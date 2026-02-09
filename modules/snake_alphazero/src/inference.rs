use snake_engine::Game;
use ort::session::{Session, builder::SessionBuilder};
use ort::value::Value;
use ndarray::{Array4, ArrayView};
use parking_lot::Mutex;

pub struct InferenceEngine {
    session: Mutex<Session>,
}

impl InferenceEngine {
    pub fn new(model_path: &str) -> Self {
        // Initialize ORT (global)
        let _ = ort::init()
            .with_name("SnakeAlphaZero")
            .commit();

        let session = Session::builder()
            .unwrap()
            .commit_from_file(model_path)
            .expect("Failed to load model");

        Self { session: Mutex::new(session) }
    }

    pub fn predict(&self, game: &Game) -> (Vec<f32>, f32) {
        let input_tensor = self.encode_state(game);
        let input_tensor_values = Value::from_array(input_tensor.into_dyn()).unwrap();

        let mut session = self.session.lock();
        let outputs = session.run(ort::inputs![input_tensor_values]).expect("Inference failed");
        
        // Output 0: Policy logits (1, 4)
        // Output 1: Value (1, 1)
        let (policy_shape, policy_data) = outputs["policy"].try_extract_tensor::<f32>().unwrap();
        let (value_shape, value_data) = outputs["value"].try_extract_tensor::<f32>().unwrap();
        
        let policy_shape_usize: Vec<usize> = policy_shape.iter().map(|&x| x as usize).collect();
        let value_shape_usize: Vec<usize> = value_shape.iter().map(|&x| x as usize).collect();

        let policy_logits_view = ArrayView::from_shape(policy_shape_usize, policy_data).unwrap().into_dimensionality::<ndarray::Ix2>().unwrap();
        let value = ArrayView::from_shape(value_shape_usize, value_data).unwrap().into_dimensionality::<ndarray::Ix2>().unwrap()[[0, 0]];

        // Softmax using ndarray
        let logits = policy_logits_view.row(0);
        let max_logit = logits.fold(f32::NEG_INFINITY, |a: f32, &b| a.max(b));
        let exp_logits = logits.mapv(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();
        let policy = (exp_logits / sum_exp).to_vec();

        (policy, value)
    }

    fn encode_state(&self, game: &Game) -> Array4<f32> {
        let width = game.grid_width() as usize;
        let height = game.grid_height() as usize;
        let mut tensor = Array4::<f32>::zeros((1, 4, height, width));

        // Channel 0: Body
        for &(x, y) in game.snake() {
            tensor[[0, 0, y as usize, x as usize]] = 1.0;
        }

        // Channel 1: Head
        if let Some(&(hx, hy)) = game.snake().front() {
             tensor[[0, 1, hy as usize, hx as usize]] = 1.0;
        }

        // Channel 2: Food
        let (fx, fy) = game.food();
        tensor[[0, 2, fy as usize, fx as usize]] = 1.0;

        // Channel 3: Hunger (Global value filled in whole channel)
        let hunger_ratio = game.steps_since_eat() as f32 / game.max_steps_without_food() as f32;
        tensor.slice_mut(ndarray::s![0, 3, .., ..]).fill(hunger_ratio);

        tensor
    }
}
