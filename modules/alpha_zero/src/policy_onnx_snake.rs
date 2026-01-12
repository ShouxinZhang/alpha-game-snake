use std::path::Path;

use tract_onnx::prelude::*;

use crate::env::Environment;
use crate::onnx::OnnxModel;
use crate::policy::PolicyValueFn;
use crate::snake::env::{SnakeAction, SnakeEnv, SnakeState};
use crate::snake::features;

pub struct SnakeOnnxPolicyValue {
    model: OnnxModel,
    in_dim: usize,
}

impl SnakeOnnxPolicyValue {
    pub fn load(path: impl AsRef<Path>, in_dim: usize) -> TractResult<Self> {
        // We assume 8 channels.
        // in_dim = 8 * w * h
        // Shape for CNN: [1, 8, h, w] or similar.
        // Since PyTorch View(-1, 8, 5, 5) just interprets the flat buffer in order,
        // we can actually stick with flattening here IF the ONNX model input is defined as flat
        // and then reshaped inside the model, OR we must provide the correct shape if the model
        // input demands it.
        // The export_onnx.py script exports a model that takes 'x' and does x.view(...).
        // BUT, ONNX export usually traces the shapes.
        // If the dummy input in export_onnx.py was [1, 200], then the ONNX input is likely [1, 200].
        // Let's check `export_onnx.py`.
        
        // Wait, in export_onnx.py:
        // dummy = torch.zeros((1, in_dim), dtype=torch.float32)
        // torch.onnx.export(..., dummy, ...)
        // The detailed forward pass does: x = x.view(-1, self.channels, self.hw, self.hw)
        
        // So the ONNX model EXPECTS a flat [1, 200] input, and the RESHAPE is PART of the model graph.
        // Therefore, we do NOT need to reshape in Rust. The input shape [1, in_dim] is correct.
        
        // So why is it failing?
        // Maybe the dimensions aren't matching perfectly, or we need to debug the error.
        
        // But wait, in the previous turn I updated train/train_snake.py AND export_onnx.py.
        // Did I update the dummy input shape in export_onnx.py?
        // export_onnx.py: dummy = torch.zeros((1, in_dim), ...)
        // This is still flat.
        // forward: x = x.view(...)
        
        // So the interface is correct.
        
        // Let's look at the error handling in policy_onnx_snake.rs.
        // It silences errors. I should print the error to stderr to confirm.
        
        let input_fact = InferenceFact::dt_shape(f32::datum_type(), tvec!(1, in_dim as i64));
        let model = OnnxModel::load(path, input_fact)?;
        Ok(Self { model, in_dim })
    }

    fn eval_raw(&self, state: &SnakeState) -> TractResult<([f32; 4], f32)> {
        let feat = features::encode(state);
        let mut x = vec![0.0f32; self.in_dim];
        let n = feat.len().min(self.in_dim);
        x[..n].copy_from_slice(&feat[..n]);
        let input_arr = tract_ndarray::Array2::from_shape_vec((1, self.in_dim), x)?;
        let input: Tensor = input_arr.into();
        let outputs = self.model.run(input)?;

        let policy_logits = outputs[0].to_array_view::<f32>()?;
        let value = outputs[1].to_array_view::<f32>()?;

        let logits = [
            policy_logits[[0, 0]],
            policy_logits[[0, 1]],
            policy_logits[[0, 2]],
            policy_logits[[0, 3]],
        ];
        let v = value[[0]];
        Ok((logits, v))
    }
}

impl SnakeAction {
    fn to_index(self) -> usize {
        match self {
            SnakeAction::Up => 0,
            SnakeAction::Down => 1,
            SnakeAction::Left => 2,
            SnakeAction::Right => 3,
        }
    }
}

impl PolicyValueFn<SnakeEnv> for SnakeOnnxPolicyValue {
    fn evaluate(&self, env: &SnakeEnv, state: &SnakeState, actions: &[SnakeAction]) -> (Vec<f32>, f32) {
        let (logits, v) = match self.eval_raw(state) {
            Ok(x) => x,
            Err(e) => {
                // Log failure to aid debugging
                eprintln!("ONNX eval failed: {:?}", e);
                let p = if actions.is_empty() { 0.0 } else { 1.0 / actions.len() as f32 };
                return (vec![p; actions.len()], 0.0);
            }
        };

        // Mask illegal actions
        let legal = env.legal_actions(state);
        let mut mask = [false; 4];
        for a in legal {
            mask[a.to_index()] = true;
        }

        // Softmax
        let mut max_logit = f32::NEG_INFINITY;
        for i in 0..4 {
            if mask[i] {
                max_logit = max_logit.max(logits[i]);
            }
        }
        if !max_logit.is_finite() {
            let p = if actions.is_empty() { 0.0 } else { 1.0 / actions.len() as f32 };
            return (vec![p; actions.len()], v);
        }

        let mut exp_sum = 0.0f32;
        let mut probs_full = [0.0f32; 4];
        for i in 0..4 {
            if mask[i] {
                let e = (logits[i] - max_logit).exp();
                probs_full[i] = e;
                exp_sum += e;
            }
        }
        if exp_sum > 0.0 {
            for p in &mut probs_full {
                *p /= exp_sum;
            }
        }

        let mut priors = Vec::with_capacity(actions.len());
        for &a in actions {
            priors.push(probs_full[a.to_index()]);
        }

        (priors, v)
    }
}
