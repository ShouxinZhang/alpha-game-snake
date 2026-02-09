use snake_engine::{Game, GameStatus, Direction};
use crate::mcts::{MCTS, MCTSConfig};
use crate::inference::InferenceEngine;
use std::sync::Arc;
use serde::{Serialize, Deserialize};

/// A single training sample from self-play
#[derive(Serialize, Deserialize, Clone)]
pub struct TrainingSample {
    /// 4-channel state encoding: [Body, Head, Food, Hunger]
    /// Shape: (4, H, W) flattened to Vec
    pub state: Vec<Vec<f32>>,
    /// MCTS policy distribution [Up, Down, Left, Right]
    pub policy: Vec<f32>,
    /// Game outcome from this player's perspective: 1.0 (win), -1.0 (loss), 0.0 (draw/ongoing)
    pub value: f32,
    /// Grid dimensions
    pub width: i32,
    pub height: i32,
}

/// Statistics from a single game
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct GameStats {
    pub steps: usize,
    pub score: u32,
    pub is_victory: bool,
    pub final_reward: f32,
}

/// Configuration for self-play games
#[derive(Clone, Copy)]
pub struct SelfPlayConfig {
    pub grid_width: i32,
    pub grid_height: i32,
    pub mcts_config: MCTSConfig,
    pub temperature: f32,
    pub temperature_threshold: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            grid_width: 10,
            grid_height: 10,
            mcts_config: MCTSConfig::default(),
            temperature: 1.0,
            temperature_threshold: 15,
        }
    }
}

/// Self-play worker that generates training data
pub struct SelfPlayWorker {
    config: SelfPlayConfig,
    inference_engine: Arc<InferenceEngine>,
}

impl SelfPlayWorker {
    pub fn new(config: SelfPlayConfig, inference_engine: Arc<InferenceEngine>) -> Self {
        Self { config, inference_engine }
    }

    /// Play a single game and return training samples + stats
    pub fn play_game(&self, seed: usize) -> (Vec<TrainingSample>, GameStats) {
        let mut game = Game::new(self.config.grid_width, self.config.grid_height);
        game.reset_with_random(seed);
        
        let mut mcts = MCTS::new(self.config.mcts_config, self.inference_engine.clone());
        let mut samples: Vec<TrainingSample> = Vec::new();
        let mut step_count = 0;
        
        while game.status() == GameStatus::Running {
            // Run MCTS
            let policy = mcts.search(&game);
            
            // Encode state
            let state = self.encode_state(&game);
            
            // Store sample (value will be filled at game end)
            samples.push(TrainingSample {
                state,
                policy: policy.clone(),
                value: 0.0, // Placeholder
                width: self.config.grid_width,
                height: self.config.grid_height,
            });
            
            // Select action based on policy
            let action = self.select_action(&policy, step_count);
            
            // Apply action
            game.queue_direction(action);
            game.step(seed.wrapping_add(step_count));
            
            step_count += 1;
        }
        
        // Fill in final values
        let is_victory = game.status() == GameStatus::Victory;
        let final_value = if is_victory { 1.0 } else { -1.0 };
        
        // Assign value to all samples
        for sample in &mut samples {
            sample.value = final_value;
        }
        
        let stats = GameStats {
            steps: step_count,
            score: game.score(),
            is_victory,
            final_reward: final_value,
        };
        
        (samples, stats)
    }
    
    /// Encode game state into 4-channel tensor representation
    fn encode_state(&self, game: &Game) -> Vec<Vec<f32>> {
        let width = game.grid_width() as usize;
        let height = game.grid_height() as usize;
        let mut channels: Vec<Vec<f32>> = vec![vec![0.0; width * height]; 4];
        
        // Channel 0: Body
        for &(x, y) in game.snake() {
            let idx = y as usize * width + x as usize;
            channels[0][idx] = 1.0;
        }
        
        // Channel 1: Head
        if let Some(&(hx, hy)) = game.snake().front() {
            let idx = hy as usize * width + hx as usize;
            channels[1][idx] = 1.0;
        }
        
        // Channel 2: Food
        let (fx, fy) = game.food();
        let idx = fy as usize * width + fx as usize;
        channels[2][idx] = 1.0;
        
        // Channel 3: Hunger (global value)
        let hunger = game.steps_since_eat() as f32 / game.max_steps_without_food() as f32;
        for i in 0..channels[3].len() {
            channels[3][i] = hunger;
        }
        
        channels
    }
    
    /// Select action from policy, with temperature
    fn select_action(&self, policy: &[f32], step_count: usize) -> Direction {
        let directions = [Direction::Up, Direction::Down, Direction::Left, Direction::Right];
        
        if step_count < self.config.temperature_threshold {
            // Sample proportionally to policy (exploration)
            let sum: f32 = policy.iter().sum();
            if sum <= 0.0 {
                return directions[0]; // Fallback
            }
            
            let normalized: Vec<f32> = policy.iter().map(|p| p / sum).collect();
            let mut r: f32 = rand::random();
            
            for (i, &p) in normalized.iter().enumerate() {
                r -= p;
                if r <= 0.0 {
                    return directions[i];
                }
            }
            directions[3] // Fallback
        } else {
            // Greedy (exploitation)
            let max_idx = policy
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            directions[max_idx]
        }
    }
}
