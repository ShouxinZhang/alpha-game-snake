use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub iter: u64,
    pub stage_size: u64,
    pub mode: String,
    pub total_loss: f64,
    pub ppo_loss: f64,
    pub mae_loss: f64,
    pub icm_loss: f64,
    pub avg_env_reward: f64,
    pub avg_intrinsic_reward: f64,
    pub avg_score: f64,
    pub recent_avg_length: f64,
    pub episodes_finished: u64,
    #[serde(default)]
    pub policy_up: Option<f64>,
    #[serde(default)]
    pub policy_down: Option<f64>,
    #[serde(default)]
    pub policy_left: Option<f64>,
    #[serde(default)]
    pub policy_right: Option<f64>,
}

impl MetricsSnapshot {
    pub fn policy_probs(&self) -> Option<[f32; 4]> {
        let up = self.policy_up?;
        let down = self.policy_down?;
        let left = self.policy_left?;
        let right = self.policy_right?;
        let sum = up + down + left + right;
        if sum <= f64::EPSILON {
            return None;
        }

        Some([
            (up / sum) as f32,
            (down / sum) as f32,
            (left / sum) as f32,
            (right / sum) as f32,
        ])
    }
}

pub struct MetricsClient {
    path: Option<PathBuf>,
}

impl MetricsClient {
    pub fn new(path: Option<PathBuf>) -> Self {
        Self { path }
    }

    pub fn poll_latest(&self) -> Option<MetricsSnapshot> {
        let path = self.path.as_ref()?;
        let raw = fs::read_to_string(path).ok()?;

        for line in raw.lines().rev() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(parsed) = serde_json::from_str::<MetricsSnapshot>(line) {
                return Some(parsed);
            }
        }
        None
    }
}
