use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub rolling_avg_reward: f64,
    pub avg_score: f64,
    pub avg_steps: f64,
    pub loss: f64,
    #[serde(default)]
    pub policy_up: Option<f64>,
    #[serde(default)]
    pub policy_down: Option<f64>,
    #[serde(default)]
    pub policy_left: Option<f64>,
    #[serde(default)]
    pub policy_right: Option<f64>,
    #[serde(default)]
    pub episodes_finished: Option<u64>,
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
        let line = raw.lines().last()?;
        serde_json::from_str::<MetricsSnapshot>(line).ok()
    }
}
