use serde::{Deserialize, Serialize};

pub const OBS_CHANNELS: usize = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}

impl Action {
    pub fn from_index(index: i64) -> Option<Self> {
        match index {
            0 => Some(Self::Up),
            1 => Some(Self::Down),
            2 => Some(Self::Left),
            3 => Some(Self::Right),
            _ => None,
        }
    }

    pub fn opposite(self) -> Self {
        match self {
            Self::Up => Self::Down,
            Self::Down => Self::Up,
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }

    pub fn delta(self) -> (isize, isize) {
        match self {
            Self::Up => (0, -1),
            Self::Down => (0, 1),
            Self::Left => (-1, 0),
            Self::Right => (1, 0),
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::Up => 0,
            Self::Down => 1,
            Self::Left => 2,
            Self::Right => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point {
    pub x: usize,
    pub y: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameConfig {
    pub board_w: usize,
    pub board_h: usize,
    pub max_steps: u32,
    pub seed: u64,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            board_w: 12,
            board_h: 12,
            max_steps: 512,
            seed: 7,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    pub num_envs: usize,
    pub num_threads: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            num_envs: 16,
            num_threads: 24,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepOutput {
    pub obs: Vec<f32>,
    pub reward: f32,
    pub done: bool,
    pub score: i32,
    pub legal_actions_mask: [u8; 4],
}

#[derive(Debug, Clone)]
pub struct BatchObs {
    pub obs: Vec<f32>,
    pub shape: [usize; 4],
}

#[derive(Debug, Clone)]
pub struct BatchStepOutput {
    pub obs: Vec<f32>,
    pub shape: [usize; 4],
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub scores: Vec<i32>,
    pub legal_actions_mask: Vec<[u8; 4]>,
}
