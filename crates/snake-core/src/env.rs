use std::collections::VecDeque;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::types::{Action, GameConfig, Point, StepOutput, OBS_CHANNELS};

#[derive(Debug, Clone)]
pub struct SnakeEnv {
    config: GameConfig,
    rng: ChaCha8Rng,
    snake: VecDeque<Point>,
    dir: Action,
    food: Point,
    done: bool,
    score: i32,
    steps: u32,
}

impl SnakeEnv {
    pub fn new(config: GameConfig, seed: u64) -> Self {
        let mut env = Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            config,
            snake: VecDeque::new(),
            dir: Action::Right,
            food: Point { x: 0, y: 0 },
            done: false,
            score: 0,
            steps: 0,
        };
        env.reset();
        env
    }

    pub fn reseed(&mut self, seed: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(seed);
        self.reset();
    }

    pub fn reset(&mut self) -> Vec<f32> {
        self.done = false;
        self.score = 0;
        self.steps = 0;
        self.dir = Action::Right;
        self.snake.clear();

        let cx = self.config.board_w / 2;
        let cy = self.config.board_h / 2;
        self.snake.push_back(Point { x: cx - 1, y: cy });
        self.snake.push_back(Point { x: cx, y: cy });
        self.snake.push_back(Point { x: cx + 1, y: cy });
        self.spawn_food();
        self.observation()
    }

    pub fn step(&mut self, requested_action: Action) -> StepOutput {
        if self.done {
            return StepOutput {
                obs: self.observation(),
                reward: 0.0,
                done: true,
                score: self.score,
                legal_actions_mask: self.legal_actions_mask(),
            };
        }

        let mut reward = -0.01;
        let action = if requested_action == self.dir.opposite() {
            reward -= 0.1;
            self.dir
        } else {
            requested_action
        };
        self.dir = action;

        let head = self
            .snake
            .back()
            .copied()
            .expect("snake should always have at least one segment");
        let (dx, dy) = action.delta();
        let next_x = head.x as isize + dx;
        let next_y = head.y as isize + dy;

        if next_x < 0
            || next_y < 0
            || next_x >= self.config.board_w as isize
            || next_y >= self.config.board_h as isize
        {
            self.done = true;
            reward -= 1.0;
            return StepOutput {
                obs: self.observation(),
                reward,
                done: true,
                score: self.score,
                legal_actions_mask: self.legal_actions_mask(),
            };
        }

        let next = Point {
            x: next_x as usize,
            y: next_y as usize,
        };
        let eating = next.x == self.food.x && next.y == self.food.y;

        let mut collision = false;
        for (idx, segment) in self.snake.iter().enumerate() {
            let is_tail = idx == 0;
            if !eating && is_tail {
                continue;
            }
            if segment.x == next.x && segment.y == next.y {
                collision = true;
                break;
            }
        }

        if collision {
            self.done = true;
            reward -= 1.0;
            return StepOutput {
                obs: self.observation(),
                reward,
                done: true,
                score: self.score,
                legal_actions_mask: self.legal_actions_mask(),
            };
        }

        self.snake.push_back(next);
        if eating {
            self.score += 1;
            reward += 1.0;
            self.spawn_food();
        } else {
            let _ = self.snake.pop_front();
        }

        self.steps += 1;
        if self.steps >= self.config.max_steps {
            self.done = true;
            reward -= 0.2;
        }

        StepOutput {
            obs: self.observation(),
            reward,
            done: self.done,
            score: self.score,
            legal_actions_mask: self.legal_actions_mask(),
        }
    }

    pub fn observation(&self) -> Vec<f32> {
        let mut obs = vec![0.0; OBS_CHANNELS * self.config.board_h * self.config.board_w];
        let plane_size = self.config.board_h * self.config.board_w;

        for segment in &self.snake {
            let idx = segment.y * self.config.board_w + segment.x;
            obs[plane_size + idx] = 1.0;
        }
        if let Some(head) = self.snake.back() {
            let idx = head.y * self.config.board_w + head.x;
            obs[idx] = 1.0;
        }

        let food_idx = self.food.y * self.config.board_w + self.food.x;
        obs[2 * plane_size + food_idx] = 1.0;

        let dir_plane = 3 + self.dir.index();
        for i in 0..plane_size {
            obs[dir_plane * plane_size + i] = 1.0;
        }

        obs
    }

    pub fn legal_actions_mask(&self) -> [u8; 4] {
        let mut mask = [1_u8; 4];
        mask[self.dir.opposite().index()] = 0;
        mask
    }

    pub fn done(&self) -> bool {
        self.done
    }

    pub fn score(&self) -> i32 {
        self.score
    }

    pub fn config(&self) -> &GameConfig {
        &self.config
    }

    pub fn debug_set_food(&mut self, x: usize, y: usize) {
        self.food = Point { x, y };
    }

    pub fn debug_set_snake(&mut self, segments_from_tail_to_head: &[Point], dir: Action) {
        self.snake.clear();
        for p in segments_from_tail_to_head {
            self.snake.push_back(*p);
        }
        self.dir = dir;
        self.done = false;
        self.steps = 0;
    }

    fn spawn_food(&mut self) {
        if self.snake.len() >= self.config.board_w * self.config.board_h {
            self.done = true;
            return;
        }

        loop {
            let x = self.rng.gen_range(0..self.config.board_w);
            let y = self.rng.gen_range(0..self.config.board_h);
            let occupied = self.snake.iter().any(|s| s.x == x && s.y == y);
            if !occupied {
                self.food = Point { x, y };
                break;
            }
        }
    }
}
