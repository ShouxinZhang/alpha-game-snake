use std::collections::VecDeque;

use crate::env::{Environment, Player};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum SnakeAction {
    Up,
    Down,
    Left,
    Right,
}

impl SnakeAction {
    pub fn delta(self) -> (i32, i32) {
        match self {
            Self::Up => (0, -1),
            Self::Down => (0, 1),
            Self::Left => (-1, 0),
            Self::Right => (1, 0),
        }
    }

    pub fn is_opposite(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Up, Self::Down)
                | (Self::Down, Self::Up)
                | (Self::Left, Self::Right)
                | (Self::Right, Self::Left)
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SnakeStatus {
    Running,
    GameOver,
    Victory,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SnakeGameOverReason {
    HitWall,
    HitSelf,
    Stall,
    Loop,
}

#[derive(Clone, Debug)]
pub struct SnakeState {
    pub width: i32,
    pub height: i32,

    pub snake: VecDeque<(i32, i32)>,
    pub direction: SnakeAction,
    pub food: (i32, i32),

    pub status: SnakeStatus,
    pub game_over_reason: Option<SnakeGameOverReason>,

    pub score: u32,
    pub steps_since_food: u32,

    /// Rolling history of state hashes for loop detection.
    /// Kept bounded by `SnakeEnv::history_limit`.
    pub history_hashes: VecDeque<u64>,

    pub rng: u64,
}

impl SnakeState {
    fn in_bounds(&self, cell: (i32, i32)) -> bool {
        cell.0 >= 0 && cell.0 < self.width && cell.1 >= 0 && cell.1 < self.height
    }

    fn cell_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    fn to_index(&self, cell: (i32, i32)) -> usize {
        (cell.1 as usize) * (self.width as usize) + (cell.0 as usize)
    }

    fn from_index(&self, idx: usize) -> (i32, i32) {
        let w = self.width as usize;
        ((idx % w) as i32, (idx / w) as i32)
    }
}

#[derive(Clone, Debug)]
pub struct SnakeEnv {
    pub width: i32,
    pub height: i32,

    /// Steps without eating food allowed before we consider it stalling and force game over.
    /// This is the code-level "no circling" guard.
    pub stall_limit: u32,

    /// Kill immediately when a previously seen state repeats.
    pub kill_on_loop: bool,

    /// Max number of recent hashes to keep for loop detection.
    pub history_limit: usize,

    /// Per-step hunger penalty.
    pub step_penalty: f32,

    /// Reward for eating food.
    pub food_reward: f32,

    /// Terminal penalties.
    pub hit_wall_penalty: f32,
    pub hit_self_penalty: f32,
    pub stall_penalty: f32,
}

impl SnakeEnv {
    pub fn new(width: i32, height: i32) -> Self {
        let cell_count = (width * height).max(1) as u32;
        Self {
            width,
            height,
            stall_limit: cell_count.saturating_mul(4),
            // Loop Detection Tuning:
            // Previous: true. Immediate kill on ANY repeating state.
            // Problem: Snake needs to "wait" or loop safely to let its tail pass or wait for path to open.
            // Fix: false. Allow loops, let step_penalty or stall_limit handle infinite stalls naturally.
            //      Or strictly, allow *non-fatal* loops but kill only on specific 3-move cycles?
            //      Simplest for training is to disable instant-kill.
            kill_on_loop: false, 
            history_limit: (cell_count as usize).saturating_mul(8).max(256),
            // Reward Tuning:
            // Previous: -0.1 step, 1.0 food.
            // Problem: If steps > 10, net reward < 0. Agent learns to suicide.
            // Fix: -0.01 step. Eating is always profitable if < 100 steps.
            step_penalty: -0.01,
            food_reward: 1.0,
            // Normalize Death Penalties:
            // Previous: Wall -10, Self -1. Agent prefers hitting self to quit early.
            // Fix: All deaths -1.0 (or -2.0 to ensure larger than step noise).
            hit_wall_penalty: -1.0,
            hit_self_penalty: -1.0,
            stall_penalty: -1.0,
        }
    }

    pub fn reset(&self, seed: u64) -> SnakeState {
        let mut snake = VecDeque::new();
        let head_x = self.width / 2;
        let head_y = self.height / 2;
        snake.push_front((head_x, head_y));
        snake.push_back((head_x - 1, head_y));
        snake.push_back((head_x - 2, head_y));

        let mut s = SnakeState {
            width: self.width,
            height: self.height,
            snake,
            direction: SnakeAction::Right,
            food: (0, 0),
            status: SnakeStatus::Running,
            game_over_reason: None,
            score: 0,
            steps_since_food: 0,
            history_hashes: VecDeque::new(),
            rng: seed,
        };
        s.food = self.spawn_food(&mut s);
        let h = self.state_hash(&s);
        s.history_hashes.push_back(h);
        s
    }

    fn next_rng(rng: &mut u64) -> u64 {
        // Deterministic LCG
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        *rng
    }

    fn spawn_food(&self, state: &mut SnakeState) -> (i32, i32) {
        let total = state.cell_count();
        if total == 0 {
            return (0, 0);
        }

        // Build free cells list (O(N)); for training/MCTS on small boards this is OK.
        let mut occupied = vec![false; total];
        for &cell in state.snake.iter() {
            if state.in_bounds(cell) {
                occupied[state.to_index(cell)] = true;
            }
        }
        let mut free = Vec::new();
        for idx in 0..total {
            if !occupied[idx] {
                free.push(idx);
            }
        }
        if free.is_empty() {
            return (0, 0);
        }

        let r = (Self::next_rng(&mut state.rng) >> 32) as usize;
        let pick = free[r % free.len()];
        state.from_index(pick)
    }

    fn apply_action(direction: SnakeAction, action: SnakeAction) -> SnakeAction {
        if action.is_opposite(direction) {
            direction
        } else {
            action
        }
    }

    fn mix64(mut x: u64) -> u64 {
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
        x ^= x >> 33;
        x
    }

    fn state_hash(&self, s: &SnakeState) -> u64 {
        // Hash includes: snake body coords, food coord, direction, board size.
        let mut h = 0xcbf29ce484222325u64;
        h ^= Self::mix64((s.width as u64) << 32 | (s.height as u64));
        h = h.wrapping_mul(0x100000001b3);

        let dir_id = match s.direction {
            SnakeAction::Up => 0u64,
            SnakeAction::Down => 1u64,
            SnakeAction::Left => 2u64,
            SnakeAction::Right => 3u64,
        };
        h ^= Self::mix64(dir_id);
        h = h.wrapping_mul(0x100000001b3);

        let (fx, fy) = s.food;
        h ^= Self::mix64(((fx as u64) << 32) ^ (fy as u64));
        h = h.wrapping_mul(0x100000001b3);

        // Include full body so cycle detection is exact.
        for &(x, y) in s.snake.iter() {
            h ^= Self::mix64(((x as u64) << 32) ^ (y as u64));
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

impl Environment for SnakeEnv {
    type State = SnakeState;
    type Action = SnakeAction;

    fn player_to_move(&self, _state: &Self::State) -> Player {
        1
    }

    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action> {
        if state.status != SnakeStatus::Running {
            return Vec::new();
        }
        let cur = state.direction;
        [SnakeAction::Up, SnakeAction::Down, SnakeAction::Left, SnakeAction::Right]
            .into_iter()
            .filter(|&a| !a.is_opposite(cur))
            .collect()
    }

    fn next_state(&self, state: &Self::State, action: Self::Action) -> Self::State {
        // Keep backward-compat callers working; ignore reward.
        self.step(state, action).0
    }

    fn step(&self, state: &Self::State, action: Self::Action) -> (Self::State, f32) {
        let mut s = state.clone();
        if s.status != SnakeStatus::Running {
            return (s, 0.0);
        }

        // Anti-stalling guard: if too long without eating, force game over.
        if s.steps_since_food >= self.stall_limit {
            s.status = SnakeStatus::GameOver;
            s.game_over_reason = Some(SnakeGameOverReason::Stall);
            return (s, self.stall_penalty);
        }

        s.direction = Self::apply_action(s.direction, action);
        let (dx, dy) = s.direction.delta();
        let (hx, hy) = *s.snake.front().expect("snake has head");
        let new_head = (hx + dx, hy + dy);

        if !s.in_bounds(new_head) {
            s.status = SnakeStatus::GameOver;
            s.game_over_reason = Some(SnakeGameOverReason::HitWall);
            return (s, self.hit_wall_penalty);
        }

        let will_grow = new_head == s.food;
        let tail = s.snake.back().copied();

        // Check self collision; allow moving into tail if tail will move away this step.
        let hits_body = s.snake.iter().any(|&c| c == new_head);
        if hits_body {
            if !(tail == Some(new_head) && !will_grow) {
                s.status = SnakeStatus::GameOver;
                s.game_over_reason = Some(SnakeGameOverReason::HitSelf);
                return (s, self.hit_self_penalty);
            }
        }

        s.snake.push_front(new_head);

        let mut reward = self.step_penalty;

        if will_grow {
            s.score += 1;
            s.steps_since_food = 0;
            reward += self.food_reward;

            if s.snake.len() >= s.cell_count() {
                s.status = SnakeStatus::Victory;
                return (s, reward);
            }

            s.food = self.spawn_food(&mut s);
        } else {
            s.steps_since_food = s.steps_since_food.saturating_add(1);
            s.snake.pop_back();
        }

        // Loop detection: repeat once => dead.
        if self.kill_on_loop {
            let h = self.state_hash(&s);
            if s.history_hashes.iter().any(|&x| x == h) {
                s.status = SnakeStatus::GameOver;
                s.game_over_reason = Some(SnakeGameOverReason::Loop);
                return (s, self.stall_penalty);
            }
            s.history_hashes.push_back(h);
            while s.history_hashes.len() > self.history_limit {
                s.history_hashes.pop_front();
            }
        }

        // Another anti-stalling check post-step (covers the exact boundary condition)
        if s.steps_since_food >= self.stall_limit {
            s.status = SnakeStatus::GameOver;
            s.game_over_reason = Some(SnakeGameOverReason::Stall);
            return (s, self.stall_penalty);
        }

        (s, reward)
    }

    fn terminal_value(&self, state: &Self::State) -> Option<f32> {
        match state.status {
            SnakeStatus::Running => None,
            // Terminal reward is emitted on the last transition via `step()`.
            SnakeStatus::GameOver | SnakeStatus::Victory => Some(0.0),
        }
    }
}
