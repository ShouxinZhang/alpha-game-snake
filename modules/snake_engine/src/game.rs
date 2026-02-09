use std::collections::VecDeque;
use super::{Direction, Grid, BoardCache};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameStatus {
    Running,
    GameOver,
    Victory,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameOverReason {
    HitWall,
    HitSelf,
    Starvation,
}

/// 贪吃蛇核心规则引擎（不依赖任何 UI 库）
#[derive(Clone)]
pub struct Game {
    snake: VecDeque<(i32, i32)>,
    direction: Direction,
    input_queue: VecDeque<Direction>,
    food: (i32, i32),
    score: u32,
    status: GameStatus,
    game_over_reason: Option<GameOverReason>,
    cache: BoardCache,
    steps_since_eat: usize,
    max_steps_without_food: usize,
}

impl Game {
    const INPUT_QUEUE_CAPACITY: usize = 2;

    pub fn new(width: i32, height: i32) -> Self {
        let grid = Grid::new(width, height);
        let cache = BoardCache::new(grid);
        let mut game = Self {
            snake: VecDeque::new(),
            direction: Direction::Right,
            input_queue: VecDeque::new(),
            food: (0, 0),
            score: 0,
            status: GameStatus::Running,
            game_over_reason: None,
            cache,
            steps_since_eat: 0,
            max_steps_without_food: (width * height * 2) as usize, // Default T_max
        };
        game.reset_with_random(0);
        game
    }

    // ─────────────────────────────────────────────────────
    // 公共只读访问器
    // ─────────────────────────────────────────────────────
    pub fn snake(&self) -> &VecDeque<(i32, i32)> {
        &self.snake
    }

    pub fn food(&self) -> (i32, i32) {
        self.food
    }

    pub fn score(&self) -> u32 {
        self.score
    }

    pub fn status(&self) -> GameStatus {
        self.status
    }

    pub fn game_over_reason(&self) -> Option<GameOverReason> {
        self.game_over_reason
    }

    pub fn direction(&self) -> Direction {
        self.direction
    }

    pub fn grid_width(&self) -> i32 {
        self.cache.grid().width
    }

    pub fn grid_height(&self) -> i32 {
        self.cache.grid().height
    }

    pub fn steps_since_eat(&self) -> usize {
        self.steps_since_eat
    }

    pub fn max_steps_without_food(&self) -> usize {
        self.max_steps_without_food
    }

    // ─────────────────────────────────────────────────────
    // 输入队列（让外部把方向推入，step 时消费）
    // ─────────────────────────────────────────────────────
    pub fn queue_direction(&mut self, next: Direction) {
        let last_effective = self.input_queue.back().copied().unwrap_or(self.direction);
        if next == last_effective || next.is_opposite(last_effective) {
            return;
        }
        if self.input_queue.len() >= Self::INPUT_QUEUE_CAPACITY {
            return;
        }
        self.input_queue.push_back(next);
    }

    // ─────────────────────────────────────────────────────
    // 重置 / 改变尺寸
    // ─────────────────────────────────────────────────────
    pub fn reset_with_random(&mut self, rand_seed: usize) {
        self.snake.clear();
        let head_x = self.cache.grid().width / 2;
        let head_y = self.cache.grid().height / 2;
        self.snake.push_front((head_x, head_y));
        self.snake.push_back((head_x - 1, head_y));
        self.snake.push_back((head_x - 2, head_y));
        self.direction = Direction::Right;
        self.input_queue.clear();
        self.score = 0;
        self.status = GameStatus::Running;
        self.game_over_reason = None;
        self.steps_since_eat = 0;
        self.cache.rebuild(self.snake.iter());
        self.food = self.spawn_food(rand_seed);
    }

    pub fn resize(&mut self, width: i32, height: i32, rand_seed: usize) {
        self.cache.resize(width, height);
        self.reset_with_random(rand_seed);
    }

    // ─────────────────────────────────────────────────────
    // 核心推进（一步）
    // ─────────────────────────────────────────────────────
    pub fn step(&mut self, rand_seed: usize) {
        self.apply_queued_direction();
        let (dx, dy) = self.direction.delta();
        let (head_x, head_y) = *self.snake.front().expect("snake has a head");
        let new_head = (head_x + dx, head_y + dy);

        if !self.cache.grid().in_bounds(new_head) {
            self.status = GameStatus::GameOver;
            self.game_over_reason = Some(GameOverReason::HitWall);
            return;
        }

        let will_grow = new_head == self.food;
        let new_idx = self.cache.grid().to_index(new_head);
        let tail = self.snake.back().copied();
        let tail_idx = tail.map(|cell| self.cache.grid().to_index(cell));
        let hits_body = self.cache.is_occupied(new_idx);
        if hits_body && !(tail_idx == Some(new_idx) && !will_grow) {
            self.status = GameStatus::GameOver;
            self.game_over_reason = Some(GameOverReason::HitSelf);
            return;
        }

        // Starvation Check
        self.steps_since_eat += 1;
        if self.steps_since_eat >= self.max_steps_without_food {
             self.status = GameStatus::GameOver;
             self.game_over_reason = Some(GameOverReason::Starvation);
             return;
        }

        self.snake.push_front(new_head);
        self.cache.set_occupied(new_idx);

        if will_grow {
            self.score += 1;
            self.steps_since_eat = 0; // Reset hunger
            let total_cells = self.cache.grid().cell_count() as i32;
            if self.snake.len() as i32 >= total_cells {
                self.status = GameStatus::Victory;
                return;
            }
            self.food = self.spawn_food(rand_seed);
        } else if let Some(old_tail) = self.snake.pop_back() {
            if old_tail != new_head {
                let old_tail_idx = self.cache.grid().to_index(old_tail);
                self.cache.set_empty(old_tail_idx);
            }
        }
    }

    // ─────────────────────────────────────────────────────
    // 私有辅助
    // ─────────────────────────────────────────────────────
    fn apply_queued_direction(&mut self) {
        if let Some(next) = self.input_queue.pop_front() {
            if !next.is_opposite(self.direction) {
                self.direction = next;
            }
        }
    }

    fn spawn_food(&self, rand_seed: usize) -> (i32, i32) {
        self.cache.random_free_cell(rand_seed).unwrap_or((0, 0))
    }
}
