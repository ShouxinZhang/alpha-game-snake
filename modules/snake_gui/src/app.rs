use macroquad::prelude::*;
use macroquad::rand::gen_range;

use crate::domain::{Game, GameStatus};
use crate::input::{InputHandler, Action};
use crate::settings::{SettingsEditor, SettingsResult, DEFAULT_GRID_WIDTH, DEFAULT_GRID_HEIGHT};
use crate::ui::{BoardLayout, ConsoleLayout, Renderer};

const STEP_TIME: f32 = 0.12;

/// 应用层：编排游戏逻辑、输入、UI、设置的状态机
pub struct App {
    game: Game,
    paused: bool,
    step_timer: f32,
    settings: Option<SettingsEditor>,
    settings_toggled_this_frame: bool,
    // 性能统计
    last_frame_ms: f32,
    last_fps: f32,
    last_step_ms: f32,
}

impl App {
    pub fn new() -> Self {
        Self {
            game: Game::new(DEFAULT_GRID_WIDTH, DEFAULT_GRID_HEIGHT),
            paused: false,
            step_timer: 0.0,
            settings: None,
            settings_toggled_this_frame: false,
            last_frame_ms: 0.0,
            last_fps: 0.0,
            last_step_ms: 0.0,
        }
    }

    /// 每帧调用：处理输入 → 更新逻辑 → 渲染
    pub fn tick(&mut self) {
        let dt = get_frame_time();
        self.last_frame_ms = dt * 1000.0;
        self.last_fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };

        self.settings_toggled_this_frame = false;

        self.handle_input();
        self.update(dt);
        self.render();
    }

    // ─────────────────────────────────────────────────────
    // 输入处理
    // ─────────────────────────────────────────────────────
    fn handle_input(&mut self) {
        let actions = InputHandler::poll();
        for action in actions {
            match action {
                Action::ToggleSettings => {
                    if self.settings.is_some() {
                        self.settings = None;
                    } else {
                        self.open_settings();
                    }

                    // 关键：避免在同一帧里打开 settings 又立刻执行 settings.update()
                    // 导致 Tab 按下当帧被当作“关闭”处理。
                    self.settings_toggled_this_frame = true;
                    return;
                }
                Action::Move(dir) => {
                    if self.settings.is_some() {
                        continue;
                    }
                    if self.game.status() == GameStatus::Running {
                        self.game.queue_direction(dir);
                    }
                }
                Action::TogglePause => {
                    if self.settings.is_some() {
                        continue;
                    }
                    if self.game.status() == GameStatus::Running {
                        self.paused = !self.paused;
                    }
                }
                Action::Restart => {
                    if self.settings.is_some() {
                        continue;
                    }
                    if self.game.status() != GameStatus::Running {
                        self.game.reset_with_random(self.rand_seed());
                    }
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────
    // 逻辑更新
    // ─────────────────────────────────────────────────────
    fn update(&mut self, dt: f32) {
        // Tab 切换 settings 的那一帧，跳过更新，避免“打开即关闭”。
        if self.settings_toggled_this_frame {
            return;
        }

        // 设置面板更新
        if let Some(ref mut editor) = self.settings {
            match editor.update(dt) {
                SettingsResult::Close => {
                    self.settings = None;
                }
                SettingsResult::Apply(w, h) => {
                    self.game.resize(w, h, self.rand_seed());
                    self.paused = false;
                    self.step_timer = 0.0;
                    self.settings = None;
                }
                SettingsResult::None => {}
            }
            return;
        }

        // 游戏结束时不推进
        if self.game.status() != GameStatus::Running {
            return;
        }

        // 暂停时不推进
        if self.paused {
            return;
        }

        // 步进计时器
        self.step_timer += dt;
        while self.step_timer >= STEP_TIME {
            self.step_timer -= STEP_TIME;
            let before = get_time();
            self.game.step(self.rand_seed());
            let after = get_time();
            self.last_step_ms = ((after - before) * 1000.0) as f32;

            if self.game.status() != GameStatus::Running {
                break;
            }
        }
    }

    // ─────────────────────────────────────────────────────
    // 渲染
    // ─────────────────────────────────────────────────────
    fn render(&self) {
        let board = BoardLayout::compute(
            screen_width(),
            screen_height(),
            self.game.grid_width(),
            self.game.grid_height(),
        );
        let console = ConsoleLayout::compute(screen_width(), screen_height());

        Renderer::draw_frame(
            &board,
            &console,
            self.game.snake(),
            self.game.food(),
            self.game.status(),
            self.game.game_over_reason(),
            self.paused,
            self.game.score(),
            self.game.grid_width(),
            self.game.grid_height(),
            self.last_fps,
            self.last_frame_ms,
            self.last_step_ms,
            self.settings.as_ref(),
        );
    }

    // ─────────────────────────────────────────────────────
    // 辅助
    // ─────────────────────────────────────────────────────
    fn open_settings(&mut self) {
        self.settings = Some(SettingsEditor::new(
            self.game.grid_width(),
            self.game.grid_height(),
        ));
    }

    fn rand_seed(&self) -> usize {
        gen_range(0, i32::MAX) as usize
    }
}
