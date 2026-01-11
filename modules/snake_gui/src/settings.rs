use macroquad::prelude::*;

pub const DEFAULT_GRID_WIDTH: i32 = 30;
pub const DEFAULT_GRID_HEIGHT: i32 = 20;
pub const MIN_GRID_WIDTH: i32 = 4;
pub const MIN_GRID_HEIGHT: i32 = 4;
pub const DEFAULT_CELL_SIZE: f32 = 24.0;
pub const MIN_CELL_SIZE: f32 = 12.0;
pub const WINDOW_WIDTH: f32 = DEFAULT_GRID_WIDTH as f32 * DEFAULT_CELL_SIZE;
pub const WINDOW_HEIGHT: f32 = DEFAULT_GRID_HEIGHT as f32 * DEFAULT_CELL_SIZE;
pub const MAX_GRID_WIDTH: i32 = (WINDOW_WIDTH / MIN_CELL_SIZE) as i32;
pub const MAX_GRID_HEIGHT: i32 = (WINDOW_HEIGHT / MIN_CELL_SIZE) as i32;

/// 设置面板编辑器状态
pub struct SettingsEditor {
    pub current_width: i32,
    pub current_height: i32,
    pub pending_width: i32,
    pub pending_height: i32,
    pub input: String,
    pub error_timer: f32,
}

/// 设置面板操作结果
pub enum SettingsResult {
    None,
    Close,
    Apply(i32, i32),
}

impl SettingsEditor {
    pub fn new(current_width: i32, current_height: i32) -> Self {
        Self {
            current_width,
            current_height,
            pending_width: current_width,
            pending_height: current_height,
            input: String::new(),
            error_timer: 0.0,
        }
    }

    /// 解析输入字符串，提取两个数字
    fn parse_input(input: &str) -> Option<(i32, i32)> {
        let mut numbers = input
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse::<i32>().ok());
        let w = numbers.next()?;
        let h = numbers.next()?;
        Some((w, h))
    }

    /// 每帧调用，处理键盘输入，返回操作结果
    pub fn update(&mut self, dt: f32) -> SettingsResult {
        // 错误提示倒计时
        if self.error_timer > 0.0 {
            self.error_timer = (self.error_timer - dt).max(0.0);
        }

        // Esc 或 Tab 关闭
        if is_key_pressed(KeyCode::Escape) || is_key_pressed(KeyCode::Tab) {
            return SettingsResult::Close;
        }

        // 退格删除
        if is_key_pressed(KeyCode::Backspace) {
            self.input.pop();
        }

        // 字符输入
        while let Some(ch) = get_char_pressed() {
            let ok = ch.is_ascii_digit() || matches!(ch, 'x' | 'X' | ' ' | ',' | ';');
            if ok && self.input.len() < 12 {
                self.input.push(ch);
            }
        }

        // 实时解析预览
        if let Some((w, h)) = Self::parse_input(&self.input) {
            self.pending_width = w.clamp(MIN_GRID_WIDTH, MAX_GRID_WIDTH);
            self.pending_height = h.clamp(MIN_GRID_HEIGHT, MAX_GRID_HEIGHT);
        }

        // Enter 应用
        if is_key_pressed(KeyCode::Enter) {
            if let Some((w, h)) = Self::parse_input(&self.input) {
                let w = w.clamp(MIN_GRID_WIDTH, MAX_GRID_WIDTH);
                let h = h.clamp(MIN_GRID_HEIGHT, MAX_GRID_HEIGHT);
                return SettingsResult::Apply(w, h);
            }
            self.error_timer = 1.8;
        }

        // 方向键微调
        let mut adjusted = false;
        if is_key_pressed(KeyCode::Left) || is_key_pressed(KeyCode::A) {
            self.pending_width = (self.pending_width - 1).max(MIN_GRID_WIDTH);
            adjusted = true;
        } else if is_key_pressed(KeyCode::Right) || is_key_pressed(KeyCode::D) {
            self.pending_width = (self.pending_width + 1).min(MAX_GRID_WIDTH);
            adjusted = true;
        }
        if is_key_pressed(KeyCode::Up) || is_key_pressed(KeyCode::W) {
            self.pending_height = (self.pending_height + 1).min(MAX_GRID_HEIGHT);
            adjusted = true;
        } else if is_key_pressed(KeyCode::Down) || is_key_pressed(KeyCode::S) {
            self.pending_height = (self.pending_height - 1).max(MIN_GRID_HEIGHT);
            adjusted = true;
        }
        if adjusted {
            self.input = format!("{}x{}", self.pending_width, self.pending_height);
        }

        SettingsResult::None
    }
}
