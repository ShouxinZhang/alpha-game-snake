use macroquad::prelude::*;
use crate::domain::Direction;

/// 用户意图（从按键翻译而来）
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Move(Direction),
    TogglePause,
    ToggleSettings,
    Restart,
}

/// 输入处理器：把键盘事件翻译成 Action
pub struct InputHandler;

impl InputHandler {
    /// 检测本帧按键，返回动作列表（通常 0~2 个）
    pub fn poll() -> Vec<Action> {
        let mut actions = Vec::new();

        // 方向键
        if is_key_pressed(KeyCode::Up) || is_key_pressed(KeyCode::W) {
            actions.push(Action::Move(Direction::Up));
        }
        if is_key_pressed(KeyCode::Down) || is_key_pressed(KeyCode::S) {
            actions.push(Action::Move(Direction::Down));
        }
        if is_key_pressed(KeyCode::Left) || is_key_pressed(KeyCode::A) {
            actions.push(Action::Move(Direction::Left));
        }
        if is_key_pressed(KeyCode::Right) || is_key_pressed(KeyCode::D) {
            actions.push(Action::Move(Direction::Right));
        }

        // 功能键
        if is_key_pressed(KeyCode::P) {
            actions.push(Action::TogglePause);
        }
        if is_key_pressed(KeyCode::Tab) {
            actions.push(Action::ToggleSettings);
        }
        if is_key_pressed(KeyCode::R) {
            actions.push(Action::Restart);
        }

        actions
    }
}
