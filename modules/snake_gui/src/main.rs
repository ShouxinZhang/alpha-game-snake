//! Snake GUI - 贪吃蛇图形界面
//!
//! 模块结构:
//! - domain: 纯规则引擎（不依赖 UI）
//! - ui: 布局计算与渲染
//! - input: 键盘事件翻译
//! - settings: 设置面板编辑器
//! - app: 应用层编排

mod domain;
mod ui;
mod input;
mod settings;
mod app;

use macroquad::prelude::*;
use settings::{WINDOW_WIDTH, WINDOW_HEIGHT};
use ui::CONSOLE_HEIGHT;

fn window_conf() -> Conf {
    Conf {
        window_title: "Snake GUI".to_owned(),
        window_width: WINDOW_WIDTH as i32,
        window_height: (WINDOW_HEIGHT + CONSOLE_HEIGHT) as i32,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut app = app::App::new();
    loop {
        app.tick();
        next_frame().await;
    }
}
