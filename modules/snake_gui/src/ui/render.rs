use macroquad::prelude::*;
use super::layout::{BoardLayout, ConsoleLayout};
use snake_engine::{GameStatus, GameOverReason};
use crate::settings::SettingsEditor;

/// 渲染上下文，负责所有绘制操作
pub struct Renderer;

impl Renderer {
    // ─────────────────────────────────────────────────────
    // 颜色常量
    // ─────────────────────────────────────────────────────
    const BG_COLOR: Color = Color::new(0.071, 0.086, 0.094, 1.0);           // 18,22,24
    const BOARD_BG: Color = Color::new(0.102, 0.118, 0.133, 1.0);           // 26,30,34
    const FOOD_COLOR: Color = Color::new(0.922, 0.345, 0.345, 1.0);         // 235,88,88
    const SNAKE_HEAD: Color = Color::new(0.471, 0.863, 0.471, 1.0);         // 120,220,120
    const SNAKE_BODY: Color = Color::new(0.314, 0.706, 0.353, 1.0);         // 80,180,90
    const CONSOLE_BG: Color = Color::new(0.047, 0.055, 0.063, 0.922);       // 12,14,16,235
    const CONSOLE_LINE: Color = Color::new(0.157, 0.173, 0.188, 1.0);       // 40,44,48
    const TEXT_FG: Color = Color::new(0.863, 0.863, 0.863, 1.0);            // 220,220,220
    const TEXT_MUTED: Color = Color::new(0.667, 0.667, 0.667, 1.0);         // 170,170,170
    const OVERLAY_DIM: Color = Color::new(0.0, 0.0, 0.0, 0.353);            // 0,0,0,90
    const PANEL_BG: Color = Color::new(0.094, 0.110, 0.125, 1.0);           // 24,28,32
    const PANEL_BORDER: Color = Color::new(0.235, 0.275, 0.314, 1.0);       // 60,70,80
    const ERROR_COLOR: Color = Color::new(1.0, 0.471, 0.471, 1.0);          // 255,120,120

    // ─────────────────────────────────────────────────────
    // 整体绘制入口
    // ─────────────────────────────────────────────────────
    pub fn draw_frame(
        board: &BoardLayout,
        console: &ConsoleLayout,
        snake: &std::collections::VecDeque<(i32, i32)>,
        food: (i32, i32),
        status: GameStatus,
        game_over_reason: Option<GameOverReason>,
        paused: bool,
        score: u32,
        grid_width: i32,
        grid_height: i32,
        fps: f32,
        frame_ms: f32,
        step_ms: f32,
        settings: Option<&SettingsEditor>,
    ) {
        clear_background(Self::BG_COLOR);

        // 棋盘背景
        draw_rectangle(board.origin_x, board.origin_y, board.width, board.height, Self::BOARD_BG);

        // 食物（胜利时不画）
        if status != GameStatus::Victory {
            Self::draw_cell(food, board, Self::FOOD_COLOR);
        }

        // 蛇身
        for (i, &cell) in snake.iter().enumerate() {
            let color = if i == 0 { Self::SNAKE_HEAD } else { Self::SNAKE_BODY };
            Self::draw_cell(cell, board, color);
        }

        // 控制台
        Self::draw_console(
            console,
            status,
            game_over_reason,
            paused,
            score,
            snake.len(),
            grid_width,
            grid_height,
            fps,
            frame_ms,
            step_ms,
            settings.is_some(),
        );

        // 设置面板
        if let Some(s) = settings {
            Self::draw_settings_panel(board, s);
        }
    }

    // ─────────────────────────────────────────────────────
    // 单元格
    // ─────────────────────────────────────────────────────
    fn draw_cell(cell: (i32, i32), layout: &BoardLayout, color: Color) {
        let (px, py, w, h) = layout.cell_rect(cell);
        draw_rectangle(px, py, w, h, color);
    }

    // ─────────────────────────────────────────────────────
    // 控制台面板
    // ─────────────────────────────────────────────────────
    fn draw_console(
        layout: &ConsoleLayout,
        status: GameStatus,
        game_over_reason: Option<GameOverReason>,
        paused: bool,
        score: u32,
        snake_len: usize,
        grid_width: i32,
        grid_height: i32,
        fps: f32,
        frame_ms: f32,
        step_ms: f32,
        settings_open: bool,
    ) {
        draw_rectangle(layout.origin_x, layout.origin_y, layout.width, layout.height, Self::CONSOLE_BG);
        draw_line(
            layout.origin_x,
            layout.origin_y,
            layout.origin_x + layout.width,
            layout.origin_y,
            2.0,
            Self::CONSOLE_LINE,
        );

        let x = layout.origin_x + 12.0;
        let mut y = layout.origin_y + 26.0;
        let line_h = 20.0;

        let status_str = match status {
            GameStatus::Running => "Running",
            GameStatus::GameOver => "GameOver",
            GameStatus::Victory => "Victory",
        };

        let reason_str = if status == GameStatus::GameOver {
            match game_over_reason {
                Some(GameOverReason::HitWall) => " | Reason: Hit Wall",
                Some(GameOverReason::HitSelf) => " | Reason: Hit Self",
                Some(GameOverReason::Starvation) => " | Reason: Starved",
                None => " | Reason: Unknown",
            }
        } else {
            ""
        };
        let paused_str = if paused { " | Paused" } else { "" };

        let line1 = format!(
            "Score: {} | Len: {} | Grid: {}x{} | Status: {}{}{}",
            score, snake_len, grid_width, grid_height, status_str, reason_str, paused_str
        );
        draw_text(&line1, x, y, 18.0, Self::TEXT_FG);
        y += line_h;

        let line2 = format!("FPS: {:.0} | Frame: {:.2} ms | Step: {:.3} ms", fps, frame_ms, step_ms);
        draw_text(&line2, x, y, 18.0, Self::TEXT_FG);
        y += line_h;

        let line3 = if settings_open {
            "Controls: type 30x20 | Backspace(delete) | Enter(apply) | Esc/Tab(close)"
        } else {
            "Controls: WASD/Arrow(move) | Tab(settings) | P(pause) | R(restart)"
        };
        draw_text(line3, x, y, 16.0, Self::TEXT_MUTED);
    }

    // ─────────────────────────────────────────────────────
    // 设置面板
    // ─────────────────────────────────────────────────────
    fn draw_settings_panel(board: &BoardLayout, settings: &SettingsEditor) {
        let panel_width = 420.0;
        let panel_height = 210.0;
        let panel_x = (board.width - panel_width) * 0.5;
        let panel_y = (board.height - panel_height) * 0.5;

        // 遮罩
        draw_rectangle(board.origin_x, board.origin_y, board.width, board.height, Self::OVERLAY_DIM);
        // 面板
        draw_rectangle(panel_x, panel_y, panel_width, panel_height, Self::PANEL_BG);
        draw_rectangle_lines(panel_x, panel_y, panel_width, panel_height, 2.0, Self::PANEL_BORDER);

        draw_text("Settings", panel_x + 16.0, panel_y + 32.0, 24.0, Self::TEXT_FG);

        let current_text = format!("Current: {} x {}", settings.current_width, settings.current_height);
        draw_text(&current_text, panel_x + 16.0, panel_y + 62.0, 18.0, Color::new(0.784, 0.784, 0.784, 1.0));

        let input_text = format!("Input: {}", settings.input);
        draw_text(&input_text, panel_x + 16.0, panel_y + 92.0, 20.0, Color::new(0.902, 0.902, 0.902, 1.0));

        let preview_text = format!("Preview: {} x {}", settings.pending_width, settings.pending_height);
        draw_text(&preview_text, panel_x + 16.0, panel_y + 120.0, 18.0, Color::new(0.745, 0.745, 0.745, 1.0));

        draw_text(
            "Tip: type like 30x20 or 30 20, then Enter",
            panel_x + 16.0,
            panel_y + 148.0,
            16.0,
            Self::TEXT_MUTED,
        );
        draw_text(
            "Arrows: fine-tune | Enter: apply | Esc/Tab: close",
            panel_x + 16.0,
            panel_y + 172.0,
            16.0,
            Self::TEXT_MUTED,
        );

        if settings.error_timer > 0.0 {
            draw_text(
                "Invalid input: please enter two numbers (e.g. 30x20)",
                panel_x + 16.0,
                panel_y + 198.0,
                16.0,
                Self::ERROR_COLOR,
            );
        }
    }
}
