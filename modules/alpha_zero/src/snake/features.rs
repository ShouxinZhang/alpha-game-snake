use crate::snake::env::{SnakeAction, SnakeState};

/// Feature encoder: multi-channel board + direction one-hot.
///
/// Channels (C=8):
/// 0 head
/// 1 body (excluding head)
/// 2 food
/// 3 walls (board border)
/// 4 dir_up (broadcast)
/// 5 dir_down (broadcast)
/// 6 dir_left (broadcast)
/// 7 dir_right (broadcast)
///
/// Layout: channel-first flattened vector of length C*H*W.
pub fn encode(state: &SnakeState) -> Vec<f32> {
    let w = state.width as usize;
    let h = state.height as usize;
    let c = 8usize;

    let mut out = vec![0.0f32; c * w * h];

    let idx = |ch: usize, x: usize, y: usize, w: usize| -> usize { ch * w * h + y * w + x };

    // NOTE:
    // Channel 3 (walls) is intentionally left as all-zeros.
    // Border collisions are handled by the environment rules, and marking the border as walls
    // in features previously caused the model to "think" the usable board was smaller.

    // Snake
    if let Some(&(hx, hy)) = state.snake.front() {
        if hx >= 0 && hy >= 0 {
            let x = hx as usize;
            let y = hy as usize;
            if x < w && y < h {
                out[idx(0, x, y, w)] = 1.0;
            }
        }
    }
    for &(x, y) in state.snake.iter().skip(1) {
        if x >= 0 && y >= 0 {
            let xx = x as usize;
            let yy = y as usize;
            if xx < w && yy < h {
                out[idx(1, xx, yy, w)] = 1.0;
            }
        }
    }

    // Food
    let (fx, fy) = state.food;
    if fx >= 0 && fy >= 0 {
        let x = fx as usize;
        let y = fy as usize;
        if x < w && y < h {
            out[idx(2, x, y, w)] = 1.0;
        }
    }

    // Direction broadcast
    let (du, dd, dl, dr) = match state.direction {
        SnakeAction::Up => (1.0, 0.0, 0.0, 0.0),
        SnakeAction::Down => (0.0, 1.0, 0.0, 0.0),
        SnakeAction::Left => (0.0, 0.0, 1.0, 0.0),
        SnakeAction::Right => (0.0, 0.0, 0.0, 1.0),
    };
    for y in 0..h {
        for x in 0..w {
            out[idx(4, x, y, w)] = du;
            out[idx(5, x, y, w)] = dd;
            out[idx(6, x, y, w)] = dl;
            out[idx(7, x, y, w)] = dr;
        }
    }

    out
}
