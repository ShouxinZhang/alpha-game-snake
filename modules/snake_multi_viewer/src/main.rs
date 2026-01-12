use alpha_zero::snake::{SnakeEnv, SnakeStatus};
use alpha_zero::Environment;
use alpha_zero::{SnakeOnnxPolicyValue, PolicyValueFn};
use macroquad::prelude::*;
use ::rand::rngs::StdRng;
use ::rand::{Rng, RngCore, SeedableRng};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

const AGENTS: usize = 32;
const COLS: usize = 8;
const ROWS: usize = 4;
const GRID_W: i32 = 5;
const GRID_H: i32 = 5;
const CONSOLE_H: f32 = 200.0;
const LOG_LINES_MAX: usize = 12;

fn window_conf() -> Conf {
    Conf {
        window_title: "Snake Multi Viewer".to_string(),
        window_width: 1000,
        window_height: 650,
        ..Default::default()
    }
}

struct Agent {
    state: alpha_zero::snake::SnakeState,
    rng: StdRng,
    cur_reward: f32,
    cur_steps: u32,
    cur_value: f32, // Visualize the AI's confidence

    episodes: u32,
    wins: u32,
    sum_reward: f32,
    sum_steps: u32,
    done_cooldown: u32,
}

struct TrainCtl {
    running: bool,
    child: Arc<Mutex<Option<Child>>>,
    rx: Option<mpsc::Receiver<String>>,
}

impl TrainCtl {
    fn new() -> Self {
        Self {
            running: false,
            child: Arc::new(Mutex::new(None)),
            rx: None,
        }
    }

    fn start(&mut self, rounds: u64) {
        if self.running {
            return;
        }
        let (tx, rx) = mpsc::channel::<String>();
        self.rx = Some(rx);
        self.running = true;

        let child_slot = self.child.clone();
        thread::spawn(move || {
            // Intelligent path discovery: check if we are in root or in modules/snake_multi_viewer
            let mut alpha_zero_dir = PathBuf::from("../alpha_zero");
            if !alpha_zero_dir.exists() {
                alpha_zero_dir = PathBuf::from("modules/alpha_zero");
            }

            let script = PathBuf::from("train/loop_snake_5x5.sh");
            let script_path = alpha_zero_dir.join(&script);

            if !script_path.exists() {
                let _ = tx.send(format!("[ui] ERROR: script not found at {:?}", script_path));
                return;
            }

            let mut cmd = Command::new("bash");
            cmd.current_dir(&alpha_zero_dir)
                .arg(&script)
                .arg(rounds.to_string())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());

            // Ensure the Python-side training logs flush line-by-line.
            cmd.env("PYTHONUNBUFFERED", "1")
                .env("CARGO_TERM_COLOR", "never")
                .env("TERM", "dumb");

            match cmd.spawn() {
                Ok(child) => {
                    let _ = tx.send("[train] started".to_string());

                    // Store child so Stop can kill it.
                    {
                        let mut lock = child_slot.lock().unwrap();
                        *lock = Some(child);
                    }

                    // Take stdout/stderr from the stored child.
                    let (stdout, stderr) = {
                        let mut lock = child_slot.lock().unwrap();
                        let child = lock.as_mut().unwrap();
                        (child.stdout.take(), child.stderr.take())
                    };

                    if let Some(out) = stdout {
                        let txo = tx.clone();
                        thread::spawn(move || {
                            let reader = BufReader::new(out);
                            for line in reader.lines().flatten() {
                                let _ = txo.send(format!("[train] {}", line));
                            }
                        });
                    }
                    if let Some(err) = stderr {
                        let txe = tx.clone();
                        thread::spawn(move || {
                            let reader = BufReader::new(err);
                            for line in reader.lines().flatten() {
                                let _ = txe.send(format!("[train] {}", line));
                            }
                        });
                    }

                    // Wait for completion (or Stop kills it).
                    let status = {
                        let mut lock = child_slot.lock().unwrap();
                        if let Some(child) = lock.as_mut() {
                            child.wait().ok()
                        } else {
                            None
                        }
                    };
                    {
                        let mut lock = child_slot.lock().unwrap();
                        *lock = None;
                    }
                    let _ = tx.send(format!("[train] exited: {:?}", status));
                    // If it exited immediately with an error code, it's bad.
                }
                Err(e) => {
                    let _ = tx.send(format!("[train] failed to start: {}. Make sure 'bash' is in PATH.", e));
                }
            }
        });
    }

    fn stop(&mut self) {
        if !self.running {
            return;
        }
        if let Ok(mut lock) = self.child.lock() {
            if let Some(mut child) = lock.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        self.running = false;
    }
}

fn truncate_text(mut s: String, max_w: f32, font_size: u16) -> String {
    let m = measure_text(&s, None, font_size, 1.0);
    if m.width <= max_w {
        return s;
    }
    let mut chars: Vec<char> = s.chars().collect();
    if chars.len() <= 4 {
        return "…".to_string();
    }

    // binary search the max prefix length that fits
    let mut lo = 0usize;
    let mut hi = chars.len();
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        let cand: String = chars[..mid].iter().collect::<String>() + "…";
        if measure_text(&cand, None, font_size, 1.0).width <= max_w {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    chars.truncate(lo);
    s = chars.into_iter().collect::<String>();
    s.push('…');
    s
}

impl Agent {
    fn new(env: &SnakeEnv, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let state = env.reset(rng.next_u64());
        Self {
            state,
            rng,
            cur_reward: 0.0,
            cur_steps: 0,
            cur_value: 0.0,
            episodes: 0,
            wins: 0,
            sum_reward: 0.0,
            sum_steps: 0,
            done_cooldown: 0,
        }
    }

    fn reset(&mut self, env: &SnakeEnv) {
        self.state = env.reset(self.rng.next_u64());
        self.cur_reward = 0.0;
        self.cur_steps = 0;
        self.cur_value = 0.0;
        self.done_cooldown = 0;
    }

    fn step_once(&mut self, env: &SnakeEnv, policy: Option<&SnakeOnnxPolicyValue>) {
        if self.state.status != SnakeStatus::Running {
            if self.done_cooldown > 0 {
                self.done_cooldown -= 1;
                return;
            }
            self.episodes += 1;
            if self.state.status == SnakeStatus::Victory {
                self.wins += 1;
            }

            // roll current episode stats into history
            self.sum_reward += self.cur_reward;
            self.sum_steps = self.sum_steps.saturating_add(self.cur_steps);

            self.reset(env);
            return;
        }

        let actions = env.legal_actions(&self.state);
        if actions.is_empty() {
            self.done_cooldown = 20;
            return;
        }

        let a = if let Some(pol) = policy {
            let (priors, val) = pol.evaluate(env, &self.state, &actions);
            self.cur_value = val;
            
            // GREEDY SELECTION (Argmax)
            // Instead of weighted random (which explores), we pick the BEST move.
            // This represents the model's true capability.
            let mut best_idx = 0;
            let mut best_p = -1.0;
            // Add small noise to break ties randomly instead of always picking first
            for (i, &p) in priors.iter().enumerate() {
                let jitter = self.rng.gen::<f32>() * 1e-5; 
                if p + jitter > best_p {
                    best_p = p + jitter;
                    best_idx = i;
                }
            }
            actions[best_idx]
        } else {
            self.cur_value = 0.0;
            actions[self.rng.gen_range(0..actions.len())]
        };

        let (ns, r) = env.step(&self.state, a);
        self.state = ns;
        self.cur_reward += r;
        self.cur_steps += 1;

        if self.state.status != SnakeStatus::Running {
            self.done_cooldown = 20;
        }
    }
}

fn draw_board(env: &SnakeEnv, agent: &Agent, x0: f32, y0: f32, w: f32, h: f32, selected: bool) {
    let pad = 6.0;
    let header_h = 0.0;

    let border = if selected { YELLOW } else { DARKGRAY };
    draw_rectangle_lines(x0, y0, w, h, 2.0, border);

    let board_w = w - pad * 2.0;
    let board_h = h - pad * 2.0 - header_h;
    let cell = (board_w / GRID_W as f32).min(board_h / GRID_H as f32);

    let bx = x0 + pad;
    let by = y0 + pad + header_h;

    // background
    draw_rectangle(bx, by, cell * GRID_W as f32, cell * GRID_H as f32, Color::new(0.08, 0.08, 0.10, 1.0));

    // grid lines
    for iy in 0..=GRID_H {
        let y = by + iy as f32 * cell;
        draw_line(bx, y, bx + cell * GRID_W as f32, y, 1.0, Color::new(0.15, 0.15, 0.18, 1.0));
    }
    for ix in 0..=GRID_W {
        let x = bx + ix as f32 * cell;
        draw_line(x, by, x, by + cell * GRID_H as f32, 1.0, Color::new(0.15, 0.15, 0.18, 1.0));
    }

    // food
    let (fx, fy) = agent.state.food;
    if fx >= 0 && fy >= 0 {
        let rx = bx + fx as f32 * cell;
        let ry = by + fy as f32 * cell;
        draw_rectangle(rx + 2.0, ry + 2.0, cell - 4.0, cell - 4.0, RED);
    }

    // snake (body then head)
    for (i, &(sx, sy)) in agent.state.snake.iter().rev().enumerate() {
        let rx = bx + sx as f32 * cell;
        let ry = by + sy as f32 * cell;
        let c = if i == agent.state.snake.len() - 1 { GREEN } else { Color::new(0.2, 0.7, 0.2, 1.0) };
        draw_rectangle(rx + 2.0, ry + 2.0, cell - 4.0, cell - 4.0, c);
    }

    // status overlay
    if agent.state.status != SnakeStatus::Running {
        let s = match agent.state.status {
            SnakeStatus::Victory => "VICTORY",
            SnakeStatus::GameOver => "GAME OVER",
            SnakeStatus::Running => "",
        };
        draw_rectangle(bx, by + cell * 2.0, cell * GRID_W as f32, cell, Color::new(0.0, 0.0, 0.0, 0.55));
        draw_text(s, bx + 6.0, by + cell * 2.0 + cell * 0.75, 20.0, WHITE);
    }

    // NOTE: currently env is unused in rendering; keep signature for future (e.g., draw reason)
    let _ = env;
}

fn draw_button(rect: Rect, label: &str, enabled: bool) -> bool {
    let (mx, my) = mouse_position();
    let hovered = rect.contains(Vec2::new(mx, my));
    let bg = if !enabled {
        Color::new(0.14, 0.14, 0.16, 1.0)
    } else if hovered {
        Color::new(0.22, 0.22, 0.26, 1.0)
    } else {
        Color::new(0.18, 0.18, 0.21, 1.0)
    };
    draw_rectangle(rect.x, rect.y, rect.w, rect.h, bg);
    draw_rectangle_lines(rect.x, rect.y, rect.w, rect.h, 1.0, Color::new(0.35, 0.35, 0.4, 1.0));
    draw_text(label, rect.x + 10.0, rect.y + rect.h * 0.72, 18.0, WHITE);

    enabled && hovered && is_mouse_button_pressed(MouseButton::Left)
}

fn draw_console(
    agents: &[Agent],
    selected: usize,
    log_lines: &VecDeque<String>,
    train_running: bool,
    start_enabled: bool,
    stop_enabled: bool,
    has_model: bool,
) -> (bool, bool) {
    let y0 = screen_height() - CONSOLE_H;
    draw_rectangle(0.0, y0, screen_width(), CONSOLE_H, Color::new(0.03, 0.03, 0.035, 1.0));
    draw_line(0.0, y0, screen_width(), y0, 2.0, Color::new(0.12, 0.12, 0.14, 1.0));

    // header row
    let train_state = if train_running { "TRAIN:RUN" } else { "TRAIN:STOP" };
    draw_text(train_state, 12.0, y0 + 24.0, 18.0, if train_running { GREEN } else { GRAY });

    let (m_txt, m_clr) = if has_model {
        ("MODEL: LOADED", SKYBLUE)
    } else {
        ("MODEL: RANDOM (TRAINING REQUIRED)", ORANGE)
    };
    draw_text(m_txt, 140.0, y0 + 24.0, 18.0, m_clr);

    // buttons (top-right)
    let btn_y = y0 + 8.0;
    let start_rect = Rect::new(screen_width() - 260.0, btn_y, 120.0, 28.0);
    let stop_rect = Rect::new(screen_width() - 130.0, btn_y, 120.0, 28.0);
    let start_clicked = draw_button(start_rect, "Start Train", start_enabled);
    let stop_clicked = draw_button(stop_rect, "Stop Train", stop_enabled);

    // log panel
    let log_x = 12.0;
    let log_y = y0 + 38.0;
    let log_w = screen_width() - 24.0;
    let log_h = CONSOLE_H - 38.0 - 64.0;
    draw_rectangle(log_x, log_y, log_w, log_h, Color::new(0.05, 0.05, 0.06, 1.0));
    draw_rectangle_lines(log_x, log_y, log_w, log_h, 1.0, Color::new(0.12, 0.12, 0.14, 1.0));

    // rolling log (older -> newer)
    let font = 18u16;
    let max_text_w = log_w - 16.0;
    let mut y = log_y + 22.0;
    for line in log_lines.iter() {
        let s = truncate_text(line.clone(), max_text_w, font);
        draw_text(&s, log_x + 8.0, y, font as f32, GRAY);
        y += 20.0;
        if y > log_y + log_h - 6.0 {
            break;
        }
    }

    let a = &agents[selected];
    let sel_wr = if a.episodes == 0 { 0.0 } else { a.wins as f32 / a.episodes as f32 };
    let sel_avg_r = if a.episodes == 0 { 0.0 } else { a.sum_reward / a.episodes as f32 };
    let sel_avg_steps = if a.episodes == 0 { 0.0 } else { a.sum_steps as f32 / a.episodes as f32 };
    let reason = match a.state.game_over_reason {
        Some(r) => format!("{:?}", r),
        None => "-".to_string(),
    };
    let line2 = format!(
        "selected #{:02} | {:?} ({}) | cur: score={} steps={} R={:.1} V={:.2} | hist: ep={} wr={:.2} avgR={:.2} avgS={:.1}",
        selected,
        a.state.status,
        reason,
        a.state.score,
        a.cur_steps,
        a.cur_reward,
        a.cur_value,
        a.episodes,
        sel_wr,
        sel_avg_r,
        sel_avg_steps
    );
    let sel_text = truncate_text(line2, screen_width() - 24.0, 18);
    draw_text(&sel_text, 12.0, y0 + CONSOLE_H - 40.0, 18.0, LIGHTGRAY);

    let help = "Space pause | S step | R reset | click select | +/- speed";
    draw_text(help, 12.0, y0 + CONSOLE_H - 16.0, 18.0, Color::new(0.55, 0.55, 0.6, 1.0));

    (start_clicked, stop_clicked)
}

#[macroquad::main(window_conf)]
async fn main() {
    let env = SnakeEnv::new(GRID_W, GRID_H);
    // Keep defaults from alpha_zero::snake/env.rs (rewards + anti-loop)

    let mut agents: Vec<Agent> = (0..AGENTS)
        .map(|i| Agent::new(&env, 0x5EED_u64 ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
        .collect();

    let mut paused = true;
    let mut speed = 1u32; // steps per frame
    let mut selected: usize = 0;

    let mut log_lines: VecDeque<String> = VecDeque::new();
    let mut last_logged_eps: u64 = 0;

    let mut train_ctl = TrainCtl::new();

    let mut policy: Option<SnakeOnnxPolicyValue> = None;
    // Use absolute path from executable location
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    
    // Try multiple possible locations for the model
    let possible_paths = vec![
        PathBuf::from("../alpha_zero/train/models/snake_policy_value.onnx"),
        PathBuf::from("modules/alpha_zero/train/models/snake_policy_value.onnx"),
        exe_dir.join("../../alpha_zero/train/models/snake_policy_value.onnx"),
        exe_dir.join("../../../modules/alpha_zero/train/models/snake_policy_value.onnx"),
    ];
    
    let mut model_path = PathBuf::from("../alpha_zero/train/models/snake_policy_value.onnx");
    for p in &possible_paths {
        if p.exists() {
            model_path = p.clone();
            break;
        }
    }

    let mut last_reload_check = std::time::Instant::now();
    let mut last_mod_time: Option<std::time::SystemTime> = None;
    let in_dim = (8 * GRID_W * GRID_H) as usize;

    log_lines.push_back(format!("[ui] model path: {:?} exists={}", model_path, model_path.exists()));

    loop {
        clear_background(Color::new(0.06, 0.06, 0.07, 1.0));

        if last_reload_check.elapsed().as_secs_f32() > 1.0 {
            last_reload_check = std::time::Instant::now();
            if model_path.exists() {
                if let Ok(meta) = std::fs::metadata(&model_path) {
                    if let Ok(mtime) = meta.modified() {
                        let changed = match last_mod_time {
                            Some(t) => mtime > t,
                            None => true,
                        };
                        if changed {
                            match SnakeOnnxPolicyValue::load(&model_path, in_dim) {
                                Ok(p) => {
                                    policy = Some(p);
                                    last_mod_time = Some(mtime);
                                    log_lines.push_back("[ui] model reloaded".to_string());
                                }
                                Err(e) => {
                                    log_lines.push_back(format!("[ui] model load failed: {e:?}"));
                                    while log_lines.len() > LOG_LINES_MAX {
                                        log_lines.pop_front();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // controls
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        if is_key_pressed(KeyCode::R) {
            for a in &mut agents {
                a.reset(&env);
            }
        }
        if is_key_pressed(KeyCode::Equal) || is_key_pressed(KeyCode::KpAdd) {
            speed = (speed + 1).min(20);
        }
        if is_key_pressed(KeyCode::Minus) || is_key_pressed(KeyCode::KpSubtract) {
            speed = speed.saturating_sub(1).max(1);
        }
        let single_step = is_key_pressed(KeyCode::S);

        if is_mouse_button_pressed(MouseButton::Left) {
            let (mx, my) = mouse_position();
            // ignore clicks in console area (buttons / log)
            if my < screen_height() - CONSOLE_H {
                let tile_w = screen_width() / COLS as f32;
                let usable_h = (screen_height() - CONSOLE_H).max(100.0);
                let tile_h = usable_h / ROWS as f32;
                let cx = (mx / tile_w).floor() as i32;
                let cy = (my / tile_h).floor() as i32;
                if cx >= 0 && cy >= 0 {
                    let idx = (cy as usize) * COLS + (cx as usize);
                    if idx < agents.len() {
                        selected = idx;
                    }
                }
            }
        }

        if !paused {
            for _ in 0..speed {
                for a in &mut agents {
                    a.step_once(&env, policy.as_ref());
                }
            }
        } else if single_step {
            for a in &mut agents {
                a.step_once(&env, policy.as_ref());
            }
        }

        // Append summary line every 10 completed episodes (across all agents)
        let mut total_eps: u64 = 0;
        let mut total_wins: u64 = 0;
        let mut sum_reward: f64 = 0.0;
        let mut sum_steps: u64 = 0;
        for a in &agents {
            total_eps += a.episodes as u64;
            total_wins += a.wins as u64;
            sum_reward += a.sum_reward as f64;
            sum_steps += a.sum_steps as u64;
        }
        if total_eps >= last_logged_eps + 10 {
            last_logged_eps = total_eps;
            let wr = if total_eps == 0 { 0.0 } else { total_wins as f64 / total_eps as f64 };
            let avg_r = if total_eps == 0 { 0.0 } else { sum_reward / total_eps as f64 };
            let avg_steps = if total_eps == 0 { 0.0 } else { sum_steps as f64 / total_eps as f64 };
            let status = if paused { "PAUSED" } else { "RUN" };
            let line = format!(
                "{} | agents={} | episodes={} | win_rate={:.2} | avg_return/ep={:.2} | avg_steps/ep={:.2} | speed={}",
                status,
                agents.len(),
                total_eps,
                wr,
                avg_r,
                avg_steps,
                speed
            );
            log_lines.push_back(line);
            while log_lines.len() > LOG_LINES_MAX {
                log_lines.pop_front();
            }
        }

        // Drain training output
        if let Some(rx) = train_ctl.rx.as_ref() {
            while let Ok(line) = rx.try_recv() {
                log_lines.push_back(line);
                while log_lines.len() > LOG_LINES_MAX {
                    log_lines.pop_front();
                }
            }
        }

        // layout
        let usable_h = (screen_height() - CONSOLE_H).max(100.0);
        let tile_w = screen_width() / COLS as f32;
        let tile_h = usable_h / ROWS as f32;

        for i in 0..agents.len() {
            let col = i % COLS;
            let row = i / COLS;
            let x0 = col as f32 * tile_w + 6.0;
            let y0 = row as f32 * tile_h + 6.0;
            let w = tile_w - 12.0;
            let h = tile_h - 12.0;
            draw_board(&env, &agents[i], x0, y0, w, h, i == selected);
        }

        let (start_clicked, stop_clicked) = draw_console(
            &agents,
            selected,
            &log_lines,
            train_ctl.running,
            !train_ctl.running,
            train_ctl.running,
            policy.is_some(),
        );
        if start_clicked {
            train_ctl.start(999999);
            paused = false; // Start moving agents so user sees activity
            log_lines.push_back("[ui] start train".to_string());
            while log_lines.len() > LOG_LINES_MAX {
                log_lines.pop_front();
            }
        }
        if stop_clicked {
            train_ctl.stop();
            log_lines.push_back("[ui] stop train".to_string());
            while log_lines.len() > LOG_LINES_MAX {
                log_lines.pop_front();
            }
        }

        next_frame().await;
    }
}
