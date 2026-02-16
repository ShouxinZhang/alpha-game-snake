mod panels {
    pub mod env_grid;
    pub mod focus_agent;
}

use std::time::{Duration, Instant};

use anyhow::Result;
use eframe::egui::{self, Color32, RichText};
use rand::Rng;

use crate::panels::env_grid::draw_env_grid;
use crate::panels::focus_agent::FocusPanel;
use snake_core::{Action, BatchConfig, BatchEnv, GameConfig};

fn main() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1460.0, 820.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Snake Game Monitor",
        options,
        Box::new(|_| Ok(Box::new(GameMonitorApp::default()))),
    )
    .map_err(|err| anyhow::anyhow!("eframe run failed: {err}"))
}

struct GameMonitorApp {
    env: BatchEnv,
    selected_env: usize,
    scores: Vec<i32>,
    running: bool,
    sleep_mode: bool,
    sim_speed: f32,
    last_tick: Instant,
    total_steps: u64,
    obs: Vec<f32>,
    shape: [usize; 4],
    focus_panel: FocusPanel,
    episodes_finished: u64,
    avg_score: f64,
    rolling_reward: f64,
}

impl Default for GameMonitorApp {
    fn default() -> Self {
        let mut env = BatchEnv::new(
            GameConfig {
                board_w: 12,
                board_h: 12,
                max_steps: 512,
                seed: 7,
            },
            BatchConfig {
                num_envs: 16,
                num_threads: 24,
            },
        );
        let obs = env.reset();

        Self {
            env,
            selected_env: 0,
            scores: vec![0; 16],
            running: true,
            sleep_mode: false,
            sim_speed: 8.0,
            last_tick: Instant::now(),
            total_steps: 0,
            obs: obs.obs,
            shape: obs.shape,
            focus_panel: FocusPanel::new(),
            episodes_finished: 0,
            avg_score: 0.0,
            rolling_reward: 0.0,
        }
    }
}

impl eframe::App for GameMonitorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.tick_simulation();

        egui::TopBottomPanel::top("top_controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .add(egui::Button::new(RichText::new("▶ Run").color(Color32::GREEN)))
                    .clicked()
                {
                    self.running = true;
                    self.sleep_mode = false;
                }
                if ui.button("⏸ Pause All").clicked() {
                    self.running = false;
                }
                if ui.button("☾ Sleep Mode").clicked() {
                    self.sleep_mode = !self.sleep_mode;
                }
                ui.separator();
                ui.label(format!("Simulation Speed: {:.1}x", self.sim_speed));
                ui.add(egui::Slider::new(&mut self.sim_speed, 1.0..=20.0));
            });
        });

        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(Color32::from_rgb(7, 15, 30)))
            .show(ctx, |ui| {
                ui.horizontal_top(|ui| {
                    ui.vertical(|ui| {
                        draw_env_grid(
                            ui,
                            &self.obs,
                            self.shape,
                            &mut self.selected_env,
                            &self.scores,
                        );
                    });
                    ui.add_space(12.0);
                    ui.vertical(|ui| {
                        self.focus_panel
                            .ui(ui, &self.obs, self.shape, self.selected_env);
                    });
                });
            });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Total Steps: {} | Active Envs: {} | Selected Env: {} | Episodes Finished: {} | Avg Score: {:.2} | Rolling Reward: {:.4}",
                    self.total_steps,
                    self.shape[0],
                    self.selected_env,
                    self.episodes_finished,
                    self.avg_score,
                    self.rolling_reward
                ));
            });
        });

        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

impl GameMonitorApp {
    fn tick_simulation(&mut self) {
        if !self.running {
            return;
        }
        if self.sleep_mode {
            std::thread::sleep(Duration::from_millis(16));
        }

        let interval = Duration::from_millis((1000.0 / self.sim_speed.max(1.0)) as u64);
        if self.last_tick.elapsed() < interval {
            return;
        }
        self.last_tick = Instant::now();

        let mut rng = rand::thread_rng();
        let mut actions = Vec::with_capacity(self.shape[0]);
        for _ in 0..self.shape[0] {
            actions.push(match rng.gen_range(0..4) {
                0 => Action::Up,
                1 => Action::Down,
                2 => Action::Left,
                _ => Action::Right,
            });
        }

        let step_out = self.env.step(&actions);
        self.total_steps += self.shape[0] as u64;

        self.obs = step_out.obs;
        self.shape = step_out.shape;
        self.scores = step_out.scores.clone();

        let reward_mean = if !step_out.rewards.is_empty() {
            step_out.rewards.iter().copied().sum::<f32>() as f64 / step_out.rewards.len() as f64
        } else {
            0.0
        };

        self.rolling_reward = self.rolling_reward * 0.98 + reward_mean * 0.02;
        self.avg_score = if self.scores.is_empty() {
            0.0
        } else {
            self.scores.iter().copied().sum::<i32>() as f64 / self.scores.len() as f64
        };

        if let Some(mask) = step_out.legal_actions_mask.get(self.selected_env) {
            let mut norm = [0.0_f32; 4];
            let mut sum = 0.0;
            for (i, v) in mask.iter().enumerate() {
                norm[i] = *v as f32;
                sum += norm[i];
            }
            if sum > 0.0 {
                for v in &mut norm {
                    *v /= sum;
                }
            }
            self.focus_panel.set_policy(norm);
        }

        let reset_ids = self.env.reset_done();
        self.episodes_finished += reset_ids.len() as u64;
    }
}
