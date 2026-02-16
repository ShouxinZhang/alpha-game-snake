use egui::{Color32, RichText, Sense, Stroke};

pub fn draw_env_grid(
    ui: &mut egui::Ui,
    obs: &[f32],
    shape: [usize; 4],
    selected: &mut usize,
    scores: &[i32],
) {
    let num_envs = shape[0];
    let channels = shape[1];
    let board_h = shape[2];
    let board_w = shape[3];
    let env_obs_size = channels * board_h * board_w;

    let cols = (num_envs as f32).sqrt().ceil() as usize;
    ui.label(RichText::new(format!("Concurrent Environments ({} Threads)", num_envs)).heading());

    egui::Grid::new("env_grid")
        .num_columns(cols)
        .spacing([8.0, 8.0])
        .show(ui, |ui| {
            for env_idx in 0..num_envs {
                let frame = egui::Frame::default()
                    .stroke(if *selected == env_idx {
                        Stroke::new(2.0, Color32::from_rgb(241, 196, 15))
                    } else {
                        Stroke::new(1.0, Color32::from_gray(80))
                    })
                    .fill(Color32::from_rgb(7, 19, 38))
                    .inner_margin(egui::Margin::same(4.0));

                frame.show(ui, |ui| {
                    let desired_size = egui::vec2(110.0, 110.0);
                    let (rect, resp) = ui.allocate_exact_size(desired_size, Sense::click());
                    if resp.clicked() {
                        *selected = env_idx;
                    }

                    let painter = ui.painter_at(rect);
                    let cell_w = rect.width() / board_w as f32;
                    let cell_h = rect.height() / board_h as f32;
                    let start = env_idx * env_obs_size;
                    let plane = board_w * board_h;

                    for y in 0..board_h {
                        for x in 0..board_w {
                            let idx = y * board_w + x;
                            let head = obs[start + idx] > 0.5;
                            let body = obs[start + plane + idx] > 0.5;
                            let food = obs[start + 2 * plane + idx] > 0.5;

                            let color = if head {
                                Color32::from_rgb(160, 255, 130)
                            } else if body {
                                Color32::from_rgb(68, 176, 94)
                            } else if food {
                                Color32::from_rgb(255, 138, 101)
                            } else {
                                Color32::from_rgb(11, 25, 48)
                            };

                            let x0 = rect.left() + x as f32 * cell_w;
                            let y0 = rect.top() + y as f32 * cell_h;
                            let cell_rect = egui::Rect::from_min_size(
                                egui::pos2(x0, y0),
                                egui::vec2(cell_w - 1.0, cell_h - 1.0),
                            );
                            painter.rect_filled(cell_rect, 1.0, color);
                        }
                    }
                });

                let score = scores.get(env_idx).copied().unwrap_or_default();
                ui.label(format!("Env #{env_idx} | Score {score}"));

                if (env_idx + 1) % cols == 0 {
                    ui.end_row();
                }
            }
        });
}
