use egui::{Color32, RichText, Stroke};

pub struct FocusPanel {
    last_policy: [f32; 4],
}

impl FocusPanel {
    pub fn new() -> Self {
        Self {
            last_policy: [0.25; 4],
        }
    }

    pub fn set_policy(&mut self, policy: [f32; 4]) {
        self.last_policy = policy;
    }

    pub fn ui(&self, ui: &mut egui::Ui, obs: &[f32], shape: [usize; 4], selected: usize) {
        let channels = shape[1];
        let board_h = shape[2];
        let board_w = shape[3];
        let env_obs_size = channels * board_h * board_w;
        let plane = board_h * board_w;

        ui.label(RichText::new("Selected Agent Focus (ViT + RL State)").heading());

        let frame = egui::Frame::default()
            .stroke(Stroke::new(1.0, Color32::from_gray(80)))
            .fill(Color32::from_rgb(16, 25, 35))
            .inner_margin(egui::Margin::same(8.0));

        frame.show(ui, |ui| {
            let desired = egui::vec2(420.0, 220.0);
            let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
            let painter = ui.painter_at(rect);

            let board_rect = egui::Rect::from_min_size(rect.min, egui::vec2(280.0, 220.0));
            let cell_w = board_rect.width() / board_w as f32;
            let cell_h = board_rect.height() / board_h as f32;

            let start = selected * env_obs_size;
            for y in 0..board_h {
                for x in 0..board_w {
                    let idx = y * board_w + x;
                    let head = obs[start + idx] > 0.5;
                    let body = obs[start + plane + idx] > 0.5;
                    let food = obs[start + 2 * plane + idx] > 0.5;

                    let color = if head {
                        Color32::from_rgb(254, 212, 116)
                    } else if body {
                        Color32::from_rgb(255, 109, 67)
                    } else if food {
                        Color32::from_rgb(255, 240, 170)
                    } else {
                        Color32::from_rgb(56, 17, 25)
                    };

                    let x0 = board_rect.left() + x as f32 * cell_w;
                    let y0 = board_rect.top() + y as f32 * cell_h;
                    let cell_rect = egui::Rect::from_min_size(
                        egui::pos2(x0, y0),
                        egui::vec2(cell_w - 1.0, cell_h - 1.0),
                    );
                    painter.rect_filled(cell_rect, 1.0, color);
                }
            }

            let mut saliency = vec![0.02_f32; plane];
            let mut head_pos: Option<(usize, usize)> = None;
            let mut food_pos: Option<(usize, usize)> = None;
            for y in 0..board_h {
                for x in 0..board_w {
                    let idx = y * board_w + x;
                    let head = obs[start + idx] > 0.5;
                    let body = obs[start + plane + idx] > 0.5;
                    let food = obs[start + 2 * plane + idx] > 0.5;
                    if body {
                        saliency[idx] += 0.30;
                    }
                    if head {
                        saliency[idx] += 0.90;
                        head_pos = Some((x, y));
                    }
                    if food {
                        saliency[idx] += 1.0;
                        food_pos = Some((x, y));
                    }
                }
            }

            if let Some((hx, hy)) = head_pos {
                for y in 0..board_h {
                    for x in 0..board_w {
                        let idx = y * board_w + x;
                        let dist = hx.abs_diff(x) + hy.abs_diff(y);
                        saliency[idx] += 0.20 / (1.0 + dist as f32);
                    }
                }
            }
            if let Some((fx, fy)) = food_pos {
                for y in 0..board_h {
                    for x in 0..board_w {
                        let idx = y * board_w + x;
                        let dist = fx.abs_diff(x) + fy.abs_diff(y);
                        saliency[idx] += 0.35 / (1.0 + dist as f32);
                    }
                }
            }

            let mut max_saliency = 1e-6_f32;
            for value in &saliency {
                if *value > max_saliency {
                    max_saliency = *value;
                }
            }

            let heatmap_rect = egui::Rect::from_min_size(
                egui::pos2(board_rect.right() + 12.0, board_rect.top()),
                egui::vec2(126.0, 126.0),
            );
            painter.rect_filled(heatmap_rect, 2.0, Color32::from_rgb(45, 20, 20));

            let heat_cell_w = heatmap_rect.width() / board_w as f32;
            let heat_cell_h = heatmap_rect.height() / board_h as f32;
            for y in 0..board_h {
                for x in 0..board_w {
                    let idx = y * board_w + x;
                    let p = (saliency[idx] / max_saliency).clamp(0.0, 1.0);
                    let r = (35.0 + p * 220.0) as u8;
                    let g = (10.0 + p * 140.0) as u8;
                    let b = (20.0 + p * 40.0) as u8;
                    let block = egui::Rect::from_min_size(
                        egui::pos2(
                            heatmap_rect.left() + x as f32 * heat_cell_w,
                            heatmap_rect.top() + y as f32 * heat_cell_h,
                        ),
                        egui::vec2(heat_cell_w - 1.0, heat_cell_h - 1.0),
                    );
                    painter.rect_filled(block, 1.0, Color32::from_rgb(r, g, b));
                }
            }

            painter.text(
                egui::pos2(heatmap_rect.left(), heatmap_rect.bottom() + 12.0),
                egui::Align2::LEFT_TOP,
                format!(
                    "UP {:.0}%  DOWN {:.0}%\nLEFT {:.0}% RIGHT {:.0}%",
                    self.last_policy[0] * 100.0,
                    self.last_policy[1] * 100.0,
                    self.last_policy[2] * 100.0,
                    self.last_policy[3] * 100.0,
                ),
                egui::FontId::monospace(12.0),
                Color32::LIGHT_BLUE,
            );
            painter.text(
                egui::pos2(heatmap_rect.left(), heatmap_rect.top() - 4.0),
                egui::Align2::LEFT_BOTTOM,
                "State Heatmap",
                egui::FontId::monospace(11.0),
                Color32::GRAY,
            );
        });
    }
}
