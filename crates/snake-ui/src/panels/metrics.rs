use egui::RichText;
use egui_plot::{Line, Plot, PlotPoints};

use crate::bridge::metrics_client::MetricsSnapshot;

const WINDOW: usize = 300;

pub struct MetricsPanel {
    x: f64,
    env_reward: Vec<[f64; 2]>,
    avg_score: Vec<[f64; 2]>,
    total_loss: Vec<[f64; 2]>,
    intrinsic: Vec<[f64; 2]>,
}

impl MetricsPanel {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            env_reward: Vec::new(),
            avg_score: Vec::new(),
            total_loss: Vec::new(),
            intrinsic: Vec::new(),
        }
    }

    pub fn push(&mut self, metric: &MetricsSnapshot) {
        self.x += 1.0;
        self.env_reward.push([self.x, metric.avg_env_reward]);
        self.avg_score.push([self.x, metric.avg_score]);
        self.total_loss.push([self.x, metric.total_loss]);
        self.intrinsic.push([self.x, metric.avg_intrinsic_reward]);

        Self::trim(&mut self.env_reward);
        Self::trim(&mut self.avg_score);
        Self::trim(&mut self.total_loss);
        Self::trim(&mut self.intrinsic);
    }

    pub fn ui(&self, ui: &mut egui::Ui, latest: Option<&MetricsSnapshot>) {
        ui.label(RichText::new("Training Metrics").heading());

        if let Some(metric) = latest {
            ui.label(format!(
                "iter={} | stage={} | mode={} | loss={:.4} | env_reward={:.4} | score={:.3}",
                metric.iter,
                metric.stage_size,
                metric.mode,
                metric.total_loss,
                metric.avg_env_reward,
                metric.avg_score,
            ));
        } else {
            ui.label("Waiting for artifacts/metrics/latest.jsonl ...");
        }

        let spacing = 10.0;
        let plot_width = ((ui.available_width() - spacing).max(500.0)) / 2.0;
        let plot_height = 140.0;

        egui::Grid::new("metrics_grid").num_columns(2).show(ui, |ui| {
            Self::plot(ui, "Avg Env Reward", &self.env_reward, plot_width, plot_height);
            Self::plot(ui, "Avg Score", &self.avg_score, plot_width, plot_height);
            ui.end_row();
            Self::plot(ui, "Total Loss", &self.total_loss, plot_width, plot_height);
            Self::plot(ui, "Avg Intrinsic Reward", &self.intrinsic, plot_width, plot_height);
        });
    }

    fn trim(series: &mut Vec<[f64; 2]>) {
        if series.len() > WINDOW {
            let drop_count = series.len() - WINDOW;
            series.drain(0..drop_count);
        }
    }

    fn plot(ui: &mut egui::Ui, title: &str, series: &[[f64; 2]], width: f32, height: f32) {
        Plot::new(title).width(width).height(height).show(ui, |plot_ui| {
            let points = PlotPoints::new(series.to_vec());
            plot_ui.line(Line::new(points).name(title));
        });
    }
}
