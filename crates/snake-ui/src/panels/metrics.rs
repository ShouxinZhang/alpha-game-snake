use egui::RichText;
use egui_plot::{Line, Plot, PlotPoints};

use crate::bridge::metrics_client::MetricsSnapshot;

const WINDOW: usize = 300;

pub struct MetricsPanel {
    rolling_reward: Vec<[f64; 2]>,
    avg_score: Vec<[f64; 2]>,
    avg_steps: Vec<[f64; 2]>,
    loss: Vec<[f64; 2]>,
    x: f64,
}

impl MetricsPanel {
    pub fn new() -> Self {
        Self {
            rolling_reward: Vec::new(),
            avg_score: Vec::new(),
            avg_steps: Vec::new(),
            loss: Vec::new(),
            x: 0.0,
        }
    }

    pub fn push(&mut self, metric: MetricsSnapshot) {
        self.x += 1.0;
        self.rolling_reward.push([self.x, metric.rolling_avg_reward]);
        self.avg_score.push([self.x, metric.avg_score]);
        self.avg_steps.push([self.x, metric.avg_steps]);
        self.loss.push([self.x, metric.loss]);

        Self::trim(&mut self.rolling_reward);
        Self::trim(&mut self.avg_score);
        Self::trim(&mut self.avg_steps);
        Self::trim(&mut self.loss);
    }

    pub fn ui(&self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Aggregate Training Metrics").heading());
        let spacing = 10.0;
        let plot_width = ((ui.available_width() - spacing).max(440.0)) / 2.0;
        let plot_height = 125.0;
        egui::Grid::new("metrics_grid").num_columns(2).show(ui, |ui| {
            Self::plot(ui, "Rolling Avg Reward", &self.rolling_reward, plot_width, plot_height);
            Self::plot(ui, "Avg Score", &self.avg_score, plot_width, plot_height);
            ui.end_row();
            Self::plot(ui, "Avg Survival Steps", &self.avg_steps, plot_width, plot_height);
            Self::plot(ui, "Loss", &self.loss, plot_width, plot_height);
        });
    }

    fn trim(series: &mut Vec<[f64; 2]>) {
        if series.len() > WINDOW {
            let drop_count = series.len() - WINDOW;
            series.drain(0..drop_count);
        }
    }

    fn plot(ui: &mut egui::Ui, title: &str, series: &[[f64; 2]], width: f32, height: f32) {
        Plot::new(title)
            .width(width)
            .height(height)
            .show(ui, |plot_ui| {
                let points = PlotPoints::new(series.to_vec());
                plot_ui.line(Line::new(points).name(title));
            });
    }
}
