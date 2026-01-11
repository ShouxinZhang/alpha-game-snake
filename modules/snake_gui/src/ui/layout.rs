pub const CONSOLE_HEIGHT: f32 = 120.0;

/// 游戏棋盘布局信息
pub struct BoardLayout {
    pub origin_x: f32,
    pub origin_y: f32,
    pub width: f32,
    pub height: f32,
    pub cell_w: f32,
    pub cell_h: f32,
}

impl BoardLayout {
    pub fn compute(screen_w: f32, screen_h: f32, grid_width: i32, grid_height: i32) -> Self {
        let usable_h = (screen_h - CONSOLE_HEIGHT).max(1.0);
        let width = screen_w;
        let height = usable_h;
        let cell_w = width / grid_width as f32;
        let cell_h = height / grid_height as f32;

        Self {
            origin_x: 0.0,
            origin_y: 0.0,
            width,
            height,
            cell_w,
            cell_h,
        }
    }

    pub fn cell_rect(&self, cell: (i32, i32)) -> (f32, f32, f32, f32) {
        let (x, y) = cell;
        let px = self.origin_x + x as f32 * self.cell_w;
        let py = self.origin_y + y as f32 * self.cell_h;
        let w = (self.cell_w - 2.0).max(1.0);
        let h = (self.cell_h - 2.0).max(1.0);
        (px + 1.0, py + 1.0, w, h)
    }
}

/// 控制台面板布局信息
pub struct ConsoleLayout {
    pub origin_x: f32,
    pub origin_y: f32,
    pub width: f32,
    pub height: f32,
}

impl ConsoleLayout {
    pub fn compute(screen_w: f32, screen_h: f32) -> Self {
        Self {
            origin_x: 0.0,
            origin_y: (screen_h - CONSOLE_HEIGHT).max(0.0),
            width: screen_w,
            height: CONSOLE_HEIGHT.min(screen_h),
        }
    }
}
