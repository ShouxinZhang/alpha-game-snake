/// 网格坐标与索引转换工具
#[derive(Clone)]
pub struct Grid {
    pub width: i32,
    pub height: i32,
}

impl Grid {
    pub fn new(width: i32, height: i32) -> Self {
        Self { width, height }
    }

    pub fn cell_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    pub fn to_index(&self, cell: (i32, i32)) -> usize {
        let (x, y) = cell;
        (y as usize) * (self.width as usize) + (x as usize)
    }

    pub fn from_index(&self, idx: usize) -> (i32, i32) {
        let w = self.width as usize;
        ((idx % w) as i32, (idx / w) as i32)
    }

    pub fn in_bounds(&self, cell: (i32, i32)) -> bool {
        cell.0 >= 0 && cell.0 < self.width && cell.1 >= 0 && cell.1 < self.height
    }
}

/// 棋盘占用状态缓存，支持 O(1) 空闲格子随机选取
#[derive(Clone)]
pub struct BoardCache {
    grid: Grid,
    occupied: Vec<bool>,
    free_cells: Vec<usize>,
    free_pos: Vec<usize>,
}

impl BoardCache {
    pub fn new(grid: Grid) -> Self {
        let cell_count = grid.cell_count();
        Self {
            grid,
            occupied: vec![false; cell_count],
            free_cells: (0..cell_count).collect(),
            free_pos: (0..cell_count).collect(),
        }
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    pub fn resize(&mut self, width: i32, height: i32) {
        self.grid = Grid::new(width, height);
        let cell_count = self.grid.cell_count();
        self.occupied.clear();
        self.occupied.resize(cell_count, false);
        self.free_cells.clear();
        self.free_cells.reserve(cell_count);
        self.free_pos.clear();
        self.free_pos.resize(cell_count, usize::MAX);
        for idx in 0..cell_count {
            self.free_pos[idx] = self.free_cells.len();
            self.free_cells.push(idx);
        }
    }

    pub fn rebuild<'a>(&mut self, snake: impl Iterator<Item = &'a (i32, i32)>) {
        let cell_count = self.grid.cell_count();
        self.occupied.clear();
        self.occupied.resize(cell_count, false);

        for &cell in snake {
            let idx = self.grid.to_index(cell);
            self.occupied[idx] = true;
        }

        self.free_cells.clear();
        self.free_cells.reserve(cell_count);
        self.free_pos.clear();
        self.free_pos.resize(cell_count, usize::MAX);

        for idx in 0..cell_count {
            if !self.occupied[idx] {
                self.free_pos[idx] = self.free_cells.len();
                self.free_cells.push(idx);
            }
        }
    }

    pub fn is_occupied(&self, idx: usize) -> bool {
        self.occupied[idx]
    }

    pub fn set_occupied(&mut self, idx: usize) {
        if self.occupied[idx] {
            return;
        }
        self.occupied[idx] = true;
        let pos = self.free_pos[idx];
        if pos == usize::MAX {
            return;
        }
        let last = *self
            .free_cells
            .last()
            .expect("free_cells must contain idx when free_pos is set");
        self.free_cells.swap_remove(pos);
        self.free_pos[idx] = usize::MAX;
        if pos < self.free_cells.len() {
            self.free_pos[last] = pos;
        }
    }

    pub fn set_empty(&mut self, idx: usize) {
        if !self.occupied[idx] {
            return;
        }
        self.occupied[idx] = false;
        if self.free_pos[idx] != usize::MAX {
            return;
        }
        self.free_pos[idx] = self.free_cells.len();
        self.free_cells.push(idx);
    }

    pub fn free_cells(&self) -> &[usize] {
        &self.free_cells
    }

    pub fn random_free_cell(&self, rand_index: usize) -> Option<(i32, i32)> {
        if self.free_cells.is_empty() {
            return None;
        }
        let idx = self.free_cells[rand_index % self.free_cells.len()];
        Some(self.grid.from_index(idx))
    }
}
