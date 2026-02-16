#![cfg(feature = "python")]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::batch::BatchEnv;
use crate::types::{Action, BatchConfig, GameConfig, OBS_CHANNELS};

#[pyclass(name = "RustBatchEnv")]
pub struct PyRustBatchEnv {
    inner: BatchEnv,
}

#[pymethods]
impl PyRustBatchEnv {
    #[new]
    #[pyo3(signature = (board_w, board_h, max_steps, seed, num_envs, num_threads))]
    fn new(
        board_w: usize,
        board_h: usize,
        max_steps: u32,
        seed: u64,
        num_envs: usize,
        num_threads: usize,
    ) -> Self {
        let config = GameConfig {
            board_w,
            board_h,
            max_steps,
            seed,
        };
        let batch = BatchConfig {
            num_envs,
            num_threads,
        };
        Self {
            inner: BatchEnv::new(config, batch),
        }
    }

    fn reset(&mut self) -> (Vec<f32>, (usize, usize, usize, usize)) {
        let batch_obs = self.inner.reset();
        (
            batch_obs.obs,
            (
                batch_obs.shape[0],
                batch_obs.shape[1],
                batch_obs.shape[2],
                batch_obs.shape[3],
            ),
        )
    }

    fn step(
        &mut self,
        actions: Vec<i64>,
    ) -> PyResult<(
        Vec<f32>,
        (usize, usize, usize, usize),
        Vec<f32>,
        Vec<bool>,
        Vec<i32>,
        Vec<u8>,
    )> {
        let mut decoded = Vec::with_capacity(actions.len());
        for act in actions {
            let action = Action::from_index(act)
                .ok_or_else(|| PyValueError::new_err(format!("invalid action: {act}")))?;
            decoded.push(action);
        }

        let result = self.inner.step(&decoded);
        let mut legal_flat = Vec::with_capacity(result.legal_actions_mask.len() * 4);
        for mask in result.legal_actions_mask {
            legal_flat.extend(mask);
        }

        Ok((
            result.obs,
            (
                result.shape[0],
                result.shape[1],
                result.shape[2],
                result.shape[3],
            ),
            result.rewards,
            result.dones,
            result.scores,
            legal_flat,
        ))
    }

    fn reset_done(&mut self) -> Vec<usize> {
        self.inner.reset_done()
    }

    fn get_obs_tensor(&self) -> (Vec<f32>, (usize, usize, usize, usize)) {
        let batch_obs = self.inner.get_obs_tensor();
        (
            batch_obs.obs,
            (
                batch_obs.shape[0],
                batch_obs.shape[1],
                batch_obs.shape[2],
                batch_obs.shape[3],
            ),
        )
    }

    fn seed(&mut self, seed: u64) {
        self.inner.seed(seed);
    }

    fn obs_channels(&self) -> usize {
        OBS_CHANNELS
    }
}

#[pymodule]
fn snake_core_py(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRustBatchEnv>()?;
    module.add("OBS_CHANNELS", OBS_CHANNELS)?;
    Ok(())
}
