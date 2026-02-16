use rayon::prelude::*;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;

use crate::env::SnakeEnv;
use crate::types::{Action, BatchConfig, BatchObs, BatchStepOutput, GameConfig, OBS_CHANNELS};

#[derive(Debug)]
pub struct BatchEnv {
    config: GameConfig,
    batch_config: BatchConfig,
    envs: Vec<SnakeEnv>,
    thread_pool: ThreadPool,
}

impl BatchEnv {
    pub fn new(config: GameConfig, batch_config: BatchConfig) -> Self {
        let thread_count = batch_config.num_threads.max(1);
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .thread_name(|idx| format!("snake-core-worker-{idx}"))
            .build()
            .expect("failed to build snake core rayon thread pool");

        let envs = (0..batch_config.num_envs)
            .map(|idx| SnakeEnv::new(config.clone(), config.seed + idx as u64))
            .collect();

        Self {
            config,
            batch_config,
            envs,
            thread_pool,
        }
    }

    pub fn reset(&mut self) -> BatchObs {
        let obs_parts = self
            .thread_pool
            .install(|| self.envs.par_iter_mut().map(SnakeEnv::reset).collect::<Vec<_>>());
        self.flatten_obs(obs_parts)
    }

    pub fn step(&mut self, actions: &[Action]) -> BatchStepOutput {
        assert_eq!(
            actions.len(),
            self.batch_config.num_envs,
            "actions.len must equal num_envs"
        );

        let outputs = self.thread_pool.install(|| {
            self.envs
                .par_iter_mut()
                .zip(actions.par_iter().copied())
                .map(|(env, action)| env.step(action))
                .collect::<Vec<_>>()
        });

        let plane_size = self.config.board_w * self.config.board_h;
        let single_obs_size = OBS_CHANNELS * plane_size;

        let mut obs = Vec::with_capacity(outputs.len() * single_obs_size);
        let mut rewards = Vec::with_capacity(outputs.len());
        let mut dones = Vec::with_capacity(outputs.len());
        let mut scores = Vec::with_capacity(outputs.len());
        let mut legal_actions_mask = Vec::with_capacity(outputs.len());

        for output in outputs {
            obs.extend(output.obs);
            rewards.push(output.reward);
            dones.push(output.done);
            scores.push(output.score);
            legal_actions_mask.push(output.legal_actions_mask);
        }

        BatchStepOutput {
            obs,
            shape: [
                self.batch_config.num_envs,
                OBS_CHANNELS,
                self.config.board_h,
                self.config.board_w,
            ],
            rewards,
            dones,
            scores,
            legal_actions_mask,
        }
    }

    pub fn reset_done(&mut self) -> Vec<usize> {
        let mut ids = Vec::new();
        for (idx, env) in self.envs.iter_mut().enumerate() {
            if env.done() {
                let _ = env.reset();
                ids.push(idx);
            }
        }
        ids
    }

    pub fn get_obs_tensor(&self) -> BatchObs {
        let obs_parts = self
            .thread_pool
            .install(|| self.envs.par_iter().map(SnakeEnv::observation).collect::<Vec<_>>());
        self.flatten_obs(obs_parts)
    }

    pub fn seed(&mut self, seed: u64) {
        for (idx, env) in self.envs.iter_mut().enumerate() {
            env.reseed(seed + idx as u64);
        }
    }

    pub fn config(&self) -> &GameConfig {
        &self.config
    }

    pub fn num_envs(&self) -> usize {
        self.batch_config.num_envs
    }

    fn flatten_obs(&self, obs_parts: Vec<Vec<f32>>) -> BatchObs {
        let plane_size = self.config.board_w * self.config.board_h;
        let single_obs_size = OBS_CHANNELS * plane_size;
        let mut obs = Vec::with_capacity(self.batch_config.num_envs * single_obs_size);
        for part in obs_parts {
            obs.extend(part);
        }
        BatchObs {
            obs,
            shape: [
                self.batch_config.num_envs,
                OBS_CHANNELS,
                self.config.board_h,
                self.config.board_w,
            ],
        }
    }
}
