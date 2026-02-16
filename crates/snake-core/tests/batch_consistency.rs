use snake_core::{Action, BatchConfig, BatchEnv, GameConfig, SnakeEnv};

#[test]
fn single_env_and_batch_env_should_be_consistent_under_same_seed() {
    let config = GameConfig {
        board_w: 9,
        board_h: 9,
        max_steps: 64,
        seed: 42,
    };

    let mut single = SnakeEnv::new(config.clone(), config.seed);
    let mut batch = BatchEnv::new(
        config,
        BatchConfig {
            num_envs: 1,
            num_threads: 1,
        },
    );

    let _ = single.reset();
    let _ = batch.reset();

    let actions = [
        Action::Right,
        Action::Down,
        Action::Down,
        Action::Left,
        Action::Up,
        Action::Right,
    ];

    for action in actions {
        let single_out = single.step(action);
        let batch_out = batch.step(&[action]);

        assert_eq!(single_out.done, batch_out.dones[0]);
        assert_eq!(single_out.score, batch_out.scores[0]);
        assert!((single_out.reward - batch_out.rewards[0]).abs() < 1e-6);
    }
}
