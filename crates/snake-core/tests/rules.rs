use snake_core::{Action, GameConfig, Point, SnakeEnv};

#[test]
fn wall_collision_should_end_episode() {
    let config = GameConfig {
        board_w: 5,
        board_h: 5,
        max_steps: 100,
        seed: 1,
    };
    let mut env = SnakeEnv::new(config, 1);

    let mut done = false;
    for _ in 0..10 {
        let out = env.step(Action::Right);
        done = out.done;
        if done {
            assert!(out.reward < 0.0);
            break;
        }
    }

    assert!(done, "snake should collide with wall in a small board");
}

#[test]
fn eating_food_should_increase_score() {
    let config = GameConfig {
        board_w: 7,
        board_h: 7,
        max_steps: 100,
        seed: 2,
    };
    let mut env = SnakeEnv::new(config, 2);

    env.debug_set_snake(
        &[
            Point { x: 2, y: 3 },
            Point { x: 3, y: 3 },
            Point { x: 4, y: 3 },
        ],
        Action::Right,
    );
    env.debug_set_food(5, 3);

    let out = env.step(Action::Right);
    assert_eq!(out.score, 1);
    assert!(out.reward > 0.0);
    assert!(!out.done);
}

#[test]
fn self_collision_should_end_episode() {
    let config = GameConfig {
        board_w: 8,
        board_h: 8,
        max_steps: 100,
        seed: 3,
    };
    let mut env = SnakeEnv::new(config, 3);

    env.debug_set_snake(
        &[
            Point { x: 3, y: 4 },
            Point { x: 3, y: 5 },
            Point { x: 4, y: 5 },
            Point { x: 5, y: 5 },
            Point { x: 5, y: 4 },
            Point { x: 4, y: 4 },
        ],
        Action::Left,
    );

    let out = env.step(Action::Down);
    assert!(out.done, "moving into own body must terminate the episode");
    assert!(out.reward < 0.0);
}

#[test]
fn reverse_action_should_be_blocked() {
    let config = GameConfig {
        board_w: 8,
        board_h: 8,
        max_steps: 100,
        seed: 4,
    };
    let mut env = SnakeEnv::new(config, 4);

    let out = env.step(Action::Left);
    assert!(!out.done);
    assert_eq!(out.legal_actions_mask[Action::Left.index()], 0);
}
