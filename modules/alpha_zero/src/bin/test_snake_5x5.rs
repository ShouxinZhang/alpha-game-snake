use alpha_zero::policy::UniformPolicyValue;
use alpha_zero::snake::{SnakeEnv, SnakeStatus};
use alpha_zero::{Environment, Mcts, MctsConfig};

fn parse_arg_u64(args: &[String], key: &str, default: u64) -> u64 {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn main() {
    // Example:
    // cargo run --bin test_snake_5x5 -- --games 50 --sim 200
    let args: Vec<String> = std::env::args().collect();
    let games = parse_arg_u64(&args, "--games", 50) as usize;
    let sim = parse_arg_u64(&args, "--sim", 200) as usize;

    let env = SnakeEnv::new(5, 5);
    let mcts = Mcts::new(MctsConfig {
        num_simulations: sim,
        c_puct: 1.5,
    });
    let pv = UniformPolicyValue;

    let mut win = 0usize;
    let mut lose = 0usize;
    let mut total_return = 0.0f32;
    let mut total_steps = 0usize;
    let mut total_final_len = 0usize;
    let mut total_score = 0u32;

    for g in 0..games {
        let mut s = env.reset(0xC0FFEE_u64 ^ (g as u64));
        let mut ret = 0.0f32;
        let mut steps = 0usize;

        while s.status == SnakeStatus::Running {
            let a = mcts
                .select_action(&env, &pv, &s, 0.0, &mut rand::thread_rng())
                .expect("no action");
            let (ns, r) = env.step(&s, a);
            s = ns;
            ret += r;
            steps += 1;
            if steps > 10_000 {
                break;
            }
        }

        total_return += ret;
        total_steps += steps;
        total_final_len += s.snake.len();
        total_score += s.score;

        match s.status {
            SnakeStatus::Victory => win += 1,
            SnakeStatus::GameOver => lose += 1,
            SnakeStatus::Running => {}
        }

        if (g + 1) % 10 == 0 {
            let n = (g + 1) as f32;
            println!(
                "games={} avg_reward={:.3} avg_steps={:.2} avg_final_len={:.2} avg_score={:.2} win_rate={:.2}",
                g + 1,
                total_return / n,
                (total_steps as f32) / n,
                (total_final_len as f32) / n,
                (total_score as f32) / n,
                (win as f32) / n
            );
        }
    }

    println!(
        "done games={} win={} lose={} avg_reward={:.3} avg_steps={:.2} avg_final_len={:.2} avg_score={:.2}",
        games,
        win,
        lose,
        total_return / (games as f32),
        (total_steps as f32) / (games as f32),
        (total_final_len as f32) / (games as f32),
        (total_score as f32) / (games as f32)
    );
}
