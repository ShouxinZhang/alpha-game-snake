use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use alpha_zero::snake::{SnakeAction, SnakeEnv, SnakeStatus};
use alpha_zero::{Environment, Mcts, MctsConfig, RootPolicy, SnakeOnnxPolicyValue, UniformPolicyValue};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

#[derive(Clone)]
struct Sample {
    features: Vec<f32>,
    pi: [f32; 4],
    reward: f32,
    z: f32,
}

fn parse_arg_string(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn parse_arg_u64(args: &[String], key: &str, default: u64) -> u64 {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_arg_f32(args: &[String], key: &str, default: f32) -> f32 {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn action_to_index(a: SnakeAction) -> usize {
    match a {
        SnakeAction::Up => 0,
        SnakeAction::Down => 1,
        SnakeAction::Left => 2,
        SnakeAction::Right => 3,
    }
}

fn index_to_action(i: usize) -> SnakeAction {
    match i {
        0 => SnakeAction::Up,
        1 => SnakeAction::Down,
        2 => SnakeAction::Left,
        _ => SnakeAction::Right,
    }
}

fn root_policy_to_pi(policy: RootPolicy<SnakeAction>) -> [f32; 4] {
    let mut pi = [0.0f32; 4];
    let total: u32 = policy.visits.iter().map(|(_, n)| *n).sum();
    if total == 0 {
        return [0.25; 4];
    }
    for (a, n) in policy.visits {
        pi[action_to_index(a)] = (n as f32) / (total as f32);
    }
    pi
}

fn apply_temperature(mut pi: [f32; 4], temp: f32) -> [f32; 4] {
    if temp <= 0.0 {
        // greedy
        let mut best = 0usize;
        for i in 1..4 {
            if pi[i] > pi[best] {
                best = i;
            }
        }
        let mut out = [0.0f32; 4];
        out[best] = 1.0;
        return out;
    }
    if (temp - 1.0).abs() < f32::EPSILON {
        return pi;
    }
    let inv_t = 1.0 / temp;
    for p in &mut pi {
        *p = p.powf(inv_t);
    }
    let s: f32 = pi.iter().sum();
    if s > 0.0 {
        for p in &mut pi {
            *p /= s;
        }
    }
    pi
}

fn sample_action_from_pi(pi: [f32; 4], rng: &mut impl Rng) -> SnakeAction {
    let weights: Vec<f32> = pi.to_vec();
    let dist = WeightedIndex::new(&weights).unwrap_or_else(|_| WeightedIndex::new(vec![1.0; 4]).unwrap());
    index_to_action(dist.sample(rng))
}

fn main() {
    // Example:
    // cargo run --bin selfplay_snake -- --out train/data/snake_5x5.csv --games 200 --sim 200 --temp 1.0 --threads 32 --w 5 --h 5
    let args: Vec<String> = std::env::args().collect();
    let out = parse_arg_string(&args, "--out", "snake.csv");
    let games = parse_arg_u64(&args, "--games", 200) as usize;
    let sim = parse_arg_u64(&args, "--sim", 200) as usize;
    let temp = parse_arg_f32(&args, "--temp", 1.0);
    let threads = parse_arg_u64(&args, "--threads", 32) as usize;
    let w = parse_arg_u64(&args, "--w", 5) as i32;
    let h = parse_arg_u64(&args, "--h", 5) as i32;
    let onnx_path = parse_arg_string(&args, "--onnx", "");

    let out_path = PathBuf::from(out);
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).expect("create output dir");
        }
    }

    let f = fs::File::create(&out_path).expect("create output file");
    let mut wtr = BufWriter::new(f);

    let in_dim = (8 * (w as usize) * (h as usize)) as usize;

    // header
    for i in 0..in_dim {
        write!(wtr, "f{}{}", i, if i + 1 == in_dim { "," } else { "," }).unwrap();
    }
    writeln!(wtr, "pi0,pi1,pi2,pi3,z").unwrap();

    let threads = threads.max(1).min(games.max(1));
    let games_per_thread = (games + threads - 1) / threads;

    // (csv_lines, rows, return, steps, win)
    let (tx, rx) = mpsc::channel::<(String, usize, f32, usize, bool)>();

    for t in 0..threads {
        let tx = tx.clone();
        let onnx_path = onnx_path.clone();
        let start = t * games_per_thread;
        let end = ((t + 1) * games_per_thread).min(games);
        if start >= end {
            continue;
        }

        thread::spawn(move || {
            let env = SnakeEnv::new(w, h);
            let mcts = Mcts::new(MctsConfig {
                num_simulations: sim,
                c_puct: 1.5,
            });
            let pv_uniform = UniformPolicyValue;
            let pv_onnx = if onnx_path.is_empty() {
                None
            } else {
                SnakeOnnxPolicyValue::load(&onnx_path, in_dim).ok()
            };

            let base_seed: u64 = 0x51A9_EED_u64 ^ (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed);

            for g in start..end {
                let mut s = env.reset(base_seed ^ (g as u64));
                let mut episode: Vec<Sample> = Vec::new();
                let mut ret = 0.0f32;
                let mut steps = 0usize;

                while s.status == SnakeStatus::Running {
                    let policy = if let Some(ref pv) = pv_onnx {
                        mcts.root_policy(&env, pv, &s)
                    } else {
                        mcts.root_policy(&env, &pv_uniform, &s)
                    };
                    let pi = apply_temperature(root_policy_to_pi(policy), temp);

                    let features = alpha_zero::snake::features::encode(&s);
                    episode.push(Sample {
                        features,
                        pi,
                        reward: 0.0,
                        z: 0.0,
                    });

                    let a = sample_action_from_pi(pi, &mut rng);
                    let (ns, r) = env.step(&s, a);
                    s = ns;
                    ret += r;
                    steps += 1;

                    if let Some(last) = episode.last_mut() {
                        last.reward = r;
                    }

                    if steps > 10_000 {
                        break;
                    }
                }

                // Backfill z as undiscounted return-to-go
                let mut gret = 0.0f32;
                for sample in episode.iter_mut().rev() {
                    gret += sample.reward;
                    sample.z = gret;
                }

                let win = s.status == SnakeStatus::Victory;

                let mut csv = String::new();
                for sample in episode {
                    for (i, v) in sample.features.iter().enumerate() {
                        csv.push_str(&format!("{:.3}", v));
                        csv.push(',');
                        let _ = i;
                    }
                    csv.push_str(&format!(
                        "{:.6},{:.6},{:.6},{:.6},{:.3}\n",
                        sample.pi[0], sample.pi[1], sample.pi[2], sample.pi[3], sample.z
                    ));
                }
                let rows = csv.lines().count();
                let _ = tx.send((csv, rows, ret, steps, win));
            }
        });
    }
    drop(tx);

    let mut total_rows = 0usize;
    let mut total_games = 0usize;
    let mut win_games = 0usize;
    let mut sum_return = 0.0f64;
    let mut sum_steps = 0usize;

    for (csv, rows, ret, steps, win) in rx {
        wtr.write_all(csv.as_bytes()).expect("write rows");
        total_rows += rows;
        total_games += 1;
        if win {
            win_games += 1;
        }
        sum_return += ret as f64;
        sum_steps += steps;

        if total_games % 10 == 0 {
            let avg_return = sum_return / (total_games as f64);
            let avg_steps = (sum_steps as f64) / (total_games as f64);
            let win_rate = (win_games as f64) / (total_games as f64);
            eprintln!(
                "games={} avg_return={:.3} avg_steps={:.2} win_rate={:.2}",
                total_games, avg_return, avg_steps, win_rate
            );
        }
    }

    wtr.flush().expect("flush");
    eprintln!(
        "wrote {} rows from {} games to {} (threads={})",
        total_rows,
        total_games,
        out_path.display(),
        threads
    );
}
