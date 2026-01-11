use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use alpha_zero::{Environment, Mcts, MctsConfig, Player, RootPolicy, UniformPolicyValue};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

#[derive(Clone, Debug)]
struct NimState {
    pile: u8,
    player: Player,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum NimAction {
    Take1,
    Take2,
    Take3,
}

impl NimAction {
    fn to_index(self) -> usize {
        match self {
            Self::Take1 => 0,
            Self::Take2 => 1,
            Self::Take3 => 2,
        }
    }

    fn from_index(i: usize) -> Self {
        match i {
            0 => Self::Take1,
            1 => Self::Take2,
            _ => Self::Take3,
        }
    }
}

struct Nim;

impl Environment for Nim {
    type State = NimState;
    type Action = NimAction;

    fn player_to_move(&self, state: &Self::State) -> Player {
        state.player
    }

    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action> {
        let mut a = Vec::new();
        if state.pile >= 1 {
            a.push(NimAction::Take1);
        }
        if state.pile >= 2 {
            a.push(NimAction::Take2);
        }
        if state.pile >= 3 {
            a.push(NimAction::Take3);
        }
        a
    }

    fn next_state(&self, state: &Self::State, action: Self::Action) -> Self::State {
        let take = match action {
            NimAction::Take1 => 1,
            NimAction::Take2 => 2,
            NimAction::Take3 => 3,
        };
        NimState {
            pile: state.pile.saturating_sub(take),
            player: -state.player,
        }
    }

    fn terminal_value(&self, state: &Self::State) -> Option<f32> {
        if state.pile == 0 {
            // Player to move has no move; they lost.
            Some(-1.0)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct Sample {
    pile: u8,
    player: Player,
    pi: [f32; 3],
    z: f32,
}

fn root_policy_to_fixed_pi(policy: RootPolicy<NimAction>) -> [f32; 3] {
    let mut pi = [0.0f32; 3];
    for (a, n) in policy.visits {
        pi[a.to_index()] = n as f32;
    }
    let s: f32 = pi.iter().sum();
    if s > 0.0 {
        for p in &mut pi {
            *p /= s;
        }
    } else {
        pi = [1.0 / 3.0; 3];
    }
    pi
}

fn sample_action_from_pi<R: Rng + ?Sized>(pi: [f32; 3], rng: &mut R) -> NimAction {
    let weights = pi.map(|p| p.max(0.0));
    if let Ok(dist) = WeightedIndex::new(weights) {
        NimAction::from_index(dist.sample(rng))
    } else {
        NimAction::Take1
    }
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

fn parse_arg_string(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    // Example:
    // cargo run --bin selfplay_nim -- --out train/data/nim.csv --games 200 --sim 200 --temp 1.0
    let args: Vec<String> = std::env::args().collect();
    let out = parse_arg_string(&args, "--out", "nim.csv");
    let games = parse_arg_u64(&args, "--games", 200) as usize;
    let sim = parse_arg_u64(&args, "--sim", 200) as usize;
    let temp = parse_arg_f32(&args, "--temp", 1.0);
    let max_pile = parse_arg_u64(&args, "--max_pile", 12) as u8;
    let threads = parse_arg_u64(&args, "--threads", 32) as usize;

    let out_path = PathBuf::from(out);
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).expect("create output dir");
        }
    }
    let f = fs::File::create(&out_path).expect("create output file");
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "pile,player,pi_take1,pi_take2,pi_take3,z"
    )
    .expect("write header");

    let threads = threads.max(1).min(games.max(1));
    let (tx, rx) = mpsc::channel::<(String, usize, i32, usize)>();
    // payload: (csv_lines, rows, start_player_result(+1 win / -1 lose), episode_len)

    let base_seed: u64 = 0xA1F4_5EED_u64;
    let games_per_thread = (games + threads - 1) / threads;

    for t in 0..threads {
        let tx = tx.clone();
        let start = t * games_per_thread;
        let end = ((t + 1) * games_per_thread).min(games);
        if start >= end {
            continue;
        }

        thread::spawn(move || {
            let env = Nim;
            let pv = UniformPolicyValue;
            let mcts = Mcts::new(MctsConfig {
                num_simulations: sim as usize,
                c_puct: 1.5,
            });
            let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed ^ (t as u64));

            for g in start..end {
                // Make each game deterministic per index
                let mut grng = rand::rngs::StdRng::seed_from_u64(base_seed ^ (g as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let start_pile = grng.gen_range(1..=max_pile);
                let mut state = NimState {
                    pile: start_pile,
                    player: 1,
                };

                let mut episode: Vec<Sample> = Vec::new();
                let mut steps = 0usize;

                while env.terminal_value(&state).is_none() {
                    let policy = mcts.root_policy(&env, &pv, &state);
                    let mut pi = root_policy_to_fixed_pi(policy);

                    // Temperature adjustment on pi (simple power transform)
                    if temp > 0.0 && (temp - 1.0).abs() > f32::EPSILON {
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
                    }

                    episode.push(Sample {
                        pile: state.pile,
                        player: state.player,
                        pi,
                        z: 0.0, // fill after terminal
                    });

                    let action = sample_action_from_pi(pi, &mut rng);
                    state = env.next_state(&state, action);
                    steps += 1;
                }

                // Winner is the previous player (player_to_move at terminal has no move and loses).
                let terminal_player = env.player_to_move(&state);
                let winner: Player = -terminal_player;
                let result = if winner == 1 { 1 } else { -1 };

                // Terminal value is from perspective of player_to_move at terminal.
                let mut z = env.terminal_value(&state).unwrap_or(0.0);
                let mut z_player = terminal_player;

                // Backfill z for each sample from its own player perspective.
                for s in episode.iter_mut().rev() {
                    if s.player != z_player {
                        z = -z;
                        z_player = s.player;
                    }
                    s.z = z;
                }

                let mut csv = String::new();
                for s in episode {
                    csv.push_str(&format!(
                        "{},{},{:.6},{:.6},{:.6},{:.3}\n",
                        s.pile, s.player, s.pi[0], s.pi[1], s.pi[2], s.z
                    ));
                }
                let rows = csv.lines().count();
                let _ = tx.send((csv, rows, result, steps));
            }
        });
    }
    drop(tx);

    let mut total_rows = 0usize;
    let mut total_games = 0usize;
    let mut win_games = 0usize;
    let mut sum_return = 0i64;
    let mut sum_len = 0usize;

    for (csv, rows, result, steps) in rx {
        w.write_all(csv.as_bytes()).expect("write rows");
        total_rows += rows;
        total_games += 1;
        if result > 0 {
            win_games += 1;
        }
        sum_return += result as i64;
        sum_len += steps;

        if total_games % 10 == 0 {
            let avg_return = (sum_return as f64) / (total_games as f64);
            let avg_len = (sum_len as f64) / (total_games as f64);
            let win_rate = (win_games as f64) / (total_games as f64);
            eprintln!(
                "games={} avg_return={:.3} avg_len={:.2} win_rate={:.2}",
                total_games, avg_return, avg_len, win_rate
            );
        }
    }

    w.flush().expect("flush");
    eprintln!("wrote {} rows from {} games to {} (threads={})", total_rows, total_games, out_path.display(), threads);
}
