use alpha_zero::{Environment, Mcts, MctsConfig, Player, UniformPolicyValue};
use rand::thread_rng;

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
            Some(-1.0)
        } else {
            None
        }
    }
}

fn main() {
    let env = Nim;
    let pv = UniformPolicyValue;
    let mcts = Mcts::new(MctsConfig {
        num_simulations: 200,
        c_puct: 1.5,
    });

    let mut rng = thread_rng();

    let mut s = NimState { pile: 12, player: 1 };
    println!("Start Nim with pile={} (player={})", s.pile, s.player);

    while env.terminal_value(&s).is_none() {
        let a = mcts
            .select_action(&env, &pv, &s, 0.0, &mut rng)
            .expect("no action");
        println!("player {} takes {:?} (pile={})", s.player, a, s.pile);
        s = env.next_state(&s, a);
    }

    let v = env.terminal_value(&s).unwrap();
    println!("Terminal reached. player_to_move={} value={}", s.player, v);
}
