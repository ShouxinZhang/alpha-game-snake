use std::hash::Hash;

pub type Player = i8;

/// Environment for AlphaZero-style MCTS.
///
/// - `terminal_value` returns value from the perspective of `player_to_move(state)`.
///   For zero-sum games, typical range is [-1, 1].
pub trait Environment {
    type State: Clone;
    type Action: Copy + Eq + Hash;

    fn player_to_move(&self, state: &Self::State) -> Player;

    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action>;

    fn next_state(&self, state: &Self::State, action: Self::Action) -> Self::State;

    /// Transition with per-step reward.
    ///
    /// Reward is interpreted from the perspective of `player_to_move(state)`.
    /// Default implementation returns zero reward.
    fn step(&self, state: &Self::State, action: Self::Action) -> (Self::State, f32) {
        (self.next_state(state, action), 0.0)
    }

    fn terminal_value(&self, state: &Self::State) -> Option<f32>;
}

pub(crate) fn transform_value(value: f32, from_player: Player, to_player: Player) -> f32 {
    // For single-player (1 vs 1), this is identity.
    // For 2-player zero-sum (alternating 1 and -1), this is also identity mathematically
    // if we treat player as the sign flip factor? No.
    // If player 1 has value V, player -1 has value -V.
    // To convert V from P1 to P-1: -V.
    // Logic: v_to = v_from * (from * to)?
    // If from=1, to=-1 => v * -1. Correct.
    // If from=-1, to=1 => v * -1. Correct.
    // If from=1, to=1 => v * 1. Correct.
    // This logic works for standard +1/-1 zero sum.
    // For single player snake, player is always 1. So it returns value.
    if from_player == to_player {
        value
    } else {
        -value 
    }
}
