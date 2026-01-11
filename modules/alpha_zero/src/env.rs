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
    // Convert `value` measured from `from_player` perspective into `to_player` perspective.
    // If players alternate (+1/-1), this becomes negation when switching sides.
    if from_player == to_player {
        value
    } else {
        value * (from_player as f32) / (to_player as f32)
    }
}
