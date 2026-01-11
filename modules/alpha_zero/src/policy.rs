use crate::env::Environment;

/// Policy + value evaluator.
///
/// Returns:
/// - a prior probability for each provided action (same order, each >= 0, sum doesn't have to be 1)
/// - a value estimate in [-1,1] (recommended) from the perspective of `env.player_to_move(state)`.
pub trait PolicyValueFn<E: Environment> {
    fn evaluate(&self, env: &E, state: &E::State, actions: &[E::Action]) -> (Vec<f32>, f32);
}

/// A minimal baseline: uniform priors and zero value.
#[derive(Default, Clone, Copy)]
pub struct UniformPolicyValue;

impl<E: Environment> PolicyValueFn<E> for UniformPolicyValue {
    fn evaluate(&self, _env: &E, _state: &E::State, actions: &[E::Action]) -> (Vec<f32>, f32) {
        if actions.is_empty() {
            return (Vec::new(), 0.0);
        }
        let p = 1.0 / (actions.len() as f32);
        (vec![p; actions.len()], 0.0)
    }
}
