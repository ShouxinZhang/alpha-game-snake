use std::collections::HashMap;

use rand::distributions::WeightedIndex;
use rand::prelude::*;

use crate::env::{transform_value, Environment, Player};
use crate::policy::PolicyValueFn;

#[derive(Debug, Clone, Copy)]
pub struct MctsConfig {
    pub num_simulations: usize,
    pub c_puct: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 200,
            c_puct: 1.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RootPolicy<A> {
    pub visits: Vec<(A, u32)>,
}

impl<A: Copy + PartialEq> RootPolicy<A> {
    pub fn to_probabilities(&self, temperature: f32) -> Vec<(A, f32)> {
        let mut out = Vec::with_capacity(self.visits.len());
        if self.visits.is_empty() {
            return out;
        }

        if temperature <= 0.0 {
            let (best_action, _best_n) = self
                .visits
                .iter()
                .copied()
                .max_by_key(|(_, n)| *n)
                .unwrap();
            out.push((best_action, 1.0));
            for (a, _) in self.visits.iter().copied() {
                if a != best_action {
                    out.push((a, 0.0));
                }
            }
            return out;
        }

        let inv_t = 1.0 / temperature;
        let mut weights = Vec::with_capacity(self.visits.len());
        for &(_, n) in &self.visits {
            weights.push((n as f32).powf(inv_t));
        }
        let sum: f32 = weights.iter().sum();
        for ((a, _), w) in self.visits.iter().copied().zip(weights) {
            out.push((a, if sum > 0.0 { w / sum } else { 0.0 }));
        }
        out
    }
}

#[derive(Debug, Clone)]
struct Edge<A> {
    action: A,
    prior: f32,
    reward: f32,
    child: usize,
}

#[derive(Debug, Clone)]
struct Node<S, A> {
    state: S,
    player: Player,
    n_visits: u32,
    value_sum: f32,
    edges: Vec<Edge<A>>,
    // Cache for quick lookup (optional); kept small.
    edge_index: HashMap<A, usize>,
    expanded: bool,
}

impl<S, A: Copy + Eq + std::hash::Hash> Node<S, A> {
    fn new(state: S, player: Player) -> Self {
        Self {
            state,
            player,
            n_visits: 0,
            value_sum: 0.0,
            edges: Vec::new(),
            edge_index: HashMap::new(),
            expanded: false,
        }
    }

    fn mean_value(&self) -> f32 {
        if self.n_visits == 0 {
            0.0
        } else {
            self.value_sum / (self.n_visits as f32)
        }
    }
}

pub struct Mcts {
    cfg: MctsConfig,
}

impl Mcts {
    pub fn new(cfg: MctsConfig) -> Self {
        Self { cfg }
    }

    pub fn root_policy<E, PV>(&self, env: &E, policy_value: &PV, root_state: &E::State) -> RootPolicy<E::Action>
    where
        E: Environment,
        E::Action: std::fmt::Debug,
        PV: PolicyValueFn<E>,
    {
        let mut arena: Vec<Node<E::State, E::Action>> = Vec::new();
        let root_player = env.player_to_move(root_state);
        arena.push(Node::new(root_state.clone(), root_player));

        for _ in 0..self.cfg.num_simulations {
            self.simulate(env, policy_value, &mut arena, 0);
        }

        let root = &arena[0];
        let mut visits = Vec::with_capacity(root.edges.len());
        for edge in &root.edges {
            let child = &arena[edge.child];
            visits.push((edge.action, child.n_visits));
        }
        RootPolicy { visits }
    }

    pub fn select_action<E, PV, R>(
        &self,
        env: &E,
        policy_value: &PV,
        root_state: &E::State,
        temperature: f32,
        rng: &mut R,
    ) -> Option<E::Action>
    where
        E: Environment,
        E::Action: std::fmt::Debug,
        PV: PolicyValueFn<E>,
        R: Rng + ?Sized,
    {
        let policy = self.root_policy(env, policy_value, root_state);
        let probs = policy.to_probabilities(temperature);
        if probs.is_empty() {
            return None;
        }

        // Sample
        let actions: Vec<E::Action> = probs.iter().map(|(a, _)| *a).collect();
        let weights: Vec<f32> = probs.iter().map(|(_, p)| *p).collect();
        let dist = WeightedIndex::new(&weights).ok()?;
        Some(actions[dist.sample(rng)])
    }

    fn simulate<E, PV>(&self, env: &E, policy_value: &PV, arena: &mut Vec<Node<E::State, E::Action>>, root_id: usize)
    where
        E: Environment,
        E::Action: Copy + Eq + std::hash::Hash + std::fmt::Debug,
        PV: PolicyValueFn<E>,
    {
        let mut path_nodes: Vec<usize> = Vec::new();
        let mut path_rewards: Vec<f32> = Vec::new();
        let mut node_id = root_id;

        loop {
            path_nodes.push(node_id);

            // Terminal?
            let terminal = {
                let node = &arena[node_id];
                env.terminal_value(&node.state)
            };
            if let Some(v_terminal) = terminal {
                self.backprop(arena, &path_nodes, &path_rewards, v_terminal, arena[node_id].player);
                return;
            }

            // Expand?
            if !arena[node_id].expanded {
                let (v_leaf, leaf_player) = self.expand(env, policy_value, arena, node_id);
                self.backprop(arena, &path_nodes, &path_rewards, v_leaf, leaf_player);
                return;
            }

            // Select next child
            let next = self.select_child(arena, node_id);
            match next {
                Some((child_id, reward)) => {
                    path_rewards.push(reward);
                    node_id = child_id;
                }
                None => {
                    // No legal moves; treat as draw/zero.
                    self.backprop(arena, &path_nodes, &path_rewards, 0.0, arena[node_id].player);
                    return;
                }
            }
        }
    }

    fn expand<E, PV>(
        &self,
        env: &E,
        policy_value: &PV,
        arena: &mut Vec<Node<E::State, E::Action>>,
        node_id: usize,
    ) -> (f32, Player)
    where
        E: Environment,
        E::Action: Copy + Eq + std::hash::Hash,
        PV: PolicyValueFn<E>,
    {
        let (actions, player, state_clone) = {
            let node = &arena[node_id];
            let actions = env.legal_actions(&node.state);
            (actions, node.player, node.state.clone())
        };

        if actions.is_empty() {
            arena[node_id].expanded = true;
            return (0.0, player);
        }

        let (mut priors, v) = policy_value.evaluate(env, &state_clone, &actions);
        if priors.len() != actions.len() {
            priors = vec![1.0; actions.len()];
        }

        // Normalize priors (avoid NaN)
        let sum: f32 = priors.iter().map(|x| x.max(0.0)).sum();
        let mut norm_priors = Vec::with_capacity(priors.len());
        if sum > 0.0 {
            for p in priors {
                norm_priors.push(p.max(0.0) / sum);
            }
        } else {
            let p = 1.0 / (actions.len() as f32);
            norm_priors.resize(actions.len(), p);
        }

        // Create children
        let mut edges = Vec::with_capacity(actions.len());
        let mut edge_index = HashMap::with_capacity(actions.len());
        for (i, (&a, &p)) in actions.iter().zip(norm_priors.iter()).enumerate() {
            let (next_state, reward) = env.step(&state_clone, a);
            let next_player = env.player_to_move(&next_state);
            let child_id = arena.len();
            arena.push(Node::new(next_state, next_player));
            edges.push(Edge {
                action: a,
                prior: p,
                reward,
                child: child_id,
            });
            edge_index.insert(a, i);
        }

        let node = &mut arena[node_id];
        node.edges = edges;
        node.edge_index = edge_index;
        node.expanded = true;

        (v, player)
    }

    fn select_child<S, A>(&self, arena: &Vec<Node<S, A>>, node_id: usize) -> Option<(usize, f32)>
    where
        A: Copy + Eq + std::hash::Hash,
    {
        let node = &arena[node_id];
        if node.edges.is_empty() {
            return None;
        }

        let parent_n = node.n_visits.max(1) as f32;
        let parent_player = node.player;

        let mut best_score = f32::NEG_INFINITY;
        let mut best_child = None;
        let mut best_reward = 0.0;

        for edge in &node.edges {
            let child = &arena[edge.child];

            let q = if child.n_visits == 0 {
                // If we haven't visited this child, assume value is 0 (neutral).
                // Or use edge.reward + leaf_value (from prior network if we stored it?)
                // Here we just use the immediate reward.
                edge.reward
            } else {
                let child_q = child.mean_value();
                edge.reward + 0.99f32 * transform_value(child_q, child.player, parent_player)
            };

            let u = self.cfg.c_puct
                * edge.prior
                * (parent_n.sqrt() / (1.0 + (child.n_visits as f32)));

            let score = q + u;
            if score > best_score {
                best_score = score;
                best_child = Some(edge.child);
                best_reward = edge.reward;
            }
        }

        best_child.map(|c| (c, best_reward))
    }

    fn backprop<S, A>(
        &self,
        arena: &mut Vec<Node<S, A>>,
        path_nodes: &[usize],
        path_rewards: &[f32],
        leaf_value: f32,
        leaf_player: Player,
    )
    where
        A: Copy + Eq + std::hash::Hash,
    {
        // Walk from leaf to root
        let mut v = leaf_value;
        let mut v_player = leaf_player;

        // path_rewards is aligned with parent indices: reward[i] is from path_nodes[i] -> path_nodes[i+1]
        for (i, &node_id) in path_nodes.iter().enumerate().rev() {
            let reward = if i < path_rewards.len() { path_rewards[i] } else { 0.0 };
            let node_player = arena[node_id].player;
            let v_node = reward + 0.99f32 * transform_value(v, v_player, node_player);

            let node = &mut arena[node_id];
            node.n_visits += 1;
            node.value_sum += v_node;

            v = v_node;
            v_player = node_player;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nim::{NimEnv, NimState};
    use crate::policy::UniformPolicyValue;

    #[test]
    fn mcts_runs_and_returns_policy() {
        let env = NimEnv;
        let pv = UniformPolicyValue;
        let mcts = Mcts::new(MctsConfig {
            num_simulations: 50,
            c_puct: 1.5,
        });

        let root = NimState { pile: 7, player: 1 };
        let pol = mcts.root_policy(&env, &pv, &root);
        assert!(!pol.visits.is_empty());
        let sum: u32 = pol.visits.iter().map(|(_, n)| *n).sum();
        assert!(sum > 0);
    }
}
