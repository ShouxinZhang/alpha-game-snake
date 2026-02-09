use snake_engine::{Game, GameStatus, Direction};
use parking_lot::Mutex;
use std::sync::Arc;
use rand::Rng;

pub const DEFAULT_NUM_THREADS: usize = 32;

#[derive(Clone, Copy)]
pub struct MCTSConfig {
    pub num_threads: usize,
    pub num_simulations: usize,
    pub cpuct: f32,
    pub virtual_loss: f32,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            num_threads: DEFAULT_NUM_THREADS,
            num_simulations: 200,
            cpuct: 1.0,
            virtual_loss: 3.0,
        }
    }
}

struct Node {
    state: Game,
    visit_count: f32,
    value_sum: f32,
    prior_prob: f32,
    children: Vec<(Direction, usize)>, // Action -> Index in Tree
    is_expanded: bool,
}

impl Node {
    fn new(state: Game, prior: f32) -> Self {
        Self {
            state,
            visit_count: 0.0,
            value_sum: 0.0,
            prior_prob: prior,
            children: Vec::new(),
            is_expanded: false,
        }
    }

    fn ucb_score(&self, parent_visit: f32, cpuct: f32) -> f32 {
        let q_value = if self.visit_count > 0.0 {
            self.value_sum / self.visit_count
        } else {
            0.0
        };
        let u_value = cpuct * self.prior_prob * (parent_visit.sqrt() / (1.0 + self.visit_count));
        q_value + u_value
    }
}

use crate::inference::InferenceEngine;

pub struct MCTS {
    config: MCTSConfig,
    tree: Arc<Mutex<Vec<Node>>>,
    inference_engine: Arc<InferenceEngine>,
}

impl MCTS {
    pub fn new(config: MCTSConfig, inference_engine: Arc<InferenceEngine>) -> Self {
        Self {
            config,
            tree: Arc::new(Mutex::new(Vec::new())),
            inference_engine,
        }
    }

    pub fn search(&mut self, game: &Game) -> Vec<f32> {
        // Reset tree for new search (reusing capacity would be better optimization)
        {
            let mut tree = self.tree.lock();
            tree.clear();
            tree.push(Node::new(game.clone(), 1.0)); // Root node
        }

        let num_threads = self.config.num_threads;
        let simulations_per_thread = self.config.num_simulations / num_threads;

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let tree = self.tree.clone();
                let config = self.config;
                let engine = self.inference_engine.clone();
                s.spawn(move || {
                    let mut rng = rand::thread_rng();
                    for _ in 0..simulations_per_thread {
                        Self::simulate(&tree, &engine, config, &mut rng);
                    }
                });
            }
        });

        // Compute policy from root visit counts
        let tree = self.tree.lock();
        let root = &tree[0];
        
        let mut policy = vec![0.0; 4]; // Up, Down, Left, Right
        let sum_visits: f32 = root.children.iter().map(|&(_, idx)| tree[idx].visit_count).sum();

        if sum_visits > 0.0 {
            for &(dir, idx) in &root.children {
                let dir_idx = match dir {
                    Direction::Up => 0,
                    Direction::Down => 1,
                    Direction::Left => 2,
                    Direction::Right => 3,
                };
                policy[dir_idx] = tree[idx].visit_count / sum_visits;
            }
        } else {
            // Fallback: uniform
             for x in policy.iter_mut() { *x = 0.25; }
        }
        
        policy
    }

    fn simulate(tree: &Arc<Mutex<Vec<Node>>>, engine: &InferenceEngine, config: MCTSConfig, rng: &mut impl Rng) {
        let mut node_idx = 0;
        let mut path = Vec::new();
        
        // 1. Selection
        loop {
            path.push(node_idx);
            
            let mut data = tree.lock();
            
            // Apply Virtual Loss and check conditions
            let (is_expanded, is_terminal, parent_visit, children_indices) = {
                let node = &mut data[node_idx];
                
                // Virtual Loss
                node.visit_count += config.virtual_loss;
                node.value_sum -= config.virtual_loss;
                
                let is_terminal = node.state.status() != GameStatus::Running;
                (node.is_expanded, is_terminal, node.visit_count, node.children.clone())
            };

            if is_terminal || !is_expanded {
                break;
            }

            // Find best child (read-only access to data)
            let mut best_score = -f32::INFINITY;
            let mut best_idx = None;

            for &(_, child_idx) in &children_indices {
                // Ensure we don't access out of bounds (though indices should be valid)
                if let Some(child) = data.get(child_idx) {
                    let score = child.ucb_score(parent_visit, config.cpuct);
                    if score > best_score {
                        best_score = score;
                        best_idx = Some(child_idx);
                    }
                }
            }

            if let Some(idx) = best_idx {
                node_idx = idx;
            } else {
                break;
            }
        }

        // 2. Expansion & Evaluation
        let last_idx = *path.last().unwrap();
        let value = {
             let mut data = tree.lock();
             let status = data[last_idx].state.status();
             
             if status != GameStatus::Running {
                  match status {
                      GameStatus::Victory => 1.0,
                      _ => -1.0,
                  }
             } else {
                  // Check if already expanded (by another thread)
                  if !data[last_idx].is_expanded {
                      let state = data[last_idx].state.clone();
                      
                      // ** Network Inference **
                      // Run inference OUTSIDE the lock? 
                      // Ideally yes, but here we need to hold lock to ensure we are the expander?
                      // Actually, if we release lock, another thread might expand. 
                      // Double-check locking pattern?
                      // Simple implementation: Hold lock, run inference (serialized).
                      // Better optimization: Drop lock, run inference, Re-lock, check if expanded.
                      // For now: Serialize inside lock (easier logic, less race bugs).
                      // Note: InferenceEngine uses Mutex too, so it's serialized anyway.
                      
                      drop(data); // Drop lock to run inference? Engine has its own lock.
                      // But if we drop `data`, other thread might expand `last_idx`.
                      // Then when we re-acquire, we see `is_expanded=true`.
                      
                      let (policy, value) = engine.predict(&state);
                      
                      let mut data = tree.lock(); // Re-acquire
                      
                      if !data[last_idx].is_expanded {
                           let mut new_children_indices = Vec::new();

                           for (i, &dir) in [Direction::Up, Direction::Down, Direction::Left, Direction::Right].iter().enumerate() {
                                let mut next_state = state.clone();
                                next_state.queue_direction(dir);
                                let seed = rng.gen();
                                next_state.step(seed);
                                
                                let prior = policy[match dir {
                                    Direction::Up => 0,
                                    Direction::Down => 1,
                                    Direction::Left => 2,
                                    Direction::Right => 3,
                                }];
                                
                                let child_node = Node::new(next_state, prior);
                                let child_idx = data.len();
                                data.push(child_node);
                                new_children_indices.push((dir, child_idx));
                           }
                           
                           // Update parent
                           let parent = &mut data[last_idx];
                           parent.children = new_children_indices;
                           parent.is_expanded = true;
                           
                           value
                      } else {
                           // Already expanded by someone else. 
                           // Treat as leaf value? Or just skip?
                           // Actually we should just read the value from the updated node?
                           // Or standard AlphaZero doesn't update value if already expanded?
                           // Just return the computed value for backup.
                           value
                      }
                  } else {
                      // Already expanded
                      0.0 
                  }
             }
        };

        // 3. Backup
        {
            let mut data = tree.lock();
            for &idx in &path {
                let node = &mut data[idx];
                node.visit_count -= config.virtual_loss; // Restore virtual loss
                node.value_sum += config.virtual_loss;
                
                node.visit_count += 1.0;
                node.value_sum += value;
            }
        }
    }
}
