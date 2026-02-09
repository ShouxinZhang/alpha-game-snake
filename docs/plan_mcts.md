# MCTS Implementation Plan

## Goal
Implement a thread-safe, high-performance Monte Carlo Tree Search (MCTS) for Snake AlphaZero.

## Components

### 1. `Node` Struct
- **State**: `GameState` (from `snake_engine`) or hash/snapshot.
- **Stats**: `visit_count` (N), `total_value` (W), `mean_value` (Q), `prior_prob` (P).
- **Children**: `HashMap<Direction, NodeId>` or `Vec<(Direction, NodeId)>`.
- **Status**: `Unexpanded`, `Expanded`, `Terminal`.

### 2. `MCTS` Struct Data
- **Tree**: `Vec<Node>` (Arena allocation for performance) or `DashMap` (for parallel access, though simple mutex might suffice for read-heavy).
- **Root**: Index of current root.

### 3. Search Loop (Parallel)
Inside `search(game)`:
1.  **Selection**: Traverse from root to leaf using PUCT formula: $Q + U = Q + c_{puct} \cdot P \cdot \frac{\sqrt{\sum N}}{1 + N}$.
2.  **Expansion & Evaluation**:
    -   If leaf is not terminal:
        -   Call `inference_engine.predict(state)` to get Policy ($P$) and Value ($v$).
        -   Create child nodes for valid moves.
    -   If leaf is terminal:
        -   $v = \text{Game Result}$ (1.0 or -1.0).
3.  **Backup**: Propagate $v$ up the path. Update $N$, $W$, $Q$.

### 4. Virtual Loss (for Parallelism)
To prevent all threads from exploring the same path:
-   Add "virtual loss" to a node when traversing down.
-   Remove it after backup.

## Verification
-   **Unit Tests**:
    -   Test PUCT selection logic.
    -   Test Backup updates.
    -   Test "Mate in 1" scenario (should always find food).
