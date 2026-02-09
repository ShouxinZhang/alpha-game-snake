# 贪吃蛇 AlphaZero 实现设计文档

## 1. 概述
本文档概述了在 `alpha-game-snake` 仓库中为贪吃蛇游戏实现基于 AlphaZero 的 AI 架构。
为实现高性能和模块化，我们将采用**混合架构**：
- **Rust**: 高性能游戏环境、MCTS（蒙特卡洛树搜索）和自我对弈数据生成。
- **Python**: 使用 PyTorch 进行深度神经网络 (CNN) 训练。

## 2. 目录结构与模块
遵循 `AGENTS.md` 中的原则，所有新的实验性代码将包含在其自己的模块中，以避免污染根目录。

```
alpha-game-snake/
├── modules/
│   ├── snake_gui/           # 现有的 GUI（将依赖 snake_engine）
│   ├── snake_engine/        # [新增] 纯逻辑（无头模式），从 snake_gui 重构而来
│   └── snake_alphazero/     # [新增] AlphaZero 实现
│       ├── src/             # Rust: MCTS、自我对弈 Worker、ONNX 推理
│       ├── trainer/         # Python: PyTorch 训练循环、数据集加载器
│       └── models/          # .onnx 和 .pt 模型文件存储
└── docs/
    └── design_snake_alphazero.md  # 本文档
```

## 3. 架构组件

### 3.1. 游戏引擎 (`snake_engine`)
*从 `snake_gui` 重构现有逻辑为独立库。*
- **职责**:
    - 状态管理（Grid 网格、Snake 蛇身、Food 食物）。
    - 规则执行（碰撞、生长）。
    - 支持无头执行（无 Macroquad 依赖）。

### 3.2. 推理与 MCTS (`snake_alphazero/src`)
- **MCTS**: 在 Rust 中自定义实现。
- **推理**: 使用 `ort` (ONNX Runtime Rust 绑定) 直接从 Rust 查询神经网络，无 Python 开销。
- **输出**: 生成回放缓冲区 (State, Policy, Value) 并保存到磁盘（如 Protobuf 或二进制 JSON）。

### 3.3. 训练循环 (`snake_alphazero/trainer`)
- **框架**: PyTorch。
- **输入**: 加载由 Rust 生成的回放缓冲区数据。
- **输出**: 将训练好的模型导出为 ONNX 格式，供 Rust Worker 重新加载。
- **网络架构 (CNN)**:
    - **输入**: `(C, H, W)` 张量。
        - 通道 0: 蛇身 (1=身体, 0=空）。
        - 通道 1: 蛇头位置 (1=蛇头, 0=空）。
        - 通道 2: 食物位置 (1=食物, 0=空）。
        - 通道 3: **饥饿度 (Hunger)** (全图填充相同的值 $t_{since\_eat} / T_{max}$，表示归一化的饥饿程度)。
    - **主干网络 (Backbone)**: 残差塔 (3-5 个 ResNet Block)。
    - **输出头 (Heads)**:
        - 策略头 (Policy Head): Conv1x1 -> FC -> Softmax (4 个输出: 上、下、左、右)。
        - 价值头 (Value Head): Conv1x1 -> FC -> Tanh (1 个输出: -1 到 1)。

### 3.4. 奖励设计与反绕圈机制 (Reward & Anti-Looping)
由于贪吃蛇是单人游戏，我们将调整标准的 AlphaZero 奖励机制，并引入**饥饿**概念以防止消极绕圈：
"饿死"（Starvation）是指：如果蛇在连续 $T_{max}$（最大步数）内没有吃到任何食物，系统将强制判定游戏结束，并视为失败（给予负奖励）。
- **目标**: 最大化每局游戏的得分，同时兼顾效率。
- **状态定义**:
    - $T_{max}$: 最大饥饿步数 (依赖于网格大小，例如: $2 \times W \times H$)。
    - $t$: 当前未进食步数。
- **即时奖励 (Immediate Reward)** $r_t$:
    - **吃到食物**: +1.0 (重置 $t=0$)
    - **游戏结束 (死亡/撞墙/撞身)**: -1.0
    - **饿死 (Starvation)** ($t > T_{max}$): -1.0 (强制结束游戏，杜绝无限绕圈)
    - **时间惩罚 (Time Penalty)**: -0.01 (每走一步扣除微小分数，迫使蛇寻找最短路径，而不是在安全区绕圈)
    - **胜利 (填满全图)**: +1.0
- **价值目标 (Value Target)** $z$:
    - 价值网络 $v(s)$ 将预测从当前状态 $s$ 开始的**预期最终得分**或**归一化胜率**。
    - 引入时间惩罚后，AlphaZero 会倾向于更早吃到食物以减少惩罚累积。

## 4. 工作流程（"循环"）

1.  **数据生成阶段 (Rust)**:
    -   Worker 加载 `latest_model.onnx`。
    -   使用 MCTS 并行运行 N 局游戏。
    -   将游戏步骤 (state, prob, outcome) 保存到 `data/gen_X.json`。
2.  **训练阶段 (Python)**:
    -   监视器 (Watcher) 检测新数据文件。
    -   使用经验回放 (Experience Replay) 更新权重。
    -   每 K 步导出新的 `latest_model.onnx`。
3.  **评估**:
    -   单独的 Rust 进程运行新模型 vs 旧模型对弈。
    -   如果胜率 > 55%，则接受新模型。

## 5. 开发路线图
1.  **重构**: 从 `snake_gui` 中提取 `snake_engine`。
2.  **Python 基线**: 创建简单的 Python CNN 和训练脚本。
3.  **Rust MCTS**: 在 Rust 中实现 MCTS。
4.  **集成**: 通过文件系统连接 Rust 数据生成与 Python 训练。
