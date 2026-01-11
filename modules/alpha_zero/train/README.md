# Alpha Zero Train (PyTorch)

这个子模块提供 **AlphaZero 训练闭环** 的最小可用版本：

1. 用 Rust 的 `alpha_zero`（MCTS）生成自博弈数据（CSV）
2. 用 PyTorch 训练一个 `policy + value` 网络
3. 导出 ONNX（可选），方便后续在 Rust 端推理接入

> 当前 demo 环境为 Nim（取 1~3 个石子）。后续要接入 Snake，只需要替换“状态特征编码”和“动作空间定义”。

## 业务上能干什么

- 产出一个可持续迭代的 AI 能力：**数据 → 训练 → 模型**
- 让 MCTS 不再靠均匀先验，而是逐步学习更强的策略（更像 AlphaZero）

## Quickstart

### 1) 生成数据

```bash
cd ..
cargo run --bin selfplay_nim -- --out train/data/nim.csv --games 200 --sim 200 --temp 1.0
```

说明：数据生成默认使用 `--threads 32`（CPU 32 线程）。你也可以手动覆盖：

```bash
cargo run --bin selfplay_nim -- --out train/data/nim.csv --games 200 --sim 200 --temp 1.0 --threads 16
```

### 2) 安装依赖（在你已有 venv 下）

```bash
pip install -r requirements.txt
```

### 3) 训练

```bash
python train_nim.py --data data/nim.csv --epochs 30 --batch 256 --lr 1e-3 --out_dir models
```

训练日志会额外输出评估指标（默认每个 epoch 评估 10 局）：

- `avg_return`：平均回报（Nim: 赢 +1 / 输 -1）
- `avg_len`：平均对局步数
- `win_rate`：先手胜率

### 4) 导出 ONNX（可选）

```bash
python export_onnx.py --ckpt models/nim_policy_value.pt --out models/nim_policy_value.onnx
```

## 数据格式（CSV）

每行一个训练样本：

- `pile`: 当前剩余石子数
- `player`: 当前轮到的玩家（+1/-1）
- `pi_take1, pi_take2, pi_take3`: MCTS 得到的动作分布（目标策略）
- `z`: 终局回传价值（从“当前 player_to_move”视角）

