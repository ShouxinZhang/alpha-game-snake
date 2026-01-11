# Alpha Zero (Module)

这个模块提供一个**轻量的 AlphaZero 风格决策框架骨架**（核心是 MCTS + Policy/Value 接口抽象），用于后续把“训练出来的策略/价值网络”接入到游戏或业务决策中。

## 业务上能干什么

- 给任意“可模拟”的环境接入 **MCTS 推演**，输出下一步动作建议（可用于 AI 对战、自动寻路、策略选择等）。
- 提供“自博弈数据采集 / PyTorch 训练 / ONNX 导出”的最小闭环（见 `train/`）。

## 优点

- 独立子模块：不侵入现有 `snake_gui`，方便实验与迭代。
- 低依赖：只依赖 `rand`。
- 通用接口：环境与策略模型通过 trait 连接，后续可接入真实网络推理。

## 缺点 / 当前边界

- 当前训练闭环是最小版：只覆盖 demo 环境（Nim），不包含分布式自博弈、复杂数据治理等重工程能力。
- 环境需要实现 trait（状态可 clone，动作可枚举）。

## Run demo

```bash
cargo run --bin demo_nim
```

## 主要 API

- `Environment`：定义状态、动作、玩家轮转、终局价值。
- `PolicyValueFn`：给定状态返回动作先验概率与价值估计。
- `Mcts`：基于 PUCT 的 MCTS 推演，输出 root policy 或直接采样动作。
