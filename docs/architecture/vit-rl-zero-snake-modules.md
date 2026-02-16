# ViT + AlphaZero Snake 模块架构

## 业务目标

- 在单机（RTX 5090 Laptop + 32 线程 CPU）上实现可持续训练吞吐的 Snake 强化学习系统。
- 支持并行环境、策略价值网络训练、实时监控、断点续训。

## 模块边界

1. `crates/snake-core`
- 纯 Rust 环境内核，提供批量并发 `reset/step/reset_done/get_obs_tensor`。
- 对 Python 暴露 PyO3 接口（feature: `python`）。

2. `python/snake_rl`
- 训练域：ViT、AlphaZero planner、self-play、replay、train loop。
- 通过 `runtime/rust_env.py` 优先连接 Rust 内核，失败时退回 Python fallback。

3. `crates/snake-ui`
- 监控域：并发网格、焦点智能体、曲线指标、控制条。
- 读取 `artifacts/metrics/latest.jsonl` 的实时指标。

4. `configs/`
- `runtime/hardware.auto.toml`: 机器探测和自动调参上限。
- `train/alpha_zero_vit.yaml`: 训练配置源。

## 数据流

1. `train_loop.py` 创建 `RustBatchEnv`。
2. `collect_self_play_batch` 采样观测，调用 planner 产生策略和动作。
3. 环境返回 `(obs, reward, done, score, legal_mask)`。
4. 转移样本写入 replay，训练器更新 ViT 策略价值网络。
5. 指标输出到 `artifacts/metrics/latest.jsonl`，UI 侧轮询显示。

## 风险控制

- MCTS CPU 压力：降低每轮 simulation，优先保证 steps/sec。
- UI 与训练争用：UI 刷新上限 20 FPS。
- 桥接延迟：优先 Rust backend，fallback 仅用于开发期可运行保障。
