# snake_multi_viewer

用 Macroquad 做一个“32 个小窗口同时观察”的 Snake 调试器。

## 业务用途

- 一屏同时观察 32 个 agent 的状态演化（5x5 默认）
- 便于调 reward/防苟活/动作合法性/MCTS 行为（后续可接入）

## Run

```bash
cd modules/snake_multi_viewer
./start.sh
```

## Controls

- `Space`: pause/resume
- `S`: single step (when paused)
- `R`: reset all agents
- `+` / `-`: speed up / slow down
- Mouse click: select agent (highlight)

## Train

窗口右下角有 `Start Train` / `Stop Train` 按钮：

- Start：后台启动 `modules/alpha_zero/train/loop_snake_5x5.sh`（大轮次，直到你点 Stop）
- Stop：kill 后台训练进程

训练 stdout/stderr 会逐行回显到窗口底部控制台（滚动显示最近几行）。
