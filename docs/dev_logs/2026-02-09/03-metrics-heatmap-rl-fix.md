# 开发日志：指标/热力图/RL无效问题修复

## 1. 用户原始请求（引用）
> 分析一下问题，我觉得需要有avg reward, score, 生存step, loss，然后热力图有问题，此外RL没有效果。

## 2. 轮次记录（背景/意图/思考摘要）
- 背景：UI 指标曲线接近平线，热力图展示为固定 2x2 色块，训练日志 reward/score 长期低位抖动。
- 意图：定位并修复“指标表达失真 + 热力图失真 + RL 信号弱”的核心原因。
- 摘要：
  - 指标侧：`avg_steps` 被错误写成常数 `steps_per_iter`，`rolling_avg_reward` 实际是瞬时值。
  - 训练侧：self-play 未正确传递 legal mask；value target 仅用即时奖励，缺少 bootstrap TD 目标。
  - 展示侧：热力图使用合法动作掩码归一化，导致长期 `33/33/0/33` 的伪策略图。
  - 可运维侧：`torch.compile` 下 checkpoint 存 `_orig_mod` 前缀，影响后续加载与评估。

## 3. 修改时间（精确到秒）
- 开始时间：2026-02-09 15:58:xx CST
- 结束时间：2026-02-09 16:08:15 CST

## 4. 文件清单（路径/操作/时间/说明）
- `python/snake_rl/trainers/self_play.py` / 修改 / 16:0x / 修复 legal mask 传递；改为 TD target；统计完成回合 reward/score/survival steps；输出策略均值与完成回合数。
- `python/snake_rl/trainers/train_loop.py` / 修改 / 16:0x / 引入 metrics rolling window；记录 instant+rolling 指标；写出 policy_up/down/left/right；checkpoint 导出 state_dict 兼容 `torch.compile`。
- `configs/train/alpha_zero_vit.yaml` / 修改 / 16:0x / 增加 `optim.gamma` 与 `runtime.metrics_window`。
- `crates/snake-ui/src/bridge/metrics_client.rs` / 修改 / 16:0x / 扩展指标结构，支持策略概率与完成回合读取。
- `crates/snake-ui/src/panels/metrics.rs` / 修改 / 16:0x / 图表改为动态尺寸，确保 4 项（avg reward/score/survival steps/loss）可见。
- `crates/snake-ui/src/panels/focus_agent.rs` / 修改 / 16:0x / 热力图改为基于状态的网格 saliency（head/body/food + 距离衰减），替换原 2x2 假热力图。
- `crates/snake-ui/src/main.rs` / 修改 / 16:0x / 使用 trainer 输出策略驱动焦点策略显示，并作为 UI 侧 action hint；状态栏新增 episodes finished。

## 5. 变更说明（方案/影响范围/风险控制）
- 方案：
  - 训练信号增强：从“一步即时奖励监督”升级为“TD(1) bootstrap 目标”。
  - 指标语义纠正：分离瞬时与滚动统计，避免伪平稳曲线误导。
  - 热力图语义纠正：从 action-mask 伪热力图改为状态热区图 + trainer policy 概率。
- 影响范围：Python 训练链路与 Rust UI 展示链路。
- 风险控制：
  - 先做本地冒烟训练验证输出字段；
  - 保持配置向后兼容（新增字段有默认值）；
  - UI 对新增指标字段按 `Option` 读取，不依赖强一致。

## 6. 验证结果（check/test/build）
- `cargo check -p snake-ui`：通过。
- `PYTHONPATH=python .venv/bin/python -m py_compile python/snake_rl/trainers/self_play.py python/snake_rl/trainers/train_loop.py`：通过。
- 训练冒烟（CPU，小配置）通过，示例输出含动态指标与策略概率：
  - `avg_steps` 不再固定常数
  - `episodes_finished` 正常累积
  - `policy_up/down/left/right` 已写入 `latest.jsonl`
- `bash scripts/check_errors.sh`：通过。

## 7. Git 锚点（branch/commit/tag）
- branch: `main`
- base commit: `d29926afedb03b5d93bda8578afe6c77d1c69897`
- tag/checkpoint: N/A（本轮未提交）
