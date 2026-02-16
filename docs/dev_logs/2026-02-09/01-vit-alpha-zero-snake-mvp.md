# 开发日志：ViT + AlphaZero Snake MVP

## 1. 用户原始请求（引用）
> 用 rust + python 写一个 ViT + RL zero 的 snake game，先调查并保存 CPU/GPU/内存信息以最大化利用机器性能，然后模块化构建算法和架构，最后逐步完成整体模块构建。随后用户明确要求按实施计划直接实现。

## 2. 轮次记录（背景 / 意图 / 思考摘要）
- 背景：仓库初始仅有规范脚本，业务代码为空。
- 意图：从 0 搭建可训练 MVP，包含 Rust 高性能环境、Python 训练、PyO3 桥接与 Rust egui 监控面板。
- 执行摘要：
  - 先完成硬件探测与自动调参落盘；
  - 再完成 `snake-core` 环境内核与测试；
  - 再完成 PyO3 bridge + Python fallback runtime；
  - 再完成 ViT + AlphaZero 训练闭环；
  - 最后完成 UI 面板并执行全链路验证与质量门禁。

## 3. 修改时间（精确到秒）
- 开始时间：2026-02-09 11:38:21 CST
- 结束时间：2026-02-09 11:50:42 CST

## 4. 文件清单（路径 / 操作 / 时间 / 说明）
- `Cargo.toml` / 新增 / 11:44:xx / Rust workspace 与统一依赖。
- `.gitignore` / 新增 / 11:44:xx / Rust/Python/训练产物忽略规则。
- `crates/snake-core/Cargo.toml` / 新增 / 11:44:xx / 核心仿真 crate 配置（含 PyO3 feature）。
- `crates/snake-core/src/types.rs` / 新增 / 11:44:xx / 公共类型与批量输出结构。
- `crates/snake-core/src/env.rs` / 新增 / 11:44:xx / 单环境 Snake 规则实现。
- `crates/snake-core/src/batch.rs` / 新增 / 11:44:xx / Rayon 并行批量环境实现。
- `crates/snake-core/src/python_api.rs` / 新增 / 11:44:xx / PyO3 批量环境 API。
- `crates/snake-core/src/lib.rs` / 新增 / 11:44:xx / 导出模块与公共接口。
- `crates/snake-core/tests/rules.rs` / 新增 / 11:44:xx / 撞墙、自撞、吃果、反向动作测试。
- `crates/snake-core/tests/batch_consistency.rs` / 新增 / 11:44:xx / 单环境与批环境一致性测试。
- `crates/snake-ui/Cargo.toml` / 新增 / 11:44:xx / UI crate 配置。
- `crates/snake-ui/src/main.rs` / 新增 / 11:44:xx / 控制条、主布局、模拟驱动。
- `crates/snake-ui/src/panels/env_grid.rs` / 新增 / 11:44:xx / 16 并发网格展示。
- `crates/snake-ui/src/panels/focus_agent.rs` / 新增 / 11:44:xx / 选中智能体视图与策略热力。
- `crates/snake-ui/src/panels/metrics.rs` / 新增 / 11:44:xx / 四宫格指标曲线。
- `crates/snake-ui/src/bridge/metrics_client.rs` / 新增 / 11:44:xx / JSONL 指标轮询读取。
- `python/snake_rl/__init__.py` / 新增 / 11:44:xx / Python 包入口。
- `python/snake_rl/models/vit_policy_value.py` / 新增 / 11:44:xx / ViT 策略价值网络。
- `python/snake_rl/mcts/alpha_zero_mcts.py` / 新增 / 11:44:xx / AlphaZero planner（轻量 MCTS 近似）。
- `python/snake_rl/runtime/rust_env.py` / 新增 / 11:44:xx / Rust backend + Python fallback 向量环境。
- `python/snake_rl/runtime/replay_buffer.py` / 新增 / 11:44:xx / 回放缓冲。
- `python/snake_rl/trainers/self_play.py` / 新增 / 11:44:xx / 自博弈采样流程。
- `python/snake_rl/trainers/train_loop.py` / 新增 / 11:44:xx / 训练主循环、指标与 checkpoint。
- `python/snake_rl/trainers/train_loop.py` / 修改 / 11:50:xx / `GradScaler` API 升级到 `torch.amp.GradScaler`，消除弃用告警。
- `python/snake_rl/*/__init__.py` / 新增 / 11:44:xx / 子模块包声明。
- `configs/train/alpha_zero_vit.yaml` / 新增 / 11:44:xx / 训练配置 schema 实例。
- `scripts/system_probe/probe_hardware.py` / 新增 / 11:44:xx / 硬件探测与自动配置生成。
- `configs/runtime/hardware.auto.toml` / 新增 / 11:44:xx / 自动调参配置（由 probe 生成）。
- `docs/system_profile/2026-02-09-hardware-baseline.md` / 新增 / 11:44:xx / 硬件基线文档（由 probe 生成）。
- `docs/architecture/vit-rl-zero-snake-modules.md` / 新增 / 11:44:xx / 模块架构说明。
- `docs/architecture/repo-metadata.json` / 新增 / 11:48:xx / 仓库元数据（scan/update）。
- `docs/architecture/repository-structure.md` / 新增 / 11:48:xx / 自动目录结构文档。
- `Cargo.lock` / 新增 / 11:39-11:46 / Rust 依赖锁定（cargo 生成）。

## 5. 变更说明（方案 / 影响范围 / 风险控制）
- 方案：按“硬件基线 -> 环境内核 -> 桥接 -> 训练闭环 -> UI”顺序分治实现，优先保证可训练与可观测。
- 影响范围：新增 Rust/Python 主业务目录、训练配置、系统探测脚本、架构文档与仓库结构元数据。
- 风险控制：
  - 提供 Python fallback env，避免本地未编译 PyO3 时训练阻塞；
  - 批量环境与规则行为以 Rust tests 固化；
  - 通过 `check_errors.sh` 做语法与静态门禁；
  - UI 与训练指标通过 JSONL 解耦，降低互相阻塞风险。

## 6. 验证结果（check / test / build）
- `cargo test -p snake-core`：通过（规则与一致性测试全部通过）。
- `cargo check -p snake-core --features python`：通过（PyO3 桥接可编译）。
- `cargo check -p snake-ui`：通过（egui UI 可编译）。
- `PYTHONPATH=python .venv/bin/python -m snake_rl.trainers.train_loop --config /tmp/alpha_zero_vit_smoke.yaml`：通过（2 iter 冒烟训练，checkpoint 正常写入）。
- `bash scripts/check_errors.sh`：通过（Python 语法、未使用导入、__all__ 校验通过；Node/TS项按仓库现状跳过）。
- `node scripts/repo-metadata/scripts/scan.mjs --update`：通过。
- `node scripts/repo-metadata/scripts/generate-structure-md.mjs`：通过。

## 7. Git 锚点（branch / commit / tag）
- branch: `main`
- base commit: `d29926afedb03b5d93bda8578afe6c77d1c69897`
- tag/checkpoint: `N/A`（本轮未执行提交或打标）
