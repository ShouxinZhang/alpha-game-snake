# 开发日志：初始化分支清理算法栈

## 1. 用户原始请求（引用）

> 这是一个全新的branch, 我希望保持一个初始化状态，也就是说仅仅包含游戏主体以及skills这些  
> 算法相关的东西全部clear

## 2. 轮次记录（背景 / 意图 / 思考摘要）

- 背景：当前分支同时包含游戏主体（Rust core + UI）与训练/算法栈（Python RL、训练配置、训练文档、重启脚本等）。
- 意图：将分支收敛为“初始化状态”，仅保留游戏主体与 `.agents/skills` 能力。
- 思考摘要：
  - 先对齐清理范围并获得用户确认；
  - 大改前建立可回滚锚点（backup branch + checkpoint tag）；
  - 删除算法代码与文档后，同步 Rust 依赖与 UI 逻辑，避免残留编译入口；
  - 执行质量门禁与测试，最后同步仓库结构文档。

## 3. 修改时间

- 开始时间：2026-02-17 01:30:46 +0800
- 主要改动时间：2026-02-17 01:31:00 - 01:33:14 +0800
- 记录时间：2026-02-17 01:33:14 +0800

## 4. 文件清单（路径 / 操作 / 时间 / 说明）

- `python/snake_rl/**` / 删除 / 01:31:xx / 清理全部 RL 训练代码（models/mcts/runtime/trainers）。
- `configs/train/alpha_zero_vit.yaml` / 删除 / 01:31:xx / 清理训练配置。
- `configs/runtime/hardware.auto.toml` / 删除 / 01:31:xx / 清理训练硬件配置。
- `scripts/system_probe/probe_hardware.py` / 删除 / 01:31:xx / 清理训练硬件探测脚本。
- `restart.sh` / 删除 / 01:31:xx / 清理训练+UI 联动重启脚本。
- `docs/architecture/vit-rl-system-principles.md` / 删除 / 01:31:xx / 清理算法原理文档。
- `docs/architecture/vit-rl-zero-snake-modules.md` / 删除 / 01:31:xx / 清理 RL 架构文档。
- `docs/system_profile/2026-02-09-hardware-baseline.md` / 删除 / 01:31:xx / 清理训练硬件基线文档。
- `docs/dev_logs/2026-02-09/*.md` / 删除 / 01:31:xx / 清理历史训练相关开发日志。
- `crates/snake-core/src/python_api.rs` / 删除 / 01:31:xx / 清理 Python 桥接入口。
- `crates/snake-core/src/lib.rs` / 修改 / 01:31:xx / 移除 `python_api` 模块导出。
- `crates/snake-core/Cargo.toml` / 修改 / 01:31:xx / 移除 `python` feature、`pyo3` 依赖和 `cdylib` 输出。
- `crates/snake-ui/src/bridge/metrics_client.rs` / 删除 / 01:31:xx / 清理训练 metrics 文件读取桥接。
- `crates/snake-ui/src/panels/metrics.rs` / 删除 / 01:31:xx / 清理训练指标图表面板。
- `crates/snake-ui/src/main.rs` / 修改 / 01:31:xx / UI 改为纯游戏监控，不再读取训练 metrics。
- `crates/snake-ui/src/panels/focus_agent.rs` / 修改 / 01:31:xx / 文案去 RL 指向，保留棋盘焦点展示。
- `crates/snake-ui/Cargo.toml` / 修改 / 01:31:xx / 移除 `egui_plot/serde/serde_json` 依赖。
- `Cargo.toml` / 修改 / 01:31:xx / 移除 workspace 级 `pyo3/egui_plot/serde_json` 依赖。
- `Cargo.lock` / 修改 / 01:32:xx / 依赖锁文件随 `cargo test` 自动更新。
- `docs/architecture/repo-metadata.json` / 修改 / 01:32:xx / 结构扫描更新。
- `docs/architecture/repository-structure.md` / 修改 / 01:32:xx / 目录树同步为当前初始化态。

## 5. 变更说明（方案 / 影响范围 / 风险控制）

- 方案：按“先删算法资产 -> 再清 Rust 引用 -> 最后校验与文档同步”的顺序执行，确保删除后仍可编译运行。
- 影响范围：训练链路、算法文档、训练脚本已全部移除；保留 `crates/snake-core`、`crates/snake-ui` 和 `.agents/skills`。
- 风险控制：
  - 删除前创建回滚点（backup branch + checkpoint tag）；
  - 对 UI 做最小行为改造，避免引入新业务；
  - 执行强制质量门禁与 Rust 测试。

## 6. 验证结果（check / test / build）

- `bash scripts/check_errors.sh`：通过（失败 0；Python 相关检查通过；Node 步骤按脚本规则跳过）。
- `cargo test`：通过（`snake-core` 规则测试与 batch 一致性测试均通过）。
- `node scripts/repo-metadata/scripts/scan.mjs --update`：通过。
- `node scripts/repo-metadata/scripts/generate-structure-md.mjs`：通过。

## 7. Git 锚点（branch / commit / tag / backup）

- 当前分支：`0217`
- 执行前 commit：`f280486`
- backup 分支：`backup/2026-02-17-init-cleanup`
- checkpoint tag：`checkpoint/2026-02-17-init-cleanup`
- 本轮提交：`N/A`（按用户请求完成改动与验证，未执行 commit）
