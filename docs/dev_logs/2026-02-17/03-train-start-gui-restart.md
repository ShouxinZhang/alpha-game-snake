# 开发日志：打通可启动训练 + GUI 监控 + restart.sh 一键启动

## 1. 用户原始请求（引用）

> 继续到“可启动训练”
> 以及可视化GUI窗口呢？
> restart.sh一键启动呢？

## 2. 轮次记录（背景 / 意图 / 思考摘要）

- 背景：上一轮已恢复算法链路与 PyO3 桥接，但尚未完成“可直接启动训练 + GUI 监控 + 一键脚本”的交付闭环。
- 意图：将训练启动、可视化监控、运维启动方式合并为可执行路径，降低本地启动复杂度。
- 思考摘要：
  - 复用现有 `snake-ui` 画面，不改现有棋盘监控能力，只增训练指标面板；
  - 复用 `artifacts/metrics/latest.jsonl` 作为训练与 GUI 的解耦接口；
  - 用 `restart.sh` 统一进程管理、日志收集与 PyO3 扩展构建。

## 3. 修改时间

- 开始时间：2026-02-17 04:03:xx +0800
- 主要改动时间：2026-02-17 04:04:xx - 04:08:xx +0800
- 记录时间：2026-02-17 04:08:07 +0800

## 4. 文件清单（路径 / 操作 / 时间 / 说明）

- `Cargo.toml` / 修改 / 04:04:xx / 新增 `egui_plot`、`serde_json` workspace 依赖。
- `crates/snake-ui/Cargo.toml` / 修改 / 04:04:xx / 接入 `egui_plot`、`serde`、`serde_json`。
- `crates/snake-ui/src/bridge/mod.rs` / 新增 / 04:04:xx / 桥接模块入口。
- `crates/snake-ui/src/bridge/metrics_client.rs` / 新增 / 04:04:xx / 训练指标 JSONL 轮询读取。
- `crates/snake-ui/src/panels/mod.rs` / 新增 / 04:05:xx / 面板模块入口。
- `crates/snake-ui/src/panels/metrics.rs` / 新增 / 04:05:xx / 训练曲线面板（reward/score/loss/intrinsic）。
- `crates/snake-ui/src/main.rs` / 修改 / 04:05:xx / 接入训练指标轮询与可视化展示。
- `restart.sh` / 新增 / 04:06:xx / 一键启动脚本（训练+UI、进程清理、日志、可选扩展构建）。
- `docs/architecture/repo-metadata.json` / 修改 / 04:07:xx / 结构扫描同步新增 UI/脚本路径。
- `docs/architecture/repository-structure.md` / 修改 / 04:07:xx / 目录结构同步。

## 5. 变更说明（方案 / 影响范围 / 风险控制）

- 方案：通过指标文件解耦训练与 GUI，避免 UI 与训练进程强耦合；运维侧通过 `restart.sh` 标准化启动入口。
- 影响范围：
  - `snake-ui` 增加训练监控能力，不影响原有棋盘可视化；
  - 训练模块无协议变更，仅消费既有 `latest.jsonl` 输出；
  - 新增根目录一键启动脚本。
- 风险控制：
  - `restart.sh` 在启动前清理历史进程，降低端口/资源冲突；
  - PyO3 扩展构建失败时提供回退路径（Python fallback）；
  - 保持训练配置可覆盖（`--config`），避免硬编码。

## 6. 验证结果（check / test / build）

- `PYTHONPATH=python .venv/bin/python -m snake_rl.trainers.train_loop --config /tmp/vit_mae_icm_ppo_smoke.yaml`：通过（2 iter 完成，日志显示 `mode=rust`）。
- `cargo check -p snake-ui`：通过。
- `bash restart.sh --train-only --config /tmp/vit_mae_icm_ppo_smoke.yaml --skip-build-ext`：通过（脚本启动训练并输出迭代日志）。
- `bash restart.sh --ui-only --skip-build-ext`：在当前会话可编译并执行到 `Running target/debug/snake-ui` 日志；GUI 持续驻留受当前终端/桌面会话约束（无功能性崩溃日志）。
- `bash scripts/check_errors.sh`：通过（失败 0）。
- `cargo test`：通过（`snake-core` / `snake-ui`）。
- `node scripts/repo-metadata/scripts/scan.mjs --update`：通过。
- `node scripts/repo-metadata/scripts/generate-structure-md.mjs`：通过。

## 7. Git 锚点（branch / commit / tag / backup）

- 当前分支：`0217`
- 当前基线 commit：`a8b983f`
- 参考保护点：`backup/2026-02-17-reconnect-ppo-mae-icm`
- 参考 checkpoint：`checkpoint/2026-02-17-reconnect-ppo-mae-icm`
- 本轮提交：`N/A`（未执行 commit）
