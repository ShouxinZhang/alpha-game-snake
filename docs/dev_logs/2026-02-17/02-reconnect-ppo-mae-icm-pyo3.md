# 开发日志：按 test.txt 接入 PPO+ViT/MAE+ICM，并恢复 PyO3 桥接

## 1. 用户原始请求（引用）

> 根据test.txt, 衔接算法模块

> 确认按上述方案实施并恢复 pyo3 桥接

## 2. 轮次记录（背景 / 意图 / 思考摘要）

- 背景：当前分支在 2026-02-17 已清理算法栈，`python/snake_rl` 仅剩 `__pycache__`，训练链路不可用。
- 意图：基于 `test.txt` 的新算法方案（课程学习 + ViT/MAE + ICM + PPO 联合损失）重建训练域，并恢复 Rust 到 Python 的高性能桥接。
- 思考摘要：
  - 先从历史提交恢复可复用的 PyO3 接口形态，降低桥接回归风险；
  - 再按最小边界在 `python/snake_rl` 分层落地 `runtime/models/trainers`；
  - 以门禁与测试收口，确保代码可编译、可校验、可追溯。

## 3. 修改时间

- 开始时间：2026-02-17 01:42:xx +0800
- 主要改动时间：2026-02-17 01:44:xx - 01:52:xx +0800
- 记录时间：2026-02-17 01:52:44 +0800

## 4. 文件清单（路径 / 操作 / 时间 / 说明）

- `Cargo.toml` / 修改 / 01:44:xx / 恢复 workspace 级 `pyo3` 依赖。
- `crates/snake-core/Cargo.toml` / 修改 / 01:44:xx / 恢复 `cdylib`、`python` feature、可选 `pyo3`。
- `crates/snake-core/src/lib.rs` / 修改 / 01:44:xx / 恢复 `python_api` 模块导出（feature gate）。
- `crates/snake-core/src/python_api.rs` / 新增 / 01:44:xx / 恢复 `RustBatchEnv` PyO3 模块与 API。
- `python/snake_rl/__init__.py` / 新增 / 01:46:xx / 恢复训练包入口。
- `python/snake_rl/runtime/__init__.py` / 新增 / 01:46:xx / 运行时子包入口。
- `python/snake_rl/runtime/rust_env.py` / 新增 / 01:46:xx / Rust 优先 + Python fallback 的批量环境桥接。
- `python/snake_rl/models/__init__.py` / 新增 / 01:46:xx / 模型子包入口。
- `python/snake_rl/models/vit_mae_agent.py` / 新增 / 01:46:xx / `ViT_MAE_Encoder` 与 `ViTSnakeAgent`。
- `python/snake_rl/trainers/__init__.py` / 新增 / 01:46:xx / 训练子包入口。
- `python/snake_rl/trainers/curriculum.py` / 新增 / 01:46:xx / 课程学习与 15x15 zero-padding 对齐。
- `python/snake_rl/trainers/joint_loss.py` / 新增 / 01:46:xx / PPO+MAE+ICM 联合损失。
- `python/snake_rl/trainers/train_loop.py` / 新增 / 01:46:xx / 训练主循环、GAE、课程晋级、指标与 checkpoint。
- `configs/train/vit_mae_icm_ppo.yaml` / 新增 / 01:46:xx / 新训练配置模板。
- `Cargo.lock` / 修改 / 01:47:xx / 由于恢复 `pyo3` 依赖自动更新。
- `python/snake_rl/runtime/rust_env.py` / 修改 / 01:51-01:52 / 增加 `snake_core_py` / `snake_core` 双模块名导入与本地扩展自动装载兼容。
- `docs/architecture/repo-metadata.json` / 修改 / 01:50:xx / 结构扫描同步（补录 `configs/` 与 `python/` 新路径）。
- `docs/architecture/repository-structure.md` / 修改 / 01:50:xx / 目录结构文档同步（展示 `configs/`、`python/`）。

## 5. 变更说明（方案 / 影响范围 / 风险控制）

- 方案：恢复桥接能力后，将 `test.txt` 的算法设计拆分进 `runtime/models/trainers`，形成可维护的模块边界。
- 影响范围：
  - Rust 侧新增可选 Python 绑定能力（不影响默认 feature 的游戏内核行为）；
  - Python 侧恢复训练闭环与配置入口；
  - UI 与核心规则逻辑不变。
- 风险控制：
  - 大改前已创建回滚锚点（backup branch + checkpoint tag）；
  - 桥接接口沿用历史稳定签名；
  - 完整执行质量门禁、Rust 测试与结构同步。

## 6. 验证结果（check / test / build）

- `python3 -m py_compile ...`：通过。
- `cargo test -p snake-core --no-run`：通过。
- `cargo test -p snake-core --features python --no-run`：通过（验证 PyO3 特性可编译）。
- `cargo build -p snake-core --features python` + Python 动态加载 `target/debug/libsnake_core.so`：通过（`snake_core_py.RustBatchEnv` 可实例化并返回观测）。
- `bash scripts/check_errors.sh`：通过（失败 0；Python 检查通过；Node 步骤按脚本规则跳过）。
- `bash scripts/check_errors.sh`（二次复跑，含兼容补丁后）：通过。
- `bash scripts/check_errors.sh`（三次复跑，含本地扩展自动装载后）：通过。
- `cargo test`：通过（`snake-core` 与 `snake-ui` 测试全部通过）。
- `node scripts/repo-metadata/scripts/scan.mjs --update`：通过。
- `node scripts/repo-metadata/scripts/generate-structure-md.mjs`：通过。
- 运行时烟测（导入 `torch` 并执行最小 loss 前向）：失败，环境缺少 `torch`（`ModuleNotFoundError: No module named 'torch'`）。
- 运行时烟测（`PYTHONPATH=python` 直接创建 `RustBatchEnv`）：失败，环境缺少 `numpy`（`ModuleNotFoundError: No module named 'numpy'`）。

## 7. Git 锚点（branch / commit / tag / backup）

- 当前分支：`0217`
- 执行前 commit：`a8b983f`
- backup 分支：`backup/2026-02-17-reconnect-ppo-mae-icm`
- checkpoint tag：`checkpoint/2026-02-17-reconnect-ppo-mae-icm`
- 本轮提交：`N/A`（按请求完成改动与验证，未执行 commit）
