# 开发日志：新增 restart.sh

## 1. 用户原始请求（引用）
> create a restart.sh

## 2. 轮次对话记录（背景/意图/思考摘要）
- 背景：仓库已具备训练与 UI 启动入口，但缺少统一重启脚本。
- 意图：提供一键重启能力，减少手动停止旧进程、重启训练和 UI 的重复操作成本。
- 执行摘要：
  - 新增根目录 `restart.sh`；
  - 支持停止旧进程并按参数重启 trainer / UI；
  - 补充脚本帮助信息与参数检查；
  - 完成语法验证、质量门禁、结构文档同步与日志落盘。

## 3. 修改时间（精确到秒）
- 开始时间：2026-02-09 12:45:17 CST
- 结束时间：2026-02-09 12:45:55 CST

## 4. 文件清单（路径/操作/时间/说明）
- `restart.sh` / 新增 / 12:45:xx / 一键重启脚本，包含进程清理与重启逻辑。
- `docs/architecture/repo-metadata.json` / 修改 / 12:45:xx / 扫描后新增 `restart.sh` 与 `docs/dev_logs` 节点。
- `docs/architecture/repository-structure.md` / 修改 / 12:45:xx / 目录结构同步更新。
- `docs/dev_logs/2026-02-09/02-create-restart-sh.md` / 新增 / 12:45:xx / 本轮开发日志。

## 5. 变更说明（方案、影响范围、风险控制）
- 方案：新增 `restart.sh` 并采用“先停旧进程，再按选项拉起新进程”的稳定重启模式。
- 影响范围：仅新增运维辅助脚本与文档元数据，不改动训练/环境/UI业务逻辑。
- 风险控制：
  - 脚本内校验 `.venv/bin/python` 和配置文件存在性；
  - 默认日志落盘 `logs/train.log` 与 `logs/ui.log`，便于故障定位；
  - 支持 `--train-only` / `--ui-only` 降低误操作影响。

## 6. 验证结果（check/test/build）
- `bash -n restart.sh`：通过。
- `./restart.sh --help`：通过（参数帮助输出正确）。
- `bash scripts/check_errors.sh`：通过（Python 检查通过，Node/TS 按仓库现状跳过）。
- `node scripts/repo-metadata/scripts/scan.mjs --update`：通过。
- `node scripts/repo-metadata/scripts/generate-structure-md.mjs`：通过。

## 7. Git 锚点（branch/commit/tag）
- branch: `main`
- base commit: `d29926afedb03b5d93bda8578afe6c77d1c69897`
- tag/checkpoint: `N/A`（本轮未执行提交/打标签）
