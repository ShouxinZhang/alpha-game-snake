# 04 - 修复 GUI 退出后 trainer 不被杀死的问题

## 用户原始请求

> 似乎退出GUI不会后台杀死训练程序？

## 背景与意图

`restart.sh` 使用 `nohup ... &` 独立启动 trainer 和 UI 两个进程，
两者完全脱离，关闭 GUI 窗口后 trainer 继续在后台运行，用户需要手动 kill。

## 修改时间

2026-02-17 （本轮对话）

## 文件清单

| 路径 | 操作 | 说明 |
|------|------|------|
| restart.sh | 修改 | 移除 nohup，增加 cleanup / trap / wait 逻辑 |

## 变更说明

### 方案

1. 移除 `nohup`，让 trainer/UI 作为 `restart.sh` 的子进程运行。
2. 增加 `cleanup()` 函数，负责 kill 所有子进程。
3. 注册 `trap cleanup EXIT INT TERM`，脚本退出时自动清理。
4. 同时运行 UI + trainer 时，`wait $UI_PID` 前台等待 UI 退出后触发 cleanup 杀死 trainer。
5. 单独运行 UI / trainer 时，同样前台等待并在退出时清理。

### 影响范围

- 仅 `restart.sh` 启动行为变更：从 fire-and-forget 改为前台管理模式。
- 用户体验：关闭 GUI 窗口 → trainer 自动停止。Ctrl+C 也会停止全部。

### 风险控制

- 如需后台独立运行 trainer（如 SSH 训练），可使用 `--train-only` + `tmux/screen`。

## 验证结果

- `bash scripts/check_errors.sh` → 通过 3 / 失败 0 / 跳过 5

## Git 锚点

- 分支: `0217`
