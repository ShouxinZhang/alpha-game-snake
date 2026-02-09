# 模块化机械规则（脚本版）

本文件定义 `check-modularity.mjs` 的可机读规则，不依赖人工主观打分。

## 状态机

1. 命中任一 `BLOCK` 规则：输出 `BLOCK`
2. 无 `BLOCK` 且命中任一 `REFINE` 规则：输出 `REFINE`
3. 未命中任何规则：输出 `PASS`

## 规则清单

| 代码 | 严重级别 | 触发条件 | 默认阈值（policy） |
|---|---|---|---|
| `CIRCULAR_DEPENDENCY` | `BLOCK` | 内部依赖图存在环路 | `maxCircularDeps = 0` |
| `REVERSE_LAYER_IMPORT` | `BLOCK` | 低层反向依赖高层 | `maxReverseLayerImports = 0` |
| `FILE_TOO_LARGE` | `REFINE` | 单文件行数过大 | `maxFileLines = 400` |
| `UNUSED_DECLARATION` | `REFINE` | 变量/参数/局部函数声明后未被读取 | 由 `check-unused-symbols.mjs` 检测 |
| `UNUSED_IMPORT_DECLARATION` | `REFINE` | import 声明未被使用 | 由 `check-unused-symbols.mjs` 检测 |
| `UNUSED_EXPORT` | `REFINE` | 导出符号未被其他模块导入或再导出 | 由 `check-unused-symbols.mjs` 检测 |
| `HIGH_FAN_OUT` | `REFINE` | 单文件依赖数过大 | `maxFanOut = 12` |
| `DEEP_RELATIVE_IMPORT` | `REFINE` | 相对导入层级过深 | `maxRelativeImportDepth = 3` |
| `PUBLIC_API_TOO_WIDE` | `REFINE` | 入口文件导出面过宽 | `maxPublicExports = 25` |
| `ORPHAN_FILES` | `REFINE` | 孤立文件数量超标 | `maxOrphanFiles = 0` |
| `UNKNOWN_LAYER_FILES` | `REFINE` | 未归层文件数量超标 | `maxUnknownLayerFiles = 0` |

## 报告结构

脚本输出 JSON 包含：

1. `summary.status`: `PASS | REFINE | BLOCK`
2. `summary.metrics`: 量化指标（循环数、反向依赖数、孤立文件数等）
3. `findings[]`: 问题明细（`severity`、`code`、`files`、`suggestion`、`evidence`）

## 推荐用法（LLM 分治）

1. 先修 `BLOCK`（保证框架正确）
2. 再收敛 `REFINE`（控制变更半径）
3. 每轮修复后重跑脚本，直到满足当前阶段目标

## 误删防护（UNUSED_EXPORT）

1. `UNUSED_EXPORT` 的语义是“导出未被消费”，不是“定义一定无用”。
2. 修复顺序必须是：先移除 `export`，再评估定义是否可删。
3. 若定义仍被模块内类型/函数签名使用，应保留定义本体。
