# dependency-guard

目标：把架构约束从文档规则变成可执行阻断。

执行要点：
1. 基于 `madge` 生成依赖图与循环依赖列表
2. 根据 `scripts/review/config/policy.json` 检查禁止依赖边
3. 输出依赖门禁报告并给出阻断结果

MCP 工具：`review_dependency_gate`

常用参数：
1. `policy`：策略文件路径（可选，默认 `scripts/review/config/policy.json`）
2. `output`：报告输出路径（可选）

判定依据：
1. `summary.cycleCount` 是否超过阈值
2. `summary.forbiddenCount` 是否超过阈值
3. `summary.passed` 是否为 `true`

fallback 脚本：

```bash
node scripts/review/scripts/dependency-gate.mjs --project-root . --policy scripts/review/config/policy.json
```
