# collect-context

目标：用最低成本收集可回放的评审上下文。

执行要点：
1. 记录当前 `branch`、`HEAD`、时间戳
2. 收集改动文件列表（工作区模式或 `base..head` 模式）
3. 输出结构化 JSON，供后续子步骤复用

MCP 工具：`review_collect_context`

常用参数：
1. `base` / `head`：指定 diff 区间（可选）
2. `output`：上下文输出路径（可选）

期望输出：
1. `refs.branch` / `refs.commit`
2. `changedFiles` 与 `totalChangedFiles`

fallback 脚本：

```bash
node scripts/review/scripts/collect-context.mjs --project-root . --base main --head HEAD
```
