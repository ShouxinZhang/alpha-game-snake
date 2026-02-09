# Review Pipeline

目标：以最简流程实现严格机械评审。

## 一键执行

```bash
bash scripts/review/run.sh
```

## 可选参数

```bash
bash scripts/review/run.sh --base main --head HEAD --llm-report scripts/review/input/llm-review.json
```

## 结果文件

- `scripts/review/artifacts/<run-id>/context.json`
- `scripts/review/artifacts/<run-id>/dependency-report.json`
- `scripts/review/artifacts/<run-id>/llm-validation.json`
- `scripts/review/artifacts/<run-id>/review-result.json`
- `scripts/review/artifacts/review-result.latest.json`

## 默认严格策略

- `BLOCK` 或 `HUMAN` 均返回非 0 退出码，便于 CI 阻断。

## MCP Tools

`scripts/review/mcp-server.mjs` 提供以下 MCP tools：

- `review_collect_context`
- `review_dependency_gate`
- `review_validate_llm`
- `review_run`

本地启动（stdio）:

```bash
node scripts/review/mcp-server.mjs
```
