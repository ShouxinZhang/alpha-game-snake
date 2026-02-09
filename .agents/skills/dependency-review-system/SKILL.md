---
name: dependency-review-system
description: 依赖关系优先的 AI Review 总调度。机械门禁 + LLM 结构化判读。
---

# dependency-review-system（通用版）

## 目标

- 机械门禁阻断高风险提交
- LLM 输出结构化风险报告
- 统一结论：`PASS | BLOCK | HUMAN`

## 推荐执行（最简）

```bash
bash scripts/review/run.sh
```

## 子能力

1. collect-context
2. dependency-gate
3. validate-llm-report
4. decision-gate

## 关键输入

- 代码状态（workspace 或 diff range）
- `scripts/review/config/policy.json`
- `scripts/review/input/llm-review.json`（可选）
