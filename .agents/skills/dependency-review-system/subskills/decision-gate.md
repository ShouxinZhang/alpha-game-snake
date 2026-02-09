# decision-gate

目标：将多源结果统一为一个机器可消费结论。

判定顺序：
1. 机械门禁失败 -> `BLOCK`
2. 机械门禁通过但 LLM 缺失/低置信度 -> `HUMAN`
3. 机械门禁通过且 LLM 有阻断项 -> `BLOCK`
4. 其余 -> `PASS`

MCP 工具：`review_run`

常用参数：
1. `base` / `head`：可选 diff 区间
2. `llmReport`：LLM 报告路径（可选）
3. `allowHuman`：是否允许 `HUMAN` 状态返回成功退出码
4. `output`：最终结果输出路径（可选）

输出文件：`review-result.json`
退出码：
- `0` = PASS
- `1` = BLOCK 或 HUMAN（默认严格）

fallback 脚本：

```bash
node scripts/review/scripts/run-review.mjs --project-root . --llm-report scripts/review/input/llm-review.json
```
