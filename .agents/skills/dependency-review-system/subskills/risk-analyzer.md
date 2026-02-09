# risk-analyzer

目标：确保 LLM 风险结论可审计、可判定。

执行要点：
1. 校验 LLM 报告字段完整性（风险等级、证据、置信度）
2. 按阈值判断 `PASS/BLOCK/HUMAN`
3. 输出校验报告，作为最终裁决输入

MCP 工具：`review_validate_llm`

常用参数：
1. `input`：LLM 报告路径（默认 `scripts/review/input/llm-review.json`）
2. `policy`：阈值策略路径（可选）
3. `output`：校验结果输出路径（可选）

关键输出字段：
1. `status`：`pass` / `block` / `human_required`
2. `valid`：结构化校验是否通过
3. `reasons`：不通过原因或低置信度原因
4. `parsed`：原始 LLM 报告快照（用于追溯）

fallback 脚本：

```bash
node scripts/review/scripts/validate-llm-report.mjs --project-root . --input scripts/review/input/llm-review.json
```
