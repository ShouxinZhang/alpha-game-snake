# test-gap-mapper

目标：把风险发现映射到最小测试补齐动作。

执行要点：
1. 读取依赖门禁与 LLM 风险报告
2. 对每个风险项给出测试建议（单测/集成测试/回归场景）
3. 将建议写入最终审查结果，便于人工快速执行

MCP 使用方式：
1. 先调用 `review_validate_llm`
2. 读取输出中的 `parsed.advisories`
3. 将每条 advisory 映射到测试动作（优先最小补测）

当前版本说明：
1. 暂无独立 MCP tool
2. 建议由 `review_validate_llm` 的结构化输出驱动测试补齐
