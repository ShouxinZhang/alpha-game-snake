---
name: dev-logs
description: 开发日志记录规范。每次开发循环后必须记录。
---

# 开发日志规范（通用版）

路径：

```text
docs/dev_logs/{YYYY-MM-DD}/{NN}-{summary}.md
```

## 必填内容

1. 用户原始请求（引用）
2. 轮次对话记录（背景/意图/LLM 思考摘要）
3. 修改时间（精确到秒）
4. 文件清单（路径/操作/时间/说明）
5. 变更说明（方案、影响范围、风险控制）
6. 验证结果（check/test/build）
7. Git 锚点（branch/commit/tag）

## 原则

- 禁止写入敏感信息明文
- 保证几天后可复盘
