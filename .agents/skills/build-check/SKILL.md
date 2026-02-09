---
name: build-check
description: 代码构建全链路质量门禁。代码改动后必须执行。
---

# 构建检查规范（通用版）

## 默认命令

```bash
bash scripts/check_errors.sh
```

## 补充测试

```bash
npm test
```

## 执行要求

1. 每次代码改动后执行。
2. 如失败，修复并重跑，直至全部通过。
3. 将结果记录到 dev log。
