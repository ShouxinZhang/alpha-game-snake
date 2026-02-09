---
name: modularization-governance
description: 机械量化模块化治理。输出 PASS/REFINE/BLOCK 并形成修复闭环。
---

# 模块化治理工作流（通用版）

## Step 1：定义策略

- 使用 `references/modularity-policy.template.json`
- 设置 layerOrder、entryFiles、thresholds

## Step 2：运行检查

```bash
node .agents/skills/modularization-governance/scripts/check-modularity.mjs \
  --target <module-path> \
  --policy .agents/skills/modularization-governance/references/modularity-policy.template.json \
  --output .agents/skills/modularization-governance/artifacts/<module>.modularity-report.json

node .agents/skills/modularization-governance/scripts/check-unused-symbols.mjs \
  --target <module-path> \
  --tsconfig <tsconfig-path> \
  --export-ignore-files index.ts,index.tsx \
  --output .agents/skills/modularization-governance/artifacts/<module>.unused-report.json
```

## Step 3：按报告修复

- 先修 `BLOCK`
- 再收敛 `REFINE`
- 每轮修复后复跑

## Step 4：最终门禁

```bash
bash scripts/check_errors.sh
```
