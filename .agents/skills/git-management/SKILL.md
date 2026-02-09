---
name: git-management
description: Git 管理规范。阶段成果或大改前后建立可追溯、可回滚节奏。
---

# Git 管理规范（通用版）

## 阶段提交

```bash
git add -A
git commit -m "<type>: <milestone-summary>"
```

## 大改前保护点

```bash
git branch backup/<date>-<topic>
git tag -a checkpoint/<date>-<topic> -m "before large change"
```

## 质量门禁

```bash
bash scripts/check_errors.sh
npm test
```

## 日志回写

- 回写 `branch / commit / tag / backup` 到对应 dev log
