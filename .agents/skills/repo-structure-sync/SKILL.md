---
name: repo-structure-sync
description: 仓库架构文档同步。结构变更后必须更新仓库元数据与结构文档。
---

# 仓库结构同步规范（通用版）

## 触发条件

- 新增/删除/移动文件或目录
- 新增依赖或 npm scripts

## 执行流程

1. 扫描并更新元数据：

```bash
node scripts/repo-metadata/scripts/scan.mjs --update
```

2. 生成结构文档：

```bash
node scripts/repo-metadata/scripts/generate-structure-md.mjs
```

3. 补充新增路径描述（如有）

## 约束

- 目录树自动生成区域不可手工改写
