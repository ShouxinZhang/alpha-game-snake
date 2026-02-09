# repo-metadata (Generic)

通用仓库结构元数据工具集。

## 功能

- 扫描 git 追踪路径并维护 `docs/architecture/repo-metadata.json`
- 生成 `docs/architecture/repository-structure.md` 中目录树
- 提供 JSON ⇄ PostgreSQL 同步能力
- 提供 MCP Server（可选）

## 目录

```text
scripts/repo-metadata/
├── lib/
├── scripts/
├── sql/
├── mcp-server.mjs
└── README.md
```

## 常用命令

```bash
# 扫描并更新 JSON
node scripts/repo-metadata/scripts/scan.mjs --update

# 从 JSON 生成结构文档目录树
node scripts/repo-metadata/scripts/generate-structure-md.mjs

# 手工设置某路径描述
node scripts/repo-metadata/scripts/crud.mjs set --path src --description "source code"
```

## PostgreSQL（可选）

```bash
psql "$DATABASE_URL" -f scripts/repo-metadata/sql/001_init.sql
DATABASE_URL='postgres://...' node scripts/repo-metadata/scripts/sync-json-to-postgres.mjs
DATABASE_URL='postgres://...' node scripts/repo-metadata/scripts/sync-to-json.mjs
```

## 默认输出

- `docs/architecture/repo-metadata.json`
- `docs/architecture/repository-structure.md`

## 配置建议

`repo-metadata.json` 中 `config.scanIgnore` 建议至少包含：

```json
[
  "docs/dev_logs/**",
  "docs/private_context/**"
]
```
