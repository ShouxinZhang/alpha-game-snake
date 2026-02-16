# repo-metadata (SQLite)

通用仓库结构元数据工具集，使用 SQLite 作为存储后端。

## 功能

- 扫描 git 追踪路径并维护 `docs/architecture/repo-metadata.db`
- 生成 `docs/architecture/repository-structure.md` 中目录树
- 支持导出为 JSON 格式（方便查看）
- 提供 MCP Server（可选）

## 目录

```text
scripts/repo-metadata/
├── lib/
│   └── shared.mjs          # 核心库: SQLite 操作 + 工具函数
├── scripts/
│   ├── scan.mjs             # 扫描 & 同步
│   ├── crud.mjs             # CRUD 操作
│   ├── tree.mjs             # 彩色 tree 可视化
│   ├── generate-structure-md.mjs  # 生成目录树文档
│   ├── export-json.mjs      # 导出为 JSON
│   └── migrate-json-to-sqlite.mjs # 从旧 JSON 迁移
├── mcp-server.mjs           # MCP Server
├── package.json
└── README.md
```

## 常用命令

```bash
# 扫描并更新数据库
node scripts/repo-metadata/scripts/scan.mjs --update

# 可视化仓库树结构（彩色、带描述）
node scripts/repo-metadata/scripts/tree.mjs
node scripts/repo-metadata/scripts/tree.mjs --depth 4 --path crates

# 从数据库生成结构文档目录树
node scripts/repo-metadata/scripts/generate-structure-md.mjs

# 手工设置某路径描述
node scripts/repo-metadata/scripts/crud.mjs set --path src --description "source code"

# 导出为 JSON（方便查看/备份）
node scripts/repo-metadata/scripts/export-json.mjs

# 从旧 JSON 迁移到 SQLite（一次性）
node scripts/repo-metadata/scripts/migrate-json-to-sqlite.mjs
```

## 数据库

SQLite 数据库文件位于 `docs/architecture/repo-metadata.db`，包含两张表：

- **config** — 键值配置（scanIgnore、generateMdDepth 等）
- **nodes** — 仓库路径元数据（path、type、description、detail、tags 等）

## 默认输出

- `docs/architecture/repo-metadata.db`（主存储）
- `docs/architecture/repository-structure.md`（生成的目录树文档）

## 配置

通过 `config` 表管理，默认值：

| key | 默认值 | 说明 |
|-----|--------|------|
| `scanIgnore` | `["docs/dev_logs/**", "docs/private_context/**"]` | 扫描忽略的 glob 模式 |
| `generateMdDepth` | `2` | 目录树展开深度 |
