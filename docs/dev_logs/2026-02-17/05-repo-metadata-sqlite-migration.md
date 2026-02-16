# 05 — repo-metadata 从 JSON/PostgreSQL 迁移到 SQLite

## 用户指令

> 关于仓库架构，我希望用 SQLite 储存，我觉得这个要方便很多。能这样子做吗？

## 时间戳

- 开始: 2026-02-17 05:35:xx
- 结束: 2026-02-17 05:42:xx

## 变更文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/repo-metadata/lib/shared.mjs` | 重写 | 核心库从 JSON 文件读写改为 SQLite (better-sqlite3)，新增 openMetadataDb、config/node CRUD、importFromJson、exportToJson 等函数 |
| `scripts/repo-metadata/mcp-server.mjs` | 重写 | 8 个 MCP Tool 全部改为 SQLite 后端；删除 PG sync_db tool；新增 export_json tool |
| `scripts/repo-metadata/scripts/scan.mjs` | 重写 | 扫描比对改为读写 SQLite |
| `scripts/repo-metadata/scripts/crud.mjs` | 重写 | CRUD 操作改为 SQLite |
| `scripts/repo-metadata/scripts/generate-structure-md.mjs` | 重写 | 从 SQLite 读取节点生成目录树 |
| `scripts/repo-metadata/scripts/export-json.mjs` | 新增 | SQLite → JSON 导出工具 |
| `scripts/repo-metadata/scripts/migrate-json-to-sqlite.mjs` | 新增 | 一次性迁移脚本：JSON → SQLite |
| `scripts/repo-metadata/scripts/sync-json-to-postgres.mjs` | 删除 | 不再需要 PostgreSQL |
| `scripts/repo-metadata/scripts/sync-to-json.mjs` | 删除 | 不再需要 PostgreSQL |
| `scripts/repo-metadata/sql/001_init.sql` | 删除 | PG 建表脚本已用内嵌 SQLite schema 替代 |
| `scripts/repo-metadata/package.json` | 修改 | `pg` → `better-sqlite3`，版本升至 2.0.0 |
| `scripts/repo-metadata/README.md` | 重写 | 文档更新为 SQLite 方案 |
| `.gitignore` | 修改 | 添加 `node_modules/` |
| `docs/architecture/repo-metadata.db` | 新增 | SQLite 数据库文件（125 条记录迁移成功） |

## 变更详情

### 架构变更
- **存储后端**: `repo-metadata.json` (JSON flat file) + PostgreSQL (可选) → `repo-metadata.db` (SQLite)
- **依赖**: `pg` → `better-sqlite3`
- **数据库 schema**: 两张表 `config` (键值配置) + `nodes` (路径元数据)，通过 `openMetadataDb()` 首次打开时自动初始化
- **同步 API 改为同步调用**: better-sqlite3 是同步 API，代码更简洁; fs 操作也改为 sync 版本

### 保留能力
- `export-json.mjs` 和 MCP `repo_metadata_export_json` tool 可随时将 SQLite 导出为 JSON
- `migrate-json-to-sqlite.mjs` 提供一次性迁移路径

### 向后兼容
- CLI 命令名和参数不变（scan、crud、generate-structure-md）
- MCP Tool 名称和参数不变（除删除 sync_db、新增 export_json）
- `.agents/skills/repo-structure-sync/SKILL.md` 中引用的命令路径无需改动

## 验证结果

```
✅ 迁移完成: 125 条记录已导入 SQLite
✅ scan.mjs 正常: 发现 44 目录 82 文件
✅ crud.mjs get/list 正常
✅ generate-structure-md.mjs 正常
✅ export-json.mjs 正常
✅ 质量门禁: 通过 3 / 失败 0 / 跳过 5
```
