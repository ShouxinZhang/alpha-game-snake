#!/usr/bin/env node
/**
 * sync-json-to-postgres.mjs — 将 repo-metadata.json 推送到 PostgreSQL
 *
 * 用法:
 *   DATABASE_URL='postgres://...' node sync-json-to-postgres.mjs
 *
 * 行为: 以 JSON 为 source of truth，全量 upsert 到 PG
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Client } from 'pg';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '../../../');
const metadataPath = path.join(repoRoot, 'docs', 'architecture', 'repo-metadata.json');
const databaseUrl = process.env.DATABASE_URL;

if (!databaseUrl) {
  console.error('❌ 缺少 DATABASE_URL');
  process.exit(1);
}

function parentPathOf(p) {
  const parent = path.dirname(p);
  return parent === '.' ? null : parent;
}

async function main() {
  const content = await fs.readFile(metadataPath, 'utf8');
  const metadata = JSON.parse(content);
  const nodes = metadata.nodes;
  const entries = Object.entries(nodes);

  if (entries.length === 0) {
    console.log('ℹ repo-metadata.json 为空，无数据可同步。');
    return;
  }

  const client = new Client({ connectionString: databaseUrl });
  await client.connect();

  try {
    await client.query('begin');

    // 按路径深度排序，确保父路径先插入
    const sorted = entries.sort(([a], [b]) => {
      const da = a.split('/').length;
      const db = b.split('/').length;
      return da - db || a.localeCompare(b);
    });

    let upserted = 0;

    for (const [nodePath, node] of sorted) {
      const parentPath = parentPathOf(nodePath);

      await client.query(
        `
        insert into repo_metadata_nodes
          (path, type, description, detail, tags, parent_path, sort_order, updated_by)
        values ($1, $2, $3, $4, $5, $6, $7, $8)
        on conflict (path) do update set
          type        = excluded.type,
          description = excluded.description,
          detail      = excluded.detail,
          tags        = excluded.tags,
          parent_path = excluded.parent_path,
          sort_order  = excluded.sort_order,
          updated_by  = excluded.updated_by
        `,
        [
          nodePath,
          node.type,
          node.description || null,
          node.detail || null,
          node.tags ?? [],
          parentPath,
          node.sortOrder ?? 0,
          node.updatedBy ?? 'scan',
        ],
      );

      upserted++;
    }

    // 删除 JSON 中不存在的记录
    const pathSet = new Set(entries.map(([p]) => p));
    const dbRows = await client.query('select path from repo_metadata_nodes');
    let deleted = 0;

    for (const row of dbRows.rows) {
      if (!pathSet.has(row.path)) {
        await client.query('delete from repo_metadata_nodes where path = $1', [row.path]);
        deleted++;
      }
    }

    await client.query('commit');
    console.log(`✅ JSON → PG 同步完成: upsert ${upserted}, 删除 ${deleted}`);
  } catch (err) {
    await client.query('rollback');
    throw err;
  } finally {
    await client.end();
  }
}

main().catch((err) => {
  console.error(`❌ 同步失败: ${err.message}`);
  process.exitCode = 1;
});
