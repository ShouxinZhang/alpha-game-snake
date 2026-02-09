#!/usr/bin/env node
/**
 * sync-to-json.mjs — 从 PostgreSQL 拉取 repo_metadata_nodes 到 repo-metadata.json
 *
 * 用法:
 *   DATABASE_URL='postgres://...' node sync-to-json.mjs
 *
 * 行为: 以 PG 为 source of truth，生成 repo-metadata.json
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

async function loadExistingConfig() {
  try {
    const content = await fs.readFile(metadataPath, 'utf8');
    const metadata = JSON.parse(content);
    return metadata.config ?? {};
  } catch {
    return {
      scanIgnore: [
        'docs/dev_logs/**',
        'docs/private_context/**',
      ],
      generateMdDepth: 2,
    };
  }
}

async function main() {
  const client = new Client({ connectionString: databaseUrl });
  await client.connect();

  try {
    const result = await client.query(`
      select path, type, description, detail, tags, sort_order, updated_by, updated_at
      from repo_metadata_nodes
      order by path
    `);

    if (result.rows.length === 0) {
      console.log('ℹ PG 表为空，无数据可同步。');
      return;
    }

    const config = await loadExistingConfig();
    const nodes = {};

    for (const row of result.rows) {
      nodes[row.path] = {
        type: row.type,
        description: row.description ?? '',
        detail: row.detail ?? '',
        tags: row.tags ?? [],
        updatedBy: row.updated_by ?? 'scan',
        updatedAt: row.updated_at?.toISOString() ?? new Date().toISOString(),
      };
    }

    const metadata = {
      version: 1,
      config,
      updatedAt: new Date().toISOString(),
      nodes,
    };

    await fs.mkdir(path.dirname(metadataPath), { recursive: true });
    await fs.writeFile(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`, 'utf8');

    console.log(`✅ PG → JSON 同步完成: ${result.rows.length} 条记录`);
  } finally {
    await client.end();
  }
}

main().catch((err) => {
  console.error(`❌ 同步失败: ${err.message}`);
  process.exitCode = 1;
});
