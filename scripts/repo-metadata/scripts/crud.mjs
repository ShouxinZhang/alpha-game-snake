#!/usr/bin/env node
/**
 * crud.mjs — 仓库元数据 CRUD（直接操作 SQLite 数据库）
 *
 * 用法:
 *   node crud.mjs get    --path <path>
 *   node crud.mjs set    --path <path> [--description <text>] [--detail <text>] [--tags <a,b,c>] [--type <file|directory>] [--updated-by <scan|llm|human>]
 *   node crud.mjs delete --path <path>
 *   node crud.mjs list   [--undescribed] [--type <file|directory>] [--max-depth <n>] [--tag <tag>]
 *   node crud.mjs batch-set < descriptions.json
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  openMetadataDb,
  getNode,
  upsertNode,
  deleteNodeByPath,
  listNodes,
  parseFlags,
} from '../lib/shared.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '../../../');
const dbPath = path.join(repoRoot, 'docs', 'architecture', 'repo-metadata.db');

/* ------------------------------------------------------------------ */
/*  用法提示                                                           */
/* ------------------------------------------------------------------ */

function printUsage() {
  console.log(`
仓库元数据 CRUD (SQLite)

用法:
  node crud.mjs get    --path <path>
  node crud.mjs set    --path <path> [--description <text>] [--detail <text>] [--tags <a,b,c>] [--type <file|directory>] [--updated-by <scan|llm|human>]
  node crud.mjs delete --path <path>
  node crud.mjs list   [--undescribed] [--type <file|directory>] [--max-depth <n>] [--tag <tag>]
  node crud.mjs batch-set < descriptions.json

batch-set 输入格式 (JSON):
  [
    { "path": "src", "description": "源代码目录", "detail": "...", "tags": ["core"] },
    ...
  ]
`);
}

/* ------------------------------------------------------------------ */
/*  CRUD 操作                                                          */
/* ------------------------------------------------------------------ */

function cmdGet(db, flags) {
  const p = flags.path;
  if (!p) throw new Error('get 需要 --path');

  const node = getNode(db, p);
  if (!node) {
    console.error(`❌ 路径不存在: ${p}`);
    process.exitCode = 1;
    return;
  }

  console.log(JSON.stringify({ path: p, ...node }, null, 2));
}

function cmdSet(db, flags) {
  const p = flags.path;
  if (!p) throw new Error('set 需要 --path');

  const fields = {
    updatedBy: flags['updated-by'] ?? 'human',
  };
  if ('description' in flags) fields.description = flags.description;
  if ('detail' in flags) fields.detail = flags.detail;
  if ('tags' in flags) fields.tags = flags.tags.split(',').map((t) => t.trim()).filter(Boolean);
  if ('type' in flags) fields.type = flags.type;

  upsertNode(db, p, fields);
  console.log(`✅ 已更新: ${p}`);
}

function cmdDelete(db, flags) {
  const p = flags.path;
  if (!p) throw new Error('delete 需要 --path');

  const existing = getNode(db, p);
  if (!existing) {
    console.error(`❌ 路径不存在: ${p}`);
    process.exitCode = 1;
    return;
  }

  const { cascaded } = deleteNodeByPath(db, p);
  console.log(`✅ 已删除: ${p}${cascaded > 0 ? ` (+ ${cascaded} 个子路径)` : ''}`);
}

function cmdList(db, flags) {
  const maxDepth = flags['max-depth'] ? parseInt(flags['max-depth'], 10) : null;
  const filterType = flags.type ?? null;
  const filterTag = flags.tag ?? null;
  const onlyUndescribed = flags.undescribed === 'true';

  const entries = listNodes(db, {
    type: filterType,
    tag: filterTag,
    maxDepth,
    undescribedOnly: onlyUndescribed,
  });

  if (entries.length === 0) {
    console.log('没有匹配的条目。');
    return;
  }

  const maxPathLen = Math.min(
    Math.max(...entries.map((n) => n.path.length)),
    60,
  );

  for (const node of entries) {
    const typeIcon = node.type === 'directory' ? '📁' : '📄';
    const desc = node.description || '(未描述)';
    const padding = ' '.repeat(Math.max(1, maxPathLen - node.path.length + 2));
    console.log(`${typeIcon} ${node.path}${padding}${desc}`);
  }

  console.log(`\n共 ${entries.length} 条`);
}

async function cmdBatchSet(db) {
  // 从 stdin 读取 JSON 数组
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  const input = Buffer.concat(chunks).toString('utf8');

  let items;
  try {
    items = JSON.parse(input);
  } catch {
    throw new Error('无法解析 stdin JSON，格式应为 [{ "path": "...", "description": "..." }, ...]');
  }

  if (!Array.isArray(items)) {
    throw new Error('输入应为 JSON 数组');
  }

  let updated = 0;

  const batch = db.transaction(() => {
    for (const item of items) {
      if (!item.path) {
        console.warn('⚠️ 跳过: 缺少 path 字段');
        continue;
      }

      const existing = getNode(db, item.path);
      if (!existing) {
        console.warn(`⚠️ 跳过: 路径不在数据库中: ${item.path}`);
        continue;
      }

      upsertNode(db, item.path, {
        description: item.description,
        detail: item.detail,
        tags: item.tags,
        updatedBy: item.updatedBy ?? 'llm',
      });
      updated++;
    }
  });
  batch();

  console.log(`✅ 批量更新完成: ${updated}/${items.length} 条`);
}

/* ------------------------------------------------------------------ */
/*  主入口                                                             */
/* ------------------------------------------------------------------ */

async function main() {
  const [command, ...rest] = process.argv.slice(2);

  if (!command) {
    printUsage();
    process.exit(1);
  }

  const flags = parseFlags(rest);
  const db = openMetadataDb(dbPath);

  try {
    switch (command) {
      case 'get':
        cmdGet(db, flags);
        break;
      case 'set':
        cmdSet(db, flags);
        break;
      case 'delete':
        cmdDelete(db, flags);
        break;
      case 'list':
        cmdList(db, flags);
        break;
      case 'batch-set':
        await cmdBatchSet(db);
        break;
      default:
        printUsage();
        process.exit(1);
    }
  } finally {
    db.close();
  }
}

main().catch((err) => {
  console.error(`❌ 执行失败: ${err.message}`);
  process.exitCode = 1;
});
