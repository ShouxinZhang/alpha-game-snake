#!/usr/bin/env node
/**
 * migrate-json-to-sqlite.mjs — 将旧的 repo-metadata.json 迁移到 SQLite 数据库
 *
 * 用法:
 *   node migrate-json-to-sqlite.mjs [--json <path>]
 *
 * 行为:
 *   1. 读取 repo-metadata.json
 *   2. 创建（或打开）repo-metadata.db
 *   3. 将所有节点和配置导入 SQLite
 *
 * 注意: 这是一次性迁移脚本，迁移成功后可安全删除 repo-metadata.json
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  openMetadataDb,
  importFromJson,
  parseFlags,
} from '../lib/shared.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '../../../');
const defaultJsonPath = path.join(repoRoot, 'docs', 'architecture', 'repo-metadata.json');
const dbPath = path.join(repoRoot, 'docs', 'architecture', 'repo-metadata.db');

async function main() {
  const flags = parseFlags(process.argv.slice(2));
  const jsonPath = flags.json ? path.resolve(repoRoot, flags.json) : defaultJsonPath;

  // 检查 JSON 文件是否存在
  if (!fs.existsSync(jsonPath)) {
    console.error(`❌ JSON 文件不存在: ${jsonPath}`);
    process.exit(1);
  }

  console.log(`📂 读取 JSON: ${path.relative(repoRoot, jsonPath)}`);
  const content = fs.readFileSync(jsonPath, 'utf8');
  const jsonData = JSON.parse(content);

  const nodeCount = Object.keys(jsonData.nodes ?? {}).length;
  console.log(`   找到 ${nodeCount} 个节点`);

  if (nodeCount === 0) {
    console.log('ℹ JSON 为空，无需迁移。');
    return;
  }

  // 打开 SQLite 数据库
  console.log(`💾 打开数据库: ${path.relative(repoRoot, dbPath)}`);
  const db = openMetadataDb(dbPath);

  try {
    const imported = importFromJson(db, jsonData);
    console.log(`✅ 迁移完成: ${imported} 条记录已导入 SQLite`);
    console.log(`\n💡 数据库文件: ${path.relative(repoRoot, dbPath)}`);
    console.log('💡 确认无误后，可删除旧的 repo-metadata.json');
  } finally {
    db.close();
  }
}

main().catch((err) => {
  console.error(`❌ 迁移失败: ${err.message}`);
  process.exitCode = 1;
});
