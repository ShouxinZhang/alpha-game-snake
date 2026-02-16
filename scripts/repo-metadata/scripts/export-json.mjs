#!/usr/bin/env node
/**
 * export-json.mjs — 将 SQLite 数据库导出为 JSON 文件
 *
 * 用法:
 *   node export-json.mjs [--output <path>]
 *
 * 选项:
 *   --output <path>   输出文件路径（默认: docs/architecture/repo-metadata.json）
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  openMetadataDb,
  exportToJson,
  parseFlags,
} from '../lib/shared.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '../../../');
const dbPath = path.join(repoRoot, 'docs', 'architecture', 'repo-metadata.db');

async function main() {
  const flags = parseFlags(process.argv.slice(2));
  const outputPath = flags.output
    ? path.resolve(repoRoot, flags.output)
    : path.join(repoRoot, 'docs', 'architecture', 'repo-metadata.json');

  const db = openMetadataDb(dbPath);

  try {
    const json = exportToJson(db);
    const jsonStr = JSON.stringify(json, null, 2);

    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, `${jsonStr}\n`, 'utf8');

    const nodeCount = Object.keys(json.nodes).length;
    console.log(`✅ 已导出到 ${path.relative(repoRoot, outputPath)}（${nodeCount} 条记录）`);
  } finally {
    db.close();
  }
}

main().catch((err) => {
  console.error(`❌ 导出失败: ${err.message}`);
  process.exitCode = 1;
});
