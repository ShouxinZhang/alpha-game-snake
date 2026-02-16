#!/usr/bin/env node
/**
 * generate-structure-md.mjs — 从 SQLite 数据库生成 repository-structure.md 的目录树部分
 *
 * 用法:
 *   node generate-structure-md.mjs [--depth N]
 *
 * 行为:
 *   1. 读取 repo-metadata.db
 *   2. 生成 ASCII 目录树（默认展开到第 2 层）
 *   3. 替换 repository-structure.md 中 <!-- REPO-TREE-START --> 到 <!-- REPO-TREE-END --> 之间的内容
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildTree,
  openMetadataDb,
  getGenerateMdDepth,
  getAllNodes,
  parseFlags,
  renderTree,
  updateStructureMdSync,
} from '../lib/shared.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '../../../');
const dbPath = path.join(repoRoot, 'docs', 'architecture', 'repo-metadata.db');
const structureMdPath = path.join(repoRoot, 'docs', 'architecture', 'repository-structure.md');

async function main() {
  const flags = parseFlags(process.argv.slice(2));

  const db = openMetadataDb(dbPath);

  try {
    const depth = flags.depth
      ? parseInt(flags.depth, 10)
      : getGenerateMdDepth(db);

    const nodes = getAllNodes(db);

    if (nodes.length === 0) {
      console.error('❌ 数据库中没有节点数据。');
      console.error('请先运行 scan.mjs --update 生成元数据。');
      process.exit(1);
    }

    // 构建树并渲染
    const tree = buildTree(nodes);
    const treeContent = renderTree(tree, depth);

    // 更新 Markdown
    updateStructureMdSync(structureMdPath, treeContent);

    console.log(`✅ 已更新 repository-structure.md（${nodes.length} 个节点，展开 ${depth} 层）`);
  } finally {
    db.close();
  }
}

main().catch((err) => {
  console.error(`❌ 生成失败: ${err.message}`);
  process.exitCode = 1;
});
