#!/usr/bin/env node
/**
 * generate-structure-md.mjs — 从 repo-metadata.json 生成 repository-structure.md 的目录树部分
 *
 * 用法:
 *   node generate-structure-md.mjs [--depth N]
 *
 * 行为:
 *   1. 读取 repo-metadata.json
 *   2. 生成 ASCII 目录树（默认展开到第 2 层）
 *   3. 替换 repository-structure.md 中 <!-- REPO-TREE-START --> 到 <!-- REPO-TREE-END --> 之间的内容
 *   4. 如果文件中没有标记，则在第一个 ``` 代码块位置插入
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildTree,
  loadMetadata,
  parseFlags,
  renderTree,
  updateStructureMd,
} from '../lib/shared.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '../../../');
const metadataPath = path.join(repoRoot, 'docs', 'architecture', 'repo-metadata.json');
const structureMdPath = path.join(repoRoot, 'docs', 'architecture', 'repository-structure.md');

/* ------------------------------------------------------------------ */
/*  主入口                                                             */
/* ------------------------------------------------------------------ */

async function main() {
  const flags = parseFlags(process.argv.slice(2));

  // 读取 JSON
  let metadata;
  try {
    metadata = await loadMetadata(metadataPath);
  } catch (err) {
    console.error(`❌ 无法读取 repo-metadata.json: ${err.message}`);
    console.error('请先运行 scan.mjs --update 生成元数据。');
    process.exit(1);
  }

  const depth = flags.depth
    ? parseInt(flags.depth, 10)
    : (metadata.config?.generateMdDepth ?? 2);

  if (Object.keys(metadata.nodes).length === 0) {
    console.error('❌ repo-metadata.json 中没有节点数据。');
    process.exit(1);
  }

  // 构建树并渲染
  const tree = buildTree(metadata.nodes);
  const treeContent = renderTree(tree, depth);

  // 更新 Markdown
  await updateStructureMd(structureMdPath, treeContent);

  const nodeCount = Object.keys(metadata.nodes).length;
  console.log(`✅ 已更新 repository-structure.md（${nodeCount} 个节点，展开 ${depth} 层）`);
}

main().catch((err) => {
  console.error(`❌ 生成失败: ${err.message}`);
  process.exitCode = 1;
});
