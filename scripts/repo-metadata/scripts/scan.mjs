#!/usr/bin/env node
/**
 * scan.mjs — 扫描仓库目录结构，对比 repo-metadata.json，报告新增/删除/未描述条目
 *
 * 用法:
 *   node scan.mjs [--max-depth N] [--update]
 *
 * 选项:
 *   --max-depth N   最大扫描深度（默认: 无限制，扫到叶子节点）
 *   --update        自动更新 repo-metadata.json（添加新条目、移除已删除条目）
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  depthOf,
  getTrackedPaths,
  globToRegex,
  loadMetadata,
  parseFlags,
  saveMetadata,
  shouldIgnore,
} from '../lib/shared.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '../../../');
const metadataPath = path.join(repoRoot, 'docs', 'architecture', 'repo-metadata.json');

/* ------------------------------------------------------------------ */
/*  核心逻辑                                                           */
/* ------------------------------------------------------------------ */

function buildIgnoreMatchers(patterns) {
  return (patterns ?? []).map(globToRegex);
}

async function main() {
  const flags = parseFlags(process.argv.slice(2));
  const maxDepth = flags['max-depth'] ? parseInt(flags['max-depth'], 10) : null;
  const shouldUpdate = flags.update === 'true';

  console.log('📁 Scanning repository...');

  const { fileSet, dirSet } = getTrackedPaths(repoRoot);
  const metadata = await loadMetadata(metadataPath);
  const ignoreMatchers = buildIgnoreMatchers(metadata.config?.scanIgnore);

  // 构建磁盘路径 → 类型映射
  const diskPaths = new Map();
  for (const d of dirSet) {
    if (!shouldIgnore(d, ignoreMatchers)) {
      diskPaths.set(d, 'directory');
    }
  }
  for (const f of fileSet) {
    if (!shouldIgnore(f, ignoreMatchers)) {
      diskPaths.set(f, 'file');
    }
  }

  // 应用深度过滤
  const filteredPaths = maxDepth
    ? new Map([...diskPaths].filter(([p]) => depthOf(p) <= maxDepth))
    : diskPaths;

  const dirCount = [...filteredPaths.values()].filter((t) => t === 'directory').length;
  const fileCount = [...filteredPaths.values()].filter((t) => t === 'file').length;
  console.log(`Found ${dirCount} directories, ${fileCount} files`);

  const existingPaths = new Set(Object.keys(metadata.nodes));

  // 对比: 新增 / 删除 / 未描述
  const added = [];
  const undescribed = [];

  for (const [p, type] of filteredPaths) {
    if (!existingPaths.has(p)) {
      added.push({ path: p, type });
    } else if (!metadata.nodes[p].description) {
      undescribed.push(p);
    }
  }

  const removed = [];
  for (const p of existingPaths) {
    if (!filteredPaths.has(p)) {
      removed.push(p);
    }
  }

  // 输出报告
  if (added.length > 0) {
    console.log(`\n🆕 New paths (${added.length}):`);
    for (const { path: p, type } of added.sort((a, b) => a.path.localeCompare(b.path))) {
      console.log(`  + ${p}  (${type})`);
    }
  }

  if (removed.length > 0) {
    console.log(`\n🗑️  Removed paths (${removed.length}):`);
    for (const p of removed.sort()) {
      console.log(`  - ${p}`);
    }
  }

  if (undescribed.length > 0) {
    console.log(`\n⚠️  Undescribed paths (${undescribed.length}):`);
    for (const p of undescribed.sort()) {
      console.log(`  ? ${p}`);
    }
  }

  if (added.length === 0 && removed.length === 0) {
    console.log('\n✅ Metadata is up to date with filesystem.');
  }

  // 更新 JSON
  if (shouldUpdate) {
    const now = new Date().toISOString();

    for (const { path: p, type } of added) {
      metadata.nodes[p] = {
        type,
        description: '',
        detail: '',
        tags: [],
        updatedBy: 'scan',
        updatedAt: now,
      };
    }

    for (const p of removed) {
      delete metadata.nodes[p];
    }

    // 按路径排序 nodes
    const sorted = Object.keys(metadata.nodes).sort();
    const orderedNodes = {};
    for (const key of sorted) {
      orderedNodes[key] = metadata.nodes[key];
    }
    metadata.nodes = orderedNodes;

    await saveMetadata(metadataPath, metadata);
    console.log(`\n✅ Updated repo-metadata.json: ${added.length} added, ${removed.length} removed`);
  } else if (added.length > 0 || removed.length > 0) {
    console.log('\n💡 Run with --update to apply changes to repo-metadata.json');
  }

  console.log(
    `\nSummary: ${added.length} new, ${removed.length} removed, ${undescribed.length} undescribed`,
  );
}

main().catch((err) => {
  console.error(`❌ Scan failed: ${err.message}`);
  process.exitCode = 1;
});
