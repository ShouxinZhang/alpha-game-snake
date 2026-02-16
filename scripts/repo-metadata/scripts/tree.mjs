#!/usr/bin/env node
/**
 * tree.mjs — 在终端中可视化仓库结构树（带描述注释，支持彩色输出）
 *
 * 用法:
 *   node tree.mjs [--depth N] [--no-color] [--path <subpath>] [--dirs-only]
 *
 * 选项:
 *   --depth N       展开深度（默认: 3）
 *   --no-color      关闭彩色输出
 *   --path <path>   只显示指定子树
 *   --dirs-only     只显示目录节点（适合 LLM 快速理解模块架构）
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  openMetadataDb,
  getGenerateMdDepth,
  getAllNodes,
  getNode,
  parseFlags,
} from '../lib/shared.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '../../../');
const dbPath = path.join(repoRoot, 'docs', 'architecture', 'repo-metadata.db');

/* ------------------------------------------------------------------ */
/*  ANSI Colors                                                        */
/* ------------------------------------------------------------------ */

const COLORS = {
  reset: '\x1b[0m',
  dim: '\x1b[2m',
  bold: '\x1b[1m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  magenta: '\x1b[35m',
  gray: '\x1b[90m',
  white: '\x1b[37m',
};

function c(color, text, useColor) {
  return useColor ? `${COLORS[color]}${text}${COLORS.reset}` : text;
}

/* ------------------------------------------------------------------ */
/*  Tree Building                                                      */
/* ------------------------------------------------------------------ */

function buildTreeFromNodes(nodes, rootPath) {
  const root = { name: rootPath || 'REPO', children: new Map(), meta: null };

  for (const node of nodes) {
    let relPath = node.path;
    if (rootPath) {
      if (!relPath.startsWith(rootPath)) continue;
      relPath = relPath === rootPath ? '' : relPath.slice(rootPath.length + 1);
      if (!relPath) {
        root.meta = node;
        continue;
      }
    }

    const parts = relPath.split('/');
    let current = root;
    for (const part of parts) {
      if (!current.children.has(part)) {
        current.children.set(part, { name: part, children: new Map(), meta: null });
      }
      current = current.children.get(part);
    }
    current.meta = node;
  }

  return root;
}

/* ------------------------------------------------------------------ */
/*  Tree Rendering with Colors                                         */
/* ------------------------------------------------------------------ */

function renderColorTree(root, maxDepth, useColor, dirsOnly = false) {
  const lines = [];

  // Root line
  const rootDesc = root.meta?.description || '';
  const rootName = c('bold', `📦 ${root.name}/`, useColor);
  if (rootDesc) {
    lines.push(`${rootName}  ${c('gray', `# ${rootDesc}`, useColor)}`);
  } else {
    lines.push(rootName);
  }

  // Stats
  let dirCount = 0;
  let fileCount = 0;
  let describedCount = 0;

  function countAll(node) {
    for (const [, child] of node.children) {
      const isDir = child.meta?.type === 'directory' || child.children.size > 0;
      if (isDir) dirCount++;
      else fileCount++;
      if (child.meta?.description) describedCount++;
      countAll(child);
    }
  }
  countAll(root);

  function renderChildren(node, prefix, currentDepth) {
    if (currentDepth >= maxDepth) return;

    let entries = [...node.children.entries()].sort(([aName, aNode], [bName, bNode]) => {
      const aIsDir = aNode.meta?.type === 'directory' || aNode.children.size > 0;
      const bIsDir = bNode.meta?.type === 'directory' || bNode.children.size > 0;
      if (aIsDir !== bIsDir) return aIsDir ? -1 : 1;
      return aName.localeCompare(bName);
    });

    // --dirs-only: filter out file entries
    if (dirsOnly) {
      entries = entries.filter(([, child]) => {
        return child.meta?.type === 'directory' || child.children.size > 0;
      });
    }

    entries.forEach(([name, child], index) => {
      const isLast = index === entries.length - 1;
      const connector = isLast ? '└── ' : '├── ';
      const isDir = child.meta?.type === 'directory' || child.children.size > 0;

      // Icon and name
      let icon, nameColor;
      if (isDir) {
        icon = '📁';
        nameColor = 'blue';
      } else {
        // Pick icon by extension
        const ext = name.split('.').pop();
        if (['rs'].includes(ext)) { icon = '🦀'; nameColor = 'yellow'; }
        else if (['py', 'pyc'].includes(ext)) { icon = '🐍'; nameColor = 'green'; }
        else if (['mjs', 'js', 'ts'].includes(ext)) { icon = '⚡'; nameColor = 'cyan'; }
        else if (['md'].includes(ext)) { icon = '📝'; nameColor = 'white'; }
        else if (['json', 'yaml', 'yml', 'toml'].includes(ext)) { icon = '⚙️'; nameColor = 'magenta'; }
        else if (['sh'].includes(ext)) { icon = '🔧'; nameColor = 'green'; }
        else if (['lock'].includes(ext)) { icon = '🔒'; nameColor = 'gray'; }
        else { icon = '📄'; nameColor = 'white'; }
      }

      const displayName = isDir ? `${name}/` : name;
      const coloredName = c(nameColor, displayName, useColor);

      // Description annotation
      const desc = child.meta?.description || '';
      const padTo = 50;
      const rawLen = prefix.length + connector.length + displayName.length + 2; // +2 for icon + space
      const padding = desc ? ' '.repeat(Math.max(1, padTo - rawLen)) : '';
      const comment = desc ? `${padding}${c('gray', `# ${desc}`, useColor)}` : '';

      // Collapsed indicator
      const hasHiddenChildren = isDir && child.children.size > 0 && currentDepth + 1 >= maxDepth;
      const collapsed = hasHiddenChildren ? c('dim', ` [${child.children.size}]`, useColor) : '';

      lines.push(`${c('dim', prefix + connector, useColor)}${icon} ${coloredName}${collapsed}${comment}`);

      if (isDir && currentDepth + 1 < maxDepth) {
        const childPrefix = prefix + (isLast ? '    ' : '│   ');
        renderChildren(child, childPrefix, currentDepth + 1);
      }
    });
  }

  renderChildren(root, '', 0);

  // Summary — adapt wording for dirs-only mode
  if (dirsOnly) {
    lines.push('');
    lines.push(c('dim', `─── ${dirCount} 模块 (仅目录视图) ───`, useColor));
    return lines.join('\n');
  }

  // Summary
  const total = dirCount + fileCount;
  lines.push('');
  lines.push(c('dim', `─── ${dirCount} 目录, ${fileCount} 文件, ${describedCount}/${total} 已描述 ───`, useColor));

  return lines.join('\n');
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

async function main() {
  const flags = parseFlags(process.argv.slice(2));
  const depth = flags.depth ? parseInt(flags.depth, 10) : 3;
  const useColor = flags['no-color'] !== 'true';
  const subPath = flags.path ?? null;
  const dirsOnly = flags['dirs-only'] === 'true';

  const db = openMetadataDb(dbPath);

  try {
    const nodes = getAllNodes(db);

    if (nodes.length === 0) {
      console.error('❌ 数据库为空，请先运行 scan.mjs --update');
      process.exit(1);
    }

    const tree = buildTreeFromNodes(nodes, subPath);

    // If subpath specified, attach its metadata
    if (subPath) {
      const meta = getNode(db, subPath);
      if (meta) tree.meta = meta;
    }

    const output = renderColorTree(tree, depth, useColor, dirsOnly);
    console.log(output);
  } finally {
    db.close();
  }
}

main().catch((err) => {
  console.error(`❌ ${err.message}`);
  process.exitCode = 1;
});
