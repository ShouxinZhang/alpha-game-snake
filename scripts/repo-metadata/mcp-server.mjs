#!/usr/bin/env node
/**
 * repo-metadata MCP Server (SQLite backend)
 *
 * 提供仓库元数据 CRUD、扫描、生成架构文档等 MCP Tools，
 * 供 LLM 直接调用，无需拼终端命令。
 *
 * 传输方式: stdio（VS Code Copilot 标准集成）
 */
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import {
  buildTree,
  depthOf,
  getTrackedPaths,
  globToRegex,
  openMetadataDb,
  getIgnoreMatchers,
  getGenerateMdDepth,
  getNode,
  upsertNode,
  deleteNodeByPath,
  listNodes,
  getAllNodes,
  getAllPaths,
  renderTree,
  updateStructureMdSync,
  exportToJson,
  shouldIgnore,
} from './lib/shared.mjs';

/* ------------------------------------------------------------------ */
/*  路径常量                                                           */
/* ------------------------------------------------------------------ */

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '../../');
const dbPath = path.join(repoRoot, 'docs', 'architecture', 'repo-metadata.db');
const structureMdPath = path.join(repoRoot, 'docs', 'architecture', 'repository-structure.md');

/* ------------------------------------------------------------------ */
/*  MCP Server 定义                                                    */
/* ------------------------------------------------------------------ */

const server = new McpServer({
  name: 'repo-metadata',
  version: '2.0.0',
});

// ─── Tool 1: scan ─────────────────────────────────────────

server.tool(
  'repo_metadata_scan',
  '扫描仓库目录结构，对比 SQLite 数据库，报告新增/删除/未描述的条目。可选自动更新。',
  {
    update: z.boolean().optional().default(false).describe('是否自动更新数据库'),
    maxDepth: z.number().optional().describe('最大扫描深度（默认: 无限制）'),
  },
  async ({ update, maxDepth }) => {
    const db = openMetadataDb(dbPath);
    try {
      const { fileSet, dirSet } = getTrackedPaths(repoRoot);
      const ignoreMatchers = getIgnoreMatchers(db);

      const diskPaths = new Map();
      for (const d of dirSet) {
        if (!shouldIgnore(d, ignoreMatchers)) diskPaths.set(d, 'directory');
      }
      for (const f of fileSet) {
        if (!shouldIgnore(f, ignoreMatchers)) diskPaths.set(f, 'file');
      }

      const filteredPaths = maxDepth
        ? new Map([...diskPaths].filter(([p]) => depthOf(p) <= maxDepth))
        : diskPaths;

      const existingPaths = getAllPaths(db);

      const added = [];
      const undescribed = [];

      for (const [p, type] of filteredPaths) {
        if (!existingPaths.has(p)) {
          added.push({ path: p, type });
        } else {
          const node = getNode(db, p);
          if (node && !node.description) {
            undescribed.push(p);
          }
        }
      }

      const removed = [];
      for (const p of existingPaths) {
        if (!filteredPaths.has(p)) removed.push(p);
      }

      if (update) {
        const upsertBatch = db.transaction(() => {
          for (const { path: p, type } of added) {
            upsertNode(db, p, { type, updatedBy: 'scan' });
          }
          for (const p of removed) {
            deleteNodeByPath(db, p);
          }
        });
        upsertBatch();
      }

      const lines = [];
      lines.push(`扫描完成: ${filteredPaths.size} 个路径`);
      if (added.length > 0) {
        lines.push(`\n🆕 新增 (${added.length}):`);
        for (const { path: p, type } of added.sort((a, b) => a.path.localeCompare(b.path))) {
          lines.push(`  + ${p}  (${type})`);
        }
      }
      if (removed.length > 0) {
        lines.push(`\n🗑️ 已删除 (${removed.length}):`);
        for (const p of removed.sort()) lines.push(`  - ${p}`);
      }
      if (undescribed.length > 0) {
        lines.push(`\n⚠️ 未描述 (${undescribed.length}):`);
        for (const p of undescribed.sort()) lines.push(`  ? ${p}`);
      }
      if (added.length === 0 && removed.length === 0 && undescribed.length === 0) {
        lines.push('\n✅ 元数据与文件系统完全同步，所有条目已描述。');
      }
      if (update) {
        lines.push(`\n✅ 已更新数据库: ${added.length} added, ${removed.length} removed`);
      }

      return { content: [{ type: 'text', text: lines.join('\n') }] };
    } finally {
      db.close();
    }
  },
);

// ─── Tool 2: get ──────────────────────────────────────────

server.tool(
  'repo_metadata_get',
  '获取指定路径的元数据详情（描述、标签、类型等）。',
  {
    path: z.string().describe('相对路径，如 "src/components"'),
  },
  async ({ path: nodePath }) => {
    const db = openMetadataDb(dbPath);
    try {
      const node = getNode(db, nodePath);
      if (!node) {
        return { content: [{ type: 'text', text: `❌ 路径不存在: ${nodePath}` }] };
      }
      return {
        content: [{ type: 'text', text: JSON.stringify({ path: nodePath, ...node }, null, 2) }],
      };
    } finally {
      db.close();
    }
  },
);

// ─── Tool 3: set ──────────────────────────────────────────

server.tool(
  'repo_metadata_set',
  '设置/更新指定路径的元数据（描述、标签等）。路径不存在时自动创建。',
  {
    path: z.string().describe('相对路径'),
    description: z.string().optional().describe('一句话描述'),
    detail: z.string().optional().describe('详细说明'),
    tags: z.array(z.string()).optional().describe('标签数组'),
    type: z.enum(['file', 'directory']).optional().describe('类型'),
  },
  async ({ path: nodePath, description, detail, tags, type }) => {
    const db = openMetadataDb(dbPath);
    try {
      upsertNode(db, nodePath, { description, detail, tags, type, updatedBy: 'llm' });
      return { content: [{ type: 'text', text: `✅ 已更新: ${nodePath}` }] };
    } finally {
      db.close();
    }
  },
);

// ─── Tool 4: batch_set ────────────────────────────────────

server.tool(
  'repo_metadata_batch_set',
  '批量设置多条路径的描述信息。适合 LLM 一次性补写多个新增条目。',
  {
    items: z
      .array(
        z.object({
          path: z.string().describe('相对路径'),
          description: z.string().optional().describe('一句话描述'),
          detail: z.string().optional().describe('详细说明'),
          tags: z.array(z.string()).optional().describe('标签数组'),
        }),
      )
      .describe('要更新的条目数组'),
  },
  async ({ items }) => {
    const db = openMetadataDb(dbPath);
    try {
      let updated = 0;
      let skipped = 0;

      const batch = db.transaction(() => {
        for (const item of items) {
          const existing = getNode(db, item.path);
          if (!existing) {
            skipped++;
            continue;
          }
          upsertNode(db, item.path, {
            description: item.description,
            detail: item.detail,
            tags: item.tags,
            updatedBy: 'llm',
          });
          updated++;
        }
      });
      batch();

      return {
        content: [
          { type: 'text', text: `✅ 批量更新完成: ${updated}/${items.length} 条 (跳过 ${skipped})` },
        ],
      };
    } finally {
      db.close();
    }
  },
);

// ─── Tool 5: list ─────────────────────────────────────────

server.tool(
  'repo_metadata_list',
  '列出仓库元数据条目。支持按类型、标签、深度、是否未描述过滤。',
  {
    type: z.enum(['file', 'directory']).optional().describe('过滤类型'),
    tag: z.string().optional().describe('过滤标签'),
    maxDepth: z.number().optional().describe('最大深度'),
    undescribedOnly: z.boolean().optional().default(false).describe('只显示未描述的条目'),
  },
  async ({ type, tag, maxDepth, undescribedOnly }) => {
    const db = openMetadataDb(dbPath);
    try {
      const entries = listNodes(db, { type, tag, maxDepth, undescribedOnly });

      if (entries.length === 0) {
        return { content: [{ type: 'text', text: '没有匹配的条目。' }] };
      }

      const lines = entries.map((node) => {
        const icon = node.type === 'directory' ? '📁' : '📄';
        const desc = node.description || '(未描述)';
        return `${icon} ${node.path} — ${desc}`;
      });
      lines.push(`\n共 ${entries.length} 条`);

      return { content: [{ type: 'text', text: lines.join('\n') }] };
    } finally {
      db.close();
    }
  },
);

// ─── Tool 6: delete ───────────────────────────────────────

server.tool(
  'repo_metadata_delete',
  '删除指定路径的元数据条目（级联删除子路径）。',
  {
    path: z.string().describe('要删除的相对路径'),
  },
  async ({ path: nodePath }) => {
    const db = openMetadataDb(dbPath);
    try {
      const existing = getNode(db, nodePath);
      if (!existing) {
        return { content: [{ type: 'text', text: `❌ 路径不存在: ${nodePath}` }] };
      }

      const { deleted, cascaded } = deleteNodeByPath(db, nodePath);
      return {
        content: [
          {
            type: 'text',
            text: `✅ 已删除: ${nodePath}${cascaded > 0 ? ` (+ ${cascaded} 个子路径)` : ''}`,
          },
        ],
      };
    } finally {
      db.close();
    }
  },
);

// ─── Tool 7: generate_md ─────────────────────────────────

server.tool(
  'repo_metadata_generate_md',
  '从 SQLite 数据库生成/更新 repository-structure.md 中的目录树。',
  {
    depth: z.number().optional().describe('目录树展开深度（默认: config.generateMdDepth 或 2）'),
  },
  async ({ depth }) => {
    const db = openMetadataDb(dbPath);
    try {
      const treeDepth = depth ?? getGenerateMdDepth(db);
      const nodes = getAllNodes(db);

      if (nodes.length === 0) {
        return { content: [{ type: 'text', text: '❌ 数据库中没有节点数据。' }] };
      }

      const tree = buildTree(nodes);
      const treeContent = renderTree(tree, treeDepth);
      updateStructureMdSync(structureMdPath, treeContent);

      return {
        content: [
          {
            type: 'text',
            text: `✅ 已更新 repository-structure.md（${nodes.length} 个节点，展开 ${treeDepth} 层）`,
          },
        ],
      };
    } finally {
      db.close();
    }
  },
);

// ─── Tool 8: export_json ─────────────────────────────────

server.tool(
  'repo_metadata_export_json',
  '将 SQLite 数据库导出为 JSON 格式（输出到 stdout 或文件）。',
  {
    outputPath: z
      .string()
      .optional()
      .describe('输出文件路径（相对于仓库根目录），不指定则输出到 stdout'),
  },
  async ({ outputPath }) => {
    const db = openMetadataDb(dbPath);
    try {
      const json = exportToJson(db);
      const jsonStr = JSON.stringify(json, null, 2);

      if (outputPath) {
        const fullPath = path.resolve(repoRoot, outputPath);
        fs.mkdirSync(path.dirname(fullPath), { recursive: true });
        fs.writeFileSync(fullPath, `${jsonStr}\n`, 'utf8');
        return {
          content: [{ type: 'text', text: `✅ 已导出到 ${outputPath}（${Object.keys(json.nodes).length} 条）` }],
        };
      }

      return { content: [{ type: 'text', text: jsonStr }] };
    } finally {
      db.close();
    }
  },
);

// ─── Tool 9: tree ────────────────────────────────────────

server.tool(
  'repo_metadata_tree',
  '以 ASCII 树形结构可视化仓库目录，带描述注释和文件类型图标。',
  {
    depth: z.number().optional().default(3).describe('展开深度（默认: 3）'),
    path: z.string().optional().describe('只显示指定子树（如 "crates/snake-core"）'),
  },
  async ({ depth, path: subPath }) => {
    const db = openMetadataDb(dbPath);
    try {
      const nodes = getAllNodes(db);
      if (nodes.length === 0) {
        return { content: [{ type: 'text', text: '❌ 数据库为空。' }] };
      }

      // Build tree (optionally filtered to subpath)
      const root = { name: subPath || 'REPO', children: new Map(), meta: null };
      for (const node of nodes) {
        let relPath = node.path;
        if (subPath) {
          if (!relPath.startsWith(subPath)) continue;
          relPath = relPath === subPath ? '' : relPath.slice(subPath.length + 1);
          if (!relPath) { root.meta = node; continue; }
        }
        const parts = relPath.split('/');
        let cur = root;
        for (const part of parts) {
          if (!cur.children.has(part)) {
            cur.children.set(part, { name: part, children: new Map(), meta: null });
          }
          cur = cur.children.get(part);
        }
        cur.meta = node;
      }

      const treeContent = renderTree(root, depth);
      return { content: [{ type: 'text', text: treeContent }] };
    } finally {
      db.close();
    }
  },
);

/* ------------------------------------------------------------------ */
/*  启动 Server                                                        */
/* ------------------------------------------------------------------ */

const transport = new StdioServerTransport();
await server.connect(transport);
