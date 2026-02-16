import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import Database from 'better-sqlite3';

export const MARKER_START = '<!-- REPO-TREE-START -->';
export const MARKER_END = '<!-- REPO-TREE-END -->';

/* ------------------------------------------------------------------ */
/*  CLI helpers                                                        */
/* ------------------------------------------------------------------ */

export function parseFlags(args) {
  const flags = {};
  for (let i = 0; i < args.length; i += 1) {
    const token = args[i];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    const next = args[i + 1];
    if (!next || next.startsWith('--')) {
      flags[key] = 'true';
    } else {
      flags[key] = next;
      i += 1;
    }
  }
  return flags;
}

/* ------------------------------------------------------------------ */
/*  Git helpers                                                        */
/* ------------------------------------------------------------------ */

export function globToRegex(pattern) {
  const re = pattern
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/\*\*/g, '{{GLOBSTAR}}')
    .replace(/\*/g, '[^/]*')
    .replace(/\?/g, '[^/]')
    .replace(/\{\{GLOBSTAR\}\}/g, '.*');
  return new RegExp(`^${re}$`);
}

export function getTrackedPaths(repoRoot) {
  const output = execSync('git ls-files', { cwd: repoRoot, encoding: 'utf8' });
  const files = output.trim().split('\n').filter(Boolean);
  const fileSet = new Set(files);
  const dirSet = new Set();

  for (const file of files) {
    const parts = file.split('/');
    let current = '';
    for (let i = 0; i < parts.length - 1; i += 1) {
      current = current ? `${current}/${parts[i]}` : parts[i];
      dirSet.add(current);
    }
  }

  return { fileSet, dirSet };
}

export function shouldIgnore(filePath, matchers) {
  return matchers.some((matcher) => matcher.test(filePath));
}

export function depthOf(filePath) {
  return filePath.split('/').length;
}

/* ------------------------------------------------------------------ */
/*  SQLite — 初始化 & 连接                                             */
/* ------------------------------------------------------------------ */

const SCHEMA_SQL = `
create table if not exists config (
  key   text primary key,
  value text not null
);

create table if not exists nodes (
  path         text primary key,
  type         text not null check (type in ('directory', 'file')),
  description  text not null default '',
  detail       text not null default '',
  tags         text not null default '[]',       -- JSON array stored as text
  parent_path  text,
  sort_order   integer not null default 0,
  updated_by   text not null default 'scan',
  created_at   text not null default (datetime('now')),
  updated_at   text not null default (datetime('now'))
);

create index if not exists idx_nodes_parent
  on nodes(parent_path, sort_order, path);

-- 自动更新 updated_at
create trigger if not exists trg_nodes_updated_at
after update on nodes
for each row
begin
  update nodes set updated_at = datetime('now') where path = new.path;
end;
`;

const DEFAULT_CONFIG = {
  scanIgnore: JSON.stringify(['docs/dev_logs/**', 'docs/private_context/**']),
  generateMdDepth: '2',
};

/**
 * 打开 (或创建) 元数据 SQLite 数据库，返回 better-sqlite3 Database 实例。
 */
export function openMetadataDb(dbPath) {
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new Database(dbPath);
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');
  db.exec(SCHEMA_SQL);

  // 初始化默认 config（若表空）
  const hasConfig = db.prepare('select count(*) as cnt from config').get();
  if (hasConfig.cnt === 0) {
    const insert = db.prepare('insert or ignore into config (key, value) values (?, ?)');
    for (const [k, v] of Object.entries(DEFAULT_CONFIG)) {
      insert.run(k, v);
    }
  }

  return db;
}

/* ------------------------------------------------------------------ */
/*  SQLite — Config helpers                                            */
/* ------------------------------------------------------------------ */

export function getConfig(db, key) {
  const row = db.prepare('select value from config where key = ?').get(key);
  return row?.value ?? null;
}

export function setConfig(db, key, value) {
  db.prepare(
    'insert into config (key, value) values (?, ?) on conflict(key) do update set value = excluded.value',
  ).run(key, String(value));
}

export function getIgnoreMatchers(db) {
  const raw = getConfig(db, 'scanIgnore');
  const patterns = raw ? JSON.parse(raw) : [];
  return patterns.map(globToRegex);
}

export function getGenerateMdDepth(db) {
  const raw = getConfig(db, 'generateMdDepth');
  return raw ? parseInt(raw, 10) : 2;
}

/* ------------------------------------------------------------------ */
/*  SQLite — Node CRUD                                                 */
/* ------------------------------------------------------------------ */

export function getNode(db, nodePath) {
  const row = db.prepare('select * from nodes where path = ?').get(nodePath);
  if (!row) return null;
  return { ...row, tags: JSON.parse(row.tags) };
}

export function upsertNode(db, nodePath, fields) {
  const existing = getNode(db, nodePath);
  if (existing) {
    const desc = fields.description ?? existing.description;
    const detail = fields.detail ?? existing.detail;
    const tags = fields.tags ? JSON.stringify(fields.tags) : JSON.stringify(existing.tags);
    const type = fields.type ?? existing.type;
    const updatedBy = fields.updatedBy ?? 'llm';
    db.prepare(
      'update nodes set description=?, detail=?, tags=?, type=?, updated_by=? where path=?',
    ).run(desc, detail, tags, type, updatedBy, nodePath);
  } else {
    const parentPath = path.dirname(nodePath);
    db.prepare(`
      insert into nodes (path, type, description, detail, tags, parent_path, updated_by)
      values (?, ?, ?, ?, ?, ?, ?)
    `).run(
      nodePath,
      fields.type ?? 'directory',
      fields.description ?? '',
      fields.detail ?? '',
      JSON.stringify(fields.tags ?? []),
      parentPath === '.' ? null : parentPath,
      fields.updatedBy ?? 'scan',
    );
  }
}

export function deleteNodeByPath(db, nodePath) {
  const prefix = `${nodePath}/`;
  const cascaded = db.prepare("delete from nodes where path like ?").run(`${prefix}%`).changes;
  const deleted = db.prepare('delete from nodes where path = ?').run(nodePath).changes;
  return { deleted, cascaded };
}

export function listNodes(db, { type, tag, maxDepth, undescribedOnly } = {}) {
  let sql = 'select * from nodes where 1=1';
  const params = [];

  if (type) {
    sql += ' and type = ?';
    params.push(type);
  }
  if (maxDepth) {
    // depth = number of '/' separators + 1
    sql += ` and (length(path) - length(replace(path, '/', ''))) < ?`;
    params.push(maxDepth);
  }
  if (undescribedOnly) {
    sql += " and (description = '' or description is null)";
  }
  sql += ' order by path';

  let rows = db.prepare(sql).all(...params);

  if (tag) {
    rows = rows.filter((r) => {
      const tags = JSON.parse(r.tags);
      return tags.includes(tag);
    });
  }

  return rows.map((r) => ({ ...r, tags: JSON.parse(r.tags) }));
}

export function getAllNodes(db) {
  const rows = db.prepare('select * from nodes order by path').all();
  return rows.map((r) => ({ ...r, tags: JSON.parse(r.tags) }));
}

export function getAllPaths(db) {
  return new Set(db.prepare('select path from nodes').all().map((r) => r.path));
}

/* ------------------------------------------------------------------ */
/*  Tree rendering                                                     */
/* ------------------------------------------------------------------ */

export function buildTree(nodes) {
  const root = { name: 'REPO', children: new Map(), meta: null };
  for (const node of nodes) {
    const parts = node.path.split('/');
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

export function renderTree(root, maxDepth) {
  const lines = [`${root.name}/`];

  function renderChildren(node, prefix, currentDepth) {
    if (currentDepth >= maxDepth) return;

    const entries = [...node.children.entries()].sort(([aName, aNode], [bName, bNode]) => {
      const aIsDir = aNode.meta?.type === 'directory' || aNode.children.size > 0;
      const bIsDir = bNode.meta?.type === 'directory' || bNode.children.size > 0;
      if (aIsDir !== bIsDir) return aIsDir ? -1 : 1;
      return aName.localeCompare(bName);
    });

    entries.forEach(([name, child], index) => {
      const isLast = index === entries.length - 1;
      const connector = isLast ? '└── ' : '├── ';
      const isDir = child.meta?.type === 'directory' || child.children.size > 0;
      const displayName = isDir ? `${name}/` : name;
      const desc = child.meta?.description || '';
      const nameWidth = prefix.length + connector.length + displayName.length;
      const padTo = 45;
      const padding = desc ? ' '.repeat(Math.max(1, padTo - nameWidth)) : '';
      const comment = desc ? `${padding}# ${desc}` : '';

      lines.push(`${prefix}${connector}${displayName}${comment}`);

      if (isDir && currentDepth + 1 < maxDepth) {
        const childPrefix = prefix + (isLast ? '    ' : '│   ');
        renderChildren(child, childPrefix, currentDepth + 1);
      }
    });
  }

  renderChildren(root, '', 0);
  return lines.join('\n');
}

/* ------------------------------------------------------------------ */
/*  Markdown update                                                    */
/* ------------------------------------------------------------------ */

export function updateStructureMdSync(structureMdPath, treeContent) {
  let markdown;
  try {
    markdown = fs.readFileSync(structureMdPath, 'utf8');
  } catch {
    markdown = `# Repository Structure\n\n## 目录结构\n\n${MARKER_START}\n\`\`\`\n\`\`\`\n${MARKER_END}\n`;
  }

  const startIndex = markdown.indexOf(MARKER_START);
  const endIndex = markdown.indexOf(MARKER_END);
  const treeBlock = `${MARKER_START}\n\`\`\`\n${treeContent}\n\`\`\`\n${MARKER_END}`;

  if (startIndex !== -1 && endIndex !== -1) {
    markdown = markdown.slice(0, startIndex) + treeBlock + markdown.slice(endIndex + MARKER_END.length);
  } else {
    const sectionMatch = markdown.match(/## 目录结构\s*\n/);
    if (sectionMatch) {
      const sectionStart = sectionMatch.index + sectionMatch[0].length;
      const codeBlockMatch = markdown.slice(sectionStart).match(/```[\s\S]*?```/);
      if (codeBlockMatch) {
        const blockStart = sectionStart + codeBlockMatch.index;
        const blockEnd = blockStart + codeBlockMatch[0].length;
        markdown = markdown.slice(0, blockStart) + treeBlock + markdown.slice(blockEnd);
      } else {
        markdown = markdown.slice(0, sectionStart) + '\n' + treeBlock + '\n' + markdown.slice(sectionStart);
      }
    } else {
      markdown += `\n## 目录结构\n\n${treeBlock}\n`;
    }
  }

  fs.writeFileSync(structureMdPath, markdown, 'utf8');
}

/* ------------------------------------------------------------------ */
/*  JSON export helper                                                 */
/* ------------------------------------------------------------------ */

export function exportToJson(db) {
  const nodes = getAllNodes(db);
  const nodesObj = {};
  for (const n of nodes) {
    nodesObj[n.path] = {
      type: n.type,
      description: n.description,
      detail: n.detail,
      tags: n.tags,
      updatedBy: n.updated_by,
      updatedAt: n.updated_at,
    };
  }

  const scanIgnoreRaw = getConfig(db, 'scanIgnore');
  const generateMdDepthRaw = getConfig(db, 'generateMdDepth');

  return {
    version: 1,
    config: {
      scanIgnore: scanIgnoreRaw ? JSON.parse(scanIgnoreRaw) : [],
      generateMdDepth: generateMdDepthRaw ? parseInt(generateMdDepthRaw, 10) : 2,
    },
    updatedAt: new Date().toISOString(),
    nodes: nodesObj,
  };
}

/* ------------------------------------------------------------------ */
/*  JSON import helper (for migration)                                 */
/* ------------------------------------------------------------------ */

export function importFromJson(db, jsonData) {
  const nodes = Object.entries(jsonData.nodes ?? {});
  if (nodes.length === 0) return 0;

  // Sort by depth to ensure parents first
  nodes.sort(([a], [b]) => a.split('/').length - b.split('/').length || a.localeCompare(b));

  const insert = db.prepare(`
    insert or replace into nodes (path, type, description, detail, tags, parent_path, updated_by, updated_at)
    values (?, ?, ?, ?, ?, ?, ?, ?)
  `);

  const importAll = db.transaction(() => {
    for (const [nodePath, node] of nodes) {
      const parentPath = path.dirname(nodePath);
      insert.run(
        nodePath,
        node.type,
        node.description ?? '',
        node.detail ?? '',
        JSON.stringify(node.tags ?? []),
        parentPath === '.' ? null : parentPath,
        node.updatedBy ?? 'scan',
        node.updatedAt ?? new Date().toISOString(),
      );
    }
  });

  importAll();

  // Import config
  if (jsonData.config) {
    if (jsonData.config.scanIgnore) {
      setConfig(db, 'scanIgnore', JSON.stringify(jsonData.config.scanIgnore));
    }
    if (jsonData.config.generateMdDepth !== undefined) {
      setConfig(db, 'generateMdDepth', String(jsonData.config.generateMdDepth));
    }
  }

  return nodes.length;
}
