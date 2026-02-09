import { execSync } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';

export const MARKER_START = '<!-- REPO-TREE-START -->';
export const MARKER_END = '<!-- REPO-TREE-END -->';

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

export async function loadMetadata(metadataPath) {
  try {
    const content = await fs.readFile(metadataPath, 'utf8');
    return JSON.parse(content);
  } catch {
    return {
      version: 1,
      config: {
        scanIgnore: ['docs/dev_logs/**', 'docs/private_context/**'],
        generateMdDepth: 2,
      },
      updatedAt: new Date().toISOString(),
      nodes: {},
    };
  }
}

export async function saveMetadata(metadataPath, metadata) {
  metadata.updatedAt = new Date().toISOString();
  const sorted = Object.keys(metadata.nodes).sort();
  const orderedNodes = {};
  for (const key of sorted) {
    orderedNodes[key] = metadata.nodes[key];
  }
  metadata.nodes = orderedNodes;

  await fs.mkdir(path.dirname(metadataPath), { recursive: true });
  await fs.writeFile(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`, 'utf8');
}

export function buildTree(nodes) {
  const root = { name: 'REPO', children: new Map(), meta: null };
  for (const [nodePath, meta] of Object.entries(nodes)) {
    const parts = nodePath.split('/');
    let current = root;

    for (const part of parts) {
      if (!current.children.has(part)) {
        current.children.set(part, { name: part, children: new Map(), meta: null });
      }
      current = current.children.get(part);
    }

    current.meta = meta;
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

export async function updateStructureMd(structureMdPath, treeContent) {
  let markdown;
  try {
    markdown = await fs.readFile(structureMdPath, 'utf8');
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

  await fs.writeFile(structureMdPath, markdown, 'utf8');
}
