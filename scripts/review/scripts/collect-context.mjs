#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith('--')) {
      args[key] = true;
      continue;
    }
    args[key] = next;
    i += 1;
  }
  return args;
}

function runGit(command, cwd) {
  try {
    return execSync(command, { cwd, stdio: ['ignore', 'pipe', 'pipe'], encoding: 'utf8' }).trim();
  } catch {
    return '';
  }
}

function parseStatusLines(output) {
  const files = [];
  for (const line of output.split('\n')) {
    if (!line.trim()) continue;
    const trimmed = line.trim();
    if (trimmed.startsWith('?? ')) {
      files.push(trimmed.slice(3).trim());
      continue;
    }
    const statusCode = line.slice(0, 2);
    let filePath = line.slice(3).trim();
    if (filePath.includes(' -> ')) {
      const parts = filePath.split(' -> ');
      filePath = parts[parts.length - 1].trim();
    }
    files.push(filePath);
    if (statusCode.includes('D')) {
      files.push(`${filePath} (deleted)`);
    }
  }
  return [...new Set(files)].sort();
}

function collectChangedFiles(cwd, base, head) {
  if (base && head) {
    const diffOut = runGit(`git diff --name-only --diff-filter=ACMRTUXB ${base}..${head}`, cwd);
    return diffOut ? diffOut.split('\n').map((item) => item.trim()).filter(Boolean).sort() : [];
  }
  const statusOut = runGit('git status --porcelain', cwd);
  return parseStatusLines(statusOut);
}

const args = parseArgs(process.argv.slice(2));
const projectRoot = path.resolve(args['project-root'] || process.cwd());
const outputPath = path.resolve(args.output || path.join(projectRoot, 'scripts/review/artifacts/context.json'));
const base = args.base;
const head = args.head;

const payload = {
  generatedAt: new Date().toISOString(),
  mode: base && head ? 'range' : 'workspace',
  refs: {
    base: base || null,
    head: head || null,
    branch: runGit('git rev-parse --abbrev-ref HEAD', projectRoot) || null,
    commit: runGit('git rev-parse --short HEAD', projectRoot) || null
  },
  changedFiles: collectChangedFiles(projectRoot, base, head),
  totalChangedFiles: 0
};

payload.totalChangedFiles = payload.changedFiles.length;
fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
console.log(outputPath);
