#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

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

function toPosix(p) {
  return p.split(path.sep).join('/');
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function readText(filePath) {
  return fs.readFileSync(filePath, 'utf8');
}

function isFile(filePath) {
  try {
    return fs.statSync(filePath).isFile();
  } catch {
    return false;
  }
}

function isDirectory(dirPath) {
  try {
    return fs.statSync(dirPath).isDirectory();
  } catch {
    return false;
  }
}

function collectFiles(rootDir, exts, excludeContains) {
  const result = [];
  const stack = [rootDir];

  while (stack.length > 0) {
    const current = stack.pop();
    const entries = fs.readdirSync(current, { withFileTypes: true });

    for (const entry of entries) {
      const abs = path.join(current, entry.name);
      const rel = toPosix(path.relative(rootDir, abs));

      if (excludeContains.some((token) => rel.includes(token))) {
        continue;
      }

      if (entry.isDirectory()) {
        stack.push(abs);
        continue;
      }

      if (entry.isFile() && exts.some((ext) => entry.name.endsWith(ext))) {
        result.push(abs);
      }
    }
  }

  return result;
}

function parseImports(code) {
  const specs = [];
  const importExportRegex = /(?:import|export)\s+(?:[^'"`]*?from\s+)?['"]([^'"`]+)['"]/g;
  const dynamicImportRegex = /import\(\s*['"]([^'"`]+)['"]\s*\)/g;

  let match = importExportRegex.exec(code);
  while (match) {
    specs.push(match[1]);
    match = importExportRegex.exec(code);
  }

  match = dynamicImportRegex.exec(code);
  while (match) {
    specs.push(match[1]);
    match = dynamicImportRegex.exec(code);
  }

  return specs;
}

function countExports(code) {
  const exportRegex = /^\s*export\s+/gm;
  const matches = code.match(exportRegex);
  return matches ? matches.length : 0;
}

function resolveInternalImport(spec, importerAbs, targetAbs, extensions) {
  if (!spec.startsWith('.')) {
    return null;
  }

  const base = path.resolve(path.dirname(importerAbs), spec);
  const candidates = [];

  candidates.push(base);
  for (const ext of extensions) {
    candidates.push(`${base}${ext}`);
  }
  for (const ext of extensions) {
    candidates.push(path.join(base, `index${ext}`));
  }

  for (const candidate of candidates) {
    if (!isFile(candidate)) continue;
    const normalized = path.resolve(candidate);
    if (!normalized.startsWith(targetAbs)) continue;
    return normalized;
  }

  return null;
}

function detectLayer(relPath, layerOrder) {
  const normalized = toPosix(relPath);
  const segments = normalized.split('/');
  const first = segments[0] || '';

  if (layerOrder.includes(first)) {
    return first;
  }

  return 'unassigned';
}

function countRelativeDepth(spec) {
  const segments = spec.split('/');
  return segments.filter((s) => s === '..').length;
}

function tarjanScc(graph) {
  let index = 0;
  const indices = new Map();
  const lowlink = new Map();
  const onStack = new Set();
  const stack = [];
  const sccs = [];

  function strongConnect(node) {
    indices.set(node, index);
    lowlink.set(node, index);
    index += 1;
    stack.push(node);
    onStack.add(node);

    const edges = graph.get(node) || [];
    for (const neighbor of edges) {
      if (!indices.has(neighbor)) {
        strongConnect(neighbor);
        lowlink.set(node, Math.min(lowlink.get(node), lowlink.get(neighbor)));
      } else if (onStack.has(neighbor)) {
        lowlink.set(node, Math.min(lowlink.get(node), indices.get(neighbor)));
      }
    }

    if (lowlink.get(node) === indices.get(node)) {
      const component = [];
      let w = null;
      while (w !== node) {
        w = stack.pop();
        onStack.delete(w);
        component.push(w);
      }
      sccs.push(component);
    }
  }

  for (const node of graph.keys()) {
    if (!indices.has(node)) {
      strongConnect(node);
    }
  }

  return sccs;
}

function buildFinding({ severity, code, message, files, suggestion, evidence }) {
  return {
    severity,
    code,
    message,
    files,
    suggestion,
    evidence
  };
}

const args = parseArgs(process.argv.slice(2));
const cwd = path.resolve(args['project-root'] || process.cwd());

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const skillRoot = path.resolve(scriptDir, '..');
const defaultPolicy = path.join(skillRoot, 'references', 'modularity-policy.template.json');

const targetRel = args.target;
if (!targetRel) {
  console.error('Missing required argument: --target <path>');
  process.exit(2);
}

const targetAbs = path.resolve(cwd, targetRel);
if (!isDirectory(targetAbs)) {
  console.error(`Target directory not found: ${targetAbs}`);
  process.exit(2);
}

const policyPath = path.resolve(cwd, args.policy || defaultPolicy);
if (!isFile(policyPath)) {
  console.error(`Policy file not found: ${policyPath}`);
  process.exit(2);
}

const policy = readJson(policyPath);
const extensions = policy.extensions || ['.ts', '.tsx'];
const excludeContains = policy.excludeContains || ['__tests__', '.test.', '.spec.', '.d.ts'];
const layerOrder = policy.layerOrder || ['ui', 'hooks', 'model', 'api', 'infra'];
const entryFiles = policy.entryFiles || ['index.ts', 'index.tsx'];
const thresholds = {
  maxFileLines: 400,
  maxFanOut: 12,
  maxRelativeImportDepth: 3,
  maxPublicExports: 25,
  maxOrphanFiles: 0,
  maxUnknownLayerFiles: 0,
  maxReverseLayerImports: 0,
  maxCircularDeps: 0,
  ...(policy.thresholds || {})
};

const filesAbs = collectFiles(targetAbs, extensions, excludeContains).sort();
const fileSet = new Set(filesAbs);
const relByAbs = new Map(filesAbs.map((abs) => [abs, toPosix(path.relative(targetAbs, abs))]));
const graph = new Map();
const inboundCount = new Map();
const fileStats = new Map();
const findings = [];

for (const abs of filesAbs) {
  const rel = relByAbs.get(abs);
  graph.set(rel, []);
  inboundCount.set(rel, 0);
}

let totalImportCount = 0;
let externalImportCount = 0;
const reverseLayerEdges = [];
const deepRelativeImports = [];

for (const abs of filesAbs) {
  const rel = relByAbs.get(abs);
  const code = readText(abs);
  const imports = parseImports(code);
  const uniqueImports = [...new Set(imports)];
  const lineCount = code.split('\n').length;
  const exportCount = countExports(code);
  const currentLayer = detectLayer(rel, layerOrder);

  let internalFanOut = 0;
  let externalFanOut = 0;

  for (const spec of uniqueImports) {
    totalImportCount += 1;

    const depth = countRelativeDepth(spec);
    if (depth > thresholds.maxRelativeImportDepth) {
      deepRelativeImports.push({ rel, spec, depth });
    }

    const resolved = resolveInternalImport(spec, abs, targetAbs, extensions);
    if (resolved && fileSet.has(resolved)) {
      const targetRelPath = relByAbs.get(resolved);
      graph.get(rel).push(targetRelPath);
      inboundCount.set(targetRelPath, (inboundCount.get(targetRelPath) || 0) + 1);
      internalFanOut += 1;

      const targetLayer = detectLayer(targetRelPath, layerOrder);
      const fromIdx = layerOrder.indexOf(currentLayer);
      const toIdx = layerOrder.indexOf(targetLayer);
      if (fromIdx !== -1 && toIdx !== -1 && fromIdx > toIdx) {
        reverseLayerEdges.push({ from: rel, to: targetRelPath, fromLayer: currentLayer, toLayer: targetLayer });
      }
    } else {
      externalFanOut += 1;
      externalImportCount += 1;
    }
  }

  fileStats.set(rel, {
    lineCount,
    importCount: uniqueImports.length,
    internalFanOut,
    externalFanOut,
    layer: currentLayer,
    exportCount
  });

  if (lineCount > thresholds.maxFileLines) {
    findings.push(
      buildFinding({
        severity: 'REFINE',
        code: 'FILE_TOO_LARGE',
        message: `${rel} 行数 ${lineCount} 超过阈值 ${thresholds.maxFileLines}`,
        files: [rel],
        suggestion: '拆分为更小职责单元（先抽纯逻辑，再拆状态与视图）。',
        evidence: { lineCount, threshold: thresholds.maxFileLines }
      })
    );
  }

  if (uniqueImports.length > thresholds.maxFanOut) {
    findings.push(
      buildFinding({
        severity: 'REFINE',
        code: 'HIGH_FAN_OUT',
        message: `${rel} 依赖数 ${uniqueImports.length} 超过阈值 ${thresholds.maxFanOut}`,
        files: [rel],
        suggestion: '收敛外部依赖到 facade/adapter，避免单文件扇出过大。',
        evidence: { importCount: uniqueImports.length, threshold: thresholds.maxFanOut }
      })
    );
  }
}

for (const edge of reverseLayerEdges) {
  findings.push(
    buildFinding({
      severity: 'BLOCK',
      code: 'REVERSE_LAYER_IMPORT',
      message: `反向依赖: ${edge.from} (${edge.fromLayer}) -> ${edge.to} (${edge.toLayer})`,
      files: [edge.from, edge.to],
      suggestion: '调整调用方向，或下沉共享逻辑到更低层模块。',
      evidence: edge
    })
  );
}

for (const item of deepRelativeImports) {
  findings.push(
    buildFinding({
      severity: 'REFINE',
      code: 'DEEP_RELATIVE_IMPORT',
      message: `${item.rel} 使用深层相对路径 ${item.spec}（深度 ${item.depth}）`,
      files: [item.rel],
      suggestion: '使用模块出口文件（index/facade）替代深层相对导入。',
      evidence: item
    })
  );
}

const sccs = tarjanScc(graph);
const cycles = [];
for (const component of sccs) {
  if (component.length > 1) {
    cycles.push(component);
    continue;
  }
  const [node] = component;
  if ((graph.get(node) || []).includes(node)) {
    cycles.push(component);
  }
}

for (const cycle of cycles) {
  findings.push(
    buildFinding({
      severity: 'BLOCK',
      code: 'CIRCULAR_DEPENDENCY',
      message: `发现循环依赖: ${cycle.join(' -> ')}`,
      files: cycle,
      suggestion: '提取共享接口到独立层，打断环路。',
      evidence: { cycle }
    })
  );
}

const entrySet = new Set();
for (const rel of fileStats.keys()) {
  if (entryFiles.some((name) => rel === name || rel.endsWith(`/${name}`))) {
    entrySet.add(rel);
  }
}

for (const rel of entrySet) {
  const stats = fileStats.get(rel);
  if (!stats) continue;
  if (stats.exportCount > thresholds.maxPublicExports) {
    findings.push(
      buildFinding({
        severity: 'REFINE',
        code: 'PUBLIC_API_TOO_WIDE',
        message: `${rel} 导出数 ${stats.exportCount} 超过阈值 ${thresholds.maxPublicExports}`,
        files: [rel],
        suggestion: '收敛公共导出，仅保留稳定 API。',
        evidence: { exportCount: stats.exportCount, threshold: thresholds.maxPublicExports }
      })
    );
  }
}

const orphanFiles = [];
for (const rel of fileStats.keys()) {
  const inbound = inboundCount.get(rel) || 0;
  if (inbound === 0 && !entrySet.has(rel)) {
    orphanFiles.push(rel);
  }
}

if (orphanFiles.length > thresholds.maxOrphanFiles) {
  findings.push(
    buildFinding({
      severity: 'REFINE',
      code: 'ORPHAN_FILES',
      message: `孤立文件数 ${orphanFiles.length} 超过阈值 ${thresholds.maxOrphanFiles}`,
      files: orphanFiles.slice(0, 50),
      suggestion: '删除死代码或将入口纳入 policy.entryFiles。',
      evidence: { orphanCount: orphanFiles.length, threshold: thresholds.maxOrphanFiles }
    })
  );
}

const unknownLayerFiles = [];
for (const [rel, stats] of fileStats.entries()) {
  if (stats.layer === 'unassigned') {
    unknownLayerFiles.push(rel);
  }
}

if (unknownLayerFiles.length > thresholds.maxUnknownLayerFiles) {
  findings.push(
    buildFinding({
      severity: 'REFINE',
      code: 'UNKNOWN_LAYER_FILES',
      message: `未归层文件数 ${unknownLayerFiles.length} 超过阈值 ${thresholds.maxUnknownLayerFiles}`,
      files: unknownLayerFiles.slice(0, 50),
      suggestion: '将文件移动到已定义层目录，或扩展 layerOrder。',
      evidence: { unknownLayerFiles: unknownLayerFiles.length, threshold: thresholds.maxUnknownLayerFiles }
    })
  );
}

const metrics = {
  totalFiles: filesAbs.length,
  totalImports: totalImportCount,
  externalImports: externalImportCount,
  circularDeps: cycles.length,
  reverseLayerImports: reverseLayerEdges.length,
  deepRelativeImports: deepRelativeImports.length,
  orphanFiles: orphanFiles.length,
  unknownLayerFiles: unknownLayerFiles.length,
  oversizedFiles: [...fileStats.values()].filter((s) => s.lineCount > thresholds.maxFileLines).length,
  highFanOutFiles: [...fileStats.values()].filter((s) => s.importCount > thresholds.maxFanOut).length
};

const hardViolations =
  metrics.circularDeps > thresholds.maxCircularDeps ||
  metrics.reverseLayerImports > thresholds.maxReverseLayerImports;

const hasRefine = findings.some((item) => item.severity === 'REFINE');
const status = hardViolations ? 'BLOCK' : hasRefine ? 'REFINE' : 'PASS';

const summary = {
  status,
  blockCount: findings.filter((item) => item.severity === 'BLOCK').length,
  refineCount: findings.filter((item) => item.severity === 'REFINE').length,
  metrics
};

const report = {
  generatedAt: new Date().toISOString(),
  target: toPosix(path.relative(cwd, targetAbs)),
  policyPath: toPosix(path.relative(cwd, policyPath)),
  thresholds,
  summary,
  findings
};

const outputPath = path.resolve(
  cwd,
  args.output || path.join(skillRoot, 'artifacts', `${path.basename(targetAbs)}.modularity-report.json`)
);

ensureDir(path.dirname(outputPath));
fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');

console.log(`Status: ${status}`);
console.log(`Report: ${toPosix(path.relative(cwd, outputPath))}`);
console.log(`BLOCK: ${summary.blockCount}, REFINE: ${summary.refineCount}`);

if (status === 'PASS') {
  process.exit(0);
}
if (status === 'REFINE') {
  process.exit(1);
}
process.exit(2);
