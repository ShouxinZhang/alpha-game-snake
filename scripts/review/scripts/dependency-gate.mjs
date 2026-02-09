#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

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

function globToRegex(glob) {
  const escaped = glob
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/\*\*/g, '::DOUBLE_STAR::')
    .replace(/\*/g, '[^/]*')
    .replace(/::DOUBLE_STAR::/g, '.*');
  return new RegExp(`^${escaped}$`);
}

function runMadge(args, cwd) {
  const result = spawnSync('npx', args, {
    cwd,
    encoding: 'utf8'
  });

  return {
    code: result.status ?? 1,
    stdout: result.stdout || '',
    stderr: result.stderr || ''
  };
}

function getSegmentName(modulePath, rootPrefix) {
  if (!modulePath.startsWith(`${rootPrefix}/`)) return null;
  const parts = modulePath.split('/');
  return parts.length > 1 ? parts[1] : null;
}

const args = parseArgs(process.argv.slice(2));
const projectRoot = path.resolve(args['project-root'] || process.cwd());
const policyPath = path.resolve(args.policy || path.join(projectRoot, 'scripts/review/config/policy.json'));
const outputPath = path.resolve(args.output || path.join(projectRoot, 'scripts/review/artifacts/dependency-report.json'));

const policy = JSON.parse(fs.readFileSync(policyPath, 'utf8'));
const madgeCfg = policy.madge || {};
const madgeCwd = path.resolve(projectRoot, madgeCfg.cwd || '.');
const madgeSource = madgeCfg.source || 'src';
const madgeTsConfig = madgeCfg.tsconfig || 'tsconfig.json';
const madgeExts = madgeCfg.extensions || 'ts,tsx';

const graphRun = runMadge(['madge', '--ts-config', madgeTsConfig, '--extensions', madgeExts, '--json', madgeSource], madgeCwd);
const cycleRun = runMadge(['madge', '--ts-config', madgeTsConfig, '--extensions', madgeExts, '--circular', '--json', madgeSource], madgeCwd);

let graph = {};
let cycles = [];
let parseErrors = [];

if (graphRun.code === 0) {
  try {
    graph = JSON.parse(graphRun.stdout || '{}');
  } catch (error) {
    parseErrors.push(`依赖图 JSON 解析失败: ${String(error)}`);
  }
} else {
  parseErrors.push(`madge 依赖图生成失败: ${graphRun.stderr.trim() || graphRun.stdout.trim()}`);
}

if (cycleRun.code === 0) {
  try {
    cycles = JSON.parse(cycleRun.stdout || '[]');
  } catch (error) {
    parseErrors.push(`循环依赖 JSON 解析失败: ${String(error)}`);
  }
} else {
  parseErrors.push(`madge 循环依赖检查失败: ${cycleRun.stderr.trim() || cycleRun.stdout.trim()}`);
}

const ruleMatchers = (policy.layerRules || []).map((rule) => ({
  ...rule,
  fromRegex: globToRegex(rule.from),
  toRegex: globToRegex(rule.to)
}));

const forbiddenEdges = [];
for (const [from, dependencies] of Object.entries(graph)) {
  for (const to of dependencies) {
    for (const rule of ruleMatchers) {
      if (rule.fromRegex.test(from) && rule.toRegex.test(to)) {
        forbiddenEdges.push({
          type: 'layer-rule',
          rule: rule.name,
          reason: rule.reason,
          from,
          to
        });
      }
    }

    const crossSegmentPolicy = policy.crossSegmentRule;
    if (crossSegmentPolicy?.enabled) {
      const rootPrefix = crossSegmentPolicy.rootPrefix || 'features';
      const fromSegment = getSegmentName(from, rootPrefix);
      const toSegment = getSegmentName(to, rootPrefix);
      if (fromSegment && toSegment && fromSegment !== toSegment) {
        const allowed = crossSegmentPolicy.allowIndexImportOnly
          ? to === `${rootPrefix}/${toSegment}/index.ts`
          : false;

        if (!allowed) {
          forbiddenEdges.push({
            type: 'cross-segment',
            rule: 'cross-segment-only-index',
            reason: crossSegmentPolicy.reason,
            from,
            to
          });
        }
      }
    }
  }
}

const uniqueForbiddenEdges = [];
const seen = new Set();
for (const edge of forbiddenEdges) {
  const key = `${edge.type}|${edge.rule}|${edge.from}|${edge.to}`;
  if (seen.has(key)) continue;
  seen.add(key);
  uniqueForbiddenEdges.push(edge);
}

const thresholds = policy.thresholds || {};
const cycleCount = cycles.length;
const forbiddenCount = uniqueForbiddenEdges.length;
const passCycles = cycleCount <= (thresholds.maxCircularDependencies ?? 0);
const passForbidden = forbiddenCount <= (thresholds.maxForbiddenDependencies ?? 0);
const passParser = parseErrors.length === 0;
const passed = passCycles && passForbidden && passParser;

const report = {
  generatedAt: new Date().toISOString(),
  policyPath: path.relative(projectRoot, policyPath),
  summary: {
    modules: Object.keys(graph).length,
    cycleCount,
    forbiddenCount,
    parseErrorCount: parseErrors.length,
    passed
  },
  thresholds: {
    maxCircularDependencies: thresholds.maxCircularDependencies ?? 0,
    maxForbiddenDependencies: thresholds.maxForbiddenDependencies ?? 0
  },
  cycles,
  forbiddenEdges: uniqueForbiddenEdges,
  parseErrors
};

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
console.log(outputPath);
process.exit(passed ? 0 : 1);
