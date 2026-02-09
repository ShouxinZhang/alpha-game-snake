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

function runCommand(name, cmd, commandArgs, options = {}) {
  const startAt = Date.now();
  const run = spawnSync(cmd, commandArgs, {
    cwd: options.cwd,
    encoding: 'utf8'
  });

  const combinedOutput = `${run.stdout || ''}${run.stderr || ''}`.trim();
  if (options.logPath) {
    fs.writeFileSync(options.logPath, `${combinedOutput}\n`, 'utf8');
  }

  return {
    name,
    code: run.status ?? 1,
    durationMs: Date.now() - startAt,
    logPath: options.logPath || null,
    outputPreview: combinedOutput.split('\n').slice(-20)
  };
}

function readJsonSafe(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

const args = parseArgs(process.argv.slice(2));
const projectRoot = path.resolve(args['project-root'] || process.cwd());
const reviewRoot = path.join(projectRoot, 'scripts/review');
const scriptsDir = path.join(reviewRoot, 'scripts');
const policyPath = path.join(reviewRoot, 'config/policy.json');
const policy = readJsonSafe(policyPath) || {};
const artifactsRoot = path.join(reviewRoot, 'artifacts');
const runId = new Date().toISOString().replace(/[:.]/g, '-');
const runDir = path.join(artifactsRoot, runId);
const outputPath = path.resolve(args.output || path.join(runDir, 'review-result.json'));
const llmReportPath = args['llm-report'] ? path.resolve(args['llm-report']) : path.join(projectRoot, 'scripts/review/input/llm-review.json');
const allowHuman = args['allow-human'] === true || args['allow-human'] === 'true';

fs.mkdirSync(runDir, { recursive: true });

const contextFile = path.join(runDir, 'context.json');
const dependencyFile = path.join(runDir, 'dependency-report.json');
const llmValidationFile = path.join(runDir, 'llm-validation.json');
const buildLog = path.join(runDir, 'build-check.log');
const testLog = path.join(runDir, 'test.log');

function parseCommandSpec(spec, fallback) {
  if (Array.isArray(spec) && spec.length > 0) {
    return { cmd: spec[0], args: spec.slice(1) };
  }
  if (typeof spec === 'string' && spec.trim() !== '') {
    return { cmd: 'bash', args: ['-lc', spec] };
  }
  return fallback;
}

const buildCommand = parseCommandSpec(
  policy?.commands?.build,
  { cmd: 'bash', args: ['scripts/check_errors.sh'] }
);
const testCommand = parseCommandSpec(
  policy?.commands?.test,
  { cmd: 'npm', args: ['test'] }
);

runCommand('collect-context', 'node', [
  path.join(scriptsDir, 'collect-context.mjs'),
  '--project-root',
  projectRoot,
  '--output',
  contextFile,
  ...(args.base ? ['--base', args.base] : []),
  ...(args.head ? ['--head', args.head] : [])
]);

const buildStep = runCommand('build-check', buildCommand.cmd, buildCommand.args, {
  cwd: projectRoot,
  logPath: buildLog
});

const testStep = runCommand('unit-test', testCommand.cmd, testCommand.args, {
  cwd: projectRoot,
  logPath: testLog
});

const dependencyStep = runCommand('dependency-gate', 'node', [
  path.join(scriptsDir, 'dependency-gate.mjs'),
  '--project-root',
  projectRoot,
  '--policy',
  policyPath,
  '--output',
  dependencyFile
], {
  cwd: projectRoot,
  logPath: path.join(runDir, 'dependency-gate.log')
});

runCommand('llm-validation', 'node', [
  path.join(scriptsDir, 'validate-llm-report.mjs'),
  '--project-root',
  projectRoot,
  '--policy',
  policyPath,
  '--input',
  llmReportPath,
  '--output',
  llmValidationFile
], {
  cwd: projectRoot,
  logPath: path.join(runDir, 'llm-validation.log')
});

const dependencyReport = readJsonSafe(dependencyFile);
const llmValidation = readJsonSafe(llmValidationFile);

let finalStatus = 'pass';
const blockReasons = [];
const humanReasons = [];

if (buildStep.code !== 0) {
  finalStatus = 'block';
  blockReasons.push('构建质量门禁失败（check_errors.sh）');
}
if (testStep.code !== 0) {
  finalStatus = 'block';
  blockReasons.push(`自动化测试失败（${testCommand.cmd} ${testCommand.args.join(' ')}）`);
}
if (!dependencyReport || dependencyStep.code !== 0 || dependencyReport.summary?.passed !== true) {
  finalStatus = 'block';
  blockReasons.push('依赖关系门禁失败（循环依赖或禁止边超限）');
}

if (llmValidation) {
  if (llmValidation.status === 'block') {
    finalStatus = 'block';
    blockReasons.push('LLM 风险报告存在阻断项');
  } else if (llmValidation.status === 'human_required' && finalStatus !== 'block') {
    finalStatus = 'human';
    humanReasons.push(...(llmValidation.reasons || []));
  }
} else if (finalStatus !== 'block') {
  finalStatus = 'human';
  humanReasons.push('LLM 风险报告校验结果缺失');
}

const result = {
  generatedAt: new Date().toISOString(),
  runId,
  status: finalStatus.toUpperCase(),
  steps: {
    buildCheck: {
      passed: buildStep.code === 0,
      logPath: path.relative(projectRoot, buildLog)
    },
    unitTest: {
      passed: testStep.code === 0,
      logPath: path.relative(projectRoot, testLog)
    },
    dependencyGate: {
      passed: dependencyStep.code === 0 && dependencyReport?.summary?.passed === true,
      reportPath: path.relative(projectRoot, dependencyFile)
    },
    llmValidation: {
      status: (llmValidation?.status || 'human_required').toUpperCase(),
      reportPath: path.relative(projectRoot, llmValidationFile)
    }
  },
  reasons: {
    block: blockReasons,
    human: humanReasons
  },
  artifacts: {
    context: path.relative(projectRoot, contextFile),
    dependency: path.relative(projectRoot, dependencyFile),
    llmValidation: path.relative(projectRoot, llmValidationFile)
  }
};

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
fs.writeFileSync(path.join(artifactsRoot, 'review-result.latest.json'), `${JSON.stringify(result, null, 2)}\n`, 'utf8');

console.log(`Review status: ${result.status}`);
console.log(`Result file: ${path.relative(projectRoot, outputPath)}`);

if (result.status === 'PASS') {
  process.exit(0);
}
if (result.status === 'HUMAN' && allowHuman) {
  process.exit(0);
}
process.exit(1);
