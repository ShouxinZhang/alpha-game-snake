#!/usr/bin/env node
/**
 * review-pipeline MCP Server
 *
 * 提供 review 链路相关工具：
 * - review_collect_context
 * - review_dependency_gate
 * - review_validate_llm
 * - review_run
 */
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '../../');
const reviewRoot = path.join(repoRoot, 'scripts', 'review');
const scriptsDir = path.join(reviewRoot, 'scripts');

function resolvePath(inputPath) {
  if (!inputPath) return null;
  if (path.isAbsolute(inputPath)) return inputPath;
  return path.resolve(repoRoot, inputPath);
}

function relativeToRepo(targetPath) {
  return path.relative(repoRoot, targetPath);
}

function makeArtifactPath(fileName) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  return path.join(reviewRoot, 'artifacts', 'mcp', stamp, fileName);
}

function runNodeScript(scriptFile, args = []) {
  const fullScriptPath = path.join(scriptsDir, scriptFile);
  const commandArgs = [fullScriptPath, '--project-root', repoRoot, ...args];

  const run = spawnSync('node', commandArgs, {
    cwd: repoRoot,
    encoding: 'utf8'
  });

  const stdout = run.stdout || '';
  const stderr = run.stderr || '';
  const exitCode = run.status ?? 1;

  return {
    exitCode,
    stdout,
    stderr,
    outputPreview: `${stdout}${stderr}`.trim().split('\n').filter(Boolean).slice(-20)
  };
}

function readJsonSafe(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

function summarizeExecution(title, result, extraLines = []) {
  const lines = [];
  lines.push(`${title}`);
  lines.push(`exitCode: ${result.exitCode}`);
  if (extraLines.length > 0) {
    lines.push(...extraLines);
  }
  if (result.outputPreview.length > 0) {
    lines.push('outputPreview:');
    for (const line of result.outputPreview) {
      lines.push(`  ${line}`);
    }
  }
  return lines.join('\n');
}

const server = new McpServer({
  name: 'review-pipeline',
  version: '1.0.0'
});

server.tool(
  'review_collect_context',
  '收集本次评审的 Git 上下文（分支、提交、改动文件）。',
  {
    base: z.string().optional().describe('可选，diff 起点，如 main'),
    head: z.string().optional().describe('可选，diff 终点，如 HEAD'),
    output: z.string().optional().describe('可选，输出路径（相对仓库根目录或绝对路径）')
  },
  async ({ base, head, output }) => {
    const outputPath = resolvePath(output) || makeArtifactPath('context.json');
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });

    const args = ['--output', outputPath];
    if (base) args.push('--base', base);
    if (head) args.push('--head', head);

    const run = runNodeScript('collect-context.mjs', args);
    const payload = readJsonSafe(outputPath);
    const text = summarizeExecution('review_collect_context', run, [
      `output: ${relativeToRepo(outputPath)}`,
      `changedFiles: ${payload?.totalChangedFiles ?? 0}`
    ]);

    return { content: [{ type: 'text', text }] };
  }
);

server.tool(
  'review_dependency_gate',
  '执行依赖关系门禁（循环依赖 + 禁止边规则）。',
  {
    policy: z.string().optional().describe('可选，策略文件路径（默认 scripts/review/config/policy.json）'),
    output: z.string().optional().describe('可选，输出路径（相对仓库根目录或绝对路径）')
  },
  async ({ policy, output }) => {
    const outputPath = resolvePath(output) || makeArtifactPath('dependency-report.json');
    const policyPath = resolvePath(policy) || path.join(reviewRoot, 'config', 'policy.json');
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });

    const run = runNodeScript('dependency-gate.mjs', [
      '--policy',
      policyPath,
      '--output',
      outputPath
    ]);

    const report = readJsonSafe(outputPath);
    const text = summarizeExecution('review_dependency_gate', run, [
      `policy: ${relativeToRepo(policyPath)}`,
      `output: ${relativeToRepo(outputPath)}`,
      `passed: ${report?.summary?.passed === true ? 'true' : 'false'}`,
      `cycleCount: ${report?.summary?.cycleCount ?? 'N/A'}`,
      `forbiddenCount: ${report?.summary?.forbiddenCount ?? 'N/A'}`
    ]);

    return { content: [{ type: 'text', text }] };
  }
);

server.tool(
  'review_validate_llm',
  '校验 LLM Review 报告的结构完整性与置信度阈值。',
  {
    input: z.string().optional().describe('可选，LLM 报告路径（默认 scripts/review/input/llm-review.json）'),
    policy: z.string().optional().describe('可选，策略文件路径（默认 scripts/review/config/policy.json）'),
    output: z.string().optional().describe('可选，输出路径（相对仓库根目录或绝对路径）')
  },
  async ({ input, policy, output }) => {
    const inputPath = resolvePath(input) || path.join(reviewRoot, 'input', 'llm-review.json');
    const policyPath = resolvePath(policy) || path.join(reviewRoot, 'config', 'policy.json');
    const outputPath = resolvePath(output) || makeArtifactPath('llm-validation.json');
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });

    const run = runNodeScript('validate-llm-report.mjs', [
      '--input',
      inputPath,
      '--policy',
      policyPath,
      '--output',
      outputPath
    ]);

    const validation = readJsonSafe(outputPath);
    const text = summarizeExecution('review_validate_llm', run, [
      `input: ${relativeToRepo(inputPath)}`,
      `policy: ${relativeToRepo(policyPath)}`,
      `output: ${relativeToRepo(outputPath)}`,
      `status: ${(validation?.status || 'human_required').toUpperCase()}`,
      `valid: ${validation?.valid === true ? 'true' : 'false'}`
    ]);

    return { content: [{ type: 'text', text }] };
  }
);

server.tool(
  'review_run',
  '执行完整 review 链路并输出统一结论（PASS/BLOCK/HUMAN）。',
  {
    base: z.string().optional().describe('可选，diff 起点，如 main'),
    head: z.string().optional().describe('可选，diff 终点，如 HEAD'),
    llmReport: z.string().optional().describe('可选，LLM 报告路径'),
    allowHuman: z.boolean().optional().default(false).describe('true 时允许 HUMAN 状态返回成功退出码'),
    output: z.string().optional().describe('可选，结果输出路径（默认自动生成）')
  },
  async ({ base, head, llmReport, allowHuman, output }) => {
    const outputPath = resolvePath(output) || makeArtifactPath('review-result.json');
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });

    const args = ['--output', outputPath];
    if (base) args.push('--base', base);
    if (head) args.push('--head', head);
    if (llmReport) args.push('--llm-report', resolvePath(llmReport));
    if (allowHuman) args.push('--allow-human', 'true');

    const run = runNodeScript('run-review.mjs', args);
    const result = readJsonSafe(outputPath);

    const text = summarizeExecution('review_run', run, [
      `output: ${relativeToRepo(outputPath)}`,
      `status: ${(result?.status || 'UNKNOWN').toUpperCase()}`,
      `blockReasons: ${(result?.reasons?.block || []).length}`,
      `humanReasons: ${(result?.reasons?.human || []).length}`
    ]);

    return { content: [{ type: 'text', text }] };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
