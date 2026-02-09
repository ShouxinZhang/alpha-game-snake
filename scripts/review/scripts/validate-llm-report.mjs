#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

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

function isIssueArray(value) {
  if (!Array.isArray(value)) return false;
  return value.every((item) => {
    if (!item || typeof item !== 'object') return false;
    if (typeof item.title !== 'string' || item.title.trim() === '') return false;
    if (typeof item.reason !== 'string' || item.reason.trim() === '') return false;
    if (!Array.isArray(item.evidence) || item.evidence.length === 0) return false;
    return item.evidence.every((entry) => typeof entry === 'string' && entry.trim() !== '');
  });
}

const args = parseArgs(process.argv.slice(2));
const projectRoot = path.resolve(args['project-root'] || process.cwd());
const policyPath = path.resolve(args.policy || path.join(projectRoot, 'scripts/review/config/policy.json'));
const inputPath = args.input ? path.resolve(args.input) : path.join(projectRoot, 'scripts/review/input/llm-review.json');
const outputPath = path.resolve(args.output || path.join(projectRoot, 'scripts/review/artifacts/llm-validation.json'));

const policy = JSON.parse(fs.readFileSync(policyPath, 'utf8'));
const minLlmConfidence = policy.thresholds?.minLlmConfidence ?? 0.75;

const result = {
  generatedAt: new Date().toISOString(),
  inputPath: path.relative(projectRoot, inputPath),
  present: false,
  valid: false,
  status: 'human_required',
  reasons: [],
  parsed: null
};

if (!fs.existsSync(inputPath)) {
  result.reasons.push('LLM 风险报告缺失');
} else {
  result.present = true;
  try {
    const parsed = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
    result.parsed = parsed;

    const riskLevels = new Set(['low', 'medium', 'high', 'critical']);
    if (typeof parsed.summary !== 'string' || parsed.summary.trim() === '') {
      result.reasons.push('字段 summary 缺失或为空');
    }
    if (!riskLevels.has(parsed.riskLevel)) {
      result.reasons.push('字段 riskLevel 非法（需 low/medium/high/critical）');
    }
    if (typeof parsed.confidence !== 'number' || Number.isNaN(parsed.confidence) || parsed.confidence < 0 || parsed.confidence > 1) {
      result.reasons.push('字段 confidence 非法（需 0~1 数值）');
    }
    if (!isIssueArray(parsed.blockingIssues || [])) {
      result.reasons.push('字段 blockingIssues 非法（需数组，且每项包含 title/reason/evidence[]）');
    }
    if (!isIssueArray(parsed.advisories || [])) {
      result.reasons.push('字段 advisories 非法（需数组，且每项包含 title/reason/evidence[]）');
    }

    result.valid = result.reasons.length === 0;
    if (result.valid) {
      if ((parsed.blockingIssues || []).length > 0) {
        result.status = 'block';
      } else if (parsed.confidence < minLlmConfidence) {
        result.status = 'human_required';
        result.reasons.push(`置信度 ${parsed.confidence} 低于阈值 ${minLlmConfidence}`);
      } else {
        result.status = 'pass';
      }
    }
  } catch (error) {
    result.reasons.push(`LLM 风险报告 JSON 解析失败: ${String(error)}`);
  }
}

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
console.log(outputPath);
process.exit(0);
