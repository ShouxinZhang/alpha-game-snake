#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import ts from 'typescript';

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

function toPosix(filePath) {
  return filePath.split(path.sep).join('/');
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function normalizeAbs(filePath) {
  return path.resolve(filePath);
}

function isUnderTarget(filePath, targetAbs) {
  const normalized = normalizeAbs(filePath);
  return normalized === targetAbs || normalized.startsWith(`${targetAbs}${path.sep}`);
}

function mapFindingCode(message) {
  if (message.includes('All imports in import declaration are unused')) {
    return 'UNUSED_IMPORT_DECLARATION';
  }
  if (message.includes('is declared but its value is never read')) {
    return 'UNUSED_DECLARATION';
  }
  if (message.includes('is declared but never used')) {
    return 'UNUSED_DECLARATION';
  }
  if (message.includes('is never read')) {
    return 'UNUSED_DECLARATION';
  }
  return 'UNUSED_SYMBOL';
}

function extractSymbolName(message) {
  const match = message.match(/'([^']+)'/);
  return match ? match[1] : null;
}

function hasModifier(node, modifierKind) {
  return Boolean(node.modifiers?.some((m) => m.kind === modifierKind));
}

function resolveModuleToFile(sourceFile, specifierText, compilerOptions) {
  const resolved = ts.resolveModuleName(specifierText, sourceFile.fileName, compilerOptions, ts.sys)
    .resolvedModule?.resolvedFileName;
  if (!resolved) return null;
  if (resolved.endsWith('.d.ts')) return null;
  return normalizeAbs(resolved);
}

function addUsage(usageMap, filePath, exportName) {
  if (!usageMap.has(filePath)) {
    usageMap.set(filePath, new Set());
  }
  usageMap.get(filePath).add(exportName);
}

function getDeclarationLoc(node, cwd) {
  const sf = node.getSourceFile();
  const { line, character } = sf.getLineAndCharacterOfPosition(node.getStart());
  return {
    file: toPosix(path.relative(cwd, sf.fileName)),
    line: line + 1,
    column: character + 1
  };
}

function collectLocalExportsFromSourceFile(sourceFile, cwd) {
  const exportsList = [];

  for (const stmt of sourceFile.statements) {
    if (ts.isExportAssignment(stmt)) {
      const loc = getDeclarationLoc(stmt, cwd);
      exportsList.push({ exportName: 'default', ...loc });
      continue;
    }

    if (ts.isFunctionDeclaration(stmt) || ts.isClassDeclaration(stmt) || ts.isInterfaceDeclaration(stmt) || ts.isTypeAliasDeclaration(stmt) || ts.isEnumDeclaration(stmt)) {
      if (!hasModifier(stmt, ts.SyntaxKind.ExportKeyword)) continue;
      const isDefault = hasModifier(stmt, ts.SyntaxKind.DefaultKeyword);
      const exportName = isDefault ? 'default' : stmt.name?.text;
      if (!exportName) continue;
      const loc = getDeclarationLoc(stmt, cwd);
      exportsList.push({ exportName, ...loc });
      continue;
    }

    if (ts.isVariableStatement(stmt)) {
      if (!hasModifier(stmt, ts.SyntaxKind.ExportKeyword)) continue;
      const isDefault = hasModifier(stmt, ts.SyntaxKind.DefaultKeyword);
      for (const decl of stmt.declarationList.declarations) {
        if (!ts.isIdentifier(decl.name)) continue;
        const exportName = isDefault ? 'default' : decl.name.text;
        const loc = getDeclarationLoc(decl, cwd);
        exportsList.push({ exportName, ...loc });
      }
      continue;
    }

    if (ts.isExportDeclaration(stmt)) {
      if (stmt.moduleSpecifier) {
        continue;
      }

      if (!stmt.exportClause) {
        continue;
      }

      if (ts.isNamedExports(stmt.exportClause)) {
        for (const element of stmt.exportClause.elements) {
          const exportName = element.name.text;
          const loc = getDeclarationLoc(element, cwd);
          exportsList.push({ exportName, ...loc });
        }
      } else if (ts.isNamespaceExport(stmt.exportClause)) {
        const exportName = stmt.exportClause.name.text;
        const loc = getDeclarationLoc(stmt.exportClause, cwd);
        exportsList.push({ exportName, ...loc });
      }
    }
  }

  return exportsList;
}

const args = parseArgs(process.argv.slice(2));
const cwd = path.resolve(args['project-root'] || process.cwd());
const targetRel = args.target;
if (!targetRel) {
  console.error('Missing required argument: --target <path>');
  process.exit(2);
}

const targetAbs = normalizeAbs(path.resolve(cwd, targetRel));
if (!fs.existsSync(targetAbs)) {
  console.error(`Target not found: ${targetAbs}`);
  process.exit(2);
}

const tsconfigPath = normalizeAbs(path.resolve(cwd, args.tsconfig || 'tsconfig.json'));
if (!fs.existsSync(tsconfigPath)) {
  console.error(`tsconfig not found: ${tsconfigPath}`);
  process.exit(2);
}

const ignoreUnderscore = args['ignore-underscore'] !== 'false';
const checkUnusedExports = args['check-unused-exports'] !== 'false';
const exportIgnoreFiles = (args['export-ignore-files'] || '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);

const reportPath = normalizeAbs(
  path.resolve(
    cwd,
    args.output || `.agents/skills/modularization-governance/artifacts/${path.basename(targetAbs)}.unused-report.json`
  )
);

const configDir = path.dirname(tsconfigPath);
const configRead = ts.readConfigFile(tsconfigPath, ts.sys.readFile);
if (configRead.error) {
  const errorText = ts.flattenDiagnosticMessageText(configRead.error.messageText, '\n');
  console.error(`Failed to read tsconfig: ${errorText}`);
  process.exit(2);
}

const parsed = ts.parseJsonConfigFileContent(
  configRead.config,
  ts.sys,
  configDir,
  {
    noEmit: true,
    noUnusedLocals: true,
    noUnusedParameters: true,
    pretty: false
  },
  tsconfigPath
);

const program = ts.createProgram({
  rootNames: parsed.fileNames,
  options: parsed.options,
  projectReferences: parsed.projectReferences
});

const findings = [];
const seen = new Set();

const allDiagnostics = ts.getPreEmitDiagnostics(program);
for (const diagnostic of allDiagnostics) {
  if (!diagnostic.file || typeof diagnostic.start !== 'number') {
    continue;
  }

  const fileNameAbs = normalizeAbs(diagnostic.file.fileName);
  if (!isUnderTarget(fileNameAbs, targetAbs)) {
    continue;
  }

  const message = ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n');
  const maybeUnused =
    message.includes('never used') ||
    message.includes('never read') ||
    message.includes('unused');
  if (!maybeUnused) {
    continue;
  }

  const symbolName = extractSymbolName(message);
  if (ignoreUnderscore && symbolName && symbolName.startsWith('_')) {
    continue;
  }

  const position = diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start);
  const relFile = toPosix(path.relative(cwd, fileNameAbs));
  const code = mapFindingCode(message);
  const key = `${code}:${relFile}:${position.line + 1}:${position.character + 1}:${message}`;
  if (seen.has(key)) {
    continue;
  }
  seen.add(key);

  findings.push({
    severity: 'REFINE',
    code,
    tsCode: diagnostic.code,
    message,
    symbol: symbolName,
    file: relFile,
    line: position.line + 1,
    column: position.character + 1,
    suggestion:
      code === 'UNUSED_IMPORT_DECLARATION'
        ? '删除未使用 import，或改为 type-only 导入。'
        : '删除未使用符号，或将其接入实际调用路径。'
  });
}

let exportFindings = 0;
if (checkUnusedExports) {
  const sourceFiles = program
    .getSourceFiles()
    .filter((sf) => !sf.isDeclarationFile && /\.(tsx?|jsx?)$/.test(sf.fileName));

  const localExportsByFile = new Map();
  for (const sf of sourceFiles) {
    const abs = normalizeAbs(sf.fileName);
    if (!isUnderTarget(abs, targetAbs)) continue;

    const rel = toPosix(path.relative(cwd, abs));
    if (exportIgnoreFiles.some((token) => rel.includes(token))) {
      continue;
    }

    localExportsByFile.set(abs, collectLocalExportsFromSourceFile(sf, cwd));
  }

  const importUsageMap = new Map();
  for (const sf of sourceFiles) {
    for (const stmt of sf.statements) {
      if (!ts.isImportDeclaration(stmt) && !ts.isExportDeclaration(stmt)) {
        continue;
      }

      if (!stmt.moduleSpecifier || !ts.isStringLiteral(stmt.moduleSpecifier)) {
        continue;
      }

      const resolvedAbs = resolveModuleToFile(sf, stmt.moduleSpecifier.text, parsed.options);
      if (!resolvedAbs || !localExportsByFile.has(resolvedAbs)) {
        continue;
      }

      if (ts.isImportDeclaration(stmt)) {
        const clause = stmt.importClause;
        if (!clause) continue;

        if (clause.name) {
          addUsage(importUsageMap, resolvedAbs, 'default');
        }

        if (clause.namedBindings) {
          if (ts.isNamespaceImport(clause.namedBindings)) {
            addUsage(importUsageMap, resolvedAbs, '*');
          } else if (ts.isNamedImports(clause.namedBindings)) {
            for (const el of clause.namedBindings.elements) {
              const importedName = el.propertyName ? el.propertyName.text : el.name.text;
              addUsage(importUsageMap, resolvedAbs, importedName);
            }
          }
        }
        continue;
      }

      if (ts.isExportDeclaration(stmt)) {
        if (!stmt.exportClause) {
          addUsage(importUsageMap, resolvedAbs, '*');
          continue;
        }

        if (ts.isNamedExports(stmt.exportClause)) {
          for (const el of stmt.exportClause.elements) {
            const importedName = el.propertyName ? el.propertyName.text : el.name.text;
            addUsage(importUsageMap, resolvedAbs, importedName);
          }
        } else if (ts.isNamespaceExport(stmt.exportClause)) {
          addUsage(importUsageMap, resolvedAbs, '*');
        }
      }
    }
  }

  for (const [fileAbs, exportedItems] of localExportsByFile.entries()) {
    if (!exportedItems.length) continue;
    const usage = importUsageMap.get(fileAbs) || new Set();

    for (const item of exportedItems) {
      if (item.exportName === 'default' && usage.has('default')) {
        continue;
      }
      if (usage.has('*') || usage.has(item.exportName)) {
        continue;
      }

      const key = `UNUSED_EXPORT:${item.file}:${item.line}:${item.column}:${item.exportName}`;
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      exportFindings += 1;

      findings.push({
        severity: 'REFINE',
        code: 'UNUSED_EXPORT',
        tsCode: null,
        message: `导出符号 '${item.exportName}' 未被任何模块导入或再导出`,
        symbol: item.exportName,
        file: item.file,
        line: item.line,
        column: item.column,
        suggestion: '优先移除 export 暴露并保留定义；若定义在模块内也无引用，再删除。'
      });
    }
  }
}

const fileCounter = new Map();
const codeCounter = new Map();
for (const item of findings) {
  fileCounter.set(item.file, (fileCounter.get(item.file) || 0) + 1);
  codeCounter.set(item.code, (codeCounter.get(item.code) || 0) + 1);
}

const summary = {
  status: findings.length > 0 ? 'REFINE' : 'PASS',
  findingCount: findings.length,
  filesWithFindings: fileCounter.size,
  byCode: Object.fromEntries([...codeCounter.entries()].sort((a, b) => a[0].localeCompare(b[0]))),
  byFile: Object.fromEntries([...fileCounter.entries()].sort((a, b) => a[0].localeCompare(b[0]))),
  exportFindings
};

const report = {
  generatedAt: new Date().toISOString(),
  target: toPosix(path.relative(cwd, targetAbs)),
  tsconfig: toPosix(path.relative(cwd, tsconfigPath)),
  options: {
    ignoreUnderscore,
    checkUnusedExports,
    exportIgnoreFiles
  },
  summary,
  findings
};

ensureDir(path.dirname(reportPath));
fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');

console.log(`Status: ${summary.status}`);
console.log(`Report: ${toPosix(path.relative(cwd, reportPath))}`);
console.log(`Findings: ${summary.findingCount}`);

process.exit(summary.status === 'PASS' ? 0 : 1);
