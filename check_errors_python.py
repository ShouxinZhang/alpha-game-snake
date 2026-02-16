#!/usr/bin/env python3
"""Python 代码错误检查脚本。

支持的检查项：
  1. 语法错误 (py_compile / ast)
  2. 基本代码规范 (pycodestyle / flake8，如已安装)
  3. 常见编程问题 — 未使用的 import、变量遮蔽、缩进混乱等
  4. 类型标注检查 — None 默认值与类型不匹配、torch.compile 类型丢失等
  5. 类型检查 (pyright / mypy，如已安装)

用法：
  python check_errors_python.py                   # 检查当前目录所有 .py 文件
  python check_errors_python.py file1.py file2.py  # 检查指定文件
  python check_errors_python.py --dir src/         # 检查指定目录
"""

from __future__ import annotations

import argparse
import ast
import py_compile
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ───────────────────────── 数据结构 ─────────────────────────

@dataclass
class Issue:
    file: str
    line: int
    col: int
    level: str  # ERROR / WARNING / INFO
    code: str   # 例如 E001, W001
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}: [{self.level}] {self.code} {self.message}"


@dataclass
class CheckResult:
    file: str
    issues: List[Issue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.level == "ERROR" for i in self.issues)


# ───────────────────────── 1. 语法检查 ─────────────────────────

def check_syntax(filepath: str) -> List[Issue]:
    """使用 py_compile 和 ast 检查语法错误。"""
    issues: List[Issue] = []

    # py_compile
    try:
        py_compile.compile(filepath, doraise=True)
    except py_compile.PyCompileError as e:
        line = getattr(e, "lineno", 0) or 0
        col = getattr(e, "offset", 0) or 0
        issues.append(Issue(filepath, line, col, "ERROR", "E001", f"语法错误: {e.msg}"))
        return issues  # 语法错误严重，直接返回

    # ast 解析（可捕获一些 py_compile 遗漏的情况）
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="replace")
        ast.parse(source, filename=filepath)
    except SyntaxError as e:
        issues.append(Issue(
            filepath,
            e.lineno or 0,
            e.offset or 0,
            "ERROR", "E002",
            f"AST 解析失败: {e.msg}",
        ))

    return issues


# ───────────────────────── 2. 基础静态分析 ─────────────────────────

class _BasicAnalyzer(ast.NodeVisitor):
    """用纯 AST 做轻量检查，不依赖第三方库。"""

    def __init__(self, filepath: str, source: str):
        self.filepath = filepath
        self.source = source
        self.lines = source.splitlines()
        self.issues: List[Issue] = []
        self._imported_names: dict[str, int] = {}  # name -> lineno
        self._used_names: set[str] = set()
        self._scope_stack: list[set[str]] = [set()]
        self._in_function = False

    # ── import 检查 ──

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self._imported_names[name] = node.lineno
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.issues.append(Issue(
                    self.filepath, node.lineno, node.col_offset,
                    "WARNING", "W001", f"通配符导入 'from {node.module} import *' 不推荐使用",
                ))
            else:
                name = alias.asname or alias.name
                self._imported_names[name] = node.lineno
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self._used_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._used_names.add(node.attr)
        self.generic_visit(node)

    # ── 裸 except 检查 ──

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.issues.append(Issue(
                self.filepath, node.lineno, node.col_offset,
                "WARNING", "W002", "裸 except 会捕获所有异常（包括 KeyboardInterrupt），建议指定异常类型",
            ))
        self.generic_visit(node)

    # ── 可变默认参数 & 类型标注 ──

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_mutable_defaults(node)
        self._check_too_many_args(node)
        self._check_none_default_without_optional(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_mutable_defaults(node)  # type: ignore[arg-type]
        self._check_too_many_args(node)  # type: ignore[arg-type]
        self._check_none_default_without_optional(node)  # type: ignore[arg-type]
        self.generic_visit(node)

    def _check_mutable_defaults(self, node: ast.FunctionDef) -> None:
        for default in node.args.defaults + node.args.kw_defaults:
            if default is None:
                continue
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.issues.append(Issue(
                    self.filepath, default.lineno, default.col_offset,
                    "WARNING", "W003",
                    f"函数 '{node.name}' 使用了可变默认参数，可能导致意外共享状态",
                ))

    def _check_too_many_args(self, node: ast.FunctionDef) -> None:
        args = node.args
        total = len(args.args) + len(args.posonlyargs) + len(args.kwonlyargs)
        if args.vararg:
            total += 1
        if args.kwarg:
            total += 1
        if total > 10:
            self.issues.append(Issue(
                self.filepath, node.lineno, node.col_offset,
                "INFO", "I001",
                f"函数 '{node.name}' 有 {total} 个参数，建议精简",
            ))

    @staticmethod
    def _annotation_allows_none(annotation: ast.expr) -> bool:
        """判断类型标注是否允许 None 值。"""
        # Optional[X] -> typing.Optional 或 X | None
        if isinstance(annotation, ast.Constant) and annotation.value is None:
            return True
        if isinstance(annotation, ast.Name) and annotation.id in ("None", "Optional", "Any"):
            return True
        if isinstance(annotation, ast.Attribute) and annotation.attr in ("Optional", "Any"):
            return True
        # Optional[X] 解析为 Subscript: Optional[X]
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name) and annotation.value.id == "Optional":
                return True
            if isinstance(annotation.value, ast.Attribute) and annotation.value.attr == "Optional":
                return True
        # X | None (Python 3.10+ union syntax)
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            if _BasicAnalyzer._annotation_allows_none(annotation.left):
                return True
            if _BasicAnalyzer._annotation_allows_none(annotation.right):
                return True
        # Union[X, None]
        if isinstance(annotation, ast.Subscript):
            func = annotation.value
            if (isinstance(func, ast.Name) and func.id == "Union") or \
               (isinstance(func, ast.Attribute) and func.attr == "Union"):
                # Union 的参数是 Tuple
                slice_node = annotation.slice
                if isinstance(slice_node, ast.Tuple):
                    for elt in slice_node.elts:
                        if _BasicAnalyzer._annotation_allows_none(elt):
                            return True
        return False

    def _check_none_default_without_optional(self, node: ast.FunctionDef) -> None:
        """检查参数默认值为 None 但类型标注不含 Optional/None 的情况。"""
        args_list = node.args.args
        defaults = node.args.defaults
        # defaults 对应 args 的最后 len(defaults) 个参数
        offset = len(args_list) - len(defaults)
        for i, default in enumerate(defaults):
            if not (isinstance(default, ast.Constant) and default.value is None):
                continue
            arg = args_list[offset + i]
            if arg.annotation is None:
                continue  # 没有类型标注，跳过
            if not self._annotation_allows_none(arg.annotation):
                type_str = ast.unparse(arg.annotation) if hasattr(ast, "unparse") else "<type>"
                self.issues.append(Issue(
                    self.filepath, arg.lineno, arg.col_offset,
                    "ERROR", "E010",
                    f"参数 '{arg.arg}' 类型标注为 '{type_str}' 但默认值为 None，"
                    f"应使用 'Optional[{type_str}]' 或 '{type_str} | None'",
                ))

        # 同样检查 keyword-only 参数
        kw_defaults = node.args.kw_defaults
        for i, default in enumerate(kw_defaults):
            if default is None:
                continue
            if not (isinstance(default, ast.Constant) and default.value is None):
                continue
            arg = node.args.kwonlyargs[i]
            if arg.annotation is None:
                continue
            if not self._annotation_allows_none(arg.annotation):
                type_str = ast.unparse(arg.annotation) if hasattr(ast, "unparse") else "<type>"
                self.issues.append(Issue(
                    self.filepath, arg.lineno, arg.col_offset,
                    "ERROR", "E010",
                    f"参数 '{arg.arg}' 类型标注为 '{type_str}' 但默认值为 None，"
                    f"应使用 'Optional[{type_str}]' 或 '{type_str} | None'",
                ))

    # ── 行级检查 ──

    def _check_lines(self) -> None:
        has_tabs = False
        has_spaces = False
        for i, line in enumerate(self.lines, start=1):
            # 过长行
            if len(line) > 150:
                self.issues.append(Issue(
                    self.filepath, i, 150,
                    "INFO", "I002",
                    f"行长度 {len(line)} 超过 150 字符",
                ))
            # 混合缩进
            stripped = line.lstrip()
            if stripped:
                indent = line[: len(line) - len(stripped)]
                if "\t" in indent:
                    has_tabs = True
                if " " in indent:
                    has_spaces = True
            # 行尾空格
            if line.rstrip() != line and line.strip():
                self.issues.append(Issue(
                    self.filepath, i, len(line.rstrip()),
                    "INFO", "I003", "行尾有多余空格",
                ))

        if has_tabs and has_spaces:
            self.issues.append(Issue(
                self.filepath, 1, 0,
                "WARNING", "W004", "文件中混合使用了 Tab 和空格缩进",
            ))

    # ── 未使用 import ──

    def _check_unused_imports(self) -> None:
        # 一些常见的 side-effect import 不报未使用
        side_effect_modules = {"__future__", "annotations", "typing", "typing_extensions"}
        for name, lineno in self._imported_names.items():
            if name not in self._used_names and name not in side_effect_modules:
                self.issues.append(Issue(
                    self.filepath, lineno, 0,
                    "WARNING", "W005",
                    f"导入的 '{name}' 未被使用",
                ))

    # ── torch.compile 类型检查 ──

    def visit_Assign(self, node: ast.Assign) -> None:
        """检测 model = torch.compile(model) 后对 model 调用 .parameters() 等 nn.Module 属性。"""
        self._check_torch_compile_reassign(node)
        self._check_method_alias_type_mismatch(node)
        self.generic_visit(node)

    def _check_torch_compile_reassign(self, node: ast.Assign) -> None:
        # 匹配: x = torch.compile(x) 或 x = torch.compile(x, ...)
        if not isinstance(node.value, ast.Call):
            return
        call = node.value
        func = call.func
        is_torch_compile = False
        if isinstance(func, ast.Attribute) and func.attr == "compile":
            if isinstance(func.value, ast.Name) and func.value.id == "torch":
                is_torch_compile = True
        if not is_torch_compile:
            return

        # 拿到被赋值的变量名
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return
        var_name = node.targets[0].id

        # 拿到 torch.compile 的第一个参数名
        arg_name = None
        if call.args and isinstance(call.args[0], ast.Name):
            arg_name = call.args[0].id

        # 只在变量名与参数名相同时告警（即 model = torch.compile(model) 覆盖了原变量）
        if var_name == arg_name:
            self.issues.append(Issue(
                self.filepath, node.lineno, node.col_offset,
                "WARNING", "W010",
                f"'{var_name} = torch.compile({var_name})' 会使 '{var_name}' 的类型变为"
                f" 通用可调用对象，后续调用 .parameters()/.state_dict() 等 nn.Module 属性"
                f" 将产生类型错误。建议保留原始模型引用，例如: compiled_{var_name} = torch.compile({var_name})",
            ))

    def _check_method_alias_type_mismatch(self, node: ast.Assign) -> None:
        """检测类中 visit_AsyncFunctionDef = visit_FunctionDef 这类别名赋值的类型不兼容。"""
        # 仅在类定义内部检查
        if not isinstance(node.value, ast.Name):
            return
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return
        target_name = node.targets[0].id
        source_name = node.value.id

        # 常见的 AST visitor 别名模式：visit_AsyncXxx = visit_Xxx
        # 这些方法的参数类型不同（AsyncFunctionDef vs FunctionDef）
        async_sync_pairs = {
            ("visit_AsyncFunctionDef", "visit_FunctionDef"),
            ("visit_AsyncFor", "visit_For"),
            ("visit_AsyncWith", "visit_With"),
        }
        if (target_name, source_name) in async_sync_pairs:
            self.issues.append(Issue(
                self.filepath, node.lineno, node.col_offset,
                "WARNING", "W011",
                f"'{target_name} = {source_name}' 会导致类型不兼容——"
                f"'{target_name}' 期望的参数类型与 '{source_name}' 不同。"
                f"建议定义独立方法避免类型检查器报错",
            ))

    # ── torch.amp 导出检查 ──

    def visit_Call(self, node: ast.Call) -> None:
        """检测 torch.amp.autocast / torch.amp.GradScaler 等未正式导出的用法。"""
        self._check_torch_amp_usage(node)
        self.generic_visit(node)

    def _check_torch_amp_usage(self, node: ast.Call, from_with: bool = False) -> None:
        func = node.func
        # 匹配 torch.amp.X 模式
        if not isinstance(func, ast.Attribute):
            return
        attr_name = func.attr  # e.g. "autocast", "GradScaler"
        parent = func.value
        if not isinstance(parent, ast.Attribute):
            return
        if parent.attr != "amp":
            return
        if not isinstance(parent.value, ast.Name):
            return
        if parent.value.id != "torch":
            return

        # 防止重复报告：如果从 visit_With 调用，标记节点；visit_Call 遇到已标记节点则跳过
        node_id = id(node)
        if not hasattr(self, "_amp_reported"):
            self._amp_reported: set[int] = set()
        if node_id in self._amp_reported:
            return
        self._amp_reported.add(node_id)

        deprecated_apis = {
            "autocast": "torch.cuda.amp.autocast 或 torch.autocast('cuda')",
            "GradScaler": "torch.cuda.amp.GradScaler",
        }

        if attr_name in deprecated_apis:
            suggestion = deprecated_apis[attr_name]
            self.issues.append(Issue(
                self.filepath, node.lineno, node.col_offset,
                "ERROR", "E011",
                f"'torch.amp.{attr_name}' 未从 torch.amp 模块正式导出，"
                f"类型检查器将报错。应使用 {suggestion}",
            ))

    # ── with 语句中的 torch.amp 检查 ──

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            # with 语句的 context_expr 会被 visit_Call 单独处理，这里标记跳过
            if isinstance(item.context_expr, ast.Call):
                self._check_torch_amp_usage(item.context_expr, from_with=True)
        self.generic_visit(node)

    def run(self) -> List[Issue]:
        try:
            tree = ast.parse(self.source, filename=self.filepath)
        except SyntaxError:
            return self.issues  # 语法阶段已报告
        self.visit(tree)
        self._check_lines()
        self._check_unused_imports()
        return self.issues


def check_basic_analysis(filepath: str) -> List[Issue]:
    source = Path(filepath).read_text(encoding="utf-8", errors="replace")
    analyzer = _BasicAnalyzer(filepath, source)
    return analyzer.run()


# ───────────────────────── 3. 外部工具 (可选) ─────────────────────────

def _run_external(cmd: List[str], filepath: str, tool_name: str, level: str = "WARNING") -> List[Issue]:
    """运行外部命令，解析其输出为 Issue 列表。"""
    try:
        result = subprocess.run(
            cmd + [filepath],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return [Issue(filepath, 0, 0, "WARNING", "T001", f"{tool_name} 运行超时")]

    issues: List[Issue] = []
    for line in result.stdout.splitlines():
        # 常见格式: file:line:col: CODE message
        parts = line.split(":", 3)
        if len(parts) >= 4:
            try:
                ln = int(parts[1])
                co = int(parts[2]) if parts[2].strip().isdigit() else 0
                msg = parts[3].strip()
                issues.append(Issue(filepath, ln, co, level, f"{tool_name}", msg))
            except (ValueError, IndexError):
                continue
    return issues


def check_flake8(filepath: str) -> List[Issue]:
    return _run_external(
        ["flake8", "--max-line-length", "150", "--select", "E,W,F"],
        filepath, "flake8",
    )


def check_mypy(filepath: str) -> List[Issue]:
    return _run_external(
        ["mypy", "--ignore-missing-imports", "--no-error-summary"],
        filepath, "mypy", level="WARNING",
    )


def check_pyright(filepath: str) -> List[Issue]:
    """使用 pyright 进行类型检查（需安装: pip install pyright）。"""
    try:
        result = subprocess.run(
            ["pyright", "--outputjson", filepath],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return [Issue(filepath, 0, 0, "WARNING", "T002", "pyright 运行超时")]

    issues: List[Issue] = []
    try:
        import json as _json
        data = _json.loads(result.stdout)
        for diag in data.get("generalDiagnostics", []):
            severity = diag.get("severity", "information")
            level = {"error": "ERROR", "warning": "WARNING"}.get(severity, "INFO")
            rng = diag.get("range", {})
            start = rng.get("start", {})
            line = start.get("line", 0) + 1  # pyright 行号从 0 开始
            col = start.get("character", 0)
            rule = diag.get("rule", "")
            msg = diag.get("message", "")
            code_str = f"pyright({rule})" if rule else "pyright"
            issues.append(Issue(filepath, line, col, level, code_str, msg))
    except (ValueError, KeyError):
        pass
    return issues


# ───────────────────────── 汇总 & 报告 ─────────────────────────

def check_file(
    filepath: str,
    use_flake8: bool = True,
    use_mypy: bool = False,
    use_pyright: bool = False,
) -> CheckResult:
    result = CheckResult(file=filepath)

    # 1) 语法
    syntax_issues = check_syntax(filepath)
    result.issues.extend(syntax_issues)
    if any(i.level == "ERROR" for i in syntax_issues):
        return result  # 有语法错误则跳过后续

    # 2) 基础静态分析（含类型标注检查）
    result.issues.extend(check_basic_analysis(filepath))

    # 3) flake8
    if use_flake8:
        result.issues.extend(check_flake8(filepath))

    # 4) mypy
    if use_mypy:
        result.issues.extend(check_mypy(filepath))

    # 5) pyright
    if use_pyright:
        result.issues.extend(check_pyright(filepath))

    return result


def collect_py_files(paths: List[str]) -> List[str]:
    """从路径列表收集所有 .py 文件。"""
    files: List[str] = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".py":
            files.append(str(path))
        elif path.is_dir():
            for f in sorted(path.rglob("*.py")):
                # 跳过虚拟环境和隐藏目录
                parts = f.parts
                if any(part.startswith(".") or part in ("__pycache__", ".venv", "venv", "env", "node_modules") for part in parts):
                    continue
                files.append(str(f))
    return files


def print_report(results: List[CheckResult]) -> int:
    """打印检查报告，返回总错误数。"""
    total_errors = 0
    total_warnings = 0
    total_info = 0

    for r in results:
        if not r.issues:
            continue
        print(f"\n{'─' * 60}")
        print(f"📄 {r.file}")
        print(f"{'─' * 60}")
        for issue in sorted(r.issues, key=lambda i: (i.line, i.col)):
            icon = {"ERROR": "❌", "WARNING": "⚠️ ", "INFO": "ℹ️ "}.get(issue.level, "  ")
            print(f"  {icon} 行 {issue.line:>4}:{issue.col:<3} {issue.code:<8} {issue.message}")

        errors = sum(1 for i in r.issues if i.level == "ERROR")
        warnings = sum(1 for i in r.issues if i.level == "WARNING")
        infos = sum(1 for i in r.issues if i.level == "INFO")
        total_errors += errors
        total_warnings += warnings
        total_info += infos

    # 汇总
    print(f"\n{'═' * 60}")
    files_checked = len(results)
    files_with_issues = sum(1 for r in results if r.issues)
    print(f"✅ 检查完成: {files_checked} 个文件, {files_with_issues} 个有问题")
    print(f"   ❌ 错误: {total_errors}  ⚠️  警告: {total_warnings}  ℹ️  提示: {total_info}")
    print(f"{'═' * 60}")

    if total_errors == 0 and total_warnings == 0:
        print("🎉 没有发现错误和警告！")

    return total_errors


# ───────────────────────── CLI ─────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Python 代码错误检查脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              python check_errors_python.py                    # 检查当前目录
              python check_errors_python.py snake_rl_parallel.py
              python check_errors_python.py --dir src/ --mypy
        """),
    )
    parser.add_argument("files", nargs="*", help="要检查的 Python 文件（默认当前目录所有 .py）")
    parser.add_argument("--dir", "-d", default=None, help="要检查的目录")
    parser.add_argument("--no-flake8", action="store_true", help="跳过 flake8 检查")
    parser.add_argument("--mypy", action="store_true", help="启用 mypy 类型检查")
    parser.add_argument("--pyright", action="store_true", help="启用 pyright 类型检查")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    # 确定要检查的文件
    targets = args.files if args.files else []
    if args.dir:
        targets.append(args.dir)
    if not targets:
        targets = ["."]

    py_files = collect_py_files(targets)
    if not py_files:
        print("未找到 .py 文件。")
        sys.exit(0)

    if args.verbose:
        print(f"将检查 {len(py_files)} 个文件...")

    results: List[CheckResult] = []
    for f in py_files:
        if args.verbose:
            print(f"  检查 {f} ...")
        results.append(check_file(
            f,
            use_flake8=not args.no_flake8,
            use_mypy=args.mypy,
            use_pyright=args.pyright,
        ))

    total_errors = print_report(results)
    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    main()
