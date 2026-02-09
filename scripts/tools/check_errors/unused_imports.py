from __future__ import annotations

import argparse
import ast
import os
from dataclasses import dataclass
from typing import Iterable


EXCLUDE_DIR_NAMES_DEFAULT = {".venv", "venv", "__pycache__", "old_files"}


@dataclass(frozen=True)
class UnusedImport:
    file_path: str
    line: int
    name: str

    def format(self) -> str:
        return f"  ERROR: In {self.file_path}:{self.line}, '{self.name}' is imported but never used."


def _iter_py_files(root: str, exclude_dir_names: set[str]) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dir_names]
        for filename in filenames:
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def _is_type_checking_test(test: ast.expr) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if (
        isinstance(test, ast.Attribute)
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
        and test.attr == "TYPE_CHECKING"
    ):
        return True
    return False


def _collect_exported_all_strings(module: ast.Module) -> set[str]:
    exported: set[str] = set()
    for stmt in module.body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                value = stmt.value
                if isinstance(value, (ast.List, ast.Tuple)):
                    for elt in value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exported.add(elt.value)
    return exported


def _collect_exempt_import_lines(module: ast.Module) -> set[int]:
    exempt: set[int] = set()
    for stmt in module.body:
        if isinstance(stmt, ast.If) and _is_type_checking_test(stmt.test):
            for inner in stmt.body:
                if isinstance(inner, (ast.Import, ast.ImportFrom)) and getattr(inner, "lineno", None):
                    exempt.add(int(inner.lineno))
    return exempt


def _bound_names_from_import(node: ast.AST) -> list[tuple[str, int]]:
    results: list[tuple[str, int]] = []
    lineno = int(getattr(node, "lineno", 0) or 0)

    if isinstance(node, ast.Import):
        for alias in node.names:
            bound = alias.asname or alias.name.split(".", 1)[0]
            results.append((bound, lineno))
    elif isinstance(node, ast.ImportFrom):
        if node.module == "__future__":
            return []
        for alias in node.names:
            if alias.name == "*":
                continue
            bound = alias.asname or alias.name
            results.append((bound, lineno))

    return results


class _UsedNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.used: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.used.add(node.id)
        self.generic_visit(node)


def scan_file(path: str) -> list[UnusedImport]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            src = f.read()

    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError:
        return []

    used_collector = _UsedNameCollector()
    used_collector.visit(tree)

    exported = _collect_exported_all_strings(tree)
    used_names = used_collector.used | exported
    exempt_lines = _collect_exempt_import_lines(tree)

    imported: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported.extend(_bound_names_from_import(node))

    unused: list[UnusedImport] = []
    for name, lineno in imported:
        if lineno in exempt_lines:
            continue
        if name not in used_names:
            unused.append(UnusedImport(file_path=path, line=lineno, name=name))

    return unused


def scan_project(root: str, exclude_dir_names: set[str]) -> list[UnusedImport]:
    all_unused: list[UnusedImport] = []
    for pyfile in _iter_py_files(root, exclude_dir_names):
        all_unused.extend(scan_file(pyfile))
    return all_unused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect unused imports via AST scan")
    parser.add_argument("project_dir")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude directory name (can be repeated). Default excludes: .venv, venv, __pycache__, old_files",
    )

    args = parser.parse_args(argv)
    exclude_dir_names = set(EXCLUDE_DIR_NAMES_DEFAULT)
    exclude_dir_names.update(args.exclude)

    unused = scan_project(args.project_dir, exclude_dir_names)
    if unused:
        for item in unused:
            print(item.format())
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
