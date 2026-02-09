from __future__ import annotations

import argparse
import ast
import importlib
import os
from dataclasses import dataclass
from typing import Iterable


EXCLUDE_DIR_NAMES_DEFAULT = {".venv", "venv", "__pycache__", "old_files"}


@dataclass(frozen=True)
class DunderAllIssue:
    init_path: str
    message: str

    def format(self) -> str:
        return f"  ERROR: In {self.init_path}, {self.message}"


def _iter_init_files(root: str, exclude_dir_names: set[str]) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dir_names]
        for filename in filenames:
            if filename == "__init__.py":
                yield os.path.join(dirpath, filename)


def _module_name_from_init(project_dir: str, init_path: str) -> str:
    rel_path = os.path.relpath(init_path, project_dir)
    mod_path = rel_path[: -len("/__init__.py")] if rel_path.endswith("/__init__.py") else rel_path
    return mod_path.replace(os.sep, ".")


def _file_has_dunder_all(init_path: str) -> bool:
    """Fast AST pre-check: return True only if the file declares __all__."""
    try:
        with open(init_path, "r", encoding="utf-8") as f:
            src = f.read()
    except (UnicodeDecodeError, OSError):
        return False
    if "__all__" not in src:
        return False
    try:
        tree = ast.parse(src, filename=init_path)
    except SyntaxError:
        return False
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return True
    return False


def validate_init(project_dir: str, init_path: str) -> list[DunderAllIssue]:
    # Skip files that don't declare __all__ — avoids unnecessary import attempts
    if not _file_has_dunder_all(init_path):
        return []

    module_name = _module_name_from_init(project_dir, init_path)

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        return [
            DunderAllIssue(
                init_path=init_path,
                message=f"failed to import module '{module_name}' to validate __all__: {e}",
            )
        ]

    exports = getattr(mod, "__all__", None)
    if exports is None:
        return []

    issues: list[DunderAllIssue] = []
    if not isinstance(exports, (list, tuple)):
        issues.append(
            DunderAllIssue(
                init_path=init_path,
                message=f"__all__ must be a list/tuple of strings, got {type(exports).__name__}",
            )
        )
        return issues

    for item in exports:
        if not isinstance(item, str):
            issues.append(
                DunderAllIssue(
                    init_path=init_path,
                    message=f"__all__ contains non-string element: {item!r} ({type(item).__name__})",
                )
            )
            continue
        if not hasattr(mod, item):
            issues.append(
                DunderAllIssue(
                    init_path=init_path,
                    message=f"'{item}' is specified in __all__ but is not present in module",
                )
            )

    return issues


def validate_project(project_dir: str, exclude_dir_names: set[str]) -> list[DunderAllIssue]:
    issues: list[DunderAllIssue] = []
    for init_path in _iter_init_files(project_dir, exclude_dir_names):
        issues.extend(validate_init(project_dir, init_path))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate __all__ exports in packages")
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

    issues = validate_project(args.project_dir, exclude_dir_names)
    if issues:
        for issue in issues:
            print(issue.format())
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
