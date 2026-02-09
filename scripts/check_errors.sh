#!/usr/bin/env bash
#
# check_errors.sh - Generic quality gate script
#
# Usage:
#   bash scripts/check_errors.sh
#   bash scripts/check_errors.sh --lint
#   bash scripts/check_errors.sh --tsc
#   bash scripts/check_errors.sh --build
#   bash scripts/check_errors.sh --python
#
# Customization (optional env):
#   QUALITY_DEPENDENCY_CMD="npm ci"
#   QUALITY_TYPECHECK_CMD="npm run typecheck"
#   QUALITY_LINT_CMD="npm run lint"
#   QUALITY_BUILD_CMD="npm run build"
#   QUALITY_PYTHON_TEST_CMD="python -m pytest"
#   QUALITY_PYTHON_EXCLUDE_DIRS=".venv,venv,__pycache__,old_files,node_modules"
#

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

TOTAL_ERRORS=0
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
RESULTS=()

record_result() {
  local step_name=$1
  local exit_code=$2
  local output=$3

  if [ "$exit_code" -eq 0 ]; then
    echo -e "  ${GREEN}✔ $step_name 通过${NC}"
    RESULTS+=("${GREEN}✔ $step_name${NC}")
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    echo -e "  ${RED}✘ $step_name 失败${NC}"
    if [ -n "$output" ]; then
      echo -e "${YELLOW}$output${NC}" | head -30
    fi
    RESULTS+=("${RED}✘ $step_name${NC}")
    FAIL_COUNT=$((FAIL_COUNT + 1))
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
  fi
}

run_cmd_step() {
  local name="$1"
  local cmd="$2"
  local output=""
  local code=0
  output=$(cd "$PROJECT_ROOT" && bash -lc "$cmd" 2>&1) || code=$?
  record_result "$name" "$code" "$output"
}

detect_typecheck_cmd() {
  if [ -n "${QUALITY_TYPECHECK_CMD:-}" ]; then
    echo "$QUALITY_TYPECHECK_CMD"
    return
  fi
  if [ -f "$PROJECT_ROOT/package.json" ] && cd "$PROJECT_ROOT" && npm run | grep -qE ' typecheck\b'; then
    echo "npm run typecheck"
    return
  fi
  if [ -f "$PROJECT_ROOT/tsconfig.json" ]; then
    echo "npx tsc --noEmit"
    return
  fi
  echo ""
}

detect_lint_cmd() {
  if [ -n "${QUALITY_LINT_CMD:-}" ]; then
    echo "$QUALITY_LINT_CMD"
    return
  fi
  if [ -f "$PROJECT_ROOT/package.json" ] && cd "$PROJECT_ROOT" && npm run | grep -qE ' lint\b'; then
    echo "npm run lint"
    return
  fi
  echo ""
}

detect_build_cmd() {
  if [ -n "${QUALITY_BUILD_CMD:-}" ]; then
    echo "$QUALITY_BUILD_CMD"
    return
  fi
  if [ -f "$PROJECT_ROOT/package.json" ] && cd "$PROJECT_ROOT" && npm run | grep -qE ' build\b'; then
    echo "npm run build"
    return
  fi
  echo ""
}

check_dependencies() {
  local cmd="${QUALITY_DEPENDENCY_CMD:-}"
  if [ -z "$cmd" ]; then
    # 没有 package.json 则跳过 Node.js 依赖检查
    if [ ! -f "$PROJECT_ROOT/package.json" ]; then
      echo -e "  ${YELLOW}⚠ 未发现 package.json，Node.js 依赖检查已跳过${NC}"
      RESULTS+=("${YELLOW}⚠ 依赖检查 跳过${NC}")
      SKIP_COUNT=$((SKIP_COUNT + 1))
      return
    fi
    if [ -d "$PROJECT_ROOT/node_modules" ]; then
      record_result "依赖检查" 0 ""
      return
    fi
    if [ -f "$PROJECT_ROOT/package-lock.json" ]; then
      cmd="npm ci"
    else
      cmd="npm install"
    fi
  fi

  local output=""
  local code=0
  output=$(cd "$PROJECT_ROOT" && bash -lc "$cmd" 2>&1) || code=$?
  record_result "依赖检查/安装" "$code" "$output"
}

run_typecheck() {
  local cmd
  cmd="$(detect_typecheck_cmd)"
  if [ -z "$cmd" ]; then
    echo -e "  ${YELLOW}⚠ 未发现 typecheck 命令，已跳过${NC}"
    RESULTS+=("${YELLOW}⚠ Typecheck 跳过${NC}")
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi
  run_cmd_step "TypeScript/Typecheck" "$cmd"
}

run_lint() {
  local cmd
  cmd="$(detect_lint_cmd)"
  if [ -z "$cmd" ]; then
    echo -e "  ${YELLOW}⚠ 未发现 lint 命令，已跳过${NC}"
    RESULTS+=("${YELLOW}⚠ Lint 跳过${NC}")
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi
  run_cmd_step "Lint" "$cmd"
}

run_build() {
  local cmd
  cmd="$(detect_build_cmd)"
  if [ -z "$cmd" ]; then
    echo -e "  ${YELLOW}⚠ 未发现 build 命令，已跳过${NC}"
    RESULTS+=("${YELLOW}⚠ Build 跳过${NC}")
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi
  run_cmd_step "Build" "$cmd"
}

# ────────────────────────── Python 检查 ──────────────────────────

has_python_files() {
  local exclude_pattern="${QUALITY_PYTHON_EXCLUDE_DIRS:-.venv,venv,__pycache__,old_files,node_modules}"
  local prune_args=()
  IFS=',' read -ra dirs <<< "$exclude_pattern"
  for d in "${dirs[@]}"; do
    prune_args+=(-name "$d" -o)
  done
  # Remove trailing -o
  unset 'prune_args[${#prune_args[@]}-1]'

  local found
  found=$(find "$PROJECT_ROOT" \( "${prune_args[@]}" \) -prune -o -name "*.py" -print -quit 2>/dev/null)
  [ -n "$found" ]
}

_python_find_args() {
  local exclude_pattern="${QUALITY_PYTHON_EXCLUDE_DIRS:-.venv,venv,__pycache__,old_files,node_modules}"
  local prune_args=()
  IFS=',' read -ra dirs <<< "$exclude_pattern"
  for d in "${dirs[@]}"; do
    prune_args+=(-name "$d" -o)
  done
  unset 'prune_args[${#prune_args[@]}-1]'
  echo "${prune_args[@]}"
}

run_python_syntax() {
  if ! has_python_files; then
    echo -e "  ${YELLOW}⚠ 未发现 Python 文件，语法检查已跳过${NC}"
    RESULTS+=("${YELLOW}⚠ Python 语法检查 跳过${NC}")
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi

  local python_cmd
  python_cmd="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
  if [ -z "$python_cmd" ]; then
    echo -e "  ${YELLOW}⚠ 未发现 python 解释器，已跳过${NC}"
    RESULTS+=("${YELLOW}⚠ Python 语法检查 跳过 (无解释器)${NC}")
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi

  echo -e "  ${BLUE}▶ Python 语法检查 (py_compile)...${NC}"

  local exclude_pattern="${QUALITY_PYTHON_EXCLUDE_DIRS:-.venv,venv,__pycache__,old_files,node_modules}"
  local prune_args=()
  IFS=',' read -ra dirs <<< "$exclude_pattern"
  for d in "${dirs[@]}"; do
    prune_args+=(-name "$d" -o)
  done
  unset 'prune_args[${#prune_args[@]}-1]'

  local errors=""
  local err_count=0
  while IFS= read -r pyfile; do
    local output=""
    output=$("$python_cmd" -m py_compile "$pyfile" 2>&1) || {
      errors+="  $pyfile: $output"$'\n'
      err_count=$((err_count + 1))
    }
  done < <(find "$PROJECT_ROOT" \( "${prune_args[@]}" \) -prune -o -name "*.py" -print 2>/dev/null)

  if [ "$err_count" -eq 0 ]; then
    record_result "Python 语法检查" 0 ""
  else
    record_result "Python 语法检查 ($err_count 个错误)" 1 "$errors"
  fi
}

run_python_unused_imports() {
  local tool_path="$SCRIPT_DIR/tools/check_errors/unused_imports.py"
  if [ ! -f "$tool_path" ]; then
    echo -e "  ${YELLOW}⚠ 未找到 unused_imports.py 工具，已跳过${NC}"
    RESULTS+=("${YELLOW}⚠ Python 未使用导入检查 跳过${NC}")
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi

  if ! has_python_files; then
    echo -e "  ${YELLOW}⚠ 未发现 Python 文件，未使用导入检查已跳过${NC}"
    RESULTS+=("${YELLOW}⚠ Python 未使用导入检查 跳过${NC}")
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi

  local python_cmd
  python_cmd="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
  if [ -z "$python_cmd" ]; then
    echo -e "  ${YELLOW}⚠ 未发现 python 解释器，已跳过${NC}"
    RESULTS+=("${YELLOW}⚠ Python 未使用导入检查 跳过 (无解释器)${NC}")
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi

  echo -e "  ${BLUE}▶ Python 未使用导入检查...${NC}"
  local output=""
  local code=0
  output=$("$python_cmd" "$tool_path" "$PROJECT_ROOT" 2>&1) || code=$?
  record_result "Python 未使用导入检查" "$code" "$output"
}

run_python_dunder_all() {
  local tool_path="$SCRIPT_DIR/tools/check_errors/validate_dunder_all.py"
  if [ ! -f "$tool_path" ]; then
    echo -e "  ${YELLOW}⚠ 未找到 validate_dunder_all.py 工具，已跳过${NC}"
    RESULTS+=("${YELLOW}⚠ Python __all__ 校验 跳过${NC}")
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi

  if ! has_python_files; then
    echo -e "  ${YELLOW}⚠ 未发现 Python 文件，__all__ 校验已跳过${NC}"
    RESULTS+=("${YELLOW}⚠ Python __all__ 校验 跳过${NC}")
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi

  local python_cmd
  python_cmd="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
  if [ -z "$python_cmd" ]; then
    echo -e "  ${YELLOW}⚠ 未发现 python 解释器，已跳过${NC}"
    RESULTS+=("${YELLOW}⚠ Python __all__ 校验 跳过 (无解释器)${NC}")
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi

  echo -e "  ${BLUE}▶ Python __all__ 导出校验...${NC}"

  # 收集所有 Python 应用根目录（含 requirements.txt / setup.py / pyproject.toml）
  local app_roots=()
  while IFS= read -r marker; do
    local root
    root="$(dirname "$marker")"
    app_roots+=("$root")
  done < <(find "$PROJECT_ROOT" \
    \( -name ".venv" -o -name "venv" -o -name "__pycache__" -o -name "node_modules" \) -prune \
    -o \( -name "requirements.txt" -o -name "setup.py" -o -name "pyproject.toml" \) -print 2>/dev/null)

  # 如果没找到任何应用根，回退到项目根目录
  if [ ${#app_roots[@]} -eq 0 ]; then
    app_roots=("$PROJECT_ROOT")
  fi

  local total_code=0
  local all_output=""
  for root in "${app_roots[@]}"; do
    local output=""
    local code=0
    output=$(cd "$root" && PYTHONPATH="$root:${PYTHONPATH:-}" "$python_cmd" "$tool_path" "$root" 2>&1) || code=$?
    if [ "$code" -ne 0 ]; then
      all_output+="[$root]"$'\n'"$output"$'\n'
    fi
    total_code=$((total_code + code))
  done

  record_result "Python __all__ 校验" "$total_code" "$all_output"
}

run_python_tests() {
  local python_cmd
  python_cmd="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
  if [ -z "$python_cmd" ]; then
    echo -e "  ${YELLOW}⚠ 未发现 python 解释器，已跳过${NC}"
    RESULTS+=("${YELLOW}⚠ Python 测试 跳过 (无解释器)${NC}")
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi

  local cmd="${QUALITY_PYTHON_TEST_CMD:-}"
  if [ -z "$cmd" ]; then
    # 查找测试文件
    local test_file
    test_file=$(find "$PROJECT_ROOT" \
      \( -name ".venv" -o -name "venv" -o -name "__pycache__" -o -name "node_modules" \) -prune \
      -o \( -name "test_*.py" -o -name "*_test.py" \) -print | head -1)
    if [ -z "$test_file" ]; then
      echo -e "  ${YELLOW}⚠ 未发现 Python 测试文件，已跳过${NC}"
      RESULTS+=("${YELLOW}⚠ Python 测试 跳过${NC}")
      SKIP_COUNT=$((SKIP_COUNT + 1))
      return
    fi

    # 收集所有包含测试文件的独立目录根（取含 __init__.py 或 tests/ 的最近父级）
    local test_roots=()
    while IFS= read -r tf; do
      local dir
      dir="$(dirname "$tf")"
      # 找到包含 tests 目录的应用根目录
      local app_root="$dir"
      while [ "$app_root" != "$PROJECT_ROOT" ] && [ "$app_root" != "/" ]; do
        local parent
        parent="$(dirname "$app_root")"
        # 如果父目录有 requirements.txt / setup.py / pyproject.toml，那就是应用根
        if [ -f "$parent/requirements.txt" ] || [ -f "$parent/setup.py" ] || [ -f "$parent/pyproject.toml" ]; then
          app_root="$parent"
          break
        fi
        app_root="$parent"
      done
      # 去重
      local already=false
      for r in "${test_roots[@]+"${test_roots[@]}"}"; do
        if [ "$r" = "$app_root" ]; then
          already=true
          break
        fi
      done
      if ! $already; then
        test_roots+=("$app_root")
      fi
    done < <(find "$PROJECT_ROOT" \
      \( -name ".venv" -o -name "venv" -o -name "__pycache__" -o -name "node_modules" \) -prune \
      -o \( -name "test_*.py" -o -name "*_test.py" \) -print 2>/dev/null)

    # 在每个测试根目录运行测试
    local total_code=0
    local all_output=""
    for root in "${test_roots[@]+"${test_roots[@]}"}"; do
      local output=""
      local code=0
      if "$python_cmd" -m pytest --version &>/dev/null; then
        output=$(cd "$root" && PYTHONPATH="$root:${PYTHONPATH:-}" "$python_cmd" -m pytest --tb=short -q 2>&1) || code=$?
      else
        # 找到包含 test_*.py 的子目录
        local test_start_dirs=()
        while IFS= read -r tf; do
          local td
          td="$(dirname "$tf")"
          local rel
          rel="$(python3 -c "import os; print(os.path.relpath('$td', '$root'))")"
          local dup=false
          for existing in "${test_start_dirs[@]+"${test_start_dirs[@]}"}"; do
            if [ "$existing" = "$rel" ]; then dup=true; break; fi
          done
          if ! $dup; then test_start_dirs+=("$rel"); fi
        done < <(find "$root" \( -name ".venv" -o -name "__pycache__" \) -prune -o -name "test_*.py" -print 2>/dev/null)

        for start_dir in "${test_start_dirs[@]+"${test_start_dirs[@]}"}"; do
          local sub_output=""
          sub_output=$(cd "$root" && PYTHONPATH="$root:${PYTHONPATH:-}" "$python_cmd" -m unittest discover -s "$start_dir" -p 'test_*.py' 2>&1) || code=$?
          output+="$sub_output"$'\n'
        done
      fi
      all_output+="[$root]"$'\n'"$output"$'\n'
      total_code=$((total_code + code))
    done

    record_result "Python 测试" "$total_code" "$all_output"
    return
  fi

  echo -e "  ${BLUE}▶ Python 测试...${NC}"
  run_cmd_step "Python 测试" "$cmd"
}

run_python_all() {
  if ! has_python_files; then
    echo -e "  ${YELLOW}⚠ 未发现 Python 文件，Python 检查已跳过${NC}"
    RESULTS+=("${YELLOW}⚠ Python 检查 跳过${NC}")
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi

  echo -e "\n${BOLD}${BLUE}── Python 质量检查 ──${NC}"
  run_python_syntax
  run_python_unused_imports
  run_python_dunder_all
  run_python_tests
}

# ────────────────────────────────────────────────────────────────

print_header() {
  echo ""
  echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}${CYAN}║     🔍 Generic Repo - 质量门禁检查               ║${NC}"
  echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${NC}"
  echo -e "${CYAN}  时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
  echo -e "${CYAN}  目录: ${PROJECT_ROOT}${NC}"
  echo ""
}

print_summary() {
  echo ""
  echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}${CYAN}║     📊 检查汇总报告                              ║${NC}"
  echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${NC}"
  echo ""

  for result in "${RESULTS[@]}"; do
    echo -e "  $result"
  done

  echo ""
  echo -e "  ${GREEN}通过: $PASS_COUNT${NC}  ${RED}失败: $FAIL_COUNT${NC}  ${YELLOW}跳过: $SKIP_COUNT${NC}"
  echo ""
}

main() {
  local mode="${1:-all}"
  print_header

  case "$mode" in
    --python)
      run_python_all
      ;;
    --lint)
      check_dependencies
      run_lint
      ;;
    --tsc)
      check_dependencies
      run_typecheck
      ;;
    --build)
      check_dependencies
      run_build
      ;;
    all|*)
      check_dependencies
      echo -e "\n${BOLD}${BLUE}── JS/TS 质量检查 ──${NC}"
      run_typecheck
      run_lint
      run_build
      run_python_all
      ;;
  esac

  print_summary
  exit "$FAIL_COUNT"
}

main "$@"
