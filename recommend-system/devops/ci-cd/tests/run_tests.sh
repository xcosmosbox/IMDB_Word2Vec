#!/usr/bin/env bash
# =============================================================================
# 测试运行脚本
# 
# 用途: 运行所有 CI/CD 相关的单元测试
# 用法: ./run_tests.sh [options]
# 
# 选项:
#   --bats     - 只运行 Bash 脚本测试
#   --python   - 只运行 Python 测试
#   --all      - 运行所有测试 (默认)
#   --verbose  - 详细输出
# =============================================================================

set -euo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# 默认配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_BATS="true"
RUN_PYTHON="true"
VERBOSE=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --bats)
            RUN_BATS="true"
            RUN_PYTHON="false"
            shift
            ;;
        --python)
            RUN_BATS="false"
            RUN_PYTHON="true"
            shift
            ;;
        --all)
            RUN_BATS="true"
            RUN_PYTHON="true"
            shift
            ;;
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        -h|--help)
            echo "用法: $(basename "$0") [--bats|--python|--all] [--verbose]"
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            exit 1
            ;;
    esac
done

cd "${SCRIPT_DIR}"

FAILED=false

# =============================================================================
# 运行 Bats 测试
# =============================================================================
if [[ "${RUN_BATS}" == "true" ]]; then
    log_info "运行 Bash 脚本测试..."
    
    # 检查 bats 是否安装
    if command -v bats &> /dev/null; then
        BATS_ARGS=()
        if [[ -n "${VERBOSE}" ]]; then
            BATS_ARGS+=("--tap")
        fi
        
        # 运行所有 .bats 文件
        for test_file in test_*.bats; do
            if [[ -f "${test_file}" ]]; then
                log_info "  运行 ${test_file}..."
                if bats "${BATS_ARGS[@]}" "${test_file}"; then
                    log_success "  ✓ ${test_file}"
                else
                    log_error "  ✗ ${test_file}"
                    FAILED=true
                fi
            fi
        done
    else
        log_error "bats 未安装。请安装: npm install -g bats 或 brew install bats-core"
        log_info "跳过 Bash 脚本测试"
    fi
fi

# =============================================================================
# 运行 Python 测试
# =============================================================================
if [[ "${RUN_PYTHON}" == "true" ]]; then
    log_info "运行 Python 测试..."
    
    # 检查 pytest 是否安装
    if command -v pytest &> /dev/null; then
        PYTEST_ARGS=()
        if [[ -n "${VERBOSE}" ]]; then
            PYTEST_ARGS+=("-v")
        fi
        
        # 安装依赖
        pip install -q pyyaml pytest
        
        # 运行 Python 测试
        if pytest "${PYTEST_ARGS[@]}" test_*.py; then
            log_success "Python 测试通过"
        else
            log_error "Python 测试失败"
            FAILED=true
        fi
    else
        log_error "pytest 未安装。请安装: pip install pytest"
        log_info "跳过 Python 测试"
    fi
fi

# =============================================================================
# 总结
# =============================================================================
echo ""
if [[ "${FAILED}" == "true" ]]; then
    log_error "部分测试失败"
    exit 1
else
    log_success "所有测试通过"
fi

