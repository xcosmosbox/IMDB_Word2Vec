#!/usr/bin/env bash
# =============================================================================
# 测试脚本
# 
# 用途: 运行 Go、Python 和前端的单元测试、集成测试
# 用法: ./test.sh [target] [options]
# 
# 目标:
#   go        - 运行 Go 测试
#   python    - 运行 Python 测试
#   frontend  - 运行前端测试
#   all       - 运行所有测试 (默认)
# 
# 选项:
#   --coverage         - 生成覆盖率报告
#   --integration      - 运行集成测试
#   --race             - 启用竞态检测 (Go)
#   --verbose          - 详细输出
#   --fail-fast        - 遇到失败立即停止
# =============================================================================

set -euo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# 默认配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
COVERAGE="${COVERAGE:-false}"
INTEGRATION="${INTEGRATION:-false}"
RACE="${RACE:-false}"
VERBOSE="${VERBOSE:-false}"
FAIL_FAST="${FAIL_FAST:-false}"

# 覆盖率输出目录
COVERAGE_DIR="${PROJECT_ROOT}/recommend-system/coverage"

# 解析命令行参数
parse_args() {
    TARGET="all"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            go|python|frontend|integration|all)
                TARGET="$1"
                shift
                ;;
            --coverage)
                COVERAGE="true"
                shift
                ;;
            --integration)
                INTEGRATION="true"
                shift
                ;;
            --race)
                RACE="true"
                shift
                ;;
            --verbose)
                VERBOSE="true"
                shift
                ;;
            --fail-fast)
                FAIL_FAST="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
用法: $(basename "$0") [target] [options]

目标:
  go          运行 Go 测试
  python      运行 Python 测试
  frontend    运行前端测试
  integration 运行集成测试
  all         运行所有测试 (默认)

选项:
  --coverage    生成覆盖率报告
  --integration 包含集成测试
  --race        启用竞态检测 (Go)
  --verbose     显示详细输出
  --fail-fast   遇到失败立即停止
  -h, --help    显示此帮助信息

环境变量:
  DATABASE_URL  PostgreSQL 连接字符串 (集成测试需要)
  REDIS_URL     Redis 连接字符串 (集成测试需要)

示例:
  $(basename "$0") go --coverage --race
  $(basename "$0") python --verbose
  $(basename "$0") all --coverage
EOF
}

# 初始化覆盖率目录
init_coverage() {
    if [[ "${COVERAGE}" == "true" ]]; then
        mkdir -p "${COVERAGE_DIR}"
        log_info "覆盖率报告目录: ${COVERAGE_DIR}"
    fi
}

# 运行 Go 测试
test_go() {
    log_info "运行 Go 测试..."
    
    cd "${PROJECT_ROOT}/recommend-system"
    
    # 构建测试参数
    GO_TEST_ARGS=("-v")
    
    if [[ "${RACE}" == "true" ]]; then
        GO_TEST_ARGS+=("-race")
    fi
    
    if [[ "${COVERAGE}" == "true" ]]; then
        GO_TEST_ARGS+=("-coverprofile=${COVERAGE_DIR}/go-coverage.out")
        GO_TEST_ARGS+=("-covermode=atomic")
    fi
    
    if [[ "${FAIL_FAST}" == "true" ]]; then
        GO_TEST_ARGS+=("-failfast")
    fi
    
    if [[ "${INTEGRATION}" == "true" ]]; then
        GO_TEST_ARGS+=("-tags=integration")
    else
        GO_TEST_ARGS+=("-short")
    fi
    
    # 运行测试
    log_info "  执行: go test ${GO_TEST_ARGS[*]} ./..."
    go test "${GO_TEST_ARGS[@]}" ./...
    
    # 生成 HTML 覆盖率报告
    if [[ "${COVERAGE}" == "true" ]]; then
        go tool cover -html="${COVERAGE_DIR}/go-coverage.out" -o "${COVERAGE_DIR}/go-coverage.html"
        
        # 计算覆盖率百分比
        COVERAGE_PCT=$(go tool cover -func="${COVERAGE_DIR}/go-coverage.out" | grep total | awk '{print $3}')
        log_success "  Go 覆盖率: ${COVERAGE_PCT}"
    fi
    
    log_success "Go 测试完成"
}

# 运行 Python 测试
test_python() {
    log_info "运行 Python 测试..."
    
    cd "${PROJECT_ROOT}/recommend-system/algorithm"
    
    # 检查 pytest
    if ! command -v pytest &> /dev/null; then
        log_info "  安装 pytest..."
        pip install --quiet pytest pytest-cov pytest-asyncio
    fi
    
    # 构建测试参数
    PYTEST_ARGS=()
    
    if [[ "${VERBOSE}" == "true" ]]; then
        PYTEST_ARGS+=("-v")
    fi
    
    if [[ "${COVERAGE}" == "true" ]]; then
        PYTEST_ARGS+=("--cov=.")
        PYTEST_ARGS+=("--cov-report=xml:${COVERAGE_DIR}/python-coverage.xml")
        PYTEST_ARGS+=("--cov-report=html:${COVERAGE_DIR}/python-coverage-html")
    fi
    
    if [[ "${FAIL_FAST}" == "true" ]]; then
        PYTEST_ARGS+=("-x")
    fi
    
    # 排除模型仓库目录
    PYTEST_ARGS+=("--ignore=serving/model_repository")
    
    # 运行测试
    log_info "  执行: pytest ${PYTEST_ARGS[*]}"
    pytest "${PYTEST_ARGS[@]}" || {
        if [[ "${FAIL_FAST}" == "true" ]]; then
            log_error "Python 测试失败"
            exit 1
        fi
    }
    
    log_success "Python 测试完成"
}

# 运行前端测试
test_frontend() {
    log_info "运行前端测试..."
    
    FRONTEND_APPS=("user-app" "admin")
    
    for app in "${FRONTEND_APPS[@]}"; do
        APP_DIR="${PROJECT_ROOT}/recommend-system/frontend/${app}"
        
        if [[ ! -d "${APP_DIR}" ]]; then
            log_warn "目录不存在: ${APP_DIR}"
            continue
        fi
        
        log_info "  测试 ${app}..."
        
        cd "${APP_DIR}"
        
        # 安装依赖
        if [[ ! -d "node_modules" ]]; then
            npm ci --silent
        fi
        
        # 构建测试参数
        NPM_TEST_ARGS=("--run")
        
        if [[ "${COVERAGE}" == "true" ]]; then
            NPM_TEST_ARGS+=("--coverage")
            
            # 设置覆盖率输出目录
            mkdir -p "${COVERAGE_DIR}/${app}"
        fi
        
        # 运行测试
        npm run test -- "${NPM_TEST_ARGS[@]}" || {
            if [[ "${FAIL_FAST}" == "true" ]]; then
                log_error "${app} 测试失败"
                exit 1
            fi
        }
        
        # 复制覆盖率报告
        if [[ "${COVERAGE}" == "true" ]] && [[ -d "coverage" ]]; then
            cp -r coverage/* "${COVERAGE_DIR}/${app}/" 2>/dev/null || true
        fi
        
        log_success "  ✓ ${app}"
    done
    
    log_success "前端测试完成"
}

# 运行集成测试
test_integration() {
    log_info "运行集成测试..."
    
    # 检查必要的环境变量
    if [[ -z "${DATABASE_URL:-}" ]]; then
        log_warn "DATABASE_URL 未设置，使用默认值"
        export DATABASE_URL="postgres://test:test@localhost:5432/recommend_test?sslmode=disable"
    fi
    
    if [[ -z "${REDIS_URL:-}" ]]; then
        log_warn "REDIS_URL 未设置，使用默认值"
        export REDIS_URL="redis://localhost:6379"
    fi
    
    cd "${PROJECT_ROOT}/recommend-system"
    
    # 运行 Go 集成测试
    log_info "  运行 Go 集成测试..."
    go test -v -tags=integration ./tests/integration/...
    
    # 运行 Python 集成测试
    log_info "  运行 Python 集成测试..."
    cd "${PROJECT_ROOT}/recommend-system/algorithm"
    pytest -v -m integration --ignore=serving/model_repository || true
    
    log_success "集成测试完成"
}

# 生成综合覆盖率报告
generate_coverage_report() {
    if [[ "${COVERAGE}" != "true" ]]; then
        return
    fi
    
    log_info "生成综合覆盖率报告..."
    
    # 创建报告索引
    cat > "${COVERAGE_DIR}/index.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>测试覆盖率报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        .report-link { display: block; margin: 10px 0; padding: 10px; background: #f5f5f5; text-decoration: none; color: #333; }
        .report-link:hover { background: #e5e5e5; }
    </style>
</head>
<body>
    <h1>测试覆盖率报告</h1>
    <p>生成时间: $(date)</p>
    <a class="report-link" href="go-coverage.html">Go 覆盖率报告</a>
    <a class="report-link" href="python-coverage-html/index.html">Python 覆盖率报告</a>
    <a class="report-link" href="user-app/index.html">User App 覆盖率报告</a>
    <a class="report-link" href="admin/index.html">Admin 覆盖率报告</a>
</body>
</html>
EOF

    log_success "覆盖率报告: ${COVERAGE_DIR}/index.html"
}

# 主函数
main() {
    parse_args "$@"
    
    log_info "=== 开始测试 ==="
    log_info "目标: ${TARGET}"
    log_info "覆盖率: ${COVERAGE}"
    log_info "集成测试: ${INTEGRATION}"
    
    START_TIME=$(date +%s)
    
    init_coverage
    
    FAILED=false
    
    case "${TARGET}" in
        go)
            test_go || FAILED=true
            ;;
        python)
            test_python || FAILED=true
            ;;
        frontend)
            test_frontend || FAILED=true
            ;;
        integration)
            test_integration || FAILED=true
            ;;
        all)
            test_go || FAILED=true
            test_python || FAILED=true
            test_frontend || FAILED=true
            if [[ "${INTEGRATION}" == "true" ]]; then
                test_integration || FAILED=true
            fi
            ;;
        *)
            log_error "未知目标: ${TARGET}"
            exit 1
            ;;
    esac
    
    generate_coverage_report
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [[ "${FAILED}" == "true" ]]; then
        log_error "=== 测试失败 (耗时: ${DURATION}s) ==="
        exit 1
    else
        log_success "=== 测试完成 (耗时: ${DURATION}s) ==="
    fi
}

main "$@"

