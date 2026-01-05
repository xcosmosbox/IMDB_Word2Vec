#!/bin/bash
# =============================================================================
# 性能测试运行脚本 (Linux/macOS)
#
# 使用方法:
#   ./run-tests.sh baseline              # 运行基线测试
#   ./run-tests.sh stress                # 运行压力测试
#   ./run-tests.sh spike                 # 运行峰值测试
#   ./run-tests.sh all                   # 运行所有测试
#   ./run-tests.sh locust baseline       # 使用 Locust 运行基线测试
#   TEST_ENV=dev ./run-tests.sh baseline # 指定环境
# =============================================================================

set -euo pipefail

# ============================================================================
# 配置
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
K6_DIR="${PROJECT_ROOT}/load/k6"
LOCUST_DIR="${PROJECT_ROOT}/load/locust"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
REPORTS_DIR="${PROJECT_ROOT}/reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 环境变量
BASE_URL="${BASE_URL:-http://localhost:8080}"
TEST_ENV="${TEST_ENV:-local}"
API_KEY="${API_KEY:-test-api-key}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 日志函数
# ============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%H:%M:%S') $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $(date '+%H:%M:%S') $1"
}

print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║           生成式推荐系统 - 性能测试套件                           ║"
    echo "║           Generative Recommendation System - Performance Test    ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# ============================================================================
# 检查依赖
# ============================================================================

check_k6() {
    if ! command -v k6 &> /dev/null; then
        log_error "K6 未安装。请先安装 K6:"
        echo "  macOS: brew install k6"
        echo "  Linux: sudo apt install k6 / sudo dnf install k6"
        echo "  Docker: docker pull grafana/k6"
        echo "  更多信息: https://k6.io/docs/getting-started/installation/"
        return 1
    fi
    log_info "K6 版本: $(k6 version)"
    return 0
}

check_locust() {
    if ! command -v locust &> /dev/null; then
        log_error "Locust 未安装。请先安装 Locust:"
        echo "  pip install locust"
        return 1
    fi
    log_info "Locust 版本: $(locust --version)"
    return 0
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装。"
        return 1
    fi
    log_info "Python 版本: $(python3 --version)"
    return 0
}

# ============================================================================
# 健康检查
# ============================================================================

health_check() {
    local url="${BASE_URL}/health"
    log_step "检查目标服务健康状态: ${url}"
    
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        if curl -sf "${url}" -o /dev/null --connect-timeout 5; then
            log_info "✓ 目标服务健康"
            return 0
        fi
        retry=$((retry + 1))
        log_warn "健康检查失败，重试 ${retry}/${max_retries}..."
        sleep 2
    done
    
    log_error "目标服务不可用: ${url}"
    return 1
}

# ============================================================================
# 创建目录
# ============================================================================

setup_directories() {
    mkdir -p "${RESULTS_DIR}"
    mkdir -p "${RESULTS_DIR}/k6"
    mkdir -p "${RESULTS_DIR}/locust"
    mkdir -p "${RESULTS_DIR}/reports"
    log_info "结果目录: ${RESULTS_DIR}"
}

# ============================================================================
# K6 测试运行器
# ============================================================================

run_k6_baseline() {
    log_step "运行 K6 基线测试..."
    
    local output_file="${RESULTS_DIR}/k6/baseline_${TIMESTAMP}.json"
    local log_file="${RESULTS_DIR}/k6/baseline_${TIMESTAMP}.log"
    
    k6 run \
        --out "json=${output_file}" \
        --env "BASE_URL=${BASE_URL}" \
        --env "TEST_ENV=${TEST_ENV}" \
        --env "API_KEY=${API_KEY}" \
        "${K6_DIR}/scenarios/baseline.js" \
        2>&1 | tee "${log_file}"
    
    log_info "基线测试完成"
    log_info "结果文件: ${output_file}"
}

run_k6_stress() {
    log_step "运行 K6 压力测试..."
    
    local output_file="${RESULTS_DIR}/k6/stress_${TIMESTAMP}.json"
    local log_file="${RESULTS_DIR}/k6/stress_${TIMESTAMP}.log"
    
    k6 run \
        --out "json=${output_file}" \
        --env "BASE_URL=${BASE_URL}" \
        --env "TEST_ENV=${TEST_ENV}" \
        --env "API_KEY=${API_KEY}" \
        "${K6_DIR}/scenarios/stress.js" \
        2>&1 | tee "${log_file}"
    
    log_info "压力测试完成"
    log_info "结果文件: ${output_file}"
}

run_k6_spike() {
    log_step "运行 K6 峰值测试..."
    
    local output_file="${RESULTS_DIR}/k6/spike_${TIMESTAMP}.json"
    local log_file="${RESULTS_DIR}/k6/spike_${TIMESTAMP}.log"
    
    k6 run \
        --out "json=${output_file}" \
        --env "BASE_URL=${BASE_URL}" \
        --env "TEST_ENV=${TEST_ENV}" \
        --env "API_KEY=${API_KEY}" \
        "${K6_DIR}/scenarios/spike.js" \
        2>&1 | tee "${log_file}"
    
    log_info "峰值测试完成"
    log_info "结果文件: ${output_file}"
}

# ============================================================================
# Locust 测试运行器
# ============================================================================

run_locust_baseline() {
    log_step "运行 Locust 基线测试..."
    
    local output_file="${RESULTS_DIR}/locust/baseline_${TIMESTAMP}"
    
    cd "${LOCUST_DIR}"
    
    locust \
        -f locustfile.py \
        --host "${BASE_URL}" \
        --headless \
        -u 100 \
        -r 10 \
        -t 5m \
        --html "${output_file}.html" \
        --csv "${output_file}" \
        2>&1 | tee "${output_file}.log"
    
    log_info "Locust 基线测试完成"
    log_info "结果文件: ${output_file}.html"
}

run_locust_stress() {
    log_step "运行 Locust 压力测试..."
    
    local output_file="${RESULTS_DIR}/locust/stress_${TIMESTAMP}"
    
    cd "${LOCUST_DIR}"
    
    locust \
        -f locustfile.py \
        --host "${BASE_URL}" \
        --headless \
        -u 1000 \
        -r 50 \
        -t 10m \
        --html "${output_file}.html" \
        --csv "${output_file}" \
        2>&1 | tee "${output_file}.log"
    
    log_info "Locust 压力测试完成"
    log_info "结果文件: ${output_file}.html"
}

run_locust_spike() {
    log_step "运行 Locust 峰值测试..."
    
    local output_file="${RESULTS_DIR}/locust/spike_${TIMESTAMP}"
    
    cd "${LOCUST_DIR}"
    
    locust \
        -f locustfile.py \
        --host "${BASE_URL}" \
        --headless \
        -u 5000 \
        -r 500 \
        -t 2m \
        --html "${output_file}.html" \
        --csv "${output_file}" \
        2>&1 | tee "${output_file}.log"
    
    log_info "Locust 峰值测试完成"
    log_info "结果文件: ${output_file}.html"
}

# ============================================================================
# 报告生成
# ============================================================================

generate_reports() {
    log_step "生成测试报告..."
    
    if ! check_python; then
        log_warn "跳过报告生成 (Python 不可用)"
        return
    fi
    
    # 查找所有 JSON 结果文件并生成 HTML 报告
    for json_file in "${RESULTS_DIR}/k6"/*.json; do
        if [[ -f "${json_file}" ]]; then
            local html_file="${json_file%.json}.html"
            log_info "生成报告: ${html_file}"
            
            python3 "${REPORTS_DIR}/generate-report.py" \
                "${json_file}" \
                -o "${html_file}" \
                2>/dev/null || log_warn "报告生成失败: ${json_file}"
        fi
    done
    
    log_info "报告生成完成"
}

# ============================================================================
# 显示帮助
# ============================================================================

show_help() {
    echo "使用方法: $0 [tool] <test_type>"
    echo ""
    echo "工具 (可选):"
    echo "  k6       使用 K6 运行测试 (默认)"
    echo "  locust   使用 Locust 运行测试"
    echo ""
    echo "测试类型:"
    echo "  baseline  基线测试 (100 RPS, 5分钟)"
    echo "  stress    压力测试 (逐步增加到 1000 RPS, 10分钟)"
    echo "  spike     峰值测试 (突发 5000 RPS, 2分钟)"
    echo "  all       运行所有测试"
    echo ""
    echo "环境变量:"
    echo "  BASE_URL   目标服务 URL (默认: http://localhost:8080)"
    echo "  TEST_ENV   测试环境 (local/dev/prod, 默认: local)"
    echo "  API_KEY    API 密钥 (默认: test-api-key)"
    echo "  RESULTS_DIR 结果输出目录"
    echo ""
    echo "示例:"
    echo "  $0 baseline                          # K6 基线测试"
    echo "  $0 locust stress                     # Locust 压力测试"
    echo "  BASE_URL=http://api.example.com $0 all  # 指定 URL 运行所有测试"
    echo ""
}

# ============================================================================
# 主函数
# ============================================================================

main() {
    print_banner
    
    # 解析参数
    local tool="k6"
    local test_type=""
    
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    if [[ $# -eq 1 ]]; then
        test_type="$1"
    elif [[ $# -eq 2 ]]; then
        tool="$1"
        test_type="$2"
    else
        show_help
        exit 1
    fi
    
    # 显示配置
    log_info "配置信息:"
    log_info "  工具: ${tool}"
    log_info "  测试类型: ${test_type}"
    log_info "  目标 URL: ${BASE_URL}"
    log_info "  环境: ${TEST_ENV}"
    echo ""
    
    # 检查依赖
    if [[ "${tool}" == "k6" ]]; then
        check_k6 || exit 1
    elif [[ "${tool}" == "locust" ]]; then
        check_locust || exit 1
    else
        log_error "未知工具: ${tool}"
        show_help
        exit 1
    fi
    
    # 设置目录
    setup_directories
    
    # 健康检查
    health_check || exit 1
    
    # 运行测试
    case "${test_type}" in
        baseline)
            if [[ "${tool}" == "k6" ]]; then
                run_k6_baseline
            else
                run_locust_baseline
            fi
            ;;
        stress)
            if [[ "${tool}" == "k6" ]]; then
                run_k6_stress
            else
                run_locust_stress
            fi
            ;;
        spike)
            if [[ "${tool}" == "k6" ]]; then
                run_k6_spike
            else
                run_locust_spike
            fi
            ;;
        all)
            if [[ "${tool}" == "k6" ]]; then
                run_k6_baseline
                run_k6_stress
                run_k6_spike
            else
                run_locust_baseline
                run_locust_stress
                run_locust_spike
            fi
            ;;
        *)
            log_error "未知测试类型: ${test_type}"
            show_help
            exit 1
            ;;
    esac
    
    # 生成报告
    generate_reports
    
    echo ""
    log_info "=========================================="
    log_info "  所有测试完成!"
    log_info "  结果目录: ${RESULTS_DIR}"
    log_info "=========================================="
}

# 运行主函数
main "$@"

