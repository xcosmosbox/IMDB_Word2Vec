#!/bin/bash
# =============================================================================
# UGT 模型性能基准测试脚本
# 
# 功能：
# 1. 对 Triton 服务进行性能测试
# 2. 测试不同批次大小和序列长度
# 3. 生成性能报告
#
# Author: Person F (MLOps Engineer)
# =============================================================================

set -e

# =============================================================================
# 配置参数
# =============================================================================

# Triton 配置
TRITON_URL="${TRITON_URL:-localhost:8001}"
MODEL_NAME="${MODEL_NAME:-ugt_recommend}"

# 测试配置
NUM_WARMUP="${NUM_WARMUP:-100}"
NUM_REQUESTS="${NUM_REQUESTS:-10000}"
CONCURRENCY="${CONCURRENCY:-1}"

# 测试的批次大小和序列长度
BATCH_SIZES="${BATCH_SIZES:-1 8 16 32}"
SEQ_LENGTHS="${SEQ_LENGTHS:-32 64 128 256 512}"

# 输出配置
OUTPUT_DIR="${OUTPUT_DIR:-./benchmark_results}"
REPORT_FILE="${OUTPUT_DIR}/benchmark_report_$(date +%Y%m%d_%H%M%S).txt"

# =============================================================================
# 辅助函数
# =============================================================================

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_success() {
    echo "[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# =============================================================================
# 检查服务状态
# =============================================================================

check_triton_health() {
    log_info "检查 Triton 服务状态..."
    
    # 提取主机和端口
    HOST=$(echo $TRITON_URL | cut -d: -f1)
    PORT=$(echo $TRITON_URL | cut -d: -f2)
    HTTP_PORT=$((PORT - 1))  # gRPC 端口通常是 HTTP 端口 + 1
    
    if curl -s "http://${HOST}:${HTTP_PORT}/v2/health/ready" | grep -q "true"; then
        log_success "Triton 服务就绪"
    else
        log_error "Triton 服务不可用: http://${HOST}:${HTTP_PORT}"
        exit 1
    fi
    
    # 检查模型是否加载
    if curl -s "http://${HOST}:${HTTP_PORT}/v2/models/${MODEL_NAME}/ready" | grep -q "true"; then
        log_success "模型 ${MODEL_NAME} 已加载"
    else
        log_error "模型 ${MODEL_NAME} 未加载"
        exit 1
    fi
}

# =============================================================================
# 使用 Python 进行基准测试
# =============================================================================

run_python_benchmark() {
    log_info "使用 Python 进行基准测试..."
    
    mkdir -p "$OUTPUT_DIR"
    
    python -c "
import sys
sys.path.insert(0, '.')
import json
from datetime import datetime
from algorithm.serving import TritonBenchmark, BenchmarkConfig, ExportConfig

# 配置
benchmark_config = BenchmarkConfig(
    triton_url='$TRITON_URL',
    num_warmup_requests=$NUM_WARMUP,
    num_requests=$NUM_REQUESTS,
    concurrency=$CONCURRENCY,
    test_batch_sizes=tuple(map(int, '$BATCH_SIZES'.split())),
    test_seq_lengths=tuple(map(int, '$SEQ_LENGTHS'.split())),
)

export_config = ExportConfig(model_name='$MODEL_NAME')

# 创建基准测试器
benchmark = TritonBenchmark(benchmark_config, export_config)

print('=' * 80)
print('UGT 模型性能基准测试')
print('=' * 80)
print(f'Triton URL: $TRITON_URL')
print(f'模型名称: $MODEL_NAME')
print(f'预热请求: $NUM_WARMUP')
print(f'测试请求: $NUM_REQUESTS')
print(f'并发数: $CONCURRENCY')
print('=' * 80)

# 运行完整的参数扫描测试
try:
    results = benchmark.run_sweep('$TRITON_URL', '$MODEL_NAME')
    
    # 生成报告
    report = benchmark.generate_report(results, '$REPORT_FILE')
    print(report)
    
    # 保存 JSON 结果
    json_file = '$OUTPUT_DIR/benchmark_results_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'
    with open(json_file, 'w') as f:
        json_results = {k: v.to_dict() for k, v in results.items()}
        json.dump(json_results, f, indent=2)
    print(f'\n结果已保存到: {json_file}')
    print(f'报告已保存到: $REPORT_FILE')

except Exception as e:
    print(f'基准测试失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
}

# =============================================================================
# 使用 perf_analyzer 进行基准测试（如果可用）
# =============================================================================

run_perf_analyzer() {
    if ! command -v perf_analyzer &> /dev/null; then
        log_info "perf_analyzer 不可用，跳过"
        return
    fi
    
    log_info "使用 perf_analyzer 进行基准测试..."
    
    mkdir -p "$OUTPUT_DIR"
    
    for batch_size in $BATCH_SIZES; do
        log_info "测试批次大小: $batch_size"
        
        OUTPUT_FILE="${OUTPUT_DIR}/perf_bs${batch_size}_$(date +%Y%m%d_%H%M%S).json"
        
        perf_analyzer \
            -m "$MODEL_NAME" \
            -u "$TRITON_URL" \
            --concurrency-range 1:${CONCURRENCY}:1 \
            -b "$batch_size" \
            --percentile=50,90,95,99 \
            -f "$OUTPUT_FILE" \
            || log_error "批次大小 $batch_size 测试失败"
    done
}

# =============================================================================
# 生成汇总报告
# =============================================================================

generate_summary() {
    log_info "生成汇总报告..."
    
    {
        echo "=============================================="
        echo "UGT 模型性能测试汇总报告"
        echo "=============================================="
        echo "测试时间: $(date)"
        echo "Triton URL: $TRITON_URL"
        echo "模型名称: $MODEL_NAME"
        echo ""
        echo "测试配置:"
        echo "  预热请求: $NUM_WARMUP"
        echo "  测试请求: $NUM_REQUESTS"
        echo "  并发数: $CONCURRENCY"
        echo "  批次大小: $BATCH_SIZES"
        echo "  序列长度: $SEQ_LENGTHS"
        echo ""
        echo "=============================================="
        echo ""
    } >> "$REPORT_FILE"
    
    log_success "报告已保存到: $REPORT_FILE"
}

# =============================================================================
# 主函数
# =============================================================================

main() {
    log_info "=============================================="
    log_info "UGT 模型性能基准测试"
    log_info "=============================================="
    
    # 解析命令行参数
    USE_PERF_ANALYZER=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --perf-analyzer)
                USE_PERF_ANALYZER=true
                shift
                ;;
            --url)
                TRITON_URL="$2"
                shift 2
                ;;
            --model)
                MODEL_NAME="$2"
                shift 2
                ;;
            --requests)
                NUM_REQUESTS="$2"
                shift 2
                ;;
            --concurrency)
                CONCURRENCY="$2"
                shift 2
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --url URL           Triton gRPC URL (默认: localhost:8001)"
                echo "  --model NAME        模型名称 (默认: ugt_recommend)"
                echo "  --requests N        测试请求数 (默认: 10000)"
                echo "  --concurrency N     并发数 (默认: 1)"
                echo "  --perf-analyzer     使用 Triton perf_analyzer"
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                exit 1
                ;;
        esac
    done
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    
    # 检查服务
    check_triton_health
    
    # 运行基准测试
    if [ "$USE_PERF_ANALYZER" = true ]; then
        run_perf_analyzer
    else
        run_python_benchmark
    fi
    
    # 生成汇总
    generate_summary
    
    log_info "=============================================="
    log_success "基准测试完成"
    log_info "结果目录: $OUTPUT_DIR"
    log_info "=============================================="
}

# 运行主函数
main "$@"

