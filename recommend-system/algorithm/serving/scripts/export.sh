#!/bin/bash
# =============================================================================
# UGT 模型导出和部署脚本
# 
# 功能：
# 1. 导出 PyTorch 模型为 ONNX 格式
# 2. 使用 TensorRT 优化模型
# 3. 生成 Triton Inference Server 配置
# 4. 启动 Triton 服务
#
# Author: Person F (MLOps Engineer)
# =============================================================================

set -e  # 遇到错误立即退出

# =============================================================================
# 配置参数
# =============================================================================

# 模型配置
MODEL_NAME="${MODEL_NAME:-ugt_recommend}"
MODEL_VERSION="${MODEL_VERSION:-1}"
PRECISION="${PRECISION:-fp16}"  # fp32, fp16, int8
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"

# 路径配置
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./checkpoints/ugt_best.pt}"
ONNX_OUTPUT="${ONNX_OUTPUT:-./models/${MODEL_NAME}.onnx}"
TRT_OUTPUT="${TRT_OUTPUT:-./models/${MODEL_NAME}.plan}"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-./model_repository}"

# Triton 配置
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.01-py3}"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-8000}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8001}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-8002}"

# GPU 配置
GPU_IDS="${GPU_IDS:-0}"

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

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "命令 '$1' 未找到，请先安装"
        exit 1
    fi
}

# =============================================================================
# Step 1: 环境检查
# =============================================================================

check_environment() {
    log_info "检查环境..."
    
    # 检查 Python
    check_command python
    
    # 检查必要的 Python 包
    python -c "import torch" 2>/dev/null || {
        log_error "PyTorch 未安装"
        exit 1
    }
    
    # 检查 CUDA
    if ! nvidia-smi &> /dev/null; then
        log_error "NVIDIA GPU 驱动未安装或 GPU 不可用"
        exit 1
    fi
    
    log_success "环境检查通过"
}

# =============================================================================
# Step 2: 导出 ONNX 模型
# =============================================================================

export_onnx() {
    log_info "导出 ONNX 模型..."
    log_info "  输入: $CHECKPOINT_PATH"
    log_info "  输出: $ONNX_OUTPUT"
    
    # 创建输出目录
    mkdir -p "$(dirname "$ONNX_OUTPUT")"
    
    # 运行导出脚本
    python -c "
import sys
sys.path.insert(0, '.')
import torch
from algorithm.serving import ServingExporter, ExportConfig

# 加载模型
print('加载模型...')
# 注意：这里需要根据实际的模型加载方式修改
# model = torch.load('$CHECKPOINT_PATH')

# 使用 Mock 模型进行演示
class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 512)
    
    def forward(self, *args, **kwargs):
        batch_size = args[0].shape[0] if args else 1
        return (
            torch.randint(0, 16384, (batch_size, 50, 3)),
            torch.randn(batch_size, 50)
        )
    
    def generate(self, **kwargs):
        return [[(1, 2, 3)] * 50]

model = MockModel()

# 配置
config = ExportConfig(
    model_name='$MODEL_NAME',
    precision='$PRECISION',
    max_batch_size=$MAX_BATCH_SIZE,
    max_seq_length=$MAX_SEQ_LENGTH,
)

# 导出
exporter = ServingExporter(config)
onnx_path = exporter.export_onnx(model, '$ONNX_OUTPUT', config)
print(f'ONNX 模型已导出: {onnx_path}')
"
    
    if [ -f "$ONNX_OUTPUT" ]; then
        log_success "ONNX 模型导出成功: $ONNX_OUTPUT"
    else
        log_error "ONNX 模型导出失败"
        exit 1
    fi
}

# =============================================================================
# Step 3: TensorRT 优化
# =============================================================================

optimize_tensorrt() {
    log_info "TensorRT 优化..."
    log_info "  输入: $ONNX_OUTPUT"
    log_info "  输出: $TRT_OUTPUT"
    log_info "  精度: $PRECISION"
    
    # 创建输出目录
    mkdir -p "$(dirname "$TRT_OUTPUT")"
    
    # 使用 trtexec 工具（如果可用）
    if command -v trtexec &> /dev/null; then
        log_info "使用 trtexec 进行优化..."
        
        PRECISION_FLAG=""
        if [ "$PRECISION" = "fp16" ]; then
            PRECISION_FLAG="--fp16"
        elif [ "$PRECISION" = "int8" ]; then
            PRECISION_FLAG="--int8"
        fi
        
        trtexec \
            --onnx="$ONNX_OUTPUT" \
            --saveEngine="$TRT_OUTPUT" \
            $PRECISION_FLAG \
            --workspace=4096 \
            --minShapes=encoder_l1_ids:1x1,encoder_l2_ids:1x1,encoder_l3_ids:1x1,encoder_positions:1x1,encoder_token_types:1x1,encoder_mask:1x1 \
            --optShapes=encoder_l1_ids:32x512,encoder_l2_ids:32x512,encoder_l3_ids:32x512,encoder_positions:32x512,encoder_token_types:32x512,encoder_mask:32x512 \
            --maxShapes=encoder_l1_ids:${MAX_BATCH_SIZE}x${MAX_SEQ_LENGTH},encoder_l2_ids:${MAX_BATCH_SIZE}x${MAX_SEQ_LENGTH},encoder_l3_ids:${MAX_BATCH_SIZE}x${MAX_SEQ_LENGTH},encoder_positions:${MAX_BATCH_SIZE}x${MAX_SEQ_LENGTH},encoder_token_types:${MAX_BATCH_SIZE}x${MAX_SEQ_LENGTH},encoder_mask:${MAX_BATCH_SIZE}x${MAX_SEQ_LENGTH} \
            --verbose
    else
        log_info "使用 Python TensorRT API 进行优化..."
        
        python -c "
import sys
sys.path.insert(0, '.')
from algorithm.serving import TensorRTOptimizer, ExportConfig

config = ExportConfig(
    model_name='$MODEL_NAME',
    precision='$PRECISION',
    max_batch_size=$MAX_BATCH_SIZE,
    max_seq_length=$MAX_SEQ_LENGTH,
)

optimizer = TensorRTOptimizer(config)
engine_path = optimizer.optimize('$ONNX_OUTPUT', '$TRT_OUTPUT')
print(f'TensorRT 引擎已生成: {engine_path}')
"
    fi
    
    if [ -f "$TRT_OUTPUT" ]; then
        log_success "TensorRT 优化完成: $TRT_OUTPUT"
    else
        log_error "TensorRT 优化失败"
        exit 1
    fi
}

# =============================================================================
# Step 4: 生成 Triton 配置
# =============================================================================

generate_triton_config() {
    log_info "生成 Triton 配置..."
    log_info "  模型仓库: $MODEL_REPOSITORY"
    
    # 创建目录结构
    MODEL_DIR="${MODEL_REPOSITORY}/${MODEL_NAME}"
    VERSION_DIR="${MODEL_DIR}/${MODEL_VERSION}"
    
    mkdir -p "$VERSION_DIR"
    
    # 复制模型文件
    if [ -f "$TRT_OUTPUT" ]; then
        cp "$TRT_OUTPUT" "${VERSION_DIR}/model.plan"
        log_info "  复制 TensorRT 引擎到 ${VERSION_DIR}/model.plan"
    elif [ -f "$ONNX_OUTPUT" ]; then
        cp "$ONNX_OUTPUT" "${VERSION_DIR}/model.onnx"
        log_info "  复制 ONNX 模型到 ${VERSION_DIR}/model.onnx"
    fi
    
    # 生成 config.pbtxt
    python -c "
import sys
sys.path.insert(0, '.')
from algorithm.serving import TritonConfigGenerator, ExportConfig, TritonConfig

export_config = ExportConfig(
    model_name='$MODEL_NAME',
    precision='$PRECISION',
    max_batch_size=$MAX_BATCH_SIZE,
    max_seq_length=$MAX_SEQ_LENGTH,
)

triton_config = TritonConfig(
    platform='tensorrt_plan' if '$PRECISION' != 'fp32' else 'onnxruntime_onnx',
    instance_count=2,
    gpus=tuple(map(int, '$GPU_IDS'.split(','))),
)

generator = TritonConfigGenerator(export_config, triton_config)
config_path = generator.generate('$MODEL_REPOSITORY')
print(f'Triton 配置已生成: {config_path}')
"
    
    log_success "Triton 配置生成完成"
}

# =============================================================================
# Step 5: 启动 Triton Server
# =============================================================================

start_triton() {
    log_info "启动 Triton Inference Server..."
    log_info "  镜像: $TRITON_IMAGE"
    log_info "  模型仓库: $MODEL_REPOSITORY"
    log_info "  HTTP 端口: $TRITON_HTTP_PORT"
    log_info "  gRPC 端口: $TRITON_GRPC_PORT"
    
    # 检查 Docker
    check_command docker
    
    # 停止已有容器
    docker stop triton-ugt 2>/dev/null || true
    docker rm triton-ugt 2>/dev/null || true
    
    # 启动 Triton
    docker run -d \
        --name triton-ugt \
        --gpus "device=${GPU_IDS}" \
        -p ${TRITON_HTTP_PORT}:8000 \
        -p ${TRITON_GRPC_PORT}:8001 \
        -p ${TRITON_METRICS_PORT}:8002 \
        -v "$(realpath ${MODEL_REPOSITORY}):/models" \
        ${TRITON_IMAGE} \
        tritonserver --model-repository=/models
    
    log_info "等待 Triton 启动..."
    sleep 10
    
    # 检查服务状态
    if curl -s "http://localhost:${TRITON_HTTP_PORT}/v2/health/ready" | grep -q "true"; then
        log_success "Triton Server 启动成功"
        log_info "  HTTP: http://localhost:${TRITON_HTTP_PORT}"
        log_info "  gRPC: localhost:${TRITON_GRPC_PORT}"
        log_info "  Metrics: http://localhost:${TRITON_METRICS_PORT}/metrics"
    else
        log_error "Triton Server 启动失败"
        docker logs triton-ugt
        exit 1
    fi
}

# =============================================================================
# 主函数
# =============================================================================

main() {
    log_info "=============================================="
    log_info "UGT 模型部署流水线"
    log_info "=============================================="
    log_info "模型名称: $MODEL_NAME"
    log_info "精度: $PRECISION"
    log_info "最大批次: $MAX_BATCH_SIZE"
    log_info "最大序列长度: $MAX_SEQ_LENGTH"
    log_info "=============================================="
    
    # 解析命令行参数
    SKIP_EXPORT=false
    SKIP_TRT=false
    SKIP_TRITON=false
    START_SERVER=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-export)
                SKIP_EXPORT=true
                shift
                ;;
            --skip-trt)
                SKIP_TRT=true
                shift
                ;;
            --skip-triton-config)
                SKIP_TRITON=true
                shift
                ;;
            --start-server)
                START_SERVER=true
                shift
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --skip-export       跳过 ONNX 导出"
                echo "  --skip-trt          跳过 TensorRT 优化"
                echo "  --skip-triton-config 跳过 Triton 配置生成"
                echo "  --start-server      启动 Triton Server"
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                exit 1
                ;;
        esac
    done
    
    # 执行流水线
    check_environment
    
    if [ "$SKIP_EXPORT" = false ]; then
        export_onnx
    fi
    
    if [ "$SKIP_TRT" = false ]; then
        optimize_tensorrt
    fi
    
    if [ "$SKIP_TRITON" = false ]; then
        generate_triton_config
    fi
    
    if [ "$START_SERVER" = true ]; then
        start_triton
    fi
    
    log_info "=============================================="
    log_success "部署流水线完成"
    log_info "=============================================="
}

# 运行主函数
main "$@"

