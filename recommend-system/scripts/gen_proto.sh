#!/bin/bash
# gen_proto.sh - Proto 代码生成脚本
#
# 该脚本用于从 .proto 文件生成 Go 代码。
#
# 依赖安装：
#   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
#   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
#
# 使用方法：
#   chmod +x scripts/gen_proto.sh
#   ./scripts/gen_proto.sh

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 目录配置
PROTO_DIR="${PROJECT_ROOT}/proto"
OUT_DIR="${PROJECT_ROOT}"

log_info "Project root: ${PROJECT_ROOT}"
log_info "Proto directory: ${PROTO_DIR}"
log_info "Output directory: ${OUT_DIR}"

# 检查 protoc 是否安装
check_protoc() {
    if ! command -v protoc &> /dev/null; then
        log_error "protoc is not installed. Please install Protocol Buffers compiler."
        log_info "Installation guide:"
        log_info "  - macOS: brew install protobuf"
        log_info "  - Ubuntu: apt-get install -y protobuf-compiler"
        log_info "  - Windows: choco install protoc"
        exit 1
    fi
    log_info "protoc version: $(protoc --version)"
}

# 检查 Go 插件是否安装
check_go_plugins() {
    if ! command -v protoc-gen-go &> /dev/null; then
        log_warn "protoc-gen-go not found. Installing..."
        go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
    fi
    
    if ! command -v protoc-gen-go-grpc &> /dev/null; then
        log_warn "protoc-gen-go-grpc not found. Installing..."
        go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
    fi
    
    log_info "Go plugins ready"
}

# 确保输出目录存在
ensure_output_dirs() {
    mkdir -p "${OUT_DIR}/proto/recommend/v1"
    mkdir -p "${OUT_DIR}/proto/user/v1"
    mkdir -p "${OUT_DIR}/proto/item/v1"
}

# 生成推荐服务 Proto
generate_recommend_proto() {
    log_info "Generating recommend service proto..."
    
    protoc \
        --proto_path="${PROTO_DIR}" \
        --go_out="${OUT_DIR}" \
        --go_opt=paths=source_relative \
        --go-grpc_out="${OUT_DIR}" \
        --go-grpc_opt=paths=source_relative \
        "${PROTO_DIR}/recommend/v1/recommend.proto"
    
    log_info "Recommend service proto generated"
}

# 生成用户服务 Proto
generate_user_proto() {
    log_info "Generating user service proto..."
    
    protoc \
        --proto_path="${PROTO_DIR}" \
        --go_out="${OUT_DIR}" \
        --go_opt=paths=source_relative \
        --go-grpc_out="${OUT_DIR}" \
        --go-grpc_opt=paths=source_relative \
        "${PROTO_DIR}/user/v1/user.proto"
    
    log_info "User service proto generated"
}

# 生成物品服务 Proto
generate_item_proto() {
    log_info "Generating item service proto..."
    
    protoc \
        --proto_path="${PROTO_DIR}" \
        --go_out="${OUT_DIR}" \
        --go_opt=paths=source_relative \
        --go-grpc_out="${OUT_DIR}" \
        --go-grpc_opt=paths=source_relative \
        "${PROTO_DIR}/item/v1/item.proto"
    
    log_info "Item service proto generated"
}

# 生成所有 Proto
generate_all() {
    generate_recommend_proto
    generate_user_proto
    generate_item_proto
}

# 清理生成的文件
clean() {
    log_info "Cleaning generated files..."
    
    find "${OUT_DIR}/proto" -name "*.pb.go" -delete
    find "${OUT_DIR}/proto" -name "*_grpc.pb.go" -delete
    
    log_info "Clean completed"
}

# 显示帮助
show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  all       Generate all proto files (default)"
    echo "  recommend Generate recommend service proto"
    echo "  user      Generate user service proto"
    echo "  item      Generate item service proto"
    echo "  clean     Clean generated files"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0          # Generate all proto files"
    echo "  $0 all      # Generate all proto files"
    echo "  $0 user     # Generate only user service proto"
    echo "  $0 clean    # Clean all generated files"
}

# 主函数
main() {
    local command="${1:-all}"
    
    log_info "Starting proto generation..."
    
    check_protoc
    check_go_plugins
    ensure_output_dirs
    
    case "$command" in
        all)
            generate_all
            ;;
        recommend)
            generate_recommend_proto
            ;;
        user)
            generate_user_proto
            ;;
        item)
            generate_item_proto
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
    
    log_info "Proto generation completed successfully!"
}

# 运行主函数
main "$@"

