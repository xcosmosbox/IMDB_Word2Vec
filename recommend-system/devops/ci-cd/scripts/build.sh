#!/usr/bin/env bash
# =============================================================================
# 构建脚本
# 
# 用途: 构建 Go 服务、Python 包、前端应用和 Docker 镜像
# 用法: ./build.sh [target] [options]
# 
# 目标:
#   go        - 构建 Go 服务
#   python    - 构建 Python 包
#   frontend  - 构建前端应用
#   docker    - 构建 Docker 镜像
#   all       - 构建所有 (默认)
# 
# 选项:
#   --version VERSION   - 指定版本号
#   --registry URL      - Docker 镜像仓库地址
#   --push              - 推送 Docker 镜像
#   --platform PLATFORM - Docker 目标平台 (linux/amd64, linux/arm64)
#   --no-cache          - 不使用缓存
#   --verbose           - 详细输出
# =============================================================================

set -euo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# 默认配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VERSION="${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo "dev")}"
REGISTRY="${REGISTRY:-localhost:5000}"
PUSH="${PUSH:-false}"
PLATFORM="${PLATFORM:-linux/amd64}"
NO_CACHE="${NO_CACHE:-false}"
VERBOSE="${VERBOSE:-false}"

# Go 构建配置
GO_SERVICES=("recommend-service" "user-service" "item-service")
GO_OUTPUT_DIR="${PROJECT_ROOT}/recommend-system/bin"
GOOS="${GOOS:-linux}"
GOARCH="${GOARCH:-amd64}"

# Docker 配置
DOCKER_IMAGES=(
    "recommend-service:deployments/docker/Dockerfile"
    "user-service:deployments/docker/Dockerfile.user"
    "item-service:deployments/docker/Dockerfile.item"
    "ugt-inference:algorithm/Dockerfile"
    "user-app:frontend/user-app/Dockerfile"
    "admin-app:frontend/admin/Dockerfile"
)

# 解析命令行参数
parse_args() {
    TARGET="all"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            go|python|frontend|docker|all)
                TARGET="$1"
                shift
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --push)
                PUSH="true"
                shift
                ;;
            --platform)
                PLATFORM="$2"
                shift 2
                ;;
            --no-cache)
                NO_CACHE="true"
                shift
                ;;
            --verbose)
                VERBOSE="true"
                set -x
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
  go        构建 Go 服务
  python    构建 Python 包
  frontend  构建前端应用
  docker    构建 Docker 镜像
  all       构建所有 (默认)

选项:
  --version VERSION   指定版本号 (默认: git tag)
  --registry URL      Docker 镜像仓库地址 (默认: localhost:5000)
  --push              推送 Docker 镜像到仓库
  --platform PLATFORM Docker 目标平台 (默认: linux/amd64)
  --no-cache          不使用构建缓存
  --verbose           显示详细输出
  -h, --help          显示此帮助信息

环境变量:
  VERSION             版本号
  REGISTRY            Docker 镜像仓库
  GOOS                Go 目标操作系统
  GOARCH              Go 目标架构

示例:
  $(basename "$0") go --version v1.0.0
  $(basename "$0") docker --registry myregistry.com --push
  $(basename "$0") all --version v1.0.0 --push
EOF
}

# 构建 Go 服务
build_go() {
    log_info "构建 Go 服务..."
    
    cd "${PROJECT_ROOT}/recommend-system"
    
    # 创建输出目录
    mkdir -p "${GO_OUTPUT_DIR}"
    
    # 设置构建参数
    LDFLAGS="-s -w"
    LDFLAGS="${LDFLAGS} -X main.Version=${VERSION}"
    LDFLAGS="${LDFLAGS} -X main.BuildTime=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    LDFLAGS="${LDFLAGS} -X main.GitCommit=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    
    # 构建每个服务
    for service in "${GO_SERVICES[@]}"; do
        log_info "  构建 ${service}..."
        
        OUTPUT="${GO_OUTPUT_DIR}/${service}"
        if [[ "${GOOS}" == "windows" ]]; then
            OUTPUT="${OUTPUT}.exe"
        fi
        
        CGO_ENABLED=0 GOOS="${GOOS}" GOARCH="${GOARCH}" \
            go build \
            -ldflags="${LDFLAGS}" \
            -trimpath \
            -o "${OUTPUT}" \
            "./cmd/${service}"
        
        log_success "  ✓ ${service} -> ${OUTPUT}"
    done
    
    log_success "Go 服务构建完成"
}

# 构建 Python 包
build_python() {
    log_info "构建 Python 包..."
    
    cd "${PROJECT_ROOT}/recommend-system/algorithm"
    
    # 检查 Python 环境
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    # 安装构建依赖
    pip install --quiet build wheel
    
    # 构建 wheel 包
    python -m build --wheel --outdir dist/
    
    log_success "Python 包构建完成: $(ls dist/*.whl)"
}

# 构建前端应用
build_frontend() {
    log_info "构建前端应用..."
    
    FRONTEND_APPS=("user-app" "admin")
    
    for app in "${FRONTEND_APPS[@]}"; do
        APP_DIR="${PROJECT_ROOT}/recommend-system/frontend/${app}"
        
        if [[ ! -d "${APP_DIR}" ]]; then
            log_warn "目录不存在: ${APP_DIR}"
            continue
        fi
        
        log_info "  构建 ${app}..."
        
        cd "${APP_DIR}"
        
        # 安装依赖
        if [[ -f "package-lock.json" ]]; then
            npm ci --silent
        else
            npm install --silent
        fi
        
        # 设置环境变量
        export VITE_APP_VERSION="${VERSION}"
        
        # 构建
        npm run build
        
        log_success "  ✓ ${app} -> ${APP_DIR}/dist"
    done
    
    log_success "前端应用构建完成"
}

# 构建 Docker 镜像
build_docker() {
    log_info "构建 Docker 镜像..."
    
    cd "${PROJECT_ROOT}/recommend-system"
    
    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装"
        exit 1
    fi
    
    # 构建参数
    BUILD_ARGS=()
    BUILD_ARGS+=("--build-arg" "VERSION=${VERSION}")
    BUILD_ARGS+=("--build-arg" "BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)")
    
    if [[ "${NO_CACHE}" == "true" ]]; then
        BUILD_ARGS+=("--no-cache")
    fi
    
    # 检查是否使用 buildx
    USE_BUILDX=false
    if docker buildx version &> /dev/null; then
        USE_BUILDX=true
        # 创建或使用 builder
        docker buildx create --name multiarch --use 2>/dev/null || docker buildx use multiarch
    fi
    
    # 构建每个镜像
    for image_config in "${DOCKER_IMAGES[@]}"; do
        IFS=':' read -r image_name dockerfile <<< "${image_config}"
        
        if [[ ! -f "${dockerfile}" ]]; then
            log_warn "Dockerfile 不存在: ${dockerfile}, 跳过 ${image_name}"
            continue
        fi
        
        IMAGE_TAG="${REGISTRY}/${image_name}:${VERSION}"
        IMAGE_TAG_LATEST="${REGISTRY}/${image_name}:latest"
        
        log_info "  构建 ${image_name}..."
        
        if [[ "${USE_BUILDX}" == "true" ]]; then
            BUILDX_ARGS=("--platform" "${PLATFORM}")
            
            if [[ "${PUSH}" == "true" ]]; then
                BUILDX_ARGS+=("--push")
            else
                BUILDX_ARGS+=("--load")
            fi
            
            docker buildx build \
                "${BUILD_ARGS[@]}" \
                "${BUILDX_ARGS[@]}" \
                -t "${IMAGE_TAG}" \
                -t "${IMAGE_TAG_LATEST}" \
                -f "${dockerfile}" \
                .
        else
            docker build \
                "${BUILD_ARGS[@]}" \
                -t "${IMAGE_TAG}" \
                -t "${IMAGE_TAG_LATEST}" \
                -f "${dockerfile}" \
                .
            
            if [[ "${PUSH}" == "true" ]]; then
                docker push "${IMAGE_TAG}"
                docker push "${IMAGE_TAG_LATEST}"
            fi
        fi
        
        log_success "  ✓ ${IMAGE_TAG}"
    done
    
    log_success "Docker 镜像构建完成"
}

# 主函数
main() {
    parse_args "$@"
    
    log_info "=== 开始构建 ==="
    log_info "版本: ${VERSION}"
    log_info "目标: ${TARGET}"
    log_info "镜像仓库: ${REGISTRY}"
    
    START_TIME=$(date +%s)
    
    case "${TARGET}" in
        go)
            build_go
            ;;
        python)
            build_python
            ;;
        frontend)
            build_frontend
            ;;
        docker)
            build_docker
            ;;
        all)
            build_go
            build_python
            build_frontend
            build_docker
            ;;
        *)
            log_error "未知目标: ${TARGET}"
            exit 1
            ;;
    esac
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    log_success "=== 构建完成 (耗时: ${DURATION}s) ==="
}

main "$@"

