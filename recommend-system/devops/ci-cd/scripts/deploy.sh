#!/usr/bin/env bash
# =============================================================================
# 部署脚本
# 
# 用途: 将服务部署到 Kubernetes 集群
# 用法: ./deploy.sh [environment] [options]
# 
# 环境:
#   dev       - 开发环境
#   staging   - 预发布环境
#   prod      - 生产环境
# 
# 选项:
#   --version VERSION      - 指定部署版本
#   --services SERVICES    - 指定部署的服务 (逗号分隔)
#   --canary               - 金丝雀发布
#   --canary-weight N      - 金丝雀流量权重 (百分比)
#   --dry-run              - 试运行模式
#   --wait                 - 等待部署完成
#   --timeout DURATION     - 部署超时时间 (默认: 5m)
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
K8S_DIR="${PROJECT_ROOT}/recommend-system/devops/kubernetes"

# 环境配置
declare -A ENV_NAMESPACES=(
    [dev]="recommend-dev"
    [staging]="recommend-staging"
    [prod]="recommend-prod"
)

declare -A ENV_REPLICAS=(
    [dev]=1
    [staging]=2
    [prod]=3
)

# 部署参数
ENVIRONMENT=""
VERSION="${VERSION:-latest}"
SERVICES="${SERVICES:-recommend-service,user-service,item-service,ugt-inference}"
CANARY="${CANARY:-false}"
CANARY_WEIGHT="${CANARY_WEIGHT:-10}"
DRY_RUN="${DRY_RUN:-false}"
WAIT="${WAIT:-true}"
TIMEOUT="${TIMEOUT:-5m}"
REGISTRY="${REGISTRY:-localhost:5000}"

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            dev|staging|prod)
                ENVIRONMENT="$1"
                shift
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            --services)
                SERVICES="$2"
                shift 2
                ;;
            --canary)
                CANARY="true"
                shift
                ;;
            --canary-weight)
                CANARY_WEIGHT="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --wait)
                WAIT="true"
                shift
                ;;
            --no-wait)
                WAIT="false"
                shift
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
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
    
    if [[ -z "${ENVIRONMENT}" ]]; then
        log_error "必须指定环境 (dev, staging, prod)"
        exit 1
    fi
}

show_help() {
    cat << EOF
用法: $(basename "$0") <environment> [options]

环境:
  dev       开发环境
  staging   预发布环境
  prod      生产环境

选项:
  --version VERSION      指定部署版本 (默认: latest)
  --services SERVICES    指定部署的服务，逗号分隔 (默认: 所有服务)
  --canary               使用金丝雀发布
  --canary-weight N      金丝雀流量权重，百分比 (默认: 10)
  --dry-run              试运行模式，不实际部署
  --wait                 等待部署完成 (默认)
  --no-wait              不等待部署完成
  --timeout DURATION     部署超时时间 (默认: 5m)
  --registry URL         Docker 镜像仓库
  -h, --help             显示此帮助信息

环境变量:
  KUBECONFIG            Kubernetes 配置文件路径
  REGISTRY              Docker 镜像仓库地址

示例:
  $(basename "$0") dev --version v1.0.0
  $(basename "$0") prod --canary --canary-weight 5
  $(basename "$0") staging --services recommend-service,user-service
EOF
}

# 检查前置条件
check_prerequisites() {
    log_info "检查前置条件..."
    
    # 检查 kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl 未安装"
        exit 1
    fi
    
    # 检查集群连接
    if ! kubectl cluster-info &> /dev/null; then
        log_error "无法连接到 Kubernetes 集群"
        exit 1
    fi
    
    # 检查命名空间
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_info "创建命名空间: ${NAMESPACE}"
        kubectl create namespace "${NAMESPACE}"
    fi
    
    log_success "前置条件检查通过"
}

# 准备部署清单
prepare_manifests() {
    log_info "准备部署清单..."
    
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    REPLICAS="${ENV_REPLICAS[${ENVIRONMENT}]}"
    
    # 创建临时目录
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf ${TEMP_DIR}" EXIT
    
    # 使用 Kustomize 生成清单
    OVERLAY_DIR="${K8S_DIR}/overlays/${ENVIRONMENT}"
    
    if [[ -d "${OVERLAY_DIR}" ]]; then
        # 更新 Kustomization 中的镜像版本
        cd "${OVERLAY_DIR}"
        
        IFS=',' read -ra SERVICE_ARRAY <<< "${SERVICES}"
        for service in "${SERVICE_ARRAY[@]}"; do
            service=$(echo "${service}" | xargs)
            kustomize edit set image "${service}=${REGISTRY}/${service}:${VERSION}" 2>/dev/null || true
        done
        
        kustomize build . > "${TEMP_DIR}/manifests.yaml"
        cd -
    else
        # 使用基础清单
        cp "${K8S_DIR}/base/"*.yaml "${TEMP_DIR}/" 2>/dev/null || true
    fi
    
    echo "${TEMP_DIR}/manifests.yaml"
}

# 部署服务
deploy_services() {
    log_info "部署服务到 ${ENVIRONMENT} 环境..."
    
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    
    IFS=',' read -ra SERVICE_ARRAY <<< "${SERVICES}"
    
    for service in "${SERVICE_ARRAY[@]}"; do
        service=$(echo "${service}" | xargs)
        IMAGE="${REGISTRY}/${service}:${VERSION}"
        
        log_info "  部署 ${service} (${IMAGE})..."
        
        KUBECTL_ARGS=()
        if [[ "${DRY_RUN}" == "true" ]]; then
            KUBECTL_ARGS+=("--dry-run=client")
        fi
        
        # 更新 Deployment 镜像
        kubectl set image "deployment/${service}" \
            "${service}=${IMAGE}" \
            -n "${NAMESPACE}" \
            "${KUBECTL_ARGS[@]}" 2>/dev/null || {
                log_warn "  Deployment ${service} 不存在，尝试创建..."
                create_deployment "${service}" "${IMAGE}"
            }
        
        log_success "  ✓ ${service}"
    done
}

# 创建 Deployment
create_deployment() {
    local service=$1
    local image=$2
    local namespace="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    local replicas="${ENV_REPLICAS[${ENVIRONMENT}]}"
    
    # 生成基础 Deployment
    cat << EOF | kubectl apply -f - ${DRY_RUN:+--dry-run=client}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${service}
  namespace: ${namespace}
  labels:
    app: ${service}
    version: ${VERSION}
spec:
  replicas: ${replicas}
  selector:
    matchLabels:
      app: ${service}
  template:
    metadata:
      labels:
        app: ${service}
        version: ${VERSION}
    spec:
      containers:
      - name: ${service}
        image: ${image}
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
EOF
}

# 金丝雀部署
deploy_canary() {
    log_info "执行金丝雀部署 (权重: ${CANARY_WEIGHT}%)..."
    
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    
    IFS=',' read -ra SERVICE_ARRAY <<< "${SERVICES}"
    
    for service in "${SERVICE_ARRAY[@]}"; do
        service=$(echo "${service}" | xargs)
        IMAGE="${REGISTRY}/${service}:${VERSION}"
        CANARY_NAME="${service}-canary"
        
        log_info "  创建金丝雀 ${CANARY_NAME}..."
        
        # 创建金丝雀 Deployment
        cat << EOF | kubectl apply -f - ${DRY_RUN:+--dry-run=client}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${CANARY_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: ${service}
    version: ${VERSION}
    canary: "true"
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${service}
      canary: "true"
  template:
    metadata:
      labels:
        app: ${service}
        version: ${VERSION}
        canary: "true"
    spec:
      containers:
      - name: ${service}
        image: ${IMAGE}
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
EOF
        
        log_success "  ✓ ${CANARY_NAME}"
    done
}

# 等待部署完成
wait_for_rollout() {
    if [[ "${WAIT}" != "true" ]]; then
        return
    fi
    
    log_info "等待部署完成 (超时: ${TIMEOUT})..."
    
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    
    IFS=',' read -ra SERVICE_ARRAY <<< "${SERVICES}"
    
    FAILED=false
    for service in "${SERVICE_ARRAY[@]}"; do
        service=$(echo "${service}" | xargs)
        
        DEPLOYMENT_NAME="${service}"
        if [[ "${CANARY}" == "true" ]]; then
            DEPLOYMENT_NAME="${service}-canary"
        fi
        
        log_info "  等待 ${DEPLOYMENT_NAME}..."
        
        if ! kubectl rollout status "deployment/${DEPLOYMENT_NAME}" \
            -n "${NAMESPACE}" \
            --timeout="${TIMEOUT}"; then
            log_error "  ✗ ${DEPLOYMENT_NAME} 部署超时"
            FAILED=true
        else
            log_success "  ✓ ${DEPLOYMENT_NAME}"
        fi
    done
    
    if [[ "${FAILED}" == "true" ]]; then
        log_error "部分服务部署失败"
        exit 1
    fi
}

# 验证部署
verify_deployment() {
    log_info "验证部署..."
    
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    
    # 检查 Pod 状态
    UNHEALTHY_PODS=$(kubectl get pods -n "${NAMESPACE}" \
        -o jsonpath='{.items[?(@.status.phase!="Running")].metadata.name}')
    
    if [[ -n "${UNHEALTHY_PODS}" ]]; then
        log_warn "发现不健康的 Pod: ${UNHEALTHY_PODS}"
    fi
    
    # 显示部署状态
    log_info "当前部署状态:"
    kubectl get deployments -n "${NAMESPACE}" -o wide
    
    log_info "Pod 状态:"
    kubectl get pods -n "${NAMESPACE}" -o wide
    
    log_success "部署验证完成"
}

# 记录部署信息
record_deployment() {
    log_info "记录部署信息..."
    
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    
    # 创建部署记录 ConfigMap
    RECORD=$(cat << EOF
{
    "version": "${VERSION}",
    "environment": "${ENVIRONMENT}",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "services": "${SERVICES}",
    "canary": "${CANARY}",
    "deployer": "${USER:-unknown}"
}
EOF
)
    
    kubectl create configmap deployment-record \
        --from-literal=record.json="${RECORD}" \
        -n "${NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f - ${DRY_RUN:+--dry-run=client}
}

# 主函数
main() {
    parse_args "$@"
    
    log_info "=== 开始部署 ==="
    log_info "环境: ${ENVIRONMENT}"
    log_info "版本: ${VERSION}"
    log_info "服务: ${SERVICES}"
    log_info "金丝雀: ${CANARY}"
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warn "试运行模式 - 不会实际部署"
    fi
    
    START_TIME=$(date +%s)
    
    check_prerequisites
    
    if [[ "${CANARY}" == "true" ]]; then
        deploy_canary
    else
        deploy_services
    fi
    
    wait_for_rollout
    verify_deployment
    record_deployment
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    log_success "=== 部署完成 (耗时: ${DURATION}s) ==="
}

main "$@"

