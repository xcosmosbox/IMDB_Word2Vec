#!/usr/bin/env bash
# =============================================================================
# 回滚脚本
# 
# 用途: 将服务回滚到之前的版本
# 用法: ./rollback.sh [environment] [options]
# 
# 环境:
#   dev       - 开发环境
#   staging   - 预发布环境
#   prod      - 生产环境
# 
# 选项:
#   --services SERVICES    - 指定回滚的服务 (逗号分隔)
#   --revision N           - 回滚到指定的修订版本
#   --version VERSION      - 回滚到指定的版本标签
#   --history              - 显示部署历史
#   --dry-run              - 试运行模式
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

# 环境配置
declare -A ENV_NAMESPACES=(
    [dev]="recommend-dev"
    [staging]="recommend-staging"
    [prod]="recommend-prod"
)

# 回滚参数
ENVIRONMENT=""
SERVICES="${SERVICES:-recommend-service,user-service,item-service,ugt-inference}"
REVISION=""
TARGET_VERSION=""
SHOW_HISTORY="${SHOW_HISTORY:-false}"
DRY_RUN="${DRY_RUN:-false}"
REGISTRY="${REGISTRY:-localhost:5000}"

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            dev|staging|prod)
                ENVIRONMENT="$1"
                shift
                ;;
            --services)
                SERVICES="$2"
                shift 2
                ;;
            --revision)
                REVISION="$2"
                shift 2
                ;;
            --version)
                TARGET_VERSION="$2"
                shift 2
                ;;
            --history)
                SHOW_HISTORY="true"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
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
  --services SERVICES    指定回滚的服务，逗号分隔 (默认: 所有服务)
  --revision N           回滚到指定的修订版本号
  --version VERSION      回滚到指定的版本标签
  --history              显示部署历史
  --dry-run              试运行模式，不实际回滚
  -h, --help             显示此帮助信息

环境变量:
  KUBECONFIG            Kubernetes 配置文件路径
  REGISTRY              Docker 镜像仓库地址

示例:
  $(basename "$0") prod --history
  $(basename "$0") prod --revision 2
  $(basename "$0") staging --version v1.0.0
  $(basename "$0") dev --services recommend-service
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
        log_error "命名空间 ${NAMESPACE} 不存在"
        exit 1
    fi
    
    log_success "前置条件检查通过"
}

# 显示部署历史
show_deployment_history() {
    log_info "显示部署历史..."
    
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    
    IFS=',' read -ra SERVICE_ARRAY <<< "${SERVICES}"
    
    for service in "${SERVICE_ARRAY[@]}"; do
        service=$(echo "${service}" | xargs)
        
        echo ""
        echo "=== ${service} 部署历史 ==="
        echo ""
        
        # 获取部署历史
        kubectl rollout history "deployment/${service}" -n "${NAMESPACE}" 2>/dev/null || {
            log_warn "未找到 ${service} 的部署历史"
            continue
        }
        
        # 显示每个修订版本的详细信息
        REVISIONS=$(kubectl rollout history "deployment/${service}" -n "${NAMESPACE}" \
            -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}' 2>/dev/null)
        
        # 获取当前镜像
        CURRENT_IMAGE=$(kubectl get deployment "${service}" -n "${NAMESPACE}" \
            -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null)
        echo "当前镜像: ${CURRENT_IMAGE}"
        
        # 获取 ReplicaSet 历史
        echo ""
        echo "ReplicaSet 历史:"
        kubectl get rs -n "${NAMESPACE}" -l app="${service}" \
            -o custom-columns=NAME:.metadata.name,DESIRED:.spec.replicas,CURRENT:.status.replicas,READY:.status.readyReplicas,AGE:.metadata.creationTimestamp \
            --sort-by=.metadata.creationTimestamp
    done
}

# 获取之前的版本
get_previous_version() {
    local service=$1
    local namespace="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    
    # 从部署记录中获取
    RECORD=$(kubectl get configmap deployment-record -n "${namespace}" \
        -o jsonpath='{.data.record\.json}' 2>/dev/null || echo "{}")
    
    if [[ "${RECORD}" != "{}" ]]; then
        PREV_VERSION=$(echo "${RECORD}" | jq -r '.version // empty')
        if [[ -n "${PREV_VERSION}" ]]; then
            echo "${PREV_VERSION}"
            return
        fi
    fi
    
    # 从 ReplicaSet 获取
    PREV_IMAGE=$(kubectl get rs -n "${namespace}" -l app="${service}" \
        --sort-by=.metadata.creationTimestamp \
        -o jsonpath='{.items[-2].spec.template.spec.containers[0].image}' 2>/dev/null)
    
    if [[ -n "${PREV_IMAGE}" ]]; then
        echo "${PREV_IMAGE##*:}"
    fi
}

# 回滚到指定修订版本
rollback_to_revision() {
    log_info "回滚到修订版本 ${REVISION}..."
    
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    
    IFS=',' read -ra SERVICE_ARRAY <<< "${SERVICES}"
    
    FAILED=false
    for service in "${SERVICE_ARRAY[@]}"; do
        service=$(echo "${service}" | xargs)
        
        log_info "  回滚 ${service} 到修订版本 ${REVISION}..."
        
        KUBECTL_ARGS=()
        if [[ "${DRY_RUN}" == "true" ]]; then
            KUBECTL_ARGS+=("--dry-run=client")
        fi
        
        if kubectl rollout undo "deployment/${service}" \
            --to-revision="${REVISION}" \
            -n "${NAMESPACE}" \
            "${KUBECTL_ARGS[@]}"; then
            log_success "  ✓ ${service}"
        else
            log_error "  ✗ ${service} 回滚失败"
            FAILED=true
        fi
    done
    
    if [[ "${FAILED}" == "true" ]]; then
        exit 1
    fi
}

# 回滚到指定版本标签
rollback_to_version() {
    log_info "回滚到版本 ${TARGET_VERSION}..."
    
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    
    IFS=',' read -ra SERVICE_ARRAY <<< "${SERVICES}"
    
    FAILED=false
    for service in "${SERVICE_ARRAY[@]}"; do
        service=$(echo "${service}" | xargs)
        IMAGE="${REGISTRY}/${service}:${TARGET_VERSION}"
        
        log_info "  回滚 ${service} 到 ${IMAGE}..."
        
        KUBECTL_ARGS=()
        if [[ "${DRY_RUN}" == "true" ]]; then
            KUBECTL_ARGS+=("--dry-run=client")
        fi
        
        if kubectl set image "deployment/${service}" \
            "${service}=${IMAGE}" \
            -n "${NAMESPACE}" \
            "${KUBECTL_ARGS[@]}"; then
            log_success "  ✓ ${service}"
        else
            log_error "  ✗ ${service} 回滚失败"
            FAILED=true
        fi
    done
    
    if [[ "${FAILED}" == "true" ]]; then
        exit 1
    fi
}

# 回滚到上一版本
rollback_to_previous() {
    log_info "回滚到上一版本..."
    
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    
    IFS=',' read -ra SERVICE_ARRAY <<< "${SERVICES}"
    
    FAILED=false
    for service in "${SERVICE_ARRAY[@]}"; do
        service=$(echo "${service}" | xargs)
        
        log_info "  回滚 ${service}..."
        
        KUBECTL_ARGS=()
        if [[ "${DRY_RUN}" == "true" ]]; then
            KUBECTL_ARGS+=("--dry-run=client")
        fi
        
        if kubectl rollout undo "deployment/${service}" \
            -n "${NAMESPACE}" \
            "${KUBECTL_ARGS[@]}"; then
            log_success "  ✓ ${service}"
        else
            log_error "  ✗ ${service} 回滚失败"
            FAILED=true
        fi
    done
    
    if [[ "${FAILED}" == "true" ]]; then
        exit 1
    fi
}

# 等待回滚完成
wait_for_rollback() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        return
    fi
    
    log_info "等待回滚完成..."
    
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    
    IFS=',' read -ra SERVICE_ARRAY <<< "${SERVICES}"
    
    for service in "${SERVICE_ARRAY[@]}"; do
        service=$(echo "${service}" | xargs)
        
        log_info "  等待 ${service}..."
        
        if kubectl rollout status "deployment/${service}" \
            -n "${NAMESPACE}" \
            --timeout=5m; then
            log_success "  ✓ ${service} 回滚完成"
        else
            log_warn "  ${service} 回滚超时，请手动检查"
        fi
    done
}

# 清理金丝雀部署
cleanup_canary() {
    log_info "清理金丝雀部署..."
    
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    
    IFS=',' read -ra SERVICE_ARRAY <<< "${SERVICES}"
    
    for service in "${SERVICE_ARRAY[@]}"; do
        service=$(echo "${service}" | xargs)
        CANARY_NAME="${service}-canary"
        
        if kubectl get deployment "${CANARY_NAME}" -n "${NAMESPACE}" &> /dev/null; then
            log_info "  删除 ${CANARY_NAME}..."
            
            KUBECTL_ARGS=()
            if [[ "${DRY_RUN}" == "true" ]]; then
                KUBECTL_ARGS+=("--dry-run=client")
            fi
            
            kubectl delete deployment "${CANARY_NAME}" \
                -n "${NAMESPACE}" \
                --ignore-not-found \
                "${KUBECTL_ARGS[@]}"
            
            log_success "  ✓ ${CANARY_NAME}"
        fi
    done
}

# 验证回滚
verify_rollback() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        return
    fi
    
    log_info "验证回滚..."
    
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    
    # 等待一段时间让服务稳定
    sleep 10
    
    # 检查 Pod 状态
    UNHEALTHY_PODS=$(kubectl get pods -n "${NAMESPACE}" \
        -o jsonpath='{.items[?(@.status.phase!="Running")].metadata.name}')
    
    if [[ -n "${UNHEALTHY_PODS}" ]]; then
        log_warn "发现不健康的 Pod: ${UNHEALTHY_PODS}"
    else
        log_success "所有 Pod 运行正常"
    fi
    
    # 显示当前状态
    log_info "当前部署状态:"
    kubectl get deployments -n "${NAMESPACE}" -o wide
}

# 记录回滚
record_rollback() {
    log_info "记录回滚信息..."
    
    NAMESPACE="${ENV_NAMESPACES[${ENVIRONMENT}]}"
    
    # 创建回滚记录
    RECORD=$(cat << EOF
{
    "action": "rollback",
    "environment": "${ENVIRONMENT}",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "services": "${SERVICES}",
    "target_revision": "${REVISION:-previous}",
    "target_version": "${TARGET_VERSION:-auto}",
    "executor": "${USER:-unknown}"
}
EOF
)
    
    if [[ "${DRY_RUN}" != "true" ]]; then
        kubectl create configmap rollback-record \
            --from-literal=record.json="${RECORD}" \
            -n "${NAMESPACE}" \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
}

# 发送通知
send_notification() {
    if [[ -z "${SLACK_WEBHOOK_URL:-}" ]]; then
        return
    fi
    
    log_info "发送通知..."
    
    PAYLOAD=$(cat << EOF
{
    "text": "🔄 服务回滚完成",
    "attachments": [{
        "color": "warning",
        "fields": [
            {"title": "环境", "value": "${ENVIRONMENT}", "short": true},
            {"title": "服务", "value": "${SERVICES}", "short": true},
            {"title": "目标版本", "value": "${TARGET_VERSION:-上一版本}", "short": true},
            {"title": "执行者", "value": "${USER:-unknown}", "short": true}
        ]
    }]
}
EOF
)
    
    curl -s -X POST -H 'Content-type: application/json' \
        --data "${PAYLOAD}" \
        "${SLACK_WEBHOOK_URL}" > /dev/null || true
}

# 主函数
main() {
    parse_args "$@"
    
    # 只显示历史
    if [[ "${SHOW_HISTORY}" == "true" ]]; then
        check_prerequisites
        show_deployment_history
        exit 0
    fi
    
    log_info "=== 开始回滚 ==="
    log_info "环境: ${ENVIRONMENT}"
    log_info "服务: ${SERVICES}"
    if [[ -n "${REVISION}" ]]; then
        log_info "目标修订版本: ${REVISION}"
    elif [[ -n "${TARGET_VERSION}" ]]; then
        log_info "目标版本: ${TARGET_VERSION}"
    else
        log_info "目标: 上一版本"
    fi
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warn "试运行模式 - 不会实际回滚"
    fi
    
    # 确认回滚
    if [[ "${ENVIRONMENT}" == "prod" ]] && [[ "${DRY_RUN}" != "true" ]]; then
        echo ""
        log_warn "⚠️  即将回滚生产环境!"
        read -p "确认继续? (输入 'yes' 确认): " CONFIRM
        if [[ "${CONFIRM}" != "yes" ]]; then
            log_info "回滚已取消"
            exit 0
        fi
    fi
    
    START_TIME=$(date +%s)
    
    check_prerequisites
    
    # 执行回滚
    if [[ -n "${REVISION}" ]]; then
        rollback_to_revision
    elif [[ -n "${TARGET_VERSION}" ]]; then
        rollback_to_version
    else
        rollback_to_previous
    fi
    
    cleanup_canary
    wait_for_rollback
    verify_rollback
    record_rollback
    send_notification
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    log_success "=== 回滚完成 (耗时: ${DURATION}s) ==="
}

main "$@"

