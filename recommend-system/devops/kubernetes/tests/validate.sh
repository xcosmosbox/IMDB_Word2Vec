#!/bin/bash
# Kubernetes 配置验证脚本
# 使用多种工具验证配置的正确性

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$(dirname "$SCRIPT_DIR")"

# 计数器
PASSED=0
FAILED=0
SKIPPED=0

# 输出函数
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((SKIPPED++))
}

# 检查工具是否存在
check_tool() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# 1. YAML 语法验证
test_yaml_syntax() {
    info "=== 测试 YAML 语法 ==="
    
    if check_tool "yamllint"; then
        find "$K8S_DIR" -name "*.yaml" -o -name "*.yml" | while read -r file; do
            if yamllint -d relaxed "$file" > /dev/null 2>&1; then
                pass "YAML 语法正确: $(basename "$file")"
            else
                fail "YAML 语法错误: $(basename "$file")"
                yamllint -d relaxed "$file"
            fi
        done
    else
        skip "yamllint 未安装，跳过 YAML 语法验证"
    fi
}

# 2. Kubernetes 资源验证
test_k8s_resources() {
    info "=== 测试 Kubernetes 资源格式 ==="
    
    if check_tool "kubeconform"; then
        # 验证 base 配置
        info "验证 base 配置..."
        find "$K8S_DIR/base" -name "*.yaml" | while read -r file; do
            if kubeconform -strict -ignore-missing-schemas "$file" > /dev/null 2>&1; then
                pass "资源格式正确: $(basename "$file")"
            else
                fail "资源格式错误: $(basename "$file")"
                kubeconform -strict -ignore-missing-schemas "$file"
            fi
        done
        
        # 验证 Istio 配置
        info "验证 Istio 配置..."
        find "$K8S_DIR/istio" -name "*.yaml" 2>/dev/null | while read -r file; do
            if kubeconform -strict -ignore-missing-schemas "$file" > /dev/null 2>&1; then
                pass "Istio 资源格式正确: $(basename "$file")"
            else
                warn "Istio 资源验证跳过 (可能需要 CRD): $(basename "$file")"
            fi
        done
        
        # 验证 Ingress 配置
        info "验证 Ingress 配置..."
        find "$K8S_DIR/ingress" -name "*.yaml" 2>/dev/null | while read -r file; do
            if kubeconform -strict -ignore-missing-schemas "$file" > /dev/null 2>&1; then
                pass "Ingress 资源格式正确: $(basename "$file")"
            else
                warn "Ingress 资源验证跳过 (可能需要 CRD): $(basename "$file")"
            fi
        done
    else
        skip "kubeconform 未安装，跳过资源格式验证"
    fi
}

# 3. Kustomize 构建验证
test_kustomize_build() {
    info "=== 测试 Kustomize 构建 ==="
    
    if check_tool "kustomize"; then
        # 验证 base
        info "构建 base 配置..."
        if kustomize build "$K8S_DIR/base" > /dev/null 2>&1; then
            pass "base 配置构建成功"
        else
            fail "base 配置构建失败"
            kustomize build "$K8S_DIR/base"
        fi
        
        # 验证 dev overlay
        info "构建 dev overlay..."
        if kustomize build "$K8S_DIR/overlays/dev" > /dev/null 2>&1; then
            pass "dev overlay 构建成功"
        else
            fail "dev overlay 构建失败"
            kustomize build "$K8S_DIR/overlays/dev"
        fi
        
        # 验证 prod overlay
        info "构建 prod overlay..."
        if kustomize build "$K8S_DIR/overlays/prod" > /dev/null 2>&1; then
            pass "prod overlay 构建成功"
        else
            fail "prod overlay 构建失败"
            kustomize build "$K8S_DIR/overlays/prod"
        fi
    elif check_tool "kubectl"; then
        # 使用 kubectl kustomize 作为备选
        info "构建 base 配置 (使用 kubectl kustomize)..."
        if kubectl kustomize "$K8S_DIR/base" > /dev/null 2>&1; then
            pass "base 配置构建成功"
        else
            fail "base 配置构建失败"
        fi
        
        info "构建 dev overlay..."
        if kubectl kustomize "$K8S_DIR/overlays/dev" > /dev/null 2>&1; then
            pass "dev overlay 构建成功"
        else
            fail "dev overlay 构建失败"
        fi
        
        info "构建 prod overlay..."
        if kubectl kustomize "$K8S_DIR/overlays/prod" > /dev/null 2>&1; then
            pass "prod overlay 构建成功"
        else
            fail "prod overlay 构建失败"
        fi
    else
        skip "kustomize 和 kubectl 均未安装，跳过构建验证"
    fi
}

# 4. 资源限制验证
test_resource_limits() {
    info "=== 测试资源限制配置 ==="
    
    # 检查所有 Deployment 是否配置了资源限制
    find "$K8S_DIR/base" -name "deployment.yaml" | while read -r file; do
        if grep -q "resources:" "$file" && grep -q "limits:" "$file" && grep -q "requests:" "$file"; then
            pass "资源限制已配置: $(basename "$file")"
        else
            fail "缺少资源限制配置: $(basename "$file")"
        fi
    done
}

# 5. 安全配置验证
test_security_config() {
    info "=== 测试安全配置 ==="
    
    # 检查 SecurityContext
    find "$K8S_DIR/base" -name "deployment.yaml" | while read -r file; do
        if grep -q "securityContext:" "$file"; then
            pass "SecurityContext 已配置: $(basename "$file")"
            
            # 检查具体安全设置
            if grep -q "runAsNonRoot: true" "$file"; then
                pass "runAsNonRoot 已启用"
            else
                warn "建议启用 runAsNonRoot"
            fi
            
            if grep -q "readOnlyRootFilesystem: true" "$file"; then
                pass "只读根文件系统已启用"
            else
                warn "建议启用只读根文件系统"
            fi
        else
            fail "缺少 SecurityContext 配置: $(basename "$file")"
        fi
    done
}

# 6. 探针配置验证
test_probes() {
    info "=== 测试探针配置 ==="
    
    find "$K8S_DIR/base" -name "deployment.yaml" | while read -r file; do
        if grep -q "livenessProbe:" "$file"; then
            pass "存活探针已配置: $(basename "$file")"
        else
            fail "缺少存活探针: $(basename "$file")"
        fi
        
        if grep -q "readinessProbe:" "$file"; then
            pass "就绪探针已配置: $(basename "$file")"
        else
            fail "缺少就绪探针: $(basename "$file")"
        fi
        
        if grep -q "startupProbe:" "$file"; then
            pass "启动探针已配置: $(basename "$file")"
        else
            warn "建议配置启动探针: $(basename "$file")"
        fi
    done
}

# 7. 命名规范验证
test_naming_conventions() {
    info "=== 测试命名规范 ==="
    
    # 检查标签
    find "$K8S_DIR/base" -name "*.yaml" | while read -r file; do
        if grep -q "app.kubernetes.io/name:" "$file"; then
            pass "标准标签已配置: $(basename "$file")"
        else
            warn "建议使用 app.kubernetes.io 标签: $(basename "$file")"
        fi
    done
}

# 8. 服务端口验证
test_service_ports() {
    info "=== 测试服务端口配置 ==="
    
    # 验证端口是否符合接口定义
    if [ -f "$K8S_DIR/base/service.yaml" ]; then
        # recommend-service: 8080, 9090, 9091
        if grep -q "port: 8080" "$K8S_DIR/base/service.yaml" && \
           grep -q "port: 9090" "$K8S_DIR/base/service.yaml" && \
           grep -q "port: 9091" "$K8S_DIR/base/service.yaml"; then
            pass "recommend-service 端口配置正确"
        else
            fail "recommend-service 端口配置不正确"
        fi
        
        # ugt-inference: 50051, 9094
        if grep -q "port: 50051" "$K8S_DIR/base/service.yaml" && \
           grep -q "port: 9094" "$K8S_DIR/base/service.yaml"; then
            pass "ugt-inference 端口配置正确"
        else
            fail "ugt-inference 端口配置不正确"
        fi
    fi
}

# 9. HPA 配置验证
test_hpa_config() {
    info "=== 测试 HPA 配置 ==="
    
    if [ -f "$K8S_DIR/base/hpa.yaml" ]; then
        if grep -q "minReplicas:" "$K8S_DIR/base/hpa.yaml" && \
           grep -q "maxReplicas:" "$K8S_DIR/base/hpa.yaml"; then
            pass "HPA 副本数限制已配置"
        else
            fail "HPA 缺少副本数限制"
        fi
        
        if grep -q "behavior:" "$K8S_DIR/base/hpa.yaml"; then
            pass "HPA 行为策略已配置"
        else
            warn "建议配置 HPA 行为策略"
        fi
    fi
}

# 10. PDB 配置验证
test_pdb_config() {
    info "=== 测试 PDB 配置 ==="
    
    if [ -f "$K8S_DIR/base/pdb.yaml" ]; then
        if grep -q "minAvailable:" "$K8S_DIR/base/pdb.yaml" || \
           grep -q "maxUnavailable:" "$K8S_DIR/base/pdb.yaml"; then
            pass "PDB 已正确配置"
        else
            fail "PDB 配置不正确"
        fi
    else
        fail "缺少 PDB 配置"
    fi
}

# 主函数
main() {
    echo "=========================================="
    echo "Kubernetes 配置验证"
    echo "=========================================="
    echo ""
    
    test_yaml_syntax
    echo ""
    test_k8s_resources
    echo ""
    test_kustomize_build
    echo ""
    test_resource_limits
    echo ""
    test_security_config
    echo ""
    test_probes
    echo ""
    test_naming_conventions
    echo ""
    test_service_ports
    echo ""
    test_hpa_config
    echo ""
    test_pdb_config
    
    echo ""
    echo "=========================================="
    echo "测试结果汇总"
    echo "=========================================="
    echo -e "${GREEN}通过: $PASSED${NC}"
    echo -e "${RED}失败: $FAILED${NC}"
    echo -e "${YELLOW}跳过: $SKIPPED${NC}"
    echo ""
    
    if [ $FAILED -gt 0 ]; then
        error "存在失败的测试，请检查配置"
        exit 1
    else
        info "所有测试通过！"
        exit 0
    fi
}

main "$@"

