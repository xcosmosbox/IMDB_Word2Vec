#!/usr/bin/env bats
# =============================================================================
# deploy.sh 单元测试
# 
# 使用 bats (Bash Automated Testing System) 测试框架
# 运行: bats tests/test_deploy.bats
# =============================================================================

setup() {
    TEST_DIR="$(cd "$(dirname "$BATS_TEST_FILENAME")" && pwd)"
    SCRIPTS_DIR="${TEST_DIR}/../scripts"
    TEMP_DIR=$(mktemp -d)
    
    # 设置模拟环境
    export PROJECT_ROOT="${TEMP_DIR}"
    export VERSION="test-v1.0.0"
    export REGISTRY="test-registry.com"
    export DRY_RUN="true"  # 始终使用 dry-run 模式进行测试
}

teardown() {
    rm -rf "${TEMP_DIR}"
}

# =============================================================================
# 帮助信息测试
# =============================================================================

@test "deploy.sh --help 显示帮助信息" {
    run bash "${SCRIPTS_DIR}/deploy.sh" --help
    [ "$status" -eq 0 ]
    [[ "$output" == *"用法"* ]]
    [[ "$output" == *"环境"* ]]
    [[ "$output" == *"dev"* ]]
    [[ "$output" == *"staging"* ]]
    [[ "$output" == *"prod"* ]]
}

# =============================================================================
# 环境验证测试
# =============================================================================

@test "deploy.sh 必须指定环境" {
    run bash "${SCRIPTS_DIR}/deploy.sh"
    [ "$status" -eq 1 ]
    [[ "$output" == *"必须指定环境"* ]]
}

@test "deploy.sh 接受 dev 环境" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; parse_args dev; echo \$ENVIRONMENT"
    [[ "$output" == *"dev"* ]]
}

@test "deploy.sh 接受 staging 环境" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; parse_args staging; echo \$ENVIRONMENT"
    [[ "$output" == *"staging"* ]]
}

@test "deploy.sh 接受 prod 环境" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; parse_args prod; echo \$ENVIRONMENT"
    [[ "$output" == *"prod"* ]]
}

# =============================================================================
# 参数解析测试
# =============================================================================

@test "deploy.sh 解析 --version 参数" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; parse_args dev --version v2.0.0; echo \$VERSION"
    [[ "$output" == *"v2.0.0"* ]]
}

@test "deploy.sh 解析 --services 参数" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; parse_args dev --services recommend-service; echo \$SERVICES"
    [[ "$output" == *"recommend-service"* ]]
}

@test "deploy.sh 解析 --canary 参数" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; parse_args dev --canary; echo \$CANARY"
    [[ "$output" == *"true"* ]]
}

@test "deploy.sh 解析 --canary-weight 参数" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; parse_args dev --canary --canary-weight 20; echo \$CANARY_WEIGHT"
    [[ "$output" == *"20"* ]]
}

@test "deploy.sh 解析 --timeout 参数" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; parse_args dev --timeout 10m; echo \$TIMEOUT"
    [[ "$output" == *"10m"* ]]
}

@test "deploy.sh 解析 --dry-run 参数" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; parse_args dev --dry-run; echo \$DRY_RUN"
    [[ "$output" == *"true"* ]]
}

@test "deploy.sh 解析 --no-wait 参数" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; parse_args dev --no-wait; echo \$WAIT"
    [[ "$output" == *"false"* ]]
}

# =============================================================================
# 命名空间映射测试
# =============================================================================

@test "deploy.sh dev 环境映射到 recommend-dev 命名空间" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; \
        parse_args dev; \
        echo \${ENV_NAMESPACES[\$ENVIRONMENT]}"
    [[ "$output" == *"recommend-dev"* ]]
}

@test "deploy.sh staging 环境映射到 recommend-staging 命名空间" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; \
        parse_args staging; \
        echo \${ENV_NAMESPACES[\$ENVIRONMENT]}"
    [[ "$output" == *"recommend-staging"* ]]
}

@test "deploy.sh prod 环境映射到 recommend-prod 命名空间" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; \
        parse_args prod; \
        echo \${ENV_NAMESPACES[\$ENVIRONMENT]}"
    [[ "$output" == *"recommend-prod"* ]]
}

# =============================================================================
# 副本数映射测试
# =============================================================================

@test "deploy.sh dev 环境默认 1 个副本" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; \
        parse_args dev; \
        echo \${ENV_REPLICAS[\$ENVIRONMENT]}"
    [[ "$output" == *"1"* ]]
}

@test "deploy.sh staging 环境默认 2 个副本" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; \
        parse_args staging; \
        echo \${ENV_REPLICAS[\$ENVIRONMENT]}"
    [[ "$output" == *"2"* ]]
}

@test "deploy.sh prod 环境默认 3 个副本" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; \
        parse_args prod; \
        echo \${ENV_REPLICAS[\$ENVIRONMENT]}"
    [[ "$output" == *"3"* ]]
}

# =============================================================================
# 组合参数测试
# =============================================================================

@test "deploy.sh 支持完整的参数组合" {
    run bash -c "source ${SCRIPTS_DIR}/deploy.sh 2>/dev/null; \
        parse_args prod --version v1.0.0 --canary --canary-weight 5 --timeout 10m --dry-run; \
        echo \"ENV=\$ENVIRONMENT VER=\$VERSION CANARY=\$CANARY WEIGHT=\$CANARY_WEIGHT\""
    [[ "$output" == *"ENV=prod"* ]]
    [[ "$output" == *"VER=v1.0.0"* ]]
    [[ "$output" == *"CANARY=true"* ]]
    [[ "$output" == *"WEIGHT=5"* ]]
}

