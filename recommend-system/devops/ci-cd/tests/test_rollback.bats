#!/usr/bin/env bats
# =============================================================================
# rollback.sh 单元测试
# 
# 使用 bats (Bash Automated Testing System) 测试框架
# 运行: bats tests/test_rollback.bats
# =============================================================================

setup() {
    TEST_DIR="$(cd "$(dirname "$BATS_TEST_FILENAME")" && pwd)"
    SCRIPTS_DIR="${TEST_DIR}/../scripts"
    TEMP_DIR=$(mktemp -d)
    
    export PROJECT_ROOT="${TEMP_DIR}"
    export DRY_RUN="true"
}

teardown() {
    rm -rf "${TEMP_DIR}"
}

# =============================================================================
# 帮助信息测试
# =============================================================================

@test "rollback.sh --help 显示帮助信息" {
    run bash "${SCRIPTS_DIR}/rollback.sh" --help
    [ "$status" -eq 0 ]
    [[ "$output" == *"用法"* ]]
    [[ "$output" == *"环境"* ]]
    [[ "$output" == *"--revision"* ]]
    [[ "$output" == *"--version"* ]]
    [[ "$output" == *"--history"* ]]
}

# =============================================================================
# 环境验证测试
# =============================================================================

@test "rollback.sh 必须指定环境" {
    run bash "${SCRIPTS_DIR}/rollback.sh"
    [ "$status" -eq 1 ]
    [[ "$output" == *"必须指定环境"* ]]
}

@test "rollback.sh 接受 dev 环境" {
    run bash -c "source ${SCRIPTS_DIR}/rollback.sh 2>/dev/null; parse_args dev; echo \$ENVIRONMENT"
    [[ "$output" == *"dev"* ]]
}

@test "rollback.sh 接受 staging 环境" {
    run bash -c "source ${SCRIPTS_DIR}/rollback.sh 2>/dev/null; parse_args staging; echo \$ENVIRONMENT"
    [[ "$output" == *"staging"* ]]
}

@test "rollback.sh 接受 prod 环境" {
    run bash -c "source ${SCRIPTS_DIR}/rollback.sh 2>/dev/null; parse_args prod; echo \$ENVIRONMENT"
    [[ "$output" == *"prod"* ]]
}

# =============================================================================
# 参数解析测试
# =============================================================================

@test "rollback.sh 解析 --revision 参数" {
    run bash -c "source ${SCRIPTS_DIR}/rollback.sh 2>/dev/null; parse_args dev --revision 3; echo \$REVISION"
    [[ "$output" == *"3"* ]]
}

@test "rollback.sh 解析 --version 参数" {
    run bash -c "source ${SCRIPTS_DIR}/rollback.sh 2>/dev/null; parse_args dev --version v1.0.0; echo \$TARGET_VERSION"
    [[ "$output" == *"v1.0.0"* ]]
}

@test "rollback.sh 解析 --services 参数" {
    run bash -c "source ${SCRIPTS_DIR}/rollback.sh 2>/dev/null; parse_args dev --services recommend-service; echo \$SERVICES"
    [[ "$output" == *"recommend-service"* ]]
}

@test "rollback.sh 解析 --history 参数" {
    run bash -c "source ${SCRIPTS_DIR}/rollback.sh 2>/dev/null; parse_args dev --history; echo \$SHOW_HISTORY"
    [[ "$output" == *"true"* ]]
}

@test "rollback.sh 解析 --dry-run 参数" {
    run bash -c "source ${SCRIPTS_DIR}/rollback.sh 2>/dev/null; parse_args dev --dry-run; echo \$DRY_RUN"
    [[ "$output" == *"true"* ]]
}

# =============================================================================
# 命名空间映射测试
# =============================================================================

@test "rollback.sh 正确映射 dev 命名空间" {
    run bash -c "source ${SCRIPTS_DIR}/rollback.sh 2>/dev/null; \
        parse_args dev; \
        echo \${ENV_NAMESPACES[\$ENVIRONMENT]}"
    [[ "$output" == *"recommend-dev"* ]]
}

@test "rollback.sh 正确映射 prod 命名空间" {
    run bash -c "source ${SCRIPTS_DIR}/rollback.sh 2>/dev/null; \
        parse_args prod; \
        echo \${ENV_NAMESPACES[\$ENVIRONMENT]}"
    [[ "$output" == *"recommend-prod"* ]]
}

# =============================================================================
# 组合参数测试
# =============================================================================

@test "rollback.sh 支持 --revision 和 --services 组合" {
    run bash -c "source ${SCRIPTS_DIR}/rollback.sh 2>/dev/null; \
        parse_args dev --revision 2 --services recommend-service,user-service --dry-run; \
        echo \"REV=\$REVISION SERVICES=\$SERVICES DRY_RUN=\$DRY_RUN\""
    [[ "$output" == *"REV=2"* ]]
    [[ "$output" == *"SERVICES=recommend-service,user-service"* ]]
    [[ "$output" == *"DRY_RUN=true"* ]]
}

@test "rollback.sh 支持 --version 参数回滚到指定版本" {
    run bash -c "source ${SCRIPTS_DIR}/rollback.sh 2>/dev/null; \
        parse_args staging --version v1.2.3; \
        echo \"ENV=\$ENVIRONMENT VERSION=\$TARGET_VERSION\""
    [[ "$output" == *"ENV=staging"* ]]
    [[ "$output" == *"VERSION=v1.2.3"* ]]
}

# =============================================================================
# 默认值测试
# =============================================================================

@test "rollback.sh 默认服务列表包含所有服务" {
    run bash -c "source ${SCRIPTS_DIR}/rollback.sh 2>/dev/null; echo \$SERVICES"
    [[ "$output" == *"recommend-service"* ]]
    [[ "$output" == *"user-service"* ]]
    [[ "$output" == *"item-service"* ]]
    [[ "$output" == *"ugt-inference"* ]]
}

@test "rollback.sh SHOW_HISTORY 默认为 false" {
    run bash -c "source ${SCRIPTS_DIR}/rollback.sh 2>/dev/null; parse_args dev; echo \$SHOW_HISTORY"
    [[ "$output" == *"false"* ]]
}

