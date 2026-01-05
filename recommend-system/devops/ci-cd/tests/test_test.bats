#!/usr/bin/env bats
# =============================================================================
# test.sh 单元测试
# 
# 使用 bats (Bash Automated Testing System) 测试框架
# 运行: bats tests/test_test.bats
# =============================================================================

setup() {
    TEST_DIR="$(cd "$(dirname "$BATS_TEST_FILENAME")" && pwd)"
    SCRIPTS_DIR="${TEST_DIR}/../scripts"
    TEMP_DIR=$(mktemp -d)
    
    export PROJECT_ROOT="${TEMP_DIR}"
}

teardown() {
    rm -rf "${TEMP_DIR}"
}

# =============================================================================
# 帮助信息测试
# =============================================================================

@test "test.sh --help 显示帮助信息" {
    run bash "${SCRIPTS_DIR}/test.sh" --help
    [ "$status" -eq 0 ]
    [[ "$output" == *"用法"* ]]
    [[ "$output" == *"目标"* ]]
    [[ "$output" == *"--coverage"* ]]
    [[ "$output" == *"--integration"* ]]
}

# =============================================================================
# 目标验证测试
# =============================================================================

@test "test.sh 接受 'go' 目标" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; parse_args go; echo \$TARGET"
    [[ "$output" == *"go"* ]]
}

@test "test.sh 接受 'python' 目标" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; parse_args python; echo \$TARGET"
    [[ "$output" == *"python"* ]]
}

@test "test.sh 接受 'frontend' 目标" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; parse_args frontend; echo \$TARGET"
    [[ "$output" == *"frontend"* ]]
}

@test "test.sh 接受 'integration' 目标" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; parse_args integration; echo \$TARGET"
    [[ "$output" == *"integration"* ]]
}

@test "test.sh 接受 'all' 目标" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; parse_args all; echo \$TARGET"
    [[ "$output" == *"all"* ]]
}

@test "test.sh 默认目标是 'all'" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; parse_args; echo \$TARGET"
    [[ "$output" == *"all"* ]]
}

# =============================================================================
# 参数解析测试
# =============================================================================

@test "test.sh 解析 --coverage 参数" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; parse_args go --coverage; echo \$COVERAGE"
    [[ "$output" == *"true"* ]]
}

@test "test.sh 解析 --integration 参数" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; parse_args go --integration; echo \$INTEGRATION"
    [[ "$output" == *"true"* ]]
}

@test "test.sh 解析 --race 参数" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; parse_args go --race; echo \$RACE"
    [[ "$output" == *"true"* ]]
}

@test "test.sh 解析 --verbose 参数" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; parse_args go --verbose; echo \$VERBOSE"
    [[ "$output" == *"true"* ]]
}

@test "test.sh 解析 --fail-fast 参数" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; parse_args go --fail-fast; echo \$FAIL_FAST"
    [[ "$output" == *"true"* ]]
}

# =============================================================================
# 默认值测试
# =============================================================================

@test "test.sh COVERAGE 默认为 false" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; echo \$COVERAGE"
    [[ "$output" == *"false"* ]]
}

@test "test.sh INTEGRATION 默认为 false" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; echo \$INTEGRATION"
    [[ "$output" == *"false"* ]]
}

@test "test.sh RACE 默认为 false" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; echo \$RACE"
    [[ "$output" == *"false"* ]]
}

@test "test.sh FAIL_FAST 默认为 false" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; echo \$FAIL_FAST"
    [[ "$output" == *"false"* ]]
}

# =============================================================================
# 组合参数测试
# =============================================================================

@test "test.sh 支持多个参数组合" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; \
        parse_args go --coverage --race --verbose --fail-fast; \
        echo \"TARGET=\$TARGET COVERAGE=\$COVERAGE RACE=\$RACE VERBOSE=\$VERBOSE FAIL_FAST=\$FAIL_FAST\""
    [[ "$output" == *"TARGET=go"* ]]
    [[ "$output" == *"COVERAGE=true"* ]]
    [[ "$output" == *"RACE=true"* ]]
    [[ "$output" == *"VERBOSE=true"* ]]
    [[ "$output" == *"FAIL_FAST=true"* ]]
}

@test "test.sh 支持 all 目标加 integration" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; \
        parse_args all --integration --coverage; \
        echo \"TARGET=\$TARGET INTEGRATION=\$INTEGRATION COVERAGE=\$COVERAGE\""
    [[ "$output" == *"TARGET=all"* ]]
    [[ "$output" == *"INTEGRATION=true"* ]]
    [[ "$output" == *"COVERAGE=true"* ]]
}

# =============================================================================
# 日志函数测试
# =============================================================================

@test "test.sh log_info 函数正常工作" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; log_info 'test message'"
    [[ "$output" == *"INFO"* ]]
    [[ "$output" == *"test message"* ]]
}

@test "test.sh log_success 函数正常工作" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; log_success 'success'"
    [[ "$output" == *"SUCCESS"* ]]
}

@test "test.sh log_error 函数正常工作" {
    run bash -c "source ${SCRIPTS_DIR}/test.sh 2>/dev/null; log_error 'error' 2>&1"
    [[ "$output" == *"ERROR"* ]]
}

