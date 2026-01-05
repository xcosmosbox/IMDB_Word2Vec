#!/usr/bin/env bats
# =============================================================================
# build.sh 单元测试
# 
# 使用 bats (Bash Automated Testing System) 测试框架
# 安装: npm install -g bats 或 brew install bats-core
# 运行: bats tests/
# =============================================================================

# 测试辅助函数
setup() {
    # 设置测试目录
    TEST_DIR="$(cd "$(dirname "$BATS_TEST_FILENAME")" && pwd)"
    SCRIPTS_DIR="${TEST_DIR}/../scripts"
    
    # 创建临时目录
    TEMP_DIR=$(mktemp -d)
    
    # 模拟项目结构
    mkdir -p "${TEMP_DIR}/recommend-system/cmd/recommend-service"
    mkdir -p "${TEMP_DIR}/recommend-system/cmd/user-service"
    mkdir -p "${TEMP_DIR}/recommend-system/cmd/item-service"
    mkdir -p "${TEMP_DIR}/recommend-system/algorithm"
    mkdir -p "${TEMP_DIR}/recommend-system/frontend/user-app"
    mkdir -p "${TEMP_DIR}/recommend-system/frontend/admin"
    mkdir -p "${TEMP_DIR}/recommend-system/bin"
    
    # 创建模拟 Go 文件
    cat > "${TEMP_DIR}/recommend-system/cmd/recommend-service/main.go" << 'EOF'
package main
func main() {}
EOF
    
    # 创建 go.mod
    cat > "${TEMP_DIR}/recommend-system/go.mod" << 'EOF'
module recommend-system
go 1.21
EOF
    
    # 导出变量
    export PROJECT_ROOT="${TEMP_DIR}"
    export VERSION="test-v1.0.0"
    export REGISTRY="test-registry.com"
}

teardown() {
    # 清理临时目录
    rm -rf "${TEMP_DIR}"
}

# =============================================================================
# 帮助信息测试
# =============================================================================

@test "build.sh --help 显示帮助信息" {
    run bash "${SCRIPTS_DIR}/build.sh" --help
    [ "$status" -eq 0 ]
    [[ "$output" == *"用法"* ]]
    [[ "$output" == *"目标"* ]]
    [[ "$output" == *"选项"* ]]
}

@test "build.sh -h 显示帮助信息" {
    run bash "${SCRIPTS_DIR}/build.sh" -h
    [ "$status" -eq 0 ]
    [[ "$output" == *"用法"* ]]
}

# =============================================================================
# 参数解析测试
# =============================================================================

@test "build.sh 解析 --version 参数" {
    # 跳过实际构建，只测试参数解析
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; parse_args --version v2.0.0 go; echo \$VERSION"
    [[ "$output" == *"v2.0.0"* ]]
}

@test "build.sh 解析 --registry 参数" {
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; parse_args --registry myregistry.com go; echo \$REGISTRY"
    [[ "$output" == *"myregistry.com"* ]]
}

@test "build.sh 解析 --push 参数" {
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; parse_args --push docker; echo \$PUSH"
    [[ "$output" == *"true"* ]]
}

@test "build.sh 拒绝未知参数" {
    run bash "${SCRIPTS_DIR}/build.sh" --unknown-param
    [ "$status" -eq 1 ]
    [[ "$output" == *"未知参数"* ]]
}

# =============================================================================
# 目标验证测试
# =============================================================================

@test "build.sh 接受 'go' 目标" {
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; parse_args go; echo \$TARGET"
    [[ "$output" == *"go"* ]]
}

@test "build.sh 接受 'python' 目标" {
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; parse_args python; echo \$TARGET"
    [[ "$output" == *"python"* ]]
}

@test "build.sh 接受 'frontend' 目标" {
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; parse_args frontend; echo \$TARGET"
    [[ "$output" == *"frontend"* ]]
}

@test "build.sh 接受 'docker' 目标" {
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; parse_args docker; echo \$TARGET"
    [[ "$output" == *"docker"* ]]
}

@test "build.sh 接受 'all' 目标" {
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; parse_args all; echo \$TARGET"
    [[ "$output" == *"all"* ]]
}

@test "build.sh 默认目标是 'all'" {
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; parse_args; echo \$TARGET"
    [[ "$output" == *"all"* ]]
}

# =============================================================================
# 日志函数测试
# =============================================================================

@test "log_info 输出蓝色日志" {
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; log_info 'test message'"
    [[ "$output" == *"INFO"* ]]
    [[ "$output" == *"test message"* ]]
}

@test "log_success 输出绿色日志" {
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; log_success 'success message'"
    [[ "$output" == *"SUCCESS"* ]]
    [[ "$output" == *"success message"* ]]
}

@test "log_warn 输出黄色日志" {
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; log_warn 'warning message'"
    [[ "$output" == *"WARN"* ]]
    [[ "$output" == *"warning message"* ]]
}

@test "log_error 输出红色日志到 stderr" {
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; log_error 'error message' 2>&1"
    [[ "$output" == *"ERROR"* ]]
    [[ "$output" == *"error message"* ]]
}

# =============================================================================
# 组合参数测试
# =============================================================================

@test "build.sh 支持多个参数组合" {
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; \
        parse_args go --version v1.0.0 --verbose; \
        echo \"TARGET=\$TARGET VERSION=\$VERSION VERBOSE=\$VERBOSE\""
    [[ "$output" == *"TARGET=go"* ]]
    [[ "$output" == *"VERSION=v1.0.0"* ]]
    [[ "$output" == *"VERBOSE=true"* ]]
}

@test "build.sh docker 目标支持 --push 和 --platform" {
    run bash -c "source ${SCRIPTS_DIR}/build.sh 2>/dev/null; \
        parse_args docker --push --platform linux/arm64 --registry myregistry.com; \
        echo \"TARGET=\$TARGET PUSH=\$PUSH PLATFORM=\$PLATFORM REGISTRY=\$REGISTRY\""
    [[ "$output" == *"TARGET=docker"* ]]
    [[ "$output" == *"PUSH=true"* ]]
    [[ "$output" == *"PLATFORM=linux/arm64"* ]]
    [[ "$output" == *"REGISTRY=myregistry.com"* ]]
}

