#!/bin/bash
# =============================================================================
# PostgreSQL 数据库备份验证脚本
# 
# 项目：生成式推荐系统
# 模块：数据库管理
# 版本：1.0.0
# 
# 功能：
#   - 验证备份文件完整性
#   - 测试恢复到临时数据库
#   - 比较源数据库和恢复数据库
#   - 生成验证报告
# =============================================================================

set -euo pipefail

# =============================================================================
# 配置变量
# =============================================================================
BACKUP_DIR="${BACKUP_DIR:-/backups}"
BACKUP_LOG_DIR="${BACKUP_LOG_DIR:-/var/log/backup}"

DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-recommend}"
DB_USER="${POSTGRES_USER:-postgres}"
DB_PASSWORD="${POSTGRES_PASSWORD:-}"

VERIFY_DB_NAME="${VERIFY_DB_NAME:-recommend_verify}"

SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
CLEANUP_VERIFY_DB="${CLEANUP_VERIFY_DB:-true}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${BACKUP_LOG_DIR}/verify_${TIMESTAMP}.log"
REPORT_FILE="${BACKUP_DIR}/verify_report_${TIMESTAMP}.json"

# =============================================================================
# 日志函数
# =============================================================================
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() { log "INFO" "$1"; }
log_warn() { log "WARN" "$1"; }
log_error() { log "ERROR" "$1"; }
log_success() { log "SUCCESS" "$1"; }

# =============================================================================
# 使用说明
# =============================================================================
usage() {
    cat <<EOF
Usage: $0 [OPTIONS] <backup_file>

Verify PostgreSQL backup integrity and recoverability

Arguments:
  backup_file           Path to backup file or 'latest' for most recent

Options:
  --no-cleanup          Don't remove verification database after test
  --compare             Compare with source database
  --quick               Quick verification (skip restore test)
  -h, --help            Show this help message

Examples:
  # Verify latest backup
  $0 latest

  # Verify specific backup with comparison
  $0 --compare /backups/daily/recommend_20240101.sql.gz

  # Quick verification (no restore test)
  $0 --quick latest
EOF
    exit 0
}

# =============================================================================
# 参数解析
# =============================================================================
BACKUP_FILE=""
DO_COMPARE="${DO_COMPARE:-false}"
QUICK_MODE="${QUICK_MODE:-false}"

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-cleanup)
                CLEANUP_VERIFY_DB="false"
                shift
                ;;
            --compare)
                DO_COMPARE="true"
                shift
                ;;
            --quick)
                QUICK_MODE="true"
                shift
                ;;
            -h|--help)
                usage
                ;;
            -*)
                log_error "Unknown option: $1"
                usage
                ;;
            *)
                BACKUP_FILE="$1"
                shift
                ;;
        esac
    done
    
    if [[ -z "${BACKUP_FILE}" ]]; then
        log_error "Backup file is required"
        usage
    fi
}

# =============================================================================
# 通知函数
# =============================================================================
send_notification() {
    local status="$1"
    local message="$2"
    
    [[ -z "${SLACK_WEBHOOK}" ]] && return 0
    
    local emoji=""
    case "$status" in
        "success") emoji="✅" ;;
        "warning") emoji="⚠️" ;;
        "error") emoji="❌" ;;
    esac
    
    curl -s -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"${emoji} [Backup Verify] ${message}\"}" \
        "${SLACK_WEBHOOK}" > /dev/null || true
}

# =============================================================================
# 清理函数
# =============================================================================
cleanup() {
    if [[ "${CLEANUP_VERIFY_DB}" == "true" ]]; then
        log_info "Cleaning up verification database..."
        PGPASSWORD="${DB_PASSWORD}" dropdb \
            -h "${DB_HOST}" \
            -p "${DB_PORT}" \
            -U "${DB_USER}" \
            "${VERIFY_DB_NAME}" \
            --if-exists 2>/dev/null || true
    fi
}

trap cleanup EXIT

# =============================================================================
# 解析备份文件
# =============================================================================
resolve_backup_file() {
    if [[ "${BACKUP_FILE}" == "latest" ]]; then
        BACKUP_FILE="${BACKUP_DIR}/${DB_NAME}_latest.sql.gz"
        if [[ -L "${BACKUP_FILE}" ]]; then
            BACKUP_FILE=$(readlink -f "${BACKUP_FILE}")
        fi
    elif [[ ! "${BACKUP_FILE}" == /* ]]; then
        BACKUP_FILE="${BACKUP_DIR}/${BACKUP_FILE}"
    fi
    
    if [[ ! -f "${BACKUP_FILE}" ]]; then
        log_error "Backup file not found: ${BACKUP_FILE}"
        exit 1
    fi
    
    log_info "Verifying backup: ${BACKUP_FILE}"
}

# =============================================================================
# 验证文件完整性
# =============================================================================
verify_file_integrity() {
    log_info "=== Step 1: File Integrity Check ==="
    
    local file_size=$(du -h "${BACKUP_FILE}" | cut -f1)
    local file_size_bytes=$(stat -c %s "${BACKUP_FILE}" 2>/dev/null || stat -f %z "${BACKUP_FILE}")
    
    log_info "File size: ${file_size} (${file_size_bytes} bytes)"
    
    if [[ ${file_size_bytes} -lt 1000 ]]; then
        log_error "Backup file is too small (< 1KB), likely corrupted"
        return 1
    fi
    
    # GZIP 完整性检查
    log_info "Checking GZIP integrity..."
    if ! gzip -t "${BACKUP_FILE}" 2>&1; then
        log_error "GZIP integrity check failed"
        return 1
    fi
    log_success "GZIP integrity: OK"
    
    # 校验和验证
    local checksum_file="${BACKUP_FILE}.sha256"
    if [[ -f "${checksum_file}" ]]; then
        log_info "Verifying SHA256 checksum..."
        cd "$(dirname ${BACKUP_FILE})"
        if sha256sum -c "${checksum_file}" 2>&1; then
            log_success "SHA256 checksum: OK"
        else
            log_error "SHA256 checksum verification failed"
            return 1
        fi
    else
        log_warn "No checksum file found, skipping checksum verification"
    fi
    
    return 0
}

# =============================================================================
# 验证 SQL 内容
# =============================================================================
verify_sql_content() {
    log_info "=== Step 2: SQL Content Check ==="
    
    # 检查 SQL 文件头部
    log_info "Checking SQL structure..."
    
    local head_content=$(gunzip -c "${BACKUP_FILE}" | head -100)
    
    # 检查是否包含 PostgreSQL dump 标识
    if echo "${head_content}" | grep -q "PostgreSQL database dump"; then
        log_success "PostgreSQL dump header: OK"
    else
        log_warn "PostgreSQL dump header not found"
    fi
    
    # 检查是否包含核心表
    local core_tables=("users" "items" "user_behaviors" "item_stats")
    local found_tables=0
    
    for table in "${core_tables[@]}"; do
        if gunzip -c "${BACKUP_FILE}" | grep -q "CREATE TABLE.*${table}"; then
            ((found_tables++))
        fi
    done
    
    log_info "Found ${found_tables}/${#core_tables[@]} core tables in backup"
    
    if [[ ${found_tables} -lt ${#core_tables[@]} ]]; then
        log_warn "Some core tables may be missing"
    else
        log_success "Core tables: OK"
    fi
    
    # 统计 SQL 语句
    local create_count=$(gunzip -c "${BACKUP_FILE}" | grep -c "^CREATE " || echo "0")
    local insert_count=$(gunzip -c "${BACKUP_FILE}" | grep -c "^INSERT " || echo "0")
    local index_count=$(gunzip -c "${BACKUP_FILE}" | grep -c "^CREATE INDEX" || echo "0")
    
    log_info "CREATE statements: ${create_count}"
    log_info "INSERT statements: ${insert_count}"
    log_info "INDEX statements: ${index_count}"
    
    return 0
}

# =============================================================================
# 测试恢复
# =============================================================================
test_restore() {
    log_info "=== Step 3: Restore Test ==="
    
    if [[ "${QUICK_MODE}" == "true" ]]; then
        log_info "Quick mode: Skipping restore test"
        return 0
    fi
    
    # 删除可能存在的验证数据库
    log_info "Preparing verification database..."
    PGPASSWORD="${DB_PASSWORD}" dropdb \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        "${VERIFY_DB_NAME}" \
        --if-exists
    
    # 创建验证数据库
    PGPASSWORD="${DB_PASSWORD}" createdb \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        "${VERIFY_DB_NAME}"
    
    log_info "Created verification database: ${VERIFY_DB_NAME}"
    
    # 恢复数据
    log_info "Restoring backup to verification database..."
    local start_time=$(date +%s)
    
    gunzip -c "${BACKUP_FILE}" | PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${VERIFY_DB_NAME}" \
        --quiet \
        2>> "${LOG_FILE}"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "Restore completed in ${duration} seconds"
    
    # 验证恢复的数据
    verify_restored_data
    
    return 0
}

# =============================================================================
# 验证恢复的数据
# =============================================================================
verify_restored_data() {
    log_info "=== Step 4: Data Validation ==="
    
    # 检查表数量
    local table_count=$(PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${VERIFY_DB_NAME}" \
        -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
    
    log_info "Tables in restored database: ${table_count}"
    
    # 检查核心表数据量
    declare -A table_counts
    local core_tables=("users" "items" "user_behaviors" "item_stats" "user_profiles")
    
    for table in "${core_tables[@]}"; do
        local count=$(PGPASSWORD="${DB_PASSWORD}" psql \
            -h "${DB_HOST}" \
            -p "${DB_PORT}" \
            -U "${DB_USER}" \
            -d "${VERIFY_DB_NAME}" \
            -t -c "SELECT COUNT(*) FROM ${table};" 2>/dev/null || echo "0")
        table_counts["${table}"]="${count}"
        log_info "Table ${table}: ${count} rows"
    done
    
    # 检查索引
    local index_count=$(PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${VERIFY_DB_NAME}" \
        -t -c "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public';")
    
    log_info "Indexes: ${index_count}"
    
    # 检查函数
    local function_count=$(PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${VERIFY_DB_NAME}" \
        -t -c "SELECT COUNT(*) FROM pg_proc p JOIN pg_namespace n ON p.pronamespace = n.oid WHERE n.nspname = 'public';")
    
    log_info "Functions: ${function_count}"
    
    # 检查扩展
    local extensions=$(PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${VERIFY_DB_NAME}" \
        -t -c "SELECT string_agg(extname, ', ') FROM pg_extension WHERE extname != 'plpgsql';")
    
    log_info "Extensions: ${extensions}"
    
    # 运行简单查询测试
    log_info "Running query tests..."
    
    # 测试查询 1: JOIN 查询
    local join_result=$(PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${VERIFY_DB_NAME}" \
        -t -c "SELECT COUNT(*) FROM items i LEFT JOIN item_stats s ON i.id = s.item_id;" 2>/dev/null || echo "ERROR")
    
    if [[ "${join_result}" == "ERROR" ]]; then
        log_warn "JOIN query test failed"
    else
        log_success "JOIN query test: OK (${join_result} rows)"
    fi
    
    return 0
}

# =============================================================================
# 与源数据库比较
# =============================================================================
compare_with_source() {
    if [[ "${DO_COMPARE}" != "true" ]]; then
        return 0
    fi
    
    log_info "=== Step 5: Source Comparison ==="
    
    local core_tables=("users" "items" "user_behaviors" "item_stats")
    local comparison_issues=0
    
    for table in "${core_tables[@]}"; do
        local source_count=$(PGPASSWORD="${DB_PASSWORD}" psql \
            -h "${DB_HOST}" \
            -p "${DB_PORT}" \
            -U "${DB_USER}" \
            -d "${DB_NAME}" \
            -t -c "SELECT COUNT(*) FROM ${table};" 2>/dev/null || echo "0")
        
        local verify_count=$(PGPASSWORD="${DB_PASSWORD}" psql \
            -h "${DB_HOST}" \
            -p "${DB_PORT}" \
            -U "${DB_USER}" \
            -d "${VERIFY_DB_NAME}" \
            -t -c "SELECT COUNT(*) FROM ${table};" 2>/dev/null || echo "0")
        
        source_count=$(echo "${source_count}" | tr -d ' ')
        verify_count=$(echo "${verify_count}" | tr -d ' ')
        
        if [[ "${source_count}" == "${verify_count}" ]]; then
            log_info "Table ${table}: ${source_count} = ${verify_count} ✓"
        else
            log_warn "Table ${table}: Source=${source_count}, Backup=${verify_count} (diff: $((source_count - verify_count)))"
            ((comparison_issues++))
        fi
    done
    
    if [[ ${comparison_issues} -gt 0 ]]; then
        log_warn "Found ${comparison_issues} tables with count differences"
        log_info "Note: Differences may be due to data changes since backup"
    else
        log_success "All table counts match"
    fi
    
    return 0
}

# =============================================================================
# 生成验证报告
# =============================================================================
generate_report() {
    local status="$1"
    
    log_info "Generating verification report..."
    
    cat > "${REPORT_FILE}" <<EOF
{
    "verification_timestamp": "${TIMESTAMP}",
    "backup_file": "${BACKUP_FILE}",
    "backup_size_bytes": $(stat -c %s "${BACKUP_FILE}" 2>/dev/null || stat -f %z "${BACKUP_FILE}"),
    "verification_database": "${VERIFY_DB_NAME}",
    "status": "${status}",
    "checks": {
        "file_integrity": true,
        "sql_content": true,
        "restore_test": ${QUICK_MODE:+false}${QUICK_MODE:-true},
        "data_validation": ${QUICK_MODE:+false}${QUICK_MODE:-true},
        "source_comparison": ${DO_COMPARE}
    },
    "log_file": "${LOG_FILE}"
}
EOF
    
    log_info "Report saved to: ${REPORT_FILE}"
}

# =============================================================================
# 主函数
# =============================================================================
main() {
    mkdir -p "${BACKUP_LOG_DIR}"
    
    parse_args "$@"
    
    log_info "=========================================="
    log_info "PostgreSQL Backup Verification"
    log_info "=========================================="
    
    resolve_backup_file
    
    local all_passed=true
    
    # 执行验证步骤
    if ! verify_file_integrity; then
        all_passed=false
    fi
    
    if ! verify_sql_content; then
        all_passed=false
    fi
    
    if ! test_restore; then
        all_passed=false
    fi
    
    compare_with_source
    
    # 生成报告
    if [[ "${all_passed}" == "true" ]]; then
        generate_report "PASSED"
        send_notification "success" "Backup verification passed: $(basename ${BACKUP_FILE})"
        log_success "=========================================="
        log_success "All verification checks PASSED"
        log_success "=========================================="
        exit 0
    else
        generate_report "FAILED"
        send_notification "error" "Backup verification failed: $(basename ${BACKUP_FILE})"
        log_error "=========================================="
        log_error "Verification FAILED"
        log_error "=========================================="
        exit 1
    fi
}

# =============================================================================
# 运行主函数
# =============================================================================
main "$@"

