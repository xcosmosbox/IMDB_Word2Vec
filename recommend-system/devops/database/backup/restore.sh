#!/bin/bash
# =============================================================================
# PostgreSQL 数据库恢复脚本
# 
# 项目：生成式推荐系统
# 模块：数据库管理
# 版本：1.0.0
# 
# 功能：
#   - 从本地或 S3 恢复数据库
#   - 支持时间点恢复
#   - 恢复前验证
#   - 自动备份当前数据
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

S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-backups/postgres}"
AWS_REGION="${AWS_REGION:-us-east-1}"

SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

# =============================================================================
# 参数解析
# =============================================================================
BACKUP_FILE=""
DRY_RUN="${DRY_RUN:-false}"
SKIP_CONFIRM="${SKIP_CONFIRM:-false}"
CREATE_PRE_RESTORE_BACKUP="${CREATE_PRE_RESTORE_BACKUP:-true}"
DROP_EXISTING="${DROP_EXISTING:-true}"
TARGET_DB="${TARGET_DB:-}"  # 可选：恢复到不同的数据库
FROM_S3="${FROM_S3:-false}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${BACKUP_LOG_DIR}/restore_${TIMESTAMP}.log"

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

# =============================================================================
# 使用说明
# =============================================================================
usage() {
    cat <<EOF
Usage: $0 [OPTIONS] <backup_file>

PostgreSQL Database Restore Script

Arguments:
  backup_file           Path to the backup file or 'latest' for most recent backup

Options:
  --dry-run            Show what would be done without executing
  --skip-confirm       Skip confirmation prompt
  --no-pre-backup      Skip creating backup before restore
  --keep-existing      Don't drop existing database
  --target-db NAME     Restore to a different database name
  --from-s3            Download backup from S3
  -h, --help           Show this help message

Environment Variables:
  POSTGRES_HOST        Database host (default: localhost)
  POSTGRES_PORT        Database port (default: 5432)
  POSTGRES_DB          Database name (default: recommend)
  POSTGRES_USER        Database user (default: postgres)
  POSTGRES_PASSWORD    Database password (required)
  S3_BUCKET            S3 bucket for remote backups
  S3_PREFIX            S3 prefix path (default: backups/postgres)

Examples:
  # Restore from latest backup
  $0 latest

  # Restore from specific file
  $0 /backups/daily/recommend_20240101_120000.sql.gz

  # Dry run
  $0 --dry-run latest

  # Restore from S3
  $0 --from-s3 daily/recommend_20240101_120000.sql.gz

  # Restore to different database
  $0 --target-db recommend_test latest
EOF
    exit 0
}

# =============================================================================
# 参数解析
# =============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --skip-confirm)
                SKIP_CONFIRM="true"
                shift
                ;;
            --no-pre-backup)
                CREATE_PRE_RESTORE_BACKUP="false"
                shift
                ;;
            --keep-existing)
                DROP_EXISTING="false"
                shift
                ;;
            --target-db)
                TARGET_DB="$2"
                shift 2
                ;;
            --from-s3)
                FROM_S3="true"
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
    
    if [[ -z "${SLACK_WEBHOOK}" ]]; then
        return 0
    fi
    
    local emoji=""
    case "$status" in
        "success") emoji="✅" ;;
        "warning") emoji="⚠️" ;;
        "error") emoji="❌" ;;
        "info") emoji="ℹ️" ;;
    esac
    
    curl -s -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"${emoji} [Restore] ${message}\"}" \
        "${SLACK_WEBHOOK}" > /dev/null || true
}

# =============================================================================
# 错误处理
# =============================================================================
cleanup_on_error() {
    log_error "Restore failed!"
    send_notification "error" "Database restore failed for ${TARGET_DB:-${DB_NAME}}"
    exit 1
}

trap cleanup_on_error ERR

# =============================================================================
# 解析备份文件路径
# =============================================================================
resolve_backup_file() {
    local input="$1"
    
    # 处理 "latest" 快捷方式
    if [[ "${input}" == "latest" ]]; then
        BACKUP_FILE="${BACKUP_DIR}/${DB_NAME}_latest.sql.gz"
        if [[ ! -f "${BACKUP_FILE}" ]]; then
            log_error "Latest backup symlink not found: ${BACKUP_FILE}"
            exit 1
        fi
        # 解析实际文件
        BACKUP_FILE=$(readlink -f "${BACKUP_FILE}")
        log_info "Resolved 'latest' to: ${BACKUP_FILE}"
        return
    fi
    
    # 从 S3 下载
    if [[ "${FROM_S3}" == "true" ]]; then
        download_from_s3 "${input}"
        return
    fi
    
    # 处理相对路径
    if [[ ! "${input}" == /* ]]; then
        BACKUP_FILE="${BACKUP_DIR}/${input}"
    else
        BACKUP_FILE="${input}"
    fi
    
    if [[ ! -f "${BACKUP_FILE}" ]]; then
        log_error "Backup file not found: ${BACKUP_FILE}"
        exit 1
    fi
}

# =============================================================================
# 从 S3 下载备份
# =============================================================================
download_from_s3() {
    local s3_key="$1"
    local s3_path="s3://${S3_BUCKET}/${S3_PREFIX}/${s3_key}"
    local local_file="${BACKUP_DIR}/s3_restore_${TIMESTAMP}.sql.gz"
    
    log_info "Downloading backup from S3: ${s3_path}"
    
    aws s3 cp "${s3_path}" "${local_file}" --region "${AWS_REGION}"
    
    # 下载校验和（如果存在）
    local checksum_s3="${s3_path}.sha256"
    local checksum_local="${local_file}.sha256"
    
    if aws s3 ls "${checksum_s3}" --region "${AWS_REGION}" > /dev/null 2>&1; then
        aws s3 cp "${checksum_s3}" "${checksum_local}" --region "${AWS_REGION}"
        
        # 验证校验和
        log_info "Verifying S3 backup checksum..."
        cd "$(dirname ${local_file})"
        if ! sha256sum -c "${checksum_local}"; then
            log_error "Checksum verification failed!"
            rm -f "${local_file}" "${checksum_local}"
            exit 1
        fi
        log_info "Checksum verified"
    fi
    
    BACKUP_FILE="${local_file}"
    log_info "Downloaded to: ${BACKUP_FILE}"
}

# =============================================================================
# 验证备份文件
# =============================================================================
verify_backup() {
    log_info "Verifying backup file integrity..."
    
    if ! gzip -t "${BACKUP_FILE}"; then
        log_error "Backup file is corrupted: ${BACKUP_FILE}"
        exit 1
    fi
    
    log_info "Backup file integrity verified"
    
    # 显示备份信息
    local file_size=$(du -h "${BACKUP_FILE}" | cut -f1)
    local file_date=$(stat -c %y "${BACKUP_FILE}" 2>/dev/null || stat -f %Sm "${BACKUP_FILE}")
    
    log_info "Backup file size: ${file_size}"
    log_info "Backup file date: ${file_date}"
}

# =============================================================================
# 创建恢复前备份
# =============================================================================
create_pre_restore_backup() {
    if [[ "${CREATE_PRE_RESTORE_BACKUP}" != "true" ]]; then
        log_info "Skipping pre-restore backup"
        return
    fi
    
    local pre_backup_file="${BACKUP_DIR}/pre_restore_${DB_NAME}_${TIMESTAMP}.sql.gz"
    
    log_info "Creating pre-restore backup: ${pre_backup_file}"
    
    PGPASSWORD="${DB_PASSWORD}" pg_dump \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        --format=plain \
        --no-owner \
        --no-privileges \
        2>> "${LOG_FILE}" | gzip > "${pre_backup_file}"
    
    log_info "Pre-restore backup created successfully"
}

# =============================================================================
# 执行恢复
# =============================================================================
do_restore() {
    local target_db="${TARGET_DB:-${DB_NAME}}"
    
    log_info "Starting restore to database: ${target_db}"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would restore ${BACKUP_FILE} to ${target_db}"
        log_info "[DRY RUN] DROP_EXISTING: ${DROP_EXISTING}"
        log_info "[DRY RUN] CREATE_PRE_RESTORE_BACKUP: ${CREATE_PRE_RESTORE_BACKUP}"
        return
    fi
    
    # 确认
    if [[ "${SKIP_CONFIRM}" != "true" ]]; then
        echo ""
        echo "=========================================="
        echo "WARNING: This will restore the database!"
        echo "=========================================="
        echo "Source: ${BACKUP_FILE}"
        echo "Target: ${target_db} @ ${DB_HOST}:${DB_PORT}"
        echo "Drop existing: ${DROP_EXISTING}"
        echo ""
        read -p "Are you sure you want to continue? (yes/no): " confirm
        
        if [[ "${confirm}" != "yes" ]]; then
            log_info "Restore cancelled by user"
            exit 0
        fi
    fi
    
    # 通知开始恢复
    send_notification "info" "Starting database restore to ${target_db}"
    
    # 创建恢复前备份
    if [[ "${target_db}" == "${DB_NAME}" ]]; then
        create_pre_restore_backup
    fi
    
    if [[ "${DROP_EXISTING}" == "true" ]]; then
        # 断开现有连接
        log_info "Terminating existing connections..."
        PGPASSWORD="${DB_PASSWORD}" psql \
            -h "${DB_HOST}" \
            -p "${DB_PORT}" \
            -U "${DB_USER}" \
            -d postgres \
            -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${target_db}' AND pid <> pg_backend_pid();" \
            2>> "${LOG_FILE}" || true
        
        # 删除数据库
        log_info "Dropping database: ${target_db}"
        PGPASSWORD="${DB_PASSWORD}" dropdb \
            -h "${DB_HOST}" \
            -p "${DB_PORT}" \
            -U "${DB_USER}" \
            "${target_db}" \
            --if-exists
        
        # 创建数据库
        log_info "Creating database: ${target_db}"
        PGPASSWORD="${DB_PASSWORD}" createdb \
            -h "${DB_HOST}" \
            -p "${DB_PORT}" \
            -U "${DB_USER}" \
            "${target_db}"
    fi
    
    # 恢复数据
    log_info "Restoring data from backup..."
    local start_time=$(date +%s)
    
    gunzip -c "${BACKUP_FILE}" | PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${target_db}" \
        --quiet \
        2>> "${LOG_FILE}"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_info "Data restore completed in ${duration} seconds"
    
    # 验证恢复结果
    verify_restore "${target_db}"
    
    # 分析数据库
    log_info "Analyzing database..."
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${target_db}" \
        -c "ANALYZE;"
    
    send_notification "success" "Database restore completed successfully in ${duration}s"
    
    log_info "=========================================="
    log_info "Restore completed successfully!"
    log_info "=========================================="
}

# =============================================================================
# 验证恢复结果
# =============================================================================
verify_restore() {
    local target_db="$1"
    
    log_info "Verifying restore result..."
    
    # 检查表数量
    local table_count=$(PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${target_db}" \
        -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
    
    log_info "Restored tables: ${table_count}"
    
    # 检查核心表是否存在
    local core_tables=("users" "items" "item_stats" "user_behaviors" "user_profiles")
    local missing_tables=()
    
    for table in "${core_tables[@]}"; do
        local exists=$(PGPASSWORD="${DB_PASSWORD}" psql \
            -h "${DB_HOST}" \
            -p "${DB_PORT}" \
            -U "${DB_USER}" \
            -d "${target_db}" \
            -t -c "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '${table}');")
        
        if [[ "${exists}" == *"f"* ]]; then
            missing_tables+=("${table}")
        fi
    done
    
    if [[ ${#missing_tables[@]} -gt 0 ]]; then
        log_warn "Missing core tables: ${missing_tables[*]}"
    else
        log_info "All core tables verified"
    fi
    
    # 检查数据量
    for table in "users" "items"; do
        local count=$(PGPASSWORD="${DB_PASSWORD}" psql \
            -h "${DB_HOST}" \
            -p "${DB_PORT}" \
            -U "${DB_USER}" \
            -d "${target_db}" \
            -t -c "SELECT COUNT(*) FROM ${table};" 2>/dev/null || echo "0")
        log_info "Table ${table} row count: ${count}"
    done
}

# =============================================================================
# 列出可用备份
# =============================================================================
list_backups() {
    echo "Available local backups:"
    echo ""
    
    for type in daily weekly monthly; do
        if [[ -d "${BACKUP_DIR}/${type}" ]]; then
            echo "=== ${type} ==="
            ls -lh "${BACKUP_DIR}/${type}"/*.sql.gz 2>/dev/null || echo "  No backups found"
            echo ""
        fi
    done
    
    if [[ -n "${S3_BUCKET}" ]]; then
        echo "=== S3 Backups ==="
        aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" --recursive | tail -20
    fi
}

# =============================================================================
# 主函数
# =============================================================================
main() {
    mkdir -p "${BACKUP_LOG_DIR}"
    
    if [[ "${1:-}" == "list" ]]; then
        list_backups
        exit 0
    fi
    
    parse_args "$@"
    
    log_info "=========================================="
    log_info "PostgreSQL Database Restore"
    log_info "=========================================="
    
    resolve_backup_file "${BACKUP_FILE}"
    verify_backup
    do_restore
}

# =============================================================================
# 运行主函数
# =============================================================================
main "$@"

