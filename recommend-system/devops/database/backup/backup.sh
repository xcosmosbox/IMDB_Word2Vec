#!/bin/bash
# =============================================================================
# PostgreSQL 数据库备份脚本
# 
# 项目：生成式推荐系统
# 模块：数据库管理
# 版本：1.0.0
# 
# 功能：
#   - 全量数据库备份
#   - 支持 S3 上传
#   - 自动清理过期备份
#   - Slack 通知
#   - 备份验证
# =============================================================================

set -euo pipefail

# =============================================================================
# 配置变量
# =============================================================================
# 备份目录配置
BACKUP_DIR="${BACKUP_DIR:-/backups}"
BACKUP_LOG_DIR="${BACKUP_LOG_DIR:-/var/log/backup}"

# 数据库连接配置
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-recommend}"
DB_USER="${POSTGRES_USER:-postgres}"
DB_PASSWORD="${POSTGRES_PASSWORD:-}"

# 备份保留策略
RETENTION_DAYS="${RETENTION_DAYS:-30}"
RETENTION_WEEKLY="${RETENTION_WEEKLY:-12}"
RETENTION_MONTHLY="${RETENTION_MONTHLY:-12}"

# S3 配置（可选）
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-backups/postgres}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# 通知配置
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
ENABLE_NOTIFICATION="${ENABLE_NOTIFICATION:-true}"

# 压缩配置
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-6}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

# =============================================================================
# 时间戳和文件名
# =============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE_ONLY=$(date +%Y%m%d)
DAY_OF_WEEK=$(date +%u)
DAY_OF_MONTH=$(date +%d)

# 根据日期确定备份类型
if [[ "$DAY_OF_MONTH" == "01" ]]; then
    BACKUP_TYPE="monthly"
elif [[ "$DAY_OF_WEEK" == "7" ]]; then
    BACKUP_TYPE="weekly"
else
    BACKUP_TYPE="daily"
fi

BACKUP_FILE="${BACKUP_DIR}/${BACKUP_TYPE}/${DB_NAME}_${TIMESTAMP}.sql.gz"
BACKUP_FILE_LATEST="${BACKUP_DIR}/${DB_NAME}_latest.sql.gz"
CHECKSUM_FILE="${BACKUP_FILE}.sha256"
LOG_FILE="${BACKUP_LOG_DIR}/backup_${DATE_ONLY}.log"

# =============================================================================
# 日志函数
# =============================================================================
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() {
    log "INFO" "$1"
}

log_warn() {
    log "WARN" "$1"
}

log_error() {
    log "ERROR" "$1"
}

# =============================================================================
# 通知函数
# =============================================================================
send_notification() {
    local status="$1"
    local message="$2"
    
    if [[ "${ENABLE_NOTIFICATION}" != "true" ]] || [[ -z "${SLACK_WEBHOOK}" ]]; then
        return 0
    fi
    
    local emoji=""
    local color=""
    case "$status" in
        "success")
            emoji="✅"
            color="good"
            ;;
        "warning")
            emoji="⚠️"
            color="warning"
            ;;
        "error")
            emoji="❌"
            color="danger"
            ;;
    esac
    
    local payload=$(cat <<EOF
{
    "attachments": [{
        "color": "${color}",
        "title": "Database Backup: ${DB_NAME}",
        "text": "${emoji} ${message}",
        "fields": [
            {"title": "Type", "value": "${BACKUP_TYPE}", "short": true},
            {"title": "Host", "value": "${DB_HOST}", "short": true},
            {"title": "Timestamp", "value": "${TIMESTAMP}", "short": true}
        ],
        "footer": "Recommend System Backup",
        "ts": $(date +%s)
    }]
}
EOF
)
    
    curl -s -X POST -H 'Content-type: application/json' \
        --data "${payload}" \
        "${SLACK_WEBHOOK}" > /dev/null || log_warn "Failed to send Slack notification"
}

# =============================================================================
# 错误处理
# =============================================================================
cleanup_on_error() {
    log_error "Backup failed, cleaning up..."
    
    # 删除不完整的备份文件
    [[ -f "${BACKUP_FILE}" ]] && rm -f "${BACKUP_FILE}"
    [[ -f "${CHECKSUM_FILE}" ]] && rm -f "${CHECKSUM_FILE}"
    
    send_notification "error" "Backup failed for database ${DB_NAME}"
    exit 1
}

trap cleanup_on_error ERR

# =============================================================================
# 主函数
# =============================================================================
main() {
    log_info "=========================================="
    log_info "Starting ${BACKUP_TYPE} backup for database: ${DB_NAME}"
    log_info "Backup file: ${BACKUP_FILE}"
    log_info "=========================================="
    
    # 创建目录
    mkdir -p "${BACKUP_DIR}/daily"
    mkdir -p "${BACKUP_DIR}/weekly"
    mkdir -p "${BACKUP_DIR}/monthly"
    mkdir -p "${BACKUP_LOG_DIR}"
    
    # 检查数据库连接
    log_info "Testing database connection..."
    if ! PGPASSWORD="${DB_PASSWORD}" pg_isready -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" > /dev/null 2>&1; then
        log_error "Cannot connect to database at ${DB_HOST}:${DB_PORT}"
        cleanup_on_error
    fi
    log_info "Database connection successful"
    
    # 获取数据库大小
    local db_size=$(PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
        -t -c "SELECT pg_size_pretty(pg_database_size('${DB_NAME}'));")
    log_info "Database size: ${db_size}"
    
    # 执行备份
    log_info "Starting pg_dump..."
    local start_time=$(date +%s)
    
    PGPASSWORD="${DB_PASSWORD}" pg_dump \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        --format=plain \
        --no-owner \
        --no-privileges \
        --no-comments \
        --verbose \
        --jobs="${PARALLEL_JOBS}" \
        2>> "${LOG_FILE}" | gzip -"${COMPRESSION_LEVEL}" > "${BACKUP_FILE}"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_info "pg_dump completed in ${duration} seconds"
    
    # 获取备份文件大小
    local backup_size=$(du -h "${BACKUP_FILE}" | cut -f1)
    log_info "Backup file size: ${backup_size}"
    
    # 生成校验和
    log_info "Generating checksum..."
    sha256sum "${BACKUP_FILE}" > "${CHECKSUM_FILE}"
    log_info "Checksum: $(cat ${CHECKSUM_FILE})"
    
    # 创建最新备份链接
    ln -sf "${BACKUP_FILE}" "${BACKUP_FILE_LATEST}"
    log_info "Created symlink: ${BACKUP_FILE_LATEST}"
    
    # 验证备份完整性
    log_info "Verifying backup integrity..."
    if ! gzip -t "${BACKUP_FILE}"; then
        log_error "Backup file is corrupted!"
        cleanup_on_error
    fi
    log_info "Backup integrity verified"
    
    # 上传到 S3
    if [[ -n "${S3_BUCKET}" ]]; then
        upload_to_s3
    fi
    
    # 清理旧备份
    cleanup_old_backups
    
    # 生成备份报告
    generate_backup_report "${duration}" "${backup_size}"
    
    # 发送成功通知
    send_notification "success" "Backup completed successfully. Size: ${backup_size}, Duration: ${duration}s"
    
    log_info "=========================================="
    log_info "Backup completed successfully!"
    log_info "=========================================="
}

# =============================================================================
# S3 上传函数
# =============================================================================
upload_to_s3() {
    log_info "Uploading backup to S3..."
    
    local s3_path="s3://${S3_BUCKET}/${S3_PREFIX}/${BACKUP_TYPE}/"
    
    # 上传备份文件
    aws s3 cp "${BACKUP_FILE}" "${s3_path}" \
        --region "${AWS_REGION}" \
        --storage-class STANDARD_IA
    
    # 上传校验和
    aws s3 cp "${CHECKSUM_FILE}" "${s3_path}" \
        --region "${AWS_REGION}"
    
    log_info "S3 upload completed: ${s3_path}$(basename ${BACKUP_FILE})"
}

# =============================================================================
# 清理旧备份函数
# =============================================================================
cleanup_old_backups() {
    log_info "Cleaning up old backups..."
    
    # 清理日备份
    local daily_deleted=$(find "${BACKUP_DIR}/daily" -name "${DB_NAME}_*.sql.gz" -type f -mtime +"${RETENTION_DAYS}" -delete -print | wc -l)
    log_info "Deleted ${daily_deleted} daily backups older than ${RETENTION_DAYS} days"
    
    # 清理周备份
    local weekly_deleted=$(find "${BACKUP_DIR}/weekly" -name "${DB_NAME}_*.sql.gz" -type f -mtime +$((RETENTION_WEEKLY * 7)) -delete -print | wc -l)
    log_info "Deleted ${weekly_deleted} weekly backups older than ${RETENTION_WEEKLY} weeks"
    
    # 清理月备份
    local monthly_deleted=$(find "${BACKUP_DIR}/monthly" -name "${DB_NAME}_*.sql.gz" -type f -mtime +$((RETENTION_MONTHLY * 30)) -delete -print | wc -l)
    log_info "Deleted ${monthly_deleted} monthly backups older than ${RETENTION_MONTHLY} months"
    
    # 清理对应的校验和文件
    find "${BACKUP_DIR}" -name "*.sha256" -type f | while read checksum_file; do
        local backup_file="${checksum_file%.sha256}"
        if [[ ! -f "${backup_file}" ]]; then
            rm -f "${checksum_file}"
        fi
    done
    
    # 统计当前备份数量
    local daily_count=$(find "${BACKUP_DIR}/daily" -name "${DB_NAME}_*.sql.gz" -type f | wc -l)
    local weekly_count=$(find "${BACKUP_DIR}/weekly" -name "${DB_NAME}_*.sql.gz" -type f | wc -l)
    local monthly_count=$(find "${BACKUP_DIR}/monthly" -name "${DB_NAME}_*.sql.gz" -type f | wc -l)
    
    log_info "Current backup count - Daily: ${daily_count}, Weekly: ${weekly_count}, Monthly: ${monthly_count}"
}

# =============================================================================
# 生成备份报告
# =============================================================================
generate_backup_report() {
    local duration="$1"
    local backup_size="$2"
    
    local report_file="${BACKUP_DIR}/backup_report_${DATE_ONLY}.json"
    
    cat > "${report_file}" <<EOF
{
    "backup_type": "${BACKUP_TYPE}",
    "database": "${DB_NAME}",
    "host": "${DB_HOST}",
    "timestamp": "${TIMESTAMP}",
    "backup_file": "${BACKUP_FILE}",
    "backup_size": "${backup_size}",
    "duration_seconds": ${duration},
    "checksum_file": "${CHECKSUM_FILE}",
    "s3_uploaded": $([ -n "${S3_BUCKET}" ] && echo "true" || echo "false"),
    "retention": {
        "daily_days": ${RETENTION_DAYS},
        "weekly_weeks": ${RETENTION_WEEKLY},
        "monthly_months": ${RETENTION_MONTHLY}
    },
    "success": true
}
EOF
    
    log_info "Backup report generated: ${report_file}"
}

# =============================================================================
# 运行主函数
# =============================================================================
main "$@"

