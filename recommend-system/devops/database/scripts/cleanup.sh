#!/bin/bash
# =============================================================================
# 数据库清理脚本
# 
# 项目：生成式推荐系统
# 模块：数据库管理
# 版本：1.0.0
# 
# 功能：
#   - 清理过期数据
#   - 清理旧分区
#   - 回收存储空间
#   - 更新统计信息
# =============================================================================

set -euo pipefail

# =============================================================================
# 配置变量
# =============================================================================
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-recommend}"
DB_USER="${POSTGRES_USER:-postgres}"
DB_PASSWORD="${POSTGRES_PASSWORD:-}"

# 数据保留策略
BEHAVIOR_RETENTION_DAYS="${BEHAVIOR_RETENTION_DAYS:-90}"
REQUEST_RETENTION_DAYS="${REQUEST_RETENTION_DAYS:-30}"
SESSION_RETENTION_DAYS="${SESSION_RETENTION_DAYS:-7}"
LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-30}"

# 日志
LOG_FILE="${LOG_FILE:-/var/log/cleanup.log}"

# 通知
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

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
log_success() { log "SUCCESS" "$1"; }
log_warn() { log "WARN" "$1"; }
log_error() { log "ERROR" "$1"; }

# =============================================================================
# 执行 SQL
# =============================================================================
exec_sql() {
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        -v ON_ERROR_STOP=1 \
        "$@"
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
        --data "{\"text\":\"${emoji} [DB Cleanup] ${message}\"}" \
        "${SLACK_WEBHOOK}" > /dev/null || true
}

# =============================================================================
# 清理用户行为数据
# =============================================================================
cleanup_behaviors() {
    log_info "Cleaning up user behaviors older than ${BEHAVIOR_RETENTION_DAYS} days..."
    
    local cutoff_date=$(date -d "-${BEHAVIOR_RETENTION_DAYS} days" '+%Y-%m-%d')
    
    local deleted=$(exec_sql -t -c "
        WITH deleted AS (
            DELETE FROM user_behaviors
            WHERE created_at < '${cutoff_date}'::timestamp
            RETURNING 1
        )
        SELECT COUNT(*) FROM deleted;
    " | tr -d ' ')
    
    log_success "Deleted ${deleted} old behavior records"
    echo "${deleted}"
}

# =============================================================================
# 清理推荐请求记录
# =============================================================================
cleanup_requests() {
    log_info "Cleaning up recommendation requests older than ${REQUEST_RETENTION_DAYS} days..."
    
    local cutoff_date=$(date -d "-${REQUEST_RETENTION_DAYS} days" '+%Y-%m-%d')
    
    # 先删除推荐结果
    local results_deleted=$(exec_sql -t -c "
        WITH deleted AS (
            DELETE FROM recommendation_results
            WHERE created_at < '${cutoff_date}'::timestamp
            RETURNING 1
        )
        SELECT COUNT(*) FROM deleted;
    " | tr -d ' ')
    
    log_info "Deleted ${results_deleted} old recommendation results"
    
    # 再删除推荐请求
    local requests_deleted=$(exec_sql -t -c "
        WITH deleted AS (
            DELETE FROM recommendation_requests
            WHERE created_at < '${cutoff_date}'::timestamp
            RETURNING 1
        )
        SELECT COUNT(*) FROM deleted;
    " | tr -d ' ')
    
    log_success "Deleted ${requests_deleted} old recommendation requests"
}

# =============================================================================
# 清理用户会话
# =============================================================================
cleanup_sessions() {
    log_info "Cleaning up user sessions older than ${SESSION_RETENTION_DAYS} days..."
    
    local cutoff_date=$(date -d "-${SESSION_RETENTION_DAYS} days" '+%Y-%m-%d')
    
    local deleted=$(exec_sql -t -c "
        WITH deleted AS (
            DELETE FROM user_sessions
            WHERE started_at < '${cutoff_date}'::timestamp
            RETURNING 1
        )
        SELECT COUNT(*) FROM deleted;
    " | tr -d ' ' 2>/dev/null || echo "0")
    
    log_success "Deleted ${deleted} old session records"
}

# =============================================================================
# 删除旧分区
# =============================================================================
cleanup_old_partitions() {
    log_info "Checking for old partitions to drop..."
    
    local cutoff_date=$(date -d "-${BEHAVIOR_RETENTION_DAYS} days" '+%Y_%m')
    
    exec_sql <<EOF
DO \$\$
DECLARE
    partition_name TEXT;
    partition_date TEXT;
BEGIN
    FOR partition_name IN 
        SELECT c.relname 
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind = 'r'
        AND n.nspname = 'public'
        AND c.relname LIKE 'user_behaviors_%'
        AND c.relname ~ '^user_behaviors_[0-9]{4}_[0-9]{2}$'
    LOOP
        -- 提取分区日期
        partition_date := substring(partition_name from 'user_behaviors_(.+)');
        
        -- 比较日期（简单字符串比较）
        IF partition_date < '${cutoff_date}' THEN
            RAISE NOTICE 'Dropping partition: %', partition_name;
            EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(partition_name);
        END IF;
    END LOOP;
END;
\$\$;
EOF
    
    log_success "Old partitions cleaned up"
}

# =============================================================================
# 清理已删除的用户和物品
# =============================================================================
cleanup_deleted_records() {
    log_info "Cleaning up records marked as deleted..."
    
    # 删除标记为 deleted 的用户（软删除超过 30 天）
    local users_deleted=$(exec_sql -t -c "
        WITH deleted AS (
            DELETE FROM users
            WHERE status = 'deleted'
            AND updated_at < NOW() - INTERVAL '30 days'
            RETURNING 1
        )
        SELECT COUNT(*) FROM deleted;
    " | tr -d ' ')
    
    log_info "Permanently deleted ${users_deleted} soft-deleted users"
    
    # 删除标记为 deleted 的物品
    local items_deleted=$(exec_sql -t -c "
        WITH deleted AS (
            DELETE FROM items
            WHERE status = 'deleted'
            AND updated_at < NOW() - INTERVAL '30 days'
            RETURNING 1
        )
        SELECT COUNT(*) FROM deleted;
    " | tr -d ' ')
    
    log_info "Permanently deleted ${items_deleted} soft-deleted items"
}

# =============================================================================
# 清理孤立数据
# =============================================================================
cleanup_orphan_data() {
    log_info "Cleaning up orphan data..."
    
    # 清理没有对应物品的统计数据
    local orphan_stats=$(exec_sql -t -c "
        WITH deleted AS (
            DELETE FROM item_stats
            WHERE item_id NOT IN (SELECT id FROM items)
            RETURNING 1
        )
        SELECT COUNT(*) FROM deleted;
    " | tr -d ' ')
    
    log_info "Deleted ${orphan_stats} orphan item_stats records"
    
    # 清理没有对应用户的画像数据
    local orphan_profiles=$(exec_sql -t -c "
        WITH deleted AS (
            DELETE FROM user_profiles
            WHERE user_id NOT IN (SELECT id FROM users)
            RETURNING 1
        )
        SELECT COUNT(*) FROM deleted;
    " | tr -d ' ')
    
    log_info "Deleted ${orphan_profiles} orphan user_profiles records"
}

# =============================================================================
# 回收存储空间
# =============================================================================
reclaim_storage() {
    log_info "Reclaiming storage space..."
    
    # VACUUM ANALYZE
    exec_sql -c "VACUUM ANALYZE users;"
    exec_sql -c "VACUUM ANALYZE items;"
    exec_sql -c "VACUUM ANALYZE item_stats;"
    exec_sql -c "VACUUM ANALYZE user_profiles;"
    
    log_info "Running VACUUM on partitioned tables..."
    exec_sql -c "VACUUM ANALYZE user_behaviors;"
    
    log_success "Storage reclaimed"
}

# =============================================================================
# 更新统计信息
# =============================================================================
update_statistics() {
    log_info "Updating database statistics..."
    
    exec_sql -c "ANALYZE;"
    
    log_success "Statistics updated"
}

# =============================================================================
# 刷新物化视图
# =============================================================================
refresh_materialized_views() {
    log_info "Refreshing materialized views..."
    
    exec_sql <<EOF
DO \$\$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE matviewname = 'mv_user_daily_stats') THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_daily_stats;
        RAISE NOTICE 'Refreshed mv_user_daily_stats';
    END IF;
    
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE matviewname = 'mv_item_daily_stats') THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_item_daily_stats;
        RAISE NOTICE 'Refreshed mv_item_daily_stats';
    END IF;
    
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE matviewname = 'mv_trending_items') THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_trending_items;
        RAISE NOTICE 'Refreshed mv_trending_items';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Error refreshing materialized views: %', SQLERRM;
END;
\$\$;
EOF
    
    log_success "Materialized views refreshed"
}

# =============================================================================
# 生成清理报告
# =============================================================================
generate_report() {
    log_info "Generating cleanup report..."
    
    local report=$(exec_sql -t <<EOF
SELECT json_build_object(
    'timestamp', NOW(),
    'tables', json_build_object(
        'users', (SELECT COUNT(*) FROM users),
        'items', (SELECT COUNT(*) FROM items),
        'user_behaviors', (SELECT COUNT(*) FROM user_behaviors),
        'item_stats', (SELECT COUNT(*) FROM item_stats),
        'user_profiles', (SELECT COUNT(*) FROM user_profiles)
    ),
    'database_size', pg_size_pretty(pg_database_size('${DB_NAME}')),
    'largest_tables', (
        SELECT json_agg(t)
        FROM (
            SELECT 
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS size
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC
            LIMIT 10
        ) t
    )
);
EOF
)
    
    echo "${report}" | tee -a "${LOG_FILE}"
}

# =============================================================================
# 使用说明
# =============================================================================
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Database cleanup and maintenance script

Options:
  --behaviors-days N    Retention days for behaviors (default: ${BEHAVIOR_RETENTION_DAYS})
  --requests-days N     Retention days for requests (default: ${REQUEST_RETENTION_DAYS})
  --sessions-days N     Retention days for sessions (default: ${SESSION_RETENTION_DAYS})
  --skip-vacuum         Skip VACUUM operation
  --dry-run             Show what would be done without executing
  --report-only         Only generate report
  -h, --help            Show this help message

Examples:
  # Full cleanup with defaults
  $0

  # Custom retention periods
  $0 --behaviors-days 60 --requests-days 14

  # Dry run
  $0 --dry-run

  # Report only
  $0 --report-only
EOF
    exit 0
}

# =============================================================================
# 参数解析
# =============================================================================
SKIP_VACUUM=false
DRY_RUN=false
REPORT_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --behaviors-days)
            BEHAVIOR_RETENTION_DAYS="$2"
            shift 2
            ;;
        --requests-days)
            REQUEST_RETENTION_DAYS="$2"
            shift 2
            ;;
        --sessions-days)
            SESSION_RETENTION_DAYS="$2"
            shift 2
            ;;
        --skip-vacuum)
            SKIP_VACUUM=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --report-only)
            REPORT_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# =============================================================================
# 主函数
# =============================================================================
main() {
    mkdir -p "$(dirname ${LOG_FILE})"
    
    log_info "=========================================="
    log_info "Starting Database Cleanup"
    log_info "=========================================="
    log_info "Retention: Behaviors=${BEHAVIOR_RETENTION_DAYS}d, Requests=${REQUEST_RETENTION_DAYS}d, Sessions=${SESSION_RETENTION_DAYS}d"
    
    if [[ "$REPORT_ONLY" == "true" ]]; then
        generate_report
        exit 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would clean up data with the following retention policies:"
        log_info "[DRY RUN] - Behaviors: ${BEHAVIOR_RETENTION_DAYS} days"
        log_info "[DRY RUN] - Requests: ${REQUEST_RETENTION_DAYS} days"
        log_info "[DRY RUN] - Sessions: ${SESSION_RETENTION_DAYS} days"
        exit 0
    fi
    
    local start_time=$(date +%s)
    
    # 执行清理
    cleanup_behaviors
    cleanup_requests
    cleanup_sessions
    cleanup_old_partitions
    cleanup_deleted_records
    cleanup_orphan_data
    
    # 维护操作
    if [[ "$SKIP_VACUUM" != "true" ]]; then
        reclaim_storage
    fi
    
    update_statistics
    refresh_materialized_views
    
    # 生成报告
    generate_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "=========================================="
    log_success "Database Cleanup Completed in ${duration}s"
    log_success "=========================================="
    
    send_notification "success" "Database cleanup completed in ${duration}s"
}

main

