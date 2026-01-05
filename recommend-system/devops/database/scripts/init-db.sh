#!/bin/bash
# =============================================================================
# PostgreSQL 数据库初始化脚本
# 
# 项目：生成式推荐系统
# 模块：数据库管理
# 版本：1.0.0
# 
# 功能：
#   - 创建数据库和用户
#   - 安装必要扩展
#   - 运行 Flyway 迁移
#   - 设置权限
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
DB_ADMIN_USER="${POSTGRES_ADMIN_USER:-postgres}"
DB_ADMIN_PASSWORD="${POSTGRES_ADMIN_PASSWORD:-${DB_PASSWORD}}"

# 应用用户（非超级用户）
APP_USER="${APP_DB_USER:-recommend_app}"
APP_PASSWORD="${APP_DB_PASSWORD:-}"

MIGRATIONS_DIR="${MIGRATIONS_DIR:-./migrations}"
FLYWAY_VERSION="${FLYWAY_VERSION:-9.22.3}"

LOG_FILE="${LOG_FILE:-/var/log/init-db.log}"

# =============================================================================
# 颜色输出
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# 日志函数
# =============================================================================
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local color=""
    
    case "$level" in
        "INFO") color="$BLUE" ;;
        "SUCCESS") color="$GREEN" ;;
        "WARN") color="$YELLOW" ;;
        "ERROR") color="$RED" ;;
    esac
    
    echo -e "${color}[${timestamp}] [${level}] ${message}${NC}" | tee -a "${LOG_FILE}"
}

log_info() { log "INFO" "$1"; }
log_success() { log "SUCCESS" "$1"; }
log_warn() { log "WARN" "$1"; }
log_error() { log "ERROR" "$1"; }

# =============================================================================
# 使用说明
# =============================================================================
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Initialize PostgreSQL database for Recommend System

Options:
  --create-db          Create database if not exists
  --create-users       Create application users
  --run-migrations     Run Flyway migrations
  --install-extensions Install required extensions
  --grant-permissions  Grant permissions to app user
  --full               Run full initialization (all steps)
  --skip-if-exists     Skip if database already exists
  -h, --help           Show this help message

Environment Variables:
  POSTGRES_HOST        Database host (default: localhost)
  POSTGRES_PORT        Database port (default: 5432)
  POSTGRES_DB          Database name (default: recommend)
  POSTGRES_USER        Admin user (default: postgres)
  POSTGRES_PASSWORD    Admin password (required)
  APP_DB_USER          Application user (default: recommend_app)
  APP_DB_PASSWORD      Application user password

Examples:
  # Full initialization
  $0 --full

  # Create database only
  $0 --create-db

  # Run migrations only
  $0 --run-migrations
EOF
    exit 0
}

# =============================================================================
# 参数解析
# =============================================================================
CREATE_DB=false
CREATE_USERS=false
RUN_MIGRATIONS=false
INSTALL_EXTENSIONS=false
GRANT_PERMISSIONS=false
SKIP_IF_EXISTS=false

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --create-db)
                CREATE_DB=true
                shift
                ;;
            --create-users)
                CREATE_USERS=true
                shift
                ;;
            --run-migrations)
                RUN_MIGRATIONS=true
                shift
                ;;
            --install-extensions)
                INSTALL_EXTENSIONS=true
                shift
                ;;
            --grant-permissions)
                GRANT_PERMISSIONS=true
                shift
                ;;
            --full)
                CREATE_DB=true
                CREATE_USERS=true
                RUN_MIGRATIONS=true
                INSTALL_EXTENSIONS=true
                GRANT_PERMISSIONS=true
                shift
                ;;
            --skip-if-exists)
                SKIP_IF_EXISTS=true
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
}

# =============================================================================
# 数据库连接测试
# =============================================================================
wait_for_postgres() {
    log_info "Waiting for PostgreSQL to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if PGPASSWORD="${DB_ADMIN_PASSWORD}" pg_isready \
            -h "${DB_HOST}" \
            -p "${DB_PORT}" \
            -U "${DB_ADMIN_USER}" > /dev/null 2>&1; then
            log_success "PostgreSQL is ready"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts - PostgreSQL not ready yet..."
        sleep 2
        ((attempt++))
    done
    
    log_error "PostgreSQL is not ready after $max_attempts attempts"
    return 1
}

# =============================================================================
# 创建数据库
# =============================================================================
create_database() {
    log_info "Creating database: ${DB_NAME}"
    
    # 检查数据库是否存在
    local exists=$(PGPASSWORD="${DB_ADMIN_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_ADMIN_USER}" \
        -d postgres \
        -t -c "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}';" | tr -d ' ')
    
    if [[ "$exists" == "1" ]]; then
        if [[ "$SKIP_IF_EXISTS" == "true" ]]; then
            log_info "Database ${DB_NAME} already exists, skipping"
            return 0
        else
            log_warn "Database ${DB_NAME} already exists"
            return 0
        fi
    fi
    
    # 创建数据库
    PGPASSWORD="${DB_ADMIN_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_ADMIN_USER}" \
        -d postgres \
        -c "CREATE DATABASE ${DB_NAME} WITH ENCODING='UTF8' LC_COLLATE='en_US.UTF-8' LC_CTYPE='en_US.UTF-8' TEMPLATE=template0;"
    
    log_success "Database ${DB_NAME} created successfully"
}

# =============================================================================
# 安装扩展
# =============================================================================
install_extensions() {
    log_info "Installing PostgreSQL extensions..."
    
    local extensions=(
        "uuid-ossp"
        "pg_trgm"
        "btree_gin"
    )
    
    for ext in "${extensions[@]}"; do
        log_info "Installing extension: ${ext}"
        PGPASSWORD="${DB_ADMIN_PASSWORD}" psql \
            -h "${DB_HOST}" \
            -p "${DB_PORT}" \
            -U "${DB_ADMIN_USER}" \
            -d "${DB_NAME}" \
            -c "CREATE EXTENSION IF NOT EXISTS \"${ext}\";"
    done
    
    # 尝试安装 pgvector（可能需要单独安装）
    log_info "Attempting to install pgvector extension..."
    PGPASSWORD="${DB_ADMIN_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_ADMIN_USER}" \
        -d "${DB_NAME}" \
        -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || \
        log_warn "pgvector extension not available, skipping"
    
    log_success "Extensions installed successfully"
}

# =============================================================================
# 创建用户
# =============================================================================
create_users() {
    log_info "Creating application users..."
    
    if [[ -z "$APP_PASSWORD" ]]; then
        log_warn "APP_DB_PASSWORD not set, generating random password"
        APP_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 24)
        log_info "Generated password for ${APP_USER}: ${APP_PASSWORD}"
    fi
    
    # 创建应用用户
    PGPASSWORD="${DB_ADMIN_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_ADMIN_USER}" \
        -d postgres \
        -c "
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${APP_USER}') THEN
                CREATE ROLE ${APP_USER} WITH LOGIN PASSWORD '${APP_PASSWORD}';
            ELSE
                ALTER ROLE ${APP_USER} WITH PASSWORD '${APP_PASSWORD}';
            END IF;
        END
        \$\$;
        "
    
    # 创建只读用户
    local readonly_user="${APP_USER}_readonly"
    PGPASSWORD="${DB_ADMIN_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_ADMIN_USER}" \
        -d postgres \
        -c "
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${readonly_user}') THEN
                CREATE ROLE ${readonly_user} WITH LOGIN PASSWORD '${APP_PASSWORD}_ro';
            END IF;
        END
        \$\$;
        "
    
    log_success "Users created successfully"
}

# =============================================================================
# 授权
# =============================================================================
grant_permissions() {
    log_info "Granting permissions to application user..."
    
    PGPASSWORD="${DB_ADMIN_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_ADMIN_USER}" \
        -d "${DB_NAME}" \
        -c "
        -- 授予连接权限
        GRANT CONNECT ON DATABASE ${DB_NAME} TO ${APP_USER};
        
        -- 授予 schema 使用权限
        GRANT USAGE ON SCHEMA public TO ${APP_USER};
        
        -- 授予表权限
        GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ${APP_USER};
        
        -- 授予序列权限
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ${APP_USER};
        
        -- 授予函数执行权限
        GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO ${APP_USER};
        
        -- 设置默认权限（针对未来创建的对象）
        ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO ${APP_USER};
        ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO ${APP_USER};
        ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO ${APP_USER};
        "
    
    # 只读用户权限
    local readonly_user="${APP_USER}_readonly"
    PGPASSWORD="${DB_ADMIN_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_ADMIN_USER}" \
        -d "${DB_NAME}" \
        -c "
        GRANT CONNECT ON DATABASE ${DB_NAME} TO ${readonly_user};
        GRANT USAGE ON SCHEMA public TO ${readonly_user};
        GRANT SELECT ON ALL TABLES IN SCHEMA public TO ${readonly_user};
        ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO ${readonly_user};
        " 2>/dev/null || true
    
    log_success "Permissions granted successfully"
}

# =============================================================================
# 运行迁移
# =============================================================================
run_migrations() {
    log_info "Running database migrations..."
    
    # 检查迁移目录
    if [[ ! -d "$MIGRATIONS_DIR" ]]; then
        log_error "Migrations directory not found: $MIGRATIONS_DIR"
        return 1
    fi
    
    # 使用 Flyway 或直接执行 SQL
    if command -v flyway &> /dev/null; then
        log_info "Using Flyway for migrations"
        
        flyway \
            -url="jdbc:postgresql://${DB_HOST}:${DB_PORT}/${DB_NAME}" \
            -user="${DB_ADMIN_USER}" \
            -password="${DB_ADMIN_PASSWORD}" \
            -locations="filesystem:${MIGRATIONS_DIR}" \
            -baselineOnMigrate=true \
            migrate
    else
        log_info "Flyway not found, executing SQL files directly"
        
        # 按版本顺序执行迁移文件
        for sql_file in $(ls -1 "${MIGRATIONS_DIR}"/V*.sql 2>/dev/null | sort -V); do
            log_info "Executing: $(basename $sql_file)"
            
            PGPASSWORD="${DB_ADMIN_PASSWORD}" psql \
                -h "${DB_HOST}" \
                -p "${DB_PORT}" \
                -U "${DB_ADMIN_USER}" \
                -d "${DB_NAME}" \
                -f "$sql_file" \
                -v ON_ERROR_STOP=1
            
            log_success "Completed: $(basename $sql_file)"
        done
    fi
    
    log_success "Migrations completed successfully"
}

# =============================================================================
# 验证初始化
# =============================================================================
verify_initialization() {
    log_info "Verifying database initialization..."
    
    # 检查表是否存在
    local core_tables=("users" "items" "item_stats" "user_behaviors" "user_profiles")
    local missing_tables=()
    
    for table in "${core_tables[@]}"; do
        local exists=$(PGPASSWORD="${DB_ADMIN_PASSWORD}" psql \
            -h "${DB_HOST}" \
            -p "${DB_PORT}" \
            -U "${DB_ADMIN_USER}" \
            -d "${DB_NAME}" \
            -t -c "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '${table}');" | tr -d ' ')
        
        if [[ "$exists" != "t" ]]; then
            missing_tables+=("$table")
        fi
    done
    
    if [[ ${#missing_tables[@]} -gt 0 ]]; then
        log_warn "Missing tables: ${missing_tables[*]}"
        return 1
    fi
    
    # 检查扩展
    local extensions=$(PGPASSWORD="${DB_ADMIN_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_ADMIN_USER}" \
        -d "${DB_NAME}" \
        -t -c "SELECT string_agg(extname, ', ') FROM pg_extension WHERE extname != 'plpgsql';")
    
    log_info "Installed extensions: ${extensions}"
    
    # 检查索引数量
    local index_count=$(PGPASSWORD="${DB_ADMIN_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_ADMIN_USER}" \
        -d "${DB_NAME}" \
        -t -c "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public';")
    
    log_info "Index count: ${index_count}"
    
    log_success "Database initialization verified"
    return 0
}

# =============================================================================
# 生成连接信息
# =============================================================================
generate_connection_info() {
    log_info "Database connection information:"
    echo ""
    echo "=========================================="
    echo "PostgreSQL Connection Details"
    echo "=========================================="
    echo "Host: ${DB_HOST}"
    echo "Port: ${DB_PORT}"
    echo "Database: ${DB_NAME}"
    echo "Admin User: ${DB_ADMIN_USER}"
    echo "App User: ${APP_USER}"
    echo ""
    echo "Connection String:"
    echo "postgresql://${APP_USER}:<password>@${DB_HOST}:${DB_PORT}/${DB_NAME}"
    echo ""
    echo "JDBC URL:"
    echo "jdbc:postgresql://${DB_HOST}:${DB_PORT}/${DB_NAME}"
    echo "=========================================="
}

# =============================================================================
# 主函数
# =============================================================================
main() {
    # 创建日志目录
    mkdir -p "$(dirname ${LOG_FILE})"
    
    log_info "=========================================="
    log_info "Starting Database Initialization"
    log_info "=========================================="
    
    parse_args "$@"
    
    # 如果没有指定任何操作，显示帮助
    if [[ "$CREATE_DB" == "false" ]] && \
       [[ "$CREATE_USERS" == "false" ]] && \
       [[ "$RUN_MIGRATIONS" == "false" ]] && \
       [[ "$INSTALL_EXTENSIONS" == "false" ]] && \
       [[ "$GRANT_PERMISSIONS" == "false" ]]; then
        usage
    fi
    
    # 等待数据库就绪
    wait_for_postgres
    
    # 执行各步骤
    [[ "$CREATE_DB" == "true" ]] && create_database
    [[ "$INSTALL_EXTENSIONS" == "true" ]] && install_extensions
    [[ "$CREATE_USERS" == "true" ]] && create_users
    [[ "$RUN_MIGRATIONS" == "true" ]] && run_migrations
    [[ "$GRANT_PERMISSIONS" == "true" ]] && grant_permissions
    
    # 验证
    verify_initialization
    
    # 输出连接信息
    generate_connection_info
    
    log_success "=========================================="
    log_success "Database Initialization Completed"
    log_success "=========================================="
}

# =============================================================================
# 运行主函数
# =============================================================================
main "$@"

