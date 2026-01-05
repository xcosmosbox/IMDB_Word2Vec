#!/bin/bash
# =============================================================================
# 数据库种子数据脚本
# 
# 项目：生成式推荐系统
# 模块：数据库管理
# 版本：1.0.0
# 
# 功能：
#   - 插入测试用户数据
#   - 插入测试物品数据
#   - 生成模拟行为数据
#   - 初始化统计数据
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

# 数据量配置
NUM_USERS="${NUM_USERS:-1000}"
NUM_ITEMS="${NUM_ITEMS:-5000}"
NUM_BEHAVIORS="${NUM_BEHAVIORS:-50000}"

# =============================================================================
# 日志函数
# =============================================================================
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1"
}

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
# 生成用户数据
# =============================================================================
seed_users() {
    log_info "Seeding ${NUM_USERS} users..."
    
    exec_sql <<EOF
-- 生成测试用户
INSERT INTO users (name, email, password_hash, age, gender, country, city, metadata, status)
SELECT 
    'User_' || i AS name,
    'user' || i || '@example.com' AS email,
    'hash_' || md5(random()::text) AS password_hash,
    (random() * 50 + 18)::int AS age,
    (ARRAY['male', 'female', 'other'])[floor(random() * 3 + 1)] AS gender,
    (ARRAY['China', 'USA', 'Japan', 'UK', 'Germany'])[floor(random() * 5 + 1)] AS country,
    (ARRAY['Beijing', 'Shanghai', 'New York', 'Tokyo', 'London'])[floor(random() * 5 + 1)] AS city,
    jsonb_build_object(
        'interests', ARRAY['movie', 'music', 'sports', 'tech', 'food'][(random() * 5 + 1)::int],
        'registration_source', (ARRAY['web', 'mobile', 'social'])[floor(random() * 3 + 1)]
    ) AS metadata,
    'active' AS status
FROM generate_series(1, ${NUM_USERS}) AS i
ON CONFLICT (email) DO NOTHING;
EOF
    
    log_success "Users seeded"
}

# =============================================================================
# 生成物品数据
# =============================================================================
seed_items() {
    log_info "Seeding ${NUM_ITEMS} items..."
    
    exec_sql <<EOF
-- 生成测试物品
INSERT INTO items (type, title, description, category, subcategory, tags, semantic_id, metadata, status, published_at)
SELECT 
    (ARRAY['movie', 'product', 'article', 'video'])[floor(random() * 4 + 1)] AS type,
    'Item_' || i || '_' || (ARRAY['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Documentary'])[floor(random() * 5 + 1)] AS title,
    'This is a description for item ' || i || '. It contains interesting content.' AS description,
    (ARRAY['Entertainment', 'Electronics', 'Fashion', 'Education', 'Sports'])[floor(random() * 5 + 1)] AS category,
    (ARRAY['SubA', 'SubB', 'SubC', 'SubD'])[floor(random() * 4 + 1)] AS subcategory,
    ARRAY[
        (ARRAY['popular', 'trending', 'new', 'classic', 'featured'])[floor(random() * 5 + 1)],
        (ARRAY['recommended', 'hot', 'sale', 'limited', 'exclusive'])[floor(random() * 5 + 1)]
    ] AS tags,
    ARRAY[
        (random() * 1024)::int,
        (random() * 4096)::int,
        (random() * 16384)::int
    ] AS semantic_id,
    jsonb_build_object(
        'price', (random() * 1000)::numeric(10,2),
        'rating', (random() * 5)::numeric(3,2),
        'views', (random() * 10000)::int
    ) AS metadata,
    'active' AS status,
    NOW() - (random() * INTERVAL '365 days') AS published_at
FROM generate_series(1, ${NUM_ITEMS}) AS i;
EOF
    
    log_success "Items seeded"
}

# =============================================================================
# 生成行为数据
# =============================================================================
seed_behaviors() {
    log_info "Seeding ${NUM_BEHAVIORS} user behaviors..."
    
    exec_sql <<EOF
-- 生成用户行为数据
INSERT INTO user_behaviors (user_id, item_id, action, value, context, session_id, device_type, created_at)
SELECT 
    (SELECT id FROM users ORDER BY random() LIMIT 1) AS user_id,
    (SELECT id FROM items ORDER BY random() LIMIT 1) AS item_id,
    (ARRAY['view', 'click', 'like', 'buy', 'rating', 'share'])[floor(random() * 6 + 1)] AS action,
    CASE 
        WHEN random() < 0.2 THEN (random() * 5)::numeric(3,2)
        ELSE NULL 
    END AS value,
    jsonb_build_object(
        'page', (ARRAY['home', 'search', 'detail', 'recommend'])[floor(random() * 4 + 1)],
        'position', (random() * 20)::int
    ) AS context,
    'session_' || (random() * 10000)::int AS session_id,
    (ARRAY['mobile', 'desktop', 'tablet'])[floor(random() * 3 + 1)] AS device_type,
    NOW() - (random() * INTERVAL '90 days') AS created_at
FROM generate_series(1, ${NUM_BEHAVIORS}) AS i;
EOF
    
    log_success "Behaviors seeded"
}

# =============================================================================
# 更新统计数据
# =============================================================================
update_statistics() {
    log_info "Updating item statistics..."
    
    exec_sql <<EOF
-- 更新物品统计
INSERT INTO item_stats (item_id, view_count, click_count, like_count, share_count, avg_rating, rating_count, popularity_score)
SELECT 
    i.id,
    COALESCE(SUM(CASE WHEN b.action = 'view' THEN 1 ELSE 0 END), 0) AS view_count,
    COALESCE(SUM(CASE WHEN b.action = 'click' THEN 1 ELSE 0 END), 0) AS click_count,
    COALESCE(SUM(CASE WHEN b.action = 'like' THEN 1 ELSE 0 END), 0) AS like_count,
    COALESCE(SUM(CASE WHEN b.action = 'share' THEN 1 ELSE 0 END), 0) AS share_count,
    COALESCE(AVG(b.value) FILTER (WHERE b.action = 'rating'), 0) AS avg_rating,
    COALESCE(COUNT(*) FILTER (WHERE b.action = 'rating'), 0) AS rating_count,
    (COALESCE(SUM(CASE WHEN b.action = 'view' THEN 1 ELSE 0 END), 0) * 1.0 +
     COALESCE(SUM(CASE WHEN b.action = 'click' THEN 1 ELSE 0 END), 0) * 2.0 +
     COALESCE(SUM(CASE WHEN b.action = 'like' THEN 1 ELSE 0 END), 0) * 3.0 +
     COALESCE(SUM(CASE WHEN b.action = 'buy' THEN 1 ELSE 0 END), 0) * 5.0) AS popularity_score
FROM items i
LEFT JOIN user_behaviors b ON i.id = b.item_id
GROUP BY i.id
ON CONFLICT (item_id) DO UPDATE SET
    view_count = EXCLUDED.view_count,
    click_count = EXCLUDED.click_count,
    like_count = EXCLUDED.like_count,
    share_count = EXCLUDED.share_count,
    avg_rating = EXCLUDED.avg_rating,
    rating_count = EXCLUDED.rating_count,
    popularity_score = EXCLUDED.popularity_score,
    updated_at = NOW();
EOF
    
    log_success "Statistics updated"
}

# =============================================================================
# 更新用户画像
# =============================================================================
update_user_profiles() {
    log_info "Updating user profiles..."
    
    exec_sql <<EOF
-- 更新用户画像
INSERT INTO user_profiles (user_id, total_actions, total_views, total_clicks, preferred_types, preferred_categories, last_active, is_cold_start, activity_score)
SELECT 
    u.id AS user_id,
    COALESCE(COUNT(b.id), 0) AS total_actions,
    COALESCE(COUNT(b.id) FILTER (WHERE b.action = 'view'), 0) AS total_views,
    COALESCE(COUNT(b.id) FILTER (WHERE b.action = 'click'), 0) AS total_clicks,
    COALESCE(
        jsonb_object_agg(i.type, type_count) FILTER (WHERE i.type IS NOT NULL),
        '{}'::jsonb
    ) AS preferred_types,
    COALESCE(
        jsonb_object_agg(i.category, cat_count) FILTER (WHERE i.category IS NOT NULL),
        '{}'::jsonb
    ) AS preferred_categories,
    MAX(b.created_at) AS last_active,
    CASE WHEN COUNT(b.id) < 10 THEN TRUE ELSE FALSE END AS is_cold_start,
    LEAST(COUNT(b.id)::numeric / 100, 10) AS activity_score
FROM users u
LEFT JOIN user_behaviors b ON u.id = b.user_id
LEFT JOIN items i ON b.item_id = i.id
LEFT JOIN LATERAL (
    SELECT i2.type, COUNT(*)::numeric / NULLIF(COUNT(*) OVER(), 0) AS type_count
    FROM user_behaviors b2
    JOIN items i2 ON b2.item_id = i2.id
    WHERE b2.user_id = u.id
    GROUP BY i2.type
    LIMIT 5
) type_stats ON TRUE
LEFT JOIN LATERAL (
    SELECT i3.category, COUNT(*)::numeric / NULLIF(COUNT(*) OVER(), 0) AS cat_count
    FROM user_behaviors b3
    JOIN items i3 ON b3.item_id = i3.id
    WHERE b3.user_id = u.id
    GROUP BY i3.category
    LIMIT 5
) cat_stats ON TRUE
GROUP BY u.id
ON CONFLICT (user_id) DO UPDATE SET
    total_actions = EXCLUDED.total_actions,
    total_views = EXCLUDED.total_views,
    total_clicks = EXCLUDED.total_clicks,
    last_active = EXCLUDED.last_active,
    is_cold_start = EXCLUDED.is_cold_start,
    activity_score = EXCLUDED.activity_score,
    updated_at = NOW();
EOF
    
    log_success "User profiles updated"
}

# =============================================================================
# 刷新物化视图
# =============================================================================
refresh_materialized_views() {
    log_info "Refreshing materialized views..."
    
    exec_sql <<EOF
-- 刷新物化视图（如果存在）
DO \$\$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE matviewname = 'mv_user_daily_stats') THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_daily_stats;
    END IF;
    
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE matviewname = 'mv_item_daily_stats') THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_item_daily_stats;
    END IF;
    
    IF EXISTS (SELECT 1 FROM pg_matviews WHERE matviewname = 'mv_trending_items') THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_trending_items;
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not refresh some materialized views: %', SQLERRM;
END;
\$\$;
EOF
    
    log_success "Materialized views refreshed"
}

# =============================================================================
# 验证数据
# =============================================================================
verify_data() {
    log_info "Verifying seeded data..."
    
    exec_sql -t <<EOF
SELECT 
    'Users' AS table_name, COUNT(*) AS count FROM users
UNION ALL
SELECT 
    'Items', COUNT(*) FROM items
UNION ALL
SELECT 
    'Item Stats', COUNT(*) FROM item_stats
UNION ALL
SELECT 
    'User Behaviors', COUNT(*) FROM user_behaviors
UNION ALL
SELECT 
    'User Profiles', COUNT(*) FROM user_profiles;
EOF
    
    log_success "Data verification completed"
}

# =============================================================================
# 清理现有数据
# =============================================================================
clean_data() {
    log_info "Cleaning existing seed data..."
    
    exec_sql <<EOF
-- 清理顺序需要考虑外键依赖
TRUNCATE user_behaviors CASCADE;
TRUNCATE user_sessions CASCADE;
TRUNCATE recommendation_results CASCADE;
TRUNCATE recommendation_requests CASCADE;
TRUNCATE user_profiles CASCADE;
TRUNCATE item_stats CASCADE;
TRUNCATE items CASCADE;
TRUNCATE users CASCADE;
EOF
    
    log_success "Existing data cleaned"
}

# =============================================================================
# 使用说明
# =============================================================================
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Seed database with test data

Options:
  --users NUM      Number of users to create (default: ${NUM_USERS})
  --items NUM      Number of items to create (default: ${NUM_ITEMS})
  --behaviors NUM  Number of behaviors to create (default: ${NUM_BEHAVIORS})
  --clean          Clean existing data before seeding
  --verify-only    Only verify existing data
  -h, --help       Show this help message

Examples:
  # Seed with default amounts
  $0

  # Seed with custom amounts
  $0 --users 5000 --items 10000 --behaviors 100000

  # Clean and re-seed
  $0 --clean
EOF
    exit 0
}

# =============================================================================
# 参数解析
# =============================================================================
CLEAN_DATA=false
VERIFY_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --users)
            NUM_USERS="$2"
            shift 2
            ;;
        --items)
            NUM_ITEMS="$2"
            shift 2
            ;;
        --behaviors)
            NUM_BEHAVIORS="$2"
            shift 2
            ;;
        --clean)
            CLEAN_DATA=true
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=true
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
    log_info "=========================================="
    log_info "Starting Database Seed"
    log_info "=========================================="
    log_info "Users: ${NUM_USERS}, Items: ${NUM_ITEMS}, Behaviors: ${NUM_BEHAVIORS}"
    
    if [[ "$VERIFY_ONLY" == "true" ]]; then
        verify_data
        exit 0
    fi
    
    if [[ "$CLEAN_DATA" == "true" ]]; then
        clean_data
    fi
    
    seed_users
    seed_items
    seed_behaviors
    update_statistics
    update_user_profiles
    refresh_materialized_views
    verify_data
    
    log_success "=========================================="
    log_success "Database Seed Completed"
    log_success "=========================================="
}

main

