-- =============================================================================
-- 数据库基准测试 SQL 脚本
--
-- 用于测量推荐系统数据库的性能指标。
-- 支持 PostgreSQL 数据库。
--
-- 使用方法:
--   psql -U postgres -d recommend_db -f database-benchmark.sql
--
-- 或者使用 pgbench:
--   pgbench -U postgres -d recommend_db -f database-benchmark.sql -c 10 -j 4 -T 60
-- =============================================================================

-- 设置输出格式
\timing on
\pset pager off

-- =============================================================================
-- 1. 环境准备
-- =============================================================================

-- 创建基准测试结果表 (如果不存在)
CREATE TABLE IF NOT EXISTS benchmark_results (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(255) NOT NULL,
    test_type VARCHAR(50) NOT NULL,
    execution_time_ms NUMERIC(10, 4),
    rows_affected INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 清理之前的测试结果
TRUNCATE TABLE benchmark_results;

-- 记录开始时间
INSERT INTO benchmark_results (test_name, test_type, execution_time_ms)
VALUES ('Benchmark Start', 'info', 0);

-- =============================================================================
-- 2. 表结构基准测试
-- =============================================================================

-- 2.1 用户表查询性能
\echo '=== User Table Benchmarks ==='

-- 2.1.1 主键查询 (应该非常快, < 1ms)
\echo 'Test: User lookup by ID'
EXPLAIN ANALYZE
SELECT * FROM users WHERE id = 'user_12345';

-- 2.1.2 批量用户查询
\echo 'Test: Batch user lookup (100 users)'
EXPLAIN ANALYZE
SELECT * FROM users 
WHERE id IN (
    SELECT 'user_' || generate_series(1, 100)::text
);

-- 2.1.3 用户特征向量查询
\echo 'Test: User embedding lookup'
EXPLAIN ANALYZE
SELECT id, embedding 
FROM user_embeddings 
WHERE user_id = 'user_12345';

-- =============================================================================
-- 3. 物品表查询性能
-- =============================================================================

\echo '=== Item Table Benchmarks ==='

-- 3.1 主键查询
\echo 'Test: Item lookup by ID'
EXPLAIN ANALYZE
SELECT * FROM items WHERE id = 'item_12345';

-- 3.2 分类查询
\echo 'Test: Items by category'
EXPLAIN ANALYZE
SELECT * FROM items 
WHERE category = 'action' 
ORDER BY popularity_score DESC
LIMIT 100;

-- 3.3 全文搜索
\echo 'Test: Full-text search'
EXPLAIN ANALYZE
SELECT * FROM items 
WHERE to_tsvector('english', title || ' ' || description) @@ plainto_tsquery('action movie')
LIMIT 20;

-- 3.4 标签查询 (JSON 数组)
\echo 'Test: Items by tag'
EXPLAIN ANALYZE
SELECT * FROM items 
WHERE tags @> '["action"]'::jsonb
LIMIT 100;

-- =============================================================================
-- 4. 用户行为表查询性能
-- =============================================================================

\echo '=== User Behavior Table Benchmarks ==='

-- 4.1 用户历史记录查询
\echo 'Test: User history lookup'
EXPLAIN ANALYZE
SELECT * FROM user_behaviors 
WHERE user_id = 'user_12345'
ORDER BY created_at DESC
LIMIT 100;

-- 4.2 按时间范围查询
\echo 'Test: User behaviors by time range'
EXPLAIN ANALYZE
SELECT * FROM user_behaviors 
WHERE user_id = 'user_12345'
  AND created_at >= NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;

-- 4.3 行为统计聚合
\echo 'Test: User behavior aggregation'
EXPLAIN ANALYZE
SELECT 
    user_id,
    action_type,
    COUNT(*) as count,
    AVG(EXTRACT(EPOCH FROM duration)) as avg_duration
FROM user_behaviors
WHERE user_id = 'user_12345'
GROUP BY user_id, action_type;

-- 4.4 热门物品聚合
\echo 'Test: Popular items aggregation (last 24h)'
EXPLAIN ANALYZE
SELECT 
    item_id,
    COUNT(*) as interaction_count,
    COUNT(DISTINCT user_id) as unique_users
FROM user_behaviors
WHERE created_at >= NOW() - INTERVAL '24 hours'
  AND action_type IN ('click', 'view', 'like')
GROUP BY item_id
ORDER BY interaction_count DESC
LIMIT 100;

-- =============================================================================
-- 5. 推荐缓存表查询性能
-- =============================================================================

\echo '=== Recommendation Cache Benchmarks ==='

-- 5.1 缓存命中查询
\echo 'Test: Cache hit lookup'
EXPLAIN ANALYZE
SELECT * FROM recommendation_cache
WHERE user_id = 'user_12345'
  AND scene = 'home'
  AND expires_at > NOW()
LIMIT 1;

-- 5.2 批量缓存失效
\echo 'Test: Batch cache invalidation'
EXPLAIN ANALYZE
DELETE FROM recommendation_cache
WHERE expires_at < NOW()
LIMIT 1000;

-- =============================================================================
-- 6. 向量检索性能 (使用 pgvector 扩展)
-- =============================================================================

\echo '=== Vector Search Benchmarks ==='

-- 注意: 需要安装 pgvector 扩展
-- CREATE EXTENSION IF NOT EXISTS vector;

-- 6.1 相似向量检索 (KNN)
\echo 'Test: KNN vector search'
EXPLAIN ANALYZE
SELECT id, 1 - (embedding <=> '[0.1, 0.2, ...]'::vector) as similarity
FROM item_embeddings
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 20;

-- 6.2 向量范围查询
\echo 'Test: Vector range query'
EXPLAIN ANALYZE
SELECT id
FROM item_embeddings
WHERE embedding <=> '[0.1, 0.2, ...]'::vector < 0.5
LIMIT 100;

-- =============================================================================
-- 7. 写入性能测试
-- =============================================================================

\echo '=== Write Performance Benchmarks ==='

-- 7.1 单条插入
\echo 'Test: Single row insert'
EXPLAIN ANALYZE
INSERT INTO user_behaviors (user_id, item_id, action_type, created_at)
VALUES ('benchmark_user', 'item_1', 'click', NOW());

-- 7.2 批量插入
\echo 'Test: Batch insert (100 rows)'
EXPLAIN ANALYZE
INSERT INTO user_behaviors (user_id, item_id, action_type, created_at)
SELECT 
    'benchmark_user',
    'item_' || generate_series,
    'view',
    NOW() - (generate_series || ' minutes')::interval
FROM generate_series(1, 100);

-- 7.3 更新性能
\echo 'Test: Update user profile'
EXPLAIN ANALYZE
UPDATE users
SET last_active_at = NOW(),
    behavior_count = behavior_count + 1
WHERE id = 'user_12345';

-- 7.4 批量更新
\echo 'Test: Batch update item scores'
EXPLAIN ANALYZE
UPDATE items
SET popularity_score = popularity_score + 1
WHERE category = 'action'
LIMIT 100;

-- =============================================================================
-- 8. 并发性能测试 (使用 pgbench 时有效)
-- =============================================================================

-- 为 pgbench 准备的自定义事务
-- 文件: benchmark_transaction.sql

/*
\set user_id random(1, 100000)
\set item_id random(1, 1000000)

BEGIN;

-- 读取用户
SELECT * FROM users WHERE id = 'user_' || :user_id;

-- 读取推荐
SELECT * FROM recommendation_cache 
WHERE user_id = 'user_' || :user_id 
  AND scene = 'home';

-- 记录行为
INSERT INTO user_behaviors (user_id, item_id, action_type, created_at)
VALUES ('user_' || :user_id, 'item_' || :item_id, 'click', NOW());

COMMIT;
*/

-- =============================================================================
-- 9. 索引效果分析
-- =============================================================================

\echo '=== Index Analysis ==='

-- 9.1 表索引使用统计
SELECT 
    schemaname,
    relname as table_name,
    indexrelname as index_name,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;

-- 9.2 表访问统计
SELECT 
    relname as table_name,
    seq_scan as sequential_scans,
    seq_tup_read as seq_tuples_read,
    idx_scan as index_scans,
    idx_tup_fetch as index_tuples_fetched,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY seq_scan + idx_scan DESC;

-- 9.3 慢查询分析 (需要开启 pg_stat_statements)
/*
SELECT 
    substring(query, 1, 100) as short_query,
    calls,
    round(total_time::numeric, 2) as total_time_ms,
    round(mean_time::numeric, 2) as mean_time_ms,
    round(min_time::numeric, 2) as min_time_ms,
    round(max_time::numeric, 2) as max_time_ms
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 20;
*/

-- =============================================================================
-- 10. 数据库配置建议
-- =============================================================================

\echo '=== Database Configuration Recommendations ==='

-- 10.1 当前配置
SELECT name, setting, unit, short_desc
FROM pg_settings
WHERE name IN (
    'shared_buffers',
    'effective_cache_size',
    'work_mem',
    'maintenance_work_mem',
    'max_connections',
    'random_page_cost',
    'effective_io_concurrency',
    'max_parallel_workers_per_gather'
)
ORDER BY name;

-- 10.2 推荐索引
\echo 'Recommended indexes for performance:'
\echo '  - users(id) - Primary key'
\echo '  - items(id) - Primary key'
\echo '  - items(category, popularity_score DESC)'
\echo '  - user_behaviors(user_id, created_at DESC)'
\echo '  - user_behaviors(item_id, created_at)'
\echo '  - recommendation_cache(user_id, scene, expires_at)'
\echo '  - item_embeddings USING ivfflat(embedding vector_cosine_ops)'

-- =============================================================================
-- 11. 清理测试数据
-- =============================================================================

\echo '=== Cleanup ==='

DELETE FROM user_behaviors WHERE user_id = 'benchmark_user';

-- 记录结束时间
INSERT INTO benchmark_results (test_name, test_type, execution_time_ms)
VALUES ('Benchmark End', 'info', 0);

-- 显示测试结果汇总
\echo '=== Benchmark Summary ==='
SELECT * FROM benchmark_results ORDER BY created_at;

\echo 'Database benchmark completed!'

