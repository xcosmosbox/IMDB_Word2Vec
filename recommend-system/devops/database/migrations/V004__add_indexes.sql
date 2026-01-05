-- =============================================================================
-- V004: 性能优化索引
-- 创建时间: 2024-01-04
-- 描述: 添加额外的性能优化索引、物化视图和查询优化
-- 项目: 生成式推荐系统
-- =============================================================================

-- =============================================================================
-- 复合索引优化
-- =============================================================================

-- 用户行为复合索引（优化用户历史查询）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_behaviors_user_item_action 
    ON user_behaviors(user_id, item_id, action);

-- 用户行为时间范围查询优化
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_behaviors_user_time_range 
    ON user_behaviors(user_id, created_at DESC, action);

-- 物品热度排序优化
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_item_stats_composite 
    ON item_stats(popularity_score DESC, avg_rating DESC, view_count DESC);

-- 推荐请求按场景和时间查询优化
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rec_requests_scene_time 
    ON recommendation_requests(scene, created_at DESC);

-- 实验相关查询优化
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rec_requests_exp_time 
    ON recommendation_requests(experiment_id, experiment_group, created_at DESC)
    WHERE experiment_id IS NOT NULL;

-- =============================================================================
-- 部分索引 (Partial Indexes)
-- =============================================================================

-- 活跃用户索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_active 
    ON users(last_login_at DESC)
    WHERE status = 'active' AND last_login_at > NOW() - INTERVAL '30 days';

-- 活跃物品索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_items_active_published 
    ON items(published_at DESC)
    WHERE status = 'active' AND published_at IS NOT NULL;

-- 高评分物品索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_item_stats_high_rating 
    ON item_stats(avg_rating DESC, rating_count DESC)
    WHERE avg_rating >= 4.0 AND rating_count >= 10;

-- 热门物品索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_item_stats_popular 
    ON item_stats(last_7_days_views DESC)
    WHERE last_7_days_views > 0;

-- 冷启动用户索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_profiles_cold_users 
    ON user_profiles(user_id)
    WHERE is_cold_start = TRUE;

-- 运行中实验索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_experiments_running 
    ON experiments(start_time, end_time)
    WHERE status = 'running';

-- =============================================================================
-- BRIN 索引 (适用于大表时间序列查询)
-- =============================================================================

-- 用户行为表 BRIN 索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_behaviors_created_brin 
    ON user_behaviors USING BRIN(created_at)
    WITH (pages_per_range = 128);

-- 推荐请求表 BRIN 索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rec_requests_created_brin 
    ON recommendation_requests USING BRIN(created_at)
    WITH (pages_per_range = 128);

-- 推荐结果表 BRIN 索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rec_results_created_brin 
    ON recommendation_results USING BRIN(created_at)
    WITH (pages_per_range = 128);

-- =============================================================================
-- 物化视图 - 用户行为统计（日聚合）
-- =============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_user_daily_stats AS
SELECT 
    user_id,
    DATE(created_at) AS stat_date,
    COUNT(*) AS total_actions,
    COUNT(DISTINCT item_id) AS unique_items,
    COUNT(*) FILTER (WHERE action = 'view') AS views,
    COUNT(*) FILTER (WHERE action = 'click') AS clicks,
    COUNT(*) FILTER (WHERE action = 'like') AS likes,
    COUNT(*) FILTER (WHERE action = 'buy') AS purchases,
    COUNT(*) FILTER (WHERE action = 'rating') AS ratings,
    AVG(value) FILTER (WHERE action = 'rating') AS avg_rating
FROM user_behaviors
WHERE created_at >= NOW() - INTERVAL '90 days'
GROUP BY user_id, DATE(created_at);

-- 物化视图索引
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_user_daily_unique 
    ON mv_user_daily_stats(user_id, stat_date);
CREATE INDEX IF NOT EXISTS idx_mv_user_daily_date 
    ON mv_user_daily_stats(stat_date DESC);

-- =============================================================================
-- 物化视图 - 物品行为统计（日聚合）
-- =============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_item_daily_stats AS
SELECT 
    item_id,
    DATE(created_at) AS stat_date,
    COUNT(*) AS total_actions,
    COUNT(DISTINCT user_id) AS unique_users,
    COUNT(*) FILTER (WHERE action = 'view') AS views,
    COUNT(*) FILTER (WHERE action = 'click') AS clicks,
    COUNT(*) FILTER (WHERE action = 'like') AS likes,
    COUNT(*) FILTER (WHERE action = 'buy') AS purchases,
    CASE 
        WHEN COUNT(*) FILTER (WHERE action = 'view') > 0 
        THEN COUNT(*) FILTER (WHERE action = 'click')::DECIMAL / COUNT(*) FILTER (WHERE action = 'view')
        ELSE 0 
    END AS daily_ctr
FROM user_behaviors
WHERE created_at >= NOW() - INTERVAL '90 days'
GROUP BY item_id, DATE(created_at);

-- 物化视图索引
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_item_daily_unique 
    ON mv_item_daily_stats(item_id, stat_date);
CREATE INDEX IF NOT EXISTS idx_mv_item_daily_date 
    ON mv_item_daily_stats(stat_date DESC);
CREATE INDEX IF NOT EXISTS idx_mv_item_daily_views 
    ON mv_item_daily_stats(views DESC) WHERE stat_date = CURRENT_DATE;

-- =============================================================================
-- 物化视图 - 实时热门物品 (近 24 小时)
-- =============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_trending_items AS
SELECT 
    b.item_id,
    i.title,
    i.type,
    i.category,
    COUNT(*) AS action_count,
    COUNT(DISTINCT b.user_id) AS user_count,
    COUNT(*) FILTER (WHERE b.action = 'click') AS clicks,
    COUNT(*) FILTER (WHERE b.action = 'like') AS likes,
    -- 热度分数计算
    (COUNT(*) * 1.0 + 
     COUNT(*) FILTER (WHERE b.action = 'click') * 2.0 +
     COUNT(*) FILTER (WHERE b.action = 'like') * 3.0 +
     COUNT(*) FILTER (WHERE b.action = 'buy') * 5.0) AS trending_score
FROM user_behaviors b
JOIN items i ON b.item_id = i.id
WHERE b.created_at >= NOW() - INTERVAL '24 hours'
    AND i.status = 'active'
GROUP BY b.item_id, i.title, i.type, i.category
HAVING COUNT(*) >= 10
ORDER BY trending_score DESC
LIMIT 1000;

-- 热门物品索引
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_trending_item_id 
    ON mv_trending_items(item_id);
CREATE INDEX IF NOT EXISTS idx_mv_trending_score 
    ON mv_trending_items(trending_score DESC);
CREATE INDEX IF NOT EXISTS idx_mv_trending_type 
    ON mv_trending_items(type, trending_score DESC);

-- =============================================================================
-- 物化视图刷新函数
-- =============================================================================
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_daily_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_item_daily_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_trending_items;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 查询优化函数 - 获取用户推荐候选
-- =============================================================================
CREATE OR REPLACE FUNCTION get_recommendation_candidates(
    p_user_id UUID,
    p_item_type VARCHAR(50) DEFAULT NULL,
    p_category VARCHAR(100) DEFAULT NULL,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    item_id UUID,
    title VARCHAR(500),
    type VARCHAR(50),
    category VARCHAR(100),
    popularity_score DECIMAL,
    avg_rating DECIMAL,
    semantic_id INTEGER[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        i.id AS item_id,
        i.title,
        i.type,
        i.category,
        COALESCE(s.popularity_score, 0) AS popularity_score,
        COALESCE(s.avg_rating, 0) AS avg_rating,
        i.semantic_id
    FROM items i
    LEFT JOIN item_stats s ON i.id = s.item_id
    WHERE i.status = 'active'
        AND (p_item_type IS NULL OR i.type = p_item_type)
        AND (p_category IS NULL OR i.category = p_category)
        -- 排除用户已交互过的物品
        AND NOT EXISTS (
            SELECT 1 FROM user_behaviors ub 
            WHERE ub.user_id = p_user_id 
                AND ub.item_id = i.id 
                AND ub.action IN ('click', 'buy', 'like')
                AND ub.created_at > NOW() - INTERVAL '30 days'
        )
    ORDER BY s.popularity_score DESC NULLS LAST, s.avg_rating DESC NULLS LAST
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 查询优化函数 - 获取相似物品
-- =============================================================================
CREATE OR REPLACE FUNCTION get_similar_items_by_behavior(
    p_item_id UUID,
    p_limit INTEGER DEFAULT 20
)
RETURNS TABLE (
    similar_item_id UUID,
    title VARCHAR(500),
    similarity_score DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    WITH item_users AS (
        -- 获取与目标物品有交互的用户
        SELECT DISTINCT user_id
        FROM user_behaviors
        WHERE item_id = p_item_id
            AND action IN ('click', 'like', 'buy')
            AND created_at > NOW() - INTERVAL '90 days'
        LIMIT 1000
    ),
    candidate_items AS (
        -- 获取这些用户交互过的其他物品
        SELECT 
            b.item_id,
            COUNT(DISTINCT b.user_id) AS common_users,
            SUM(CASE 
                WHEN b.action = 'buy' THEN 5
                WHEN b.action = 'like' THEN 3
                WHEN b.action = 'click' THEN 1
                ELSE 0.5
            END) AS weighted_score
        FROM user_behaviors b
        WHERE b.user_id IN (SELECT user_id FROM item_users)
            AND b.item_id != p_item_id
            AND b.action IN ('click', 'like', 'buy')
            AND b.created_at > NOW() - INTERVAL '90 days'
        GROUP BY b.item_id
        HAVING COUNT(DISTINCT b.user_id) >= 3
    )
    SELECT 
        c.item_id AS similar_item_id,
        i.title,
        (c.weighted_score / (SELECT MAX(weighted_score) FROM candidate_items))::DECIMAL(10, 4) AS similarity_score
    FROM candidate_items c
    JOIN items i ON c.item_id = i.id
    WHERE i.status = 'active'
    ORDER BY c.weighted_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 统计信息更新
-- =============================================================================
ANALYZE users;
ANALYZE items;
ANALYZE item_stats;
ANALYZE user_profiles;

