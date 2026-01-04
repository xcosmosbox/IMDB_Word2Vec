-- 生成式推荐系统数据库初始化脚本
-- PostgreSQL + pgvector

-- 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- 用户表
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(64) PRIMARY KEY,
    username VARCHAR(128) NOT NULL UNIQUE,
    email VARCHAR(256) NOT NULL UNIQUE,
    phone VARCHAR(32),
    avatar_url TEXT,
    status INTEGER NOT NULL DEFAULT 1,  -- 1:活跃 2:未激活 3:封禁 4:删除
    preferences JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_created_at ON users(created_at);

-- ============================================
-- 用户行为表
-- ============================================
CREATE TABLE IF NOT EXISTS user_behaviors (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    item_id VARCHAR(64) NOT NULL,
    item_type VARCHAR(32) NOT NULL,
    action VARCHAR(32) NOT NULL,  -- view, click, like, dislike, favorite, share, comment, purchase, rate
    value FLOAT DEFAULT 0,
    context JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 用户行为查询索引
CREATE INDEX idx_user_behaviors_user_id ON user_behaviors(user_id);
CREATE INDEX idx_user_behaviors_timestamp ON user_behaviors(timestamp DESC);
CREATE INDEX idx_user_behaviors_user_time ON user_behaviors(user_id, timestamp DESC);
CREATE INDEX idx_user_behaviors_item_id ON user_behaviors(item_id);

-- 分区表 (按月分区，用于大规模数据)
-- CREATE TABLE user_behaviors_2024_01 PARTITION OF user_behaviors
--     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- ============================================
-- 用户画像表
-- ============================================
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id VARCHAR(64) PRIMARY KEY REFERENCES users(id),
    demographics JSONB DEFAULT '{}',
    interests JSONB DEFAULT '[]',
    behavior_stats JSONB DEFAULT '{}',
    content_preferences JSONB DEFAULT '{}',
    recent_items JSONB DEFAULT '[]',
    long_term_interests JSONB DEFAULT '[]',
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- 物品表
-- ============================================
CREATE TABLE IF NOT EXISTS items (
    id VARCHAR(64) PRIMARY KEY,
    type VARCHAR(32) NOT NULL,  -- movie, video, product, article, music, book
    title VARCHAR(512) NOT NULL,
    description TEXT,
    cover_url TEXT,
    category VARCHAR(64) NOT NULL,
    sub_category VARCHAR(64),
    tags JSONB DEFAULT '[]',
    attributes JSONB DEFAULT '{}',
    status INTEGER NOT NULL DEFAULT 0,  -- 0:草稿 1:已发布 2:已下线 3:已删除
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    published_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_items_type ON items(type);
CREATE INDEX idx_items_status ON items(status);
CREATE INDEX idx_items_category ON items(category);
CREATE INDEX idx_items_updated_at ON items(updated_at DESC);

-- 全文搜索索引
CREATE INDEX idx_items_title_gin ON items USING gin(to_tsvector('english', title));

-- ============================================
-- 物品统计表
-- ============================================
CREATE TABLE IF NOT EXISTS item_stats (
    item_id VARCHAR(64) PRIMARY KEY REFERENCES items(id),
    view_count BIGINT DEFAULT 0,
    click_count BIGINT DEFAULT 0,
    like_count BIGINT DEFAULT 0,
    share_count BIGINT DEFAULT 0,
    comment_count BIGINT DEFAULT 0,
    rating_sum FLOAT DEFAULT 0,
    rating_count BIGINT DEFAULT 0,
    avg_rating FLOAT DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- 物品嵌入表 (使用 pgvector)
-- ============================================
CREATE TABLE IF NOT EXISTS item_embeddings (
    item_id VARCHAR(64) PRIMARY KEY REFERENCES items(id),
    embedding vector(256),  -- 嵌入向量维度
    semantic_id JSONB DEFAULT '[]',  -- 语义 ID 序列
    semantic_l1 INTEGER,  -- 第一层语义 ID
    semantic_l2 INTEGER,  -- 第二层语义 ID
    semantic_l3 INTEGER,  -- 第三层语义 ID
    model_version VARCHAR(32),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 向量索引 (HNSW)
CREATE INDEX idx_item_embeddings_vector ON item_embeddings USING hnsw (embedding vector_ip_ops);

-- 语义 ID 索引
CREATE INDEX idx_item_embeddings_semantic_l1 ON item_embeddings(semantic_l1);
CREATE INDEX idx_item_embeddings_semantic_l2 ON item_embeddings(semantic_l2);

-- ============================================
-- 推荐日志表
-- ============================================
CREATE TABLE IF NOT EXISTS recommend_logs (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL UNIQUE,
    user_id VARCHAR(64) NOT NULL,
    item_ids JSONB NOT NULL,
    scores JSONB,
    sources JSONB,
    model_version VARCHAR(32),
    context TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_recommend_logs_user_id ON recommend_logs(user_id);
CREATE INDEX idx_recommend_logs_timestamp ON recommend_logs(timestamp DESC);
CREATE INDEX idx_recommend_logs_user_time ON recommend_logs(user_id, timestamp DESC);

-- ============================================
-- 实验配置表
-- ============================================
CREATE TABLE IF NOT EXISTS experiments (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    description TEXT,
    traffic_ratio FLOAT NOT NULL DEFAULT 0.0,  -- 流量比例 0.0-1.0
    config JSONB DEFAULT '{}',
    status INTEGER NOT NULL DEFAULT 0,  -- 0:草稿 1:运行中 2:暂停 3:结束
    start_at TIMESTAMP WITH TIME ZONE,
    end_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- 物化视图：热门物品
-- ============================================
CREATE MATERIALIZED VIEW IF NOT EXISTS hot_items AS
SELECT 
    i.id,
    i.type,
    i.title,
    i.category,
    COALESCE(s.view_count, 0) as view_count,
    COALESCE(s.click_count, 0) as click_count,
    COALESCE(s.avg_rating, 0) as avg_rating,
    (COALESCE(s.view_count, 0) * 0.3 + COALESCE(s.click_count, 0) * 0.5 + COALESCE(s.avg_rating, 0) * 20) as hot_score
FROM items i
LEFT JOIN item_stats s ON i.id = s.item_id
WHERE i.status = 1
ORDER BY hot_score DESC
LIMIT 1000;

CREATE UNIQUE INDEX idx_hot_items_id ON hot_items(id);
CREATE INDEX idx_hot_items_type ON hot_items(type);
CREATE INDEX idx_hot_items_score ON hot_items(hot_score DESC);

-- 刷新热门物品视图的函数
CREATE OR REPLACE FUNCTION refresh_hot_items()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY hot_items;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 辅助函数
-- ============================================

-- 更新 updated_at 触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 为需要的表添加触发器
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_items_updated_at
    BEFORE UPDATE ON items
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 初始数据
-- ============================================

-- 插入测试用户
INSERT INTO users (id, username, email, status) VALUES
    ('user_001', 'test_user_1', 'test1@example.com', 1),
    ('user_002', 'test_user_2', 'test2@example.com', 1)
ON CONFLICT (id) DO NOTHING;

-- 插入测试物品
INSERT INTO items (id, type, title, category, status) VALUES
    ('item_001', 'movie', 'The Shawshank Redemption', 'Drama', 1),
    ('item_002', 'movie', 'The Godfather', 'Crime', 1),
    ('item_003', 'movie', 'Inception', 'Sci-Fi', 1)
ON CONFLICT (id) DO NOTHING;

-- 完成
SELECT 'Database initialization completed!' as message;

