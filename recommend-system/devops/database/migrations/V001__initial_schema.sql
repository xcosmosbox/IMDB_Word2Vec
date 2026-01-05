-- =============================================================================
-- V001: 初始数据库 Schema
-- 创建时间: 2024-01-01
-- 描述: 创建基础表结构，包括用户表、物品表、物品统计表
-- 项目: 生成式推荐系统
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 启用扩展
-- -----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";      -- UUID 生成
CREATE EXTENSION IF NOT EXISTS "pg_trgm";        -- 模糊搜索（三元组索引）
CREATE EXTENSION IF NOT EXISTS "btree_gin";      -- GIN 索引增强

-- =============================================================================
-- 更新时间触发器函数（全局）
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 用户表 (users)
-- =============================================================================
CREATE TABLE users (
    -- 主键
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- 基本信息
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    
    -- 人口统计学特征
    age SMALLINT CHECK (age >= 0 AND age <= 150),
    gender VARCHAR(20) CHECK (gender IN ('male', 'female', 'other', 'unknown')),
    country VARCHAR(100),
    city VARCHAR(100),
    
    -- 扩展元数据 (JSONB 格式)
    -- 示例: {"interests": ["movie", "music"], "preferences": {"language": "zh"}}
    metadata JSONB DEFAULT '{}',
    
    -- 状态
    status VARCHAR(20) DEFAULT 'active' 
        CHECK (status IN ('active', 'inactive', 'banned', 'deleted')),
    
    -- 审计字段
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE,
    
    -- 版本控制（乐观锁）
    version INTEGER DEFAULT 1
);

-- 用户表注释
COMMENT ON TABLE users IS '用户主表，存储用户基本信息和元数据';
COMMENT ON COLUMN users.id IS '用户唯一标识符 (UUID v4)';
COMMENT ON COLUMN users.metadata IS '用户扩展元数据，JSONB 格式';
COMMENT ON COLUMN users.status IS '用户状态: active/inactive/banned/deleted';

-- 用户表索引
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status) WHERE status != 'deleted';
CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_users_last_login ON users(last_login_at);
CREATE INDEX idx_users_name_trgm ON users USING gin(name gin_trgm_ops);
CREATE INDEX idx_users_metadata ON users USING gin(metadata);

-- 用户表更新时间触发器
CREATE TRIGGER users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 物品表 (items)
-- =============================================================================
CREATE TABLE items (
    -- 主键
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- 外部 ID（来自源系统）
    external_id VARCHAR(255),
    
    -- 物品类型
    type VARCHAR(50) NOT NULL 
        CHECK (type IN ('movie', 'product', 'article', 'video', 'music', 'podcast')),
    
    -- 基本信息
    title VARCHAR(500) NOT NULL,
    description TEXT,
    
    -- 分类信息
    category VARCHAR(100),
    subcategory VARCHAR(100),
    
    -- 标签 (数组格式)
    tags TEXT[] DEFAULT '{}',
    
    -- 语义 ID (来自 RQ-VAE 编码器)
    -- 格式: [level1_code, level2_code, level3_code]
    semantic_id INTEGER[] DEFAULT '{}',
    
    -- 扩展元数据
    -- 示例: {"duration": 120, "rating": 8.5, "director": "Frank Darabont"}
    metadata JSONB DEFAULT '{}',
    
    -- 状态
    status VARCHAR(20) DEFAULT 'active' 
        CHECK (status IN ('active', 'inactive', 'deleted', 'pending')),
    
    -- 发布时间
    published_at TIMESTAMP WITH TIME ZONE,
    
    -- 审计字段
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 版本控制
    version INTEGER DEFAULT 1
);

-- 物品表注释
COMMENT ON TABLE items IS '物品主表，存储所有可推荐的物品信息';
COMMENT ON COLUMN items.type IS '物品类型: movie/product/article/video/music/podcast';
COMMENT ON COLUMN items.semantic_id IS '语义 ID 数组，由 RQ-VAE 生成的层次化编码';
COMMENT ON COLUMN items.tags IS '物品标签数组';

-- 物品表索引
CREATE INDEX idx_items_type ON items(type);
CREATE INDEX idx_items_category ON items(category);
CREATE INDEX idx_items_subcategory ON items(subcategory);
CREATE INDEX idx_items_status ON items(status) WHERE status != 'deleted';
CREATE INDEX idx_items_tags ON items USING gin(tags);
CREATE INDEX idx_items_semantic_id ON items USING gin(semantic_id);
CREATE INDEX idx_items_title_trgm ON items USING gin(title gin_trgm_ops);
CREATE INDEX idx_items_created_at ON items(created_at);
CREATE INDEX idx_items_published_at ON items(published_at);
CREATE INDEX idx_items_metadata ON items USING gin(metadata);
CREATE INDEX idx_items_external_id ON items(external_id) WHERE external_id IS NOT NULL;
CREATE UNIQUE INDEX idx_items_type_external_id ON items(type, external_id) 
    WHERE external_id IS NOT NULL;

-- 物品表更新时间触发器
CREATE TRIGGER items_updated_at
    BEFORE UPDATE ON items
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 物品统计表 (item_stats)
-- =============================================================================
CREATE TABLE item_stats (
    -- 主键（与物品表关联）
    item_id UUID PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    
    -- 计数统计
    view_count BIGINT DEFAULT 0 CHECK (view_count >= 0),
    click_count BIGINT DEFAULT 0 CHECK (click_count >= 0),
    like_count BIGINT DEFAULT 0 CHECK (like_count >= 0),
    dislike_count BIGINT DEFAULT 0 CHECK (dislike_count >= 0),
    share_count BIGINT DEFAULT 0 CHECK (share_count >= 0),
    save_count BIGINT DEFAULT 0 CHECK (save_count >= 0),
    purchase_count BIGINT DEFAULT 0 CHECK (purchase_count >= 0),
    
    -- 评分统计
    avg_rating DECIMAL(3, 2) DEFAULT 0 CHECK (avg_rating >= 0 AND avg_rating <= 5),
    rating_count BIGINT DEFAULT 0 CHECK (rating_count >= 0),
    
    -- 互动时长统计（秒）
    total_duration BIGINT DEFAULT 0,
    avg_duration DECIMAL(10, 2) DEFAULT 0,
    
    -- 转化率指标
    ctr DECIMAL(5, 4) DEFAULT 0,  -- 点击率
    cvr DECIMAL(5, 4) DEFAULT 0,  -- 转化率
    
    -- 热度分数（可由定时任务更新）
    popularity_score DECIMAL(10, 4) DEFAULT 0,
    
    -- 时间窗口统计
    last_7_days_views BIGINT DEFAULT 0,
    last_30_days_views BIGINT DEFAULT 0,
    
    -- 更新时间
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 物品统计表注释
COMMENT ON TABLE item_stats IS '物品统计表，存储各类聚合指标';
COMMENT ON COLUMN item_stats.popularity_score IS '热度分数，由定时任务计算更新';
COMMENT ON COLUMN item_stats.ctr IS '点击率 = click_count / view_count';

-- 物品统计表索引
CREATE INDEX idx_item_stats_popularity ON item_stats(popularity_score DESC);
CREATE INDEX idx_item_stats_avg_rating ON item_stats(avg_rating DESC);
CREATE INDEX idx_item_stats_view_count ON item_stats(view_count DESC);

-- 物品统计表更新时间触发器
CREATE TRIGGER item_stats_updated_at
    BEFORE UPDATE ON item_stats
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 物品统计自动创建函数
-- 当新物品插入时，自动创建对应的统计记录
-- =============================================================================
CREATE OR REPLACE FUNCTION auto_create_item_stats()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO item_stats (item_id) VALUES (NEW.id) ON CONFLICT DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER items_auto_create_stats
    AFTER INSERT ON items
    FOR EACH ROW
    EXECUTE FUNCTION auto_create_item_stats();

-- =============================================================================
-- 统计更新函数（CTR/CVR）
-- =============================================================================
CREATE OR REPLACE FUNCTION update_item_ctr_cvr()
RETURNS TRIGGER AS $$
BEGIN
    -- 更新 CTR
    IF NEW.view_count > 0 THEN
        NEW.ctr = NEW.click_count::DECIMAL / NEW.view_count;
    END IF;
    
    -- 更新 CVR
    IF NEW.click_count > 0 THEN
        NEW.cvr = NEW.purchase_count::DECIMAL / NEW.click_count;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER item_stats_update_ctr_cvr
    BEFORE UPDATE ON item_stats
    FOR EACH ROW
    WHEN (
        OLD.view_count IS DISTINCT FROM NEW.view_count OR
        OLD.click_count IS DISTINCT FROM NEW.click_count OR
        OLD.purchase_count IS DISTINCT FROM NEW.purchase_count
    )
    EXECUTE FUNCTION update_item_ctr_cvr();

