-- =============================================================================
-- V002: 用户行为表
-- 创建时间: 2024-01-02
-- 描述: 添加用户行为记录表（分区表）和用户画像表
-- 项目: 生成式推荐系统
-- =============================================================================

-- =============================================================================
-- 用户行为表 (user_behaviors) - 分区表
-- 按月分区，支持高效的时间范围查询
-- =============================================================================
CREATE TABLE user_behaviors (
    -- 主键（复合主键，包含分区键）
    id UUID DEFAULT uuid_generate_v4(),
    
    -- 关联外键
    user_id UUID NOT NULL,
    item_id UUID NOT NULL,
    
    -- 行为类型
    action VARCHAR(50) NOT NULL 
        CHECK (action IN ('view', 'click', 'like', 'dislike', 'buy', 'share', 
                          'rating', 'save', 'complete', 'skip', 'add_cart', 
                          'remove_cart', 'search', 'comment')),
    
    -- 行为数值（用于评分等）
    value DECIMAL(5, 2),
    
    -- 行为上下文
    context JSONB DEFAULT '{}',
    
    -- 会话信息
    session_id VARCHAR(100),
    
    -- 设备信息
    device_type VARCHAR(50) CHECK (device_type IN ('mobile', 'tablet', 'desktop', 'tv', 'other')),
    device_os VARCHAR(50),
    app_version VARCHAR(50),
    
    -- 位置信息
    country VARCHAR(100),
    city VARCHAR(100),
    
    -- 来源追踪
    source VARCHAR(100),  -- 来源页面/模块
    position INTEGER,      -- 展示位置
    
    -- 时间戳（分区键）
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 复合主键
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- 行为表注释
COMMENT ON TABLE user_behaviors IS '用户行为记录表（按月分区），存储用户与物品的所有交互';
COMMENT ON COLUMN user_behaviors.action IS '行为类型，支持多种交互行为';
COMMENT ON COLUMN user_behaviors.context IS '行为上下文，JSON 格式存储额外信息';
COMMENT ON COLUMN user_behaviors.position IS '物品在列表中的展示位置';

-- =============================================================================
-- 创建初始分区 (2024年)
-- =============================================================================
CREATE TABLE user_behaviors_2024_01 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE user_behaviors_2024_02 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

CREATE TABLE user_behaviors_2024_03 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');

CREATE TABLE user_behaviors_2024_04 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');

CREATE TABLE user_behaviors_2024_05 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');

CREATE TABLE user_behaviors_2024_06 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');

CREATE TABLE user_behaviors_2024_07 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');

CREATE TABLE user_behaviors_2024_08 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');

CREATE TABLE user_behaviors_2024_09 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');

CREATE TABLE user_behaviors_2024_10 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');

CREATE TABLE user_behaviors_2024_11 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');

CREATE TABLE user_behaviors_2024_12 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');

-- 2025年分区
CREATE TABLE user_behaviors_2025_01 PARTITION OF user_behaviors
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE user_behaviors_2025_02 PARTITION OF user_behaviors
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

CREATE TABLE user_behaviors_2025_03 PARTITION OF user_behaviors
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

CREATE TABLE user_behaviors_2025_04 PARTITION OF user_behaviors
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');

CREATE TABLE user_behaviors_2025_05 PARTITION OF user_behaviors
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');

CREATE TABLE user_behaviors_2025_06 PARTITION OF user_behaviors
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');

CREATE TABLE user_behaviors_2025_07 PARTITION OF user_behaviors
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');

CREATE TABLE user_behaviors_2025_08 PARTITION OF user_behaviors
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');

CREATE TABLE user_behaviors_2025_09 PARTITION OF user_behaviors
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

CREATE TABLE user_behaviors_2025_10 PARTITION OF user_behaviors
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');

CREATE TABLE user_behaviors_2025_11 PARTITION OF user_behaviors
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE user_behaviors_2025_12 PARTITION OF user_behaviors
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- 2026年分区
CREATE TABLE user_behaviors_2026_01 PARTITION OF user_behaviors
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

-- =============================================================================
-- 自动创建分区函数
-- =============================================================================
CREATE OR REPLACE FUNCTION create_behavior_partition_if_needed()
RETURNS TRIGGER AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    -- 计算目标分区日期
    partition_date := DATE_TRUNC('month', NEW.created_at);
    partition_name := 'user_behaviors_' || TO_CHAR(partition_date, 'YYYY_MM');
    start_date := partition_date;
    end_date := partition_date + INTERVAL '1 month';
    
    -- 检查分区是否存在
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = partition_name
        AND n.nspname = 'public'
    ) THEN
        -- 创建分区
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF user_behaviors 
             FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            start_date,
            end_date
        );
        
        -- 为新分区创建索引
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON %I (user_id)',
            'idx_' || partition_name || '_user_id',
            partition_name
        );
        
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON %I (item_id)',
            'idx_' || partition_name || '_item_id',
            partition_name
        );
        
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON %I (action)',
            'idx_' || partition_name || '_action',
            partition_name
        );
        
        RAISE NOTICE 'Created partition: %', partition_name;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 行为表索引（在父表上）
CREATE INDEX idx_behaviors_user_id ON user_behaviors(user_id);
CREATE INDEX idx_behaviors_item_id ON user_behaviors(item_id);
CREATE INDEX idx_behaviors_action ON user_behaviors(action);
CREATE INDEX idx_behaviors_session_id ON user_behaviors(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_behaviors_user_action_time ON user_behaviors(user_id, action, created_at DESC);
CREATE INDEX idx_behaviors_item_action_time ON user_behaviors(item_id, action, created_at DESC);
CREATE INDEX idx_behaviors_context ON user_behaviors USING gin(context);

-- =============================================================================
-- 用户画像表 (user_profiles)
-- =============================================================================
CREATE TABLE user_profiles (
    -- 主键（与用户表关联）
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    
    -- 行为统计
    total_actions BIGINT DEFAULT 0,
    total_views BIGINT DEFAULT 0,
    total_clicks BIGINT DEFAULT 0,
    total_purchases BIGINT DEFAULT 0,
    
    -- 偏好分析 (JSONB 格式)
    -- 格式: {"movie": 0.8, "video": 0.5, ...}
    preferred_types JSONB DEFAULT '{}',
    
    -- 类目偏好
    -- 格式: {"action": 0.7, "comedy": 0.3, ...}
    preferred_categories JSONB DEFAULT '{}',
    
    -- 活跃时段偏好
    -- 格式: {"0": 0.1, "1": 0.05, ..., "23": 0.2}
    active_hours JSONB DEFAULT '{}',
    
    -- 活跃星期偏好
    -- 格式: {"mon": 0.1, "tue": 0.15, ...}
    active_days JSONB DEFAULT '{}',
    
    -- 价格偏好（适用于电商场景）
    price_preference JSONB DEFAULT '{}',
    
    -- 用户向量（序列化的嵌入向量）
    profile_vector BYTEA,
    
    -- 用户标签
    user_tags TEXT[] DEFAULT '{}',
    
    -- 用户细分
    user_segment VARCHAR(50),
    
    -- 冷启动状态
    is_cold_start BOOLEAN DEFAULT TRUE,
    cold_start_score DECIMAL(3, 2) DEFAULT 1.0,
    
    -- 活跃度指标
    last_active TIMESTAMP WITH TIME ZONE,
    activity_score DECIMAL(5, 2) DEFAULT 0,
    
    -- 更新时间
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 用户画像表注释
COMMENT ON TABLE user_profiles IS '用户画像表，存储用户偏好和行为分析结果';
COMMENT ON COLUMN user_profiles.preferred_types IS '物品类型偏好，JSON 格式，值为偏好权重';
COMMENT ON COLUMN user_profiles.profile_vector IS '用户嵌入向量，二进制序列化存储';
COMMENT ON COLUMN user_profiles.is_cold_start IS '是否为冷启动用户（行为数据不足）';

-- 用户画像表索引
CREATE INDEX idx_user_profiles_segment ON user_profiles(user_segment);
CREATE INDEX idx_user_profiles_last_active ON user_profiles(last_active);
CREATE INDEX idx_user_profiles_activity ON user_profiles(activity_score DESC);
CREATE INDEX idx_user_profiles_cold_start ON user_profiles(is_cold_start) WHERE is_cold_start = TRUE;
CREATE INDEX idx_user_profiles_tags ON user_profiles USING gin(user_tags);

-- 用户画像表更新时间触发器
CREATE TRIGGER user_profiles_updated_at
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 用户画像自动创建函数
-- =============================================================================
CREATE OR REPLACE FUNCTION auto_create_user_profile()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO user_profiles (user_id) VALUES (NEW.id) ON CONFLICT DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_auto_create_profile
    AFTER INSERT ON users
    FOR EACH ROW
    EXECUTE FUNCTION auto_create_user_profile();

-- =============================================================================
-- 用户会话表 (user_sessions)
-- =============================================================================
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(100) NOT NULL UNIQUE,
    
    -- 会话信息
    device_type VARCHAR(50),
    device_os VARCHAR(50),
    app_version VARCHAR(50),
    ip_address INET,
    
    -- 会话统计
    page_views INTEGER DEFAULT 0,
    actions_count INTEGER DEFAULT 0,
    duration_seconds INTEGER DEFAULT 0,
    
    -- 时间
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 元数据
    metadata JSONB DEFAULT '{}'
);

-- 会话表索引
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_started_at ON user_sessions(started_at);
CREATE INDEX idx_user_sessions_active ON user_sessions(last_activity_at) 
    WHERE ended_at IS NULL;

