-- =============================================================================
-- V003: 推荐记录表
-- 创建时间: 2024-01-03
-- 描述: 添加推荐记录、推荐结果、AB 测试相关表
-- 项目: 生成式推荐系统
-- =============================================================================

-- =============================================================================
-- 推荐请求表 (recommendation_requests)
-- 记录每次推荐请求的上下文
-- =============================================================================
CREATE TABLE recommendation_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- 请求标识
    request_id VARCHAR(100) NOT NULL UNIQUE,
    trace_id VARCHAR(100),
    
    -- 用户信息
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id VARCHAR(100),
    
    -- 请求上下文
    scene VARCHAR(100) NOT NULL,  -- 场景：homepage, detail_page, search, etc.
    page_size INTEGER DEFAULT 10,
    page_num INTEGER DEFAULT 1,
    
    -- 请求参数
    request_params JSONB DEFAULT '{}',
    
    -- 设备信息
    device_type VARCHAR(50),
    platform VARCHAR(50),
    
    -- 地理位置
    country VARCHAR(100),
    city VARCHAR(100),
    
    -- 性能指标
    latency_ms INTEGER,
    model_latency_ms INTEGER,
    
    -- 使用的模型/实验
    model_version VARCHAR(100),
    experiment_id VARCHAR(100),
    experiment_group VARCHAR(50),
    
    -- 状态
    status VARCHAR(20) DEFAULT 'success' 
        CHECK (status IN ('success', 'partial', 'failed', 'timeout')),
    error_message TEXT,
    
    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- 推荐请求表注释
COMMENT ON TABLE recommendation_requests IS '推荐请求记录表（按月分区）';
COMMENT ON COLUMN recommendation_requests.scene IS '推荐场景标识';
COMMENT ON COLUMN recommendation_requests.experiment_group IS 'AB 测试分组';

-- 创建分区（2024-2026）
CREATE TABLE recommendation_requests_2024_01 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE recommendation_requests_2024_02 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
CREATE TABLE recommendation_requests_2024_03 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');
CREATE TABLE recommendation_requests_2024_04 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');
CREATE TABLE recommendation_requests_2024_05 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');
CREATE TABLE recommendation_requests_2024_06 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');
CREATE TABLE recommendation_requests_2024_07 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');
CREATE TABLE recommendation_requests_2024_08 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');
CREATE TABLE recommendation_requests_2024_09 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');
CREATE TABLE recommendation_requests_2024_10 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');
CREATE TABLE recommendation_requests_2024_11 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');
CREATE TABLE recommendation_requests_2024_12 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');

CREATE TABLE recommendation_requests_2025_01 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE recommendation_requests_2025_02 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE recommendation_requests_2025_03 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE recommendation_requests_2025_04 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE recommendation_requests_2025_05 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE recommendation_requests_2025_06 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE recommendation_requests_2025_07 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE recommendation_requests_2025_08 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE recommendation_requests_2025_09 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE recommendation_requests_2025_10 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE recommendation_requests_2025_11 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE recommendation_requests_2025_12 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

CREATE TABLE recommendation_requests_2026_01 PARTITION OF recommendation_requests
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

-- 推荐请求表索引
CREATE INDEX idx_rec_requests_user_id ON recommendation_requests(user_id);
CREATE INDEX idx_rec_requests_scene ON recommendation_requests(scene);
CREATE INDEX idx_rec_requests_experiment ON recommendation_requests(experiment_id);
CREATE INDEX idx_rec_requests_status ON recommendation_requests(status);
CREATE INDEX idx_rec_requests_created_at ON recommendation_requests(created_at);
CREATE INDEX idx_rec_requests_latency ON recommendation_requests(latency_ms);

-- =============================================================================
-- 推荐结果表 (recommendation_results)
-- 记录每次推荐返回的物品列表
-- =============================================================================
CREATE TABLE recommendation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- 关联推荐请求
    request_id VARCHAR(100) NOT NULL,
    
    -- 物品信息
    item_id UUID NOT NULL,
    
    -- 排序位置
    position INTEGER NOT NULL,
    
    -- 模型分数
    score DECIMAL(10, 6),
    
    -- 召回来源
    recall_source VARCHAR(100),  -- semantic_id, collaborative, content, popular, etc.
    
    -- 推荐原因（可用于解释性推荐）
    reason JSONB DEFAULT '{}',
    
    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- 推荐结果表注释
COMMENT ON TABLE recommendation_results IS '推荐结果明细表（按月分区）';
COMMENT ON COLUMN recommendation_results.recall_source IS '召回来源标识';
COMMENT ON COLUMN recommendation_results.reason IS '推荐原因，用于可解释性';

-- 创建分区（2024-2026）
CREATE TABLE recommendation_results_2024_01 PARTITION OF recommendation_results
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE recommendation_results_2024_02 PARTITION OF recommendation_results
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
CREATE TABLE recommendation_results_2024_03 PARTITION OF recommendation_results
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');
CREATE TABLE recommendation_results_2024_04 PARTITION OF recommendation_results
    FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');
CREATE TABLE recommendation_results_2024_05 PARTITION OF recommendation_results
    FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');
CREATE TABLE recommendation_results_2024_06 PARTITION OF recommendation_results
    FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');
CREATE TABLE recommendation_results_2024_07 PARTITION OF recommendation_results
    FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');
CREATE TABLE recommendation_results_2024_08 PARTITION OF recommendation_results
    FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');
CREATE TABLE recommendation_results_2024_09 PARTITION OF recommendation_results
    FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');
CREATE TABLE recommendation_results_2024_10 PARTITION OF recommendation_results
    FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');
CREATE TABLE recommendation_results_2024_11 PARTITION OF recommendation_results
    FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');
CREATE TABLE recommendation_results_2024_12 PARTITION OF recommendation_results
    FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');

CREATE TABLE recommendation_results_2025_01 PARTITION OF recommendation_results
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE recommendation_results_2025_02 PARTITION OF recommendation_results
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE recommendation_results_2025_03 PARTITION OF recommendation_results
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE recommendation_results_2025_04 PARTITION OF recommendation_results
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE recommendation_results_2025_05 PARTITION OF recommendation_results
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE recommendation_results_2025_06 PARTITION OF recommendation_results
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE recommendation_results_2025_07 PARTITION OF recommendation_results
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE recommendation_results_2025_08 PARTITION OF recommendation_results
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE recommendation_results_2025_09 PARTITION OF recommendation_results
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE recommendation_results_2025_10 PARTITION OF recommendation_results
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE recommendation_results_2025_11 PARTITION OF recommendation_results
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE recommendation_results_2025_12 PARTITION OF recommendation_results
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

CREATE TABLE recommendation_results_2026_01 PARTITION OF recommendation_results
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

-- 推荐结果表索引
CREATE INDEX idx_rec_results_request_id ON recommendation_results(request_id);
CREATE INDEX idx_rec_results_item_id ON recommendation_results(item_id);
CREATE INDEX idx_rec_results_recall_source ON recommendation_results(recall_source);

-- =============================================================================
-- AB 测试实验表 (experiments)
-- =============================================================================
CREATE TABLE experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- 实验标识
    experiment_id VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- 实验配置
    config JSONB NOT NULL DEFAULT '{}',
    
    -- 流量配置
    traffic_percentage DECIMAL(5, 2) DEFAULT 0,  -- 0-100
    
    -- 分组配置
    groups JSONB NOT NULL DEFAULT '[]',
    -- 格式: [{"name": "control", "percentage": 50}, {"name": "treatment", "percentage": 50}]
    
    -- 目标指标
    primary_metric VARCHAR(100),
    secondary_metrics TEXT[] DEFAULT '{}',
    
    -- 实验状态
    status VARCHAR(20) DEFAULT 'draft' 
        CHECK (status IN ('draft', 'running', 'paused', 'completed', 'archived')),
    
    -- 时间范围
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    
    -- 创建信息
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 实验表注释
COMMENT ON TABLE experiments IS 'AB 测试实验配置表';
COMMENT ON COLUMN experiments.traffic_percentage IS '实验流量占比（0-100%）';
COMMENT ON COLUMN experiments.groups IS '实验分组配置，JSON 数组格式';

-- 实验表索引
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_experiments_start_time ON experiments(start_time);
CREATE INDEX idx_experiments_end_time ON experiments(end_time);

-- 实验表更新时间触发器
CREATE TRIGGER experiments_updated_at
    BEFORE UPDATE ON experiments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 实验指标聚合表 (experiment_metrics)
-- =============================================================================
CREATE TABLE experiment_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- 关联实验
    experiment_id VARCHAR(100) NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    
    -- 分组
    experiment_group VARCHAR(50) NOT NULL,
    
    -- 时间维度（小时级聚合）
    metric_hour TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- 样本量
    sample_count BIGINT DEFAULT 0,
    user_count BIGINT DEFAULT 0,
    
    -- 核心指标
    impressions BIGINT DEFAULT 0,
    clicks BIGINT DEFAULT 0,
    conversions BIGINT DEFAULT 0,
    
    -- 计算指标
    ctr DECIMAL(10, 6) DEFAULT 0,
    cvr DECIMAL(10, 6) DEFAULT 0,
    
    -- 业务指标
    total_revenue DECIMAL(15, 2) DEFAULT 0,
    avg_watch_time DECIMAL(10, 2) DEFAULT 0,
    
    -- 自定义指标（JSON 格式）
    custom_metrics JSONB DEFAULT '{}',
    
    -- 更新时间
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 唯一约束
    UNIQUE(experiment_id, experiment_group, metric_hour)
);

-- 实验指标表注释
COMMENT ON TABLE experiment_metrics IS '实验指标聚合表（小时级）';

-- 实验指标表索引
CREATE INDEX idx_exp_metrics_experiment_id ON experiment_metrics(experiment_id);
CREATE INDEX idx_exp_metrics_hour ON experiment_metrics(metric_hour);
CREATE INDEX idx_exp_metrics_exp_group ON experiment_metrics(experiment_id, experiment_group);

-- =============================================================================
-- 用户实验分配表 (user_experiment_assignments)
-- =============================================================================
CREATE TABLE user_experiment_assignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- 用户和实验
    user_id UUID NOT NULL,
    experiment_id VARCHAR(100) NOT NULL,
    
    -- 分配的组
    assigned_group VARCHAR(50) NOT NULL,
    
    -- 分配时间
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 唯一约束（一个用户在一个实验中只能有一个分组）
    UNIQUE(user_id, experiment_id)
);

-- 用户实验分配表索引
CREATE INDEX idx_user_exp_assign_user ON user_experiment_assignments(user_id);
CREATE INDEX idx_user_exp_assign_exp ON user_experiment_assignments(experiment_id);

