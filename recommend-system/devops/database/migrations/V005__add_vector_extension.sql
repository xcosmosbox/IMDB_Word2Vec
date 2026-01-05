-- =============================================================================
-- V005: 向量扩展与嵌入存储
-- 创建时间: 2024-01-05
-- 描述: 添加 pgvector 扩展，创建向量存储表和相似度搜索功能
-- 项目: 生成式推荐系统
-- 注意: 需要预先安装 pgvector 扩展
-- =============================================================================

-- =============================================================================
-- 安装 pgvector 扩展
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- 物品嵌入表 (item_embeddings)
-- 存储物品的向量表示
-- =============================================================================
CREATE TABLE item_embeddings (
    -- 主键（与物品表关联）
    item_id UUID PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    
    -- 嵌入向量 (256维)
    embedding vector(256) NOT NULL,
    
    -- 向量元数据
    model_version VARCHAR(50) NOT NULL DEFAULT 'v1.0',
    embedding_type VARCHAR(50) NOT NULL DEFAULT 'content',  -- content, collaborative, hybrid
    
    -- 质量指标
    quality_score DECIMAL(5, 4),  -- 向量质量分数
    
    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 物品嵌入表注释
COMMENT ON TABLE item_embeddings IS '物品嵌入向量表，存储物品的 256 维向量表示';
COMMENT ON COLUMN item_embeddings.embedding IS '物品嵌入向量 (256维)';
COMMENT ON COLUMN item_embeddings.embedding_type IS '嵌入类型: content/collaborative/hybrid';

-- 物品嵌入表触发器
CREATE TRIGGER item_embeddings_updated_at
    BEFORE UPDATE ON item_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 用户嵌入表 (user_embeddings)
-- 存储用户的向量表示
-- =============================================================================
CREATE TABLE user_embeddings (
    -- 主键（与用户表关联）
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    
    -- 嵌入向量 (256维)
    embedding vector(256) NOT NULL,
    
    -- 长期偏好向量
    long_term_embedding vector(256),
    
    -- 短期偏好向量（最近行为）
    short_term_embedding vector(256),
    
    -- 向量元数据
    model_version VARCHAR(50) NOT NULL DEFAULT 'v1.0',
    
    -- 向量基于的行为数量
    behavior_count INTEGER DEFAULT 0,
    
    -- 最后更新的行为时间
    last_behavior_at TIMESTAMP WITH TIME ZONE,
    
    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 用户嵌入表注释
COMMENT ON TABLE user_embeddings IS '用户嵌入向量表，存储用户的多维度向量表示';
COMMENT ON COLUMN user_embeddings.long_term_embedding IS '用户长期偏好向量';
COMMENT ON COLUMN user_embeddings.short_term_embedding IS '用户短期偏好向量（基于最近行为）';

-- 用户嵌入表触发器
CREATE TRIGGER user_embeddings_updated_at
    BEFORE UPDATE ON user_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 语义 ID 码本表 (semantic_codebooks)
-- 存储 RQ-VAE 生成的语义 ID 码本
-- =============================================================================
CREATE TABLE semantic_codebooks (
    id SERIAL PRIMARY KEY,
    
    -- 码本层级 (1, 2, 3)
    level INTEGER NOT NULL CHECK (level BETWEEN 1 AND 3),
    
    -- 码本 ID
    code_id INTEGER NOT NULL,
    
    -- 码本向量
    code_vector vector(256) NOT NULL,
    
    -- 码本描述（可选，用于可解释性）
    description TEXT,
    
    -- 使用频率统计
    usage_count BIGINT DEFAULT 0,
    
    -- 元数据
    metadata JSONB DEFAULT '{}',
    
    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 唯一约束
    UNIQUE(level, code_id)
);

-- 语义码本表注释
COMMENT ON TABLE semantic_codebooks IS 'RQ-VAE 语义 ID 码本表';
COMMENT ON COLUMN semantic_codebooks.level IS '码本层级: 1=粗粒度类目, 2=细粒度属性, 3=实例区分';
COMMENT ON COLUMN semantic_codebooks.code_vector IS '码本中心向量 (256维)';

-- 语义码本索引
CREATE INDEX idx_semantic_codebooks_level ON semantic_codebooks(level);
CREATE INDEX idx_semantic_codebooks_usage ON semantic_codebooks(usage_count DESC);

-- =============================================================================
-- 向量索引 (IVFFlat - 适合中等规模数据)
-- =============================================================================

-- 物品嵌入向量索引 (余弦相似度)
CREATE INDEX idx_item_embeddings_cosine ON item_embeddings 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- 物品嵌入向量索引 (内积)
CREATE INDEX idx_item_embeddings_ip ON item_embeddings 
    USING ivfflat (embedding vector_ip_ops)
    WITH (lists = 100);

-- 用户嵌入向量索引 (余弦相似度)
CREATE INDEX idx_user_embeddings_cosine ON user_embeddings 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- 用户长期偏好向量索引
CREATE INDEX idx_user_embeddings_long_term ON user_embeddings 
    USING ivfflat (long_term_embedding vector_cosine_ops)
    WITH (lists = 100)
    WHERE long_term_embedding IS NOT NULL;

-- 语义码本向量索引
CREATE INDEX idx_semantic_codebooks_vector ON semantic_codebooks 
    USING ivfflat (code_vector vector_cosine_ops)
    WITH (lists = 50);

-- =============================================================================
-- 向量相似度搜索函数
-- =============================================================================

-- 函数：基于用户向量查找相似物品
CREATE OR REPLACE FUNCTION find_similar_items_for_user(
    p_user_id UUID,
    p_item_type VARCHAR(50) DEFAULT NULL,
    p_limit INTEGER DEFAULT 50,
    p_min_similarity DECIMAL DEFAULT 0.5
)
RETURNS TABLE (
    item_id UUID,
    title VARCHAR(500),
    type VARCHAR(50),
    similarity DECIMAL
) AS $$
DECLARE
    user_vector vector(256);
BEGIN
    -- 获取用户向量
    SELECT embedding INTO user_vector
    FROM user_embeddings
    WHERE user_id = p_user_id;
    
    IF user_vector IS NULL THEN
        RAISE EXCEPTION 'User embedding not found for user_id: %', p_user_id;
    END IF;
    
    RETURN QUERY
    SELECT 
        i.id AS item_id,
        i.title,
        i.type,
        (1 - (ie.embedding <=> user_vector))::DECIMAL(10, 6) AS similarity
    FROM item_embeddings ie
    JOIN items i ON ie.item_id = i.id
    WHERE i.status = 'active'
        AND (p_item_type IS NULL OR i.type = p_item_type)
        AND (1 - (ie.embedding <=> user_vector)) >= p_min_similarity
        -- 排除已交互物品
        AND NOT EXISTS (
            SELECT 1 FROM user_behaviors ub 
            WHERE ub.user_id = p_user_id 
                AND ub.item_id = i.id 
                AND ub.action IN ('click', 'buy')
                AND ub.created_at > NOW() - INTERVAL '7 days'
        )
    ORDER BY ie.embedding <=> user_vector
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- 函数：查找相似物品
CREATE OR REPLACE FUNCTION find_similar_items(
    p_item_id UUID,
    p_limit INTEGER DEFAULT 20,
    p_same_type BOOLEAN DEFAULT TRUE
)
RETURNS TABLE (
    similar_item_id UUID,
    title VARCHAR(500),
    type VARCHAR(50),
    similarity DECIMAL
) AS $$
DECLARE
    target_vector vector(256);
    target_type VARCHAR(50);
BEGIN
    -- 获取目标物品的向量和类型
    SELECT ie.embedding, i.type INTO target_vector, target_type
    FROM item_embeddings ie
    JOIN items i ON ie.item_id = i.id
    WHERE ie.item_id = p_item_id;
    
    IF target_vector IS NULL THEN
        RAISE EXCEPTION 'Item embedding not found for item_id: %', p_item_id;
    END IF;
    
    RETURN QUERY
    SELECT 
        i.id AS similar_item_id,
        i.title,
        i.type,
        (1 - (ie.embedding <=> target_vector))::DECIMAL(10, 6) AS similarity
    FROM item_embeddings ie
    JOIN items i ON ie.item_id = i.id
    WHERE i.status = 'active'
        AND ie.item_id != p_item_id
        AND (NOT p_same_type OR i.type = target_type)
    ORDER BY ie.embedding <=> target_vector
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- 函数：查找相似用户
CREATE OR REPLACE FUNCTION find_similar_users(
    p_user_id UUID,
    p_limit INTEGER DEFAULT 50
)
RETURNS TABLE (
    similar_user_id UUID,
    similarity DECIMAL
) AS $$
DECLARE
    target_vector vector(256);
BEGIN
    -- 获取目标用户的向量
    SELECT embedding INTO target_vector
    FROM user_embeddings
    WHERE user_id = p_user_id;
    
    IF target_vector IS NULL THEN
        RAISE EXCEPTION 'User embedding not found for user_id: %', p_user_id;
    END IF;
    
    RETURN QUERY
    SELECT 
        ue.user_id AS similar_user_id,
        (1 - (ue.embedding <=> target_vector))::DECIMAL(10, 6) AS similarity
    FROM user_embeddings ue
    JOIN users u ON ue.user_id = u.id
    WHERE ue.user_id != p_user_id
        AND u.status = 'active'
    ORDER BY ue.embedding <=> target_vector
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- 函数：基于语义 ID 查找物品
CREATE OR REPLACE FUNCTION find_items_by_semantic_id(
    p_semantic_id INTEGER[],
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    item_id UUID,
    title VARCHAR(500),
    type VARCHAR(50),
    semantic_id INTEGER[],
    match_level INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        i.id AS item_id,
        i.title,
        i.type,
        i.semantic_id,
        -- 计算匹配层级
        CASE 
            WHEN i.semantic_id = p_semantic_id THEN 3  -- 完全匹配
            WHEN i.semantic_id[1:2] = p_semantic_id[1:2] THEN 2  -- Level 1+2 匹配
            WHEN i.semantic_id[1] = p_semantic_id[1] THEN 1  -- Level 1 匹配
            ELSE 0
        END AS match_level
    FROM items i
    WHERE i.status = 'active'
        AND i.semantic_id IS NOT NULL
        AND i.semantic_id[1] = p_semantic_id[1]  -- 至少 Level 1 匹配
    ORDER BY match_level DESC, i.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 批量更新嵌入向量函数
-- =============================================================================
CREATE OR REPLACE FUNCTION batch_upsert_item_embeddings(
    p_embeddings JSONB  -- 格式: [{"item_id": "uuid", "embedding": [0.1, 0.2, ...], "model_version": "v1"}]
)
RETURNS INTEGER AS $$
DECLARE
    embedding_record JSONB;
    inserted_count INTEGER := 0;
BEGIN
    FOR embedding_record IN SELECT * FROM jsonb_array_elements(p_embeddings)
    LOOP
        INSERT INTO item_embeddings (item_id, embedding, model_version)
        VALUES (
            (embedding_record->>'item_id')::UUID,
            (embedding_record->>'embedding')::vector(256),
            COALESCE(embedding_record->>'model_version', 'v1.0')
        )
        ON CONFLICT (item_id) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            model_version = EXCLUDED.model_version,
            updated_at = NOW();
        
        inserted_count := inserted_count + 1;
    END LOOP;
    
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 向量质量监控视图
-- =============================================================================
CREATE OR REPLACE VIEW v_embedding_stats AS
SELECT 
    'item_embeddings' AS table_name,
    COUNT(*) AS total_count,
    COUNT(*) FILTER (WHERE updated_at > NOW() - INTERVAL '7 days') AS recent_updates,
    AVG(vector_dims(embedding)) AS avg_dims,
    MIN(created_at) AS oldest_embedding,
    MAX(updated_at) AS latest_update
FROM item_embeddings
UNION ALL
SELECT 
    'user_embeddings' AS table_name,
    COUNT(*) AS total_count,
    COUNT(*) FILTER (WHERE updated_at > NOW() - INTERVAL '7 days') AS recent_updates,
    AVG(vector_dims(embedding)) AS avg_dims,
    MIN(created_at) AS oldest_embedding,
    MAX(updated_at) AS latest_update
FROM user_embeddings;

-- =============================================================================
-- 向量索引维护函数
-- =============================================================================
CREATE OR REPLACE FUNCTION maintain_vector_indexes()
RETURNS void AS $$
BEGIN
    -- 重建物品嵌入索引
    REINDEX INDEX CONCURRENTLY idx_item_embeddings_cosine;
    REINDEX INDEX CONCURRENTLY idx_item_embeddings_ip;
    
    -- 重建用户嵌入索引
    REINDEX INDEX CONCURRENTLY idx_user_embeddings_cosine;
    
    RAISE NOTICE 'Vector indexes maintained successfully';
END;
$$ LANGUAGE plpgsql;

