# Person E: 数据库管理

## 你的角色
你是一名 DevOps 工程师，负责实现生成式推荐系统的 **数据库管理**，包括数据库迁移、备份恢复、向量数据库管理等。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读接口定义文件：**

```
devops/interfaces.yaml
```

你需要实现的契约：

```yaml
database:
  migrations:
    format: "V{version}__{description}.sql"
    naming: snake_case
  
  backup:
    frequency: daily
    retention: 30_days
    format: pg_dump
```

---

## 你的任务

```
devops/database/
├── migrations/
│   ├── flyway.conf
│   ├── V001__initial_schema.sql
│   ├── V002__add_user_behaviors.sql
│   ├── V003__add_recommendations.sql
│   ├── V004__add_indexes.sql
│   └── V005__add_vector_extension.sql
├── backup/
│   ├── backup.sh
│   ├── restore.sh
│   ├── verify.sh
│   └── cronjob.yaml
├── milvus/
│   ├── collections.py
│   ├── indexes.py
│   └── maintenance.py
└── scripts/
    ├── init-db.sh
    ├── seed-data.sh
    └── cleanup.sh
```

---

## 1. 数据库迁移 - 初始 Schema (V001__initial_schema.sql)

```sql
-- =============================================================================
-- V001: 初始数据库 Schema
-- 创建时间: 2024-01-01
-- 描述: 创建基础表结构
-- =============================================================================

-- 启用扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- 模糊搜索

-- =============================================================================
-- 用户表
-- =============================================================================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    age SMALLINT CHECK (age >= 0 AND age <= 150),
    gender VARCHAR(20),
    metadata JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'banned')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 用户表索引
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_users_name_trgm ON users USING gin(name gin_trgm_ops);

-- 更新时间触发器
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 物品表
-- =============================================================================
CREATE TABLE items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type VARCHAR(50) NOT NULL CHECK (type IN ('movie', 'product', 'article', 'video')),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'deleted')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 物品表索引
CREATE INDEX idx_items_type ON items(type);
CREATE INDEX idx_items_category ON items(category);
CREATE INDEX idx_items_status ON items(status);
CREATE INDEX idx_items_tags ON items USING gin(tags);
CREATE INDEX idx_items_title_trgm ON items USING gin(title gin_trgm_ops);
CREATE INDEX idx_items_created_at ON items(created_at);

CREATE TRIGGER items_updated_at
    BEFORE UPDATE ON items
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 物品统计表
-- =============================================================================
CREATE TABLE item_stats (
    item_id UUID PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    view_count BIGINT DEFAULT 0,
    click_count BIGINT DEFAULT 0,
    like_count BIGINT DEFAULT 0,
    share_count BIGINT DEFAULT 0,
    avg_rating DECIMAL(3, 2) DEFAULT 0,
    rating_count BIGINT DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 统计表触发器
CREATE TRIGGER item_stats_updated_at
    BEFORE UPDATE ON item_stats
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
```

---

## 2. 用户行为表 (V002__add_user_behaviors.sql)

```sql
-- =============================================================================
-- V002: 用户行为表
-- 创建时间: 2024-01-02
-- 描述: 添加用户行为记录表
-- =============================================================================

-- 用户行为表（分区表）
CREATE TABLE user_behaviors (
    id UUID DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    item_id UUID NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    action VARCHAR(50) NOT NULL CHECK (action IN ('view', 'click', 'like', 'dislike', 'buy', 'share', 'rating')),
    value DECIMAL(5, 2),  -- 用于评分等数值型行为
    context JSONB DEFAULT '{}',
    session_id VARCHAR(100),
    device_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- 创建月度分区
CREATE TABLE user_behaviors_2024_01 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE user_behaviors_2024_02 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

CREATE TABLE user_behaviors_2024_03 PARTITION OF user_behaviors
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');

-- 创建自动分区函数
CREATE OR REPLACE FUNCTION create_behavior_partition()
RETURNS TRIGGER AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    partition_date := DATE_TRUNC('month', NEW.created_at);
    partition_name := 'user_behaviors_' || TO_CHAR(partition_date, 'YYYY_MM');
    start_date := partition_date;
    end_date := partition_date + INTERVAL '1 month';
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = partition_name
    ) THEN
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF user_behaviors FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            start_date,
            end_date
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 行为表索引（在每个分区上）
CREATE INDEX idx_behaviors_user_id ON user_behaviors(user_id);
CREATE INDEX idx_behaviors_item_id ON user_behaviors(item_id);
CREATE INDEX idx_behaviors_action ON user_behaviors(action);
CREATE INDEX idx_behaviors_user_action ON user_behaviors(user_id, action, created_at DESC);

-- 用户画像表
CREATE TABLE user_profiles (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    total_actions BIGINT DEFAULT 0,
    preferred_types JSONB DEFAULT '{}',
    preferred_categories JSONB DEFAULT '{}',
    active_hours JSONB DEFAULT '{}',
    last_active TIMESTAMP WITH TIME ZONE,
    profile_vector BYTEA,  -- 用户向量（序列化）
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TRIGGER user_profiles_updated_at
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
```

---

## 3. 备份脚本 (backup/backup.sh)

```bash
#!/bin/bash
# =============================================================================
# PostgreSQL 数据库备份脚本
# =============================================================================

set -euo pipefail

# 配置
BACKUP_DIR="${BACKUP_DIR:-/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-recommend}"
DB_USER="${POSTGRES_USER:-postgres}"
S3_BUCKET="${S3_BUCKET:-}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/${DB_NAME}_${TIMESTAMP}.sql.gz"
BACKUP_FILE_LATEST="${BACKUP_DIR}/${DB_NAME}_latest.sql.gz"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

notify_slack() {
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$1\"}" \
            "$SLACK_WEBHOOK" || true
    fi
}

# 创建备份目录
mkdir -p "$BACKUP_DIR"

log "Starting backup of database: $DB_NAME"

# 执行备份
PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --format=plain \
    --no-owner \
    --no-privileges \
    --verbose \
    2>&1 | gzip > "$BACKUP_FILE"

BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
log "Backup completed: $BACKUP_FILE (Size: $BACKUP_SIZE)"

# 创建最新备份链接
ln -sf "$BACKUP_FILE" "$BACKUP_FILE_LATEST"

# 上传到 S3（如果配置）
if [[ -n "$S3_BUCKET" ]]; then
    log "Uploading to S3: s3://$S3_BUCKET/backups/"
    aws s3 cp "$BACKUP_FILE" "s3://$S3_BUCKET/backups/$(basename $BACKUP_FILE)"
    log "S3 upload completed"
fi

# 清理旧备份
log "Cleaning up backups older than $RETENTION_DAYS days"
find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete

# 统计当前备份数量
BACKUP_COUNT=$(find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz" -type f | wc -l)
log "Current backup count: $BACKUP_COUNT"

# 验证备份
log "Verifying backup integrity..."
if gzip -t "$BACKUP_FILE"; then
    log "Backup verification passed"
    notify_slack "✅ Database backup completed: $DB_NAME ($BACKUP_SIZE)"
else
    log "ERROR: Backup verification failed!"
    notify_slack "❌ Database backup FAILED: $DB_NAME"
    exit 1
fi

log "Backup process completed successfully"
```

---

## 4. 恢复脚本 (backup/restore.sh)

```bash
#!/bin/bash
# =============================================================================
# PostgreSQL 数据库恢复脚本
# =============================================================================

set -euo pipefail

# 配置
BACKUP_DIR="${BACKUP_DIR:-/backups}"
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-recommend}"
DB_USER="${POSTGRES_USER:-postgres}"

# 参数
BACKUP_FILE="${1:-}"
DRY_RUN="${DRY_RUN:-false}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

usage() {
    echo "Usage: $0 <backup_file>"
    echo ""
    echo "Options:"
    echo "  DRY_RUN=true  Only show what would be done"
    echo ""
    echo "Examples:"
    echo "  $0 recommend_20240101_120000.sql.gz"
    echo "  $0 latest"
    echo "  DRY_RUN=true $0 recommend_latest.sql.gz"
    exit 1
}

if [[ -z "$BACKUP_FILE" ]]; then
    usage
fi

# 处理 "latest" 快捷方式
if [[ "$BACKUP_FILE" == "latest" ]]; then
    BACKUP_FILE="${BACKUP_DIR}/${DB_NAME}_latest.sql.gz"
elif [[ ! "$BACKUP_FILE" == /* ]]; then
    BACKUP_FILE="${BACKUP_DIR}/${BACKUP_FILE}"
fi

# 验证备份文件
if [[ ! -f "$BACKUP_FILE" ]]; then
    log "ERROR: Backup file not found: $BACKUP_FILE"
    exit 1
fi

log "Restore target: $BACKUP_FILE"
log "Target database: $DB_NAME @ $DB_HOST:$DB_PORT"

# 确认
if [[ "$DRY_RUN" != "true" ]]; then
    read -p "This will REPLACE all data in $DB_NAME. Continue? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        log "Restore cancelled"
        exit 0
    fi
fi

# 验证备份完整性
log "Verifying backup integrity..."
if ! gzip -t "$BACKUP_FILE"; then
    log "ERROR: Backup file is corrupted"
    exit 1
fi
log "Backup integrity verified"

if [[ "$DRY_RUN" == "true" ]]; then
    log "DRY RUN: Would restore from $BACKUP_FILE"
    log "DRY RUN: Would drop and recreate database $DB_NAME"
    exit 0
fi

# 断开现有连接
log "Terminating existing connections..."
PGPASSWORD="$POSTGRES_PASSWORD" psql \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d postgres \
    -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$DB_NAME' AND pid <> pg_backend_pid();" \
    || true

# 删除并重建数据库
log "Dropping database..."
PGPASSWORD="$POSTGRES_PASSWORD" dropdb \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    "$DB_NAME" \
    --if-exists

log "Creating database..."
PGPASSWORD="$POSTGRES_PASSWORD" createdb \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    "$DB_NAME"

# 恢复数据
log "Restoring data..."
gunzip -c "$BACKUP_FILE" | PGPASSWORD="$POSTGRES_PASSWORD" psql \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --quiet

# 验证恢复
log "Verifying restore..."
TABLE_COUNT=$(PGPASSWORD="$POSTGRES_PASSWORD" psql \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")

log "Restored tables: $TABLE_COUNT"

# 刷新统计信息
log "Analyzing database..."
PGPASSWORD="$POSTGRES_PASSWORD" psql \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    -c "ANALYZE;"

log "Restore completed successfully"
```

---

## 5. Milvus 集合管理 (milvus/collections.py)

```python
"""
Milvus 向量数据库集合管理
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
import os
from typing import Optional


class MilvusManager:
    """Milvus 集合管理器"""
    
    def __init__(
        self,
        host: str = None,
        port: str = None,
    ):
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        self._connected = False
    
    def connect(self):
        """连接到 Milvus"""
        if not self._connected:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )
            self._connected = True
            print(f"Connected to Milvus at {self.host}:{self.port}")
    
    def disconnect(self):
        """断开连接"""
        if self._connected:
            connections.disconnect("default")
            self._connected = False
    
    def create_item_embeddings_collection(
        self,
        dim: int = 256,
        collection_name: str = "item_embeddings",
    ) -> Collection:
        """
        创建物品嵌入集合
        
        Schema:
        - item_id: 物品 ID (主键)
        - embedding: 嵌入向量
        - item_type: 物品类型
        - created_at: 创建时间
        """
        self.connect()
        
        # 检查是否已存在
        if utility.has_collection(collection_name):
            print(f"Collection {collection_name} already exists")
            return Collection(collection_name)
        
        # 定义 Schema
        fields = [
            FieldSchema(
                name="item_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                is_primary=True,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=dim,
            ),
            FieldSchema(
                name="item_type",
                dtype=DataType.VARCHAR,
                max_length=32,
            ),
            FieldSchema(
                name="created_at",
                dtype=DataType.INT64,
            ),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Item embeddings for similarity search",
        )
        
        collection = Collection(
            name=collection_name,
            schema=schema,
            consistency_level="Strong",
        )
        
        print(f"Created collection: {collection_name}")
        return collection
    
    def create_user_embeddings_collection(
        self,
        dim: int = 256,
        collection_name: str = "user_embeddings",
    ) -> Collection:
        """创建用户嵌入集合"""
        self.connect()
        
        if utility.has_collection(collection_name):
            print(f"Collection {collection_name} already exists")
            return Collection(collection_name)
        
        fields = [
            FieldSchema(
                name="user_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                is_primary=True,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=dim,
            ),
            FieldSchema(
                name="updated_at",
                dtype=DataType.INT64,
            ),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="User embeddings for personalization",
        )
        
        collection = Collection(
            name=collection_name,
            schema=schema,
        )
        
        print(f"Created collection: {collection_name}")
        return collection
    
    def create_indexes(self, collection_name: str):
        """创建向量索引"""
        self.connect()
        
        collection = Collection(collection_name)
        
        # IVF_FLAT 索引（适合中等规模数据）
        index_params = {
            "metric_type": "IP",  # 内积（余弦相似度）
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        
        collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )
        
        print(f"Created index on {collection_name}.embedding")
    
    def load_collection(self, collection_name: str):
        """加载集合到内存"""
        self.connect()
        collection = Collection(collection_name)
        collection.load()
        print(f"Loaded collection: {collection_name}")
    
    def get_stats(self, collection_name: str) -> dict:
        """获取集合统计信息"""
        self.connect()
        collection = Collection(collection_name)
        
        return {
            "name": collection_name,
            "num_entities": collection.num_entities,
            "schema": str(collection.schema),
            "indexes": [str(idx) for idx in collection.indexes],
        }


def init_collections():
    """初始化所有集合"""
    manager = MilvusManager()
    
    try:
        # 创建集合
        manager.create_item_embeddings_collection(dim=256)
        manager.create_user_embeddings_collection(dim=256)
        
        # 创建索引
        manager.create_indexes("item_embeddings")
        manager.create_indexes("user_embeddings")
        
        # 加载集合
        manager.load_collection("item_embeddings")
        manager.load_collection("user_embeddings")
        
        print("\n=== Collection Stats ===")
        print(manager.get_stats("item_embeddings"))
        print(manager.get_stats("user_embeddings"))
        
    finally:
        manager.disconnect()


if __name__ == "__main__":
    init_collections()
```

---

## 注意事项

1. 使用 Flyway 管理迁移版本
2. 分区表提高查询性能
3. 备份脚本支持 S3 上传
4. Milvus 索引选择合适类型
5. 恢复前验证备份完整性

## 输出要求

请输出完整的数据库管理配置，包含：
1. 所有迁移 SQL
2. 备份恢复脚本
3. Milvus 管理脚本
4. K8s CronJob

