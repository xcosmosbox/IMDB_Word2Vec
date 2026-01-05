# 数据库管理模块

## 概述

本模块是生成式推荐系统的数据库管理组件，提供 PostgreSQL 数据库迁移、备份恢复、以及 Milvus 向量数据库管理功能。

## 目录结构

```
database/
├── migrations/               # 数据库迁移文件
│   ├── flyway.conf          # Flyway 配置
│   ├── V001__initial_schema.sql
│   ├── V002__add_user_behaviors.sql
│   ├── V003__add_recommendations.sql
│   ├── V004__add_indexes.sql
│   └── V005__add_vector_extension.sql
│
├── backup/                   # 备份恢复脚本
│   ├── backup.sh            # 备份脚本
│   ├── restore.sh           # 恢复脚本
│   ├── verify.sh            # 验证脚本
│   └── cronjob.yaml         # K8s CronJob 配置
│
├── milvus/                   # Milvus 向量数据库管理
│   ├── __init__.py
│   ├── collections.py       # 集合管理
│   ├── indexes.py           # 索引管理
│   └── maintenance.py       # 维护管理
│
├── scripts/                  # 数据库脚本
│   ├── init-db.sh           # 初始化脚本
│   ├── seed-data.sh         # 种子数据脚本
│   └── cleanup.sh           # 清理脚本
│
├── tests/                    # 单元测试
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_collections.py
│   ├── test_indexes.py
│   ├── test_maintenance.py
│   └── test_migrations.py
│
├── requirements.txt          # Python 依赖
└── README.md                 # 本文档
```

## 快速开始

### 环境要求

- PostgreSQL 15+
- Milvus 2.3+
- Python 3.9+
- Flyway 9.x（可选）

### 环境变量配置

```bash
# PostgreSQL 配置
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=recommend
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your_password

# Milvus 配置
export MILVUS_HOST=localhost
export MILVUS_PORT=19530

# S3 配置（用于备份）
export S3_BUCKET=your-backup-bucket
export AWS_REGION=us-east-1

# 通知配置
export SLACK_WEBHOOK=https://hooks.slack.com/services/xxx
```

### 安装依赖

```bash
cd recommend-system/devops/database
pip install -r requirements.txt
```

## 数据库迁移

### 迁移文件说明

| 版本 | 文件名 | 说明 |
|------|--------|------|
| V001 | initial_schema.sql | 初始 Schema：用户表、物品表、统计表 |
| V002 | add_user_behaviors.sql | 用户行为表（分区表）、用户画像表 |
| V003 | add_recommendations.sql | 推荐记录表、AB 测试表 |
| V004 | add_indexes.sql | 性能优化索引、物化视图 |
| V005 | add_vector_extension.sql | pgvector 扩展、嵌入向量表 |

### 使用 Flyway 运行迁移

```bash
# 运行迁移
flyway -configFiles=migrations/flyway.conf migrate

# 查看迁移状态
flyway -configFiles=migrations/flyway.conf info

# 验证迁移
flyway -configFiles=migrations/flyway.conf validate
```

### 手动运行迁移

```bash
# 使用初始化脚本
./scripts/init-db.sh --full

# 或直接执行 SQL
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f migrations/V001__initial_schema.sql
```

## 数据库 Schema

### 核心表

#### users（用户表）
```sql
- id: UUID (主键)
- name: VARCHAR(100)
- email: VARCHAR(255) (唯一)
- password_hash: VARCHAR(255)
- age, gender, country, city
- metadata: JSONB
- status: active/inactive/banned/deleted
- created_at, updated_at
```

#### items（物品表）
```sql
- id: UUID (主键)
- type: movie/product/article/video
- title, description
- category, subcategory
- tags: TEXT[]
- semantic_id: INTEGER[] (语义 ID)
- metadata: JSONB
- status, created_at, updated_at
```

#### user_behaviors（用户行为表 - 分区）
```sql
- id: UUID
- user_id, item_id
- action: view/click/like/buy/rating/...
- value: DECIMAL
- context: JSONB
- session_id, device_type
- created_at (分区键)
```

### 向量表（需要 pgvector）

#### item_embeddings
```sql
- item_id: UUID (主键)
- embedding: vector(256)
- model_version, embedding_type
- created_at, updated_at
```

#### user_embeddings
```sql
- user_id: UUID (主键)
- embedding: vector(256)
- long_term_embedding, short_term_embedding
- behavior_count
- created_at, updated_at
```

## 备份恢复

### 执行备份

```bash
# 手动备份
./backup/backup.sh

# 自定义备份参数
RETENTION_DAYS=60 S3_BUCKET=my-bucket ./backup/backup.sh
```

### 恢复数据

```bash
# 从最新备份恢复
./backup/restore.sh latest

# 从特定文件恢复
./backup/restore.sh /backups/daily/recommend_20240101_120000.sql.gz

# 恢复到不同数据库
./backup/restore.sh --target-db recommend_test latest

# 从 S3 恢复
./backup/restore.sh --from-s3 daily/recommend_20240101_120000.sql.gz
```

### 验证备份

```bash
# 验证最新备份
./backup/verify.sh latest

# 完整验证（包括恢复测试）
./backup/verify.sh --compare latest
```

### Kubernetes CronJob

部署自动备份：

```bash
kubectl apply -f backup/cronjob.yaml
```

备份策略：
- **每日备份**：每天 02:00，保留 30 天
- **每周备份**：每周日 03:00，保留 12 周
- **每月备份**：每月 1 日 04:00，保留 12 个月
- **验证检查**：每周一 05:00

## Milvus 向量数据库管理

### 基本用法

```python
from milvus.collections import MilvusManager, init_all_collections
from milvus.indexes import IndexManager, IndexType, MetricType
from milvus.maintenance import MaintenanceManager

# 创建管理器
manager = MilvusManager(host="localhost", port="19530")
manager.connect()

# 初始化所有集合
collections = init_all_collections(manager, dim=256)

# 创建索引
index_manager = IndexManager(manager)
index_manager.create_index(
    "item_embeddings",
    "embedding",
    index_type=IndexType.HNSW,
    metric_type=MetricType.IP,
)

# 加载集合
manager.load_collection("item_embeddings")

# 获取统计信息
stats = manager.get_stats("item_embeddings")
print(stats)

# 断开连接
manager.disconnect()
```

### 集合配置

预定义集合：

| 集合名 | 用途 | 维度 | 主键 |
|--------|------|------|------|
| item_embeddings | 物品嵌入向量 | 256 | item_id |
| user_embeddings | 用户嵌入向量 | 256 | user_id |
| query_cache | 查询缓存 | 256 | query_id |

### 索引配置

预设索引配置：

| 预设名 | 索引类型 | 适用场景 |
|--------|----------|----------|
| small_exact | FLAT | < 100K 数据，精确搜索 |
| medium_ivf | IVF_FLAT | 100K - 10M 数据 |
| large_hnsw | HNSW | 10M - 100M 数据 |
| xlarge_diskann | DISKANN | > 100M 数据 |
| high_recall | HNSW | 高召回率场景 |
| low_latency | IVF_SQ8 | 低延迟场景 |

### 维护操作

```python
from milvus.maintenance import MaintenanceManager

maintenance = MaintenanceManager(manager)

# 健康检查
health = maintenance.health_check()
print(f"Healthy: {health.is_healthy}")

# 压缩集合
result = maintenance.compact_collection("item_embeddings")

# 刷新所有集合
maintenance.flush_all_collections()

# 运行完整维护例程
results = maintenance.run_maintenance_routine()
```

## 数据库脚本

### 初始化数据库

```bash
# 完整初始化
./scripts/init-db.sh --full

# 分步执行
./scripts/init-db.sh --create-db
./scripts/init-db.sh --install-extensions
./scripts/init-db.sh --run-migrations
./scripts/init-db.sh --create-users
./scripts/init-db.sh --grant-permissions
```

### 种子数据

```bash
# 使用默认数量
./scripts/seed-data.sh

# 自定义数量
./scripts/seed-data.sh --users 10000 --items 50000 --behaviors 500000

# 清理后重新生成
./scripts/seed-data.sh --clean
```

### 数据清理

```bash
# 执行清理
./scripts/cleanup.sh

# 自定义保留期
./scripts/cleanup.sh --behaviors-days 60 --requests-days 14

# 仅生成报告
./scripts/cleanup.sh --report-only
```

## 运行测试

```bash
# 安装测试依赖
pip install -r requirements.txt

# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_collections.py -v

# 生成覆盖率报告
pytest tests/ --cov=milvus --cov-report=html

# 运行迁移文件测试
pytest tests/test_migrations.py -v
```

## 接口契约

本模块遵循 `devops/interfaces.yaml` 中定义的接口契约：

```yaml
database:
  migrations:
    format: "V{version}__{description}.sql"
    naming: snake_case
  
  backup:
    frequency: daily
    retention: 30_days
    format: pg_dump
  
  pool:
    min_connections: 5
    max_connections: 50
    idle_timeout: 300s
```

## 故障排除

### 常见问题

#### 1. 迁移失败

```bash
# 检查迁移状态
flyway info

# 修复失败的迁移
flyway repair

# 重新运行
flyway migrate
```

#### 2. Milvus 连接失败

```python
# 检查连接
manager = MilvusManager()
if not manager.connect(retry_times=5):
    print("Connection failed")
    
# 检查健康状态
from milvus.maintenance import run_health_check
health = run_health_check(manager)
print(health.to_dict())
```

#### 3. 备份失败

```bash
# 检查磁盘空间
df -h /backups

# 检查数据库连接
pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT

# 查看备份日志
tail -f /var/log/backup/backup_$(date +%Y%m%d).log
```

### 日志位置

- 备份日志: `/var/log/backup/`
- 迁移日志: Flyway 标准输出
- Milvus 日志: Python logging 模块

## 性能优化建议

### PostgreSQL

1. **连接池配置**
   - min_connections: 5
   - max_connections: 50
   - idle_timeout: 300s

2. **索引策略**
   - 使用 BRIN 索引处理时间序列数据
   - 使用 GIN 索引处理 JSONB 和数组字段
   - 使用部分索引减少索引大小

3. **分区策略**
   - 用户行为表按月分区
   - 推荐记录表按月分区
   - 自动创建新分区

### Milvus

1. **索引选择**
   - 数据量 < 100K: FLAT
   - 数据量 < 10M: IVF_FLAT
   - 数据量 < 100M: HNSW
   - 数据量 > 100M: DISKANN

2. **搜索优化**
   - 合理设置 nprobe/ef 参数
   - 使用批量查询
   - 预热常用集合

## 贡献指南

1. 遵循迁移命名规范: `V{版本号}__{描述}.sql`
2. 确保所有新功能都有单元测试
3. 更新相关文档
4. 提交前运行完整测试套件

## 版本历史

- **v1.0.0** - 初始版本
  - PostgreSQL 迁移管理
  - 备份恢复功能
  - Milvus 向量数据库支持
  - 完整的单元测试

## 联系方式

如有问题，请联系 DevOps 团队或提交 Issue。

