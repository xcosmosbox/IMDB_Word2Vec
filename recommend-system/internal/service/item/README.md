# Item Service（物品服务）开发文档

## 目录

1. [概述](#概述)
2. [架构设计](#架构设计)
3. [快速开始](#快速开始)
4. [接口说明](#接口说明)
5. [数据结构](#数据结构)
6. [配置说明](#配置说明)
7. [缓存策略](#缓存策略)
8. [向量搜索](#向量搜索)
9. [单元测试](#单元测试)
10. [常见问题](#常见问题)
11. [开发指南](#开发指南)

---

## 概述

物品服务（Item Service）是生成式推荐系统的核心服务之一，负责管理所有可推荐物品（商品、电影、文章、视频等）的生命周期。

### 主要功能

- **物品 CRUD**：物品的创建、读取、更新、删除
- **物品搜索**：关键词搜索物品
- **相似推荐**：基于向量搜索的相似物品推荐
- **统计管理**：物品浏览量、点击量、点赞量等统计
- **批量操作**：支持批量获取物品信息

### 技术栈

- **语言**: Go 1.21+
- **Web 框架**: Gin
- **数据库**: PostgreSQL
- **缓存**: Redis (多级缓存)
- **向量数据库**: Milvus

---

## 架构设计

### 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (api/item/v1)                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ handler.go - HTTP API 处理器                             ││
│  │ - 请求验证                                                ││
│  │ - 参数绑定                                                ││
│  │ - 响应封装                                                ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                   Service Layer (internal/service/item)      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ service.go - 业务逻辑层                                   ││
│  │ - 物品 CRUD 操作                                          ││
│  │ - 缓存管理                                                ││
│  │ - 向量搜索                                                ││
│  │ - 统计管理                                                ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                Repository Layer (internal/repository)        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ item_repo.go - 数据访问层                                 ││
│  │ - PostgreSQL 操作                                         ││
│  │ - Milvus 向量操作                                         ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 接口驱动开发

所有服务实现都遵循 `internal/interfaces/interfaces.go` 中定义的接口：

```go
// ItemService 物品服务接口
type ItemService interface {
    GetItem(ctx context.Context, itemID string) (*Item, error)
    CreateItem(ctx context.Context, req *CreateItemRequest) (*Item, error)
    UpdateItem(ctx context.Context, itemID string, req *UpdateItemRequest) (*Item, error)
    DeleteItem(ctx context.Context, itemID string) error
    ListItems(ctx context.Context, req *ListItemsRequest) (*ListItemsResponse, error)
    SearchItems(ctx context.Context, query string, limit int) ([]*Item, error)
    GetSimilarItems(ctx context.Context, itemID string, topK int) ([]*SimilarItem, error)
    BatchGetItems(ctx context.Context, itemIDs []string) ([]*Item, error)
    GetItemStats(ctx context.Context, itemID string) (*ItemStats, error)
}
```

---

## 快速开始

### 1. 环境准备

确保已安装以下依赖：

```bash
# Go 1.21+
go version

# PostgreSQL
psql --version

# Redis
redis-server --version

# Milvus (可选，用于向量搜索)
```

### 2. 配置文件

编辑 `configs/config.yaml`：

```yaml
server:
  mode: debug
  http_port: 8083
  read_timeout: 30s
  write_timeout: 30s
  shutdown_timeout: 30s

database:
  host: localhost
  port: 5432
  user: postgres
  password: your_password
  dbname: recommend_system

redis:
  addrs:
    - localhost:6379
  password: ""
  db: 0

milvus:
  address: localhost
  port: 19530
  collection: item_embeddings

log:
  level: debug
  format: console
  output: stdout
```

### 3. 启动服务

```bash
# 进入项目目录
cd recommend-system

# 编译
go build -o item-service ./cmd/item-service

# 运行
./item-service -config configs/config.yaml
```

### 4. 验证服务

```bash
# 健康检查
curl http://localhost:8083/health

# 创建物品
curl -X POST http://localhost:8083/api/v1/items \
  -H "Content-Type: application/json" \
  -d '{
    "type": "movie",
    "title": "Test Movie",
    "description": "A test movie",
    "category": "action"
  }'
```

---

## 接口说明

### API 端点列表

| 方法   | 路径                      | 说明           |
|--------|--------------------------|----------------|
| POST   | /api/v1/items            | 创建物品       |
| GET    | /api/v1/items            | 列出物品       |
| GET    | /api/v1/items/search     | 搜索物品       |
| GET    | /api/v1/items/:id        | 获取物品       |
| PUT    | /api/v1/items/:id        | 更新物品       |
| DELETE | /api/v1/items/:id        | 删除物品       |
| GET    | /api/v1/items/:id/similar| 获取相似物品   |
| GET    | /api/v1/items/:id/stats  | 获取物品统计   |
| POST   | /api/v1/items/:id/stats  | 更新物品统计   |
| POST   | /api/v1/items/batch      | 批量获取物品   |

### 请求/响应示例

#### 创建物品

**请求：**
```http
POST /api/v1/items
Content-Type: application/json

{
  "type": "movie",
  "title": "肖申克的救赎",
  "description": "一部经典的越狱电影",
  "category": "drama",
  "tags": ["经典", "越狱", "希望"],
  "metadata": {
    "year": 1994,
    "director": "Frank Darabont",
    "rating": 9.3
  }
}
```

**响应：**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "mov_20260104120000_a1b2c3d4",
    "type": "movie",
    "title": "肖申克的救赎",
    "description": "一部经典的越狱电影",
    "category": "drama",
    "tags": ["经典", "越狱", "希望"],
    "metadata": {
      "year": 1994,
      "director": "Frank Darabont",
      "rating": 9.3
    },
    "status": "active",
    "created_at": "2026-01-04T12:00:00Z",
    "updated_at": "2026-01-04T12:00:00Z"
  }
}
```

#### 获取相似物品

**请求：**
```http
GET /api/v1/items/mov_123456/similar?top_k=10
```

**响应：**
```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "item": {
        "id": "mov_234567",
        "type": "movie",
        "title": "阿甘正传",
        "category": "drama"
      },
      "score": 0.95
    },
    {
      "item": {
        "id": "mov_345678",
        "type": "movie",
        "title": "美丽人生",
        "category": "drama"
      },
      "score": 0.92
    }
  ]
}
```

---

## 数据结构

### Item 物品

```go
type Item struct {
    ID          string                 `json:"id"`          // 物品 ID
    Type        string                 `json:"type"`        // 类型: movie, product, article, video
    Title       string                 `json:"title"`       // 标题
    Description string                 `json:"description"` // 描述
    Category    string                 `json:"category"`    // 类目
    Tags        []string               `json:"tags"`        // 标签
    Metadata    map[string]interface{} `json:"metadata"`    // 元数据
    Status      string                 `json:"status"`      // 状态: active, inactive, deleted
    CreatedAt   time.Time              `json:"created_at"`  // 创建时间
    UpdatedAt   time.Time              `json:"updated_at"`  // 更新时间
}
```

### ItemStats 物品统计

```go
type ItemStats struct {
    ItemID     string  `json:"item_id"`     // 物品 ID
    ViewCount  int64   `json:"view_count"`  // 浏览次数
    ClickCount int64   `json:"click_count"` // 点击次数
    LikeCount  int64   `json:"like_count"`  // 点赞次数
    ShareCount int64   `json:"share_count"` // 分享次数
    AvgRating  float64 `json:"avg_rating"`  // 平均评分
}
```

### SimilarItem 相似物品

```go
type SimilarItem struct {
    Item  *Item   `json:"item"`  // 物品信息
    Score float32 `json:"score"` // 相似度分数
}
```

---

## 配置说明

### Config 服务配置

```go
type Config struct {
    // CacheTTL 缓存过期时间
    CacheTTL time.Duration
    
    // MilvusCollection Milvus 集合名称
    MilvusCollection string
    
    // DefaultPageSize 默认分页大小
    DefaultPageSize int
    
    // MaxPageSize 最大分页大小
    MaxPageSize int
    
    // EmbeddingDim 向量维度
    EmbeddingDim int
}
```

### 默认配置

```go
func DefaultConfig() *Config {
    return &Config{
        CacheTTL:         time.Hour,      // 缓存 1 小时
        MilvusCollection: "item_embeddings",
        DefaultPageSize:  20,
        MaxPageSize:      100,
        EmbeddingDim:     256,
    }
}
```

---

## 缓存策略

### 多级缓存架构

```
┌─────────────────────────────────────────────────────────────┐
│                        请求                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    L1: 本地缓存 (LocalCache)                 │
│  - 存储热点物品                                              │
│  - TTL: 5 分钟                                              │
│  - 容量: 10,000 条                                          │
└─────────────────────────────────────────────────────────────┘
                              │ 未命中
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    L2: Redis 缓存                            │
│  - 分布式缓存                                                │
│  - TTL: 1 小时                                              │
│  - 支持集群模式                                              │
└─────────────────────────────────────────────────────────────┘
                              │ 未命中
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    L3: PostgreSQL 数据库                     │
│  - 持久化存储                                                │
│  - 数据回填缓存                                              │
└─────────────────────────────────────────────────────────────┘
```

### 缓存键设计

| 键模式              | 说明           | TTL     |
|--------------------|----------------|---------|
| `item:{item_id}`   | 物品信息缓存    | 1 小时  |
| `item:embedding:{item_id}` | 物品向量缓存 | 4 小时  |

### 缓存失效策略

- **主动失效**：物品更新/删除时清除缓存
- **被动过期**：TTL 到期自动失效
- **LRU 淘汰**：本地缓存满时淘汰最少使用的条目

---

## 向量搜索

### Milvus 集成

物品服务使用 Milvus 进行向量相似度搜索，支持基于内容的相似物品推荐。

#### 集合 Schema

```go
schema := &entity.Schema{
    CollectionName: "item_embeddings",
    Fields: []*entity.Field{
        {Name: "item_id", DataType: entity.FieldTypeVarChar, PrimaryKey: true},
        {Name: "embedding", DataType: entity.FieldTypeFloatVector, Dim: 256},
        {Name: "semantic_l1", DataType: entity.FieldTypeInt32},
        {Name: "semantic_l2", DataType: entity.FieldTypeInt32},
        {Name: "semantic_l3", DataType: entity.FieldTypeInt32},
    },
}
```

#### 索引配置

- **索引类型**: HNSW
- **度量方式**: 内积 (IP)
- **参数**: M=16, efConstruction=200

#### 搜索流程

```
1. 获取目标物品的 embedding 向量
2. 调用 Milvus 进行 ANN (近似最近邻) 搜索
3. 过滤掉源物品自身
4. 获取相似物品详情
5. 返回带相似度分数的物品列表
```

---

## 单元测试

### 测试文件结构

```
recommend-system/
├── internal/service/item/
│   ├── service.go          # 服务实现
│   └── service_test.go     # 服务单元测试
└── api/item/v1/
    ├── handler.go          # API 处理器
    └── handler_test.go     # API 单元测试
```

### 运行测试

```bash
# 运行所有测试
go test ./internal/service/item/... -v

# 运行 API 测试
go test ./api/item/v1/... -v

# 运行测试并生成覆盖率报告
go test ./... -coverprofile=coverage.out
go tool cover -html=coverage.out
```

### Mock 策略

测试使用 Mock 实现来模拟外部依赖：

```go
// mockItemRepository 模拟物品仓储
type mockItemRepository struct {
    items map[string]*interfaces.Item
    // ...
}

// mockCache 模拟缓存
type mockCache struct {
    data map[string]interface{}
    // ...
}
```

### 基准测试

```bash
# 运行基准测试
go test ./internal/service/item/... -bench=. -benchmem

# 示例输出
BenchmarkGetItem-8          500000    2345 ns/op    256 B/op    4 allocs/op
BenchmarkCreateItem-8       200000    6789 ns/op    512 B/op    8 allocs/op
BenchmarkBatchGetItems-8    100000   12345 ns/op   1024 B/op   16 allocs/op
```

---

## 常见问题

### Q1: 物品 ID 如何生成？

物品 ID 使用以下规则生成：
- 格式：`{prefix}_{timestamp}_{random}`
- 前缀映射：
  - movie → `mov_`
  - product → `prd_`
  - article → `art_`
  - video → `vid_`
  - 其他 → `itm_`

### Q2: 如何处理 Milvus 不可用的情况？

当 Milvus 不可用时：
- 服务仍可正常启动
- 相似物品搜索 API 返回 503 错误
- 其他 CRUD 操作不受影响

### Q3: 缓存穿透如何处理？

- 对于不存在的物品，缓存空值（短 TTL）
- 使用布隆过滤器快速判断物品是否存在

### Q4: 如何扩展支持新的物品类型？

1. 在 `generateItemID` 方法中添加新的类型前缀
2. 更新 `CreateItemRequest` 的类型验证
3. 根据需要扩展 `Metadata` 字段

---

## 开发指南

### 添加新功能

1. **接口定义**：首先在 `interfaces.go` 中定义接口
2. **服务实现**：在 `service.go` 中实现业务逻辑
3. **API 处理器**：在 `handler.go` 中添加 HTTP 端点
4. **单元测试**：编写对应的测试用例
5. **文档更新**：更新本 README 文档

### 代码规范

- 遵循 Go 官方代码规范
- 使用有意义的变量名和函数名
- 添加详细的中文注释
- 错误处理使用统一的错误定义

### 日志规范

```go
// 使用结构化日志
logger.Info("item created",
    zap.String("item_id", item.ID),
    zap.String("type", item.Type),
    zap.Int64("latency_ms", timer.Elapsed()),
)

// 日志级别
// Debug: 调试信息
// Info: 正常操作日志
// Warn: 警告（可恢复错误）
// Error: 错误（需要关注）
```

### 性能优化建议

1. **批量操作**：使用 `BatchGetItems` 减少网络往返
2. **缓存预热**：热点物品在服务启动时预加载
3. **连接池**：合理配置数据库和 Redis 连接池
4. **异步处理**：非关键路径操作使用 goroutine 异步执行

---

## 相关链接

- [接口定义文档](../../interfaces/interfaces.go)
- [架构设计文档](../../../../docs/生成式推荐系统架构设计.md)
- [项目 README](../../../../README.md)

---

**维护者**: Person B  
**更新时间**: 2026-01-04  
**版本**: v1.0.0

