# Feature Service（特征服务）

## 概述

特征服务是生成式推荐系统的核心组件，负责实时特征提取、特征缓存存储和特征向量化。该服务实现了 `interfaces.FeatureService` 接口，为推荐系统的 UGT（Unified Generative Transformer）模型提供结构化的特征输入。

## 目录结构

```
recommend-system/internal/service/feature/
├── types.go           # 类型定义（特征结构、错误类型、常量）
├── extractor.go       # 特征提取器（从原始数据提取特征）
├── store.go           # 特征存储（缓存管理）
├── service.go         # 特征服务主体（接口实现）
├── types_test.go      # 类型定义单元测试
├── extractor_test.go  # 特征提取器单元测试
├── store_test.go      # 特征存储单元测试
├── service_test.go    # 特征服务单元测试
└── README.md          # 本文档
```

## 核心组件

### 1. 类型定义 (types.go)

#### 特征类型枚举

```go
const (
    FeatureTypeUser    FeatureType = "user"    // 用户特征
    FeatureTypeItem    FeatureType = "item"    // 物品特征
    FeatureTypeCross   FeatureType = "cross"   // 交叉特征
    FeatureTypeContext FeatureType = "context" // 上下文特征
)
```

#### 核心特征结构

| 结构体 | 说明 |
|--------|------|
| `DemographicFeatures` | 人口统计特征（年龄、性别、地区、设备） |
| `BehaviorFeatures` | 行为特征（浏览、点击、购买等） |
| `PreferenceFeatures` | 偏好特征（类目、标签、价格区间） |
| `ContentFeatures` | 内容特征（类目、标签、时长等） |
| `StatisticFeatures` | 统计特征（浏览量、点击率等） |
| `CrossFeatures` | 交叉特征（用户-物品交互） |
| `ContextFeatures` | 上下文特征（时间、设备、场景） |

#### Token 常量

```go
const (
    TokenTypeUser    = 0  // 用户 Token
    TokenTypeItem    = 1  // 物品 Token
    TokenTypeAction  = 2  // 行为 Token
    TokenTypeContext = 3  // 上下文 Token
)

const (
    TokenIDCLS int64 = 1  // [CLS] Token
    TokenIDSEP int64 = 2  // [SEP] Token
    TokenIDPAD int64 = 0  // [PAD] Token
)
```

### 2. 特征提取器 (extractor.go)

#### 创建实例

```go
extractor := NewFeatureExtractor(userRepo, itemRepo)
```

#### 主要方法

| 方法 | 说明 |
|------|------|
| `ExtractUserFeatures(ctx, userID)` | 提取用户特征 |
| `ExtractItemFeatures(ctx, itemID)` | 提取物品特征 |
| `ExtractCrossFeatures(ctx, userID, itemID)` | 提取交叉特征 |
| `ExtractContextFeatures(ctx, req)` | 提取上下文特征 |
| `BatchExtractUserFeatures(ctx, userIDs)` | 批量提取用户特征 |
| `BatchExtractItemFeatures(ctx, itemIDs)` | 批量提取物品特征 |

#### 特征提取流程

```
用户特征提取:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 获取用户信息 │ --> │ 获取行为记录 │ --> │  特征计算   │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
              ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
              │ 人口统计特征 │           │  行为特征   │           │  偏好特征   │
              └─────────────┘           └─────────────┘           └─────────────┘
```

### 3. 特征存储 (store.go)

#### 创建实例

```go
// 方式一：直接指定 TTL
store := NewFeatureStore(cache, 30*time.Minute, 60*time.Minute)

// 方式二：使用配置
config := &FeatureCacheConfig{
    UserFeatureTTL: 30 * time.Minute,
    ItemFeatureTTL: 60 * time.Minute,
}
store := NewFeatureStoreWithConfig(cache, config)
```

#### 缓存键格式

| 类型 | 键格式 |
|------|--------|
| 用户特征 | `feature:user:{userID}` |
| 物品特征 | `feature:item:{itemID}` |
| 交叉特征 | `feature:cross:{userID}:{itemID}` |

#### 主要方法

| 方法 | 说明 |
|------|------|
| `SaveUserFeatures(ctx, features)` | 保存用户特征 |
| `GetUserFeatures(ctx, userID)` | 获取用户特征 |
| `SaveItemFeatures(ctx, features)` | 保存物品特征 |
| `GetItemFeatures(ctx, itemID)` | 获取物品特征 |
| `BatchGetUserFeatures(ctx, userIDs)` | 批量获取用户特征 |
| `BatchGetItemFeatures(ctx, itemIDs)` | 批量获取物品特征 |
| `InvalidateUserFeatures(ctx, userID)` | 使用户特征失效 |
| `InvalidateItemFeatures(ctx, itemID)` | 使物品特征失效 |

#### 类型转换工具

```go
// 内部类型 -> 接口类型
interfaceFeatures := ConvertToInterfaceUserFeatures(internalFeatures)
interfaceFeatures := ConvertToInterfaceItemFeatures(internalFeatures)

// 接口类型 -> 内部类型
internalFeatures := ConvertFromInterfaceUserFeatures(interfaceFeatures)
internalFeatures := ConvertFromInterfaceItemFeatures(interfaceFeatures)
```

### 4. 特征服务 (service.go)

#### 创建实例

```go
// 方式一：使用默认配置
service := NewService(userRepo, itemRepo, cache)

// 方式二：使用自定义配置
config := &FeatureCacheConfig{
    UserFeatureTTL: 15 * time.Minute,
    ItemFeatureTTL: 45 * time.Minute,
}
service := NewServiceWithConfig(userRepo, itemRepo, cache, config)
```

#### 接口方法实现

服务实现了 `interfaces.FeatureService` 接口：

```go
type FeatureService interface {
    GetUserFeatures(ctx context.Context, userID string) (*UserFeatures, error)
    GetItemFeatures(ctx context.Context, itemID string) (*ItemFeatures, error)
    GetFeatureVector(ctx context.Context, req *FeatureVectorRequest) (*FeatureVector, error)
    BatchGetFeatureVectors(ctx context.Context, reqs []*FeatureVectorRequest) ([]*FeatureVector, error)
    RefreshUserFeatures(ctx context.Context, userID string) error
    RefreshItemFeatures(ctx context.Context, itemID string) error
}
```

#### 缓存策略

```
特征获取流程（缓存优先）:

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  查询缓存   │ --> │ 缓存命中?   │ --> │  返回特征   │
└─────────────┘     └─────────────┘     └─────────────┘
                          │ 未命中
                          ▼
                    ┌─────────────┐
                    │  提取特征   │
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │ 异步写缓存  │
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  返回特征   │
                    └─────────────┘
```

#### Token 序列化

特征向量会被序列化为 Token 序列，用于 UGT 模型输入：

```
Token 序列格式:
[CLS] [用户Token...] [物品Token...] [上下文Token...] [SEP]

示例:
[1] [1001] [1100] [100] [200] [300] [2001] [2]
 │    │      │     │     │     │     │      │
CLS  年龄   性别   L1   L2   L3   时段   SEP
```

#### 年龄分桶规则

| 年龄范围 | Token ID |
|----------|----------|
| 0-17 | 1000 |
| 18-25 | 1001 |
| 26-35 | 1002 |
| 36-45 | 1003 |
| 46-55 | 1004 |
| 56+ | 1005 |

#### 时间段分桶规则

| 时段 | 小时范围 | Token ID |
|------|----------|----------|
| night | 0-6 | 2000 |
| morning | 6-12 | 2001 |
| afternoon | 12-18 | 2002 |
| evening | 18-24 | 2003 |

## 使用示例

### 基础使用

```go
package main

import (
    "context"
    "time"
    
    "recommend-system/internal/service/feature"
    "recommend-system/internal/interfaces"
)

func main() {
    // 初始化依赖
    userRepo := // ... 用户仓库实现
    itemRepo := // ... 物品仓库实现
    cache := // ... 缓存实现
    
    // 创建服务
    svc := feature.NewService(userRepo, itemRepo, cache)
    ctx := context.Background()
    
    // 获取用户特征
    userFeatures, err := svc.GetUserFeatures(ctx, "user123")
    if err != nil {
        // 处理错误
    }
    
    // 获取物品特征
    itemFeatures, err := svc.GetItemFeatures(ctx, "item456")
    if err != nil {
        // 处理错误
    }
    
    // 获取完整特征向量
    req := &interfaces.FeatureVectorRequest{
        UserID: "user123",
        ItemID: "item456",
        Context: map[string]string{
            "device":       "mobile",
            "os":           "iOS",
            "page_context": "home",
        },
    }
    vector, err := svc.GetFeatureVector(ctx, req)
    if err != nil {
        // 处理错误
    }
    
    // 使用 vector.TokenIDs 进行模型推理
    fmt.Println("Token IDs:", vector.TokenIDs)
}
```

### 批量操作

```go
// 批量获取特征向量
reqs := []*interfaces.FeatureVectorRequest{
    {UserID: "user1", ItemID: "item1"},
    {UserID: "user2", ItemID: "item2"},
    {UserID: "user3", ItemID: "item3"},
}
vectors, err := svc.BatchGetFeatureVectors(ctx, reqs)

// 批量刷新特征
userIDs := []string{"user1", "user2", "user3"}
svc.BatchRefreshUserFeatures(ctx, userIDs)

itemIDs := []string{"item1", "item2", "item3"}
svc.BatchRefreshItemFeatures(ctx, itemIDs)
```

### 使用内部详细特征

```go
// 获取详细的用户特征
internalFeatures, err := svc.GetInternalUserFeatures(ctx, "user123")
if err == nil {
    fmt.Printf("年龄: %d\n", internalFeatures.Demographics.Age)
    fmt.Printf("总浏览: %d\n", internalFeatures.Behavior.TotalViews)
    fmt.Printf("偏好类目: %v\n", internalFeatures.Preferences.TopCategories)
}

// 获取交叉特征
crossFeatures, err := svc.GetCrossFeatures(ctx, "user123", "item456")
if err == nil {
    fmt.Printf("交互次数: %d\n", crossFeatures.Interactions)
    fmt.Printf("最后行为: %s\n", crossFeatures.LastAction)
}
```

## 依赖接口

该服务依赖以下接口（定义在 `internal/interfaces/interfaces.go`）：

### UserRepository
```go
type UserRepository interface {
    GetByID(ctx context.Context, userID string) (*User, error)
    GetByIDs(ctx context.Context, userIDs []string) ([]*User, error)
    GetBehaviors(ctx context.Context, userID string, limit int) ([]*UserBehavior, error)
    GetUserItemInteractions(ctx context.Context, userID, itemID string) ([]*UserBehavior, error)
    // ...
}
```

### ItemRepository
```go
type ItemRepository interface {
    GetByID(ctx context.Context, itemID string) (*Item, error)
    GetByIDs(ctx context.Context, itemIDs []string) ([]*Item, error)
    GetStats(ctx context.Context, itemID string) (*ItemStats, error)
    // ...
}
```

### Cache
```go
type Cache interface {
    Get(ctx context.Context, key string, value interface{}) error
    Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error
    Delete(ctx context.Context, key string) error
    Exists(ctx context.Context, key string) (bool, error)
    MGet(ctx context.Context, keys []string) ([]interface{}, error)
    MSet(ctx context.Context, kvs map[string]interface{}, ttl time.Duration) error
}
```

## 运行测试

```bash
# 运行所有测试
cd recommend-system
go test ./internal/service/feature/...

# 运行带覆盖率的测试
go test ./internal/service/feature/... -cover

# 运行详细输出的测试
go test ./internal/service/feature/... -v

# 运行特定测试
go test ./internal/service/feature/... -run TestServiceGetUserFeatures
```

## 性能优化建议

### 1. 缓存策略
- 用户特征建议 30 分钟 TTL
- 物品特征建议 60 分钟 TTL
- 高频访问物品可适当延长 TTL

### 2. 批量操作
- 使用 `BatchGetFeatureVectors` 减少网络往返
- 批量刷新时使用 goroutine 并发

### 3. 特征预计算
- 对于热门用户/物品，可定时预计算特征
- 使用消息队列监听行为事件，实时更新特征

## 错误处理

服务定义了以下错误类型：

| 错误 | 说明 |
|------|------|
| `ErrUserNotFound` | 用户不存在 |
| `ErrItemNotFound` | 物品不存在 |
| `ErrFeatureNotFound` | 特征不存在（缓存未命中） |
| `ErrCacheError` | 缓存操作失败 |
| `ErrExtractionError` | 特征提取失败 |

## 扩展开发

### 添加新的特征类型

1. 在 `types.go` 中定义特征结构
2. 在 `extractor.go` 中添加提取逻辑
3. 在 `store.go` 中添加存储逻辑
4. 在 `service.go` 中集成到服务
5. 更新 Token 序列化逻辑

### 添加新的 Token 映射

1. 在 `types.go` 中定义新的 Token 基础 ID
2. 在 `service.go` 中添加 Token 映射函数
3. 在 `serializeToTokens` 中调用新的映射函数

## 参考资料

- [生成式推荐系统架构设计](../../docs/生成式推荐系统架构设计.md)
- [接口定义](../interfaces/interfaces.go)
- [HSTU 论文](https://arxiv.org/abs/xxx)
- [OneRec 论文](https://arxiv.org/abs/xxx)

