# Person C: Feature Service（特征服务）

## 你的角色
你是一名 Go 后端工程师，负责实现生成式推荐系统的 **特征服务** 模块。

## 背景知识

特征服务是推荐系统的核心组件，负责：
- 实时特征提取和计算
- 特征存储和查询
- 用户-物品交叉特征生成
- 特征向量化（为 UGT 模型准备输入）

## 你的任务

实现以下模块：

```
recommend-system/
└── internal/service/feature/
    ├── service.go           # 特征服务主体
    ├── extractor.go         # 特征提取器
    ├── store.go             # 特征存储
    └── types.go             # 类型定义
```

---

## 1. internal/service/feature/types.go

```go
package feature

import "time"

// FeatureType 特征类型
type FeatureType string

const (
    FeatureTypeUser    FeatureType = "user"
    FeatureTypeItem    FeatureType = "item"
    FeatureTypeCross   FeatureType = "cross"
    FeatureTypeContext FeatureType = "context"
)

// UserFeatures 用户特征
type UserFeatures struct {
    UserID          string              `json:"user_id"`
    Demographics    DemographicFeatures `json:"demographics"`
    Behavior        BehaviorFeatures    `json:"behavior"`
    Preferences     PreferenceFeatures  `json:"preferences"`
    Embedding       []float32           `json:"embedding,omitempty"`
    LastUpdated     time.Time           `json:"last_updated"`
}

// DemographicFeatures 人口统计特征
type DemographicFeatures struct {
    Age      int    `json:"age"`
    Gender   string `json:"gender"`
    Location string `json:"location"`
    Device   string `json:"device"`
}

// BehaviorFeatures 行为特征
type BehaviorFeatures struct {
    TotalViews       int64   `json:"total_views"`
    TotalClicks      int64   `json:"total_clicks"`
    TotalPurchases   int64   `json:"total_purchases"`
    AvgSessionTime   float64 `json:"avg_session_time"`
    ActiveDays       int     `json:"active_days"`
    LastActiveHour   int     `json:"last_active_hour"`
    PreferredHours   []int   `json:"preferred_hours"`
}

// PreferenceFeatures 偏好特征
type PreferenceFeatures struct {
    TopCategories  []CategoryScore `json:"top_categories"`
    TopTags        []TagScore      `json:"top_tags"`
    PriceRange     [2]float64      `json:"price_range"`
    ContentLength  string          `json:"content_length"` // short, medium, long
}

type CategoryScore struct {
    Category string  `json:"category"`
    Score    float64 `json:"score"`
}

type TagScore struct {
    Tag   string  `json:"tag"`
    Score float64 `json:"score"`
}

// ItemFeatures 物品特征
type ItemFeatures struct {
    ItemID      string             `json:"item_id"`
    Type        string             `json:"type"`
    Content     ContentFeatures    `json:"content"`
    Statistics  StatisticFeatures  `json:"statistics"`
    Embedding   []float32          `json:"embedding,omitempty"`
    SemanticID  [3]int             `json:"semantic_id"` // [L1, L2, L3]
    LastUpdated time.Time          `json:"last_updated"`
}

// ContentFeatures 内容特征
type ContentFeatures struct {
    Category    string   `json:"category"`
    Tags        []string `json:"tags"`
    Price       float64  `json:"price,omitempty"`
    Duration    int      `json:"duration,omitempty"`   // 视频/电影时长（秒）
    WordCount   int      `json:"word_count,omitempty"` // 文章字数
    ReleaseDate string   `json:"release_date,omitempty"`
}

// StatisticFeatures 统计特征
type StatisticFeatures struct {
    ViewCount     int64   `json:"view_count"`
    ClickCount    int64   `json:"click_count"`
    LikeCount     int64   `json:"like_count"`
    ShareCount    int64   `json:"share_count"`
    CommentCount  int64   `json:"comment_count"`
    AvgRating     float64 `json:"avg_rating"`
    CTR           float64 `json:"ctr"`  // 点击率
    CVR           float64 `json:"cvr"`  // 转化率
}

// CrossFeatures 交叉特征
type CrossFeatures struct {
    UserID      string    `json:"user_id"`
    ItemID      string    `json:"item_id"`
    Interactions int      `json:"interactions"`      // 交互次数
    LastAction   string   `json:"last_action"`       // 最后交互类型
    LastTime     time.Time `json:"last_time"`        // 最后交互时间
    Similarity   float64  `json:"similarity"`        // 用户-物品相似度
}

// ContextFeatures 上下文特征
type ContextFeatures struct {
    Timestamp   time.Time `json:"timestamp"`
    Hour        int       `json:"hour"`
    DayOfWeek   int       `json:"day_of_week"`
    IsWeekend   bool      `json:"is_weekend"`
    Device      string    `json:"device"`
    OS          string    `json:"os"`
    Location    string    `json:"location"`
    PageContext string    `json:"page_context"`  // home, search, detail
}

// FeatureVector 特征向量（模型输入）
type FeatureVector struct {
    UserFeatures    *UserFeatures    `json:"user_features"`
    ItemFeatures    *ItemFeatures    `json:"item_features,omitempty"`
    CrossFeatures   *CrossFeatures   `json:"cross_features,omitempty"`
    ContextFeatures *ContextFeatures `json:"context_features"`
    
    // 序列化后的 Token IDs（用于 UGT 模型）
    TokenIDs        []int64          `json:"token_ids,omitempty"`
    TokenTypes      []int            `json:"token_types,omitempty"`
    Positions       []int            `json:"positions,omitempty"`
}
```

---

## 2. internal/service/feature/extractor.go

```go
package feature

import (
    "context"
    "time"
    
    "recommend-system/internal/model"
    "recommend-system/internal/repository"
)

// FeatureExtractor 特征提取器
type FeatureExtractor struct {
    userRepo repository.UserRepo
    itemRepo repository.ItemRepo
}

// NewFeatureExtractor 创建特征提取器
func NewFeatureExtractor(userRepo repository.UserRepo, itemRepo repository.ItemRepo) *FeatureExtractor {
    return &FeatureExtractor{
        userRepo: userRepo,
        itemRepo: itemRepo,
    }
}

// ExtractUserFeatures 提取用户特征
func (e *FeatureExtractor) ExtractUserFeatures(ctx context.Context, userID string) (*UserFeatures, error) {
    // 获取用户基本信息
    user, err := e.userRepo.GetByID(ctx, userID)
    if err != nil {
        return nil, err
    }
    
    // 获取用户行为
    behaviors, err := e.userRepo.GetBehaviors(ctx, userID, 1000)
    if err != nil {
        return nil, err
    }
    
    // 提取人口统计特征
    demographics := DemographicFeatures{
        Age:    user.Age,
        Gender: user.Gender,
    }
    
    // 提取行为特征
    behaviorFeatures := e.extractBehaviorFeatures(behaviors)
    
    // 提取偏好特征
    preferenceFeatures := e.extractPreferenceFeatures(behaviors)
    
    return &UserFeatures{
        UserID:       userID,
        Demographics: demographics,
        Behavior:     behaviorFeatures,
        Preferences:  preferenceFeatures,
        LastUpdated:  time.Now(),
    }, nil
}

// ExtractItemFeatures 提取物品特征
func (e *FeatureExtractor) ExtractItemFeatures(ctx context.Context, itemID string) (*ItemFeatures, error) {
    // 获取物品基本信息
    item, err := e.itemRepo.GetByID(ctx, itemID)
    if err != nil {
        return nil, err
    }
    
    // 获取物品统计
    stats, err := e.itemRepo.GetStats(ctx, itemID)
    if err != nil {
        stats = &model.ItemStats{} // 使用默认值
    }
    
    // 提取内容特征
    contentFeatures := ContentFeatures{
        Category: item.Category,
        Tags:     item.Tags,
    }
    
    // 提取统计特征
    statisticFeatures := StatisticFeatures{
        ViewCount:    stats.ViewCount,
        ClickCount:   stats.ClickCount,
        LikeCount:    stats.LikeCount,
        ShareCount:   stats.ShareCount,
        AvgRating:    stats.AvgRating,
        CTR:          e.calculateCTR(stats),
    }
    
    return &ItemFeatures{
        ItemID:      itemID,
        Type:        item.Type,
        Content:     contentFeatures,
        Statistics:  statisticFeatures,
        LastUpdated: time.Now(),
    }, nil
}

// ExtractCrossFeatures 提取交叉特征
func (e *FeatureExtractor) ExtractCrossFeatures(ctx context.Context, userID, itemID string) (*CrossFeatures, error) {
    // 查询用户对该物品的历史交互
    interactions, err := e.userRepo.GetUserItemInteractions(ctx, userID, itemID)
    if err != nil {
        return nil, err
    }
    
    cross := &CrossFeatures{
        UserID:       userID,
        ItemID:       itemID,
        Interactions: len(interactions),
    }
    
    if len(interactions) > 0 {
        lastInteraction := interactions[0]
        cross.LastAction = lastInteraction.Action
        cross.LastTime = lastInteraction.Timestamp
    }
    
    return cross, nil
}

// ExtractContextFeatures 提取上下文特征
func (e *FeatureExtractor) ExtractContextFeatures(ctx context.Context, req *ContextRequest) *ContextFeatures {
    now := time.Now()
    
    return &ContextFeatures{
        Timestamp:   now,
        Hour:        now.Hour(),
        DayOfWeek:   int(now.Weekday()),
        IsWeekend:   now.Weekday() == time.Saturday || now.Weekday() == time.Sunday,
        Device:      req.Device,
        OS:          req.OS,
        Location:    req.Location,
        PageContext: req.PageContext,
    }
}

// extractBehaviorFeatures 从行为列表提取行为特征
func (e *FeatureExtractor) extractBehaviorFeatures(behaviors []*model.UserBehavior) BehaviorFeatures {
    features := BehaviorFeatures{
        PreferredHours: make([]int, 0),
    }
    
    hourCounts := make(map[int]int)
    activeDays := make(map[string]bool)
    
    for _, b := range behaviors {
        switch b.Action {
        case "view":
            features.TotalViews++
        case "click":
            features.TotalClicks++
        case "buy", "purchase":
            features.TotalPurchases++
        }
        
        hour := b.Timestamp.Hour()
        hourCounts[hour]++
        
        day := b.Timestamp.Format("2006-01-02")
        activeDays[day] = true
    }
    
    features.ActiveDays = len(activeDays)
    
    // 找出偏好时段（出现次数 > 平均值的时段）
    if len(behaviors) > 0 {
        avgCount := float64(len(behaviors)) / 24
        for hour, count := range hourCounts {
            if float64(count) > avgCount {
                features.PreferredHours = append(features.PreferredHours, hour)
            }
        }
    }
    
    return features
}

// extractPreferenceFeatures 从行为列表提取偏好特征
func (e *FeatureExtractor) extractPreferenceFeatures(behaviors []*model.UserBehavior) PreferenceFeatures {
    features := PreferenceFeatures{
        TopCategories: make([]CategoryScore, 0),
        TopTags:       make([]TagScore, 0),
    }
    
    // 统计类别偏好
    // 实现细节...
    
    return features
}

// calculateCTR 计算点击率
func (e *FeatureExtractor) calculateCTR(stats *model.ItemStats) float64 {
    if stats.ViewCount == 0 {
        return 0
    }
    return float64(stats.ClickCount) / float64(stats.ViewCount)
}

// ContextRequest 上下文请求
type ContextRequest struct {
    Device      string `json:"device"`
    OS          string `json:"os"`
    Location    string `json:"location"`
    PageContext string `json:"page_context"`
}
```

---

## 3. internal/service/feature/store.go

```go
package feature

import (
    "context"
    "encoding/json"
    "fmt"
    "time"
    
    "recommend-system/internal/cache"
)

// FeatureStore 特征存储
type FeatureStore struct {
    cache cache.Cache
    ttl   time.Duration
}

// NewFeatureStore 创建特征存储
func NewFeatureStore(cache cache.Cache, ttl time.Duration) *FeatureStore {
    return &FeatureStore{
        cache: cache,
        ttl:   ttl,
    }
}

// SaveUserFeatures 保存用户特征
func (s *FeatureStore) SaveUserFeatures(ctx context.Context, features *UserFeatures) error {
    key := fmt.Sprintf("feature:user:%s", features.UserID)
    return s.cache.Set(ctx, key, features, s.ttl)
}

// GetUserFeatures 获取用户特征
func (s *FeatureStore) GetUserFeatures(ctx context.Context, userID string) (*UserFeatures, error) {
    key := fmt.Sprintf("feature:user:%s", userID)
    var features UserFeatures
    if err := s.cache.Get(ctx, key, &features); err != nil {
        return nil, err
    }
    return &features, nil
}

// SaveItemFeatures 保存物品特征
func (s *FeatureStore) SaveItemFeatures(ctx context.Context, features *ItemFeatures) error {
    key := fmt.Sprintf("feature:item:%s", features.ItemID)
    return s.cache.Set(ctx, key, features, s.ttl)
}

// GetItemFeatures 获取物品特征
func (s *FeatureStore) GetItemFeatures(ctx context.Context, itemID string) (*ItemFeatures, error) {
    key := fmt.Sprintf("feature:item:%s", itemID)
    var features ItemFeatures
    if err := s.cache.Get(ctx, key, &features); err != nil {
        return nil, err
    }
    return &features, nil
}

// BatchGetUserFeatures 批量获取用户特征
func (s *FeatureStore) BatchGetUserFeatures(ctx context.Context, userIDs []string) (map[string]*UserFeatures, error) {
    result := make(map[string]*UserFeatures)
    
    for _, userID := range userIDs {
        features, err := s.GetUserFeatures(ctx, userID)
        if err == nil {
            result[userID] = features
        }
    }
    
    return result, nil
}

// BatchGetItemFeatures 批量获取物品特征
func (s *FeatureStore) BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]*ItemFeatures, error) {
    result := make(map[string]*ItemFeatures)
    
    for _, itemID := range itemIDs {
        features, err := s.GetItemFeatures(ctx, itemID)
        if err == nil {
            result[itemID] = features
        }
    }
    
    return result, nil
}

// InvalidateUserFeatures 使用户特征失效
func (s *FeatureStore) InvalidateUserFeatures(ctx context.Context, userID string) error {
    key := fmt.Sprintf("feature:user:%s", userID)
    return s.cache.Delete(ctx, key)
}

// InvalidateItemFeatures 使物品特征失效
func (s *FeatureStore) InvalidateItemFeatures(ctx context.Context, itemID string) error {
    key := fmt.Sprintf("feature:item:%s", itemID)
    return s.cache.Delete(ctx, key)
}
```

---

## 4. internal/service/feature/service.go

```go
package feature

import (
    "context"
    "time"
    
    "recommend-system/internal/repository"
    "recommend-system/internal/cache"
    "recommend-system/pkg/logger"
)

// FeatureService 特征服务
type FeatureService struct {
    extractor *FeatureExtractor
    store     *FeatureStore
    logger    *logger.Logger
}

// NewFeatureService 创建特征服务
func NewFeatureService(
    userRepo repository.UserRepo,
    itemRepo repository.ItemRepo,
    cache cache.Cache,
    logger *logger.Logger,
) *FeatureService {
    return &FeatureService{
        extractor: NewFeatureExtractor(userRepo, itemRepo),
        store:     NewFeatureStore(cache, 30*time.Minute),
        logger:    logger,
    }
}

// GetUserFeatures 获取用户特征（缓存优先）
func (s *FeatureService) GetUserFeatures(ctx context.Context, userID string) (*UserFeatures, error) {
    // 先查缓存
    features, err := s.store.GetUserFeatures(ctx, userID)
    if err == nil {
        return features, nil
    }
    
    // 提取特征
    features, err = s.extractor.ExtractUserFeatures(ctx, userID)
    if err != nil {
        return nil, err
    }
    
    // 保存到缓存
    _ = s.store.SaveUserFeatures(ctx, features)
    
    return features, nil
}

// GetItemFeatures 获取物品特征（缓存优先）
func (s *FeatureService) GetItemFeatures(ctx context.Context, itemID string) (*ItemFeatures, error) {
    // 先查缓存
    features, err := s.store.GetItemFeatures(ctx, itemID)
    if err == nil {
        return features, nil
    }
    
    // 提取特征
    features, err = s.extractor.ExtractItemFeatures(ctx, itemID)
    if err != nil {
        return nil, err
    }
    
    // 保存到缓存
    _ = s.store.SaveItemFeatures(ctx, features)
    
    return features, nil
}

// GetFeatureVector 获取完整特征向量（用于模型推理）
func (s *FeatureService) GetFeatureVector(ctx context.Context, req *FeatureVectorRequest) (*FeatureVector, error) {
    vector := &FeatureVector{}
    
    // 用户特征（必需）
    userFeatures, err := s.GetUserFeatures(ctx, req.UserID)
    if err != nil {
        return nil, err
    }
    vector.UserFeatures = userFeatures
    
    // 物品特征（可选，用于排序场景）
    if req.ItemID != "" {
        itemFeatures, err := s.GetItemFeatures(ctx, req.ItemID)
        if err == nil {
            vector.ItemFeatures = itemFeatures
        }
        
        // 交叉特征
        crossFeatures, err := s.extractor.ExtractCrossFeatures(ctx, req.UserID, req.ItemID)
        if err == nil {
            vector.CrossFeatures = crossFeatures
        }
    }
    
    // 上下文特征
    vector.ContextFeatures = s.extractor.ExtractContextFeatures(ctx, &req.Context)
    
    // 序列化为 Token IDs（用于 UGT 模型）
    s.serializeToTokens(vector)
    
    return vector, nil
}

// BatchGetFeatureVectors 批量获取特征向量
func (s *FeatureService) BatchGetFeatureVectors(ctx context.Context, reqs []*FeatureVectorRequest) ([]*FeatureVector, error) {
    vectors := make([]*FeatureVector, 0, len(reqs))
    
    for _, req := range reqs {
        vector, err := s.GetFeatureVector(ctx, req)
        if err != nil {
            s.logger.Warn("Failed to get feature vector", "user_id", req.UserID, "error", err)
            continue
        }
        vectors = append(vectors, vector)
    }
    
    return vectors, nil
}

// RefreshUserFeatures 刷新用户特征
func (s *FeatureService) RefreshUserFeatures(ctx context.Context, userID string) error {
    // 使缓存失效
    _ = s.store.InvalidateUserFeatures(ctx, userID)
    
    // 重新提取并缓存
    _, err := s.GetUserFeatures(ctx, userID)
    return err
}

// RefreshItemFeatures 刷新物品特征
func (s *FeatureService) RefreshItemFeatures(ctx context.Context, itemID string) error {
    // 使缓存失效
    _ = s.store.InvalidateItemFeatures(ctx, itemID)
    
    // 重新提取并缓存
    _, err := s.GetItemFeatures(ctx, itemID)
    return err
}

// serializeToTokens 将特征向量序列化为 Token IDs
func (s *FeatureService) serializeToTokens(vector *FeatureVector) {
    // Token 类型: 0=USER, 1=ITEM, 2=ACTION, 3=CONTEXT
    tokenIDs := make([]int64, 0)
    tokenTypes := make([]int, 0)
    
    // [CLS] Token
    tokenIDs = append(tokenIDs, 1) // CLS token ID
    tokenTypes = append(tokenTypes, 3)
    
    // 用户特征 Tokens
    if vector.UserFeatures != nil {
        // 年龄分桶
        ageToken := s.getAgeToken(vector.UserFeatures.Demographics.Age)
        tokenIDs = append(tokenIDs, ageToken)
        tokenTypes = append(tokenTypes, 0)
        
        // 性别
        genderToken := s.getGenderToken(vector.UserFeatures.Demographics.Gender)
        tokenIDs = append(tokenIDs, genderToken)
        tokenTypes = append(tokenTypes, 0)
    }
    
    // 物品特征 Tokens
    if vector.ItemFeatures != nil && len(vector.ItemFeatures.SemanticID) == 3 {
        // Semantic ID L1, L2, L3
        tokenIDs = append(tokenIDs, int64(vector.ItemFeatures.SemanticID[0]))
        tokenTypes = append(tokenTypes, 1)
        tokenIDs = append(tokenIDs, int64(vector.ItemFeatures.SemanticID[1]))
        tokenTypes = append(tokenTypes, 1)
        tokenIDs = append(tokenIDs, int64(vector.ItemFeatures.SemanticID[2]))
        tokenTypes = append(tokenTypes, 1)
    }
    
    // 上下文 Tokens
    if vector.ContextFeatures != nil {
        hourToken := s.getHourToken(vector.ContextFeatures.Hour)
        tokenIDs = append(tokenIDs, hourToken)
        tokenTypes = append(tokenTypes, 3)
    }
    
    // [SEP] Token
    tokenIDs = append(tokenIDs, 2) // SEP token ID
    tokenTypes = append(tokenTypes, 3)
    
    // 位置编码
    positions := make([]int, len(tokenIDs))
    for i := range positions {
        positions[i] = i
    }
    
    vector.TokenIDs = tokenIDs
    vector.TokenTypes = tokenTypes
    vector.Positions = positions
}

// Token 映射辅助函数
func (s *FeatureService) getAgeToken(age int) int64 {
    // 年龄分桶: 0-17, 18-25, 26-35, 36-45, 46-55, 56+
    switch {
    case age < 18:
        return 1000
    case age < 26:
        return 1001
    case age < 36:
        return 1002
    case age < 46:
        return 1003
    case age < 56:
        return 1004
    default:
        return 1005
    }
}

func (s *FeatureService) getGenderToken(gender string) int64 {
    switch gender {
    case "male", "m":
        return 1100
    case "female", "f":
        return 1101
    default:
        return 1102
    }
}

func (s *FeatureService) getHourToken(hour int) int64 {
    // 时间分桶: night(0-6), morning(6-12), afternoon(12-18), evening(18-24)
    switch {
    case hour < 6:
        return 2000
    case hour < 12:
        return 2001
    case hour < 18:
        return 2002
    default:
        return 2003
    }
}

// FeatureVectorRequest 特征向量请求
type FeatureVectorRequest struct {
    UserID  string         `json:"user_id" binding:"required"`
    ItemID  string         `json:"item_id"`
    Context ContextRequest `json:"context"`
}
```

---

## 注意事项

1. **特征缓存**: 用户/物品特征使用 30 分钟缓存
2. **批量操作**: 支持批量获取以提高性能
3. **Token 化**: 特征需要序列化为 UGT 模型的输入格式
4. **实时性**: 上下文特征实时计算，用户/物品特征可缓存

## 输出要求

请输出完整的可运行代码，包含：
1. 所有 Go 文件
2. 详细的中文注释
3. 单元测试

