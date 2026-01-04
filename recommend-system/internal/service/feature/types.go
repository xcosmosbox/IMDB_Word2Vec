// Package feature 提供特征服务实现
// 特征服务是推荐系统的核心组件，负责实时特征提取、特征存储和特征向量化
package feature

import (
	"time"
)

// =============================================================================
// 特征类型常量定义
// =============================================================================

// FeatureType 特征类型枚举
type FeatureType string

const (
	// FeatureTypeUser 用户特征类型
	FeatureTypeUser FeatureType = "user"
	// FeatureTypeItem 物品特征类型
	FeatureTypeItem FeatureType = "item"
	// FeatureTypeCross 交叉特征类型
	FeatureTypeCross FeatureType = "cross"
	// FeatureTypeContext 上下文特征类型
	FeatureTypeContext FeatureType = "context"
)

// =============================================================================
// 人口统计特征
// =============================================================================

// DemographicFeatures 人口统计特征
type DemographicFeatures struct {
	Age      int    `json:"age"`       // 年龄
	Gender   string `json:"gender"`    // 性别: male, female, unknown
	Location string `json:"location"`  // 地理位置
	Device   string `json:"device"`    // 设备类型: mobile, desktop, tablet
}

// =============================================================================
// 行为特征
// =============================================================================

// BehaviorFeatures 用户行为特征
type BehaviorFeatures struct {
	TotalViews     int64   `json:"total_views"`      // 总浏览次数
	TotalClicks    int64   `json:"total_clicks"`     // 总点击次数
	TotalPurchases int64   `json:"total_purchases"`  // 总购买次数
	AvgSessionTime float64 `json:"avg_session_time"` // 平均会话时长（秒）
	ActiveDays     int     `json:"active_days"`      // 活跃天数
	LastActiveHour int     `json:"last_active_hour"` // 最后活跃时段（0-23）
	PreferredHours []int   `json:"preferred_hours"`  // 偏好时段列表
}

// =============================================================================
// 偏好特征
// =============================================================================

// PreferenceFeatures 用户偏好特征
type PreferenceFeatures struct {
	TopCategories []CategoryScore `json:"top_categories"` // 偏好类目列表
	TopTags       []TagScore      `json:"top_tags"`       // 偏好标签列表
	PriceRange    [2]float64      `json:"price_range"`    // 价格区间 [min, max]
	ContentLength string          `json:"content_length"` // 内容长度偏好: short, medium, long
}

// CategoryScore 类目偏好分数
type CategoryScore struct {
	Category string  `json:"category"` // 类目名称
	Score    float64 `json:"score"`    // 偏好分数
}

// TagScore 标签偏好分数
type TagScore struct {
	Tag   string  `json:"tag"`   // 标签名称
	Score float64 `json:"score"` // 偏好分数
}

// =============================================================================
// 内容特征
// =============================================================================

// ContentFeatures 物品内容特征
type ContentFeatures struct {
	Category    string   `json:"category"`               // 主类目
	SubCategory string   `json:"sub_category,omitempty"` // 子类目
	Tags        []string `json:"tags"`                   // 标签列表
	Price       float64  `json:"price,omitempty"`        // 价格
	Duration    int      `json:"duration,omitempty"`     // 时长（秒）
	WordCount   int      `json:"word_count,omitempty"`   // 字数
	ReleaseDate string   `json:"release_date,omitempty"` // 发布日期
}

// =============================================================================
// 统计特征
// =============================================================================

// StatisticFeatures 物品统计特征
type StatisticFeatures struct {
	ViewCount    int64   `json:"view_count"`    // 浏览次数
	ClickCount   int64   `json:"click_count"`   // 点击次数
	LikeCount    int64   `json:"like_count"`    // 点赞次数
	ShareCount   int64   `json:"share_count"`   // 分享次数
	CommentCount int64   `json:"comment_count"` // 评论次数
	AvgRating    float64 `json:"avg_rating"`    // 平均评分
	CTR          float64 `json:"ctr"`           // 点击率
	CVR          float64 `json:"cvr"`           // 转化率
}

// =============================================================================
// 交叉特征
// =============================================================================

// CrossFeatures 用户-物品交叉特征
type CrossFeatures struct {
	UserID       string    `json:"user_id"`      // 用户ID
	ItemID       string    `json:"item_id"`      // 物品ID
	Interactions int       `json:"interactions"` // 交互次数
	LastAction   string    `json:"last_action"`  // 最后交互类型
	LastTime     time.Time `json:"last_time"`    // 最后交互时间
	Similarity   float64   `json:"similarity"`   // 用户-物品相似度
}

// =============================================================================
// 上下文特征
// =============================================================================

// ContextFeatures 请求上下文特征
type ContextFeatures struct {
	Timestamp   time.Time `json:"timestamp"`    // 请求时间戳
	Hour        int       `json:"hour"`         // 小时（0-23）
	DayOfWeek   int       `json:"day_of_week"`  // 星期几（0-6）
	IsWeekend   bool      `json:"is_weekend"`   // 是否周末
	Device      string    `json:"device"`       // 设备类型
	OS          string    `json:"os"`           // 操作系统
	Location    string    `json:"location"`     // 地理位置
	PageContext string    `json:"page_context"` // 页面上下文: home, search, detail
}

// ContextRequest 上下文请求参数
type ContextRequest struct {
	Device      string `json:"device"`       // 设备类型
	OS          string `json:"os"`           // 操作系统
	Location    string `json:"location"`     // 地理位置
	PageContext string `json:"page_context"` // 页面上下文
}

// =============================================================================
// 内部特征数据结构（用于提取和存储）
// =============================================================================

// InternalUserFeatures 内部用户特征结构（包含详细特征）
type InternalUserFeatures struct {
	UserID       string              `json:"user_id"`
	Demographics DemographicFeatures `json:"demographics"`
	Behavior     BehaviorFeatures    `json:"behavior"`
	Preferences  PreferenceFeatures  `json:"preferences"`
	Embedding    []float32           `json:"embedding,omitempty"`
	LastUpdated  time.Time           `json:"last_updated"`
}

// InternalItemFeatures 内部物品特征结构（包含详细特征）
type InternalItemFeatures struct {
	ItemID      string            `json:"item_id"`
	Type        string            `json:"type"`
	Content     ContentFeatures   `json:"content"`
	Statistics  StatisticFeatures `json:"statistics"`
	Embedding   []float32         `json:"embedding,omitempty"`
	SemanticID  [3]int            `json:"semantic_id"` // [L1, L2, L3]
	LastUpdated time.Time         `json:"last_updated"`
}

// InternalFeatureVector 内部特征向量（包含完整特征）
type InternalFeatureVector struct {
	UserFeatures    *InternalUserFeatures `json:"user_features"`
	ItemFeatures    *InternalItemFeatures `json:"item_features,omitempty"`
	CrossFeatures   *CrossFeatures        `json:"cross_features,omitempty"`
	ContextFeatures *ContextFeatures      `json:"context_features"`

	// 序列化后的 Token IDs（用于 UGT 模型）
	TokenIDs   []int64 `json:"token_ids,omitempty"`
	TokenTypes []int   `json:"token_types,omitempty"`
	Positions  []int   `json:"positions,omitempty"`
}

// =============================================================================
// Token 类型常量
// =============================================================================

const (
	// TokenTypeUser 用户 Token 类型
	TokenTypeUser = 0
	// TokenTypeItem 物品 Token 类型
	TokenTypeItem = 1
	// TokenTypeAction 行为 Token 类型
	TokenTypeAction = 2
	// TokenTypeContext 上下文 Token 类型
	TokenTypeContext = 3
)

// =============================================================================
// 特殊 Token ID
// =============================================================================

const (
	// TokenIDCLS [CLS] Token ID
	TokenIDCLS int64 = 1
	// TokenIDSEP [SEP] Token ID
	TokenIDSEP int64 = 2
	// TokenIDPAD [PAD] Token ID
	TokenIDPAD int64 = 0
	// TokenIDUNK [UNK] Token ID
	TokenIDUNK int64 = 3
)

// =============================================================================
// 年龄分桶 Token 基础 ID
// =============================================================================

const (
	// AgeTokenBase 年龄 Token 基础 ID
	AgeTokenBase int64 = 1000
	// GenderTokenBase 性别 Token 基础 ID
	GenderTokenBase int64 = 1100
	// HourTokenBase 时间段 Token 基础 ID
	HourTokenBase int64 = 2000
)

// =============================================================================
// 缓存配置
// =============================================================================

// FeatureCacheConfig 特征缓存配置
type FeatureCacheConfig struct {
	UserFeatureTTL time.Duration `json:"user_feature_ttl"` // 用户特征缓存过期时间
	ItemFeatureTTL time.Duration `json:"item_feature_ttl"` // 物品特征缓存过期时间
}

// DefaultFeatureCacheConfig 返回默认缓存配置
func DefaultFeatureCacheConfig() *FeatureCacheConfig {
	return &FeatureCacheConfig{
		UserFeatureTTL: 30 * time.Minute,
		ItemFeatureTTL: 60 * time.Minute,
	}
}

// =============================================================================
// 错误定义
// =============================================================================

// FeatureError 特征服务错误类型
type FeatureError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *FeatureError) Error() string {
	return e.Message
}

// 预定义错误
var (
	// ErrUserNotFound 用户不存在
	ErrUserNotFound = &FeatureError{Code: "USER_NOT_FOUND", Message: "user not found"}
	// ErrItemNotFound 物品不存在
	ErrItemNotFound = &FeatureError{Code: "ITEM_NOT_FOUND", Message: "item not found"}
	// ErrFeatureNotFound 特征不存在
	ErrFeatureNotFound = &FeatureError{Code: "FEATURE_NOT_FOUND", Message: "feature not found"}
	// ErrCacheError 缓存错误
	ErrCacheError = &FeatureError{Code: "CACHE_ERROR", Message: "cache operation failed"}
	// ErrExtractionError 特征提取错误
	ErrExtractionError = &FeatureError{Code: "EXTRACTION_ERROR", Message: "feature extraction failed"}
)

