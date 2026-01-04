// Package interfaces 定义所有服务的接口
//
// 这是 Go 后端的核心接口定义文件，所有模块开发者必须遵循这些接口。
// 通过接口驱动开发，实现模块间解耦和可插拔设计。
//
// 使用方式：
//   - 各服务实现对应接口
//   - 依赖注入时使用接口类型
//   - 测试时使用 Mock 实现
package interfaces

import (
	"context"
	"time"
)

// =============================================================================
// 用户服务接口 (Person A 实现)
// =============================================================================

// UserService 用户服务接口
type UserService interface {
	// GetUser 获取用户信息
	GetUser(ctx context.Context, userID string) (*User, error)

	// CreateUser 创建用户
	CreateUser(ctx context.Context, req *CreateUserRequest) (*User, error)

	// UpdateUser 更新用户
	UpdateUser(ctx context.Context, userID string, req *UpdateUserRequest) (*User, error)

	// DeleteUser 删除用户
	DeleteUser(ctx context.Context, userID string) error

	// RecordBehavior 记录用户行为
	RecordBehavior(ctx context.Context, req *RecordBehaviorRequest) error

	// GetUserBehaviors 获取用户行为历史
	GetUserBehaviors(ctx context.Context, userID string, limit int) ([]*UserBehavior, error)

	// GetUserProfile 获取用户画像
	GetUserProfile(ctx context.Context, userID string) (*UserProfile, error)
}

// =============================================================================
// 物品服务接口 (Person B 实现)
// =============================================================================

// ItemService 物品服务接口
type ItemService interface {
	// GetItem 获取物品信息
	GetItem(ctx context.Context, itemID string) (*Item, error)

	// CreateItem 创建物品
	CreateItem(ctx context.Context, req *CreateItemRequest) (*Item, error)

	// UpdateItem 更新物品
	UpdateItem(ctx context.Context, itemID string, req *UpdateItemRequest) (*Item, error)

	// DeleteItem 删除物品
	DeleteItem(ctx context.Context, itemID string) error

	// ListItems 列出物品
	ListItems(ctx context.Context, req *ListItemsRequest) (*ListItemsResponse, error)

	// SearchItems 搜索物品
	SearchItems(ctx context.Context, query string, limit int) ([]*Item, error)

	// GetSimilarItems 获取相似物品（向量搜索）
	GetSimilarItems(ctx context.Context, itemID string, topK int) ([]*SimilarItem, error)

	// BatchGetItems 批量获取物品
	BatchGetItems(ctx context.Context, itemIDs []string) ([]*Item, error)

	// GetItemStats 获取物品统计
	GetItemStats(ctx context.Context, itemID string) (*ItemStats, error)
}

// =============================================================================
// 特征服务接口 (Person C 实现)
// =============================================================================

// FeatureService 特征服务接口
type FeatureService interface {
	// GetUserFeatures 获取用户特征
	GetUserFeatures(ctx context.Context, userID string) (*UserFeatures, error)

	// GetItemFeatures 获取物品特征
	GetItemFeatures(ctx context.Context, itemID string) (*ItemFeatures, error)

	// GetFeatureVector 获取完整特征向量（用于模型推理）
	GetFeatureVector(ctx context.Context, req *FeatureVectorRequest) (*FeatureVector, error)

	// BatchGetFeatureVectors 批量获取特征向量
	BatchGetFeatureVectors(ctx context.Context, reqs []*FeatureVectorRequest) ([]*FeatureVector, error)

	// RefreshUserFeatures 刷新用户特征
	RefreshUserFeatures(ctx context.Context, userID string) error

	// RefreshItemFeatures 刷新物品特征
	RefreshItemFeatures(ctx context.Context, itemID string) error
}

// =============================================================================
// 冷启动服务接口 (Person D 实现)
// =============================================================================

// ColdStartService 冷启动服务接口
type ColdStartService interface {
	// HandleNewUser 处理新用户冷启动
	HandleNewUser(ctx context.Context, user *User) (*ColdStartResult, error)

	// HandleNewItem 处理新物品冷启动
	HandleNewItem(ctx context.Context, item *Item) (*ItemColdStartResult, error)

	// GetColdStartRecommendations 获取冷启动推荐
	GetColdStartRecommendations(ctx context.Context, userID string, limit int) ([]*Item, error)

	// ExplainRecommendation 生成推荐解释
	ExplainRecommendation(ctx context.Context, userID, itemID string) (string, error)
}

// LLMClient LLM 客户端接口
type LLMClient interface {
	// Complete 文本补全
	Complete(ctx context.Context, prompt string, opts ...LLMOption) (string, error)

	// Embed 文本嵌入
	Embed(ctx context.Context, text string) ([]float32, error)

	// Chat 对话式交互
	Chat(ctx context.Context, messages []Message, opts ...LLMOption) (string, error)
}

// =============================================================================
// 推荐服务接口 (已实现，供参考)
// =============================================================================

// RecommendService 推荐服务接口
type RecommendService interface {
	// GetRecommendations 获取推荐列表
	GetRecommendations(ctx context.Context, req *RecommendRequest) (*RecommendResponse, error)

	// GetSimilarItems 获取相似物品推荐
	GetSimilarItemRecommendations(ctx context.Context, itemID string, limit int) ([]*Recommendation, error)

	// SubmitFeedback 提交反馈
	SubmitFeedback(ctx context.Context, feedback *Feedback) error
}

// =============================================================================
// 推理服务接口 (已实现，供参考)
// =============================================================================

// InferenceClient 推理客户端接口
type InferenceClient interface {
	// Infer 执行推理
	Infer(ctx context.Context, input *ModelInput) (*ModelOutput, error)

	// BatchInfer 批量推理
	BatchInfer(ctx context.Context, inputs []*ModelInput) ([]*ModelOutput, error)

	// Health 健康检查
	Health(ctx context.Context) error
}

// =============================================================================
// 数据访问层接口 (Repository)
// =============================================================================

// UserRepository 用户仓库接口
type UserRepository interface {
	GetByID(ctx context.Context, userID string) (*User, error)
	GetByIDs(ctx context.Context, userIDs []string) ([]*User, error)
	Create(ctx context.Context, user *User) error
	Update(ctx context.Context, user *User) error
	Delete(ctx context.Context, userID string) error
	GetBehaviors(ctx context.Context, userID string, limit int) ([]*UserBehavior, error)
	AddBehavior(ctx context.Context, behavior *UserBehavior) error
	GetUserItemInteractions(ctx context.Context, userID, itemID string) ([]*UserBehavior, error)
}

// ItemRepository 物品仓库接口
type ItemRepository interface {
	GetByID(ctx context.Context, itemID string) (*Item, error)
	GetByIDs(ctx context.Context, itemIDs []string) ([]*Item, error)
	Create(ctx context.Context, item *Item) error
	Update(ctx context.Context, item *Item) error
	Delete(ctx context.Context, itemID string) error
	List(ctx context.Context, itemType, category string, page, pageSize int) ([]*Item, int64, error)
	Search(ctx context.Context, query string, limit int) ([]*Item, error)
	GetStats(ctx context.Context, itemID string) (*ItemStats, error)
	IncrementStats(ctx context.Context, itemID, action string) error
	GetPopularByCategories(ctx context.Context, categories []string, limit int) ([]*Item, error)
}

// RecommendRepository 推荐仓库接口
type RecommendRepository interface {
	LogRecommendation(ctx context.Context, log *RecommendLog) error
	GetRecommendationLogs(ctx context.Context, userID string, limit int) ([]*RecommendLog, error)
	RecordExposure(ctx context.Context, userID, itemID, requestID string) error
}

// =============================================================================
// 缓存接口
// =============================================================================

// Cache 缓存接口
type Cache interface {
	Get(ctx context.Context, key string, value interface{}) error
	Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error
	Delete(ctx context.Context, key string) error
	Exists(ctx context.Context, key string) (bool, error)
	MGet(ctx context.Context, keys []string) ([]interface{}, error)
	MSet(ctx context.Context, kvs map[string]interface{}, ttl time.Duration) error
}

// =============================================================================
// 数据结构定义
// =============================================================================

// User 用户
type User struct {
	ID        string            `json:"id"`
	Name      string            `json:"name"`
	Email     string            `json:"email"`
	Age       int               `json:"age"`
	Gender    string            `json:"gender"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	CreatedAt time.Time         `json:"created_at"`
	UpdatedAt time.Time         `json:"updated_at"`
}

// UserBehavior 用户行为
type UserBehavior struct {
	UserID    string            `json:"user_id"`
	ItemID    string            `json:"item_id"`
	Action    string            `json:"action"`
	Timestamp time.Time         `json:"timestamp"`
	Context   map[string]string `json:"context,omitempty"`
}

// UserProfile 用户画像
type UserProfile struct {
	User           *User               `json:"user"`
	TotalActions   int                 `json:"total_actions"`
	PreferredTypes map[string]int      `json:"preferred_types"`
	ActiveHours    map[int]int         `json:"active_hours"`
	LastActive     time.Time           `json:"last_active"`
}

// Item 物品
type Item struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	Category    string                 `json:"category"`
	Tags        []string               `json:"tags"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	Status      string                 `json:"status"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// ItemStats 物品统计
type ItemStats struct {
	ItemID     string  `json:"item_id"`
	ViewCount  int64   `json:"view_count"`
	ClickCount int64   `json:"click_count"`
	LikeCount  int64   `json:"like_count"`
	ShareCount int64   `json:"share_count"`
	AvgRating  float64 `json:"avg_rating"`
}

// SimilarItem 相似物品
type SimilarItem struct {
	Item  *Item   `json:"item"`
	Score float32 `json:"score"`
}

// Recommendation 推荐项
type Recommendation struct {
	ItemID     string     `json:"item_id"`
	Score      float32    `json:"score"`
	Reason     string     `json:"reason,omitempty"`
	SemanticID [3]int     `json:"semantic_id,omitempty"`
}

// Feedback 反馈
type Feedback struct {
	UserID    string    `json:"user_id"`
	ItemID    string    `json:"item_id"`
	Action    string    `json:"action"`
	RequestID string    `json:"request_id,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

// RecommendLog 推荐日志
type RecommendLog struct {
	RequestID       string    `json:"request_id"`
	UserID          string    `json:"user_id"`
	Recommendations []string  `json:"recommendations"`
	Strategy        string    `json:"strategy"`
	Timestamp       time.Time `json:"timestamp"`
}

// =============================================================================
// 请求/响应结构
// =============================================================================

// CreateUserRequest 创建用户请求
type CreateUserRequest struct {
	Name   string `json:"name" binding:"required"`
	Email  string `json:"email" binding:"required,email"`
	Age    int    `json:"age"`
	Gender string `json:"gender"`
}

// UpdateUserRequest 更新用户请求
type UpdateUserRequest struct {
	Name   string `json:"name"`
	Email  string `json:"email"`
	Age    int    `json:"age"`
	Gender string `json:"gender"`
}

// RecordBehaviorRequest 记录行为请求
type RecordBehaviorRequest struct {
	UserID  string            `json:"user_id" binding:"required"`
	ItemID  string            `json:"item_id" binding:"required"`
	Action  string            `json:"action" binding:"required"`
	Context map[string]string `json:"context"`
}

// CreateItemRequest 创建物品请求
type CreateItemRequest struct {
	Type        string                 `json:"type" binding:"required"`
	Title       string                 `json:"title" binding:"required"`
	Description string                 `json:"description"`
	Category    string                 `json:"category"`
	Tags        []string               `json:"tags"`
	Metadata    map[string]interface{} `json:"metadata"`
	Embedding   []float32              `json:"embedding,omitempty"`
}

// UpdateItemRequest 更新物品请求
type UpdateItemRequest struct {
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	Category    string                 `json:"category"`
	Tags        []string               `json:"tags"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// ListItemsRequest 列出物品请求
type ListItemsRequest struct {
	Type     string `form:"type"`
	Category string `form:"category"`
	Page     int    `form:"page,default=1"`
	PageSize int    `form:"page_size,default=20"`
}

// ListItemsResponse 列出物品响应
type ListItemsResponse struct {
	Items []*Item `json:"items"`
	Total int64   `json:"total"`
	Page  int     `json:"page"`
}

// RecommendRequest 推荐请求
type RecommendRequest struct {
	UserID       string            `json:"user_id" binding:"required"`
	Limit        int               `json:"limit,default=20"`
	ExcludeItems []string          `json:"exclude_items"`
	Scene        string            `json:"scene"`
	Context      map[string]string `json:"context"`
}

// RecommendResponse 推荐响应
type RecommendResponse struct {
	Recommendations []*Recommendation `json:"recommendations"`
	RequestID       string            `json:"request_id"`
	Strategy        string            `json:"strategy"`
}

// =============================================================================
// 特征相关结构
// =============================================================================

// UserFeatures 用户特征
type UserFeatures struct {
	UserID      string    `json:"user_id"`
	Demographics map[string]interface{} `json:"demographics"`
	Behavior    map[string]interface{} `json:"behavior"`
	Preferences map[string]interface{} `json:"preferences"`
	Embedding   []float32 `json:"embedding,omitempty"`
	LastUpdated time.Time `json:"last_updated"`
}

// ItemFeatures 物品特征
type ItemFeatures struct {
	ItemID      string                 `json:"item_id"`
	Type        string                 `json:"type"`
	Content     map[string]interface{} `json:"content"`
	Statistics  map[string]interface{} `json:"statistics"`
	Embedding   []float32              `json:"embedding,omitempty"`
	SemanticID  [3]int                 `json:"semantic_id"`
	LastUpdated time.Time              `json:"last_updated"`
}

// FeatureVectorRequest 特征向量请求
type FeatureVectorRequest struct {
	UserID  string            `json:"user_id" binding:"required"`
	ItemID  string            `json:"item_id"`
	Context map[string]string `json:"context"`
}

// FeatureVector 特征向量
type FeatureVector struct {
	UserFeatures *UserFeatures `json:"user_features"`
	ItemFeatures *ItemFeatures `json:"item_features,omitempty"`
	TokenIDs     []int64       `json:"token_ids,omitempty"`
	TokenTypes   []int         `json:"token_types,omitempty"`
	Positions    []int         `json:"positions,omitempty"`
}

// =============================================================================
// 冷启动相关结构
// =============================================================================

// ColdStartResult 冷启动结果
type ColdStartResult struct {
	UserID          string                 `json:"user_id"`
	Preferences     map[string]interface{} `json:"preferences"`
	Recommendations []string               `json:"recommendations"`
	Strategy        string                 `json:"strategy"`
	CreatedAt       time.Time              `json:"created_at"`
}

// ItemColdStartResult 物品冷启动结果
type ItemColdStartResult struct {
	ItemID       string                 `json:"item_id"`
	Features     map[string]interface{} `json:"features"`
	Embedding    []float32              `json:"embedding,omitempty"`
	SimilarItems []string               `json:"similar_items"`
	Strategy     string                 `json:"strategy"`
	CreatedAt    time.Time              `json:"created_at"`
}

// =============================================================================
// LLM 相关结构
// =============================================================================

// Message LLM 消息
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// LLMOption LLM 选项
type LLMOption func(*LLMOptions)

// LLMOptions LLM 选项
type LLMOptions struct {
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
	Model       string  `json:"model"`
}

// WithMaxTokens 设置最大 Token 数
func WithMaxTokens(n int) LLMOption {
	return func(o *LLMOptions) { o.MaxTokens = n }
}

// WithTemperature 设置温度
func WithTemperature(t float64) LLMOption {
	return func(o *LLMOptions) { o.Temperature = t }
}

// WithModel 设置模型
func WithModel(model string) LLMOption {
	return func(o *LLMOptions) { o.Model = model }
}

// =============================================================================
// 推理相关结构
// =============================================================================

// ModelInput 模型输入
type ModelInput struct {
	UserSequence  []int64   `json:"user_sequence"`
	AttentionMask []int64   `json:"attention_mask"`
	TokenTypes    []int64   `json:"token_types"`
	Positions     []int64   `json:"positions"`
}

// ModelOutput 模型输出
type ModelOutput struct {
	Recommendations [][3]int  `json:"recommendations"` // [[L1, L2, L3], ...]
	Scores          []float32 `json:"scores"`
}

