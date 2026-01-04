package mocks

import (
	"context"
	"sync"
	"time"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// MockUserService - 用户服务 Mock 实现
// =============================================================================

// MockUserService Mock 用户服务
//
// 实现 interfaces.UserService 接口
type MockUserService struct {
	mu    sync.RWMutex
	users map[string]*interfaces.User

	// 调用计数器
	GetUserCalls         int
	CreateUserCalls      int
	UpdateUserCalls      int
	DeleteUserCalls      int
	RecordBehaviorCalls  int
	GetUserBehaviorsCalls int
	GetUserProfileCalls  int

	// 可配置的返回值和错误
	GetUserResult          *interfaces.User
	GetUserError           error
	CreateUserResult       *interfaces.User
	CreateUserError        error
	UpdateUserResult       *interfaces.User
	UpdateUserError        error
	DeleteUserError        error
	RecordBehaviorError    error
	GetUserBehaviorsResult []*interfaces.UserBehavior
	GetUserBehaviorsError  error
	GetUserProfileResult   *interfaces.UserProfile
	GetUserProfileError    error
}

// NewMockUserService 创建 Mock 用户服务实例
func NewMockUserService() *MockUserService {
	return &MockUserService{
		users: make(map[string]*interfaces.User),
	}
}

// GetUser 获取用户
func (m *MockUserService) GetUser(ctx context.Context, userID string) (*interfaces.User, error) {
	m.mu.Lock()
	m.GetUserCalls++
	m.mu.Unlock()

	if m.GetUserError != nil {
		return nil, m.GetUserError
	}

	if m.GetUserResult != nil {
		return m.GetUserResult, nil
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	user, ok := m.users[userID]
	if !ok {
		return nil, ErrNotFound
	}
	return user, nil
}

// CreateUser 创建用户
func (m *MockUserService) CreateUser(ctx context.Context, req *interfaces.CreateUserRequest) (*interfaces.User, error) {
	m.mu.Lock()
	m.CreateUserCalls++
	m.mu.Unlock()

	if m.CreateUserError != nil {
		return nil, m.CreateUserError
	}

	if m.CreateUserResult != nil {
		return m.CreateUserResult, nil
	}

	user := &interfaces.User{
		ID:        "user_" + time.Now().Format("20060102150405"),
		Name:      req.Name,
		Email:     req.Email,
		Age:       req.Age,
		Gender:    req.Gender,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	m.mu.Lock()
	m.users[user.ID] = user
	m.mu.Unlock()

	return user, nil
}

// UpdateUser 更新用户
func (m *MockUserService) UpdateUser(ctx context.Context, userID string, req *interfaces.UpdateUserRequest) (*interfaces.User, error) {
	m.mu.Lock()
	m.UpdateUserCalls++
	m.mu.Unlock()

	if m.UpdateUserError != nil {
		return nil, m.UpdateUserError
	}

	if m.UpdateUserResult != nil {
		return m.UpdateUserResult, nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	user, ok := m.users[userID]
	if !ok {
		return nil, ErrNotFound
	}

	if req.Name != "" {
		user.Name = req.Name
	}
	if req.Email != "" {
		user.Email = req.Email
	}
	user.UpdatedAt = time.Now()

	return user, nil
}

// DeleteUser 删除用户
func (m *MockUserService) DeleteUser(ctx context.Context, userID string) error {
	m.mu.Lock()
	m.DeleteUserCalls++
	m.mu.Unlock()

	if m.DeleteUserError != nil {
		return m.DeleteUserError
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.users, userID)
	return nil
}

// RecordBehavior 记录行为
func (m *MockUserService) RecordBehavior(ctx context.Context, req *interfaces.RecordBehaviorRequest) error {
	m.mu.Lock()
	m.RecordBehaviorCalls++
	m.mu.Unlock()

	return m.RecordBehaviorError
}

// GetUserBehaviors 获取用户行为
func (m *MockUserService) GetUserBehaviors(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error) {
	m.mu.Lock()
	m.GetUserBehaviorsCalls++
	m.mu.Unlock()

	if m.GetUserBehaviorsError != nil {
		return nil, m.GetUserBehaviorsError
	}

	return m.GetUserBehaviorsResult, nil
}

// GetUserProfile 获取用户画像
func (m *MockUserService) GetUserProfile(ctx context.Context, userID string) (*interfaces.UserProfile, error) {
	m.mu.Lock()
	m.GetUserProfileCalls++
	m.mu.Unlock()

	if m.GetUserProfileError != nil {
		return nil, m.GetUserProfileError
	}

	return m.GetUserProfileResult, nil
}

// SetUser 设置用户数据（测试用）
func (m *MockUserService) SetUser(user *interfaces.User) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.users[user.ID] = user
}

// Reset 重置状态
func (m *MockUserService) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.users = make(map[string]*interfaces.User)
	m.GetUserCalls = 0
	m.CreateUserCalls = 0
	m.UpdateUserCalls = 0
	m.DeleteUserCalls = 0
	m.RecordBehaviorCalls = 0
	m.GetUserBehaviorsCalls = 0
	m.GetUserProfileCalls = 0
}

// =============================================================================
// MockItemService - 物品服务 Mock 实现
// =============================================================================

// MockItemService Mock 物品服务
//
// 实现 interfaces.ItemService 接口
type MockItemService struct {
	mu    sync.RWMutex
	items map[string]*interfaces.Item

	// 调用计数器
	GetItemCalls         int
	CreateItemCalls      int
	UpdateItemCalls      int
	DeleteItemCalls      int
	ListItemsCalls       int
	SearchItemsCalls     int
	GetSimilarItemsCalls int
	BatchGetItemsCalls   int
	GetItemStatsCalls    int

	// 可配置的返回值和错误
	GetItemResult         *interfaces.Item
	GetItemError          error
	CreateItemResult      *interfaces.Item
	CreateItemError       error
	ListItemsResult       *interfaces.ListItemsResponse
	ListItemsError        error
	SearchItemsResult     []*interfaces.Item
	SearchItemsError      error
	GetSimilarItemsResult []*interfaces.SimilarItem
	GetSimilarItemsError  error
	BatchGetItemsResult   []*interfaces.Item
	BatchGetItemsError    error
	GetItemStatsResult    *interfaces.ItemStats
	GetItemStatsError     error
}

// NewMockItemService 创建 Mock 物品服务实例
func NewMockItemService() *MockItemService {
	return &MockItemService{
		items: make(map[string]*interfaces.Item),
	}
}

// GetItem 获取物品
func (m *MockItemService) GetItem(ctx context.Context, itemID string) (*interfaces.Item, error) {
	m.mu.Lock()
	m.GetItemCalls++
	m.mu.Unlock()

	if m.GetItemError != nil {
		return nil, m.GetItemError
	}

	if m.GetItemResult != nil {
		return m.GetItemResult, nil
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	item, ok := m.items[itemID]
	if !ok {
		return nil, ErrNotFound
	}
	return item, nil
}

// CreateItem 创建物品
func (m *MockItemService) CreateItem(ctx context.Context, req *interfaces.CreateItemRequest) (*interfaces.Item, error) {
	m.mu.Lock()
	m.CreateItemCalls++
	m.mu.Unlock()

	if m.CreateItemError != nil {
		return nil, m.CreateItemError
	}

	if m.CreateItemResult != nil {
		return m.CreateItemResult, nil
	}

	item := &interfaces.Item{
		ID:          "item_" + time.Now().Format("20060102150405"),
		Type:        req.Type,
		Title:       req.Title,
		Description: req.Description,
		Category:    req.Category,
		Tags:        req.Tags,
		Status:      "active",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	m.mu.Lock()
	m.items[item.ID] = item
	m.mu.Unlock()

	return item, nil
}

// UpdateItem 更新物品
func (m *MockItemService) UpdateItem(ctx context.Context, itemID string, req *interfaces.UpdateItemRequest) (*interfaces.Item, error) {
	m.mu.Lock()
	m.UpdateItemCalls++
	m.mu.Unlock()

	m.mu.Lock()
	defer m.mu.Unlock()

	item, ok := m.items[itemID]
	if !ok {
		return nil, ErrNotFound
	}

	if req.Title != "" {
		item.Title = req.Title
	}
	if req.Description != "" {
		item.Description = req.Description
	}
	item.UpdatedAt = time.Now()

	return item, nil
}

// DeleteItem 删除物品
func (m *MockItemService) DeleteItem(ctx context.Context, itemID string) error {
	m.mu.Lock()
	m.DeleteItemCalls++
	m.mu.Unlock()

	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.items, itemID)
	return nil
}

// ListItems 列出物品
func (m *MockItemService) ListItems(ctx context.Context, req *interfaces.ListItemsRequest) (*interfaces.ListItemsResponse, error) {
	m.mu.Lock()
	m.ListItemsCalls++
	m.mu.Unlock()

	if m.ListItemsError != nil {
		return nil, m.ListItemsError
	}

	if m.ListItemsResult != nil {
		return m.ListItemsResult, nil
	}

	return &interfaces.ListItemsResponse{
		Items: []*interfaces.Item{},
		Total: 0,
		Page:  req.Page,
	}, nil
}

// SearchItems 搜索物品
func (m *MockItemService) SearchItems(ctx context.Context, query string, limit int) ([]*interfaces.Item, error) {
	m.mu.Lock()
	m.SearchItemsCalls++
	m.mu.Unlock()

	if m.SearchItemsError != nil {
		return nil, m.SearchItemsError
	}

	return m.SearchItemsResult, nil
}

// GetSimilarItems 获取相似物品
func (m *MockItemService) GetSimilarItems(ctx context.Context, itemID string, topK int) ([]*interfaces.SimilarItem, error) {
	m.mu.Lock()
	m.GetSimilarItemsCalls++
	m.mu.Unlock()

	if m.GetSimilarItemsError != nil {
		return nil, m.GetSimilarItemsError
	}

	return m.GetSimilarItemsResult, nil
}

// BatchGetItems 批量获取物品
func (m *MockItemService) BatchGetItems(ctx context.Context, itemIDs []string) ([]*interfaces.Item, error) {
	m.mu.Lock()
	m.BatchGetItemsCalls++
	m.mu.Unlock()

	if m.BatchGetItemsError != nil {
		return nil, m.BatchGetItemsError
	}

	if m.BatchGetItemsResult != nil {
		return m.BatchGetItemsResult, nil
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]*interfaces.Item, 0)
	for _, id := range itemIDs {
		if item, ok := m.items[id]; ok {
			result = append(result, item)
		}
	}
	return result, nil
}

// GetItemStats 获取物品统计
func (m *MockItemService) GetItemStats(ctx context.Context, itemID string) (*interfaces.ItemStats, error) {
	m.mu.Lock()
	m.GetItemStatsCalls++
	m.mu.Unlock()

	if m.GetItemStatsError != nil {
		return nil, m.GetItemStatsError
	}

	return m.GetItemStatsResult, nil
}

// SetItem 设置物品数据（测试用）
func (m *MockItemService) SetItem(item *interfaces.Item) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.items[item.ID] = item
}

// Reset 重置状态
func (m *MockItemService) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.items = make(map[string]*interfaces.Item)
	m.GetItemCalls = 0
	m.CreateItemCalls = 0
	m.UpdateItemCalls = 0
	m.DeleteItemCalls = 0
}

// =============================================================================
// MockFeatureService - 特征服务 Mock 实现
// =============================================================================

// MockFeatureService Mock 特征服务
//
// 实现 interfaces.FeatureService 接口
type MockFeatureService struct {
	mu sync.RWMutex

	// 调用计数器
	GetUserFeaturesCalls        int
	GetItemFeaturesCalls        int
	GetFeatureVectorCalls       int
	BatchGetFeatureVectorsCalls int
	RefreshUserFeaturesCalls    int
	RefreshItemFeaturesCalls    int

	// 可配置的返回值和错误
	GetUserFeaturesResult        *interfaces.UserFeatures
	GetUserFeaturesError         error
	GetItemFeaturesResult        *interfaces.ItemFeatures
	GetItemFeaturesError         error
	GetFeatureVectorResult       *interfaces.FeatureVector
	GetFeatureVectorError        error
	BatchGetFeatureVectorsResult []*interfaces.FeatureVector
	BatchGetFeatureVectorsError  error
	RefreshUserFeaturesError     error
	RefreshItemFeaturesError     error
}

// NewMockFeatureService 创建 Mock 特征服务实例
func NewMockFeatureService() *MockFeatureService {
	return &MockFeatureService{}
}

// GetUserFeatures 获取用户特征
func (m *MockFeatureService) GetUserFeatures(ctx context.Context, userID string) (*interfaces.UserFeatures, error) {
	m.mu.Lock()
	m.GetUserFeaturesCalls++
	m.mu.Unlock()

	if m.GetUserFeaturesError != nil {
		return nil, m.GetUserFeaturesError
	}

	if m.GetUserFeaturesResult != nil {
		return m.GetUserFeaturesResult, nil
	}

	return &interfaces.UserFeatures{
		UserID:       userID,
		Demographics: make(map[string]interface{}),
		Behavior:     make(map[string]interface{}),
		Preferences:  make(map[string]interface{}),
		LastUpdated:  time.Now(),
	}, nil
}

// GetItemFeatures 获取物品特征
func (m *MockFeatureService) GetItemFeatures(ctx context.Context, itemID string) (*interfaces.ItemFeatures, error) {
	m.mu.Lock()
	m.GetItemFeaturesCalls++
	m.mu.Unlock()

	if m.GetItemFeaturesError != nil {
		return nil, m.GetItemFeaturesError
	}

	if m.GetItemFeaturesResult != nil {
		return m.GetItemFeaturesResult, nil
	}

	return &interfaces.ItemFeatures{
		ItemID:      itemID,
		Content:     make(map[string]interface{}),
		Statistics:  make(map[string]interface{}),
		LastUpdated: time.Now(),
	}, nil
}

// GetFeatureVector 获取特征向量
func (m *MockFeatureService) GetFeatureVector(ctx context.Context, req *interfaces.FeatureVectorRequest) (*interfaces.FeatureVector, error) {
	m.mu.Lock()
	m.GetFeatureVectorCalls++
	m.mu.Unlock()

	if m.GetFeatureVectorError != nil {
		return nil, m.GetFeatureVectorError
	}

	return m.GetFeatureVectorResult, nil
}

// BatchGetFeatureVectors 批量获取特征向量
func (m *MockFeatureService) BatchGetFeatureVectors(ctx context.Context, reqs []*interfaces.FeatureVectorRequest) ([]*interfaces.FeatureVector, error) {
	m.mu.Lock()
	m.BatchGetFeatureVectorsCalls++
	m.mu.Unlock()

	if m.BatchGetFeatureVectorsError != nil {
		return nil, m.BatchGetFeatureVectorsError
	}

	return m.BatchGetFeatureVectorsResult, nil
}

// RefreshUserFeatures 刷新用户特征
func (m *MockFeatureService) RefreshUserFeatures(ctx context.Context, userID string) error {
	m.mu.Lock()
	m.RefreshUserFeaturesCalls++
	m.mu.Unlock()

	return m.RefreshUserFeaturesError
}

// RefreshItemFeatures 刷新物品特征
func (m *MockFeatureService) RefreshItemFeatures(ctx context.Context, itemID string) error {
	m.mu.Lock()
	m.RefreshItemFeaturesCalls++
	m.mu.Unlock()

	return m.RefreshItemFeaturesError
}

// Reset 重置状态
func (m *MockFeatureService) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.GetUserFeaturesCalls = 0
	m.GetItemFeaturesCalls = 0
	m.GetFeatureVectorCalls = 0
	m.BatchGetFeatureVectorsCalls = 0
	m.RefreshUserFeaturesCalls = 0
	m.RefreshItemFeaturesCalls = 0
}

// =============================================================================
// MockColdStartService - 冷启动服务 Mock 实现
// =============================================================================

// MockColdStartService Mock 冷启动服务
//
// 实现 interfaces.ColdStartService 接口
type MockColdStartService struct {
	mu sync.RWMutex

	// 调用计数器
	HandleNewUserCalls              int
	HandleNewItemCalls              int
	GetColdStartRecommendationsCalls int
	ExplainRecommendationCalls      int

	// 可配置的返回值和错误
	HandleNewUserResult              *interfaces.ColdStartResult
	HandleNewUserError               error
	HandleNewItemResult              *interfaces.ItemColdStartResult
	HandleNewItemError               error
	GetColdStartRecommendationsResult []*interfaces.Item
	GetColdStartRecommendationsError error
	ExplainRecommendationResult      string
	ExplainRecommendationError       error
}

// NewMockColdStartService 创建 Mock 冷启动服务实例
func NewMockColdStartService() *MockColdStartService {
	return &MockColdStartService{}
}

// HandleNewUser 处理新用户
func (m *MockColdStartService) HandleNewUser(ctx context.Context, user *interfaces.User) (*interfaces.ColdStartResult, error) {
	m.mu.Lock()
	m.HandleNewUserCalls++
	m.mu.Unlock()

	if m.HandleNewUserError != nil {
		return nil, m.HandleNewUserError
	}

	if m.HandleNewUserResult != nil {
		return m.HandleNewUserResult, nil
	}

	return &interfaces.ColdStartResult{
		UserID:          user.ID,
		Preferences:     make(map[string]interface{}),
		Recommendations: []string{},
		Strategy:        "popular",
		CreatedAt:       time.Now(),
	}, nil
}

// HandleNewItem 处理新物品
func (m *MockColdStartService) HandleNewItem(ctx context.Context, item *interfaces.Item) (*interfaces.ItemColdStartResult, error) {
	m.mu.Lock()
	m.HandleNewItemCalls++
	m.mu.Unlock()

	if m.HandleNewItemError != nil {
		return nil, m.HandleNewItemError
	}

	if m.HandleNewItemResult != nil {
		return m.HandleNewItemResult, nil
	}

	return &interfaces.ItemColdStartResult{
		ItemID:       item.ID,
		Features:     make(map[string]interface{}),
		SimilarItems: []string{},
		Strategy:     "content",
		CreatedAt:    time.Now(),
	}, nil
}

// GetColdStartRecommendations 获取冷启动推荐
func (m *MockColdStartService) GetColdStartRecommendations(ctx context.Context, userID string, limit int) ([]*interfaces.Item, error) {
	m.mu.Lock()
	m.GetColdStartRecommendationsCalls++
	m.mu.Unlock()

	if m.GetColdStartRecommendationsError != nil {
		return nil, m.GetColdStartRecommendationsError
	}

	return m.GetColdStartRecommendationsResult, nil
}

// ExplainRecommendation 解释推荐
func (m *MockColdStartService) ExplainRecommendation(ctx context.Context, userID, itemID string) (string, error) {
	m.mu.Lock()
	m.ExplainRecommendationCalls++
	m.mu.Unlock()

	if m.ExplainRecommendationError != nil {
		return "", m.ExplainRecommendationError
	}

	if m.ExplainRecommendationResult != "" {
		return m.ExplainRecommendationResult, nil
	}

	return "Based on your preferences", nil
}

// Reset 重置状态
func (m *MockColdStartService) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.HandleNewUserCalls = 0
	m.HandleNewItemCalls = 0
	m.GetColdStartRecommendationsCalls = 0
	m.ExplainRecommendationCalls = 0
}

// =============================================================================
// MockLLMClient - LLM 客户端 Mock 实现
// =============================================================================

// MockLLMClient Mock LLM 客户端
//
// 实现 interfaces.LLMClient 接口
type MockLLMClient struct {
	mu sync.RWMutex

	// 调用计数器
	CompleteCalls int
	EmbedCalls    int
	ChatCalls     int

	// 可配置的返回值和错误
	CompleteResult string
	CompleteError  error
	EmbedResult    []float32
	EmbedError     error
	ChatResult     string
	ChatError      error
}

// NewMockLLMClient 创建 Mock LLM 客户端实例
func NewMockLLMClient() *MockLLMClient {
	return &MockLLMClient{
		EmbedResult: make([]float32, 768), // 默认 768 维向量
	}
}

// Complete 文本补全
func (m *MockLLMClient) Complete(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
	m.mu.Lock()
	m.CompleteCalls++
	m.mu.Unlock()

	if m.CompleteError != nil {
		return "", m.CompleteError
	}

	if m.CompleteResult != "" {
		return m.CompleteResult, nil
	}

	return "This is a mock completion response.", nil
}

// Embed 文本嵌入
func (m *MockLLMClient) Embed(ctx context.Context, text string) ([]float32, error) {
	m.mu.Lock()
	m.EmbedCalls++
	m.mu.Unlock()

	if m.EmbedError != nil {
		return nil, m.EmbedError
	}

	return m.EmbedResult, nil
}

// Chat 对话
func (m *MockLLMClient) Chat(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
	m.mu.Lock()
	m.ChatCalls++
	m.mu.Unlock()

	if m.ChatError != nil {
		return "", m.ChatError
	}

	if m.ChatResult != "" {
		return m.ChatResult, nil
	}

	return "This is a mock chat response.", nil
}

// Reset 重置状态
func (m *MockLLMClient) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.CompleteCalls = 0
	m.EmbedCalls = 0
	m.ChatCalls = 0
}

// =============================================================================
// MockInferenceClient - 推理客户端 Mock 实现
// =============================================================================

// MockInferenceClient Mock 推理客户端
//
// 实现 interfaces.InferenceClient 接口
type MockInferenceClient struct {
	mu sync.RWMutex

	// 调用计数器
	InferCalls      int
	BatchInferCalls int
	HealthCalls     int

	// 可配置的返回值和错误
	InferResult      *interfaces.ModelOutput
	InferError       error
	BatchInferResult []*interfaces.ModelOutput
	BatchInferError  error
	HealthError      error
}

// NewMockInferenceClient 创建 Mock 推理客户端实例
func NewMockInferenceClient() *MockInferenceClient {
	return &MockInferenceClient{}
}

// Infer 推理
func (m *MockInferenceClient) Infer(ctx context.Context, input *interfaces.ModelInput) (*interfaces.ModelOutput, error) {
	m.mu.Lock()
	m.InferCalls++
	m.mu.Unlock()

	if m.InferError != nil {
		return nil, m.InferError
	}

	if m.InferResult != nil {
		return m.InferResult, nil
	}

	// 返回默认模拟结果
	return &interfaces.ModelOutput{
		Recommendations: [][3]int{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
		Scores:          []float32{0.9, 0.8, 0.7},
	}, nil
}

// BatchInfer 批量推理
func (m *MockInferenceClient) BatchInfer(ctx context.Context, inputs []*interfaces.ModelInput) ([]*interfaces.ModelOutput, error) {
	m.mu.Lock()
	m.BatchInferCalls++
	m.mu.Unlock()

	if m.BatchInferError != nil {
		return nil, m.BatchInferError
	}

	if m.BatchInferResult != nil {
		return m.BatchInferResult, nil
	}

	results := make([]*interfaces.ModelOutput, len(inputs))
	for i := range inputs {
		results[i] = &interfaces.ModelOutput{
			Recommendations: [][3]int{{1, 2, 3}},
			Scores:          []float32{0.9},
		}
	}
	return results, nil
}

// Health 健康检查
func (m *MockInferenceClient) Health(ctx context.Context) error {
	m.mu.Lock()
	m.HealthCalls++
	m.mu.Unlock()

	return m.HealthError
}

// Reset 重置状态
func (m *MockInferenceClient) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.InferCalls = 0
	m.BatchInferCalls = 0
	m.HealthCalls = 0
}

// =============================================================================
// MockRecommendRepository - 推荐仓库 Mock 实现
// =============================================================================

// MockRecommendRepository Mock 推荐仓库
//
// 实现 interfaces.RecommendRepository 接口
type MockRecommendRepository struct {
	mu        sync.RWMutex
	logs      []*interfaces.RecommendLog
	exposures map[string][]string // userID -> itemIDs

	// 调用计数器
	LogRecommendationCalls    int
	GetRecommendationLogsCalls int
	RecordExposureCalls       int

	// 可配置的错误
	LogRecommendationError    error
	GetRecommendationLogsError error
	RecordExposureError       error
}

// NewMockRecommendRepository 创建 Mock 推荐仓库实例
func NewMockRecommendRepository() *MockRecommendRepository {
	return &MockRecommendRepository{
		logs:      make([]*interfaces.RecommendLog, 0),
		exposures: make(map[string][]string),
	}
}

// LogRecommendation 记录推荐日志
func (m *MockRecommendRepository) LogRecommendation(ctx context.Context, log *interfaces.RecommendLog) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.LogRecommendationCalls++

	if m.LogRecommendationError != nil {
		return m.LogRecommendationError
	}

	logCopy := *log
	m.logs = append(m.logs, &logCopy)
	return nil
}

// GetRecommendationLogs 获取推荐日志
func (m *MockRecommendRepository) GetRecommendationLogs(ctx context.Context, userID string, limit int) ([]*interfaces.RecommendLog, error) {
	m.mu.Lock()
	m.GetRecommendationLogsCalls++
	m.mu.Unlock()

	if m.GetRecommendationLogsError != nil {
		return nil, m.GetRecommendationLogsError
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	var result []*interfaces.RecommendLog
	for _, log := range m.logs {
		if log.UserID == userID {
			result = append(result, log)
		}
	}

	if limit > 0 && len(result) > limit {
		result = result[:limit]
	}

	return result, nil
}

// RecordExposure 记录曝光
func (m *MockRecommendRepository) RecordExposure(ctx context.Context, userID, itemID, requestID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.RecordExposureCalls++

	if m.RecordExposureError != nil {
		return m.RecordExposureError
	}

	m.exposures[userID] = append(m.exposures[userID], itemID)
	return nil
}

// GetExposures 获取曝光记录（测试用）
func (m *MockRecommendRepository) GetExposures(userID string) []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.exposures[userID]
}

// Reset 重置状态
func (m *MockRecommendRepository) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.logs = make([]*interfaces.RecommendLog, 0)
	m.exposures = make(map[string][]string)
	m.LogRecommendationCalls = 0
	m.GetRecommendationLogsCalls = 0
	m.RecordExposureCalls = 0
}

