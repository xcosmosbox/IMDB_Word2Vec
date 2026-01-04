package coldstart

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"recommend-system/internal/interfaces"

	"go.uber.org/zap"
)

// =============================================================================
// Mock 实现
// =============================================================================

// mockLLMClient Mock LLM 客户端
type mockLLMClient struct {
	completeFunc func(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error)
	embedFunc    func(ctx context.Context, text string) ([]float32, error)
	chatFunc     func(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error)
}

func (m *mockLLMClient) Complete(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
	if m.completeFunc != nil {
		return m.completeFunc(ctx, prompt, opts...)
	}
	return "mock complete response", nil
}

func (m *mockLLMClient) Embed(ctx context.Context, text string) ([]float32, error) {
	if m.embedFunc != nil {
		return m.embedFunc(ctx, text)
	}
	return make([]float32, 768), nil
}

func (m *mockLLMClient) Chat(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
	if m.chatFunc != nil {
		return m.chatFunc(ctx, messages, opts...)
	}
	return `{"preferred_categories": ["科技", "游戏"], "preferred_tags": ["新品"]}`, nil
}

// mockUserRepository Mock 用户仓库
type mockUserRepository struct {
	users map[string]*interfaces.User
}

func newMockUserRepository() *mockUserRepository {
	return &mockUserRepository{
		users: make(map[string]*interfaces.User),
	}
}

func (m *mockUserRepository) GetByID(ctx context.Context, userID string) (*interfaces.User, error) {
	if user, ok := m.users[userID]; ok {
		return user, nil
	}
	return nil, errors.New("user not found")
}

func (m *mockUserRepository) GetByIDs(ctx context.Context, userIDs []string) ([]*interfaces.User, error) {
	var users []*interfaces.User
	for _, id := range userIDs {
		if user, ok := m.users[id]; ok {
			users = append(users, user)
		}
	}
	return users, nil
}

func (m *mockUserRepository) Create(ctx context.Context, user *interfaces.User) error {
	m.users[user.ID] = user
	return nil
}

func (m *mockUserRepository) Update(ctx context.Context, user *interfaces.User) error {
	m.users[user.ID] = user
	return nil
}

func (m *mockUserRepository) Delete(ctx context.Context, userID string) error {
	delete(m.users, userID)
	return nil
}

func (m *mockUserRepository) GetBehaviors(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error) {
	return nil, nil
}

func (m *mockUserRepository) AddBehavior(ctx context.Context, behavior *interfaces.UserBehavior) error {
	return nil
}

func (m *mockUserRepository) GetUserItemInteractions(ctx context.Context, userID, itemID string) ([]*interfaces.UserBehavior, error) {
	return nil, nil
}

// mockItemRepository Mock 物品仓库
type mockItemRepository struct {
	items map[string]*interfaces.Item
}

func newMockItemRepository() *mockItemRepository {
	return &mockItemRepository{
		items: make(map[string]*interfaces.Item),
	}
}

func (m *mockItemRepository) GetByID(ctx context.Context, itemID string) (*interfaces.Item, error) {
	if item, ok := m.items[itemID]; ok {
		return item, nil
	}
	return nil, errors.New("item not found")
}

func (m *mockItemRepository) GetByIDs(ctx context.Context, itemIDs []string) ([]*interfaces.Item, error) {
	var items []*interfaces.Item
	for _, id := range itemIDs {
		if item, ok := m.items[id]; ok {
			items = append(items, item)
		}
	}
	return items, nil
}

func (m *mockItemRepository) Create(ctx context.Context, item *interfaces.Item) error {
	m.items[item.ID] = item
	return nil
}

func (m *mockItemRepository) Update(ctx context.Context, item *interfaces.Item) error {
	m.items[item.ID] = item
	return nil
}

func (m *mockItemRepository) Delete(ctx context.Context, itemID string) error {
	delete(m.items, itemID)
	return nil
}

func (m *mockItemRepository) List(ctx context.Context, itemType, category string, page, pageSize int) ([]*interfaces.Item, int64, error) {
	var items []*interfaces.Item
	for _, item := range m.items {
		if (itemType == "" || item.Type == itemType) && (category == "" || item.Category == category) {
			items = append(items, item)
		}
	}
	return items, int64(len(items)), nil
}

func (m *mockItemRepository) Search(ctx context.Context, query string, limit int) ([]*interfaces.Item, error) {
	return nil, nil
}

func (m *mockItemRepository) GetStats(ctx context.Context, itemID string) (*interfaces.ItemStats, error) {
	return nil, nil
}

func (m *mockItemRepository) IncrementStats(ctx context.Context, itemID, action string) error {
	return nil
}

func (m *mockItemRepository) GetPopularByCategories(ctx context.Context, categories []string, limit int) ([]*interfaces.Item, error) {
	var items []*interfaces.Item
	categorySet := make(map[string]bool)
	for _, cat := range categories {
		categorySet[cat] = true
	}

	for _, item := range m.items {
		if categorySet[item.Category] {
			items = append(items, item)
			if len(items) >= limit {
				break
			}
		}
	}
	return items, nil
}

// mockCache Mock 缓存
type mockCache struct {
	data map[string]interface{}
}

func newMockCache() *mockCache {
	return &mockCache{
		data: make(map[string]interface{}),
	}
}

func (m *mockCache) Get(ctx context.Context, key string, value interface{}) error {
	if data, ok := m.data[key]; ok {
		bytes, _ := json.Marshal(data)
		return json.Unmarshal(bytes, value)
	}
	return errors.New("not found")
}

func (m *mockCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	m.data[key] = value
	return nil
}

func (m *mockCache) Delete(ctx context.Context, key string) error {
	delete(m.data, key)
	return nil
}

func (m *mockCache) Exists(ctx context.Context, key string) (bool, error) {
	_, ok := m.data[key]
	return ok, nil
}

func (m *mockCache) MGet(ctx context.Context, keys []string) ([]interface{}, error) {
	var result []interface{}
	for _, key := range keys {
		result = append(result, m.data[key])
	}
	return result, nil
}

func (m *mockCache) MSet(ctx context.Context, kvs map[string]interface{}, ttl time.Duration) error {
	for k, v := range kvs {
		m.data[k] = v
	}
	return nil
}

// =============================================================================
// Service 创建测试
// =============================================================================

func TestNewService(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()
	logger := zap.NewNop()

	service := NewService(cfg, llm, userRepo, itemRepo, cache, logger)

	if service == nil {
		t.Fatal("service should not be nil")
	}
}

func TestNewServiceWithNilLogger(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	// 使用 nil logger，应该不会 panic
	service := NewService(cfg, llm, userRepo, itemRepo, cache, nil)

	if service == nil {
		t.Fatal("service should not be nil")
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.CacheTTL == 0 {
		t.Error("CacheTTL should have default value")
	}
	if cfg.MaxRecommendations == 0 {
		t.Error("MaxRecommendations should have default value")
	}
	if cfg.LLMTimeout == 0 {
		t.Error("LLMTimeout should have default value")
	}
	if len(cfg.DefaultCategories) == 0 {
		t.Error("DefaultCategories should have default values")
	}
}

// =============================================================================
// HandleNewUser 测试
// =============================================================================

func TestService_HandleNewUser(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{
		chatFunc: func(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
			return `{
				"preferred_categories": ["科技", "游戏"],
				"preferred_tags": ["新品", "热门"],
				"content_preference": "medium",
				"price_sensitivity": "low"
			}`, nil
		},
	}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	// 添加一些测试物品
	itemRepo.items["item1"] = &interfaces.Item{ID: "item1", Title: "科技新品", Category: "科技"}
	itemRepo.items["item2"] = &interfaces.Item{ID: "item2", Title: "游戏大作", Category: "游戏"}

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	user := &interfaces.User{
		ID:     "user1",
		Name:   "Test User",
		Age:    25,
		Gender: "male",
	}

	ctx := context.Background()
	result, err := service.HandleNewUser(ctx, user)

	if err != nil {
		t.Fatalf("HandleNewUser() error = %v", err)
	}

	if result == nil {
		t.Fatal("result should not be nil")
	}

	if result.UserID != user.ID {
		t.Errorf("UserID = %s, want %s", result.UserID, user.ID)
	}

	if result.Strategy != "llm_based" {
		t.Errorf("Strategy = %s, want 'llm_based'", result.Strategy)
	}

	if result.Preferences == nil {
		t.Error("Preferences should not be nil")
	}
}

func TestService_HandleNewUser_NilUser(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	ctx := context.Background()
	_, err := service.HandleNewUser(ctx, nil)

	if err == nil {
		t.Error("expected error for nil user")
	}
}

func TestService_HandleNewUser_LLMFailure_Fallback(t *testing.T) {
	cfg := DefaultConfig()
	cfg.EnableLLMFallback = true

	llm := &mockLLMClient{
		chatFunc: func(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
			return "", errors.New("LLM error")
		},
	}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	user := &interfaces.User{
		ID:     "user1",
		Name:   "Test User",
		Age:    25,
		Gender: "male",
	}

	ctx := context.Background()
	result, err := service.HandleNewUser(ctx, user)

	if err != nil {
		t.Fatalf("HandleNewUser() should fallback, got error = %v", err)
	}

	if result.Strategy != "demographic_fallback" {
		t.Errorf("Strategy = %s, want 'demographic_fallback'", result.Strategy)
	}
}

func TestService_HandleNewUser_LLMFailure_NoFallback(t *testing.T) {
	cfg := DefaultConfig()
	cfg.EnableLLMFallback = false

	llm := &mockLLMClient{
		chatFunc: func(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
			return "", errors.New("LLM error")
		},
	}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	user := &interfaces.User{
		ID:   "user1",
		Name: "Test User",
	}

	ctx := context.Background()
	_, err := service.HandleNewUser(ctx, user)

	if err == nil {
		t.Error("expected error when fallback is disabled")
	}
}

// =============================================================================
// HandleNewItem 测试
// =============================================================================

func TestService_HandleNewItem(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{
		chatFunc: func(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
			return `{
				"main_category": "科技",
				"sub_categories": ["电子产品", "智能设备"],
				"target_audience": ["年轻人", "科技爱好者"],
				"content_type": "产品",
				"quality_score": 0.85
			}`, nil
		},
		embedFunc: func(ctx context.Context, text string) ([]float32, error) {
			return make([]float32, 768), nil
		},
	}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	// 添加一些同类物品
	itemRepo.items["similar1"] = &interfaces.Item{ID: "similar1", Title: "类似产品", Category: "科技"}

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	item := &interfaces.Item{
		ID:          "item1",
		Title:       "新款智能手机",
		Description: "高性能智能手机",
		Category:    "科技",
		Tags:        []string{"手机", "智能"},
	}

	ctx := context.Background()
	result, err := service.HandleNewItem(ctx, item)

	if err != nil {
		t.Fatalf("HandleNewItem() error = %v", err)
	}

	if result == nil {
		t.Fatal("result should not be nil")
	}

	if result.ItemID != item.ID {
		t.Errorf("ItemID = %s, want %s", result.ItemID, item.ID)
	}

	if result.Strategy != "llm_based" {
		t.Errorf("Strategy = %s, want 'llm_based'", result.Strategy)
	}

	if result.Features == nil {
		t.Error("Features should not be nil")
	}
}

func TestService_HandleNewItem_NilItem(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	ctx := context.Background()
	_, err := service.HandleNewItem(ctx, nil)

	if err == nil {
		t.Error("expected error for nil item")
	}
}

func TestService_HandleNewItem_LLMFailure_Fallback(t *testing.T) {
	cfg := DefaultConfig()
	cfg.EnableLLMFallback = true

	llm := &mockLLMClient{
		chatFunc: func(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
			return "", errors.New("LLM error")
		},
	}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	item := &interfaces.Item{
		ID:       "item1",
		Title:    "测试物品",
		Category: "测试",
	}

	ctx := context.Background()
	result, err := service.HandleNewItem(ctx, item)

	if err != nil {
		t.Fatalf("HandleNewItem() should fallback, got error = %v", err)
	}

	if result.Strategy != "fallback" {
		t.Errorf("Strategy = %s, want 'fallback'", result.Strategy)
	}
}

// =============================================================================
// GetColdStartRecommendations 测试
// =============================================================================

func TestService_GetColdStartRecommendations(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{
		chatFunc: func(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
			return `{"preferred_categories": ["科技"]}`, nil
		},
	}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	// 添加用户和物品
	userRepo.users["user1"] = &interfaces.User{ID: "user1", Age: 25}
	itemRepo.items["item1"] = &interfaces.Item{ID: "item1", Title: "科技产品", Category: "科技"}
	itemRepo.items["item2"] = &interfaces.Item{ID: "item2", Title: "另一个产品", Category: "科技"}

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	ctx := context.Background()
	items, err := service.GetColdStartRecommendations(ctx, "user1", 10)

	if err != nil {
		t.Fatalf("GetColdStartRecommendations() error = %v", err)
	}

	// 应该返回物品（具体数量取决于仓库中匹配的物品）
	if items == nil {
		t.Error("items should not be nil")
	}
}

func TestService_GetColdStartRecommendations_EmptyUserID(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	ctx := context.Background()
	_, err := service.GetColdStartRecommendations(ctx, "", 10)

	if err == nil {
		t.Error("expected error for empty user ID")
	}
}

func TestService_GetColdStartRecommendations_FromCache(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	// 预先缓存结果
	cachedResult := &interfaces.ColdStartResult{
		UserID:          "user1",
		Recommendations: []string{"cached_item1", "cached_item2"},
		Strategy:        "cached",
	}
	cache.data["coldstart:user:user1"] = cachedResult

	// 添加物品
	itemRepo.items["cached_item1"] = &interfaces.Item{ID: "cached_item1", Title: "缓存物品1"}
	itemRepo.items["cached_item2"] = &interfaces.Item{ID: "cached_item2", Title: "缓存物品2"}

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	ctx := context.Background()
	items, err := service.GetColdStartRecommendations(ctx, "user1", 10)

	if err != nil {
		t.Fatalf("GetColdStartRecommendations() error = %v", err)
	}

	// 应该从缓存返回
	if len(items) != 2 {
		t.Errorf("len(items) = %d, want 2", len(items))
	}
}

func TestService_GetColdStartRecommendations_LimitBounds(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxRecommendations = 50

	llm := &mockLLMClient{
		chatFunc: func(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
			return `{"preferred_categories": ["科技"]}`, nil
		},
	}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	userRepo.users["user1"] = &interfaces.User{ID: "user1"}

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	ctx := context.Background()

	// 测试负数 limit
	_, err := service.GetColdStartRecommendations(ctx, "user1", -1)
	if err != nil {
		t.Errorf("should handle negative limit, got error = %v", err)
	}

	// 测试超大 limit
	_, err = service.GetColdStartRecommendations(ctx, "user1", 1000)
	if err != nil {
		t.Errorf("should handle large limit, got error = %v", err)
	}
}

// =============================================================================
// ExplainRecommendation 测试
// =============================================================================

func TestService_ExplainRecommendation(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{
		completeFunc: func(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
			return "根据您对科技产品的兴趣，为您推荐这款智能手机。", nil
		},
	}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	// 添加用户和物品
	userRepo.users["user1"] = &interfaces.User{ID: "user1", Age: 25, Gender: "male"}
	itemRepo.items["item1"] = &interfaces.Item{
		ID:          "item1",
		Title:       "智能手机",
		Category:    "科技",
		Description: "最新款智能手机",
	}

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	ctx := context.Background()
	explanation, err := service.ExplainRecommendation(ctx, "user1", "item1")

	if err != nil {
		t.Fatalf("ExplainRecommendation() error = %v", err)
	}

	if explanation == "" {
		t.Error("explanation should not be empty")
	}
}

func TestService_ExplainRecommendation_EmptyIDs(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	ctx := context.Background()

	// 空用户 ID
	_, err := service.ExplainRecommendation(ctx, "", "item1")
	if err == nil {
		t.Error("expected error for empty user ID")
	}

	// 空物品 ID
	_, err = service.ExplainRecommendation(ctx, "user1", "")
	if err == nil {
		t.Error("expected error for empty item ID")
	}
}

func TestService_ExplainRecommendation_LLMFailure(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{
		completeFunc: func(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
			return "", errors.New("LLM error")
		},
	}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	userRepo.users["user1"] = &interfaces.User{ID: "user1"}
	itemRepo.items["item1"] = &interfaces.Item{ID: "item1", Title: "测试", Category: "测试"}

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	ctx := context.Background()
	explanation, err := service.ExplainRecommendation(ctx, "user1", "item1")

	// 应该返回默认解释，不返回错误
	if err != nil {
		t.Fatalf("ExplainRecommendation() should return default, got error = %v", err)
	}

	if explanation == "" {
		t.Error("should return default explanation")
	}
}

// =============================================================================
// 辅助函数测试
// =============================================================================

func TestExtractJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "clean JSON",
			input:    `{"key": "value"}`,
			expected: `{"key": "value"}`,
		},
		{
			name:     "JSON with prefix",
			input:    `Here is the JSON: {"key": "value"}`,
			expected: `{"key": "value"}`,
		},
		{
			name:     "JSON with suffix",
			input:    `{"key": "value"} and some more text`,
			expected: `{"key": "value"}`,
		},
		{
			name:     "JSON with whitespace",
			input:    `   {"key": "value"}   `,
			expected: `{"key": "value"}`,
		},
		{
			name:     "no JSON",
			input:    "no json here",
			expected: "no json here",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractJSON(tt.input)
			if result != tt.expected {
				t.Errorf("extractJSON() = %s, want %s", result, tt.expected)
			}
		})
	}
}

func TestService_GenerateDefaultPreferences(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	tests := []struct {
		name           string
		user           *interfaces.User
		expectedLength int
	}{
		{
			name:           "young user",
			user:           &interfaces.User{ID: "1", Age: 16},
			expectedLength: 3, // 动漫、游戏、学习
		},
		{
			name:           "young adult",
			user:           &interfaces.User{ID: "2", Age: 22},
			expectedLength: 4, // 科技、游戏、娱乐、时尚
		},
		{
			name:           "adult",
			user:           &interfaces.User{ID: "3", Age: 30},
			expectedLength: 4, // 商业、科技、生活、理财
		},
		{
			name:           "middle age",
			user:           &interfaces.User{ID: "4", Age: 45},
			expectedLength: 4, // 商业、新闻、健康、家庭
		},
		{
			name:           "senior",
			user:           &interfaces.User{ID: "5", Age: 60},
			expectedLength: 4, // 健康、新闻、文化、养生
		},
		{
			name:           "unknown age",
			user:           &interfaces.User{ID: "6", Age: 0},
			expectedLength: 3, // 默认类别
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prefs := service.generateDefaultPreferences(tt.user)

			cats, ok := prefs["preferred_categories"].([]string)
			if !ok {
				t.Fatal("preferred_categories should be []string")
			}

			if len(cats) != tt.expectedLength {
				t.Errorf("len(categories) = %d, want %d", len(cats), tt.expectedLength)
			}
		})
	}
}

func TestService_ExtractCategories(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	tests := []struct {
		name        string
		preferences map[string]interface{}
		expected    int
	}{
		{
			name: "with categories",
			preferences: map[string]interface{}{
				"preferred_categories": []interface{}{"科技", "游戏"},
			},
			expected: 2,
		},
		{
			name:        "without categories",
			preferences: map[string]interface{}{},
			expected:    0,
		},
		{
			name: "wrong type",
			preferences: map[string]interface{}{
				"preferred_categories": "not an array",
			},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cats := service.extractCategories(tt.preferences)
			if len(cats) != tt.expected {
				t.Errorf("len(categories) = %d, want %d", len(cats), tt.expected)
			}
		})
	}
}

// =============================================================================
// 接口兼容性测试
// =============================================================================

func TestService_ImplementsInterface(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	// 编译时检查接口实现
	var _ interfaces.ColdStartService = service
}

// =============================================================================
// 缓存测试
// =============================================================================

func TestService_CacheInteraction(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{
		chatFunc: func(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
			return `{"preferred_categories": ["科技"]}`, nil
		},
	}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()
	cache := newMockCache()

	service := NewService(cfg, llm, userRepo, itemRepo, cache, zap.NewNop())

	user := &interfaces.User{ID: "user1", Age: 25}

	ctx := context.Background()
	result, err := service.HandleNewUser(ctx, user)

	if err != nil {
		t.Fatalf("HandleNewUser() error = %v", err)
	}

	// 验证结果已缓存
	cacheKey := "coldstart:user:user1"
	if _, ok := cache.data[cacheKey]; !ok {
		t.Error("result should be cached")
	}

	// 验证缓存的结果与返回的结果一致
	var cached interfaces.ColdStartResult
	if err := cache.Get(ctx, cacheKey, &cached); err != nil {
		t.Fatalf("failed to get cached result: %v", err)
	}

	if cached.UserID != result.UserID {
		t.Error("cached result should match returned result")
	}
}

func TestService_NilCache(t *testing.T) {
	cfg := DefaultConfig()
	llm := &mockLLMClient{
		chatFunc: func(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
			return `{"preferred_categories": ["科技"]}`, nil
		},
	}
	userRepo := newMockUserRepository()
	itemRepo := newMockItemRepository()

	// 使用 nil 缓存
	service := NewService(cfg, llm, userRepo, itemRepo, nil, zap.NewNop())

	user := &interfaces.User{ID: "user1", Age: 25}

	ctx := context.Background()
	result, err := service.HandleNewUser(ctx, user)

	// 应该正常工作，只是不缓存
	if err != nil {
		t.Fatalf("HandleNewUser() should work without cache, got error = %v", err)
	}

	if result == nil {
		t.Error("result should not be nil")
	}
}

