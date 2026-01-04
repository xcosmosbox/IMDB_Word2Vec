package feature

import (
	"context"
	"testing"
	"time"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// Mock 实现
// =============================================================================

// MockUserRepository 用户仓库 Mock
type MockUserRepository struct {
	users         map[string]*interfaces.User
	behaviors     map[string][]*interfaces.UserBehavior
	interactions  map[string][]*interfaces.UserBehavior
}

func NewMockUserRepository() *MockUserRepository {
	return &MockUserRepository{
		users:        make(map[string]*interfaces.User),
		behaviors:    make(map[string][]*interfaces.UserBehavior),
		interactions: make(map[string][]*interfaces.UserBehavior),
	}
}

func (m *MockUserRepository) GetByID(ctx context.Context, userID string) (*interfaces.User, error) {
	if user, ok := m.users[userID]; ok {
		return user, nil
	}
	return nil, ErrUserNotFound
}

func (m *MockUserRepository) GetByIDs(ctx context.Context, userIDs []string) ([]*interfaces.User, error) {
	result := make([]*interfaces.User, 0)
	for _, id := range userIDs {
		if user, ok := m.users[id]; ok {
			result = append(result, user)
		}
	}
	return result, nil
}

func (m *MockUserRepository) Create(ctx context.Context, user *interfaces.User) error {
	m.users[user.ID] = user
	return nil
}

func (m *MockUserRepository) Update(ctx context.Context, user *interfaces.User) error {
	m.users[user.ID] = user
	return nil
}

func (m *MockUserRepository) Delete(ctx context.Context, userID string) error {
	delete(m.users, userID)
	return nil
}

func (m *MockUserRepository) GetBehaviors(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error) {
	if behaviors, ok := m.behaviors[userID]; ok {
		if limit > 0 && limit < len(behaviors) {
			return behaviors[:limit], nil
		}
		return behaviors, nil
	}
	return []*interfaces.UserBehavior{}, nil
}

func (m *MockUserRepository) AddBehavior(ctx context.Context, behavior *interfaces.UserBehavior) error {
	m.behaviors[behavior.UserID] = append(m.behaviors[behavior.UserID], behavior)
	return nil
}

func (m *MockUserRepository) GetUserItemInteractions(ctx context.Context, userID, itemID string) ([]*interfaces.UserBehavior, error) {
	key := userID + ":" + itemID
	if interactions, ok := m.interactions[key]; ok {
		return interactions, nil
	}
	return []*interfaces.UserBehavior{}, nil
}

// MockItemRepository 物品仓库 Mock
type MockItemRepository struct {
	items map[string]*interfaces.Item
	stats map[string]*interfaces.ItemStats
}

func NewMockItemRepository() *MockItemRepository {
	return &MockItemRepository{
		items: make(map[string]*interfaces.Item),
		stats: make(map[string]*interfaces.ItemStats),
	}
}

func (m *MockItemRepository) GetByID(ctx context.Context, itemID string) (*interfaces.Item, error) {
	if item, ok := m.items[itemID]; ok {
		return item, nil
	}
	return nil, ErrItemNotFound
}

func (m *MockItemRepository) GetByIDs(ctx context.Context, itemIDs []string) ([]*interfaces.Item, error) {
	result := make([]*interfaces.Item, 0)
	for _, id := range itemIDs {
		if item, ok := m.items[id]; ok {
			result = append(result, item)
		}
	}
	return result, nil
}

func (m *MockItemRepository) Create(ctx context.Context, item *interfaces.Item) error {
	m.items[item.ID] = item
	return nil
}

func (m *MockItemRepository) Update(ctx context.Context, item *interfaces.Item) error {
	m.items[item.ID] = item
	return nil
}

func (m *MockItemRepository) Delete(ctx context.Context, itemID string) error {
	delete(m.items, itemID)
	return nil
}

func (m *MockItemRepository) List(ctx context.Context, itemType, category string, page, pageSize int) ([]*interfaces.Item, int64, error) {
	result := make([]*interfaces.Item, 0)
	for _, item := range m.items {
		if (itemType == "" || item.Type == itemType) && (category == "" || item.Category == category) {
			result = append(result, item)
		}
	}
	return result, int64(len(result)), nil
}

func (m *MockItemRepository) Search(ctx context.Context, query string, limit int) ([]*interfaces.Item, error) {
	return []*interfaces.Item{}, nil
}

func (m *MockItemRepository) GetStats(ctx context.Context, itemID string) (*interfaces.ItemStats, error) {
	if stats, ok := m.stats[itemID]; ok {
		return stats, nil
	}
	return &interfaces.ItemStats{}, nil
}

func (m *MockItemRepository) IncrementStats(ctx context.Context, itemID, action string) error {
	return nil
}

func (m *MockItemRepository) GetPopularByCategories(ctx context.Context, categories []string, limit int) ([]*interfaces.Item, error) {
	return []*interfaces.Item{}, nil
}

// =============================================================================
// 特征提取器测试
// =============================================================================

func TestNewFeatureExtractor(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()

	extractor := NewFeatureExtractor(userRepo, itemRepo)

	if extractor == nil {
		t.Error("extractor should not be nil")
	}
	if extractor.userRepo == nil {
		t.Error("user repo should not be nil")
	}
	if extractor.itemRepo == nil {
		t.Error("item repo should not be nil")
	}
}

func TestExtractUserFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()

	// 添加测试用户
	userRepo.users["user1"] = &interfaces.User{
		ID:     "user1",
		Name:   "Test User",
		Email:  "test@example.com",
		Age:    28,
		Gender: "male",
		Metadata: map[string]string{
			"location": "Beijing",
			"device":   "mobile",
		},
	}

	// 添加用户行为
	now := time.Now()
	userRepo.behaviors["user1"] = []*interfaces.UserBehavior{
		{UserID: "user1", ItemID: "item1", Action: "view", Timestamp: now},
		{UserID: "user1", ItemID: "item2", Action: "click", Timestamp: now.Add(-1 * time.Hour)},
		{UserID: "user1", ItemID: "item3", Action: "buy", Timestamp: now.Add(-2 * time.Hour)},
	}

	extractor := NewFeatureExtractor(userRepo, itemRepo)
	ctx := context.Background()

	features, err := extractor.ExtractUserFeatures(ctx, "user1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if features == nil {
		t.Fatal("features should not be nil")
	}

	// 验证用户 ID
	if features.UserID != "user1" {
		t.Errorf("expected user ID user1, got %s", features.UserID)
	}

	// 验证人口统计特征
	if features.Demographics.Age != 28 {
		t.Errorf("expected age 28, got %d", features.Demographics.Age)
	}
	if features.Demographics.Gender != "male" {
		t.Errorf("expected gender male, got %s", features.Demographics.Gender)
	}
	if features.Demographics.Location != "Beijing" {
		t.Errorf("expected location Beijing, got %s", features.Demographics.Location)
	}

	// 验证行为特征
	if features.Behavior.TotalViews != 1 {
		t.Errorf("expected 1 view, got %d", features.Behavior.TotalViews)
	}
	if features.Behavior.TotalClicks != 1 {
		t.Errorf("expected 1 click, got %d", features.Behavior.TotalClicks)
	}
	if features.Behavior.TotalPurchases != 1 {
		t.Errorf("expected 1 purchase, got %d", features.Behavior.TotalPurchases)
	}
}

func TestExtractUserFeaturesNotFound(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()

	extractor := NewFeatureExtractor(userRepo, itemRepo)
	ctx := context.Background()

	_, err := extractor.ExtractUserFeatures(ctx, "nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent user")
	}
}

func TestExtractItemFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()

	// 添加测试物品
	itemRepo.items["item1"] = &interfaces.Item{
		ID:          "item1",
		Type:        "movie",
		Title:       "Test Movie",
		Description: "A test movie",
		Category:    "action",
		Tags:        []string{"thriller", "sci-fi"},
		Metadata: map[string]interface{}{
			"sub_category": "superhero",
			"duration":     float64(7200),
			"release_date": "2024-01-15",
		},
	}

	// 添加物品统计
	itemRepo.stats["item1"] = &interfaces.ItemStats{
		ItemID:     "item1",
		ViewCount:  10000,
		ClickCount: 5000,
		LikeCount:  2000,
		ShareCount: 500,
		AvgRating:  4.5,
	}

	extractor := NewFeatureExtractor(userRepo, itemRepo)
	ctx := context.Background()

	features, err := extractor.ExtractItemFeatures(ctx, "item1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if features == nil {
		t.Fatal("features should not be nil")
	}

	// 验证物品 ID
	if features.ItemID != "item1" {
		t.Errorf("expected item ID item1, got %s", features.ItemID)
	}

	// 验证类型
	if features.Type != "movie" {
		t.Errorf("expected type movie, got %s", features.Type)
	}

	// 验证内容特征
	if features.Content.Category != "action" {
		t.Errorf("expected category action, got %s", features.Content.Category)
	}
	if features.Content.SubCategory != "superhero" {
		t.Errorf("expected sub category superhero, got %s", features.Content.SubCategory)
	}
	if len(features.Content.Tags) != 2 {
		t.Errorf("expected 2 tags, got %d", len(features.Content.Tags))
	}

	// 验证统计特征
	if features.Statistics.ViewCount != 10000 {
		t.Errorf("expected view count 10000, got %d", features.Statistics.ViewCount)
	}
	if features.Statistics.ClickCount != 5000 {
		t.Errorf("expected click count 5000, got %d", features.Statistics.ClickCount)
	}
	if features.Statistics.CTR != 0.5 {
		t.Errorf("expected CTR 0.5, got %f", features.Statistics.CTR)
	}
}

func TestExtractItemFeaturesNotFound(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()

	extractor := NewFeatureExtractor(userRepo, itemRepo)
	ctx := context.Background()

	_, err := extractor.ExtractItemFeatures(ctx, "nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent item")
	}
}

func TestExtractCrossFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()

	// 添加用户-物品交互
	now := time.Now()
	userRepo.interactions["user1:item1"] = []*interfaces.UserBehavior{
		{UserID: "user1", ItemID: "item1", Action: "click", Timestamp: now},
		{UserID: "user1", ItemID: "item1", Action: "view", Timestamp: now.Add(-1 * time.Hour)},
	}

	extractor := NewFeatureExtractor(userRepo, itemRepo)
	ctx := context.Background()

	features, err := extractor.ExtractCrossFeatures(ctx, "user1", "item1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if features == nil {
		t.Fatal("features should not be nil")
	}

	if features.UserID != "user1" {
		t.Errorf("expected user ID user1, got %s", features.UserID)
	}
	if features.ItemID != "item1" {
		t.Errorf("expected item ID item1, got %s", features.ItemID)
	}
	if features.Interactions != 2 {
		t.Errorf("expected 2 interactions, got %d", features.Interactions)
	}
	if features.LastAction != "click" {
		t.Errorf("expected last action click, got %s", features.LastAction)
	}
}

func TestExtractContextFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()

	extractor := NewFeatureExtractor(userRepo, itemRepo)
	ctx := context.Background()

	req := &ContextRequest{
		Device:      "mobile",
		OS:          "iOS",
		Location:    "Shanghai",
		PageContext: "home",
	}

	features := extractor.ExtractContextFeatures(ctx, req)
	if features == nil {
		t.Fatal("features should not be nil")
	}

	if features.Device != "mobile" {
		t.Errorf("expected device mobile, got %s", features.Device)
	}
	if features.OS != "iOS" {
		t.Errorf("expected OS iOS, got %s", features.OS)
	}
	if features.Location != "Shanghai" {
		t.Errorf("expected location Shanghai, got %s", features.Location)
	}
	if features.PageContext != "home" {
		t.Errorf("expected page context home, got %s", features.PageContext)
	}

	// 验证时间相关特征
	now := time.Now()
	if features.Hour != now.Hour() {
		t.Errorf("expected hour %d, got %d", now.Hour(), features.Hour)
	}
	if features.DayOfWeek != int(now.Weekday()) {
		t.Errorf("expected day of week %d, got %d", int(now.Weekday()), features.DayOfWeek)
	}
}

func TestExtractContextFeaturesNilRequest(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()

	extractor := NewFeatureExtractor(userRepo, itemRepo)
	ctx := context.Background()

	features := extractor.ExtractContextFeatures(ctx, nil)
	if features == nil {
		t.Fatal("features should not be nil even with nil request")
	}

	// 应该有默认的时间特征
	now := time.Now()
	if features.Hour != now.Hour() {
		t.Errorf("expected hour %d, got %d", now.Hour(), features.Hour)
	}
}

func TestGetActionWeight(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	extractor := NewFeatureExtractor(userRepo, itemRepo)

	tests := []struct {
		action   string
		expected float64
	}{
		{"view", 1.0},
		{"click", 2.0},
		{"like", 3.0},
		{"favorite", 4.0},
		{"share", 4.0},
		{"comment", 3.5},
		{"buy", 5.0},
		{"purchase", 5.0},
		{"rate", 4.0},
		{"unknown", 1.0}, // 默认权重
	}

	for _, tt := range tests {
		t.Run(tt.action, func(t *testing.T) {
			weight := extractor.getActionWeight(tt.action)
			if weight != tt.expected {
				t.Errorf("expected weight %f for action %s, got %f", tt.expected, tt.action, weight)
			}
		})
	}
}

func TestCalculateCTR(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	extractor := NewFeatureExtractor(userRepo, itemRepo)

	tests := []struct {
		name       string
		stats      *interfaces.ItemStats
		expectedCTR float64
	}{
		{
			name:        "Normal CTR",
			stats:       &interfaces.ItemStats{ViewCount: 1000, ClickCount: 100},
			expectedCTR: 0.1,
		},
		{
			name:        "Zero views",
			stats:       &interfaces.ItemStats{ViewCount: 0, ClickCount: 0},
			expectedCTR: 0,
		},
		{
			name:        "High CTR",
			stats:       &interfaces.ItemStats{ViewCount: 100, ClickCount: 50},
			expectedCTR: 0.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctr := extractor.calculateCTR(tt.stats)
			if ctr != tt.expectedCTR {
				t.Errorf("expected CTR %f, got %f", tt.expectedCTR, ctr)
			}
		})
	}
}

func TestBatchExtractUserFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()

	// 添加测试用户
	userRepo.users["user1"] = &interfaces.User{ID: "user1", Name: "User 1", Age: 25, Gender: "male"}
	userRepo.users["user2"] = &interfaces.User{ID: "user2", Name: "User 2", Age: 30, Gender: "female"}

	extractor := NewFeatureExtractor(userRepo, itemRepo)
	ctx := context.Background()

	result, err := extractor.BatchExtractUserFeatures(ctx, []string{"user1", "user2", "user3"})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// 应该有 2 个成功的结果（user3 不存在）
	if len(result) != 2 {
		t.Errorf("expected 2 results, got %d", len(result))
	}
	if _, ok := result["user1"]; !ok {
		t.Error("user1 should be in result")
	}
	if _, ok := result["user2"]; !ok {
		t.Error("user2 should be in result")
	}
}

func TestBatchExtractItemFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()

	// 添加测试物品
	itemRepo.items["item1"] = &interfaces.Item{ID: "item1", Type: "movie", Category: "action"}
	itemRepo.items["item2"] = &interfaces.Item{ID: "item2", Type: "book", Category: "fiction"}

	extractor := NewFeatureExtractor(userRepo, itemRepo)
	ctx := context.Background()

	result, err := extractor.BatchExtractItemFeatures(ctx, []string{"item1", "item2", "item3"})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// 应该有 2 个成功的结果（item3 不存在）
	if len(result) != 2 {
		t.Errorf("expected 2 results, got %d", len(result))
	}
	if _, ok := result["item1"]; !ok {
		t.Error("item1 should be in result")
	}
	if _, ok := result["item2"]; !ok {
		t.Error("item2 should be in result")
	}
}

