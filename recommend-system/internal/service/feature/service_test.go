package feature

import (
	"context"
	"testing"
	"time"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// 特征服务测试
// =============================================================================

func TestNewService(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	service := NewService(userRepo, itemRepo, cache)

	if service == nil {
		t.Error("service should not be nil")
	}
	if service.extractor == nil {
		t.Error("extractor should not be nil")
	}
	if service.store == nil {
		t.Error("store should not be nil")
	}
}

func TestNewServiceWithConfig(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()
	config := &FeatureCacheConfig{
		UserFeatureTTL: 15 * time.Minute,
		ItemFeatureTTL: 45 * time.Minute,
	}

	service := NewServiceWithConfig(userRepo, itemRepo, cache, config)

	if service == nil {
		t.Error("service should not be nil")
	}
}

func TestServiceGetUserFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	// 添加测试用户
	userRepo.users["user1"] = &interfaces.User{
		ID:     "user1",
		Name:   "Test User",
		Age:    25,
		Gender: "male",
		Metadata: map[string]string{
			"location": "Beijing",
		},
	}

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	// 首次获取（应从数据库提取）
	features, err := service.GetUserFeatures(ctx, "user1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if features == nil {
		t.Fatal("features should not be nil")
	}
	if features.UserID != "user1" {
		t.Errorf("expected user ID user1, got %s", features.UserID)
	}

	// 等待异步缓存完成
	time.Sleep(100 * time.Millisecond)

	// 再次获取（应从缓存获取）
	cachedFeatures, err := service.GetUserFeatures(ctx, "user1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if cachedFeatures == nil {
		t.Fatal("cached features should not be nil")
	}
}

func TestServiceGetUserFeaturesNotFound(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	_, err := service.GetUserFeatures(ctx, "nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent user")
	}
}

func TestServiceGetItemFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	// 添加测试物品
	itemRepo.items["item1"] = &interfaces.Item{
		ID:       "item1",
		Type:     "movie",
		Title:    "Test Movie",
		Category: "action",
		Tags:     []string{"thriller"},
	}
	itemRepo.stats["item1"] = &interfaces.ItemStats{
		ItemID:     "item1",
		ViewCount:  1000,
		ClickCount: 500,
	}

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	features, err := service.GetItemFeatures(ctx, "item1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if features == nil {
		t.Fatal("features should not be nil")
	}
	if features.ItemID != "item1" {
		t.Errorf("expected item ID item1, got %s", features.ItemID)
	}
	if features.Type != "movie" {
		t.Errorf("expected type movie, got %s", features.Type)
	}
}

func TestServiceGetItemFeaturesNotFound(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	_, err := service.GetItemFeatures(ctx, "nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent item")
	}
}

func TestServiceGetFeatureVector(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	// 添加测试数据
	userRepo.users["user1"] = &interfaces.User{
		ID:     "user1",
		Age:    25,
		Gender: "male",
	}
	itemRepo.items["item1"] = &interfaces.Item{
		ID:       "item1",
		Type:     "movie",
		Category: "action",
	}

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	req := &interfaces.FeatureVectorRequest{
		UserID: "user1",
		ItemID: "item1",
		Context: map[string]string{
			"device":       "mobile",
			"os":           "iOS",
			"location":     "Beijing",
			"page_context": "home",
		},
	}

	vector, err := service.GetFeatureVector(ctx, req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if vector == nil {
		t.Fatal("vector should not be nil")
	}
	if vector.UserFeatures == nil {
		t.Error("user features should not be nil")
	}
	if vector.ItemFeatures == nil {
		t.Error("item features should not be nil")
	}
	if len(vector.TokenIDs) == 0 {
		t.Error("token IDs should not be empty")
	}
	if len(vector.TokenTypes) == 0 {
		t.Error("token types should not be empty")
	}
	if len(vector.Positions) == 0 {
		t.Error("positions should not be empty")
	}
}

func TestServiceGetFeatureVectorNilRequest(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	_, err := service.GetFeatureVector(ctx, nil)
	if err == nil {
		t.Error("expected error for nil request")
	}
}

func TestServiceGetFeatureVectorEmptyUserID(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	req := &interfaces.FeatureVectorRequest{
		UserID: "",
	}

	_, err := service.GetFeatureVector(ctx, req)
	if err == nil {
		t.Error("expected error for empty user ID")
	}
}

func TestServiceGetFeatureVectorWithoutItem(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	userRepo.users["user1"] = &interfaces.User{
		ID:     "user1",
		Age:    25,
		Gender: "male",
	}

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	req := &interfaces.FeatureVectorRequest{
		UserID: "user1",
		// 不提供 ItemID
	}

	vector, err := service.GetFeatureVector(ctx, req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if vector == nil {
		t.Fatal("vector should not be nil")
	}
	if vector.UserFeatures == nil {
		t.Error("user features should not be nil")
	}
	if vector.ItemFeatures != nil {
		t.Error("item features should be nil when no item ID provided")
	}
}

func TestServiceBatchGetFeatureVectors(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	// 添加测试用户
	userRepo.users["user1"] = &interfaces.User{ID: "user1", Age: 25}
	userRepo.users["user2"] = &interfaces.User{ID: "user2", Age: 30}

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	reqs := []*interfaces.FeatureVectorRequest{
		{UserID: "user1"},
		{UserID: "user2"},
		{UserID: "user3"}, // 不存在
	}

	vectors, err := service.BatchGetFeatureVectors(ctx, reqs)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// 应该有 2 个成功的向量
	if len(vectors) != 2 {
		t.Errorf("expected 2 vectors, got %d", len(vectors))
	}
}

func TestServiceBatchGetFeatureVectorsEmpty(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	vectors, err := service.BatchGetFeatureVectors(ctx, []*interfaces.FeatureVectorRequest{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(vectors) != 0 {
		t.Errorf("expected 0 vectors, got %d", len(vectors))
	}
}

func TestServiceRefreshUserFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	userRepo.users["user1"] = &interfaces.User{
		ID:     "user1",
		Age:    25,
		Gender: "male",
	}

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	// 首先获取特征
	_, err := service.GetUserFeatures(ctx, "user1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// 刷新特征
	err = service.RefreshUserFeatures(ctx, "user1")
	if err != nil {
		t.Errorf("unexpected error refreshing: %v", err)
	}
}

func TestServiceRefreshUserFeaturesNotFound(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	err := service.RefreshUserFeatures(ctx, "nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent user")
	}
}

func TestServiceRefreshItemFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	itemRepo.items["item1"] = &interfaces.Item{
		ID:       "item1",
		Type:     "movie",
		Category: "action",
	}

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	// 首先获取特征
	_, err := service.GetItemFeatures(ctx, "item1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// 刷新特征
	err = service.RefreshItemFeatures(ctx, "item1")
	if err != nil {
		t.Errorf("unexpected error refreshing: %v", err)
	}
}

func TestServiceRefreshItemFeaturesNotFound(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	err := service.RefreshItemFeatures(ctx, "nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent item")
	}
}

// =============================================================================
// Token 序列化测试
// =============================================================================

func TestGetAgeToken(t *testing.T) {
	service := &Service{}

	tests := []struct {
		age      int
		expected int64
	}{
		{10, AgeTokenBase + 0},  // < 18
		{17, AgeTokenBase + 0},  // < 18
		{18, AgeTokenBase + 1},  // 18-25
		{25, AgeTokenBase + 1},  // 18-25
		{26, AgeTokenBase + 2},  // 26-35
		{35, AgeTokenBase + 2},  // 26-35
		{36, AgeTokenBase + 3},  // 36-45
		{45, AgeTokenBase + 3},  // 36-45
		{46, AgeTokenBase + 4},  // 46-55
		{55, AgeTokenBase + 4},  // 46-55
		{56, AgeTokenBase + 5},  // 56+
		{70, AgeTokenBase + 5},  // 56+
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			token := service.getAgeToken(tt.age)
			if token != tt.expected {
				t.Errorf("age %d: expected token %d, got %d", tt.age, tt.expected, token)
			}
		})
	}
}

func TestGetGenderToken(t *testing.T) {
	service := &Service{}

	tests := []struct {
		gender   string
		expected int64
	}{
		{"male", GenderTokenBase + 0},
		{"m", GenderTokenBase + 0},
		{"M", GenderTokenBase + 0},
		{"female", GenderTokenBase + 1},
		{"f", GenderTokenBase + 1},
		{"F", GenderTokenBase + 1},
		{"unknown", GenderTokenBase + 2},
		{"other", GenderTokenBase + 2},
		{"", GenderTokenBase + 2},
	}

	for _, tt := range tests {
		t.Run(tt.gender, func(t *testing.T) {
			token := service.getGenderToken(tt.gender)
			if token != tt.expected {
				t.Errorf("gender %s: expected token %d, got %d", tt.gender, tt.expected, token)
			}
		})
	}
}

func TestGetHourToken(t *testing.T) {
	service := &Service{}

	tests := []struct {
		hour     int
		expected int64
	}{
		{0, HourTokenBase + 0},  // night
		{3, HourTokenBase + 0},  // night
		{5, HourTokenBase + 0},  // night
		{6, HourTokenBase + 1},  // morning
		{9, HourTokenBase + 1},  // morning
		{11, HourTokenBase + 1}, // morning
		{12, HourTokenBase + 2}, // afternoon
		{15, HourTokenBase + 2}, // afternoon
		{17, HourTokenBase + 2}, // afternoon
		{18, HourTokenBase + 3}, // evening
		{21, HourTokenBase + 3}, // evening
		{23, HourTokenBase + 3}, // evening
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			token := service.getHourToken(tt.hour)
			if token != tt.expected {
				t.Errorf("hour %d: expected token %d, got %d", tt.hour, tt.expected, token)
			}
		})
	}
}

func TestSerializeToTokens(t *testing.T) {
	service := &Service{}

	vector := &interfaces.FeatureVector{
		UserFeatures: &interfaces.UserFeatures{
			UserID: "user1",
			Demographics: map[string]interface{}{
				"age":    float64(25),
				"gender": "male",
			},
		},
		ItemFeatures: &interfaces.ItemFeatures{
			ItemID:     "item1",
			SemanticID: [3]int{100, 200, 300},
		},
	}

	contextReq := &ContextRequest{
		Device:      "mobile",
		OS:          "iOS",
		Location:    "Beijing",
		PageContext: "home",
	}

	service.serializeToTokens(vector, contextReq)

	// 验证 Token IDs
	if len(vector.TokenIDs) == 0 {
		t.Error("token IDs should not be empty")
	}

	// 第一个应该是 [CLS]
	if vector.TokenIDs[0] != TokenIDCLS {
		t.Errorf("first token should be CLS (%d), got %d", TokenIDCLS, vector.TokenIDs[0])
	}

	// 最后一个应该是 [SEP]
	lastIdx := len(vector.TokenIDs) - 1
	if vector.TokenIDs[lastIdx] != TokenIDSEP {
		t.Errorf("last token should be SEP (%d), got %d", TokenIDSEP, vector.TokenIDs[lastIdx])
	}

	// 验证位置编码
	if len(vector.Positions) != len(vector.TokenIDs) {
		t.Errorf("positions length should match token IDs length")
	}
	for i, pos := range vector.Positions {
		if pos != i {
			t.Errorf("position %d should be %d, got %d", i, i, pos)
		}
	}

	// 验证 Token 类型
	if len(vector.TokenTypes) != len(vector.TokenIDs) {
		t.Errorf("token types length should match token IDs length")
	}
}

// =============================================================================
// 扩展方法测试
// =============================================================================

func TestServiceGetInternalUserFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	userRepo.users["user1"] = &interfaces.User{
		ID:     "user1",
		Age:    25,
		Gender: "male",
	}

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	features, err := service.GetInternalUserFeatures(ctx, "user1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if features == nil {
		t.Fatal("features should not be nil")
	}
	if features.UserID != "user1" {
		t.Errorf("expected user ID user1, got %s", features.UserID)
	}
	if features.Demographics.Age != 25 {
		t.Errorf("expected age 25, got %d", features.Demographics.Age)
	}
}

func TestServiceGetInternalItemFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	itemRepo.items["item1"] = &interfaces.Item{
		ID:       "item1",
		Type:     "movie",
		Category: "action",
	}

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	features, err := service.GetInternalItemFeatures(ctx, "item1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if features == nil {
		t.Fatal("features should not be nil")
	}
	if features.ItemID != "item1" {
		t.Errorf("expected item ID item1, got %s", features.ItemID)
	}
	if features.Type != "movie" {
		t.Errorf("expected type movie, got %s", features.Type)
	}
}

func TestServiceGetCrossFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	now := time.Now()
	userRepo.interactions["user1:item1"] = []*interfaces.UserBehavior{
		{UserID: "user1", ItemID: "item1", Action: "click", Timestamp: now},
	}

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	features, err := service.GetCrossFeatures(ctx, "user1", "item1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if features == nil {
		t.Fatal("features should not be nil")
	}
	if features.Interactions != 1 {
		t.Errorf("expected 1 interaction, got %d", features.Interactions)
	}
}

func TestServiceGetContextFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	req := &ContextRequest{
		Device:      "mobile",
		OS:          "iOS",
		Location:    "Beijing",
		PageContext: "home",
	}

	features := service.GetContextFeatures(ctx, req)
	if features == nil {
		t.Fatal("features should not be nil")
	}
	if features.Device != "mobile" {
		t.Errorf("expected device mobile, got %s", features.Device)
	}
}

func TestServiceBatchRefreshUserFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	userRepo.users["user1"] = &interfaces.User{ID: "user1", Age: 25}
	userRepo.users["user2"] = &interfaces.User{ID: "user2", Age: 30}

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	err := service.BatchRefreshUserFeatures(ctx, []string{"user1", "user2", "user3"})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestServiceBatchRefreshItemFeatures(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	itemRepo.items["item1"] = &interfaces.Item{ID: "item1", Type: "movie"}
	itemRepo.items["item2"] = &interfaces.Item{ID: "item2", Type: "book"}

	service := NewService(userRepo, itemRepo, cache)
	ctx := context.Background()

	err := service.BatchRefreshItemFeatures(ctx, []string{"item1", "item2", "item3"})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

// =============================================================================
// 接口实现检查
// =============================================================================

func TestServiceImplementsInterface(t *testing.T) {
	userRepo := NewMockUserRepository()
	itemRepo := NewMockItemRepository()
	cache := NewMockCache()

	service := NewService(userRepo, itemRepo, cache)

	// 编译时检查
	var _ interfaces.FeatureService = service
}

