package feature

import (
	"context"
	"testing"
	"time"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// Mock Cache 实现
// =============================================================================

// MockCache 缓存 Mock
type MockCache struct {
	data map[string]interface{}
}

func NewMockCache() *MockCache {
	return &MockCache{
		data: make(map[string]interface{}),
	}
}

func (m *MockCache) Get(ctx context.Context, key string, value interface{}) error {
	if data, ok := m.data[key]; ok {
		// 简单的类型断言赋值
		switch v := value.(type) {
		case *InternalUserFeatures:
			if src, ok := data.(*InternalUserFeatures); ok {
				*v = *src
				return nil
			}
		case *InternalItemFeatures:
			if src, ok := data.(*InternalItemFeatures); ok {
				*v = *src
				return nil
			}
		case *interfaces.UserFeatures:
			if src, ok := data.(*interfaces.UserFeatures); ok {
				*v = *src
				return nil
			}
		case *interfaces.ItemFeatures:
			if src, ok := data.(*interfaces.ItemFeatures); ok {
				*v = *src
				return nil
			}
		}
		return nil
	}
	return ErrFeatureNotFound
}

func (m *MockCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	m.data[key] = value
	return nil
}

func (m *MockCache) Delete(ctx context.Context, key string) error {
	delete(m.data, key)
	return nil
}

func (m *MockCache) Exists(ctx context.Context, key string) (bool, error) {
	_, ok := m.data[key]
	return ok, nil
}

func (m *MockCache) MGet(ctx context.Context, keys []string) ([]interface{}, error) {
	result := make([]interface{}, len(keys))
	for i, key := range keys {
		result[i] = m.data[key]
	}
	return result, nil
}

func (m *MockCache) MSet(ctx context.Context, kvs map[string]interface{}, ttl time.Duration) error {
	for k, v := range kvs {
		m.data[k] = v
	}
	return nil
}

// =============================================================================
// 特征存储测试
// =============================================================================

func TestNewFeatureStore(t *testing.T) {
	cache := NewMockCache()
	store := NewFeatureStore(cache, 30*time.Minute, 60*time.Minute)

	if store == nil {
		t.Error("store should not be nil")
	}
	if store.cache == nil {
		t.Error("cache should not be nil")
	}
	if store.userFeatureTTL != 30*time.Minute {
		t.Errorf("expected user TTL 30m, got %v", store.userFeatureTTL)
	}
	if store.itemFeatureTTL != 60*time.Minute {
		t.Errorf("expected item TTL 60m, got %v", store.itemFeatureTTL)
	}
}

func TestNewFeatureStoreWithConfig(t *testing.T) {
	cache := NewMockCache()
	config := &FeatureCacheConfig{
		UserFeatureTTL: 15 * time.Minute,
		ItemFeatureTTL: 45 * time.Minute,
	}
	store := NewFeatureStoreWithConfig(cache, config)

	if store.userFeatureTTL != 15*time.Minute {
		t.Errorf("expected user TTL 15m, got %v", store.userFeatureTTL)
	}
	if store.itemFeatureTTL != 45*time.Minute {
		t.Errorf("expected item TTL 45m, got %v", store.itemFeatureTTL)
	}
}

func TestNewFeatureStoreWithNilConfig(t *testing.T) {
	cache := NewMockCache()
	store := NewFeatureStoreWithConfig(cache, nil)

	// 应使用默认配置
	defaultConfig := DefaultFeatureCacheConfig()
	if store.userFeatureTTL != defaultConfig.UserFeatureTTL {
		t.Errorf("expected default user TTL, got %v", store.userFeatureTTL)
	}
	if store.itemFeatureTTL != defaultConfig.ItemFeatureTTL {
		t.Errorf("expected default item TTL, got %v", store.itemFeatureTTL)
	}
}

func TestSaveAndGetUserFeatures(t *testing.T) {
	cache := NewMockCache()
	store := NewFeatureStore(cache, 30*time.Minute, 60*time.Minute)
	ctx := context.Background()

	features := &InternalUserFeatures{
		UserID: "user1",
		Demographics: DemographicFeatures{
			Age:    25,
			Gender: "male",
		},
		Behavior: BehaviorFeatures{
			TotalViews: 100,
		},
		LastUpdated: time.Now(),
	}

	// 保存特征
	err := store.SaveUserFeatures(ctx, features)
	if err != nil {
		t.Errorf("unexpected error saving features: %v", err)
	}

	// 获取特征
	retrieved, err := store.GetUserFeatures(ctx, "user1")
	if err != nil {
		t.Errorf("unexpected error getting features: %v", err)
	}
	if retrieved == nil {
		t.Fatal("retrieved features should not be nil")
	}
	if retrieved.UserID != "user1" {
		t.Errorf("expected user ID user1, got %s", retrieved.UserID)
	}
	if retrieved.Demographics.Age != 25 {
		t.Errorf("expected age 25, got %d", retrieved.Demographics.Age)
	}
}

func TestSaveUserFeaturesNil(t *testing.T) {
	cache := NewMockCache()
	store := NewFeatureStore(cache, 30*time.Minute, 60*time.Minute)
	ctx := context.Background()

	err := store.SaveUserFeatures(ctx, nil)
	if err == nil {
		t.Error("expected error for nil features")
	}
}

func TestGetUserFeaturesNotFound(t *testing.T) {
	cache := NewMockCache()
	store := NewFeatureStore(cache, 30*time.Minute, 60*time.Minute)
	ctx := context.Background()

	_, err := store.GetUserFeatures(ctx, "nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent features")
	}
}

func TestSaveAndGetItemFeatures(t *testing.T) {
	cache := NewMockCache()
	store := NewFeatureStore(cache, 30*time.Minute, 60*time.Minute)
	ctx := context.Background()

	features := &InternalItemFeatures{
		ItemID: "item1",
		Type:   "movie",
		Content: ContentFeatures{
			Category: "action",
		},
		Statistics: StatisticFeatures{
			ViewCount: 1000,
		},
		SemanticID:  [3]int{100, 200, 300},
		LastUpdated: time.Now(),
	}

	// 保存特征
	err := store.SaveItemFeatures(ctx, features)
	if err != nil {
		t.Errorf("unexpected error saving features: %v", err)
	}

	// 获取特征
	retrieved, err := store.GetItemFeatures(ctx, "item1")
	if err != nil {
		t.Errorf("unexpected error getting features: %v", err)
	}
	if retrieved == nil {
		t.Fatal("retrieved features should not be nil")
	}
	if retrieved.ItemID != "item1" {
		t.Errorf("expected item ID item1, got %s", retrieved.ItemID)
	}
	if retrieved.Type != "movie" {
		t.Errorf("expected type movie, got %s", retrieved.Type)
	}
}

func TestSaveItemFeaturesNil(t *testing.T) {
	cache := NewMockCache()
	store := NewFeatureStore(cache, 30*time.Minute, 60*time.Minute)
	ctx := context.Background()

	err := store.SaveItemFeatures(ctx, nil)
	if err == nil {
		t.Error("expected error for nil features")
	}
}

func TestBatchGetUserFeatures(t *testing.T) {
	cache := NewMockCache()
	store := NewFeatureStore(cache, 30*time.Minute, 60*time.Minute)
	ctx := context.Background()

	// 保存一些特征
	store.SaveUserFeatures(ctx, &InternalUserFeatures{UserID: "user1", Demographics: DemographicFeatures{Age: 25}})
	store.SaveUserFeatures(ctx, &InternalUserFeatures{UserID: "user2", Demographics: DemographicFeatures{Age: 30}})

	// 批量获取
	found, missing, err := store.BatchGetUserFeatures(ctx, []string{"user1", "user2", "user3"})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(found) != 2 {
		t.Errorf("expected 2 found, got %d", len(found))
	}
	if len(missing) != 1 {
		t.Errorf("expected 1 missing, got %d", len(missing))
	}
	if missing[0] != "user3" {
		t.Errorf("expected missing user3, got %s", missing[0])
	}
}

func TestBatchGetItemFeatures(t *testing.T) {
	cache := NewMockCache()
	store := NewFeatureStore(cache, 30*time.Minute, 60*time.Minute)
	ctx := context.Background()

	// 保存一些特征
	store.SaveItemFeatures(ctx, &InternalItemFeatures{ItemID: "item1", Type: "movie"})
	store.SaveItemFeatures(ctx, &InternalItemFeatures{ItemID: "item2", Type: "book"})

	// 批量获取
	found, missing, err := store.BatchGetItemFeatures(ctx, []string{"item1", "item2", "item3"})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(found) != 2 {
		t.Errorf("expected 2 found, got %d", len(found))
	}
	if len(missing) != 1 {
		t.Errorf("expected 1 missing, got %d", len(missing))
	}
}

func TestInvalidateUserFeatures(t *testing.T) {
	cache := NewMockCache()
	store := NewFeatureStore(cache, 30*time.Minute, 60*time.Minute)
	ctx := context.Background()

	// 保存特征
	store.SaveUserFeatures(ctx, &InternalUserFeatures{UserID: "user1"})

	// 验证存在
	exists, _ := store.ExistsUserFeatures(ctx, "user1")
	if !exists {
		t.Error("features should exist")
	}

	// 使失效
	err := store.InvalidateUserFeatures(ctx, "user1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// 验证不存在
	exists, _ = store.ExistsUserFeatures(ctx, "user1")
	if exists {
		t.Error("features should not exist after invalidation")
	}
}

func TestInvalidateItemFeatures(t *testing.T) {
	cache := NewMockCache()
	store := NewFeatureStore(cache, 30*time.Minute, 60*time.Minute)
	ctx := context.Background()

	// 保存特征
	store.SaveItemFeatures(ctx, &InternalItemFeatures{ItemID: "item1"})

	// 验证存在
	exists, _ := store.ExistsItemFeatures(ctx, "item1")
	if !exists {
		t.Error("features should exist")
	}

	// 使失效
	err := store.InvalidateItemFeatures(ctx, "item1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// 验证不存在
	exists, _ = store.ExistsItemFeatures(ctx, "item1")
	if exists {
		t.Error("features should not exist after invalidation")
	}
}

func TestCacheKeyGeneration(t *testing.T) {
	tests := []struct {
		name     string
		fn       func(string) string
		input    string
		expected string
	}{
		{"user key", userFeatureKey, "user123", "feature:user:user123"},
		{"item key", itemFeatureKey, "item456", "feature:item:item456"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.fn(tt.input)
			if result != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, result)
			}
		})
	}
}

func TestCrossFeatureKey(t *testing.T) {
	key := crossFeatureKey("user1", "item1")
	expected := "feature:cross:user1:item1"
	if key != expected {
		t.Errorf("expected %s, got %s", expected, key)
	}
}

// =============================================================================
// 特征转换测试
// =============================================================================

func TestConvertToInterfaceUserFeatures(t *testing.T) {
	now := time.Now()
	internal := &InternalUserFeatures{
		UserID: "user1",
		Demographics: DemographicFeatures{
			Age:      25,
			Gender:   "male",
			Location: "Beijing",
			Device:   "mobile",
		},
		Behavior: BehaviorFeatures{
			TotalViews:     100,
			TotalClicks:    50,
			TotalPurchases: 10,
			AvgSessionTime: 120.5,
			ActiveDays:     30,
			LastActiveHour: 14,
			PreferredHours: []int{10, 14, 20},
		},
		Preferences: PreferenceFeatures{
			TopCategories: []CategoryScore{{Category: "electronics", Score: 0.8}},
			TopTags:       []TagScore{{Tag: "phone", Score: 0.9}},
			PriceRange:    [2]float64{100, 5000},
			ContentLength: "medium",
		},
		Embedding:   []float32{0.1, 0.2, 0.3},
		LastUpdated: now,
	}

	result := ConvertToInterfaceUserFeatures(internal)

	if result == nil {
		t.Fatal("result should not be nil")
	}
	if result.UserID != "user1" {
		t.Errorf("expected user ID user1, got %s", result.UserID)
	}
	if result.Demographics == nil {
		t.Error("demographics should not be nil")
	}
	if result.Behavior == nil {
		t.Error("behavior should not be nil")
	}
	if result.Preferences == nil {
		t.Error("preferences should not be nil")
	}
	if len(result.Embedding) != 3 {
		t.Errorf("expected 3 embedding dims, got %d", len(result.Embedding))
	}
}

func TestConvertToInterfaceUserFeaturesNil(t *testing.T) {
	result := ConvertToInterfaceUserFeatures(nil)
	if result != nil {
		t.Error("result should be nil for nil input")
	}
}

func TestConvertToInterfaceItemFeatures(t *testing.T) {
	now := time.Now()
	internal := &InternalItemFeatures{
		ItemID: "item1",
		Type:   "movie",
		Content: ContentFeatures{
			Category:    "action",
			SubCategory: "superhero",
			Tags:        []string{"thriller", "sci-fi"},
			Price:       0,
			Duration:    7200,
			ReleaseDate: "2024-01-15",
		},
		Statistics: StatisticFeatures{
			ViewCount:    10000,
			ClickCount:   5000,
			LikeCount:    2000,
			ShareCount:   500,
			CommentCount: 300,
			AvgRating:    4.5,
			CTR:          0.5,
			CVR:          0.1,
		},
		Embedding:   []float32{0.4, 0.5, 0.6},
		SemanticID:  [3]int{100, 200, 300},
		LastUpdated: now,
	}

	result := ConvertToInterfaceItemFeatures(internal)

	if result == nil {
		t.Fatal("result should not be nil")
	}
	if result.ItemID != "item1" {
		t.Errorf("expected item ID item1, got %s", result.ItemID)
	}
	if result.Type != "movie" {
		t.Errorf("expected type movie, got %s", result.Type)
	}
	if result.Content == nil {
		t.Error("content should not be nil")
	}
	if result.Statistics == nil {
		t.Error("statistics should not be nil")
	}
	if result.SemanticID[0] != 100 {
		t.Errorf("expected semantic L1 100, got %d", result.SemanticID[0])
	}
}

func TestConvertToInterfaceItemFeaturesNil(t *testing.T) {
	result := ConvertToInterfaceItemFeatures(nil)
	if result != nil {
		t.Error("result should be nil for nil input")
	}
}

func TestConvertFromInterfaceUserFeatures(t *testing.T) {
	now := time.Now()
	intf := &interfaces.UserFeatures{
		UserID: "user1",
		Demographics: map[string]interface{}{
			"age":      float64(25),
			"gender":   "male",
			"location": "Beijing",
			"device":   "mobile",
		},
		Behavior: map[string]interface{}{
			"total_views":      float64(100),
			"total_clicks":     float64(50),
			"total_purchases":  float64(10),
			"avg_session_time": float64(120.5),
			"active_days":      float64(30),
			"last_active_hour": float64(14),
		},
		Embedding:   []float32{0.1, 0.2, 0.3},
		LastUpdated: now,
	}

	result := ConvertFromInterfaceUserFeatures(intf)

	if result == nil {
		t.Fatal("result should not be nil")
	}
	if result.UserID != "user1" {
		t.Errorf("expected user ID user1, got %s", result.UserID)
	}
	if result.Demographics.Age != 25 {
		t.Errorf("expected age 25, got %d", result.Demographics.Age)
	}
	if result.Demographics.Gender != "male" {
		t.Errorf("expected gender male, got %s", result.Demographics.Gender)
	}
	if result.Behavior.TotalViews != 100 {
		t.Errorf("expected total views 100, got %d", result.Behavior.TotalViews)
	}
}

func TestConvertFromInterfaceUserFeaturesNil(t *testing.T) {
	result := ConvertFromInterfaceUserFeatures(nil)
	if result != nil {
		t.Error("result should be nil for nil input")
	}
}

func TestConvertFromInterfaceItemFeatures(t *testing.T) {
	now := time.Now()
	intf := &interfaces.ItemFeatures{
		ItemID: "item1",
		Type:   "movie",
		Content: map[string]interface{}{
			"category":     "action",
			"sub_category": "superhero",
			"tags":         []interface{}{"thriller", "sci-fi"},
			"price":        float64(0),
			"duration":     float64(7200),
		},
		Statistics: map[string]interface{}{
			"view_count":  float64(10000),
			"click_count": float64(5000),
			"like_count":  float64(2000),
			"share_count": float64(500),
			"avg_rating":  float64(4.5),
			"ctr":         float64(0.5),
		},
		Embedding:   []float32{0.4, 0.5, 0.6},
		SemanticID:  [3]int{100, 200, 300},
		LastUpdated: now,
	}

	result := ConvertFromInterfaceItemFeatures(intf)

	if result == nil {
		t.Fatal("result should not be nil")
	}
	if result.ItemID != "item1" {
		t.Errorf("expected item ID item1, got %s", result.ItemID)
	}
	if result.Type != "movie" {
		t.Errorf("expected type movie, got %s", result.Type)
	}
	if result.Content.Category != "action" {
		t.Errorf("expected category action, got %s", result.Content.Category)
	}
	if result.Statistics.ViewCount != 10000 {
		t.Errorf("expected view count 10000, got %d", result.Statistics.ViewCount)
	}
}

func TestConvertFromInterfaceItemFeaturesNil(t *testing.T) {
	result := ConvertFromInterfaceItemFeatures(nil)
	if result != nil {
		t.Error("result should be nil for nil input")
	}
}

