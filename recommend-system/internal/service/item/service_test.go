// Package item 提供物品服务单元测试
package item

import (
	"context"
	"errors"
	"testing"
	"time"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// Mock 实现
// =============================================================================

// mockItemRepository 模拟物品仓储
type mockItemRepository struct {
	items      map[string]*interfaces.Item
	stats      map[string]*interfaces.ItemStats
	createErr  error
	updateErr  error
	deleteErr  error
	getErr     error
	searchErr  error
	listErr    error
}

func newMockItemRepository() *mockItemRepository {
	return &mockItemRepository{
		items: make(map[string]*interfaces.Item),
		stats: make(map[string]*interfaces.ItemStats),
	}
}

func (m *mockItemRepository) GetByID(ctx context.Context, itemID string) (*interfaces.Item, error) {
	if m.getErr != nil {
		return nil, m.getErr
	}
	item, ok := m.items[itemID]
	if !ok {
		return nil, errors.New("item not found")
	}
	return item, nil
}

func (m *mockItemRepository) GetByIDs(ctx context.Context, itemIDs []string) ([]*interfaces.Item, error) {
	if m.getErr != nil {
		return nil, m.getErr
	}
	var result []*interfaces.Item
	for _, id := range itemIDs {
		if item, ok := m.items[id]; ok {
			result = append(result, item)
		}
	}
	return result, nil
}

func (m *mockItemRepository) Create(ctx context.Context, item *interfaces.Item) error {
	if m.createErr != nil {
		return m.createErr
	}
	m.items[item.ID] = item
	return nil
}

func (m *mockItemRepository) Update(ctx context.Context, item *interfaces.Item) error {
	if m.updateErr != nil {
		return m.updateErr
	}
	m.items[item.ID] = item
	return nil
}

func (m *mockItemRepository) Delete(ctx context.Context, itemID string) error {
	if m.deleteErr != nil {
		return m.deleteErr
	}
	delete(m.items, itemID)
	return nil
}

func (m *mockItemRepository) List(ctx context.Context, itemType, category string, page, pageSize int) ([]*interfaces.Item, int64, error) {
	if m.listErr != nil {
		return nil, 0, m.listErr
	}
	var result []*interfaces.Item
	for _, item := range m.items {
		if (itemType == "" || item.Type == itemType) && (category == "" || item.Category == category) {
			result = append(result, item)
		}
	}
	total := int64(len(result))

	// 分页处理
	start := (page - 1) * pageSize
	if start >= len(result) {
		return []*interfaces.Item{}, total, nil
	}
	end := start + pageSize
	if end > len(result) {
		end = len(result)
	}

	return result[start:end], total, nil
}

func (m *mockItemRepository) Search(ctx context.Context, query string, limit int) ([]*interfaces.Item, error) {
	if m.searchErr != nil {
		return nil, m.searchErr
	}
	var result []*interfaces.Item
	for _, item := range m.items {
		// 简单的标题匹配
		if contains(item.Title, query) {
			result = append(result, item)
			if len(result) >= limit {
				break
			}
		}
	}
	return result, nil
}

func (m *mockItemRepository) GetStats(ctx context.Context, itemID string) (*interfaces.ItemStats, error) {
	stats, ok := m.stats[itemID]
	if !ok {
		return &interfaces.ItemStats{ItemID: itemID}, nil
	}
	return stats, nil
}

func (m *mockItemRepository) IncrementStats(ctx context.Context, itemID, action string) error {
	stats, ok := m.stats[itemID]
	if !ok {
		stats = &interfaces.ItemStats{ItemID: itemID}
		m.stats[itemID] = stats
	}

	switch action {
	case "view":
		stats.ViewCount++
	case "click":
		stats.ClickCount++
	case "like":
		stats.LikeCount++
	case "share":
		stats.ShareCount++
	}
	return nil
}

func (m *mockItemRepository) GetPopularByCategories(ctx context.Context, categories []string, limit int) ([]*interfaces.Item, error) {
	var result []*interfaces.Item
	categorySet := make(map[string]bool)
	for _, c := range categories {
		categorySet[c] = true
	}

	for _, item := range m.items {
		if len(categories) == 0 || categorySet[item.Category] {
			result = append(result, item)
			if len(result) >= limit {
				break
			}
		}
	}
	return result, nil
}

// mockCache 模拟缓存
type mockCache struct {
	data   map[string]interface{}
	getErr error
	setErr error
}

func newMockCache() *mockCache {
	return &mockCache{
		data: make(map[string]interface{}),
	}
}

func (m *mockCache) Get(ctx context.Context, key string, value interface{}) error {
	if m.getErr != nil {
		return m.getErr
	}
	v, ok := m.data[key]
	if !ok {
		return errors.New("key not found")
	}
	// 简化处理：类型断言
	if item, ok := v.(*interfaces.Item); ok {
		if target, ok := value.(*interfaces.Item); ok {
			*target = *item
			return nil
		}
	}
	return errors.New("type mismatch")
}

func (m *mockCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	if m.setErr != nil {
		return m.setErr
	}
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
	results := make([]interface{}, len(keys))
	for i, key := range keys {
		results[i] = m.data[key]
	}
	return results, nil
}

func (m *mockCache) MSet(ctx context.Context, kvs map[string]interface{}, ttl time.Duration) error {
	for k, v := range kvs {
		m.data[k] = v
	}
	return nil
}

// =============================================================================
// 辅助函数
// =============================================================================

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// createTestService 创建测试用的服务实例
func createTestService() (*Service, *mockItemRepository, *mockCache) {
	repo := newMockItemRepository()
	cache := newMockCache()

	service := NewService(
		repo,
		nil, // Milvus 客户端可为空
		cache,
		DefaultConfig(),
	)

	return service, repo, cache
}

// =============================================================================
// 测试用例
// =============================================================================

// TestNewService 测试服务创建
func TestNewService(t *testing.T) {
	repo := newMockItemRepository()
	cache := newMockCache()

	// 使用默认配置
	service := NewService(repo, nil, cache, nil)
	if service == nil {
		t.Fatal("expected service to be created")
	}
	if service.config == nil {
		t.Fatal("expected default config to be used")
	}
	if service.config.CacheTTL != time.Hour {
		t.Errorf("expected default cache TTL to be 1 hour, got %v", service.config.CacheTTL)
	}

	// 使用自定义配置
	customConfig := &Config{
		CacheTTL:         30 * time.Minute,
		MilvusCollection: "test_collection",
		DefaultPageSize:  50,
		MaxPageSize:      200,
		EmbeddingDim:     128,
	}
	service2 := NewService(repo, nil, cache, customConfig)
	if service2.config.CacheTTL != 30*time.Minute {
		t.Errorf("expected custom cache TTL, got %v", service2.config.CacheTTL)
	}
}

// TestGetItem 测试获取物品
func TestGetItem(t *testing.T) {
	service, repo, _ := createTestService()
	ctx := context.Background()

	// 添加测试数据
	testItem := &interfaces.Item{
		ID:          "test_item_1",
		Type:        "movie",
		Title:       "Test Movie",
		Description: "A test movie",
		Category:    "action",
		Status:      "active",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	repo.items[testItem.ID] = testItem

	t.Run("success", func(t *testing.T) {
		item, err := service.GetItem(ctx, "test_item_1")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if item.ID != testItem.ID {
			t.Errorf("expected item ID %s, got %s", testItem.ID, item.ID)
		}
		if item.Title != testItem.Title {
			t.Errorf("expected title %s, got %s", testItem.Title, item.Title)
		}
	})

	t.Run("not found", func(t *testing.T) {
		_, err := service.GetItem(ctx, "non_existent")
		if err == nil {
			t.Fatal("expected error for non-existent item")
		}
		if err != ErrItemNotFound {
			t.Errorf("expected ErrItemNotFound, got %v", err)
		}
	})

	t.Run("empty id", func(t *testing.T) {
		_, err := service.GetItem(ctx, "")
		if err == nil {
			t.Fatal("expected error for empty item ID")
		}
		if err != ErrInvalidRequest {
			t.Errorf("expected ErrInvalidRequest, got %v", err)
		}
	})
}

// TestCreateItem 测试创建物品
func TestCreateItem(t *testing.T) {
	service, _, _ := createTestService()
	ctx := context.Background()

	t.Run("success", func(t *testing.T) {
		req := &interfaces.CreateItemRequest{
			Type:        "movie",
			Title:       "New Movie",
			Description: "A new movie",
			Category:    "drama",
			Tags:        []string{"drama", "new"},
		}

		item, err := service.CreateItem(ctx, req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if item.ID == "" {
			t.Error("expected item ID to be generated")
		}
		if item.Type != req.Type {
			t.Errorf("expected type %s, got %s", req.Type, item.Type)
		}
		if item.Title != req.Title {
			t.Errorf("expected title %s, got %s", req.Title, item.Title)
		}
		if item.Status != "active" {
			t.Errorf("expected status 'active', got %s", item.Status)
		}
	})

	t.Run("nil request", func(t *testing.T) {
		_, err := service.CreateItem(ctx, nil)
		if err == nil {
			t.Fatal("expected error for nil request")
		}
	})

	t.Run("missing required fields", func(t *testing.T) {
		// 缺少 Type
		req1 := &interfaces.CreateItemRequest{
			Title: "Test",
		}
		_, err := service.CreateItem(ctx, req1)
		if err == nil {
			t.Fatal("expected error for missing type")
		}

		// 缺少 Title
		req2 := &interfaces.CreateItemRequest{
			Type: "movie",
		}
		_, err = service.CreateItem(ctx, req2)
		if err == nil {
			t.Fatal("expected error for missing title")
		}
	})

	t.Run("with different item types", func(t *testing.T) {
		testCases := []struct {
			itemType     string
			expectPrefix string
		}{
			{"movie", "mov_"},
			{"product", "prd_"},
			{"article", "art_"},
			{"video", "vid_"},
			{"unknown", "itm_"},
		}

		for _, tc := range testCases {
			req := &interfaces.CreateItemRequest{
				Type:  tc.itemType,
				Title: "Test",
			}
			item, err := service.CreateItem(ctx, req)
			if err != nil {
				t.Fatalf("unexpected error for type %s: %v", tc.itemType, err)
			}
			if len(item.ID) < len(tc.expectPrefix) || item.ID[:len(tc.expectPrefix)] != tc.expectPrefix {
				t.Errorf("expected ID to start with %s, got %s", tc.expectPrefix, item.ID)
			}
		}
	})
}

// TestUpdateItem 测试更新物品
func TestUpdateItem(t *testing.T) {
	service, repo, _ := createTestService()
	ctx := context.Background()

	// 添加测试数据
	testItem := &interfaces.Item{
		ID:          "test_item_1",
		Type:        "movie",
		Title:       "Original Title",
		Description: "Original description",
		Category:    "action",
		Status:      "active",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	repo.items[testItem.ID] = testItem

	t.Run("success", func(t *testing.T) {
		req := &interfaces.UpdateItemRequest{
			Title:       "Updated Title",
			Description: "Updated description",
		}

		item, err := service.UpdateItem(ctx, "test_item_1", req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if item.Title != req.Title {
			t.Errorf("expected title %s, got %s", req.Title, item.Title)
		}
		if item.Description != req.Description {
			t.Errorf("expected description %s, got %s", req.Description, item.Description)
		}
	})

	t.Run("not found", func(t *testing.T) {
		req := &interfaces.UpdateItemRequest{
			Title: "New Title",
		}
		_, err := service.UpdateItem(ctx, "non_existent", req)
		if err == nil {
			t.Fatal("expected error for non-existent item")
		}
	})

	t.Run("empty id", func(t *testing.T) {
		req := &interfaces.UpdateItemRequest{
			Title: "New Title",
		}
		_, err := service.UpdateItem(ctx, "", req)
		if err == nil {
			t.Fatal("expected error for empty item ID")
		}
	})

	t.Run("nil request", func(t *testing.T) {
		_, err := service.UpdateItem(ctx, "test_item_1", nil)
		if err == nil {
			t.Fatal("expected error for nil request")
		}
	})
}

// TestDeleteItem 测试删除物品
func TestDeleteItem(t *testing.T) {
	service, repo, cache := createTestService()
	ctx := context.Background()

	t.Run("success", func(t *testing.T) {
		// 添加测试数据
		testItem := &interfaces.Item{
			ID:     "test_item_delete",
			Type:   "movie",
			Title:  "To Be Deleted",
			Status: "active",
		}
		repo.items[testItem.ID] = testItem
		cache.data["item:"+testItem.ID] = testItem

		err := service.DeleteItem(ctx, "test_item_delete")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// 验证已删除
		if _, ok := repo.items["test_item_delete"]; ok {
			t.Error("expected item to be deleted from repository")
		}
	})

	t.Run("empty id", func(t *testing.T) {
		err := service.DeleteItem(ctx, "")
		if err == nil {
			t.Fatal("expected error for empty item ID")
		}
	})
}

// TestListItems 测试列出物品
func TestListItems(t *testing.T) {
	service, repo, _ := createTestService()
	ctx := context.Background()

	// 添加测试数据
	for i := 0; i < 25; i++ {
		itemType := "movie"
		category := "action"
		if i%2 == 0 {
			itemType = "product"
			category = "electronics"
		}
		repo.items[string(rune('a'+i))] = &interfaces.Item{
			ID:       string(rune('a' + i)),
			Type:     itemType,
			Title:    "Test Item",
			Category: category,
			Status:   "active",
		}
	}

	t.Run("default pagination", func(t *testing.T) {
		resp, err := service.ListItems(ctx, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.Total != 25 {
			t.Errorf("expected total 25, got %d", resp.Total)
		}
		if len(resp.Items) > 20 {
			t.Errorf("expected max 20 items, got %d", len(resp.Items))
		}
	})

	t.Run("with pagination", func(t *testing.T) {
		req := &interfaces.ListItemsRequest{
			Page:     2,
			PageSize: 10,
		}
		resp, err := service.ListItems(ctx, req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.Page != 2 {
			t.Errorf("expected page 2, got %d", resp.Page)
		}
	})

	t.Run("filter by type", func(t *testing.T) {
		req := &interfaces.ListItemsRequest{
			Type: "movie",
		}
		resp, err := service.ListItems(ctx, req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		for _, item := range resp.Items {
			if item.Type != "movie" {
				t.Errorf("expected type 'movie', got %s", item.Type)
			}
		}
	})
}

// TestSearchItems 测试搜索物品
func TestSearchItems(t *testing.T) {
	service, repo, _ := createTestService()
	ctx := context.Background()

	// 添加测试数据
	repo.items["1"] = &interfaces.Item{ID: "1", Title: "Test Movie Alpha", Status: "active"}
	repo.items["2"] = &interfaces.Item{ID: "2", Title: "Test Movie Beta", Status: "active"}
	repo.items["3"] = &interfaces.Item{ID: "3", Title: "Other Product", Status: "active"}

	t.Run("success", func(t *testing.T) {
		items, err := service.SearchItems(ctx, "Test Movie", 10)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(items) < 2 {
			t.Errorf("expected at least 2 items, got %d", len(items))
		}
	})

	t.Run("empty query", func(t *testing.T) {
		_, err := service.SearchItems(ctx, "", 10)
		if err == nil {
			t.Fatal("expected error for empty query")
		}
	})

	t.Run("with limit", func(t *testing.T) {
		items, err := service.SearchItems(ctx, "Test", 1)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(items) > 1 {
			t.Errorf("expected max 1 item, got %d", len(items))
		}
	})
}

// TestBatchGetItems 测试批量获取物品
func TestBatchGetItems(t *testing.T) {
	service, repo, _ := createTestService()
	ctx := context.Background()

	// 添加测试数据
	for i := 1; i <= 5; i++ {
		id := string(rune('0' + i))
		repo.items[id] = &interfaces.Item{
			ID:     id,
			Type:   "movie",
			Title:  "Movie " + id,
			Status: "active",
		}
	}

	t.Run("success", func(t *testing.T) {
		items, err := service.BatchGetItems(ctx, []string{"1", "2", "3"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(items) != 3 {
			t.Errorf("expected 3 items, got %d", len(items))
		}
	})

	t.Run("empty ids", func(t *testing.T) {
		items, err := service.BatchGetItems(ctx, []string{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if items != nil && len(items) != 0 {
			t.Errorf("expected empty result, got %d items", len(items))
		}
	})

	t.Run("with non-existent ids", func(t *testing.T) {
		items, err := service.BatchGetItems(ctx, []string{"1", "non_existent", "2"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(items) != 2 {
			t.Errorf("expected 2 items, got %d", len(items))
		}
	})

	t.Run("with duplicates", func(t *testing.T) {
		items, err := service.BatchGetItems(ctx, []string{"1", "1", "2", "2"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// 返回时会按原顺序返回，所以可能有重复
		if len(items) < 2 {
			t.Errorf("expected at least 2 items, got %d", len(items))
		}
	})
}

// TestGetItemStats 测试获取物品统计
func TestGetItemStats(t *testing.T) {
	service, repo, _ := createTestService()
	ctx := context.Background()

	// 添加测试数据
	repo.stats["test_item_1"] = &interfaces.ItemStats{
		ItemID:     "test_item_1",
		ViewCount:  100,
		ClickCount: 50,
		LikeCount:  25,
		ShareCount: 10,
		AvgRating:  4.5,
	}

	t.Run("success", func(t *testing.T) {
		stats, err := service.GetItemStats(ctx, "test_item_1")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if stats.ViewCount != 100 {
			t.Errorf("expected view count 100, got %d", stats.ViewCount)
		}
		if stats.AvgRating != 4.5 {
			t.Errorf("expected avg rating 4.5, got %f", stats.AvgRating)
		}
	})

	t.Run("empty stats", func(t *testing.T) {
		stats, err := service.GetItemStats(ctx, "new_item")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if stats.ItemID != "new_item" {
			t.Errorf("expected item ID 'new_item', got %s", stats.ItemID)
		}
		if stats.ViewCount != 0 {
			t.Errorf("expected view count 0, got %d", stats.ViewCount)
		}
	})

	t.Run("empty id", func(t *testing.T) {
		_, err := service.GetItemStats(ctx, "")
		if err == nil {
			t.Fatal("expected error for empty item ID")
		}
	})
}

// TestUpdateItemStats 测试更新物品统计
func TestUpdateItemStats(t *testing.T) {
	service, repo, _ := createTestService()
	ctx := context.Background()

	t.Run("success", func(t *testing.T) {
		err := service.UpdateItemStats(ctx, "test_item_1", "view")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		stats := repo.stats["test_item_1"]
		if stats.ViewCount != 1 {
			t.Errorf("expected view count 1, got %d", stats.ViewCount)
		}

		// 再次增加
		err = service.UpdateItemStats(ctx, "test_item_1", "view")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if stats.ViewCount != 2 {
			t.Errorf("expected view count 2, got %d", stats.ViewCount)
		}
	})

	t.Run("different actions", func(t *testing.T) {
		itemID := "test_item_actions"
		actions := []string{"view", "click", "like", "share"}

		for _, action := range actions {
			err := service.UpdateItemStats(ctx, itemID, action)
			if err != nil {
				t.Fatalf("unexpected error for action %s: %v", action, err)
			}
		}

		stats := repo.stats[itemID]
		if stats.ViewCount != 1 || stats.ClickCount != 1 || stats.LikeCount != 1 || stats.ShareCount != 1 {
			t.Error("expected all counts to be 1")
		}
	})

	t.Run("invalid action", func(t *testing.T) {
		err := service.UpdateItemStats(ctx, "test_item_1", "invalid")
		if err == nil {
			t.Fatal("expected error for invalid action")
		}
	})

	t.Run("empty id", func(t *testing.T) {
		err := service.UpdateItemStats(ctx, "", "view")
		if err == nil {
			t.Fatal("expected error for empty item ID")
		}
	})

	t.Run("empty action", func(t *testing.T) {
		err := service.UpdateItemStats(ctx, "test_item_1", "")
		if err == nil {
			t.Fatal("expected error for empty action")
		}
	})
}

// TestGetSimilarItems 测试获取相似物品（无 Milvus 时的行为）
func TestGetSimilarItems(t *testing.T) {
	service, _, _ := createTestService()
	ctx := context.Background()

	t.Run("milvus not available", func(t *testing.T) {
		_, err := service.GetSimilarItems(ctx, "test_item_1", 10)
		if err == nil {
			t.Fatal("expected error when milvus is not available")
		}
		if err != ErrMilvusNotAvailable {
			t.Errorf("expected ErrMilvusNotAvailable, got %v", err)
		}
	})

	t.Run("empty id", func(t *testing.T) {
		_, err := service.GetSimilarItems(ctx, "", 10)
		if err == nil {
			t.Fatal("expected error for empty item ID")
		}
	})
}

// TestDefaultConfig 测试默认配置
func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()

	if config.CacheTTL != time.Hour {
		t.Errorf("expected CacheTTL to be 1 hour, got %v", config.CacheTTL)
	}
	if config.MilvusCollection != "item_embeddings" {
		t.Errorf("expected MilvusCollection to be 'item_embeddings', got %s", config.MilvusCollection)
	}
	if config.DefaultPageSize != 20 {
		t.Errorf("expected DefaultPageSize to be 20, got %d", config.DefaultPageSize)
	}
	if config.MaxPageSize != 100 {
		t.Errorf("expected MaxPageSize to be 100, got %d", config.MaxPageSize)
	}
	if config.EmbeddingDim != 256 {
		t.Errorf("expected EmbeddingDim to be 256, got %d", config.EmbeddingDim)
	}
}

// TestCacheAdapter 测试缓存适配器
func TestCacheAdapter(t *testing.T) {
	mockCache := newMockCache()
	adapter := NewCacheAdapter(mockCache)

	ctx := context.Background()

	t.Run("set and get", func(t *testing.T) {
		item := &interfaces.Item{
			ID:    "test_id",
			Title: "Test Title",
		}
		err := adapter.Set(ctx, "test_key", item, time.Hour)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var retrieved interfaces.Item
		err = adapter.Get(ctx, "test_key", &retrieved)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("delete", func(t *testing.T) {
		mockCache.data["to_delete"] = "value"
		err := adapter.Delete(ctx, "to_delete")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if _, ok := mockCache.data["to_delete"]; ok {
			t.Error("expected key to be deleted")
		}
	})
}

// =============================================================================
// 基准测试
// =============================================================================

// BenchmarkGetItem 基准测试：获取物品
func BenchmarkGetItem(b *testing.B) {
	service, repo, _ := createTestService()
	ctx := context.Background()

	// 添加测试数据
	repo.items["bench_item"] = &interfaces.Item{
		ID:     "bench_item",
		Type:   "movie",
		Title:  "Benchmark Movie",
		Status: "active",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = service.GetItem(ctx, "bench_item")
	}
}

// BenchmarkCreateItem 基准测试：创建物品
func BenchmarkCreateItem(b *testing.B) {
	service, _, _ := createTestService()
	ctx := context.Background()

	req := &interfaces.CreateItemRequest{
		Type:        "movie",
		Title:       "Benchmark Movie",
		Description: "A benchmark movie",
		Category:    "action",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = service.CreateItem(ctx, req)
	}
}

// BenchmarkBatchGetItems 基准测试：批量获取物品
func BenchmarkBatchGetItems(b *testing.B) {
	service, repo, _ := createTestService()
	ctx := context.Background()

	// 添加测试数据
	ids := make([]string, 100)
	for i := 0; i < 100; i++ {
		id := string(rune('a' + i%26)) + string(rune('0'+i/26))
		ids[i] = id
		repo.items[id] = &interfaces.Item{
			ID:     id,
			Type:   "movie",
			Title:  "Movie " + id,
			Status: "active",
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = service.BatchGetItems(ctx, ids[:10])
	}
}

