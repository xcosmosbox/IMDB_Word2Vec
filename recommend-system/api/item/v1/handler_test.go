// Package v1 提供物品 API v1 版本单元测试
package v1

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"recommend-system/internal/interfaces"
	"recommend-system/internal/service/item"
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
		result = append(result, item)
		if len(result) >= limit {
			break
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
	for _, item := range m.items {
		result = append(result, item)
		if len(result) >= limit {
			break
		}
	}
	return result, nil
}

// mockCache 模拟缓存
type mockCache struct {
	data map[string]interface{}
}

func newMockCache() *mockCache {
	return &mockCache{
		data: make(map[string]interface{}),
	}
}

func (m *mockCache) Get(ctx context.Context, key string, value interface{}) error {
	return errors.New("cache miss")
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
	return nil, nil
}

func (m *mockCache) MSet(ctx context.Context, kvs map[string]interface{}, ttl time.Duration) error {
	return nil
}

// =============================================================================
// 测试辅助函数
// =============================================================================

func setupTestRouter() (*gin.Engine, *item.Service, *mockItemRepository) {
	gin.SetMode(gin.TestMode)

	repo := newMockItemRepository()
	cache := newMockCache()

	service := item.NewService(
		repo,
		nil,
		cache,
		item.DefaultConfig(),
	)

	handler := NewHandler(service)

	router := gin.New()
	apiV1 := router.Group("/api/v1")
	handler.RegisterRoutes(apiV1)

	return router, service, repo
}

func performRequest(router *gin.Engine, method, path string, body interface{}) *httptest.ResponseRecorder {
	var bodyBytes []byte
	if body != nil {
		bodyBytes, _ = json.Marshal(body)
	}

	req, _ := http.NewRequest(method, path, bytes.NewBuffer(bodyBytes))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	return w
}

func parseResponse(w *httptest.ResponseRecorder) Response {
	var resp Response
	json.Unmarshal(w.Body.Bytes(), &resp)
	return resp
}

// =============================================================================
// API 测试用例
// =============================================================================

// TestCreateItemAPI 测试创建物品 API
func TestCreateItemAPI(t *testing.T) {
	router, _, _ := setupTestRouter()

	t.Run("success", func(t *testing.T) {
		req := CreateItemRequest{
			Type:        "movie",
			Title:       "Test Movie",
			Description: "A test movie",
			Category:    "action",
			Tags:        []string{"action", "test"},
		}

		w := performRequest(router, "POST", "/api/v1/items", req)

		if w.Code != http.StatusCreated {
			t.Errorf("expected status 201, got %d", w.Code)
		}

		resp := parseResponse(w)
		if resp.Code != 0 {
			t.Errorf("expected code 0, got %d", resp.Code)
		}
	})

	t.Run("missing required fields", func(t *testing.T) {
		req := CreateItemRequest{
			Description: "No type or title",
		}

		w := performRequest(router, "POST", "/api/v1/items", req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}
	})

	t.Run("invalid json", func(t *testing.T) {
		req, _ := http.NewRequest("POST", "/api/v1/items", bytes.NewBufferString("invalid json"))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}
	})
}

// TestGetItemAPI 测试获取物品 API
func TestGetItemAPI(t *testing.T) {
	router, _, repo := setupTestRouter()

	// 添加测试数据
	testItem := &interfaces.Item{
		ID:          "test_item_1",
		Type:        "movie",
		Title:       "Test Movie",
		Description: "A test movie",
		Status:      "active",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	repo.items[testItem.ID] = testItem

	t.Run("success", func(t *testing.T) {
		w := performRequest(router, "GET", "/api/v1/items/test_item_1", nil)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		resp := parseResponse(w)
		if resp.Code != 0 {
			t.Errorf("expected code 0, got %d", resp.Code)
		}
	})

	t.Run("not found", func(t *testing.T) {
		w := performRequest(router, "GET", "/api/v1/items/non_existent", nil)

		if w.Code != http.StatusNotFound {
			t.Errorf("expected status 404, got %d", w.Code)
		}
	})
}

// TestListItemsAPI 测试列出物品 API
func TestListItemsAPI(t *testing.T) {
	router, _, repo := setupTestRouter()

	// 添加测试数据
	for i := 0; i < 25; i++ {
		id := string(rune('a' + i))
		repo.items[id] = &interfaces.Item{
			ID:       id,
			Type:     "movie",
			Title:    "Movie " + id,
			Category: "action",
			Status:   "active",
		}
	}

	t.Run("default pagination", func(t *testing.T) {
		w := performRequest(router, "GET", "/api/v1/items", nil)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		resp := parseResponse(w)
		if resp.Code != 0 {
			t.Errorf("expected code 0, got %d", resp.Code)
		}
	})

	t.Run("with query params", func(t *testing.T) {
		w := performRequest(router, "GET", "/api/v1/items?type=movie&page=1&page_size=10", nil)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}
	})
}

// TestSearchItemsAPI 测试搜索物品 API
func TestSearchItemsAPI(t *testing.T) {
	router, _, repo := setupTestRouter()

	// 添加测试数据
	repo.items["1"] = &interfaces.Item{ID: "1", Title: "Test Movie", Status: "active"}
	repo.items["2"] = &interfaces.Item{ID: "2", Title: "Another Movie", Status: "active"}

	t.Run("success", func(t *testing.T) {
		w := performRequest(router, "GET", "/api/v1/items/search?q=Movie", nil)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		resp := parseResponse(w)
		if resp.Code != 0 {
			t.Errorf("expected code 0, got %d", resp.Code)
		}
	})

	t.Run("missing query", func(t *testing.T) {
		w := performRequest(router, "GET", "/api/v1/items/search", nil)

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}
	})

	t.Run("with limit", func(t *testing.T) {
		w := performRequest(router, "GET", "/api/v1/items/search?q=Movie&limit=5", nil)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}
	})
}

// TestUpdateItemAPI 测试更新物品 API
func TestUpdateItemAPI(t *testing.T) {
	router, _, repo := setupTestRouter()

	// 添加测试数据
	testItem := &interfaces.Item{
		ID:        "test_item_1",
		Type:      "movie",
		Title:     "Original Title",
		Status:    "active",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	repo.items[testItem.ID] = testItem

	t.Run("success", func(t *testing.T) {
		req := UpdateItemRequest{
			Title:       "Updated Title",
			Description: "Updated description",
		}

		w := performRequest(router, "PUT", "/api/v1/items/test_item_1", req)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		resp := parseResponse(w)
		if resp.Code != 0 {
			t.Errorf("expected code 0, got %d", resp.Code)
		}
	})

	t.Run("not found", func(t *testing.T) {
		req := UpdateItemRequest{
			Title: "New Title",
		}

		w := performRequest(router, "PUT", "/api/v1/items/non_existent", req)

		if w.Code != http.StatusNotFound {
			t.Errorf("expected status 404, got %d", w.Code)
		}
	})

	t.Run("invalid json", func(t *testing.T) {
		req, _ := http.NewRequest("PUT", "/api/v1/items/test_item_1", bytes.NewBufferString("invalid"))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}
	})
}

// TestDeleteItemAPI 测试删除物品 API
func TestDeleteItemAPI(t *testing.T) {
	router, _, repo := setupTestRouter()

	// 添加测试数据
	testItem := &interfaces.Item{
		ID:     "test_item_delete",
		Type:   "movie",
		Title:  "To Be Deleted",
		Status: "active",
	}
	repo.items[testItem.ID] = testItem

	t.Run("success", func(t *testing.T) {
		w := performRequest(router, "DELETE", "/api/v1/items/test_item_delete", nil)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		resp := parseResponse(w)
		if resp.Code != 0 {
			t.Errorf("expected code 0, got %d", resp.Code)
		}

		// 验证已删除
		if _, ok := repo.items["test_item_delete"]; ok {
			t.Error("expected item to be deleted")
		}
	})
}

// TestGetSimilarItemsAPI 测试获取相似物品 API
func TestGetSimilarItemsAPI(t *testing.T) {
	router, _, _ := setupTestRouter()

	t.Run("milvus not available", func(t *testing.T) {
		w := performRequest(router, "GET", "/api/v1/items/test_item_1/similar?top_k=10", nil)

		// 由于 Milvus 不可用，应该返回 503
		if w.Code != http.StatusServiceUnavailable && w.Code != http.StatusNotFound {
			t.Errorf("expected status 503 or 404, got %d", w.Code)
		}
	})
}

// TestGetItemStatsAPI 测试获取物品统计 API
func TestGetItemStatsAPI(t *testing.T) {
	router, _, repo := setupTestRouter()

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
		w := performRequest(router, "GET", "/api/v1/items/test_item_1/stats", nil)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		resp := parseResponse(w)
		if resp.Code != 0 {
			t.Errorf("expected code 0, got %d", resp.Code)
		}
	})
}

// TestUpdateItemStatsAPI 测试更新物品统计 API
func TestUpdateItemStatsAPI(t *testing.T) {
	router, _, _ := setupTestRouter()

	t.Run("success", func(t *testing.T) {
		req := UpdateStatsRequest{
			Action: "view",
		}

		w := performRequest(router, "POST", "/api/v1/items/test_item_1/stats", req)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}
	})

	t.Run("missing action", func(t *testing.T) {
		req := struct{}{}

		w := performRequest(router, "POST", "/api/v1/items/test_item_1/stats", req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}
	})
}

// TestBatchGetItemsAPI 测试批量获取物品 API
func TestBatchGetItemsAPI(t *testing.T) {
	router, _, repo := setupTestRouter()

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
		req := BatchGetItemsRequest{
			IDs: []string{"1", "2", "3"},
		}

		w := performRequest(router, "POST", "/api/v1/items/batch", req)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		resp := parseResponse(w)
		if resp.Code != 0 {
			t.Errorf("expected code 0, got %d", resp.Code)
		}
	})

	t.Run("empty ids", func(t *testing.T) {
		req := BatchGetItemsRequest{
			IDs: []string{},
		}

		w := performRequest(router, "POST", "/api/v1/items/batch", req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}
	})

	t.Run("too many ids", func(t *testing.T) {
		ids := make([]string, 101)
		for i := range ids {
			ids[i] = string(rune('a' + i%26))
		}
		req := BatchGetItemsRequest{
			IDs: ids,
		}

		w := performRequest(router, "POST", "/api/v1/items/batch", req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}
	})

	t.Run("missing ids field", func(t *testing.T) {
		w := performRequest(router, "POST", "/api/v1/items/batch", struct{}{})

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}
	})
}

// TestRegisterRoutes 测试路由注册
func TestRegisterRoutes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()

	repo := newMockItemRepository()
	cache := newMockCache()
	service := item.NewService(repo, nil, cache, nil)
	handler := NewHandler(service)

	apiV1 := router.Group("/api/v1")
	handler.RegisterRoutes(apiV1)

	// 验证所有路由都已注册
	routes := router.Routes()

	expectedRoutes := map[string]string{
		"POST:/api/v1/items":            "CreateItem",
		"GET:/api/v1/items":             "ListItems",
		"GET:/api/v1/items/search":      "SearchItems",
		"POST:/api/v1/items/batch":      "BatchGetItems",
		"GET:/api/v1/items/:id":         "GetItem",
		"PUT:/api/v1/items/:id":         "UpdateItem",
		"DELETE:/api/v1/items/:id":      "DeleteItem",
		"GET:/api/v1/items/:id/similar": "GetSimilarItems",
		"GET:/api/v1/items/:id/stats":   "GetItemStats",
		"POST:/api/v1/items/:id/stats":  "UpdateItemStats",
	}

	registeredRoutes := make(map[string]bool)
	for _, route := range routes {
		key := route.Method + ":" + route.Path
		registeredRoutes[key] = true
	}

	for key := range expectedRoutes {
		if !registeredRoutes[key] {
			t.Errorf("expected route %s to be registered", key)
		}
	}
}

// =============================================================================
// 基准测试
// =============================================================================

// BenchmarkCreateItemAPI 基准测试：创建物品 API
func BenchmarkCreateItemAPI(b *testing.B) {
	router, _, _ := setupTestRouter()

	req := CreateItemRequest{
		Type:        "movie",
		Title:       "Benchmark Movie",
		Description: "A benchmark movie",
		Category:    "action",
	}
	body, _ := json.Marshal(req)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		r, _ := http.NewRequest("POST", "/api/v1/items", bytes.NewBuffer(body))
		r.Header.Set("Content-Type", "application/json")
		router.ServeHTTP(w, r)
	}
}

// BenchmarkGetItemAPI 基准测试：获取物品 API
func BenchmarkGetItemAPI(b *testing.B) {
	router, _, repo := setupTestRouter()

	repo.items["bench_item"] = &interfaces.Item{
		ID:     "bench_item",
		Type:   "movie",
		Title:  "Benchmark Movie",
		Status: "active",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		r, _ := http.NewRequest("GET", "/api/v1/items/bench_item", nil)
		router.ServeHTTP(w, r)
	}
}

// BenchmarkListItemsAPI 基准测试：列出物品 API
func BenchmarkListItemsAPI(b *testing.B) {
	router, _, repo := setupTestRouter()

	for i := 0; i < 100; i++ {
		id := string(rune('a' + i%26)) + string(rune('0'+i/26))
		repo.items[id] = &interfaces.Item{
			ID:     id,
			Type:   "movie",
			Title:  "Movie " + id,
			Status: "active",
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		r, _ := http.NewRequest("GET", "/api/v1/items?page=1&page_size=20", nil)
		router.ServeHTTP(w, r)
	}
}

