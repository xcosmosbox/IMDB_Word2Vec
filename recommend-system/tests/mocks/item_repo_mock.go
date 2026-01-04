package mocks

import (
	"context"
	"sort"
	"strings"
	"sync"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// MockItemRepository - 物品仓库 Mock 实现
// =============================================================================

// MockItemRepository Mock 物品仓库
//
// 实现 interfaces.ItemRepository 接口
// 使用内存存储模拟数据库操作，支持并发安全访问
type MockItemRepository struct {
	mu    sync.RWMutex
	items map[string]*interfaces.Item
	stats map[string]*interfaces.ItemStats

	// 调用计数器
	GetByIDCalls        int
	GetByIDsCalls       int
	CreateCalls         int
	UpdateCalls         int
	DeleteCalls         int
	ListCalls           int
	SearchCalls         int
	GetStatsCalls       int
	IncrementStatsCalls int

	// 可配置的错误
	GetByIDError        error
	CreateError         error
	UpdateError         error
	DeleteError         error
	ListError           error
	SearchError         error
	GetStatsError       error
	IncrementStatsError error
}

// NewMockItemRepository 创建 Mock 物品仓库实例
func NewMockItemRepository() *MockItemRepository {
	return &MockItemRepository{
		items: make(map[string]*interfaces.Item),
		stats: make(map[string]*interfaces.ItemStats),
	}
}

// GetByID 根据ID获取物品
//
// 实现 interfaces.ItemRepository.GetByID
func (m *MockItemRepository) GetByID(ctx context.Context, itemID string) (*interfaces.Item, error) {
	m.mu.Lock()
	m.GetByIDCalls++
	m.mu.Unlock()

	if m.GetByIDError != nil {
		return nil, m.GetByIDError
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	item, ok := m.items[itemID]
	if !ok {
		return nil, ErrNotFound
	}

	// 返回副本
	itemCopy := *item
	if item.Tags != nil {
		itemCopy.Tags = make([]string, len(item.Tags))
		copy(itemCopy.Tags, item.Tags)
	}

	return &itemCopy, nil
}

// GetByIDs 批量获取物品
//
// 实现 interfaces.ItemRepository.GetByIDs
func (m *MockItemRepository) GetByIDs(ctx context.Context, itemIDs []string) ([]*interfaces.Item, error) {
	m.mu.Lock()
	m.GetByIDsCalls++
	m.mu.Unlock()

	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]*interfaces.Item, 0, len(itemIDs))
	for _, id := range itemIDs {
		if item, ok := m.items[id]; ok {
			itemCopy := *item
			if item.Tags != nil {
				itemCopy.Tags = make([]string, len(item.Tags))
				copy(itemCopy.Tags, item.Tags)
			}
			result = append(result, &itemCopy)
		}
	}

	return result, nil
}

// Create 创建物品
//
// 实现 interfaces.ItemRepository.Create
func (m *MockItemRepository) Create(ctx context.Context, item *interfaces.Item) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.CreateCalls++

	if m.CreateError != nil {
		return m.CreateError
	}

	if _, exists := m.items[item.ID]; exists {
		return ErrDuplicate
	}

	// 保存副本
	itemCopy := *item
	if item.Tags != nil {
		itemCopy.Tags = make([]string, len(item.Tags))
		copy(itemCopy.Tags, item.Tags)
	}
	m.items[item.ID] = &itemCopy

	// 初始化统计
	m.stats[item.ID] = &interfaces.ItemStats{
		ItemID: item.ID,
	}

	return nil
}

// Update 更新物品
//
// 实现 interfaces.ItemRepository.Update
func (m *MockItemRepository) Update(ctx context.Context, item *interfaces.Item) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.UpdateCalls++

	if m.UpdateError != nil {
		return m.UpdateError
	}

	if _, exists := m.items[item.ID]; !exists {
		return ErrNotFound
	}

	// 保存副本
	itemCopy := *item
	if item.Tags != nil {
		itemCopy.Tags = make([]string, len(item.Tags))
		copy(itemCopy.Tags, item.Tags)
	}
	m.items[item.ID] = &itemCopy

	return nil
}

// Delete 删除物品
//
// 实现 interfaces.ItemRepository.Delete
func (m *MockItemRepository) Delete(ctx context.Context, itemID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.DeleteCalls++

	if m.DeleteError != nil {
		return m.DeleteError
	}

	delete(m.items, itemID)
	delete(m.stats, itemID)

	return nil
}

// List 列出物品
//
// 实现 interfaces.ItemRepository.List
func (m *MockItemRepository) List(ctx context.Context, itemType, category string, page, pageSize int) ([]*interfaces.Item, int64, error) {
	m.mu.Lock()
	m.ListCalls++
	m.mu.Unlock()

	if m.ListError != nil {
		return nil, 0, m.ListError
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	// 过滤
	var filtered []*interfaces.Item
	for _, item := range m.items {
		if itemType != "" && item.Type != itemType {
			continue
		}
		if category != "" && item.Category != category {
			continue
		}
		itemCopy := *item
		filtered = append(filtered, &itemCopy)
	}

	total := int64(len(filtered))

	// 分页
	if page <= 0 {
		page = 1
	}
	if pageSize <= 0 {
		pageSize = 20
	}

	start := (page - 1) * pageSize
	if start >= len(filtered) {
		return []*interfaces.Item{}, total, nil
	}

	end := start + pageSize
	if end > len(filtered) {
		end = len(filtered)
	}

	return filtered[start:end], total, nil
}

// Search 搜索物品
//
// 实现 interfaces.ItemRepository.Search
func (m *MockItemRepository) Search(ctx context.Context, query string, limit int) ([]*interfaces.Item, error) {
	m.mu.Lock()
	m.SearchCalls++
	m.mu.Unlock()

	if m.SearchError != nil {
		return nil, m.SearchError
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	query = strings.ToLower(query)
	var results []*interfaces.Item

	for _, item := range m.items {
		// 简单的标题和描述匹配
		if strings.Contains(strings.ToLower(item.Title), query) ||
			strings.Contains(strings.ToLower(item.Description), query) {
			itemCopy := *item
			results = append(results, &itemCopy)
		}
	}

	// 限制数量
	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

// GetStats 获取物品统计
//
// 实现 interfaces.ItemRepository.GetStats
func (m *MockItemRepository) GetStats(ctx context.Context, itemID string) (*interfaces.ItemStats, error) {
	m.mu.Lock()
	m.GetStatsCalls++
	m.mu.Unlock()

	if m.GetStatsError != nil {
		return nil, m.GetStatsError
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	stats, ok := m.stats[itemID]
	if !ok {
		return nil, ErrNotFound
	}

	statsCopy := *stats
	return &statsCopy, nil
}

// IncrementStats 增加物品统计
//
// 实现 interfaces.ItemRepository.IncrementStats
func (m *MockItemRepository) IncrementStats(ctx context.Context, itemID, action string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.IncrementStatsCalls++

	if m.IncrementStatsError != nil {
		return m.IncrementStatsError
	}

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

// GetPopularByCategories 按类目获取热门物品
//
// 实现 interfaces.ItemRepository.GetPopularByCategories
func (m *MockItemRepository) GetPopularByCategories(ctx context.Context, categories []string, limit int) ([]*interfaces.Item, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	categorySet := make(map[string]bool)
	for _, c := range categories {
		categorySet[c] = true
	}

	type itemWithScore struct {
		item  *interfaces.Item
		score int64
	}

	var items []itemWithScore

	for _, item := range m.items {
		// 过滤类目
		if len(categories) > 0 && !categorySet[item.Category] {
			continue
		}

		// 计算得分（基于统计）
		score := int64(0)
		if stats, ok := m.stats[item.ID]; ok {
			score = stats.ViewCount + stats.ClickCount*2 + stats.LikeCount*3
		}

		itemCopy := *item
		items = append(items, itemWithScore{item: &itemCopy, score: score})
	}

	// 按得分排序
	sort.Slice(items, func(i, j int) bool {
		return items[i].score > items[j].score
	})

	// 限制数量
	if limit > 0 && len(items) > limit {
		items = items[:limit]
	}

	result := make([]*interfaces.Item, len(items))
	for i, iws := range items {
		result[i] = iws.item
	}

	return result, nil
}

// =============================================================================
// 测试辅助方法
// =============================================================================

// SetItem 设置物品数据（测试用）
func (m *MockItemRepository) SetItem(item *interfaces.Item) {
	m.mu.Lock()
	defer m.mu.Unlock()

	itemCopy := *item
	if item.Tags != nil {
		itemCopy.Tags = make([]string, len(item.Tags))
		copy(itemCopy.Tags, item.Tags)
	}
	m.items[item.ID] = &itemCopy

	// 初始化统计
	if _, ok := m.stats[item.ID]; !ok {
		m.stats[item.ID] = &interfaces.ItemStats{ItemID: item.ID}
	}
}

// SetItems 批量设置物品数据（测试用）
func (m *MockItemRepository) SetItems(items []*interfaces.Item) {
	for _, item := range items {
		m.SetItem(item)
	}
}

// SetItemStats 设置物品统计（测试用）
func (m *MockItemRepository) SetItemStats(stats *interfaces.ItemStats) {
	m.mu.Lock()
	defer m.mu.Unlock()

	statsCopy := *stats
	m.stats[stats.ItemID] = &statsCopy
}

// Reset 重置所有状态（测试用）
func (m *MockItemRepository) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.items = make(map[string]*interfaces.Item)
	m.stats = make(map[string]*interfaces.ItemStats)
	m.GetByIDCalls = 0
	m.GetByIDsCalls = 0
	m.CreateCalls = 0
	m.UpdateCalls = 0
	m.DeleteCalls = 0
	m.ListCalls = 0
	m.SearchCalls = 0
	m.GetStatsCalls = 0
	m.IncrementStatsCalls = 0
	m.GetByIDError = nil
	m.CreateError = nil
	m.UpdateError = nil
	m.DeleteError = nil
	m.ListError = nil
	m.SearchError = nil
	m.GetStatsError = nil
	m.IncrementStatsError = nil
}

// GetItemCount 获取物品数量（测试用）
func (m *MockItemRepository) GetItemCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return len(m.items)
}

