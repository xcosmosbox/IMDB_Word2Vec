package integration

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"recommend-system/internal/interfaces"
	"recommend-system/tests/fixtures"
	"recommend-system/tests/mocks"
)

// =============================================================================
// 物品服务集成测试
// =============================================================================

// ItemServiceIntegration 物品服务集成测试结构
type ItemServiceIntegration struct {
	itemRepo *mocks.MockItemRepository
	cache    *mocks.MockCache
}

// setupItemTestEnv 设置物品服务测试环境
func setupItemTestEnv(t *testing.T) (*ItemServiceIntegration, func()) {
	itemRepo := mocks.NewMockItemRepository()
	cache := mocks.NewMockCache()

	env := &ItemServiceIntegration{
		itemRepo: itemRepo,
		cache:    cache,
	}

	// 加载测试数据
	for _, item := range fixtures.GetAllTestItems() {
		itemRepo.SetItem(item)
	}

	// 加载统计数据
	for _, stats := range fixtures.TestItemStats {
		itemRepo.SetItemStats(stats)
	}

	cleanup := func() {
		cache.Reset()
		itemRepo.Reset()
	}

	return env, cleanup
}

// TestItemIntegration_CRUD 测试物品 CRUD 完整流程
func TestItemIntegration_CRUD(t *testing.T) {
	env, cleanup := setupItemTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 1. Create - 创建新物品
	newItem := &interfaces.Item{
		ID:          "item_integration_test",
		Type:        "movie",
		Title:       "Integration Test Movie",
		Description: "A movie for integration testing",
		Category:    "test",
		Tags:        []string{"integration", "test"},
		Status:      "active",
	}

	err := env.itemRepo.Create(ctx, newItem)
	require.NoError(t, err)
	t.Logf("Created item: %s", newItem.ID)

	// 2. Read - 读取物品
	readItem, err := env.itemRepo.GetByID(ctx, newItem.ID)
	require.NoError(t, err)
	assert.Equal(t, newItem.ID, readItem.ID)
	assert.Equal(t, newItem.Title, readItem.Title)

	// 3. Update - 更新物品
	readItem.Title = "Updated Integration Test Movie"
	readItem.Tags = append(readItem.Tags, "updated")
	err = env.itemRepo.Update(ctx, readItem)
	require.NoError(t, err)

	updatedItem, err := env.itemRepo.GetByID(ctx, newItem.ID)
	require.NoError(t, err)
	assert.Equal(t, "Updated Integration Test Movie", updatedItem.Title)
	assert.Contains(t, updatedItem.Tags, "updated")

	// 4. Delete - 删除物品
	err = env.itemRepo.Delete(ctx, newItem.ID)
	require.NoError(t, err)

	// 5. Verify deletion
	_, err = env.itemRepo.GetByID(ctx, newItem.ID)
	assert.Error(t, err)
}

// TestItemIntegration_BatchOperations 测试批量操作
func TestItemIntegration_BatchOperations(t *testing.T) {
	env, cleanup := setupItemTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 批量获取
	itemIDs := []string{"item_001", "item_002", "item_003", "non_existent"}
	items, err := env.itemRepo.GetByIDs(ctx, itemIDs)
	require.NoError(t, err)

	// 应该只返回存在的物品
	assert.Equal(t, 3, len(items))

	// 验证返回的物品
	foundIDs := make(map[string]bool)
	for _, item := range items {
		foundIDs[item.ID] = true
	}

	assert.True(t, foundIDs["item_001"])
	assert.True(t, foundIDs["item_002"])
	assert.True(t, foundIDs["item_003"])
}

// TestItemIntegration_ListAndPagination 测试列表和分页
func TestItemIntegration_ListAndPagination(t *testing.T) {
	env, cleanup := setupItemTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 测试基本列表
	items, total, err := env.itemRepo.List(ctx, "", "", 1, 5)
	require.NoError(t, err)
	assert.LessOrEqual(t, len(items), 5)
	assert.GreaterOrEqual(t, total, int64(len(items)))
	t.Logf("Total items: %d, Page 1 items: %d", total, len(items))

	// 测试按类型过滤
	movieItems, _, err := env.itemRepo.List(ctx, "movie", "", 1, 100)
	require.NoError(t, err)
	for _, item := range movieItems {
		assert.Equal(t, "movie", item.Type)
	}

	// 测试按类目过滤
	actionItems, _, err := env.itemRepo.List(ctx, "", "action", 1, 100)
	require.NoError(t, err)
	for _, item := range actionItems {
		assert.Equal(t, "action", item.Category)
	}

	// 测试分页
	page1, _, err := env.itemRepo.List(ctx, "", "", 1, 3)
	require.NoError(t, err)

	page2, _, err := env.itemRepo.List(ctx, "", "", 2, 3)
	require.NoError(t, err)

	// 确保两页没有重复
	page1IDs := make(map[string]bool)
	for _, item := range page1 {
		page1IDs[item.ID] = true
	}

	for _, item := range page2 {
		assert.False(t, page1IDs[item.ID], "Item %s appears in both pages", item.ID)
	}
}

// TestItemIntegration_Search 测试搜索功能
func TestItemIntegration_Search(t *testing.T) {
	env, cleanup := setupItemTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 搜索存在的内容
	results, err := env.itemRepo.Search(ctx, "Matrix", 10)
	require.NoError(t, err)
	assert.NotEmpty(t, results)

	// 验证搜索结果包含关键词
	found := false
	for _, item := range results {
		if item.Title == "The Matrix" {
			found = true
			break
		}
	}
	assert.True(t, found)

	// 搜索不存在的内容
	emptyResults, err := env.itemRepo.Search(ctx, "XYZABC123NonExistent", 10)
	require.NoError(t, err)
	assert.Empty(t, emptyResults)

	// 搜索限制
	limitedResults, err := env.itemRepo.Search(ctx, "the", 2)
	require.NoError(t, err)
	assert.LessOrEqual(t, len(limitedResults), 2)
}

// TestItemIntegration_Statistics 测试统计功能
func TestItemIntegration_Statistics(t *testing.T) {
	env, cleanup := setupItemTestEnv(t)
	defer cleanup()

	ctx := context.Background()
	itemID := "item_stats_test"

	// 创建测试物品
	item := &interfaces.Item{
		ID:       itemID,
		Type:     "movie",
		Title:    "Stats Test Movie",
		Category: "test",
		Status:   "active",
	}
	err := env.itemRepo.Create(ctx, item)
	require.NoError(t, err)

	// 增加统计
	for i := 0; i < 10; i++ {
		_ = env.itemRepo.IncrementStats(ctx, itemID, "view")
	}
	for i := 0; i < 5; i++ {
		_ = env.itemRepo.IncrementStats(ctx, itemID, "click")
	}
	for i := 0; i < 3; i++ {
		_ = env.itemRepo.IncrementStats(ctx, itemID, "like")
	}
	_ = env.itemRepo.IncrementStats(ctx, itemID, "share")

	// 获取统计
	stats, err := env.itemRepo.GetStats(ctx, itemID)
	require.NoError(t, err)

	assert.Equal(t, int64(10), stats.ViewCount)
	assert.Equal(t, int64(5), stats.ClickCount)
	assert.Equal(t, int64(3), stats.LikeCount)
	assert.Equal(t, int64(1), stats.ShareCount)
}

// TestItemIntegration_PopularItems 测试热门物品
func TestItemIntegration_PopularItems(t *testing.T) {
	env, cleanup := setupItemTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 获取所有类目的热门物品
	popular, err := env.itemRepo.GetPopularByCategories(ctx, []string{}, 10)
	require.NoError(t, err)
	assert.NotEmpty(t, popular)
	t.Logf("Got %d popular items", len(popular))

	// 获取特定类目的热门物品
	actionPopular, err := env.itemRepo.GetPopularByCategories(ctx, []string{"action"}, 5)
	require.NoError(t, err)
	for _, item := range actionPopular {
		assert.Equal(t, "action", item.Category)
	}

	// 验证排序（基于统计）
	if len(popular) >= 2 {
		// 第一个物品的统计应该不低于第二个
		stats1, _ := env.itemRepo.GetStats(ctx, popular[0].ID)
		stats2, _ := env.itemRepo.GetStats(ctx, popular[1].ID)
		
		if stats1 != nil && stats2 != nil {
			score1 := stats1.ViewCount + stats1.ClickCount*2 + stats1.LikeCount*3
			score2 := stats2.ViewCount + stats2.ClickCount*2 + stats2.LikeCount*3
			assert.GreaterOrEqual(t, score1, score2)
		}
	}
}

// TestItemIntegration_ConcurrentAccess 测试并发访问
func TestItemIntegration_ConcurrentAccess(t *testing.T) {
	env, cleanup := setupItemTestEnv(t)
	defer cleanup()

	ctx := context.Background()
	const numGoroutines = 50

	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines*3)

	// 并发读取
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := env.itemRepo.GetByID(ctx, "item_001")
			if err != nil {
				errors <- err
			}
		}()
	}

	// 并发搜索
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := env.itemRepo.Search(ctx, "movie", 10)
			if err != nil {
				errors <- err
			}
		}()
	}

	// 并发统计增加
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := env.itemRepo.IncrementStats(ctx, "item_001", "view")
			if err != nil {
				errors <- err
			}
		}()
	}

	wg.Wait()
	close(errors)

	// 验证没有错误
	errorCount := 0
	for err := range errors {
		t.Errorf("Concurrent access error: %v", err)
		errorCount++
	}
	assert.Equal(t, 0, errorCount)

	// 验证统计正确增加
	stats, err := env.itemRepo.GetStats(ctx, "item_001")
	require.NoError(t, err)
	assert.Equal(t, int64(numGoroutines), stats.ViewCount)
}

// TestItemIntegration_TagsHandling 测试标签处理
func TestItemIntegration_TagsHandling(t *testing.T) {
	env, cleanup := setupItemTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 创建带标签的物品
	item := &interfaces.Item{
		ID:       "item_tags_test",
		Type:     "movie",
		Title:    "Tags Test Movie",
		Category: "test",
		Tags:     []string{"tag1", "tag2", "tag3"},
		Status:   "active",
	}
	err := env.itemRepo.Create(ctx, item)
	require.NoError(t, err)

	// 读取并验证标签
	retrieved, err := env.itemRepo.GetByID(ctx, item.ID)
	require.NoError(t, err)
	assert.ElementsMatch(t, item.Tags, retrieved.Tags)

	// 更新标签
	retrieved.Tags = []string{"newtag1", "newtag2"}
	err = env.itemRepo.Update(ctx, retrieved)
	require.NoError(t, err)

	// 验证更新后的标签
	updated, err := env.itemRepo.GetByID(ctx, item.ID)
	require.NoError(t, err)
	assert.ElementsMatch(t, []string{"newtag1", "newtag2"}, updated.Tags)
}

// TestItemIntegration_TypeFiltering 测试类型过滤
func TestItemIntegration_TypeFiltering(t *testing.T) {
	env, cleanup := setupItemTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 按类型获取物品
	itemTypes := []string{"movie", "product", "video", "article"}

	for _, itemType := range itemTypes {
		items, _, err := env.itemRepo.List(ctx, itemType, "", 1, 100)
		require.NoError(t, err)

		for _, item := range items {
			assert.Equal(t, itemType, item.Type,
				"Expected type %s but got %s for item %s", itemType, item.Type, item.ID)
		}

		t.Logf("Found %d items of type %s", len(items), itemType)
	}
}

// TestItemIntegration_DataIsolation 测试数据隔离
func TestItemIntegration_DataIsolation(t *testing.T) {
	env, cleanup := setupItemTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 获取物品
	item, err := env.itemRepo.GetByID(ctx, "item_001")
	require.NoError(t, err)

	// 修改返回的物品
	originalTitle := item.Title
	item.Title = "Modified Title"

	// 再次获取，验证原数据未被修改
	itemAgain, err := env.itemRepo.GetByID(ctx, "item_001")
	require.NoError(t, err)
	assert.Equal(t, originalTitle, itemAgain.Title)
}

// TestItemIntegration_EmptyResults 测试空结果处理
func TestItemIntegration_EmptyResults(t *testing.T) {
	env, cleanup := setupItemTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 搜索不存在的内容
	results, err := env.itemRepo.Search(ctx, "xyz123nonexistent", 10)
	require.NoError(t, err)
	assert.Empty(t, results)
	assert.NotNil(t, results) // 应该返回空切片而不是 nil

	// 列表不存在的类型
	items, total, err := env.itemRepo.List(ctx, "nonexistent_type", "", 1, 10)
	require.NoError(t, err)
	assert.Empty(t, items)
	assert.Equal(t, int64(0), total)

	// 批量获取不存在的物品
	batchResult, err := env.itemRepo.GetByIDs(ctx, []string{"non1", "non2", "non3"})
	require.NoError(t, err)
	assert.Empty(t, batchResult)
}

