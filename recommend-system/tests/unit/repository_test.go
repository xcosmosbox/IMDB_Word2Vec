package unit

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"recommend-system/internal/interfaces"
	"recommend-system/tests/fixtures"
	"recommend-system/tests/mocks"
)

// =============================================================================
// UserRepository Mock 测试
// =============================================================================

// TestMockUserRepository_GetByID 测试获取用户
func TestMockUserRepository_GetByID(t *testing.T) {
	repo := mocks.NewMockUserRepository()
	ctx := context.Background()

	// 设置测试数据
	testUser := fixtures.GetTestUser("user_001")
	repo.SetUser(testUser)

	// 测试获取存在的用户
	user, err := repo.GetByID(ctx, "user_001")
	require.NoError(t, err)
	assert.Equal(t, testUser.ID, user.ID)
	assert.Equal(t, testUser.Name, user.Name)

	// 测试获取不存在的用户
	_, err = repo.GetByID(ctx, "non_existent")
	assert.Error(t, err)
	assert.Equal(t, mocks.ErrNotFound, err)

	// 验证调用计数
	assert.Equal(t, 2, repo.GetByIDCalls)
}

// TestMockUserRepository_GetByIDs 测试批量获取用户
func TestMockUserRepository_GetByIDs(t *testing.T) {
	repo := mocks.NewMockUserRepository()
	ctx := context.Background()

	// 设置测试数据
	users := fixtures.GetAllTestUsers()
	repo.SetUsers(users)

	// 测试批量获取
	result, err := repo.GetByIDs(ctx, []string{"user_001", "user_002", "non_existent"})
	require.NoError(t, err)
	assert.Len(t, result, 2) // 只返回存在的用户
}

// TestMockUserRepository_Create 测试创建用户
func TestMockUserRepository_Create(t *testing.T) {
	repo := mocks.NewMockUserRepository()
	ctx := context.Background()

	user := fixtures.CreateTestUser("user_test", "Test", "test@example.com", 25, "male")

	// 测试创建
	err := repo.Create(ctx, user)
	require.NoError(t, err)

	// 验证已创建
	created, err := repo.GetByID(ctx, "user_test")
	require.NoError(t, err)
	assert.Equal(t, user.Name, created.Name)

	// 测试重复创建
	err = repo.Create(ctx, user)
	assert.Equal(t, mocks.ErrDuplicate, err)
}

// TestMockUserRepository_Update 测试更新用户
func TestMockUserRepository_Update(t *testing.T) {
	repo := mocks.NewMockUserRepository()
	ctx := context.Background()

	// 设置测试数据
	user := fixtures.GetTestUser("user_001")
	repo.SetUser(user)

	// 更新用户
	user.Name = "Updated Name"
	err := repo.Update(ctx, user)
	require.NoError(t, err)

	// 验证更新
	updated, err := repo.GetByID(ctx, "user_001")
	require.NoError(t, err)
	assert.Equal(t, "Updated Name", updated.Name)

	// 测试更新不存在的用户
	nonExistent := fixtures.CreateTestUser("non_existent", "Test", "test@example.com", 25, "male")
	err = repo.Update(ctx, nonExistent)
	assert.Equal(t, mocks.ErrNotFound, err)
}

// TestMockUserRepository_Delete 测试删除用户
func TestMockUserRepository_Delete(t *testing.T) {
	repo := mocks.NewMockUserRepository()
	ctx := context.Background()

	// 设置测试数据
	user := fixtures.GetTestUser("user_001")
	repo.SetUser(user)

	// 删除用户
	err := repo.Delete(ctx, "user_001")
	require.NoError(t, err)

	// 验证已删除
	_, err = repo.GetByID(ctx, "user_001")
	assert.Equal(t, mocks.ErrNotFound, err)
}

// TestMockUserRepository_Behaviors 测试行为相关操作
func TestMockUserRepository_Behaviors(t *testing.T) {
	repo := mocks.NewMockUserRepository()
	ctx := context.Background()

	// 添加行为
	behavior := &interfaces.UserBehavior{
		UserID:    "user_001",
		ItemID:    "item_001",
		Action:    "click",
		Timestamp: time.Now(),
	}
	err := repo.AddBehavior(ctx, behavior)
	require.NoError(t, err)

	// 获取行为
	behaviors, err := repo.GetBehaviors(ctx, "user_001", 10)
	require.NoError(t, err)
	assert.Len(t, behaviors, 1)
	assert.Equal(t, "click", behaviors[0].Action)

	// 添加更多行为
	for i := 0; i < 5; i++ {
		behavior := fixtures.CreateTestBehavior("user_001", "item_002", "view")
		_ = repo.AddBehavior(ctx, behavior)
	}

	// 测试 limit
	behaviors, err = repo.GetBehaviors(ctx, "user_001", 3)
	require.NoError(t, err)
	assert.Len(t, behaviors, 3)
}

// TestMockUserRepository_GetUserItemInteractions 测试获取用户物品交互
func TestMockUserRepository_GetUserItemInteractions(t *testing.T) {
	repo := mocks.NewMockUserRepository()
	ctx := context.Background()

	// 添加多个行为
	behaviors := []*interfaces.UserBehavior{
		fixtures.CreateTestBehavior("user_001", "item_001", "view"),
		fixtures.CreateTestBehavior("user_001", "item_001", "click"),
		fixtures.CreateTestBehavior("user_001", "item_002", "view"),
	}
	for _, b := range behaviors {
		_ = repo.AddBehavior(ctx, b)
	}

	// 获取特定物品的交互
	interactions, err := repo.GetUserItemInteractions(ctx, "user_001", "item_001")
	require.NoError(t, err)
	assert.Len(t, interactions, 2)
}

// =============================================================================
// ItemRepository Mock 测试
// =============================================================================

// TestMockItemRepository_GetByID 测试获取物品
func TestMockItemRepository_GetByID(t *testing.T) {
	repo := mocks.NewMockItemRepository()
	ctx := context.Background()

	// 设置测试数据
	testItem := fixtures.GetTestItem("item_001")
	repo.SetItem(testItem)

	// 测试获取存在的物品
	item, err := repo.GetByID(ctx, "item_001")
	require.NoError(t, err)
	assert.Equal(t, testItem.ID, item.ID)
	assert.Equal(t, testItem.Title, item.Title)

	// 测试获取不存在的物品
	_, err = repo.GetByID(ctx, "non_existent")
	assert.Error(t, err)
}

// TestMockItemRepository_GetByIDs 测试批量获取物品
func TestMockItemRepository_GetByIDs(t *testing.T) {
	repo := mocks.NewMockItemRepository()
	ctx := context.Background()

	// 设置测试数据
	items := fixtures.GetAllTestItems()
	repo.SetItems(items)

	// 测试批量获取
	result, err := repo.GetByIDs(ctx, []string{"item_001", "item_002", "non_existent"})
	require.NoError(t, err)
	assert.Len(t, result, 2)
}

// TestMockItemRepository_Create 测试创建物品
func TestMockItemRepository_Create(t *testing.T) {
	repo := mocks.NewMockItemRepository()
	ctx := context.Background()

	item := fixtures.CreateTestItem("item_test", "movie", "Test Movie", "action", []string{"test"})

	// 测试创建
	err := repo.Create(ctx, item)
	require.NoError(t, err)

	// 验证已创建
	created, err := repo.GetByID(ctx, "item_test")
	require.NoError(t, err)
	assert.Equal(t, item.Title, created.Title)

	// 测试重复创建
	err = repo.Create(ctx, item)
	assert.Equal(t, mocks.ErrDuplicate, err)
}

// TestMockItemRepository_List 测试列表物品
func TestMockItemRepository_List(t *testing.T) {
	repo := mocks.NewMockItemRepository()
	ctx := context.Background()

	// 设置测试数据
	items := fixtures.GetAllTestItems()
	repo.SetItems(items)

	// 测试不带过滤条件
	result, total, err := repo.List(ctx, "", "", 1, 10)
	require.NoError(t, err)
	assert.Greater(t, total, int64(0))
	assert.LessOrEqual(t, len(result), 10)

	// 测试按类型过滤
	result, total, err = repo.List(ctx, "movie", "", 1, 10)
	require.NoError(t, err)
	for _, item := range result {
		assert.Equal(t, "movie", item.Type)
	}

	// 测试按类目过滤
	result, total, err = repo.List(ctx, "", "action", 1, 10)
	require.NoError(t, err)
	for _, item := range result {
		assert.Equal(t, "action", item.Category)
	}
}

// TestMockItemRepository_Search 测试搜索物品
func TestMockItemRepository_Search(t *testing.T) {
	repo := mocks.NewMockItemRepository()
	ctx := context.Background()

	// 设置测试数据
	items := fixtures.GetAllTestItems()
	repo.SetItems(items)

	// 测试搜索
	result, err := repo.Search(ctx, "Matrix", 10)
	require.NoError(t, err)
	assert.NotEmpty(t, result)
	assert.Contains(t, result[0].Title, "Matrix")

	// 测试空结果
	result, err = repo.Search(ctx, "NonExistentMovie123", 10)
	require.NoError(t, err)
	assert.Empty(t, result)
}

// TestMockItemRepository_Stats 测试物品统计
func TestMockItemRepository_Stats(t *testing.T) {
	repo := mocks.NewMockItemRepository()
	ctx := context.Background()

	// 设置测试数据
	item := fixtures.GetTestItem("item_001")
	repo.SetItem(item)

	// 增加统计
	_ = repo.IncrementStats(ctx, "item_001", "view")
	_ = repo.IncrementStats(ctx, "item_001", "view")
	_ = repo.IncrementStats(ctx, "item_001", "click")
	_ = repo.IncrementStats(ctx, "item_001", "like")

	// 获取统计
	stats, err := repo.GetStats(ctx, "item_001")
	require.NoError(t, err)
	assert.Equal(t, int64(2), stats.ViewCount)
	assert.Equal(t, int64(1), stats.ClickCount)
	assert.Equal(t, int64(1), stats.LikeCount)
}

// TestMockItemRepository_GetPopularByCategories 测试按类目获取热门物品
func TestMockItemRepository_GetPopularByCategories(t *testing.T) {
	repo := mocks.NewMockItemRepository()
	ctx := context.Background()

	// 设置测试数据
	items := fixtures.GetAllTestItems()
	repo.SetItems(items)

	// 设置统计数据
	for _, stats := range fixtures.TestItemStats {
		repo.SetItemStats(stats)
	}

	// 测试获取热门物品
	result, err := repo.GetPopularByCategories(ctx, []string{"action", "thriller"}, 5)
	require.NoError(t, err)
	assert.LessOrEqual(t, len(result), 5)
}

// =============================================================================
// Cache Mock 测试
// =============================================================================

// TestMockCache_GetSet 测试缓存获取和设置
func TestMockCache_GetSet(t *testing.T) {
	cache := mocks.NewMockCache()
	ctx := context.Background()

	// 测试设置
	err := cache.Set(ctx, "key1", "value1", time.Hour)
	require.NoError(t, err)

	// 测试获取
	var value string
	err = cache.Get(ctx, "key1", &value)
	require.NoError(t, err)
	assert.Equal(t, "value1", value)

	// 测试获取不存在的 key
	err = cache.Get(ctx, "non_existent", &value)
	assert.Equal(t, mocks.ErrCacheMiss, err)
}

// TestMockCache_Expiration 测试缓存过期
func TestMockCache_Expiration(t *testing.T) {
	cache := mocks.NewMockCache()
	ctx := context.Background()

	// 设置短期缓存
	err := cache.Set(ctx, "key1", "value1", 10*time.Millisecond)
	require.NoError(t, err)

	// 立即获取应该成功
	var value string
	err = cache.Get(ctx, "key1", &value)
	require.NoError(t, err)

	// 等待过期
	time.Sleep(20 * time.Millisecond)

	// 过期后获取应该失败
	err = cache.Get(ctx, "key1", &value)
	assert.Equal(t, mocks.ErrCacheMiss, err)
}

// TestMockCache_Delete 测试缓存删除
func TestMockCache_Delete(t *testing.T) {
	cache := mocks.NewMockCache()
	ctx := context.Background()

	// 设置并删除
	_ = cache.Set(ctx, "key1", "value1", time.Hour)
	err := cache.Delete(ctx, "key1")
	require.NoError(t, err)

	// 验证已删除
	var value string
	err = cache.Get(ctx, "key1", &value)
	assert.Equal(t, mocks.ErrCacheMiss, err)
}

// TestMockCache_Exists 测试缓存存在检查
func TestMockCache_Exists(t *testing.T) {
	cache := mocks.NewMockCache()
	ctx := context.Background()

	// 设置
	_ = cache.Set(ctx, "key1", "value1", time.Hour)

	// 测试存在
	exists, err := cache.Exists(ctx, "key1")
	require.NoError(t, err)
	assert.True(t, exists)

	// 测试不存在
	exists, err = cache.Exists(ctx, "non_existent")
	require.NoError(t, err)
	assert.False(t, exists)
}

// TestMockCache_MGetMSet 测试批量操作
func TestMockCache_MGetMSet(t *testing.T) {
	cache := mocks.NewMockCache()
	ctx := context.Background()

	// 批量设置
	kvs := map[string]interface{}{
		"key1": "value1",
		"key2": "value2",
		"key3": "value3",
	}
	err := cache.MSet(ctx, kvs, time.Hour)
	require.NoError(t, err)

	// 批量获取
	results, err := cache.MGet(ctx, []string{"key1", "key2", "non_existent"})
	require.NoError(t, err)
	assert.Len(t, results, 3)
	assert.NotNil(t, results[0])
	assert.NotNil(t, results[1])
	assert.Nil(t, results[2]) // 不存在的 key
}

// TestMockCache_ComplexTypes 测试复杂类型
func TestMockCache_ComplexTypes(t *testing.T) {
	cache := mocks.NewMockCache()
	ctx := context.Background()

	// 测试结构体
	user := fixtures.GetTestUser("user_001")
	err := cache.Set(ctx, "user:001", user, time.Hour)
	require.NoError(t, err)

	var cached interfaces.User
	err = cache.Get(ctx, "user:001", &cached)
	require.NoError(t, err)
	assert.Equal(t, user.ID, cached.ID)
	assert.Equal(t, user.Name, cached.Name)

	// 测试切片
	items := fixtures.GetAllTestItems()
	err = cache.Set(ctx, "items", items, time.Hour)
	require.NoError(t, err)

	var cachedItems []*interfaces.Item
	err = cache.Get(ctx, "items", &cachedItems)
	require.NoError(t, err)
	assert.Len(t, cachedItems, len(items))
}

// =============================================================================
// RecommendRepository Mock 测试
// =============================================================================

// TestMockRecommendRepository_LogRecommendation 测试记录推荐日志
func TestMockRecommendRepository_LogRecommendation(t *testing.T) {
	repo := mocks.NewMockRecommendRepository()
	ctx := context.Background()

	log := &interfaces.RecommendLog{
		RequestID:       "req_001",
		UserID:          "user_001",
		Recommendations: []string{"item_001", "item_002"},
		Strategy:        "model",
		Timestamp:       time.Now(),
	}

	err := repo.LogRecommendation(ctx, log)
	require.NoError(t, err)
	assert.Equal(t, 1, repo.LogRecommendationCalls)
}

// TestMockRecommendRepository_GetRecommendationLogs 测试获取推荐日志
func TestMockRecommendRepository_GetRecommendationLogs(t *testing.T) {
	repo := mocks.NewMockRecommendRepository()
	ctx := context.Background()

	// 添加日志
	for i := 0; i < 5; i++ {
		log := &interfaces.RecommendLog{
			RequestID:       "req_00" + string(rune('0'+i)),
			UserID:          "user_001",
			Recommendations: []string{"item_001"},
			Strategy:        "model",
			Timestamp:       time.Now(),
		}
		_ = repo.LogRecommendation(ctx, log)
	}

	// 获取日志
	logs, err := repo.GetRecommendationLogs(ctx, "user_001", 3)
	require.NoError(t, err)
	assert.LessOrEqual(t, len(logs), 3)
}

// TestMockRecommendRepository_RecordExposure 测试记录曝光
func TestMockRecommendRepository_RecordExposure(t *testing.T) {
	repo := mocks.NewMockRecommendRepository()
	ctx := context.Background()

	err := repo.RecordExposure(ctx, "user_001", "item_001", "req_001")
	require.NoError(t, err)

	err = repo.RecordExposure(ctx, "user_001", "item_002", "req_001")
	require.NoError(t, err)

	exposures := repo.GetExposures("user_001")
	assert.Len(t, exposures, 2)
}

