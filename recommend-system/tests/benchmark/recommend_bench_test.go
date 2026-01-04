// Package benchmark 提供性能测试
//
// 本包包含推荐系统各组件的性能基准测试，
// 用于评估和优化系统性能。
package benchmark

import (
	"context"
	"testing"
	"time"

	"recommend-system/internal/interfaces"
	"recommend-system/tests/fixtures"
	"recommend-system/tests/mocks"
)

// =============================================================================
// 推荐服务性能测试
// =============================================================================

// BenchmarkGetRecommendations 测试推荐接口性能
func BenchmarkGetRecommendations(b *testing.B) {
	// 设置
	mockUserRepo := mocks.NewMockUserRepository()
	mockItemRepo := mocks.NewMockItemRepository()
	mockCache := mocks.NewMockCache()
	mockInferClient := mocks.NewMockInferenceClient()

	// 加载测试数据
	for _, user := range fixtures.GetAllTestUsers() {
		mockUserRepo.SetUser(user)
	}
	for _, user := range fixtures.TestUsers {
		mockUserRepo.SetBehaviors(user.ID, fixtures.GetBehaviorsForUser(user.ID))
	}
	for _, item := range fixtures.GetAllTestItems() {
		mockItemRepo.SetItem(item)
	}

	ctx := context.Background()

	// 预热
	for i := 0; i < 100; i++ {
		_, _ = mockUserRepo.GetByID(ctx, "user_001")
		_, _ = mockUserRepo.GetBehaviors(ctx, "user_001", 50)
		_, _ = mockInferClient.Infer(ctx, &interfaces.ModelInput{
			UserSequence: []int64{1, 2, 3, 4, 5},
		})
	}

	// 基准测试
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 模拟推荐流程
		_, _ = mockUserRepo.GetByID(ctx, "user_001")
		_, _ = mockUserRepo.GetBehaviors(ctx, "user_001", 50)
		_, _ = mockInferClient.Infer(ctx, &interfaces.ModelInput{
			UserSequence: []int64{1, 2, 3, 4, 5},
		})
	}
}

// BenchmarkGetRecommendations_Parallel 测试并发推荐性能
func BenchmarkGetRecommendations_Parallel(b *testing.B) {
	// 设置
	mockUserRepo := mocks.NewMockUserRepository()
	mockItemRepo := mocks.NewMockItemRepository()
	mockInferClient := mocks.NewMockInferenceClient()

	for _, user := range fixtures.GetAllTestUsers() {
		mockUserRepo.SetUser(user)
	}
	for _, user := range fixtures.TestUsers {
		mockUserRepo.SetBehaviors(user.ID, fixtures.GetBehaviorsForUser(user.ID))
	}
	for _, item := range fixtures.GetAllTestItems() {
		mockItemRepo.SetItem(item)
	}

	ctx := context.Background()

	// 并行基准测试
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = mockUserRepo.GetByID(ctx, "user_001")
			_, _ = mockUserRepo.GetBehaviors(ctx, "user_001", 50)
			_, _ = mockInferClient.Infer(ctx, &interfaces.ModelInput{
				UserSequence: []int64{1, 2, 3, 4, 5},
			})
		}
	})
}

// =============================================================================
// 缓存性能测试
// =============================================================================

// BenchmarkCacheGet 测试缓存读取性能
func BenchmarkCacheGet(b *testing.B) {
	cache := mocks.NewMockCache()
	ctx := context.Background()

	// 预设数据
	testData := map[string]string{"key": "value", "name": "test"}
	_ = cache.SetDirect("test_key", testData, time.Hour)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var result map[string]string
		_ = cache.Get(ctx, "test_key", &result)
	}
}

// BenchmarkCacheSet 测试缓存写入性能
func BenchmarkCacheSet(b *testing.B) {
	cache := mocks.NewMockCache()
	ctx := context.Background()

	testData := map[string]string{"key": "value", "name": "test"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cache.Set(ctx, "test_key", testData, time.Hour)
	}
}

// BenchmarkCacheMGet 测试批量缓存读取性能
func BenchmarkCacheMGet(b *testing.B) {
	cache := mocks.NewMockCache()
	ctx := context.Background()

	// 预设数据
	for i := 0; i < 100; i++ {
		key := "key_" + string(rune('0'+i%10)) + string(rune('0'+i/10))
		_ = cache.SetDirect(key, map[string]int{"value": i}, time.Hour)
	}

	keys := make([]string, 10)
	for i := 0; i < 10; i++ {
		keys[i] = "key_" + string(rune('0'+i)) + "0"
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = cache.MGet(ctx, keys)
	}
}

// BenchmarkCacheMSet 测试批量缓存写入性能
func BenchmarkCacheMSet(b *testing.B) {
	cache := mocks.NewMockCache()
	ctx := context.Background()

	kvs := make(map[string]interface{})
	for i := 0; i < 10; i++ {
		key := "key_" + string(rune('0'+i))
		kvs[key] = map[string]int{"value": i}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cache.MSet(ctx, kvs, time.Hour)
	}
}

// BenchmarkCacheGet_Parallel 测试并发缓存读取性能
func BenchmarkCacheGet_Parallel(b *testing.B) {
	cache := mocks.NewMockCache()
	ctx := context.Background()

	testData := fixtures.GetTestUser("user_001")
	_ = cache.SetDirect("user:001", testData, time.Hour)

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			var result interfaces.User
			_ = cache.Get(ctx, "user:001", &result)
		}
	})
}

// =============================================================================
// 用户仓库性能测试
// =============================================================================

// BenchmarkUserRepoGetByID 测试用户获取性能
func BenchmarkUserRepoGetByID(b *testing.B) {
	repo := mocks.NewMockUserRepository()
	ctx := context.Background()

	// 加载测试数据
	for _, user := range fixtures.GetAllTestUsers() {
		repo.SetUser(user)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = repo.GetByID(ctx, "user_001")
	}
}

// BenchmarkUserRepoGetBehaviors 测试用户行为获取性能
func BenchmarkUserRepoGetBehaviors(b *testing.B) {
	repo := mocks.NewMockUserRepository()
	ctx := context.Background()

	// 加载测试数据
	behaviors := make([]*interfaces.UserBehavior, 100)
	for i := 0; i < 100; i++ {
		behaviors[i] = fixtures.CreateTestBehavior("user_001", "item_001", "view")
	}
	repo.SetBehaviors("user_001", behaviors)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = repo.GetBehaviors(ctx, "user_001", 50)
	}
}

// BenchmarkUserRepoAddBehavior 测试行为添加性能
func BenchmarkUserRepoAddBehavior(b *testing.B) {
	repo := mocks.NewMockUserRepository()
	ctx := context.Background()

	behavior := fixtures.CreateTestBehavior("user_001", "item_001", "view")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = repo.AddBehavior(ctx, behavior)
	}
}

// BenchmarkUserRepoGetByID_Parallel 测试并发用户获取性能
func BenchmarkUserRepoGetByID_Parallel(b *testing.B) {
	repo := mocks.NewMockUserRepository()
	ctx := context.Background()

	for _, user := range fixtures.GetAllTestUsers() {
		repo.SetUser(user)
	}

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = repo.GetByID(ctx, "user_001")
		}
	})
}

// =============================================================================
// 物品仓库性能测试
// =============================================================================

// BenchmarkItemRepoGetByID 测试物品获取性能
func BenchmarkItemRepoGetByID(b *testing.B) {
	repo := mocks.NewMockItemRepository()
	ctx := context.Background()

	for _, item := range fixtures.GetAllTestItems() {
		repo.SetItem(item)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = repo.GetByID(ctx, "item_001")
	}
}

// BenchmarkItemRepoGetByIDs 测试批量物品获取性能
func BenchmarkItemRepoGetByIDs(b *testing.B) {
	repo := mocks.NewMockItemRepository()
	ctx := context.Background()

	for _, item := range fixtures.GetAllTestItems() {
		repo.SetItem(item)
	}

	ids := []string{"item_001", "item_002", "item_003", "item_004", "item_005"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = repo.GetByIDs(ctx, ids)
	}
}

// BenchmarkItemRepoSearch 测试物品搜索性能
func BenchmarkItemRepoSearch(b *testing.B) {
	repo := mocks.NewMockItemRepository()
	ctx := context.Background()

	for _, item := range fixtures.GetAllTestItems() {
		repo.SetItem(item)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = repo.Search(ctx, "movie", 10)
	}
}

// BenchmarkItemRepoList 测试物品列表性能
func BenchmarkItemRepoList(b *testing.B) {
	repo := mocks.NewMockItemRepository()
	ctx := context.Background()

	for _, item := range fixtures.GetAllTestItems() {
		repo.SetItem(item)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = repo.List(ctx, "", "", 1, 20)
	}
}

// BenchmarkItemRepoIncrementStats 测试统计更新性能
func BenchmarkItemRepoIncrementStats(b *testing.B) {
	repo := mocks.NewMockItemRepository()
	ctx := context.Background()

	item := fixtures.GetTestItem("item_001")
	repo.SetItem(item)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = repo.IncrementStats(ctx, "item_001", "view")
	}
}

// BenchmarkItemRepoGetPopular 测试热门物品获取性能
func BenchmarkItemRepoGetPopular(b *testing.B) {
	repo := mocks.NewMockItemRepository()
	ctx := context.Background()

	for _, item := range fixtures.GetAllTestItems() {
		repo.SetItem(item)
	}
	for _, stats := range fixtures.TestItemStats {
		repo.SetItemStats(stats)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = repo.GetPopularByCategories(ctx, []string{}, 10)
	}
}

// =============================================================================
// 推理客户端性能测试
// =============================================================================

// BenchmarkInference 测试推理性能
func BenchmarkInference(b *testing.B) {
	client := mocks.NewMockInferenceClient()
	ctx := context.Background()

	input := &interfaces.ModelInput{
		UserSequence:  []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		AttentionMask: []int64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.Infer(ctx, input)
	}
}

// BenchmarkBatchInference 测试批量推理性能
func BenchmarkBatchInference(b *testing.B) {
	client := mocks.NewMockInferenceClient()
	ctx := context.Background()

	inputs := make([]*interfaces.ModelInput, 10)
	for i := 0; i < 10; i++ {
		inputs[i] = &interfaces.ModelInput{
			UserSequence:  []int64{1, 2, 3, 4, 5},
			AttentionMask: []int64{1, 1, 1, 1, 1},
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.BatchInfer(ctx, inputs)
	}
}

// BenchmarkInference_Parallel 测试并发推理性能
func BenchmarkInference_Parallel(b *testing.B) {
	client := mocks.NewMockInferenceClient()
	ctx := context.Background()

	input := &interfaces.ModelInput{
		UserSequence:  []int64{1, 2, 3, 4, 5},
		AttentionMask: []int64{1, 1, 1, 1, 1},
	}

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = client.Infer(ctx, input)
		}
	})
}

// =============================================================================
// 特征服务性能测试
// =============================================================================

// BenchmarkGetUserFeatures 测试获取用户特征性能
func BenchmarkGetUserFeatures(b *testing.B) {
	service := mocks.NewMockFeatureService()
	ctx := context.Background()

	service.GetUserFeaturesResult = fixtures.GetTestUserFeatures("user_001")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = service.GetUserFeatures(ctx, "user_001")
	}
}

// BenchmarkGetItemFeatures 测试获取物品特征性能
func BenchmarkGetItemFeatures(b *testing.B) {
	service := mocks.NewMockFeatureService()
	ctx := context.Background()

	service.GetItemFeaturesResult = fixtures.GetTestItemFeatures("item_001")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = service.GetItemFeatures(ctx, "item_001")
	}
}

// =============================================================================
// LLM 客户端性能测试
// =============================================================================

// BenchmarkLLMComplete 测试 LLM 补全性能
func BenchmarkLLMComplete(b *testing.B) {
	client := mocks.NewMockLLMClient()
	ctx := context.Background()

	prompt := "This is a test prompt for benchmarking."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.Complete(ctx, prompt)
	}
}

// BenchmarkLLMEmbed 测试 LLM 嵌入性能
func BenchmarkLLMEmbed(b *testing.B) {
	client := mocks.NewMockLLMClient()
	ctx := context.Background()

	text := "This is a test text for embedding benchmarking."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.Embed(ctx, text)
	}
}

// =============================================================================
// 综合端到端性能测试
// =============================================================================

// BenchmarkEndToEndRecommendation 测试端到端推荐性能
func BenchmarkEndToEndRecommendation(b *testing.B) {
	// 设置所有 Mock
	userRepo := mocks.NewMockUserRepository()
	itemRepo := mocks.NewMockItemRepository()
	cache := mocks.NewMockCache()
	inferClient := mocks.NewMockInferenceClient()
	featureService := mocks.NewMockFeatureService()
	recommendRepo := mocks.NewMockRecommendRepository()

	// 加载数据
	for _, user := range fixtures.GetAllTestUsers() {
		userRepo.SetUser(user)
	}
	for _, user := range fixtures.TestUsers {
		userRepo.SetBehaviors(user.ID, fixtures.GetBehaviorsForUser(user.ID))
	}
	for _, item := range fixtures.GetAllTestItems() {
		itemRepo.SetItem(item)
	}

	featureService.GetUserFeaturesResult = fixtures.GetTestUserFeatures("user_001")

	ctx := context.Background()

	// 预热
	for i := 0; i < 10; i++ {
		_ = simulateRecommendation(ctx, userRepo, itemRepo, cache, inferClient, featureService, recommendRepo, "user_001")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = simulateRecommendation(ctx, userRepo, itemRepo, cache, inferClient, featureService, recommendRepo, "user_001")
	}
}

// simulateRecommendation 模拟推荐流程
func simulateRecommendation(
	ctx context.Context,
	userRepo *mocks.MockUserRepository,
	itemRepo *mocks.MockItemRepository,
	cache *mocks.MockCache,
	inferClient *mocks.MockInferenceClient,
	featureService *mocks.MockFeatureService,
	recommendRepo *mocks.MockRecommendRepository,
	userID string,
) error {
	// 1. 获取用户
	_, err := userRepo.GetByID(ctx, userID)
	if err != nil {
		return err
	}

	// 2. 获取用户行为
	_, err = userRepo.GetBehaviors(ctx, userID, 50)
	if err != nil {
		return err
	}

	// 3. 获取用户特征
	_, err = featureService.GetUserFeatures(ctx, userID)
	if err != nil {
		return err
	}

	// 4. 执行推理
	_, err = inferClient.Infer(ctx, &interfaces.ModelInput{
		UserSequence: []int64{1, 2, 3, 4, 5},
	})
	if err != nil {
		return err
	}

	// 5. 获取物品详情
	_, err = itemRepo.GetByIDs(ctx, []string{"item_001", "item_002", "item_003"})
	if err != nil {
		return err
	}

	// 6. 记录推荐日志
	err = recommendRepo.LogRecommendation(ctx, &interfaces.RecommendLog{
		RequestID:       "req_bench",
		UserID:          userID,
		Recommendations: []string{"item_001", "item_002", "item_003"},
		Strategy:        "model",
		Timestamp:       time.Now(),
	})
	if err != nil {
		return err
	}

	return nil
}

// BenchmarkEndToEndRecommendation_Parallel 测试并发端到端推荐性能
func BenchmarkEndToEndRecommendation_Parallel(b *testing.B) {
	// 设置所有 Mock
	userRepo := mocks.NewMockUserRepository()
	itemRepo := mocks.NewMockItemRepository()
	cache := mocks.NewMockCache()
	inferClient := mocks.NewMockInferenceClient()
	featureService := mocks.NewMockFeatureService()
	recommendRepo := mocks.NewMockRecommendRepository()

	// 加载数据
	for _, user := range fixtures.GetAllTestUsers() {
		userRepo.SetUser(user)
	}
	for _, user := range fixtures.TestUsers {
		userRepo.SetBehaviors(user.ID, fixtures.GetBehaviorsForUser(user.ID))
	}
	for _, item := range fixtures.GetAllTestItems() {
		itemRepo.SetItem(item)
	}

	featureService.GetUserFeaturesResult = fixtures.GetTestUserFeatures("user_001")

	ctx := context.Background()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = simulateRecommendation(ctx, userRepo, itemRepo, cache, inferClient, featureService, recommendRepo, "user_001")
		}
	})
}

// =============================================================================
// 内存分配测试
// =============================================================================

// BenchmarkCacheGetAllocs 测试缓存读取内存分配
func BenchmarkCacheGetAllocs(b *testing.B) {
	cache := mocks.NewMockCache()
	ctx := context.Background()

	testData := fixtures.GetTestUser("user_001")
	_ = cache.SetDirect("user:001", testData, time.Hour)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var result interfaces.User
		_ = cache.Get(ctx, "user:001", &result)
	}
}

// BenchmarkUserRepoGetByIDAllocs 测试用户获取内存分配
func BenchmarkUserRepoGetByIDAllocs(b *testing.B) {
	repo := mocks.NewMockUserRepository()
	ctx := context.Background()

	for _, user := range fixtures.GetAllTestUsers() {
		repo.SetUser(user)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = repo.GetByID(ctx, "user_001")
	}
}

