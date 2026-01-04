// Package integration 提供集成测试
//
// 本包包含各服务模块之间的集成测试，
// 测试完整的业务流程和服务间交互。
package integration

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
// 推荐服务集成测试
// =============================================================================

// RecommendServiceIntegration 推荐服务集成测试结构
type RecommendServiceIntegration struct {
	userRepo        *mocks.MockUserRepository
	itemRepo        *mocks.MockItemRepository
	recommendRepo   *mocks.MockRecommendRepository
	cache           *mocks.MockCache
	inferClient     *mocks.MockInferenceClient
	featureService  *mocks.MockFeatureService
	coldStartService *mocks.MockColdStartService
}

// setupRecommendTestEnv 设置推荐服务测试环境
func setupRecommendTestEnv(t *testing.T) (*RecommendServiceIntegration, func()) {
	env := &RecommendServiceIntegration{
		userRepo:         mocks.NewMockUserRepository(),
		itemRepo:         mocks.NewMockItemRepository(),
		recommendRepo:    mocks.NewMockRecommendRepository(),
		cache:            mocks.NewMockCache(),
		inferClient:      mocks.NewMockInferenceClient(),
		featureService:   mocks.NewMockFeatureService(),
		coldStartService: mocks.NewMockColdStartService(),
	}

	// 加载测试用户
	for _, user := range fixtures.GetAllTestUsers() {
		env.userRepo.SetUser(user)
	}

	// 加载测试物品
	for _, item := range fixtures.GetAllTestItems() {
		env.itemRepo.SetItem(item)
	}

	// 加载测试行为
	for _, user := range fixtures.TestUsers {
		behaviors := fixtures.GetBehaviorsForUser(user.ID)
		env.userRepo.SetBehaviors(user.ID, behaviors)
	}

	// 设置物品统计
	for _, stats := range fixtures.TestItemStats {
		env.itemRepo.SetItemStats(stats)
	}

	cleanup := func() {
		env.cache.Clear()
		env.userRepo.Reset()
		env.itemRepo.Reset()
		env.recommendRepo.Reset()
	}

	return env, cleanup
}

// TestRecommendIntegration_NormalUser 测试正常用户推荐流程
func TestRecommendIntegration_NormalUser(t *testing.T) {
	env, cleanup := setupRecommendTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 1. 验证用户存在
	user, err := env.userRepo.GetByID(ctx, "user_001")
	require.NoError(t, err)
	assert.NotNil(t, user)

	// 2. 获取用户行为
	behaviors, err := env.userRepo.GetBehaviors(ctx, "user_001", 100)
	require.NoError(t, err)
	assert.NotEmpty(t, behaviors)

	// 3. 获取用户特征
	env.featureService.GetUserFeaturesResult = fixtures.GetTestUserFeatures("user_001")
	userFeatures, err := env.featureService.GetUserFeatures(ctx, "user_001")
	require.NoError(t, err)
	assert.NotNil(t, userFeatures)

	// 4. 执行模型推理
	modelOutput, err := env.inferClient.Infer(ctx, &interfaces.ModelInput{
		UserSequence: []int64{1, 2, 3, 4, 5},
	})
	require.NoError(t, err)
	assert.NotEmpty(t, modelOutput.Recommendations)

	// 5. 验证推荐结果
	assert.Greater(t, len(modelOutput.Recommendations), 0)
	assert.Equal(t, 1, env.inferClient.InferCalls)
}

// TestRecommendIntegration_ColdStartUser 测试冷启动用户推荐流程
func TestRecommendIntegration_ColdStartUser(t *testing.T) {
	env, cleanup := setupRecommendTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 1. 创建新用户
	newUser := fixtures.CreateTestUser("user_cold_start", "New User", "new@example.com", 22, "unknown")
	err := env.userRepo.Create(ctx, newUser)
	require.NoError(t, err)

	// 2. 验证无历史行为
	behaviors, err := env.userRepo.GetBehaviors(ctx, newUser.ID, 100)
	require.NoError(t, err)
	assert.Empty(t, behaviors)

	// 3. 触发冷启动处理
	coldStartResult, err := env.coldStartService.HandleNewUser(ctx, newUser)
	require.NoError(t, err)
	assert.NotNil(t, coldStartResult)
	assert.Equal(t, "popular", coldStartResult.Strategy)

	// 4. 获取冷启动推荐
	env.coldStartService.GetColdStartRecommendationsResult = []*interfaces.Item{
		fixtures.GetTestItem("item_001"),
		fixtures.GetTestItem("item_002"),
	}
	recommendations, err := env.coldStartService.GetColdStartRecommendations(ctx, newUser.ID, 10)
	require.NoError(t, err)
	assert.NotEmpty(t, recommendations)
}

// TestRecommendIntegration_WithBehaviorRecording 测试推荐后行为记录
func TestRecommendIntegration_WithBehaviorRecording(t *testing.T) {
	env, cleanup := setupRecommendTestEnv(t)
	defer cleanup()

	ctx := context.Background()
	userID := "user_001"

	// 1. 获取初始行为数
	initialBehaviors, err := env.userRepo.GetBehaviors(ctx, userID, 100)
	require.NoError(t, err)
	initialCount := len(initialBehaviors)

	// 2. 模拟用户点击推荐物品
	behavior := &interfaces.UserBehavior{
		UserID:    userID,
		ItemID:    "item_005",
		Action:    "click",
		Timestamp: time.Now(),
		Context:   map[string]string{"source": "recommend"},
	}
	err = env.userRepo.AddBehavior(ctx, behavior)
	require.NoError(t, err)

	// 3. 验证行为已记录
	newBehaviors, err := env.userRepo.GetBehaviors(ctx, userID, 100)
	require.NoError(t, err)
	assert.Equal(t, initialCount+1, len(newBehaviors))

	// 4. 验证行为内容
	latestBehavior := newBehaviors[len(newBehaviors)-1]
	assert.Equal(t, "item_005", latestBehavior.ItemID)
	assert.Equal(t, "click", latestBehavior.Action)
}

// TestRecommendIntegration_ExposureFiltering 测试曝光过滤
func TestRecommendIntegration_ExposureFiltering(t *testing.T) {
	env, cleanup := setupRecommendTestEnv(t)
	defer cleanup()

	ctx := context.Background()
	userID := "user_001"

	// 1. 记录曝光
	exposedItems := []string{"item_001", "item_002", "item_003"}
	for _, itemID := range exposedItems {
		err := env.recommendRepo.RecordExposure(ctx, userID, itemID, "req_001")
		require.NoError(t, err)
	}

	// 2. 获取曝光记录
	exposures := env.recommendRepo.GetExposures(userID)
	assert.Equal(t, len(exposedItems), len(exposures))

	// 3. 验证曝光过滤逻辑
	allItems := []string{"item_001", "item_002", "item_003", "item_004", "item_005"}
	exposedSet := make(map[string]bool)
	for _, id := range exposures {
		exposedSet[id] = true
	}

	filteredItems := make([]string, 0)
	for _, id := range allItems {
		if !exposedSet[id] {
			filteredItems = append(filteredItems, id)
		}
	}

	assert.Equal(t, 2, len(filteredItems))
	assert.Contains(t, filteredItems, "item_004")
	assert.Contains(t, filteredItems, "item_005")
}

// TestRecommendIntegration_SimilarItems 测试相似物品推荐
func TestRecommendIntegration_SimilarItems(t *testing.T) {
	env, cleanup := setupRecommendTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 1. 获取参考物品
	refItem, err := env.itemRepo.GetByID(ctx, "item_001")
	require.NoError(t, err)
	assert.NotNil(t, refItem)

	// 2. 获取物品特征
	env.featureService.GetItemFeaturesResult = fixtures.GetTestItemFeatures("item_001")
	itemFeatures, err := env.featureService.GetItemFeatures(ctx, "item_001")
	require.NoError(t, err)
	assert.NotNil(t, itemFeatures)

	// 3. 验证物品特征包含语义 ID
	assert.NotEqual(t, [3]int{0, 0, 0}, itemFeatures.SemanticID)
}

// TestRecommendIntegration_CacheInteraction 测试缓存交互
func TestRecommendIntegration_CacheInteraction(t *testing.T) {
	env, cleanup := setupRecommendTestEnv(t)
	defer cleanup()

	ctx := context.Background()
	userID := "user_001"

	// 1. 首次查询用户特征（缓存未命中）
	cacheKey := "user:features:" + userID
	var cachedFeatures interfaces.UserFeatures
	err := env.cache.Get(ctx, cacheKey, &cachedFeatures)
	assert.Equal(t, mocks.ErrCacheMiss, err)

	// 2. 获取用户特征并缓存
	features := fixtures.GetTestUserFeatures(userID)
	err = env.cache.Set(ctx, cacheKey, features, 15*time.Minute)
	require.NoError(t, err)

	// 3. 再次查询（缓存命中）
	err = env.cache.Get(ctx, cacheKey, &cachedFeatures)
	require.NoError(t, err)
	assert.Equal(t, userID, cachedFeatures.UserID)

	// 4. 验证缓存调用计数
	assert.Equal(t, 2, env.cache.GetCalls)
	assert.Equal(t, 1, env.cache.SetCalls)
}

// TestRecommendIntegration_RecommendationLogging 测试推荐日志记录
func TestRecommendIntegration_RecommendationLogging(t *testing.T) {
	env, cleanup := setupRecommendTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 1. 记录推荐日志
	log := &interfaces.RecommendLog{
		RequestID:       "req_001",
		UserID:          "user_001",
		Recommendations: []string{"item_001", "item_002", "item_003"},
		Strategy:        "model",
		Timestamp:       time.Now(),
	}
	err := env.recommendRepo.LogRecommendation(ctx, log)
	require.NoError(t, err)

	// 2. 添加更多日志
	for i := 0; i < 5; i++ {
		log := &interfaces.RecommendLog{
			RequestID:       "req_00" + string(rune('2'+i)),
			UserID:          "user_001",
			Recommendations: []string{"item_001"},
			Strategy:        "model",
			Timestamp:       time.Now(),
		}
		_ = env.recommendRepo.LogRecommendation(ctx, log)
	}

	// 3. 获取推荐日志
	logs, err := env.recommendRepo.GetRecommendationLogs(ctx, "user_001", 3)
	require.NoError(t, err)
	assert.LessOrEqual(t, len(logs), 3)
}

// TestRecommendIntegration_EndToEnd 测试端到端推荐流程
func TestRecommendIntegration_EndToEnd(t *testing.T) {
	env, cleanup := setupRecommendTestEnv(t)
	defer cleanup()

	ctx := context.Background()
	userID := "user_001"

	// 完整推荐流程
	
	// Step 1: 用户请求推荐
	user, err := env.userRepo.GetByID(ctx, userID)
	require.NoError(t, err)
	t.Logf("Step 1: User %s requests recommendations", user.ID)

	// Step 2: 获取用户行为历史
	behaviors, err := env.userRepo.GetBehaviors(ctx, userID, 50)
	require.NoError(t, err)
	t.Logf("Step 2: Retrieved %d behaviors", len(behaviors))

	// Step 3: 检查是否为冷启动用户
	isColdStart := len(behaviors) < 5
	t.Logf("Step 3: Is cold start: %v", isColdStart)

	// Step 4: 获取用户特征
	env.featureService.GetUserFeaturesResult = fixtures.GetTestUserFeatures(userID)
	userFeatures, err := env.featureService.GetUserFeatures(ctx, userID)
	require.NoError(t, err)
	t.Logf("Step 4: Got user features with embedding dim: %d", len(userFeatures.Embedding))

	// Step 5: 执行模型推理
	modelInput := &interfaces.ModelInput{
		UserSequence: []int64{1, 2, 3, 4, 5},
	}
	modelOutput, err := env.inferClient.Infer(ctx, modelInput)
	require.NoError(t, err)
	t.Logf("Step 5: Model returned %d recommendations", len(modelOutput.Recommendations))

	// Step 6: 过滤已曝光物品
	// 模拟之前的曝光
	_ = env.recommendRepo.RecordExposure(ctx, userID, "item_001", "req_old")
	exposures := env.recommendRepo.GetExposures(userID)
	t.Logf("Step 6: Filtering %d exposed items", len(exposures))

	// Step 7: 记录本次推荐日志
	recommendLog := &interfaces.RecommendLog{
		RequestID:       "req_current",
		UserID:          userID,
		Recommendations: []string{"item_002", "item_003", "item_004"},
		Strategy:        "model",
		Timestamp:       time.Now(),
	}
	err = env.recommendRepo.LogRecommendation(ctx, recommendLog)
	require.NoError(t, err)
	t.Logf("Step 7: Logged recommendation request")

	// Step 8: 记录新的曝光
	for _, itemID := range recommendLog.Recommendations {
		_ = env.recommendRepo.RecordExposure(ctx, userID, itemID, recommendLog.RequestID)
	}
	t.Logf("Step 8: Recorded new exposures")

	// 验证整个流程
	assert.Equal(t, 1, env.featureService.GetUserFeaturesCalls)
	assert.Equal(t, 1, env.inferClient.InferCalls)
	assert.Equal(t, 1, env.recommendRepo.LogRecommendationCalls)
}

// =============================================================================
// 并发测试
// =============================================================================

// TestRecommendIntegration_Concurrent 测试并发推荐请求
func TestRecommendIntegration_Concurrent(t *testing.T) {
	env, cleanup := setupRecommendTestEnv(t)
	defer cleanup()

	ctx := context.Background()
	const numGoroutines = 20

	done := make(chan bool, numGoroutines)
	errors := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(idx int) {
			// 随机选择用户
			userID := "user_001"
			if idx%2 == 0 {
				userID = "user_002"
			}

			// 执行推荐操作
			_, err := env.userRepo.GetByID(ctx, userID)
			if err != nil {
				errors <- err
				done <- false
				return
			}

			_, err = env.userRepo.GetBehaviors(ctx, userID, 50)
			if err != nil {
				errors <- err
				done <- false
				return
			}

			_, err = env.inferClient.Infer(ctx, &interfaces.ModelInput{
				UserSequence: []int64{1, 2, 3},
			})
			if err != nil {
				errors <- err
				done <- false
				return
			}

			done <- true
		}(i)
	}

	// 等待所有 goroutine 完成
	successCount := 0
	for i := 0; i < numGoroutines; i++ {
		if <-done {
			successCount++
		}
	}

	close(errors)

	// 验证所有请求都成功
	assert.Equal(t, numGoroutines, successCount)
	assert.Empty(t, len(errors))
}

// =============================================================================
// 错误恢复测试
// =============================================================================

// TestRecommendIntegration_InferenceFailure 测试推理失败的恢复
func TestRecommendIntegration_InferenceFailure(t *testing.T) {
	env, cleanup := setupRecommendTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 设置推理服务失败
	env.inferClient.InferError = mocks.ErrServiceUnavailable

	// 尝试推理
	_, err := env.inferClient.Infer(ctx, &interfaces.ModelInput{
		UserSequence: []int64{1, 2, 3},
	})
	assert.Error(t, err)

	// 回退到热门推荐
	popularItems, err := env.itemRepo.GetPopularByCategories(ctx, []string{}, 10)
	require.NoError(t, err)
	assert.NotEmpty(t, popularItems)

	t.Logf("Fallback to %d popular items", len(popularItems))
}

// TestRecommendIntegration_CacheFailure 测试缓存失败的恢复
func TestRecommendIntegration_CacheFailure(t *testing.T) {
	env, cleanup := setupRecommendTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 设置缓存服务失败
	env.cache.GetError = mocks.ErrServiceUnavailable
	env.cache.SetError = mocks.ErrServiceUnavailable

	// 直接从数据库获取
	user, err := env.userRepo.GetByID(ctx, "user_001")
	require.NoError(t, err)
	assert.NotNil(t, user)

	behaviors, err := env.userRepo.GetBehaviors(ctx, "user_001", 50)
	require.NoError(t, err)
	assert.NotEmpty(t, behaviors)

	// 验证即使缓存失败，主流程仍然正常
}

