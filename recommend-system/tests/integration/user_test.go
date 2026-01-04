package integration

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"recommend-system/internal/interfaces"
	"recommend-system/internal/service/user"
	"recommend-system/tests/fixtures"
	"recommend-system/tests/mocks"
)

// =============================================================================
// 用户服务集成测试
// =============================================================================

// UserServiceIntegration 用户服务集成测试结构
type UserServiceIntegration struct {
	userRepo *mocks.MockUserRepository
	cache    *mocks.MockCache
	service  *user.Service
}

// setupUserTestEnv 设置用户服务测试环境
func setupUserTestEnv(t *testing.T) (*UserServiceIntegration, func()) {
	userRepo := mocks.NewMockUserRepository()
	cache := mocks.NewMockCache()
	service := user.NewService(userRepo, cache, nil)

	env := &UserServiceIntegration{
		userRepo: userRepo,
		cache:    cache,
		service:  service,
	}

	// 加载测试数据
	for _, u := range fixtures.GetAllTestUsers() {
		userRepo.SetUser(u)
	}

	for _, u := range fixtures.TestUsers {
		behaviors := fixtures.GetBehaviorsForUser(u.ID)
		userRepo.SetBehaviors(u.ID, behaviors)
	}

	cleanup := func() {
		cache.Reset()
		userRepo.Reset()
	}

	return env, cleanup
}

// TestUserIntegration_CRUD 测试用户 CRUD 完整流程
func TestUserIntegration_CRUD(t *testing.T) {
	env, cleanup := setupUserTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 1. Create - 创建新用户
	createReq := &interfaces.CreateUserRequest{
		Name:   "Integration Test User",
		Email:  "integration@test.com",
		Age:    30,
		Gender: "male",
	}

	createdUser, err := env.service.CreateUser(ctx, createReq)
	require.NoError(t, err)
	assert.NotEmpty(t, createdUser.ID)
	t.Logf("Created user: %s", createdUser.ID)

	// 2. Read - 读取用户
	readUser, err := env.service.GetUser(ctx, createdUser.ID)
	require.NoError(t, err)
	assert.Equal(t, createdUser.ID, readUser.ID)
	assert.Equal(t, createReq.Name, readUser.Name)

	// 3. Update - 更新用户
	updateReq := &interfaces.UpdateUserRequest{
		Name: "Updated Integration User",
		Age:  31,
	}

	updatedUser, err := env.service.UpdateUser(ctx, createdUser.ID, updateReq)
	require.NoError(t, err)
	assert.Equal(t, "Updated Integration User", updatedUser.Name)
	assert.Equal(t, 31, updatedUser.Age)

	// 4. Delete - 删除用户
	err = env.service.DeleteUser(ctx, createdUser.ID)
	require.NoError(t, err)

	// 5. Verify deletion
	_, err = env.service.GetUser(ctx, createdUser.ID)
	assert.Error(t, err)
}

// TestUserIntegration_BehaviorTracking 测试用户行为跟踪
func TestUserIntegration_BehaviorTracking(t *testing.T) {
	env, cleanup := setupUserTestEnv(t)
	defer cleanup()

	ctx := context.Background()
	userID := "user_001"

	// 1. 获取初始行为数
	initialBehaviors, err := env.service.GetUserBehaviors(ctx, userID, 100)
	require.NoError(t, err)
	initialCount := len(initialBehaviors)
	t.Logf("Initial behavior count: %d", initialCount)

	// 2. 记录多个行为
	actions := []string{"view", "click", "like", "share"}
	for _, action := range actions {
		req := &interfaces.RecordBehaviorRequest{
			UserID: userID,
			ItemID: "item_new",
			Action: action,
			Context: map[string]string{
				"source": "integration_test",
			},
		}
		err := env.service.RecordBehavior(ctx, req)
		require.NoError(t, err)
	}

	// 3. 验证行为已记录
	newBehaviors, err := env.service.GetUserBehaviors(ctx, userID, 100)
	require.NoError(t, err)
	assert.Equal(t, initialCount+len(actions), len(newBehaviors))

	// 4. 验证行为内容
	for i, action := range actions {
		found := false
		for _, b := range newBehaviors {
			if b.Action == action && b.ItemID == "item_new" {
				found = true
				break
			}
		}
		assert.True(t, found, "Action %s not found (index %d)", action, i)
	}
}

// TestUserIntegration_ProfileGeneration 测试用户画像生成
func TestUserIntegration_ProfileGeneration(t *testing.T) {
	env, cleanup := setupUserTestEnv(t)
	defer cleanup()

	ctx := context.Background()
	userID := "user_001"

	// 1. 获取用户画像
	profile, err := env.service.GetUserProfile(ctx, userID)
	require.NoError(t, err)
	assert.NotNil(t, profile)
	assert.NotNil(t, profile.User)
	assert.Equal(t, userID, profile.User.ID)

	// 2. 验证画像内容
	t.Logf("User profile: TotalActions=%d, PreferredTypes=%v",
		profile.TotalActions, profile.PreferredTypes)

	assert.GreaterOrEqual(t, profile.TotalActions, 0)
	assert.NotNil(t, profile.PreferredTypes)
	assert.NotNil(t, profile.ActiveHours)
}

// TestUserIntegration_CacheConsistency 测试缓存一致性
func TestUserIntegration_CacheConsistency(t *testing.T) {
	env, cleanup := setupUserTestEnv(t)
	defer cleanup()

	ctx := context.Background()
	userID := "user_001"

	// 1. 首次获取用户（缓存未命中）
	user1, err := env.service.GetUser(ctx, userID)
	require.NoError(t, err)

	// 2. 再次获取（应该命中缓存）
	user2, err := env.service.GetUser(ctx, userID)
	require.NoError(t, err)

	// 3. 验证数据一致性
	assert.Equal(t, user1.ID, user2.ID)
	assert.Equal(t, user1.Name, user2.Name)
	assert.Equal(t, user1.Email, user2.Email)

	// 4. 更新用户
	_, err = env.service.UpdateUser(ctx, userID, &interfaces.UpdateUserRequest{
		Name: "Cache Test Updated",
	})
	require.NoError(t, err)

	// 5. 再次获取（缓存应该被清除，返回新数据）
	user3, err := env.service.GetUser(ctx, userID)
	require.NoError(t, err)
	assert.Equal(t, "Cache Test Updated", user3.Name)
}

// TestUserIntegration_ConcurrentOperations 测试并发操作
func TestUserIntegration_ConcurrentOperations(t *testing.T) {
	env, cleanup := setupUserTestEnv(t)
	defer cleanup()

	ctx := context.Background()
	userID := "user_001"

	// 并发获取用户
	const numGoroutines = 50
	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := env.service.GetUser(ctx, userID)
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
		t.Errorf("Concurrent get error: %v", err)
		errorCount++
	}
	assert.Equal(t, 0, errorCount)
}

// TestUserIntegration_ConcurrentBehaviorRecording 测试并发行为记录
func TestUserIntegration_ConcurrentBehaviorRecording(t *testing.T) {
	env, cleanup := setupUserTestEnv(t)
	defer cleanup()

	ctx := context.Background()
	userID := "user_concurrent"

	// 创建测试用户
	_, err := env.service.CreateUser(ctx, &interfaces.CreateUserRequest{
		Name:  "Concurrent User",
		Email: "concurrent@test.com",
	})
	require.NoError(t, err)

	// 并发记录行为
	const numGoroutines = 100
	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			req := &interfaces.RecordBehaviorRequest{
				UserID: userID,
				ItemID: "item_" + string(rune('0'+idx%10)),
				Action: "view",
			}
			err := env.service.RecordBehavior(ctx, req)
			if err != nil {
				errors <- err
			}
		}(i)
	}

	wg.Wait()
	close(errors)

	// 验证没有错误
	for err := range errors {
		t.Errorf("Concurrent behavior recording error: %v", err)
	}

	// 验证行为数量
	behaviors, err := env.service.GetUserBehaviors(ctx, userID, 200)
	require.NoError(t, err)
	assert.Equal(t, numGoroutines, len(behaviors))
}

// TestUserIntegration_BehaviorLimitEnforcement 测试行为限制
func TestUserIntegration_BehaviorLimitEnforcement(t *testing.T) {
	env, cleanup := setupUserTestEnv(t)
	defer cleanup()

	ctx := context.Background()
	userID := "user_limit_test"

	// 创建测试用户
	_, err := env.service.CreateUser(ctx, &interfaces.CreateUserRequest{
		Name:  "Limit Test User",
		Email: "limit@test.com",
	})
	require.NoError(t, err)

	// 添加大量行为
	const totalBehaviors = 150
	for i := 0; i < totalBehaviors; i++ {
		req := &interfaces.RecordBehaviorRequest{
			UserID: userID,
			ItemID: "item_test",
			Action: "view",
		}
		_ = env.service.RecordBehavior(ctx, req)
	}

	// 测试不同的限制值
	testCases := []struct {
		limit    int
		expected int
	}{
		{10, 10},
		{50, 50},
		{100, 100},
		{0, 100},    // 0 应该使用默认值
		{-1, 100},   // 负数应该使用默认值
	}

	for _, tc := range testCases {
		behaviors, err := env.service.GetUserBehaviors(ctx, userID, tc.limit)
		require.NoError(t, err)
		assert.LessOrEqual(t, len(behaviors), tc.expected,
			"Limit %d: expected <= %d, got %d", tc.limit, tc.expected, len(behaviors))
	}
}

// TestUserIntegration_ProfileCaching 测试画像缓存
func TestUserIntegration_ProfileCaching(t *testing.T) {
	env, cleanup := setupUserTestEnv(t)
	defer cleanup()

	ctx := context.Background()
	userID := "user_001"

	// 1. 首次获取画像
	start := time.Now()
	profile1, err := env.service.GetUserProfile(ctx, userID)
	require.NoError(t, err)
	firstDuration := time.Since(start)

	// 2. 再次获取画像（应该更快）
	start = time.Now()
	profile2, err := env.service.GetUserProfile(ctx, userID)
	require.NoError(t, err)
	secondDuration := time.Since(start)

	// 3. 验证画像一致性
	assert.Equal(t, profile1.User.ID, profile2.User.ID)
	assert.Equal(t, profile1.TotalActions, profile2.TotalActions)

	t.Logf("First request: %v, Second request: %v", firstDuration, secondDuration)
}

// TestUserIntegration_ErrorRecovery 测试错误恢复
func TestUserIntegration_ErrorRecovery(t *testing.T) {
	env, cleanup := setupUserTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 1. 测试获取不存在的用户
	_, err := env.service.GetUser(ctx, "non_existent_user")
	assert.Error(t, err)

	// 2. 测试更新不存在的用户
	_, err = env.service.UpdateUser(ctx, "non_existent_user", &interfaces.UpdateUserRequest{
		Name: "Test",
	})
	assert.Error(t, err)

	// 3. 测试无效的创建请求
	_, err = env.service.CreateUser(ctx, nil)
	assert.Error(t, err)

	// 4. 测试无效的行为记录
	err = env.service.RecordBehavior(ctx, nil)
	assert.Error(t, err)

	// 5. 验证正常操作仍然可用
	user, err := env.service.GetUser(ctx, "user_001")
	require.NoError(t, err)
	assert.NotNil(t, user)
}

// TestUserIntegration_DataIntegrity 测试数据完整性
func TestUserIntegration_DataIntegrity(t *testing.T) {
	env, cleanup := setupUserTestEnv(t)
	defer cleanup()

	ctx := context.Background()

	// 1. 创建用户
	createReq := &interfaces.CreateUserRequest{
		Name:   "Integrity Test",
		Email:  "integrity@test.com",
		Age:    25,
		Gender: "female",
	}
	created, err := env.service.CreateUser(ctx, createReq)
	require.NoError(t, err)

	// 2. 验证所有字段正确保存
	retrieved, err := env.service.GetUser(ctx, created.ID)
	require.NoError(t, err)

	assert.Equal(t, createReq.Name, retrieved.Name)
	assert.Equal(t, createReq.Email, retrieved.Email)
	assert.Equal(t, createReq.Age, retrieved.Age)
	assert.Equal(t, createReq.Gender, retrieved.Gender)
	assert.False(t, retrieved.CreatedAt.IsZero())
	assert.False(t, retrieved.UpdatedAt.IsZero())

	// 3. 部分更新不应影响其他字段
	_, err = env.service.UpdateUser(ctx, created.ID, &interfaces.UpdateUserRequest{
		Name: "Updated Name Only",
	})
	require.NoError(t, err)

	updated, err := env.service.GetUser(ctx, created.ID)
	require.NoError(t, err)

	assert.Equal(t, "Updated Name Only", updated.Name)
	assert.Equal(t, createReq.Email, updated.Email)  // 未变
	assert.Equal(t, createReq.Age, updated.Age)      // 未变
	assert.Equal(t, createReq.Gender, updated.Gender) // 未变
}

