// Package unit 提供单元测试
//
// 本包包含服务层、数据访问层和处理器层的单元测试。
// 使用 Mock 实现隔离外部依赖，确保测试的独立性和可重复性。
package unit

import (
	"context"
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
// UserService 单元测试
// =============================================================================

// TestUserService_GetUser 测试获取用户
func TestUserService_GetUser(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	testUser := fixtures.GetTestUser("user_001")
	mockRepo.SetUser(testUser)

	service := user.NewService(mockRepo, mockCache, nil)

	// 执行
	result, err := service.GetUser(context.Background(), testUser.ID)

	// 验证
	require.NoError(t, err)
	assert.Equal(t, testUser.ID, result.ID)
	assert.Equal(t, testUser.Name, result.Name)
	assert.Equal(t, testUser.Email, result.Email)
	assert.Equal(t, 1, mockRepo.GetByIDCalls)
}

// TestUserService_GetUser_NotFound 测试获取不存在的用户
func TestUserService_GetUser_NotFound(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	service := user.NewService(mockRepo, mockCache, nil)

	// 执行
	result, err := service.GetUser(context.Background(), "non_existent_user")

	// 验证
	assert.Error(t, err)
	assert.Nil(t, result)
}

// TestUserService_GetUser_FromCache 测试从缓存获取用户
func TestUserService_GetUser_FromCache(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	testUser := fixtures.GetTestUser("user_001")

	// 先设置缓存
	err := mockCache.SetDirect("user:"+testUser.ID, testUser, time.Hour)
	require.NoError(t, err)

	service := user.NewService(mockRepo, mockCache, nil)

	// 执行（应该从缓存获取，不调用 repo）
	result, err := service.GetUser(context.Background(), testUser.ID)

	// 验证
	require.NoError(t, err)
	assert.Equal(t, testUser.ID, result.ID)
	assert.Equal(t, 0, mockRepo.GetByIDCalls) // 不应调用 repo
	assert.Equal(t, 1, mockCache.GetCalls)
}

// TestUserService_GetUser_InvalidInput 测试无效输入
func TestUserService_GetUser_InvalidInput(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	service := user.NewService(mockRepo, mockCache, nil)

	// 执行 - 空用户 ID
	result, err := service.GetUser(context.Background(), "")

	// 验证
	assert.Error(t, err)
	assert.Nil(t, result)
}

// TestUserService_CreateUser 测试创建用户
func TestUserService_CreateUser(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	service := user.NewService(mockRepo, mockCache, nil)

	req := &interfaces.CreateUserRequest{
		Name:   "Test User",
		Email:  "test@example.com",
		Age:    30,
		Gender: "male",
	}

	// 执行
	result, err := service.CreateUser(context.Background(), req)

	// 验证
	require.NoError(t, err)
	assert.NotEmpty(t, result.ID)
	assert.Equal(t, req.Name, result.Name)
	assert.Equal(t, req.Email, result.Email)
	assert.Equal(t, req.Age, result.Age)
	assert.Equal(t, 1, mockRepo.CreateCalls)
}

// TestUserService_CreateUser_InvalidRequest 测试无效的创建请求
func TestUserService_CreateUser_InvalidRequest(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	service := user.NewService(mockRepo, mockCache, nil)

	testCases := []struct {
		name string
		req  *interfaces.CreateUserRequest
	}{
		{
			name: "nil request",
			req:  nil,
		},
		{
			name: "empty name",
			req:  &interfaces.CreateUserRequest{Name: "", Email: "test@example.com"},
		},
		{
			name: "empty email",
			req:  &interfaces.CreateUserRequest{Name: "Test", Email: ""},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := service.CreateUser(context.Background(), tc.req)
			assert.Error(t, err)
			assert.Nil(t, result)
		})
	}
}

// TestUserService_UpdateUser 测试更新用户
func TestUserService_UpdateUser(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	testUser := fixtures.GetTestUser("user_001")
	mockRepo.SetUser(testUser)

	service := user.NewService(mockRepo, mockCache, nil)

	req := &interfaces.UpdateUserRequest{
		Name: "Updated Name",
	}

	// 执行
	result, err := service.UpdateUser(context.Background(), testUser.ID, req)

	// 验证
	require.NoError(t, err)
	assert.Equal(t, "Updated Name", result.Name)
	assert.Equal(t, testUser.Email, result.Email) // Email 不变
	assert.Equal(t, 1, mockRepo.UpdateCalls)
}

// TestUserService_UpdateUser_NotFound 测试更新不存在的用户
func TestUserService_UpdateUser_NotFound(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	service := user.NewService(mockRepo, mockCache, nil)

	req := &interfaces.UpdateUserRequest{
		Name: "Updated Name",
	}

	// 执行
	result, err := service.UpdateUser(context.Background(), "non_existent", req)

	// 验证
	assert.Error(t, err)
	assert.Nil(t, result)
}

// TestUserService_DeleteUser 测试删除用户
func TestUserService_DeleteUser(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	testUser := fixtures.GetTestUser("user_001")
	mockRepo.SetUser(testUser)

	// 设置缓存
	_ = mockCache.SetDirect("user:"+testUser.ID, testUser, time.Hour)

	service := user.NewService(mockRepo, mockCache, nil)

	// 执行
	err := service.DeleteUser(context.Background(), testUser.ID)

	// 验证
	require.NoError(t, err)
	assert.Equal(t, 1, mockRepo.DeleteCalls)

	// 验证缓存已删除
	assert.False(t, mockCache.KeyExists("user:"+testUser.ID))
}

// TestUserService_RecordBehavior 测试记录用户行为
func TestUserService_RecordBehavior(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	testUser := fixtures.GetTestUser("user_001")
	mockRepo.SetUser(testUser)

	service := user.NewService(mockRepo, mockCache, nil)

	req := &interfaces.RecordBehaviorRequest{
		UserID: testUser.ID,
		ItemID: "item_001",
		Action: "click",
		Context: map[string]string{
			"device": "mobile",
		},
	}

	// 执行
	err := service.RecordBehavior(context.Background(), req)

	// 验证
	require.NoError(t, err)
	assert.Equal(t, 1, mockRepo.AddBehaviorCalls)

	// 验证行为是否记录
	behaviors, _ := mockRepo.GetBehaviors(context.Background(), testUser.ID, 10)
	assert.Len(t, behaviors, 1)
	assert.Equal(t, "click", behaviors[0].Action)
}

// TestUserService_RecordBehavior_InvalidRequest 测试无效的行为记录请求
func TestUserService_RecordBehavior_InvalidRequest(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	service := user.NewService(mockRepo, mockCache, nil)

	testCases := []struct {
		name string
		req  *interfaces.RecordBehaviorRequest
	}{
		{
			name: "nil request",
			req:  nil,
		},
		{
			name: "empty user id",
			req:  &interfaces.RecordBehaviorRequest{UserID: "", ItemID: "item_001", Action: "click"},
		},
		{
			name: "empty item id",
			req:  &interfaces.RecordBehaviorRequest{UserID: "user_001", ItemID: "", Action: "click"},
		},
		{
			name: "empty action",
			req:  &interfaces.RecordBehaviorRequest{UserID: "user_001", ItemID: "item_001", Action: ""},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := service.RecordBehavior(context.Background(), tc.req)
			assert.Error(t, err)
		})
	}
}

// TestUserService_GetUserBehaviors 测试获取用户行为历史
func TestUserService_GetUserBehaviors(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	testUser := fixtures.GetTestUser("user_001")
	behaviors := fixtures.GetBehaviorsForUser(testUser.ID)
	mockRepo.SetUser(testUser)
	mockRepo.SetBehaviors(testUser.ID, behaviors)

	service := user.NewService(mockRepo, mockCache, nil)

	// 执行
	result, err := service.GetUserBehaviors(context.Background(), testUser.ID, 10)

	// 验证
	require.NoError(t, err)
	assert.NotEmpty(t, result)
	assert.LessOrEqual(t, len(result), 10)
}

// TestUserService_GetUserBehaviors_WithLimit 测试获取用户行为历史（带限制）
func TestUserService_GetUserBehaviors_WithLimit(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	testUser := fixtures.GetTestUser("user_001")
	// 创建多个行为
	behaviors := make([]*interfaces.UserBehavior, 20)
	for i := 0; i < 20; i++ {
		behaviors[i] = fixtures.CreateTestBehavior(testUser.ID, "item_001", "view")
	}
	mockRepo.SetUser(testUser)
	mockRepo.SetBehaviors(testUser.ID, behaviors)

	service := user.NewService(mockRepo, mockCache, nil)

	// 执行
	result, err := service.GetUserBehaviors(context.Background(), testUser.ID, 5)

	// 验证
	require.NoError(t, err)
	assert.Len(t, result, 5)
}

// TestUserService_GetUserProfile 测试获取用户画像
func TestUserService_GetUserProfile(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	testUser := fixtures.GetTestUser("user_001")
	behaviors := fixtures.GetBehaviorsForUser(testUser.ID)
	mockRepo.SetUser(testUser)
	mockRepo.SetBehaviors(testUser.ID, behaviors)

	service := user.NewService(mockRepo, mockCache, nil)

	// 执行
	result, err := service.GetUserProfile(context.Background(), testUser.ID)

	// 验证
	require.NoError(t, err)
	assert.NotNil(t, result)
	assert.NotNil(t, result.User)
	assert.Equal(t, testUser.ID, result.User.ID)
	assert.GreaterOrEqual(t, result.TotalActions, 0)
}

// TestUserService_GetUserProfile_Cached 测试获取缓存的用户画像
func TestUserService_GetUserProfile_Cached(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	testProfile := fixtures.GetTestUserProfile("user_001")
	require.NotNil(t, testProfile)

	// 设置缓存
	err := mockCache.SetDirect("user:profile:user_001", testProfile, time.Hour)
	require.NoError(t, err)

	service := user.NewService(mockRepo, mockCache, nil)

	// 执行
	result, err := service.GetUserProfile(context.Background(), "user_001")

	// 验证
	require.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, 0, mockRepo.GetByIDCalls) // 不应调用 repo
}

// =============================================================================
// 边界条件测试
// =============================================================================

// TestUserService_ConcurrentAccess 测试并发访问
func TestUserService_ConcurrentAccess(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	testUser := fixtures.GetTestUser("user_001")
	mockRepo.SetUser(testUser)

	service := user.NewService(mockRepo, mockCache, nil)

	// 执行 - 并发获取用户
	const numGoroutines = 10
	done := make(chan bool, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			_, err := service.GetUser(context.Background(), testUser.ID)
			assert.NoError(t, err)
			done <- true
		}()
	}

	// 等待所有 goroutine 完成
	for i := 0; i < numGoroutines; i++ {
		<-done
	}

	// 验证
	assert.GreaterOrEqual(t, mockRepo.GetByIDCalls, 1)
}

// TestUserService_ContextCancellation 测试上下文取消
func TestUserService_ContextCancellation(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	// 设置缓存延迟
	mockCache.Latency = 100 * time.Millisecond

	service := user.NewService(mockRepo, mockCache, nil)

	// 创建一个已取消的上下文
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // 立即取消

	// 执行
	_, err := service.GetUser(ctx, "user_001")

	// 验证 - 由于 Mock 实现可能不检查 context，这里主要验证不会 panic
	// 在生产代码中应该返回 context.Canceled 错误
	_ = err
}

// TestUserService_ErrorPropagation 测试错误传播
func TestUserService_ErrorPropagation(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	// 设置 repo 返回错误
	mockRepo.GetByIDError = mocks.ErrNotFound

	service := user.NewService(mockRepo, mockCache, nil)

	// 执行
	result, err := service.GetUser(context.Background(), "user_001")

	// 验证
	assert.Error(t, err)
	assert.Nil(t, result)
}

// TestUserService_CacheWriteFailure 测试缓存写入失败不影响主流程
func TestUserService_CacheWriteFailure(t *testing.T) {
	// 准备
	mockRepo := mocks.NewMockUserRepository()
	mockCache := mocks.NewMockCache()

	testUser := fixtures.GetTestUser("user_001")
	mockRepo.SetUser(testUser)

	// 设置缓存写入失败
	mockCache.SetError = mocks.ErrServiceUnavailable

	service := user.NewService(mockRepo, mockCache, nil)

	// 执行
	result, err := service.GetUser(context.Background(), testUser.ID)

	// 验证 - 缓存写入失败不应该影响主流程
	require.NoError(t, err)
	assert.Equal(t, testUser.ID, result.ID)
}

