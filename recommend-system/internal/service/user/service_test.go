// Package user 提供用户服务的单元测试
package user

import (
	"context"
	"fmt"
	"testing"
	"time"

	"go.uber.org/zap"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// 测试辅助函数
// =============================================================================

// setupTestService 创建测试服务实例
func setupTestService(t *testing.T) (*Service, *MockUserRepository, *MockCache) {
	t.Helper()

	mockRepo := NewMockUserRepository()
	mockCache := NewMockCache()
	logger := zap.NewNop()

	service := NewService(mockRepo, mockCache, logger)
	return service, mockRepo, mockCache
}

// createTestUser 创建测试用户
func createTestUser(id string) *interfaces.User {
	now := time.Now()
	return &interfaces.User{
		ID:        id,
		Name:      "Test User " + id,
		Email:     id + "@example.com",
		Age:       25,
		Gender:    "male",
		Metadata:  map[string]string{"level": "vip"},
		CreatedAt: now,
		UpdatedAt: now,
	}
}

// createTestBehavior 创建测试行为
func createTestBehavior(userID, itemID, action string) *interfaces.UserBehavior {
	return &interfaces.UserBehavior{
		UserID:    userID,
		ItemID:    itemID,
		Action:    action,
		Timestamp: time.Now(),
		Context: map[string]string{
			"device_type": "mobile",
			"platform":    "ios",
		},
	}
}

// =============================================================================
// GetUser 测试
// =============================================================================

func TestService_GetUser_Success(t *testing.T) {
	service, mockRepo, mockCache := setupTestService(t)

	// 准备测试数据
	testUser := createTestUser("user_001")
	mockRepo.AddUser(testUser)

	// 执行测试
	ctx := context.Background()
	user, err := service.GetUser(ctx, "user_001")

	// 验证结果
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if user == nil {
		t.Fatal("expected user, got nil")
	}
	if user.ID != "user_001" {
		t.Errorf("expected user ID 'user_001', got '%s'", user.ID)
	}
	if user.Name != testUser.Name {
		t.Errorf("expected name '%s', got '%s'", testUser.Name, user.Name)
	}

	// 验证缓存被设置
	if mockCache.SetCalls == 0 {
		t.Error("expected cache to be set")
	}
}

func TestService_GetUser_CacheHit(t *testing.T) {
	service, _, mockCache := setupTestService(t)

	// 预设缓存
	testUser := createTestUser("user_002")
	mockCache.SetCacheValue("user:user_002", testUser, 30*time.Minute)

	// 执行测试
	ctx := context.Background()
	user, err := service.GetUser(ctx, "user_002")

	// 验证结果
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if user == nil {
		t.Fatal("expected user, got nil")
	}
	if user.ID != "user_002" {
		t.Errorf("expected user ID 'user_002', got '%s'", user.ID)
	}
}

func TestService_GetUser_NotFound(t *testing.T) {
	service, _, _ := setupTestService(t)

	// 执行测试
	ctx := context.Background()
	user, err := service.GetUser(ctx, "nonexistent")

	// 验证结果
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if user != nil {
		t.Errorf("expected nil user, got %v", user)
	}
}

func TestService_GetUser_EmptyID(t *testing.T) {
	service, _, _ := setupTestService(t)

	// 执行测试
	ctx := context.Background()
	user, err := service.GetUser(ctx, "")

	// 验证结果
	if err != ErrInvalidRequest {
		t.Errorf("expected ErrInvalidRequest, got %v", err)
	}
	if user != nil {
		t.Errorf("expected nil user, got %v", user)
	}
}

// =============================================================================
// CreateUser 测试
// =============================================================================

func TestService_CreateUser_Success(t *testing.T) {
	service, mockRepo, _ := setupTestService(t)

	// 执行测试
	ctx := context.Background()
	req := &interfaces.CreateUserRequest{
		Name:   "New User",
		Email:  "newuser@example.com",
		Age:    30,
		Gender: "female",
	}

	user, err := service.CreateUser(ctx, req)

	// 验证结果
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if user == nil {
		t.Fatal("expected user, got nil")
	}
	if user.ID == "" {
		t.Error("expected user ID to be generated")
	}
	if user.Name != "New User" {
		t.Errorf("expected name 'New User', got '%s'", user.Name)
	}
	if user.Email != "newuser@example.com" {
		t.Errorf("expected email 'newuser@example.com', got '%s'", user.Email)
	}

	// 验证用户被保存
	savedUser, err := mockRepo.GetByID(ctx, user.ID)
	if err != nil {
		t.Fatalf("expected user to be saved, got error: %v", err)
	}
	if savedUser.Name != user.Name {
		t.Error("saved user name mismatch")
	}
}

func TestService_CreateUser_NilRequest(t *testing.T) {
	service, _, _ := setupTestService(t)

	// 执行测试
	ctx := context.Background()
	user, err := service.CreateUser(ctx, nil)

	// 验证结果
	if err != ErrInvalidRequest {
		t.Errorf("expected ErrInvalidRequest, got %v", err)
	}
	if user != nil {
		t.Errorf("expected nil user, got %v", user)
	}
}

func TestService_CreateUser_MissingRequiredFields(t *testing.T) {
	service, _, _ := setupTestService(t)

	testCases := []struct {
		name string
		req  *interfaces.CreateUserRequest
	}{
		{
			name: "missing name",
			req:  &interfaces.CreateUserRequest{Email: "test@example.com"},
		},
		{
			name: "missing email",
			req:  &interfaces.CreateUserRequest{Name: "Test User"},
		},
		{
			name: "both missing",
			req:  &interfaces.CreateUserRequest{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			user, err := service.CreateUser(ctx, tc.req)

			if err == nil {
				t.Error("expected error, got nil")
			}
			if user != nil {
				t.Errorf("expected nil user, got %v", user)
			}
		})
	}
}

// =============================================================================
// UpdateUser 测试
// =============================================================================

func TestService_UpdateUser_Success(t *testing.T) {
	service, mockRepo, mockCache := setupTestService(t)

	// 准备测试数据
	testUser := createTestUser("user_003")
	mockRepo.AddUser(testUser)

	// 执行测试
	ctx := context.Background()
	req := &interfaces.UpdateUserRequest{
		Name:   "Updated Name",
		Email:  "updated@example.com",
		Age:    35,
		Gender: "other",
	}

	user, err := service.UpdateUser(ctx, "user_003", req)

	// 验证结果
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if user == nil {
		t.Fatal("expected user, got nil")
	}
	if user.Name != "Updated Name" {
		t.Errorf("expected name 'Updated Name', got '%s'", user.Name)
	}
	if user.Email != "updated@example.com" {
		t.Errorf("expected email 'updated@example.com', got '%s'", user.Email)
	}
	if user.Age != 35 {
		t.Errorf("expected age 35, got %d", user.Age)
	}

	// 验证缓存被清除
	if mockCache.DeleteCalls == 0 {
		t.Error("expected cache to be deleted")
	}
}

func TestService_UpdateUser_PartialUpdate(t *testing.T) {
	service, mockRepo, _ := setupTestService(t)

	// 准备测试数据
	testUser := createTestUser("user_004")
	mockRepo.AddUser(testUser)
	originalEmail := testUser.Email

	// 只更新名称
	ctx := context.Background()
	req := &interfaces.UpdateUserRequest{
		Name: "Only Name Updated",
	}

	user, err := service.UpdateUser(ctx, "user_004", req)

	// 验证结果
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if user.Name != "Only Name Updated" {
		t.Errorf("expected name 'Only Name Updated', got '%s'", user.Name)
	}
	// Email 应该保持不变
	if user.Email != originalEmail {
		t.Errorf("expected email '%s', got '%s'", originalEmail, user.Email)
	}
}

func TestService_UpdateUser_NotFound(t *testing.T) {
	service, _, _ := setupTestService(t)

	ctx := context.Background()
	req := &interfaces.UpdateUserRequest{Name: "Test"}

	user, err := service.UpdateUser(ctx, "nonexistent", req)

	if err == nil {
		t.Error("expected error, got nil")
	}
	if user != nil {
		t.Errorf("expected nil user, got %v", user)
	}
}

// =============================================================================
// DeleteUser 测试
// =============================================================================

func TestService_DeleteUser_Success(t *testing.T) {
	service, mockRepo, mockCache := setupTestService(t)

	// 准备测试数据
	testUser := createTestUser("user_005")
	mockRepo.AddUser(testUser)

	// 执行测试
	ctx := context.Background()
	err := service.DeleteUser(ctx, "user_005")

	// 验证结果
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	// 验证用户被删除
	_, err = mockRepo.GetByID(ctx, "user_005")
	if err == nil {
		t.Error("expected user to be deleted")
	}

	// 验证缓存被清除
	if mockCache.DeleteCalls == 0 {
		t.Error("expected cache to be deleted")
	}
}

func TestService_DeleteUser_EmptyID(t *testing.T) {
	service, _, _ := setupTestService(t)

	ctx := context.Background()
	err := service.DeleteUser(ctx, "")

	if err != ErrInvalidRequest {
		t.Errorf("expected ErrInvalidRequest, got %v", err)
	}
}

// =============================================================================
// RecordBehavior 测试
// =============================================================================

func TestService_RecordBehavior_Success(t *testing.T) {
	service, mockRepo, _ := setupTestService(t)

	// 执行测试
	ctx := context.Background()
	req := &interfaces.RecordBehaviorRequest{
		UserID: "user_006",
		ItemID: "item_001",
		Action: "view",
		Context: map[string]string{
			"device_type": "mobile",
		},
	}

	err := service.RecordBehavior(ctx, req)

	// 验证结果
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	// 验证行为被记录
	behaviors, err := mockRepo.GetBehaviors(ctx, "user_006", 10)
	if err != nil {
		t.Fatalf("expected no error getting behaviors, got %v", err)
	}
	if len(behaviors) != 1 {
		t.Errorf("expected 1 behavior, got %d", len(behaviors))
	}
	if behaviors[0].Action != "view" {
		t.Errorf("expected action 'view', got '%s'", behaviors[0].Action)
	}
}

func TestService_RecordBehavior_MissingFields(t *testing.T) {
	service, _, _ := setupTestService(t)

	testCases := []struct {
		name string
		req  *interfaces.RecordBehaviorRequest
	}{
		{
			name: "missing user_id",
			req:  &interfaces.RecordBehaviorRequest{ItemID: "item", Action: "view"},
		},
		{
			name: "missing item_id",
			req:  &interfaces.RecordBehaviorRequest{UserID: "user", Action: "view"},
		},
		{
			name: "missing action",
			req:  &interfaces.RecordBehaviorRequest{UserID: "user", ItemID: "item"},
		},
		{
			name: "nil request",
			req:  nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			err := service.RecordBehavior(ctx, tc.req)

			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

// =============================================================================
// GetUserBehaviors 测试
// =============================================================================

func TestService_GetUserBehaviors_Success(t *testing.T) {
	service, mockRepo, _ := setupTestService(t)

	// 准备测试数据
	for i := 0; i < 5; i++ {
		behavior := createTestBehavior("user_007", fmt.Sprintf("item_%03d", i), "view")
		mockRepo.AddTestBehavior(behavior)
	}

	// 执行测试
	ctx := context.Background()
	behaviors, err := service.GetUserBehaviors(ctx, "user_007", 10)

	// 验证结果
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(behaviors) != 5 {
		t.Errorf("expected 5 behaviors, got %d", len(behaviors))
	}
}

func TestService_GetUserBehaviors_WithLimit(t *testing.T) {
	service, mockRepo, _ := setupTestService(t)

	// 准备测试数据
	for i := 0; i < 10; i++ {
		behavior := createTestBehavior("user_008", fmt.Sprintf("item_%03d", i), "view")
		mockRepo.AddTestBehavior(behavior)
	}

	// 执行测试
	ctx := context.Background()
	behaviors, err := service.GetUserBehaviors(ctx, "user_008", 5)

	// 验证结果
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(behaviors) != 5 {
		t.Errorf("expected 5 behaviors, got %d", len(behaviors))
	}
}

func TestService_GetUserBehaviors_DefaultLimit(t *testing.T) {
	service, _, _ := setupTestService(t)

	// 使用默认 limit（传入 0 或负数）
	ctx := context.Background()
	behaviors, err := service.GetUserBehaviors(ctx, "user_009", 0)

	// 验证结果
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	// 应该使用默认 limit，不会出错
	_ = behaviors
}

func TestService_GetUserBehaviors_EmptyUserID(t *testing.T) {
	service, _, _ := setupTestService(t)

	ctx := context.Background()
	behaviors, err := service.GetUserBehaviors(ctx, "", 10)

	if err != ErrInvalidRequest {
		t.Errorf("expected ErrInvalidRequest, got %v", err)
	}
	if behaviors != nil {
		t.Error("expected nil behaviors")
	}
}

// =============================================================================
// GetUserProfile 测试
// =============================================================================

func TestService_GetUserProfile_Success(t *testing.T) {
	service, mockRepo, _ := setupTestService(t)

	// 准备测试数据
	testUser := createTestUser("user_010")
	mockRepo.AddUser(testUser)

	// 添加一些行为
	actions := []string{"view", "click", "view", "like"}
	for i, action := range actions {
		behavior := createTestBehavior("user_010", fmt.Sprintf("item_%03d", i), action)
		mockRepo.AddTestBehavior(behavior)
	}

	// 执行测试
	ctx := context.Background()
	profile, err := service.GetUserProfile(ctx, "user_010")

	// 验证结果
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if profile == nil {
		t.Fatal("expected profile, got nil")
	}
	if profile.User == nil {
		t.Fatal("expected profile.User, got nil")
	}
	if profile.User.ID != "user_010" {
		t.Errorf("expected user ID 'user_010', got '%s'", profile.User.ID)
	}
	if profile.TotalActions != 4 {
		t.Errorf("expected 4 total actions, got %d", profile.TotalActions)
	}
	if profile.PreferredTypes == nil {
		t.Error("expected PreferredTypes, got nil")
	}
	if profile.PreferredTypes["view"] != 2 {
		t.Errorf("expected 2 views, got %d", profile.PreferredTypes["view"])
	}
}

func TestService_GetUserProfile_UserNotFound(t *testing.T) {
	service, _, _ := setupTestService(t)

	ctx := context.Background()
	profile, err := service.GetUserProfile(ctx, "nonexistent")

	if err == nil {
		t.Error("expected error, got nil")
	}
	if profile != nil {
		t.Errorf("expected nil profile, got %v", profile)
	}
}

func TestService_GetUserProfile_CacheHit(t *testing.T) {
	service, mockRepo, mockCache := setupTestService(t)

	// 准备测试数据
	testUser := createTestUser("user_011")
	mockRepo.AddUser(testUser)

	cachedProfile := &interfaces.UserProfile{
		User:           testUser,
		TotalActions:   100,
		PreferredTypes: map[string]int{"cached": 50},
		ActiveHours:    map[int]int{10: 5},
	}
	mockCache.SetCacheValue("user:profile:user_011", cachedProfile, 15*time.Minute)

	// 执行测试
	ctx := context.Background()
	profile, err := service.GetUserProfile(ctx, "user_011")

	// 验证结果
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if profile == nil {
		t.Fatal("expected profile, got nil")
	}
	// 应该返回缓存的数据
	if profile.TotalActions != 100 {
		t.Errorf("expected 100 total actions from cache, got %d", profile.TotalActions)
	}
}

// =============================================================================
// 辅助函数测试
// =============================================================================

func TestGenerateUserID(t *testing.T) {
	// 生成多个 ID 确保唯一性
	ids := make(map[string]bool)
	for i := 0; i < 100; i++ {
		id, err := generateUserID()
		if err != nil {
			t.Fatalf("failed to generate user ID: %v", err)
		}

		// 验证格式
		if len(id) < 20 {
			t.Errorf("user ID too short: %s", id)
		}
		if id[:2] != "u_" {
			t.Errorf("user ID should start with 'u_': %s", id)
		}

		// 验证唯一性
		if ids[id] {
			t.Errorf("duplicate user ID generated: %s", id)
		}
		ids[id] = true
	}
}

func TestService_CalculatePreferredTypes(t *testing.T) {
	service, _, _ := setupTestService(t)

	behaviors := []*interfaces.UserBehavior{
		{Action: "view"},
		{Action: "view"},
		{Action: "click"},
		{Action: "like"},
		{Action: "view"},
	}

	types := service.calculatePreferredTypes(behaviors)

	if types["view"] != 3 {
		t.Errorf("expected 3 views, got %d", types["view"])
	}
	if types["click"] != 1 {
		t.Errorf("expected 1 click, got %d", types["click"])
	}
	if types["like"] != 1 {
		t.Errorf("expected 1 like, got %d", types["like"])
	}
}

func TestService_CalculateActiveHours(t *testing.T) {
	service, _, _ := setupTestService(t)

	// 创建不同时间的行为
	now := time.Now()
	behaviors := []*interfaces.UserBehavior{
		{Timestamp: time.Date(now.Year(), now.Month(), now.Day(), 10, 0, 0, 0, now.Location())},
		{Timestamp: time.Date(now.Year(), now.Month(), now.Day(), 10, 30, 0, 0, now.Location())},
		{Timestamp: time.Date(now.Year(), now.Month(), now.Day(), 14, 0, 0, 0, now.Location())},
	}

	hours := service.calculateActiveHours(behaviors)

	if hours[10] != 2 {
		t.Errorf("expected 2 activities at hour 10, got %d", hours[10])
	}
	if hours[14] != 1 {
		t.Errorf("expected 1 activity at hour 14, got %d", hours[14])
	}
}

func TestService_GetLastActiveTime(t *testing.T) {
	service, _, _ := setupTestService(t)

	// 空列表
	emptyTime := service.getLastActiveTime([]*interfaces.UserBehavior{})
	if !emptyTime.IsZero() {
		t.Error("expected zero time for empty behaviors")
	}

	// 有行为
	now := time.Now()
	behaviors := []*interfaces.UserBehavior{
		{Timestamp: now},
		{Timestamp: now.Add(-1 * time.Hour)},
	}
	lastActive := service.getLastActiveTime(behaviors)
	if !lastActive.Equal(now) {
		t.Errorf("expected last active time to be %v, got %v", now, lastActive)
	}
}

// =============================================================================
// 并发测试
// =============================================================================

func TestService_ConcurrentGetUser(t *testing.T) {
	service, mockRepo, _ := setupTestService(t)

	// 准备测试数据
	testUser := createTestUser("user_concurrent")
	mockRepo.AddUser(testUser)

	// 并发获取
	ctx := context.Background()
	done := make(chan bool)
	errChan := make(chan error)

	for i := 0; i < 100; i++ {
		go func() {
			user, err := service.GetUser(ctx, "user_concurrent")
			if err != nil {
				errChan <- err
				return
			}
			if user == nil {
				errChan <- fmt.Errorf("user is nil")
				return
			}
			done <- true
		}()
	}

	// 等待所有 goroutine 完成
	successCount := 0
	for i := 0; i < 100; i++ {
		select {
		case <-done:
			successCount++
		case err := <-errChan:
			t.Errorf("concurrent error: %v", err)
		case <-time.After(5 * time.Second):
			t.Fatal("timeout waiting for concurrent operations")
		}
	}

	if successCount != 100 {
		t.Errorf("expected 100 successful operations, got %d", successCount)
	}
}

// =============================================================================
// 边界条件测试
// =============================================================================

func TestService_GetUserBehaviors_MaxLimit(t *testing.T) {
	service, _, _ := setupTestService(t)

	// 请求超过最大限制
	ctx := context.Background()
	behaviors, err := service.GetUserBehaviors(ctx, "user_max", 10000)

	// 应该被限制到 maxBehaviorLimit
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	_ = behaviors // 验证不会 panic
}

func TestService_ContextCancellation(t *testing.T) {
	service, mockRepo, _ := setupTestService(t)

	// 设置延迟返回
	mockRepo.GetByIDFunc = func(ctx context.Context, userID string) (*interfaces.User, error) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(100 * time.Millisecond):
			return createTestUser(userID), nil
		}
	}

	// 使用已取消的 context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	user, err := service.GetUser(ctx, "user_cancel")
	if err == nil {
		t.Error("expected error for cancelled context")
	}
	if user != nil {
		t.Error("expected nil user for cancelled context")
	}
}

