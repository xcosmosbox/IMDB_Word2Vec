// Package v1 提供用户 API 处理器的单元测试
package v1

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// Mock 用户服务
// =============================================================================

// MockUserService Mock 用户服务实现
type MockUserService struct {
	GetUserFunc         func(ctx context.Context, userID string) (*interfaces.User, error)
	CreateUserFunc      func(ctx context.Context, req *interfaces.CreateUserRequest) (*interfaces.User, error)
	UpdateUserFunc      func(ctx context.Context, userID string, req *interfaces.UpdateUserRequest) (*interfaces.User, error)
	DeleteUserFunc      func(ctx context.Context, userID string) error
	RecordBehaviorFunc  func(ctx context.Context, req *interfaces.RecordBehaviorRequest) error
	GetUserBehaviorsFunc func(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error)
	GetUserProfileFunc  func(ctx context.Context, userID string) (*interfaces.UserProfile, error)
}

func (m *MockUserService) GetUser(ctx context.Context, userID string) (*interfaces.User, error) {
	if m.GetUserFunc != nil {
		return m.GetUserFunc(ctx, userID)
	}
	return nil, fmt.Errorf("not implemented")
}

func (m *MockUserService) CreateUser(ctx context.Context, req *interfaces.CreateUserRequest) (*interfaces.User, error) {
	if m.CreateUserFunc != nil {
		return m.CreateUserFunc(ctx, req)
	}
	return nil, fmt.Errorf("not implemented")
}

func (m *MockUserService) UpdateUser(ctx context.Context, userID string, req *interfaces.UpdateUserRequest) (*interfaces.User, error) {
	if m.UpdateUserFunc != nil {
		return m.UpdateUserFunc(ctx, userID, req)
	}
	return nil, fmt.Errorf("not implemented")
}

func (m *MockUserService) DeleteUser(ctx context.Context, userID string) error {
	if m.DeleteUserFunc != nil {
		return m.DeleteUserFunc(ctx, userID)
	}
	return fmt.Errorf("not implemented")
}

func (m *MockUserService) RecordBehavior(ctx context.Context, req *interfaces.RecordBehaviorRequest) error {
	if m.RecordBehaviorFunc != nil {
		return m.RecordBehaviorFunc(ctx, req)
	}
	return fmt.Errorf("not implemented")
}

func (m *MockUserService) GetUserBehaviors(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error) {
	if m.GetUserBehaviorsFunc != nil {
		return m.GetUserBehaviorsFunc(ctx, userID, limit)
	}
	return nil, fmt.Errorf("not implemented")
}

func (m *MockUserService) GetUserProfile(ctx context.Context, userID string) (*interfaces.UserProfile, error) {
	if m.GetUserProfileFunc != nil {
		return m.GetUserProfileFunc(ctx, userID)
	}
	return nil, fmt.Errorf("not implemented")
}

var _ interfaces.UserService = (*MockUserService)(nil)

// =============================================================================
// 测试辅助函数
// =============================================================================

// setupTestRouter 创建测试路由
func setupTestRouter(handler *Handler) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	apiV1 := router.Group("/api/v1")
	handler.RegisterRoutes(apiV1)
	return router
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
		CreatedAt: now,
		UpdatedAt: now,
	}
}

// performRequest 执行 HTTP 请求
func performRequest(router *gin.Engine, method, path string, body interface{}) *httptest.ResponseRecorder {
	var reqBody []byte
	if body != nil {
		reqBody, _ = json.Marshal(body)
	}

	req, _ := http.NewRequest(method, path, bytes.NewBuffer(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	return w
}

// parseResponse 解析响应
func parseResponse(w *httptest.ResponseRecorder) (Response, error) {
	var resp Response
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	return resp, err
}

// =============================================================================
// CreateUser 测试
// =============================================================================

func TestHandler_CreateUser_Success(t *testing.T) {
	mockService := &MockUserService{
		CreateUserFunc: func(ctx context.Context, req *interfaces.CreateUserRequest) (*interfaces.User, error) {
			return &interfaces.User{
				ID:        "user_new",
				Name:      req.Name,
				Email:     req.Email,
				Age:       req.Age,
				Gender:    req.Gender,
				CreatedAt: time.Now(),
				UpdatedAt: time.Now(),
			}, nil
		},
	}

	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	body := CreateUserRequest{
		Name:   "New User",
		Email:  "new@example.com",
		Age:    30,
		Gender: "female",
	}

	w := performRequest(router, "POST", "/api/v1/users", body)

	if w.Code != http.StatusCreated {
		t.Errorf("expected status %d, got %d", http.StatusCreated, w.Code)
	}

	resp, err := parseResponse(w)
	if err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}
	if resp.Code != codeSuccess {
		t.Errorf("expected code %d, got %d", codeSuccess, resp.Code)
	}
}

func TestHandler_CreateUser_InvalidRequest(t *testing.T) {
	mockService := &MockUserService{}
	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	// 缺少必填字段
	body := CreateUserRequest{
		Name: "Only Name",
		// Email is missing
	}

	w := performRequest(router, "POST", "/api/v1/users", body)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

func TestHandler_CreateUser_ServiceError(t *testing.T) {
	mockService := &MockUserService{
		CreateUserFunc: func(ctx context.Context, req *interfaces.CreateUserRequest) (*interfaces.User, error) {
			return nil, fmt.Errorf("database error")
		},
	}

	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	body := CreateUserRequest{
		Name:  "Test User",
		Email: "test@example.com",
	}

	w := performRequest(router, "POST", "/api/v1/users", body)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("expected status %d, got %d", http.StatusInternalServerError, w.Code)
	}
}

// =============================================================================
// GetUser 测试
// =============================================================================

func TestHandler_GetUser_Success(t *testing.T) {
	testUser := createTestUser("user_001")
	mockService := &MockUserService{
		GetUserFunc: func(ctx context.Context, userID string) (*interfaces.User, error) {
			if userID == "user_001" {
				return testUser, nil
			}
			return nil, fmt.Errorf("user not found")
		},
	}

	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	w := performRequest(router, "GET", "/api/v1/users/user_001", nil)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}

	resp, err := parseResponse(w)
	if err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}
	if resp.Code != codeSuccess {
		t.Errorf("expected code %d, got %d", codeSuccess, resp.Code)
	}
}

func TestHandler_GetUser_NotFound(t *testing.T) {
	mockService := &MockUserService{
		GetUserFunc: func(ctx context.Context, userID string) (*interfaces.User, error) {
			return nil, fmt.Errorf("user not found")
		},
	}

	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	w := performRequest(router, "GET", "/api/v1/users/nonexistent", nil)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected status %d, got %d", http.StatusNotFound, w.Code)
	}
}

// =============================================================================
// UpdateUser 测试
// =============================================================================

func TestHandler_UpdateUser_Success(t *testing.T) {
	mockService := &MockUserService{
		UpdateUserFunc: func(ctx context.Context, userID string, req *interfaces.UpdateUserRequest) (*interfaces.User, error) {
			return &interfaces.User{
				ID:        userID,
				Name:      req.Name,
				Email:     "original@example.com",
				UpdatedAt: time.Now(),
			}, nil
		},
	}

	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	body := UpdateUserRequest{
		Name: "Updated Name",
	}

	w := performRequest(router, "PUT", "/api/v1/users/user_001", body)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}
}

func TestHandler_UpdateUser_InvalidRequest(t *testing.T) {
	mockService := &MockUserService{}
	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	// 发送无效 JSON
	req, _ := http.NewRequest("PUT", "/api/v1/users/user_001", bytes.NewBuffer([]byte("invalid json")))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

// =============================================================================
// DeleteUser 测试
// =============================================================================

func TestHandler_DeleteUser_Success(t *testing.T) {
	mockService := &MockUserService{
		DeleteUserFunc: func(ctx context.Context, userID string) error {
			return nil
		},
	}

	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	w := performRequest(router, "DELETE", "/api/v1/users/user_001", nil)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}

	resp, err := parseResponse(w)
	if err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}
	if resp.Code != codeSuccess {
		t.Errorf("expected code %d, got %d", codeSuccess, resp.Code)
	}
}

func TestHandler_DeleteUser_Error(t *testing.T) {
	mockService := &MockUserService{
		DeleteUserFunc: func(ctx context.Context, userID string) error {
			return fmt.Errorf("delete error")
		},
	}

	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	w := performRequest(router, "DELETE", "/api/v1/users/user_001", nil)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("expected status %d, got %d", http.StatusInternalServerError, w.Code)
	}
}

// =============================================================================
// GetUserBehaviors 测试
// =============================================================================

func TestHandler_GetUserBehaviors_Success(t *testing.T) {
	mockService := &MockUserService{
		GetUserBehaviorsFunc: func(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error) {
			return []*interfaces.UserBehavior{
				{UserID: userID, ItemID: "item_001", Action: "view", Timestamp: time.Now()},
				{UserID: userID, ItemID: "item_002", Action: "click", Timestamp: time.Now()},
			}, nil
		},
	}

	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	w := performRequest(router, "GET", "/api/v1/users/user_001/behaviors?limit=10", nil)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}

	resp, err := parseResponse(w)
	if err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}
	if resp.Code != codeSuccess {
		t.Errorf("expected code %d, got %d", codeSuccess, resp.Code)
	}
}

func TestHandler_GetUserBehaviors_WithLimit(t *testing.T) {
	var capturedLimit int
	mockService := &MockUserService{
		GetUserBehaviorsFunc: func(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error) {
			capturedLimit = limit
			return []*interfaces.UserBehavior{}, nil
		},
	}

	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	w := performRequest(router, "GET", "/api/v1/users/user_001/behaviors?limit=50", nil)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}
	if capturedLimit != 50 {
		t.Errorf("expected limit 50, got %d", capturedLimit)
	}
}

// =============================================================================
// RecordBehavior 测试
// =============================================================================

func TestHandler_RecordBehavior_Success(t *testing.T) {
	mockService := &MockUserService{
		RecordBehaviorFunc: func(ctx context.Context, req *interfaces.RecordBehaviorRequest) error {
			return nil
		},
	}

	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	body := RecordBehaviorRequest{
		ItemID: "item_001",
		Action: "view",
		Context: map[string]string{
			"device_type": "mobile",
		},
	}

	w := performRequest(router, "POST", "/api/v1/users/user_001/behaviors", body)

	if w.Code != http.StatusCreated {
		t.Errorf("expected status %d, got %d", http.StatusCreated, w.Code)
	}
}

func TestHandler_RecordBehavior_InvalidRequest(t *testing.T) {
	mockService := &MockUserService{}
	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	// 缺少必填字段
	body := RecordBehaviorRequest{
		// ItemID and Action are missing
	}

	w := performRequest(router, "POST", "/api/v1/users/user_001/behaviors", body)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

// =============================================================================
// GetUserProfile 测试
// =============================================================================

func TestHandler_GetUserProfile_Success(t *testing.T) {
	testUser := createTestUser("user_001")
	mockService := &MockUserService{
		GetUserProfileFunc: func(ctx context.Context, userID string) (*interfaces.UserProfile, error) {
			return &interfaces.UserProfile{
				User:           testUser,
				TotalActions:   100,
				PreferredTypes: map[string]int{"view": 50, "click": 30, "like": 20},
				ActiveHours:    map[int]int{10: 20, 14: 30, 20: 50},
				LastActive:     time.Now(),
			}, nil
		},
	}

	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	w := performRequest(router, "GET", "/api/v1/users/user_001/profile", nil)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}

	resp, err := parseResponse(w)
	if err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}
	if resp.Code != codeSuccess {
		t.Errorf("expected code %d, got %d", codeSuccess, resp.Code)
	}
}

func TestHandler_GetUserProfile_Error(t *testing.T) {
	mockService := &MockUserService{
		GetUserProfileFunc: func(ctx context.Context, userID string) (*interfaces.UserProfile, error) {
			return nil, fmt.Errorf("profile not found")
		},
	}

	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	w := performRequest(router, "GET", "/api/v1/users/user_001/profile", nil)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("expected status %d, got %d", http.StatusInternalServerError, w.Code)
	}
}

// =============================================================================
// 路由注册测试
// =============================================================================

func TestHandler_RegisterRoutes(t *testing.T) {
	mockService := &MockUserService{}
	handler := NewHandler(mockService, zap.NewNop())

	gin.SetMode(gin.TestMode)
	router := gin.New()
	apiV1 := router.Group("/api/v1")
	handler.RegisterRoutes(apiV1)

	// 验证所有路由都已注册
	routes := router.Routes()

	expectedRoutes := map[string]string{
		"POST:/api/v1/users":               "CreateUser",
		"GET:/api/v1/users/:id":            "GetUser",
		"PUT:/api/v1/users/:id":            "UpdateUser",
		"DELETE:/api/v1/users/:id":         "DeleteUser",
		"GET:/api/v1/users/:id/behaviors":  "GetUserBehaviors",
		"POST:/api/v1/users/:id/behaviors": "RecordBehavior",
		"GET:/api/v1/users/:id/profile":    "GetUserProfile",
	}

	registeredRoutes := make(map[string]bool)
	for _, route := range routes {
		key := route.Method + ":" + route.Path
		registeredRoutes[key] = true
	}

	for route := range expectedRoutes {
		if !registeredRoutes[route] {
			t.Errorf("expected route %s to be registered", route)
		}
	}
}

// =============================================================================
// 边界条件测试
// =============================================================================

func TestHandler_EmptyUserID(t *testing.T) {
	mockService := &MockUserService{}
	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	// 注意：Gin 的路由不会匹配空 ID
	// 这里测试根路径
	w := performRequest(router, "GET", "/api/v1/users/", nil)

	// 应该返回 404，因为没有匹配的路由
	if w.Code != http.StatusNotFound && w.Code != http.StatusBadRequest {
		t.Logf("status code for empty ID: %d", w.Code)
	}
}

func TestHandler_LargeLimit(t *testing.T) {
	var capturedLimit int
	mockService := &MockUserService{
		GetUserBehaviorsFunc: func(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error) {
			capturedLimit = limit
			return []*interfaces.UserBehavior{}, nil
		},
	}

	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	// 请求超大 limit
	w := performRequest(router, "GET", "/api/v1/users/user_001/behaviors?limit=10000", nil)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}

	// 验证 limit 被限制到最大值
	if capturedLimit > maxLimit {
		t.Errorf("expected limit to be capped at %d, got %d", maxLimit, capturedLimit)
	}
}

func TestHandler_InvalidLimit(t *testing.T) {
	var capturedLimit int
	mockService := &MockUserService{
		GetUserBehaviorsFunc: func(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error) {
			capturedLimit = limit
			return []*interfaces.UserBehavior{}, nil
		},
	}

	handler := NewHandler(mockService, zap.NewNop())
	router := setupTestRouter(handler)

	// 使用无效的 limit 字符串
	w := performRequest(router, "GET", "/api/v1/users/user_001/behaviors?limit=abc", nil)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}

	// 应该使用默认 limit
	if capturedLimit != defaultLimit {
		t.Errorf("expected default limit %d, got %d", defaultLimit, capturedLimit)
	}
}

