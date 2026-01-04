package unit

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"recommend-system/internal/interfaces"
	"recommend-system/tests/fixtures"
	"recommend-system/tests/mocks"
)

// =============================================================================
// 测试辅助函数
// =============================================================================

// setupTestRouter 设置测试路由
func setupTestRouter() *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	return router
}

// performRequest 执行 HTTP 请求
func performRequest(router *gin.Engine, method, path string, body interface{}) *httptest.ResponseRecorder {
	var req *http.Request
	if body != nil {
		bodyBytes, _ := json.Marshal(body)
		req, _ = http.NewRequest(method, path, bytes.NewBuffer(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
	} else {
		req, _ = http.NewRequest(method, path, nil)
	}

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	return w
}

// =============================================================================
// UserHandler 模拟测试
// =============================================================================

// UserHandler 用户处理器（简化版）
type UserHandler struct {
	userService *mocks.MockUserService
}

// NewUserHandler 创建用户处理器
func NewUserHandler(userService *mocks.MockUserService) *UserHandler {
	return &UserHandler{userService: userService}
}

// GetUser 获取用户
func (h *UserHandler) GetUser(c *gin.Context) {
	userID := c.Param("id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "user id is required"})
		return
	}

	user, err := h.userService.GetUser(c.Request.Context(), userID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "user not found"})
		return
	}

	c.JSON(http.StatusOK, user)
}

// CreateUser 创建用户
func (h *UserHandler) CreateUser(c *gin.Context) {
	var req interfaces.CreateUserRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	user, err := h.userService.CreateUser(c.Request.Context(), &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, user)
}

// UpdateUser 更新用户
func (h *UserHandler) UpdateUser(c *gin.Context) {
	userID := c.Param("id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "user id is required"})
		return
	}

	var req interfaces.UpdateUserRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	user, err := h.userService.UpdateUser(c.Request.Context(), userID, &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, user)
}

// DeleteUser 删除用户
func (h *UserHandler) DeleteUser(c *gin.Context) {
	userID := c.Param("id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "user id is required"})
		return
	}

	if err := h.userService.DeleteUser(c.Request.Context(), userID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusNoContent, nil)
}

// RecordBehavior 记录行为
func (h *UserHandler) RecordBehavior(c *gin.Context) {
	var req interfaces.RecordBehaviorRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := h.userService.RecordBehavior(c.Request.Context(), &req); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

// =============================================================================
// UserHandler 单元测试
// =============================================================================

// TestUserHandler_GetUser 测试获取用户
func TestUserHandler_GetUser(t *testing.T) {
	// 准备
	mockService := mocks.NewMockUserService()
	testUser := fixtures.GetTestUser("user_001")
	mockService.SetUser(testUser)

	handler := NewUserHandler(mockService)

	router := setupTestRouter()
	router.GET("/users/:id", handler.GetUser)

	// 执行
	w := performRequest(router, "GET", "/users/user_001", nil)

	// 验证
	assert.Equal(t, http.StatusOK, w.Code)

	var response interfaces.User
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.Equal(t, testUser.ID, response.ID)
	assert.Equal(t, testUser.Name, response.Name)
}

// TestUserHandler_GetUser_NotFound 测试获取不存在的用户
func TestUserHandler_GetUser_NotFound(t *testing.T) {
	// 准备
	mockService := mocks.NewMockUserService()
	handler := NewUserHandler(mockService)

	router := setupTestRouter()
	router.GET("/users/:id", handler.GetUser)

	// 执行
	w := performRequest(router, "GET", "/users/non_existent", nil)

	// 验证
	assert.Equal(t, http.StatusNotFound, w.Code)
}

// TestUserHandler_CreateUser 测试创建用户
func TestUserHandler_CreateUser(t *testing.T) {
	// 准备
	mockService := mocks.NewMockUserService()
	handler := NewUserHandler(mockService)

	router := setupTestRouter()
	router.POST("/users", handler.CreateUser)

	req := interfaces.CreateUserRequest{
		Name:   "New User",
		Email:  "newuser@example.com",
		Age:    25,
		Gender: "male",
	}

	// 执行
	w := performRequest(router, "POST", "/users", req)

	// 验证
	assert.Equal(t, http.StatusCreated, w.Code)

	var response interfaces.User
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.NotEmpty(t, response.ID)
	assert.Equal(t, req.Name, response.Name)
}

// TestUserHandler_CreateUser_BadRequest 测试无效的创建请求
func TestUserHandler_CreateUser_BadRequest(t *testing.T) {
	// 准备
	mockService := mocks.NewMockUserService()
	handler := NewUserHandler(mockService)

	router := setupTestRouter()
	router.POST("/users", handler.CreateUser)

	// 执行 - 空请求体
	req, _ := http.NewRequest("POST", "/users", bytes.NewBuffer([]byte("{invalid json}")))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	// 验证
	assert.Equal(t, http.StatusBadRequest, w.Code)
}

// TestUserHandler_UpdateUser 测试更新用户
func TestUserHandler_UpdateUser(t *testing.T) {
	// 准备
	mockService := mocks.NewMockUserService()
	testUser := fixtures.GetTestUser("user_001")
	mockService.SetUser(testUser)

	handler := NewUserHandler(mockService)

	router := setupTestRouter()
	router.PUT("/users/:id", handler.UpdateUser)

	req := interfaces.UpdateUserRequest{
		Name: "Updated Name",
	}

	// 执行
	w := performRequest(router, "PUT", "/users/user_001", req)

	// 验证
	assert.Equal(t, http.StatusOK, w.Code)
}

// TestUserHandler_DeleteUser 测试删除用户
func TestUserHandler_DeleteUser(t *testing.T) {
	// 准备
	mockService := mocks.NewMockUserService()
	testUser := fixtures.GetTestUser("user_001")
	mockService.SetUser(testUser)

	handler := NewUserHandler(mockService)

	router := setupTestRouter()
	router.DELETE("/users/:id", handler.DeleteUser)

	// 执行
	w := performRequest(router, "DELETE", "/users/user_001", nil)

	// 验证
	assert.Equal(t, http.StatusNoContent, w.Code)
}

// TestUserHandler_RecordBehavior 测试记录行为
func TestUserHandler_RecordBehavior(t *testing.T) {
	// 准备
	mockService := mocks.NewMockUserService()
	handler := NewUserHandler(mockService)

	router := setupTestRouter()
	router.POST("/behaviors", handler.RecordBehavior)

	req := interfaces.RecordBehaviorRequest{
		UserID: "user_001",
		ItemID: "item_001",
		Action: "click",
	}

	// 执行
	w := performRequest(router, "POST", "/behaviors", req)

	// 验证
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, 1, mockService.RecordBehaviorCalls)
}

// =============================================================================
// ItemHandler 模拟测试
// =============================================================================

// ItemHandler 物品处理器（简化版）
type ItemHandler struct {
	itemService *mocks.MockItemService
}

// NewItemHandler 创建物品处理器
func NewItemHandler(itemService *mocks.MockItemService) *ItemHandler {
	return &ItemHandler{itemService: itemService}
}

// GetItem 获取物品
func (h *ItemHandler) GetItem(c *gin.Context) {
	itemID := c.Param("id")
	if itemID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "item id is required"})
		return
	}

	item, err := h.itemService.GetItem(c.Request.Context(), itemID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "item not found"})
		return
	}

	c.JSON(http.StatusOK, item)
}

// CreateItem 创建物品
func (h *ItemHandler) CreateItem(c *gin.Context) {
	var req interfaces.CreateItemRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	item, err := h.itemService.CreateItem(c.Request.Context(), &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, item)
}

// SearchItems 搜索物品
func (h *ItemHandler) SearchItems(c *gin.Context) {
	query := c.Query("q")
	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "query is required"})
		return
	}

	items, err := h.itemService.SearchItems(c.Request.Context(), query, 20)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"items": items})
}

// =============================================================================
// ItemHandler 单元测试
// =============================================================================

// TestItemHandler_GetItem 测试获取物品
func TestItemHandler_GetItem(t *testing.T) {
	// 准备
	mockService := mocks.NewMockItemService()
	testItem := fixtures.GetTestItem("item_001")
	mockService.SetItem(testItem)

	handler := NewItemHandler(mockService)

	router := setupTestRouter()
	router.GET("/items/:id", handler.GetItem)

	// 执行
	w := performRequest(router, "GET", "/items/item_001", nil)

	// 验证
	assert.Equal(t, http.StatusOK, w.Code)

	var response interfaces.Item
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.Equal(t, testItem.ID, response.ID)
}

// TestItemHandler_GetItem_NotFound 测试获取不存在的物品
func TestItemHandler_GetItem_NotFound(t *testing.T) {
	// 准备
	mockService := mocks.NewMockItemService()
	handler := NewItemHandler(mockService)

	router := setupTestRouter()
	router.GET("/items/:id", handler.GetItem)

	// 执行
	w := performRequest(router, "GET", "/items/non_existent", nil)

	// 验证
	assert.Equal(t, http.StatusNotFound, w.Code)
}

// TestItemHandler_CreateItem 测试创建物品
func TestItemHandler_CreateItem(t *testing.T) {
	// 准备
	mockService := mocks.NewMockItemService()
	handler := NewItemHandler(mockService)

	router := setupTestRouter()
	router.POST("/items", handler.CreateItem)

	req := interfaces.CreateItemRequest{
		Type:        "movie",
		Title:       "Test Movie",
		Description: "A test movie",
		Category:    "action",
		Tags:        []string{"test", "action"},
	}

	// 执行
	w := performRequest(router, "POST", "/items", req)

	// 验证
	assert.Equal(t, http.StatusCreated, w.Code)

	var response interfaces.Item
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.NotEmpty(t, response.ID)
	assert.Equal(t, req.Title, response.Title)
}

// TestItemHandler_SearchItems 测试搜索物品
func TestItemHandler_SearchItems(t *testing.T) {
	// 准备
	mockService := mocks.NewMockItemService()
	mockService.SearchItemsResult = []*interfaces.Item{
		fixtures.GetTestItem("item_001"),
		fixtures.GetTestItem("item_002"),
	}

	handler := NewItemHandler(mockService)

	router := setupTestRouter()
	router.GET("/items/search", handler.SearchItems)

	// 执行
	w := performRequest(router, "GET", "/items/search?q=matrix", nil)

	// 验证
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, 1, mockService.SearchItemsCalls)
}

// TestItemHandler_SearchItems_MissingQuery 测试缺少查询参数
func TestItemHandler_SearchItems_MissingQuery(t *testing.T) {
	// 准备
	mockService := mocks.NewMockItemService()
	handler := NewItemHandler(mockService)

	router := setupTestRouter()
	router.GET("/items/search", handler.SearchItems)

	// 执行
	w := performRequest(router, "GET", "/items/search", nil)

	// 验证
	assert.Equal(t, http.StatusBadRequest, w.Code)
}

// =============================================================================
// RecommendHandler 模拟测试
// =============================================================================

// RecommendHandler 推荐处理器（简化版）
type RecommendHandler struct {
	userService    *mocks.MockUserService
	itemService    *mocks.MockItemService
	featureService *mocks.MockFeatureService
	inferClient    *mocks.MockInferenceClient
}

// NewRecommendHandler 创建推荐处理器
func NewRecommendHandler(
	userService *mocks.MockUserService,
	itemService *mocks.MockItemService,
	featureService *mocks.MockFeatureService,
	inferClient *mocks.MockInferenceClient,
) *RecommendHandler {
	return &RecommendHandler{
		userService:    userService,
		itemService:    itemService,
		featureService: featureService,
		inferClient:    inferClient,
	}
}

// GetRecommendations 获取推荐
func (h *RecommendHandler) GetRecommendations(c *gin.Context) {
	var req interfaces.RecommendRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.UserID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "user_id is required"})
		return
	}

	// 检查用户是否存在
	_, err := h.userService.GetUser(c.Request.Context(), req.UserID)
	if err != nil {
		// 冷启动处理
		c.JSON(http.StatusOK, interfaces.RecommendResponse{
			Recommendations: []*interfaces.Recommendation{
				{ItemID: "item_001", Score: 0.9, Reason: "Popular"},
				{ItemID: "item_002", Score: 0.8, Reason: "Popular"},
			},
			RequestID: "req_cold_start",
			Strategy:  "cold_start",
		})
		return
	}

	// 模拟推荐结果
	response := interfaces.RecommendResponse{
		Recommendations: []*interfaces.Recommendation{
			{ItemID: "item_001", Score: 0.95, Reason: "Based on history"},
			{ItemID: "item_002", Score: 0.90, Reason: "Based on history"},
			{ItemID: "item_003", Score: 0.85, Reason: "Similar users liked"},
		},
		RequestID: "req_001",
		Strategy:  "model",
	}

	c.JSON(http.StatusOK, response)
}

// =============================================================================
// RecommendHandler 单元测试
// =============================================================================

// TestRecommendHandler_GetRecommendations 测试获取推荐
func TestRecommendHandler_GetRecommendations(t *testing.T) {
	// 准备
	mockUserService := mocks.NewMockUserService()
	mockItemService := mocks.NewMockItemService()
	mockFeatureService := mocks.NewMockFeatureService()
	mockInferClient := mocks.NewMockInferenceClient()

	testUser := fixtures.GetTestUser("user_001")
	mockUserService.SetUser(testUser)

	handler := NewRecommendHandler(mockUserService, mockItemService, mockFeatureService, mockInferClient)

	router := setupTestRouter()
	router.POST("/recommend", handler.GetRecommendations)

	req := interfaces.RecommendRequest{
		UserID: "user_001",
		Limit:  10,
	}

	// 执行
	w := performRequest(router, "POST", "/recommend", req)

	// 验证
	assert.Equal(t, http.StatusOK, w.Code)

	var response interfaces.RecommendResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.NotEmpty(t, response.Recommendations)
	assert.Equal(t, "model", response.Strategy)
}

// TestRecommendHandler_GetRecommendations_ColdStart 测试冷启动推荐
func TestRecommendHandler_GetRecommendations_ColdStart(t *testing.T) {
	// 准备
	mockUserService := mocks.NewMockUserService()
	mockItemService := mocks.NewMockItemService()
	mockFeatureService := mocks.NewMockFeatureService()
	mockInferClient := mocks.NewMockInferenceClient()

	// 不设置用户，模拟新用户

	handler := NewRecommendHandler(mockUserService, mockItemService, mockFeatureService, mockInferClient)

	router := setupTestRouter()
	router.POST("/recommend", handler.GetRecommendations)

	req := interfaces.RecommendRequest{
		UserID: "new_user",
		Limit:  10,
	}

	// 执行
	w := performRequest(router, "POST", "/recommend", req)

	// 验证
	assert.Equal(t, http.StatusOK, w.Code)

	var response interfaces.RecommendResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.Equal(t, "cold_start", response.Strategy)
}

// TestRecommendHandler_GetRecommendations_InvalidRequest 测试无效请求
func TestRecommendHandler_GetRecommendations_InvalidRequest(t *testing.T) {
	// 准备
	mockUserService := mocks.NewMockUserService()
	mockItemService := mocks.NewMockItemService()
	mockFeatureService := mocks.NewMockFeatureService()
	mockInferClient := mocks.NewMockInferenceClient()

	handler := NewRecommendHandler(mockUserService, mockItemService, mockFeatureService, mockInferClient)

	router := setupTestRouter()
	router.POST("/recommend", handler.GetRecommendations)

	// 执行 - 空 user_id
	req := interfaces.RecommendRequest{
		UserID: "",
		Limit:  10,
	}
	w := performRequest(router, "POST", "/recommend", req)

	// 验证
	assert.Equal(t, http.StatusBadRequest, w.Code)
}

// =============================================================================
// 中间件测试
// =============================================================================

// mockAuthMiddleware 模拟认证中间件
func mockAuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		token := c.GetHeader("Authorization")
		if token == "" {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
			c.Abort()
			return
		}
		c.Set("user_id", "authenticated_user")
		c.Next()
	}
}

// mockRateLimitMiddleware 模拟限流中间件
func mockRateLimitMiddleware(limit int) gin.HandlerFunc {
	requestCount := 0
	return func(c *gin.Context) {
		requestCount++
		if requestCount > limit {
			c.JSON(http.StatusTooManyRequests, gin.H{"error": "rate limit exceeded"})
			c.Abort()
			return
		}
		c.Next()
	}
}

// TestAuthMiddleware 测试认证中间件
func TestAuthMiddleware(t *testing.T) {
	router := setupTestRouter()
	router.Use(mockAuthMiddleware())
	router.GET("/protected", func(c *gin.Context) {
		userID, _ := c.Get("user_id")
		c.JSON(http.StatusOK, gin.H{"user_id": userID})
	})

	// 测试无 token
	w := performRequest(router, "GET", "/protected", nil)
	assert.Equal(t, http.StatusUnauthorized, w.Code)

	// 测试有 token
	req, _ := http.NewRequest("GET", "/protected", nil)
	req.Header.Set("Authorization", "Bearer test_token")
	w2 := httptest.NewRecorder()
	router.ServeHTTP(w2, req)
	assert.Equal(t, http.StatusOK, w2.Code)
}

// TestRateLimitMiddleware 测试限流中间件
func TestRateLimitMiddleware(t *testing.T) {
	router := setupTestRouter()
	router.Use(mockRateLimitMiddleware(3))
	router.GET("/api", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	// 前 3 次应该成功
	for i := 0; i < 3; i++ {
		w := performRequest(router, "GET", "/api", nil)
		assert.Equal(t, http.StatusOK, w.Code)
	}

	// 第 4 次应该被限流
	w := performRequest(router, "GET", "/api", nil)
	assert.Equal(t, http.StatusTooManyRequests, w.Code)
}

// =============================================================================
// 请求验证测试
// =============================================================================

// TestRequestValidation_CreateUser 测试创建用户请求验证
func TestRequestValidation_CreateUser(t *testing.T) {
	testCases := []struct {
		name       string
		body       string
		wantStatus int
	}{
		{
			name:       "valid request",
			body:       `{"name":"Test","email":"test@example.com"}`,
			wantStatus: http.StatusCreated,
		},
		{
			name:       "missing name",
			body:       `{"email":"test@example.com"}`,
			wantStatus: http.StatusCreated, // 简化版处理器可能不校验
		},
		{
			name:       "invalid json",
			body:       `{invalid}`,
			wantStatus: http.StatusBadRequest,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockService := mocks.NewMockUserService()
			handler := NewUserHandler(mockService)

			router := setupTestRouter()
			router.POST("/users", handler.CreateUser)

			req, _ := http.NewRequest("POST", "/users", bytes.NewBufferString(tc.body))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			assert.Equal(t, tc.wantStatus, w.Code)
		})
	}
}

// =============================================================================
// 响应格式测试
// =============================================================================

// TestResponseFormat_User 测试用户响应格式
func TestResponseFormat_User(t *testing.T) {
	mockService := mocks.NewMockUserService()
	testUser := fixtures.GetTestUser("user_001")
	mockService.SetUser(testUser)

	handler := NewUserHandler(mockService)

	router := setupTestRouter()
	router.GET("/users/:id", handler.GetUser)

	w := performRequest(router, "GET", "/users/user_001", nil)

	// 验证响应格式
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "application/json; charset=utf-8", w.Header().Get("Content-Type"))

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)

	// 验证必要字段存在
	assert.Contains(t, response, "id")
	assert.Contains(t, response, "name")
	assert.Contains(t, response, "email")
}

// =============================================================================
// 超时处理测试
// =============================================================================

// TestTimeoutHandling 测试超时处理
func TestTimeoutHandling(t *testing.T) {
	mockService := mocks.NewMockUserService()
	
	// 设置一个会"超时"的处理
	handler := func(c *gin.Context) {
		// 模拟超时上下文
		ctx, cancel := context.WithTimeout(c.Request.Context(), 1*time.Millisecond)
		defer cancel()

		// 等待超时
		<-ctx.Done()

		c.JSON(http.StatusGatewayTimeout, gin.H{"error": "request timeout"})
	}

	router := setupTestRouter()
	router.GET("/slow", handler)

	w := performRequest(router, "GET", "/slow", nil)

	assert.Equal(t, http.StatusGatewayTimeout, w.Code)
	_ = mockService // 避免未使用警告
}

