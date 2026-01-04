// Package v1 提供用户服务的 HTTP API 处理器
//
// 本模块实现 RESTful API 接口，处理用户相关的所有 HTTP 请求。
// 使用 Gin 框架处理路由和请求/响应。
//
// API 端点:
//   - POST   /api/v1/users           - 创建用户
//   - GET    /api/v1/users/:id       - 获取用户信息
//   - PUT    /api/v1/users/:id       - 更新用户信息
//   - DELETE /api/v1/users/:id       - 删除用户
//   - GET    /api/v1/users/:id/behaviors - 获取用户行为历史
//   - POST   /api/v1/users/:id/behaviors - 记录用户行为
//   - GET    /api/v1/users/:id/profile   - 获取用户画像
package v1

import (
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"recommend-system/internal/interfaces"
	"recommend-system/pkg/logger"
)

// =============================================================================
// 常量定义
// =============================================================================

const (
	// 默认分页参数
	defaultLimit = 20
	maxLimit     = 100

	// 响应状态码
	codeSuccess      = 0
	codeInvalidParam = 400
	codeNotFound     = 404
	codeServerError  = 500
)

// =============================================================================
// Handler 定义
// =============================================================================

// Handler 用户 API 处理器
//
// 封装用户服务，提供 HTTP API 接口
type Handler struct {
	userService interfaces.UserService // 用户服务接口
	logger      *zap.Logger            // 日志记录器
}

// NewHandler 创建用户 API 处理器
//
// 参数:
//   - userService: 用户服务接口实现
//   - log: 可选的日志记录器
//
// 返回:
//   - *Handler: API 处理器实例
func NewHandler(userService interfaces.UserService, log *zap.Logger) *Handler {
	if log == nil {
		log = logger.Logger
	}
	if log == nil {
		log = zap.NewNop()
	}

	return &Handler{
		userService: userService,
		logger:      log.Named("user-handler"),
	}
}

// =============================================================================
// 路由注册
// =============================================================================

// RegisterRoutes 注册用户相关的所有路由
//
// 将所有用户 API 端点注册到指定的路由组
//
// 参数:
//   - r: Gin 路由组，通常是 /api/v1
func (h *Handler) RegisterRoutes(r *gin.RouterGroup) {
	users := r.Group("/users")
	{
		// 用户 CRUD
		users.POST("", h.CreateUser)
		users.GET("/:id", h.GetUser)
		users.PUT("/:id", h.UpdateUser)
		users.DELETE("/:id", h.DeleteUser)

		// 用户行为
		users.GET("/:id/behaviors", h.GetUserBehaviors)
		users.POST("/:id/behaviors", h.RecordBehavior)

		// 用户画像
		users.GET("/:id/profile", h.GetUserProfile)
	}
}

// =============================================================================
// 请求/响应结构体
// =============================================================================

// Response 统一响应结构
type Response struct {
	Code    int         `json:"code"`              // 状态码，0 表示成功
	Message string      `json:"message,omitempty"` // 消息
	Data    interface{} `json:"data,omitempty"`    // 数据
}

// CreateUserRequest 创建用户请求
type CreateUserRequest struct {
	Name   string `json:"name" binding:"required"`       // 用户名，必填
	Email  string `json:"email" binding:"required,email"` // 邮箱，必填
	Age    int    `json:"age"`                           // 年龄
	Gender string `json:"gender"`                        // 性别
}

// UpdateUserRequest 更新用户请求
type UpdateUserRequest struct {
	Name   string `json:"name"`   // 用户名
	Email  string `json:"email"`  // 邮箱
	Age    int    `json:"age"`    // 年龄
	Gender string `json:"gender"` // 性别
}

// RecordBehaviorRequest 记录行为请求
type RecordBehaviorRequest struct {
	ItemID  string            `json:"item_id" binding:"required"` // 物品ID，必填
	Action  string            `json:"action" binding:"required"`  // 行为类型，必填
	Context map[string]string `json:"context"`                    // 上下文信息
}

// PaginationQuery 分页查询参数
type PaginationQuery struct {
	Page  int `form:"page,default=1"`       // 页码
	Limit int `form:"limit,default=20"`     // 每页数量
}

// =============================================================================
// API 处理方法
// =============================================================================

// CreateUser 创建用户
//
// @Summary 创建新用户
// @Description 创建一个新的用户账号
// @Tags 用户管理
// @Accept json
// @Produce json
// @Param request body CreateUserRequest true "创建用户请求"
// @Success 201 {object} Response{data=interfaces.User}
// @Failure 400 {object} Response
// @Failure 500 {object} Response
// @Router /api/v1/users [post]
func (h *Handler) CreateUser(c *gin.Context) {
	var req CreateUserRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		h.logger.Warn("invalid create user request",
			zap.Error(err),
		)
		c.JSON(http.StatusBadRequest, Response{
			Code:    codeInvalidParam,
			Message: "invalid request: " + err.Error(),
		})
		return
	}

	// 转换为接口请求
	createReq := &interfaces.CreateUserRequest{
		Name:   req.Name,
		Email:  req.Email,
		Age:    req.Age,
		Gender: req.Gender,
	}

	// 调用服务创建用户
	user, err := h.userService.CreateUser(c.Request.Context(), createReq)
	if err != nil {
		h.logger.Error("failed to create user",
			zap.String("email", req.Email),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    codeServerError,
			Message: "failed to create user: " + err.Error(),
		})
		return
	}

	h.logger.Info("user created",
		zap.String("user_id", user.ID),
		zap.String("email", user.Email),
	)

	c.JSON(http.StatusCreated, Response{
		Code:    codeSuccess,
		Message: "user created successfully",
		Data:    user,
	})
}

// GetUser 获取用户信息
//
// @Summary 获取用户信息
// @Description 根据用户ID获取用户详细信息
// @Tags 用户管理
// @Accept json
// @Produce json
// @Param id path string true "用户ID"
// @Success 200 {object} Response{data=interfaces.User}
// @Failure 404 {object} Response
// @Failure 500 {object} Response
// @Router /api/v1/users/{id} [get]
func (h *Handler) GetUser(c *gin.Context) {
	userID := c.Param("id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    codeInvalidParam,
			Message: "user id is required",
		})
		return
	}

	user, err := h.userService.GetUser(c.Request.Context(), userID)
	if err != nil {
		h.logger.Warn("user not found",
			zap.String("user_id", userID),
			zap.Error(err),
		)
		c.JSON(http.StatusNotFound, Response{
			Code:    codeNotFound,
			Message: "user not found",
		})
		return
	}

	c.JSON(http.StatusOK, Response{
		Code: codeSuccess,
		Data: user,
	})
}

// UpdateUser 更新用户信息
//
// @Summary 更新用户信息
// @Description 更新指定用户的信息
// @Tags 用户管理
// @Accept json
// @Produce json
// @Param id path string true "用户ID"
// @Param request body UpdateUserRequest true "更新用户请求"
// @Success 200 {object} Response{data=interfaces.User}
// @Failure 400 {object} Response
// @Failure 404 {object} Response
// @Failure 500 {object} Response
// @Router /api/v1/users/{id} [put]
func (h *Handler) UpdateUser(c *gin.Context) {
	userID := c.Param("id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    codeInvalidParam,
			Message: "user id is required",
		})
		return
	}

	var req UpdateUserRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, Response{
			Code:    codeInvalidParam,
			Message: "invalid request: " + err.Error(),
		})
		return
	}

	// 转换为接口请求
	updateReq := &interfaces.UpdateUserRequest{
		Name:   req.Name,
		Email:  req.Email,
		Age:    req.Age,
		Gender: req.Gender,
	}

	user, err := h.userService.UpdateUser(c.Request.Context(), userID, updateReq)
	if err != nil {
		h.logger.Error("failed to update user",
			zap.String("user_id", userID),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    codeServerError,
			Message: "failed to update user: " + err.Error(),
		})
		return
	}

	h.logger.Info("user updated",
		zap.String("user_id", userID),
	)

	c.JSON(http.StatusOK, Response{
		Code:    codeSuccess,
		Message: "user updated successfully",
		Data:    user,
	})
}

// DeleteUser 删除用户
//
// @Summary 删除用户
// @Description 删除指定的用户账号
// @Tags 用户管理
// @Accept json
// @Produce json
// @Param id path string true "用户ID"
// @Success 200 {object} Response
// @Failure 500 {object} Response
// @Router /api/v1/users/{id} [delete]
func (h *Handler) DeleteUser(c *gin.Context) {
	userID := c.Param("id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    codeInvalidParam,
			Message: "user id is required",
		})
		return
	}

	if err := h.userService.DeleteUser(c.Request.Context(), userID); err != nil {
		h.logger.Error("failed to delete user",
			zap.String("user_id", userID),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    codeServerError,
			Message: "failed to delete user: " + err.Error(),
		})
		return
	}

	h.logger.Info("user deleted",
		zap.String("user_id", userID),
	)

	c.JSON(http.StatusOK, Response{
		Code:    codeSuccess,
		Message: "user deleted successfully",
	})
}

// GetUserBehaviors 获取用户行为历史
//
// @Summary 获取用户行为历史
// @Description 获取指定用户的行为记录列表
// @Tags 用户行为
// @Accept json
// @Produce json
// @Param id path string true "用户ID"
// @Param limit query int false "返回数量限制" default(100)
// @Success 200 {object} Response{data=[]interfaces.UserBehavior}
// @Failure 500 {object} Response
// @Router /api/v1/users/{id}/behaviors [get]
func (h *Handler) GetUserBehaviors(c *gin.Context) {
	userID := c.Param("id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    codeInvalidParam,
			Message: "user id is required",
		})
		return
	}

	// 获取限制参数
	limit := defaultLimit
	if limitStr := c.Query("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
			if limit > maxLimit {
				limit = maxLimit
			}
		}
	}

	behaviors, err := h.userService.GetUserBehaviors(c.Request.Context(), userID, limit)
	if err != nil {
		h.logger.Error("failed to get user behaviors",
			zap.String("user_id", userID),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    codeServerError,
			Message: "failed to get behaviors: " + err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, Response{
		Code: codeSuccess,
		Data: behaviors,
	})
}

// RecordBehavior 记录用户行为
//
// @Summary 记录用户行为
// @Description 记录用户与物品的交互行为
// @Tags 用户行为
// @Accept json
// @Produce json
// @Param id path string true "用户ID"
// @Param request body RecordBehaviorRequest true "行为记录请求"
// @Success 201 {object} Response
// @Failure 400 {object} Response
// @Failure 500 {object} Response
// @Router /api/v1/users/{id}/behaviors [post]
func (h *Handler) RecordBehavior(c *gin.Context) {
	userID := c.Param("id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    codeInvalidParam,
			Message: "user id is required",
		})
		return
	}

	var req RecordBehaviorRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, Response{
			Code:    codeInvalidParam,
			Message: "invalid request: " + err.Error(),
		})
		return
	}

	// 转换为接口请求
	behaviorReq := &interfaces.RecordBehaviorRequest{
		UserID:  userID,
		ItemID:  req.ItemID,
		Action:  req.Action,
		Context: req.Context,
	}

	if err := h.userService.RecordBehavior(c.Request.Context(), behaviorReq); err != nil {
		h.logger.Error("failed to record behavior",
			zap.String("user_id", userID),
			zap.String("item_id", req.ItemID),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    codeServerError,
			Message: "failed to record behavior: " + err.Error(),
		})
		return
	}

	h.logger.Debug("behavior recorded",
		zap.String("user_id", userID),
		zap.String("item_id", req.ItemID),
		zap.String("action", req.Action),
	)

	c.JSON(http.StatusCreated, Response{
		Code:    codeSuccess,
		Message: "behavior recorded successfully",
	})
}

// GetUserProfile 获取用户画像
//
// @Summary 获取用户画像
// @Description 获取用户的综合画像信息，包括偏好和行为统计
// @Tags 用户画像
// @Accept json
// @Produce json
// @Param id path string true "用户ID"
// @Success 200 {object} Response{data=interfaces.UserProfile}
// @Failure 404 {object} Response
// @Failure 500 {object} Response
// @Router /api/v1/users/{id}/profile [get]
func (h *Handler) GetUserProfile(c *gin.Context) {
	userID := c.Param("id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    codeInvalidParam,
			Message: "user id is required",
		})
		return
	}

	profile, err := h.userService.GetUserProfile(c.Request.Context(), userID)
	if err != nil {
		h.logger.Error("failed to get user profile",
			zap.String("user_id", userID),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    codeServerError,
			Message: "failed to get profile: " + err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, Response{
		Code: codeSuccess,
		Data: profile,
	})
}

// =============================================================================
// 辅助方法
// =============================================================================

// success 返回成功响应
func success(c *gin.Context, data interface{}) {
	c.JSON(http.StatusOK, Response{
		Code: codeSuccess,
		Data: data,
	})
}

// successWithMessage 返回带消息的成功响应
func successWithMessage(c *gin.Context, message string, data interface{}) {
	c.JSON(http.StatusOK, Response{
		Code:    codeSuccess,
		Message: message,
		Data:    data,
	})
}

// errorResponse 返回错误响应
func errorResponse(c *gin.Context, httpCode int, code int, message string) {
	c.JSON(httpCode, Response{
		Code:    code,
		Message: message,
	})
}

