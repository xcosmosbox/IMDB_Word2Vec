// Package v1 提供推荐 API v1 版本
package v1

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"recommend-system/internal/middleware"
	"recommend-system/internal/model"
	"recommend-system/internal/service/recommend"
	"recommend-system/pkg/logger"
	"recommend-system/pkg/utils"
	"go.uber.org/zap"
)

// Handler 推荐 API 处理器
type Handler struct {
	service *recommend.Service
}

// NewHandler 创建处理器
func NewHandler(service *recommend.Service) *Handler {
	return &Handler{service: service}
}

// RegisterRoutes 注册路由
func (h *Handler) RegisterRoutes(r *gin.RouterGroup) {
	r.POST("/recommend", h.Recommend)
	r.POST("/similar", h.Similar)
	r.POST("/feedback", h.Feedback)
}

// Recommend 获取推荐列表
// @Summary 获取推荐列表
// @Description 根据用户 ID 和上下文生成个性化推荐
// @Tags 推荐
// @Accept json
// @Produce json
// @Param request body model.RecommendRequest true "推荐请求"
// @Success 200 {object} Response{data=model.RecommendResponse}
// @Failure 400 {object} Response
// @Failure 500 {object} Response
// @Router /api/v1/recommend [post]
func (h *Handler) Recommend(c *gin.Context) {
	timer := utils.NewTimer()

	var req model.RecommendRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "invalid request",
			Error:   err.Error(),
		})
		return
	}

	// 如果请求中没有 user_id，尝试从上下文获取
	if req.UserID == "" {
		req.UserID = middleware.GetUserID(c)
	}

	if req.UserID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "user_id is required",
		})
		return
	}

	// 调用服务
	resp, err := h.service.Recommend(c.Request.Context(), &req)
	if err != nil {
		logger.Error("recommend failed",
			zap.String("user_id", req.UserID),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    500,
			Message: "internal server error",
			Error:   err.Error(),
		})
		return
	}

	// 设置追踪 ID
	if traceID, exists := c.Get("trace_id"); exists {
		resp.TraceID = traceID.(string)
	}

	logger.Info("recommend success",
		zap.String("user_id", req.UserID),
		zap.Int("count", len(resp.Items)),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	c.JSON(http.StatusOK, Response{
		Code:    0,
		Message: "success",
		Data:    resp,
	})
}

// Similar 获取相似推荐
// @Summary 获取相似推荐
// @Description 根据物品 ID 获取相似物品推荐
// @Tags 推荐
// @Accept json
// @Produce json
// @Param request body model.SimilarRequest true "相似推荐请求"
// @Success 200 {object} Response{data=model.SimilarResponse}
// @Failure 400 {object} Response
// @Failure 500 {object} Response
// @Router /api/v1/similar [post]
func (h *Handler) Similar(c *gin.Context) {
	var req model.SimilarRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "invalid request",
			Error:   err.Error(),
		})
		return
	}

	// 调用服务
	resp, err := h.service.Similar(c.Request.Context(), &req)
	if err != nil {
		logger.Error("similar failed",
			zap.String("item_id", req.ItemID),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    500,
			Message: "internal server error",
			Error:   err.Error(),
		})
		return
	}

	// 设置追踪 ID
	if traceID, exists := c.Get("trace_id"); exists {
		resp.TraceID = traceID.(string)
	}

	c.JSON(http.StatusOK, Response{
		Code:    0,
		Message: "success",
		Data:    resp,
	})
}

// Feedback 提交反馈
// @Summary 提交用户反馈
// @Description 记录用户对推荐结果的行为反馈
// @Tags 推荐
// @Accept json
// @Produce json
// @Param request body model.FeedbackRequest true "反馈请求"
// @Success 200 {object} Response{data=model.FeedbackResponse}
// @Failure 400 {object} Response
// @Failure 500 {object} Response
// @Router /api/v1/feedback [post]
func (h *Handler) Feedback(c *gin.Context) {
	var req model.FeedbackRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "invalid request",
			Error:   err.Error(),
		})
		return
	}

	// 如果请求中没有 user_id，尝试从上下文获取
	if req.UserID == "" {
		req.UserID = middleware.GetUserID(c)
	}

	if req.UserID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "user_id is required",
		})
		return
	}

	// TODO: 调用反馈服务保存行为
	eventID := utils.GenerateID("evt_")

	resp := &model.FeedbackResponse{
		Success: true,
		EventID: eventID,
	}

	logger.Info("feedback received",
		zap.String("user_id", req.UserID),
		zap.String("item_id", req.ItemID),
		zap.String("action", string(req.Action)),
		zap.String("event_id", eventID),
	)

	c.JSON(http.StatusOK, Response{
		Code:    0,
		Message: "success",
		Data:    resp,
	})
}

// Response 通用响应结构
type Response struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

