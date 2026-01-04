// Package v1 提供物品 API v1 版本
//
// 物品 API 提供 RESTful 接口用于管理推荐系统中的物品（商品、电影、文章、视频等）。
// 所有接口均使用 JSON 格式进行数据交换。
//
// API 接口清单：
//   - POST   /api/v1/items        创建物品
//   - GET    /api/v1/items        列出物品
//   - GET    /api/v1/items/search 搜索物品
//   - GET    /api/v1/items/:id    获取物品
//   - PUT    /api/v1/items/:id    更新物品
//   - DELETE /api/v1/items/:id    删除物品
//   - GET    /api/v1/items/:id/similar 获取相似物品
//   - GET    /api/v1/items/:id/stats   获取物品统计
//   - POST   /api/v1/items/batch  批量获取物品
package v1

import (
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"

	"recommend-system/internal/interfaces"
	"recommend-system/internal/service/item"
	"recommend-system/pkg/logger"
	"recommend-system/pkg/utils"

	"go.uber.org/zap"
)

// =============================================================================
// Handler 定义
// =============================================================================

// Handler 物品 API 处理器
type Handler struct {
	itemService *item.Service
}

// NewHandler 创建物品 API 处理器
//
// 参数：
//   - itemService: 物品服务实例
//
// 返回：
//   - *Handler: 物品 API 处理器实例
func NewHandler(itemService *item.Service) *Handler {
	return &Handler{
		itemService: itemService,
	}
}

// RegisterRoutes 注册路由
//
// 将物品 API 路由注册到 Gin 路由组。
//
// 参数：
//   - r: Gin 路由组（通常为 /api/v1）
func (h *Handler) RegisterRoutes(r *gin.RouterGroup) {
	items := r.Group("/items")
	{
		// 创建物品
		items.POST("", h.CreateItem)
		// 列出物品
		items.GET("", h.ListItems)
		// 搜索物品
		items.GET("/search", h.SearchItems)
		// 批量获取物品
		items.POST("/batch", h.BatchGetItems)
		// 获取单个物品
		items.GET("/:id", h.GetItem)
		// 更新物品
		items.PUT("/:id", h.UpdateItem)
		// 删除物品
		items.DELETE("/:id", h.DeleteItem)
		// 获取相似物品
		items.GET("/:id/similar", h.GetSimilarItems)
		// 获取物品统计
		items.GET("/:id/stats", h.GetItemStats)
		// 更新物品统计
		items.POST("/:id/stats", h.UpdateItemStats)
	}
}

// =============================================================================
// API 处理方法
// =============================================================================

// CreateItem 创建物品
//
// @Summary      创建物品
// @Description  创建新的物品记录，支持设置物品类型、标题、描述、类目、标签等信息
// @Tags         物品管理
// @Accept       json
// @Produce      json
// @Param        request body CreateItemRequest true "创建物品请求"
// @Success      201 {object} Response{data=interfaces.Item} "创建成功"
// @Failure      400 {object} Response "请求参数错误"
// @Failure      500 {object} Response "服务器内部错误"
// @Router       /api/v1/items [post]
func (h *Handler) CreateItem(c *gin.Context) {
	timer := utils.NewTimer()
	ctx := c.Request.Context()

	var req CreateItemRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "invalid request body",
			Error:   err.Error(),
		})
		return
	}

	// 转换为接口请求结构
	createReq := &interfaces.CreateItemRequest{
		Type:        req.Type,
		Title:       req.Title,
		Description: req.Description,
		Category:    req.Category,
		Tags:        req.Tags,
		Metadata:    req.Metadata,
		Embedding:   req.Embedding,
	}

	result, err := h.itemService.CreateItem(ctx, createReq)
	if err != nil {
		logger.Error("failed to create item",
			zap.Error(err),
			zap.Int64("latency_ms", timer.Elapsed()),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    500,
			Message: "failed to create item",
			Error:   err.Error(),
		})
		return
	}

	logger.Info("item created via API",
		zap.String("item_id", result.ID),
		zap.String("type", result.Type),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	c.JSON(http.StatusCreated, Response{
		Code:    0,
		Message: "success",
		Data:    result,
	})
}

// GetItem 获取物品
//
// @Summary      获取物品
// @Description  根据物品 ID 获取物品详细信息
// @Tags         物品管理
// @Accept       json
// @Produce      json
// @Param        id path string true "物品 ID"
// @Success      200 {object} Response{data=interfaces.Item} "获取成功"
// @Failure      404 {object} Response "物品不存在"
// @Failure      500 {object} Response "服务器内部错误"
// @Router       /api/v1/items/{id} [get]
func (h *Handler) GetItem(c *gin.Context) {
	timer := utils.NewTimer()
	ctx := c.Request.Context()
	itemID := c.Param("id")

	if itemID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "item id is required",
		})
		return
	}

	result, err := h.itemService.GetItem(ctx, itemID)
	if err != nil {
		if err == item.ErrItemNotFound {
			c.JSON(http.StatusNotFound, Response{
				Code:    404,
				Message: "item not found",
			})
			return
		}

		logger.Error("failed to get item",
			zap.String("item_id", itemID),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    500,
			Message: "failed to get item",
			Error:   err.Error(),
		})
		return
	}

	logger.Debug("item fetched via API",
		zap.String("item_id", itemID),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	c.JSON(http.StatusOK, Response{
		Code:    0,
		Message: "success",
		Data:    result,
	})
}

// ListItems 列出物品
//
// @Summary      列出物品
// @Description  分页获取物品列表，支持按类型和类目筛选
// @Tags         物品管理
// @Accept       json
// @Produce      json
// @Param        type query string false "物品类型（movie, product, article, video）"
// @Param        category query string false "物品类目"
// @Param        page query int false "页码，默认 1"
// @Param        page_size query int false "每页数量，默认 20，最大 100"
// @Success      200 {object} Response{data=ListItemsResponseData} "获取成功"
// @Failure      400 {object} Response "请求参数错误"
// @Failure      500 {object} Response "服务器内部错误"
// @Router       /api/v1/items [get]
func (h *Handler) ListItems(c *gin.Context) {
	timer := utils.NewTimer()
	ctx := c.Request.Context()

	// 解析查询参数
	var req ListItemsQuery
	if err := c.ShouldBindQuery(&req); err != nil {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "invalid query parameters",
			Error:   err.Error(),
		})
		return
	}

	// 转换为接口请求结构
	listReq := &interfaces.ListItemsRequest{
		Type:     req.Type,
		Category: req.Category,
		Page:     req.Page,
		PageSize: req.PageSize,
	}

	result, err := h.itemService.ListItems(ctx, listReq)
	if err != nil {
		logger.Error("failed to list items",
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    500,
			Message: "failed to list items",
			Error:   err.Error(),
		})
		return
	}

	logger.Debug("items listed via API",
		zap.Int("count", len(result.Items)),
		zap.Int64("total", result.Total),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	c.JSON(http.StatusOK, Response{
		Code:    0,
		Message: "success",
		Data:    result,
	})
}

// SearchItems 搜索物品
//
// @Summary      搜索物品
// @Description  根据关键词搜索物品
// @Tags         物品管理
// @Accept       json
// @Produce      json
// @Param        q query string true "搜索关键词"
// @Param        limit query int false "返回数量限制，默认 10，最大 100"
// @Success      200 {object} Response{data=[]interfaces.Item} "搜索成功"
// @Failure      400 {object} Response "请求参数错误"
// @Failure      500 {object} Response "服务器内部错误"
// @Router       /api/v1/items/search [get]
func (h *Handler) SearchItems(c *gin.Context) {
	timer := utils.NewTimer()
	ctx := c.Request.Context()

	query := c.Query("q")
	if query == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "search query is required",
		})
		return
	}

	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "10"))
	if limit <= 0 {
		limit = 10
	}
	if limit > 100 {
		limit = 100
	}

	results, err := h.itemService.SearchItems(ctx, query, limit)
	if err != nil {
		logger.Error("failed to search items",
			zap.String("query", query),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    500,
			Message: "failed to search items",
			Error:   err.Error(),
		})
		return
	}

	logger.Debug("items searched via API",
		zap.String("query", query),
		zap.Int("count", len(results)),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	c.JSON(http.StatusOK, Response{
		Code:    0,
		Message: "success",
		Data:    results,
	})
}

// UpdateItem 更新物品
//
// @Summary      更新物品
// @Description  根据物品 ID 更新物品信息
// @Tags         物品管理
// @Accept       json
// @Produce      json
// @Param        id path string true "物品 ID"
// @Param        request body UpdateItemRequest true "更新物品请求"
// @Success      200 {object} Response{data=interfaces.Item} "更新成功"
// @Failure      400 {object} Response "请求参数错误"
// @Failure      404 {object} Response "物品不存在"
// @Failure      500 {object} Response "服务器内部错误"
// @Router       /api/v1/items/{id} [put]
func (h *Handler) UpdateItem(c *gin.Context) {
	timer := utils.NewTimer()
	ctx := c.Request.Context()
	itemID := c.Param("id")

	if itemID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "item id is required",
		})
		return
	}

	var req UpdateItemRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "invalid request body",
			Error:   err.Error(),
		})
		return
	}

	// 转换为接口请求结构
	updateReq := &interfaces.UpdateItemRequest{
		Title:       req.Title,
		Description: req.Description,
		Category:    req.Category,
		Tags:        req.Tags,
		Metadata:    req.Metadata,
	}

	result, err := h.itemService.UpdateItem(ctx, itemID, updateReq)
	if err != nil {
		if err == item.ErrItemNotFound {
			c.JSON(http.StatusNotFound, Response{
				Code:    404,
				Message: "item not found",
			})
			return
		}

		logger.Error("failed to update item",
			zap.String("item_id", itemID),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    500,
			Message: "failed to update item",
			Error:   err.Error(),
		})
		return
	}

	logger.Info("item updated via API",
		zap.String("item_id", itemID),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	c.JSON(http.StatusOK, Response{
		Code:    0,
		Message: "success",
		Data:    result,
	})
}

// DeleteItem 删除物品
//
// @Summary      删除物品
// @Description  根据物品 ID 删除物品（软删除）
// @Tags         物品管理
// @Accept       json
// @Produce      json
// @Param        id path string true "物品 ID"
// @Success      200 {object} Response "删除成功"
// @Failure      400 {object} Response "请求参数错误"
// @Failure      404 {object} Response "物品不存在"
// @Failure      500 {object} Response "服务器内部错误"
// @Router       /api/v1/items/{id} [delete]
func (h *Handler) DeleteItem(c *gin.Context) {
	timer := utils.NewTimer()
	ctx := c.Request.Context()
	itemID := c.Param("id")

	if itemID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "item id is required",
		})
		return
	}

	err := h.itemService.DeleteItem(ctx, itemID)
	if err != nil {
		if err == item.ErrItemNotFound {
			c.JSON(http.StatusNotFound, Response{
				Code:    404,
				Message: "item not found",
			})
			return
		}

		logger.Error("failed to delete item",
			zap.String("item_id", itemID),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    500,
			Message: "failed to delete item",
			Error:   err.Error(),
		})
		return
	}

	logger.Info("item deleted via API",
		zap.String("item_id", itemID),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	c.JSON(http.StatusOK, Response{
		Code:    0,
		Message: "item deleted successfully",
	})
}

// GetSimilarItems 获取相似物品
//
// @Summary      获取相似物品
// @Description  基于向量搜索获取与指定物品相似的物品列表
// @Tags         物品管理
// @Accept       json
// @Produce      json
// @Param        id path string true "物品 ID"
// @Param        top_k query int false "返回数量，默认 10"
// @Success      200 {object} Response{data=[]interfaces.SimilarItem} "获取成功"
// @Failure      400 {object} Response "请求参数错误"
// @Failure      404 {object} Response "物品不存在"
// @Failure      500 {object} Response "服务器内部错误"
// @Router       /api/v1/items/{id}/similar [get]
func (h *Handler) GetSimilarItems(c *gin.Context) {
	timer := utils.NewTimer()
	ctx := c.Request.Context()
	itemID := c.Param("id")

	if itemID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "item id is required",
		})
		return
	}

	topK, _ := strconv.Atoi(c.DefaultQuery("top_k", "10"))
	if topK <= 0 {
		topK = 10
	}
	if topK > 100 {
		topK = 100
	}

	results, err := h.itemService.GetSimilarItems(ctx, itemID, topK)
	if err != nil {
		if err == item.ErrItemNotFound || err == item.ErrEmbeddingNotFound {
			c.JSON(http.StatusNotFound, Response{
				Code:    404,
				Message: "item or embedding not found",
			})
			return
		}

		if err == item.ErrMilvusNotAvailable {
			c.JSON(http.StatusServiceUnavailable, Response{
				Code:    503,
				Message: "vector search service not available",
			})
			return
		}

		logger.Error("failed to get similar items",
			zap.String("item_id", itemID),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    500,
			Message: "failed to get similar items",
			Error:   err.Error(),
		})
		return
	}

	logger.Debug("similar items fetched via API",
		zap.String("item_id", itemID),
		zap.Int("count", len(results)),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	c.JSON(http.StatusOK, Response{
		Code:    0,
		Message: "success",
		Data:    results,
	})
}

// GetItemStats 获取物品统计
//
// @Summary      获取物品统计
// @Description  获取物品的浏览量、点击量、点赞量等统计信息
// @Tags         物品管理
// @Accept       json
// @Produce      json
// @Param        id path string true "物品 ID"
// @Success      200 {object} Response{data=interfaces.ItemStats} "获取成功"
// @Failure      400 {object} Response "请求参数错误"
// @Failure      404 {object} Response "物品不存在"
// @Failure      500 {object} Response "服务器内部错误"
// @Router       /api/v1/items/{id}/stats [get]
func (h *Handler) GetItemStats(c *gin.Context) {
	timer := utils.NewTimer()
	ctx := c.Request.Context()
	itemID := c.Param("id")

	if itemID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "item id is required",
		})
		return
	}

	stats, err := h.itemService.GetItemStats(ctx, itemID)
	if err != nil {
		logger.Error("failed to get item stats",
			zap.String("item_id", itemID),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    500,
			Message: "failed to get item stats",
			Error:   err.Error(),
		})
		return
	}

	logger.Debug("item stats fetched via API",
		zap.String("item_id", itemID),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	c.JSON(http.StatusOK, Response{
		Code:    0,
		Message: "success",
		Data:    stats,
	})
}

// UpdateItemStats 更新物品统计
//
// @Summary      更新物品统计
// @Description  增加物品的统计计数（浏览、点击、点赞、分享）
// @Tags         物品管理
// @Accept       json
// @Produce      json
// @Param        id path string true "物品 ID"
// @Param        request body UpdateStatsRequest true "更新统计请求"
// @Success      200 {object} Response "更新成功"
// @Failure      400 {object} Response "请求参数错误"
// @Failure      500 {object} Response "服务器内部错误"
// @Router       /api/v1/items/{id}/stats [post]
func (h *Handler) UpdateItemStats(c *gin.Context) {
	timer := utils.NewTimer()
	ctx := c.Request.Context()
	itemID := c.Param("id")

	if itemID == "" {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "item id is required",
		})
		return
	}

	var req UpdateStatsRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "invalid request body",
			Error:   err.Error(),
		})
		return
	}

	err := h.itemService.UpdateItemStats(ctx, itemID, req.Action)
	if err != nil {
		logger.Error("failed to update item stats",
			zap.String("item_id", itemID),
			zap.String("action", req.Action),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    500,
			Message: "failed to update item stats",
			Error:   err.Error(),
		})
		return
	}

	logger.Debug("item stats updated via API",
		zap.String("item_id", itemID),
		zap.String("action", req.Action),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	c.JSON(http.StatusOK, Response{
		Code:    0,
		Message: "stats updated successfully",
	})
}

// BatchGetItems 批量获取物品
//
// @Summary      批量获取物品
// @Description  根据物品 ID 列表批量获取物品信息
// @Tags         物品管理
// @Accept       json
// @Produce      json
// @Param        request body BatchGetItemsRequest true "批量获取请求"
// @Success      200 {object} Response{data=[]interfaces.Item} "获取成功"
// @Failure      400 {object} Response "请求参数错误"
// @Failure      500 {object} Response "服务器内部错误"
// @Router       /api/v1/items/batch [post]
func (h *Handler) BatchGetItems(c *gin.Context) {
	timer := utils.NewTimer()
	ctx := c.Request.Context()

	var req BatchGetItemsRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "invalid request body",
			Error:   err.Error(),
		})
		return
	}

	if len(req.IDs) == 0 {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "item ids are required",
		})
		return
	}

	// 限制批量请求大小
	if len(req.IDs) > 100 {
		c.JSON(http.StatusBadRequest, Response{
			Code:    400,
			Message: "too many item ids, max 100",
		})
		return
	}

	items, err := h.itemService.BatchGetItems(ctx, req.IDs)
	if err != nil {
		logger.Error("failed to batch get items",
			zap.Int("count", len(req.IDs)),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, Response{
			Code:    500,
			Message: "failed to batch get items",
			Error:   err.Error(),
		})
		return
	}

	logger.Debug("items batch fetched via API",
		zap.Int("requested", len(req.IDs)),
		zap.Int("returned", len(items)),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	c.JSON(http.StatusOK, Response{
		Code:    0,
		Message: "success",
		Data:    items,
	})
}

// =============================================================================
// 请求/响应结构体
// =============================================================================

// Response 通用响应结构
type Response struct {
	Code    int         `json:"code"`              // 状态码，0 表示成功
	Message string      `json:"message"`           // 提示信息
	Data    interface{} `json:"data,omitempty"`    // 响应数据
	Error   string      `json:"error,omitempty"`   // 错误详情
}

// CreateItemRequest 创建物品请求
type CreateItemRequest struct {
	Type        string                 `json:"type" binding:"required"`  // 物品类型: movie, product, article, video
	Title       string                 `json:"title" binding:"required"` // 物品标题
	Description string                 `json:"description"`              // 物品描述
	Category    string                 `json:"category"`                 // 物品类目
	Tags        []string               `json:"tags"`                     // 物品标签
	Metadata    map[string]interface{} `json:"metadata"`                 // 元数据
	Embedding   []float32              `json:"embedding"`                // 物品向量（可选）
}

// UpdateItemRequest 更新物品请求
type UpdateItemRequest struct {
	Title       string                 `json:"title"`       // 物品标题
	Description string                 `json:"description"` // 物品描述
	Category    string                 `json:"category"`    // 物品类目
	Tags        []string               `json:"tags"`        // 物品标签
	Metadata    map[string]interface{} `json:"metadata"`    // 元数据
}

// ListItemsQuery 列出物品查询参数
type ListItemsQuery struct {
	Type     string `form:"type"`               // 物品类型
	Category string `form:"category"`           // 物品类目
	Page     int    `form:"page,default=1"`     // 页码
	PageSize int    `form:"page_size,default=20"` // 每页数量
}

// ListItemsResponseData 列出物品响应数据
type ListItemsResponseData struct {
	Items []*interfaces.Item `json:"items"` // 物品列表
	Total int64              `json:"total"` // 总数
	Page  int                `json:"page"`  // 当前页码
}

// BatchGetItemsRequest 批量获取物品请求
type BatchGetItemsRequest struct {
	IDs []string `json:"ids" binding:"required"` // 物品 ID 列表
}

// UpdateStatsRequest 更新统计请求
type UpdateStatsRequest struct {
	Action string `json:"action" binding:"required"` // 行为类型: view, click, like, share
}

