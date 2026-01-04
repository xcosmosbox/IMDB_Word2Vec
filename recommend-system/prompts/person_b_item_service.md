# Person B: Item Service（物品服务）

## 你的角色
你是一名 Go 后端工程师，负责实现生成式推荐系统的 **物品服务** 模块。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读接口定义文件：**

```
recommend-system/internal/interfaces/interfaces.go
```

你需要实现的接口：

```go
// ItemService 物品服务接口
type ItemService interface {
    GetItem(ctx context.Context, itemID string) (*Item, error)
    CreateItem(ctx context.Context, req *CreateItemRequest) (*Item, error)
    UpdateItem(ctx context.Context, itemID string, req *UpdateItemRequest) (*Item, error)
    DeleteItem(ctx context.Context, itemID string) error
    ListItems(ctx context.Context, req *ListItemsRequest) (*ListItemsResponse, error)
    SearchItems(ctx context.Context, query string, limit int) ([]*Item, error)
    GetSimilarItems(ctx context.Context, itemID string, topK int) ([]*SimilarItem, error)
    BatchGetItems(ctx context.Context, itemIDs []string) ([]*Item, error)
    GetItemStats(ctx context.Context, itemID string) (*ItemStats, error)
}
```

**注意事项：**
1. 所有数据结构从 `interfaces` 包导入
2. 依赖的 `ItemRepository` 和 `Cache` 接口也在 `interfaces` 包中定义
3. 确保方法签名与接口完全一致

---

## 背景知识

物品服务管理推荐系统中的所有可推荐物品（商品、电影、文章、视频等），负责：
- 物品信息的 CRUD 操作
- 物品特征管理
- 物品向量存储（Milvus）
- 物品统计信息

## 你的任务

实现以下模块：

```
recommend-system/
├── api/item/v1/
│   └── handler.go           # HTTP API 处理器
├── cmd/item-service/
│   └── main.go              # 服务入口
└── internal/service/item/
    └── service.go           # 业务逻辑
```

---

## 1. internal/service/item/service.go

```go
package item

import (
    "context"
    "time"
    
    "recommend-system/internal/model"
    "recommend-system/internal/repository"
    "recommend-system/internal/cache"
    "recommend-system/pkg/database"
    "recommend-system/pkg/logger"
)

// ItemService 物品服务
type ItemService struct {
    itemRepo    repository.ItemRepo
    milvusDB    *database.MilvusClient
    cache       cache.Cache
    logger      *logger.Logger
}

// NewItemService 创建物品服务
func NewItemService(
    itemRepo repository.ItemRepo,
    milvusDB *database.MilvusClient,
    cache cache.Cache,
    logger *logger.Logger,
) *ItemService {
    return &ItemService{
        itemRepo: itemRepo,
        milvusDB: milvusDB,
        cache:    cache,
        logger:   logger,
    }
}

// GetItem 获取物品信息
func (s *ItemService) GetItem(ctx context.Context, itemID string) (*model.Item, error) {
    // 1. 先查缓存
    cacheKey := "item:" + itemID
    var item model.Item
    if err := s.cache.Get(ctx, cacheKey, &item); err == nil {
        return &item, nil
    }
    
    // 2. 查数据库
    itemPtr, err := s.itemRepo.GetByID(ctx, itemID)
    if err != nil {
        return nil, err
    }
    
    // 3. 写入缓存
    _ = s.cache.Set(ctx, cacheKey, itemPtr, time.Hour)
    
    return itemPtr, nil
}

// CreateItem 创建物品
func (s *ItemService) CreateItem(ctx context.Context, req *CreateItemRequest) (*model.Item, error) {
    item := &model.Item{
        ID:          generateItemID(req.Type),
        Type:        req.Type,
        Title:       req.Title,
        Description: req.Description,
        Category:    req.Category,
        Tags:        req.Tags,
        Metadata:    req.Metadata,
        Status:      "active",
        CreatedAt:   time.Now(),
        UpdatedAt:   time.Now(),
    }
    
    // 保存到数据库
    if err := s.itemRepo.Create(ctx, item); err != nil {
        return nil, err
    }
    
    // 如果有向量，保存到 Milvus
    if len(req.Embedding) > 0 {
        if err := s.SaveItemEmbedding(ctx, item.ID, req.Embedding); err != nil {
            s.logger.Warn("Failed to save embedding", "item_id", item.ID, "error", err)
        }
    }
    
    return item, nil
}

// UpdateItem 更新物品
func (s *ItemService) UpdateItem(ctx context.Context, itemID string, req *UpdateItemRequest) (*model.Item, error) {
    item, err := s.itemRepo.GetByID(ctx, itemID)
    if err != nil {
        return nil, err
    }
    
    // 更新字段
    if req.Title != "" {
        item.Title = req.Title
    }
    if req.Description != "" {
        item.Description = req.Description
    }
    if len(req.Tags) > 0 {
        item.Tags = req.Tags
    }
    if req.Metadata != nil {
        item.Metadata = req.Metadata
    }
    item.UpdatedAt = time.Now()
    
    if err := s.itemRepo.Update(ctx, item); err != nil {
        return nil, err
    }
    
    // 清除缓存
    _ = s.cache.Delete(ctx, "item:"+itemID)
    
    return item, nil
}

// DeleteItem 删除物品
func (s *ItemService) DeleteItem(ctx context.Context, itemID string) error {
    // 软删除
    if err := s.itemRepo.Delete(ctx, itemID); err != nil {
        return err
    }
    
    // 清除缓存
    _ = s.cache.Delete(ctx, "item:"+itemID)
    
    return nil
}

// ListItems 列出物品
func (s *ItemService) ListItems(ctx context.Context, req *ListItemsRequest) (*ListItemsResponse, error) {
    items, total, err := s.itemRepo.List(ctx, req.Type, req.Category, req.Page, req.PageSize)
    if err != nil {
        return nil, err
    }
    
    return &ListItemsResponse{
        Items: items,
        Total: total,
        Page:  req.Page,
    }, nil
}

// SearchItems 搜索物品
func (s *ItemService) SearchItems(ctx context.Context, query string, limit int) ([]*model.Item, error) {
    return s.itemRepo.Search(ctx, query, limit)
}

// SaveItemEmbedding 保存物品向量
func (s *ItemService) SaveItemEmbedding(ctx context.Context, itemID string, embedding []float32) error {
    return s.milvusDB.Insert(ctx, "item_embeddings", []string{itemID}, [][]float32{embedding})
}

// GetSimilarItems 获取相似物品（向量搜索）
func (s *ItemService) GetSimilarItems(ctx context.Context, itemID string, topK int) ([]*SimilarItem, error) {
    // 获取物品向量
    embeddings, err := s.milvusDB.GetByIDs(ctx, "item_embeddings", []string{itemID})
    if err != nil {
        return nil, err
    }
    if len(embeddings) == 0 {
        return nil, ErrItemNotFound
    }
    
    // 向量搜索
    results, err := s.milvusDB.Search(ctx, "item_embeddings", embeddings[0], topK+1)
    if err != nil {
        return nil, err
    }
    
    // 过滤掉自己，获取物品详情
    similarItems := make([]*SimilarItem, 0, topK)
    for _, r := range results {
        if r.ID == itemID {
            continue
        }
        
        item, err := s.GetItem(ctx, r.ID)
        if err != nil {
            continue
        }
        
        similarItems = append(similarItems, &SimilarItem{
            Item:  item,
            Score: r.Score,
        })
        
        if len(similarItems) >= topK {
            break
        }
    }
    
    return similarItems, nil
}

// GetItemStats 获取物品统计
func (s *ItemService) GetItemStats(ctx context.Context, itemID string) (*ItemStats, error) {
    stats, err := s.itemRepo.GetStats(ctx, itemID)
    if err != nil {
        return nil, err
    }
    
    return &ItemStats{
        ItemID:     itemID,
        ViewCount:  stats.ViewCount,
        ClickCount: stats.ClickCount,
        LikeCount:  stats.LikeCount,
        ShareCount: stats.ShareCount,
        AvgRating:  stats.AvgRating,
    }, nil
}

// UpdateItemStats 更新物品统计
func (s *ItemService) UpdateItemStats(ctx context.Context, itemID string, action string) error {
    return s.itemRepo.IncrementStats(ctx, itemID, action)
}

// BatchGetItems 批量获取物品
func (s *ItemService) BatchGetItems(ctx context.Context, itemIDs []string) ([]*model.Item, error) {
    // 先从缓存批量获取
    cachedItems := make(map[string]*model.Item)
    missingIDs := make([]string, 0)
    
    for _, id := range itemIDs {
        var item model.Item
        if err := s.cache.Get(ctx, "item:"+id, &item); err == nil {
            cachedItems[id] = &item
        } else {
            missingIDs = append(missingIDs, id)
        }
    }
    
    // 从数据库获取缺失的
    if len(missingIDs) > 0 {
        dbItems, err := s.itemRepo.GetByIDs(ctx, missingIDs)
        if err != nil {
            return nil, err
        }
        
        for _, item := range dbItems {
            cachedItems[item.ID] = item
            _ = s.cache.Set(ctx, "item:"+item.ID, item, time.Hour)
        }
    }
    
    // 按原顺序返回
    result := make([]*model.Item, 0, len(itemIDs))
    for _, id := range itemIDs {
        if item, ok := cachedItems[id]; ok {
            result = append(result, item)
        }
    }
    
    return result, nil
}

// 请求/响应结构体
type CreateItemRequest struct {
    Type        string            `json:"type" binding:"required"`  // movie, product, article, video
    Title       string            `json:"title" binding:"required"`
    Description string            `json:"description"`
    Category    string            `json:"category"`
    Tags        []string          `json:"tags"`
    Metadata    map[string]any    `json:"metadata"`
    Embedding   []float32         `json:"embedding"`  // 可选的物品向量
}

type UpdateItemRequest struct {
    Title       string         `json:"title"`
    Description string         `json:"description"`
    Category    string         `json:"category"`
    Tags        []string       `json:"tags"`
    Metadata    map[string]any `json:"metadata"`
}

type ListItemsRequest struct {
    Type     string `form:"type"`
    Category string `form:"category"`
    Page     int    `form:"page,default=1"`
    PageSize int    `form:"page_size,default=20"`
}

type ListItemsResponse struct {
    Items []*model.Item `json:"items"`
    Total int64         `json:"total"`
    Page  int           `json:"page"`
}

type SimilarItem struct {
    Item  *model.Item `json:"item"`
    Score float32     `json:"score"`
}

type ItemStats struct {
    ItemID     string  `json:"item_id"`
    ViewCount  int64   `json:"view_count"`
    ClickCount int64   `json:"click_count"`
    LikeCount  int64   `json:"like_count"`
    ShareCount int64   `json:"share_count"`
    AvgRating  float64 `json:"avg_rating"`
}

// 错误定义
var (
    ErrItemNotFound = errors.New("item not found")
)

// 辅助函数
func generateItemID(itemType string) string {
    prefix := map[string]string{
        "movie":   "mov",
        "product": "prd",
        "article": "art",
        "video":   "vid",
    }[itemType]
    if prefix == "" {
        prefix = "itm"
    }
    return prefix + "_" + time.Now().Format("20060102150405") + "_" + randomString(8)
}
```

---

## 2. api/item/v1/handler.go

```go
package v1

import (
    "net/http"
    "strconv"
    
    "github.com/gin-gonic/gin"
    
    "recommend-system/internal/service/item"
)

// ItemHandler 物品 API 处理器
type ItemHandler struct {
    itemService *item.ItemService
}

func NewItemHandler(itemService *item.ItemService) *ItemHandler {
    return &ItemHandler{itemService: itemService}
}

// RegisterRoutes 注册路由
func (h *ItemHandler) RegisterRoutes(r *gin.RouterGroup) {
    items := r.Group("/items")
    {
        items.POST("", h.CreateItem)
        items.GET("", h.ListItems)
        items.GET("/search", h.SearchItems)
        items.GET("/:id", h.GetItem)
        items.PUT("/:id", h.UpdateItem)
        items.DELETE("/:id", h.DeleteItem)
        items.GET("/:id/similar", h.GetSimilarItems)
        items.GET("/:id/stats", h.GetItemStats)
        items.POST("/batch", h.BatchGetItems)
    }
}

// CreateItem 创建物品
// POST /api/v1/items
func (h *ItemHandler) CreateItem(c *gin.Context) {
    var req item.CreateItemRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    result, err := h.itemService.CreateItem(c.Request.Context(), &req)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusCreated, gin.H{"code": 0, "data": result})
}

// GetItem 获取物品
// GET /api/v1/items/:id
func (h *ItemHandler) GetItem(c *gin.Context) {
    itemID := c.Param("id")
    
    result, err := h.itemService.GetItem(c.Request.Context(), itemID)
    if err != nil {
        c.JSON(http.StatusNotFound, gin.H{"error": "item not found"})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{"code": 0, "data": result})
}

// ListItems 列出物品
// GET /api/v1/items?type=movie&category=action&page=1&page_size=20
func (h *ItemHandler) ListItems(c *gin.Context) {
    var req item.ListItemsRequest
    if err := c.ShouldBindQuery(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    result, err := h.itemService.ListItems(c.Request.Context(), &req)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{"code": 0, "data": result})
}

// SearchItems 搜索物品
// GET /api/v1/items/search?q=keyword&limit=10
func (h *ItemHandler) SearchItems(c *gin.Context) {
    query := c.Query("q")
    limit, _ := strconv.Atoi(c.DefaultQuery("limit", "10"))
    
    results, err := h.itemService.SearchItems(c.Request.Context(), query, limit)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{"code": 0, "data": results})
}

// UpdateItem 更新物品
// PUT /api/v1/items/:id
func (h *ItemHandler) UpdateItem(c *gin.Context) {
    itemID := c.Param("id")
    
    var req item.UpdateItemRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    result, err := h.itemService.UpdateItem(c.Request.Context(), itemID, &req)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{"code": 0, "data": result})
}

// DeleteItem 删除物品
// DELETE /api/v1/items/:id
func (h *ItemHandler) DeleteItem(c *gin.Context) {
    itemID := c.Param("id")
    
    if err := h.itemService.DeleteItem(c.Request.Context(), itemID); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{"code": 0, "message": "deleted"})
}

// GetSimilarItems 获取相似物品
// GET /api/v1/items/:id/similar?top_k=10
func (h *ItemHandler) GetSimilarItems(c *gin.Context) {
    itemID := c.Param("id")
    topK, _ := strconv.Atoi(c.DefaultQuery("top_k", "10"))
    
    results, err := h.itemService.GetSimilarItems(c.Request.Context(), itemID, topK)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{"code": 0, "data": results})
}

// GetItemStats 获取物品统计
// GET /api/v1/items/:id/stats
func (h *ItemHandler) GetItemStats(c *gin.Context) {
    itemID := c.Param("id")
    
    stats, err := h.itemService.GetItemStats(c.Request.Context(), itemID)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{"code": 0, "data": stats})
}

// BatchGetItems 批量获取物品
// POST /api/v1/items/batch
func (h *ItemHandler) BatchGetItems(c *gin.Context) {
    var req struct {
        IDs []string `json:"ids" binding:"required"`
    }
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    items, err := h.itemService.BatchGetItems(c.Request.Context(), req.IDs)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{"code": 0, "data": items})
}
```

---

## 3. cmd/item-service/main.go

```go
package main

import (
    "context"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"
    
    "github.com/gin-gonic/gin"
    
    "recommend-system/pkg/config"
    "recommend-system/pkg/logger"
    "recommend-system/pkg/database"
    "recommend-system/internal/cache"
    "recommend-system/internal/repository"
    "recommend-system/internal/middleware"
    "recommend-system/internal/service/item"
    itemv1 "recommend-system/api/item/v1"
)

func main() {
    // 加载配置
    cfg, err := config.LoadConfig("configs/config.yaml")
    if err != nil {
        log.Fatalf("Failed to load config: %v", err)
    }
    
    // 初始化日志
    zapLogger := logger.InitLogger(cfg.Logger.Level)
    
    // 初始化 PostgreSQL
    db, err := database.InitPostgres(cfg.Database)
    if err != nil {
        log.Fatalf("Failed to connect database: %v", err)
    }
    
    // 初始化 Redis
    redisClient, err := database.InitRedis(cfg.Redis)
    if err != nil {
        log.Fatalf("Failed to connect redis: %v", err)
    }
    
    // 初始化 Milvus
    milvusClient, err := database.InitMilvus(cfg.Milvus)
    if err != nil {
        log.Fatalf("Failed to connect milvus: %v", err)
    }
    
    // 初始化 Repository
    itemRepo := repository.NewItemRepository(db, milvusClient)
    
    // 初始化 Cache
    redisCache := cache.NewRedisCache(redisClient)
    
    // 初始化 Service
    itemService := item.NewItemService(itemRepo, milvusClient, redisCache, zapLogger)
    
    // 初始化 Handler
    itemHandler := itemv1.NewItemHandler(itemService)
    
    // 设置 Gin
    router := gin.New()
    router.Use(gin.Recovery())
    router.Use(middleware.RequestLogger(zapLogger))
    router.Use(middleware.CORS())
    
    // 健康检查
    router.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{"status": "ok"})
    })
    
    // 注册路由
    apiV1 := router.Group("/api/v1")
    itemHandler.RegisterRoutes(apiV1)
    
    // 启动服务器
    srv := &http.Server{
        Addr:    ":8083",
        Handler: router,
    }
    
    go func() {
        log.Printf("Item service starting on :8083")
        if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            log.Fatalf("Failed to start server: %v", err)
        }
    }()
    
    // 优雅关闭
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit
    
    log.Println("Shutting down server...")
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    if err := srv.Shutdown(ctx); err != nil {
        log.Fatalf("Server forced to shutdown: %v", err)
    }
    
    log.Println("Server exited")
}
```

---

## API 接口规范

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/items | 创建物品 |
| GET | /api/v1/items | 列出物品 |
| GET | /api/v1/items/search | 搜索物品 |
| GET | /api/v1/items/:id | 获取物品 |
| PUT | /api/v1/items/:id | 更新物品 |
| DELETE | /api/v1/items/:id | 删除物品 |
| GET | /api/v1/items/:id/similar | 获取相似物品 |
| GET | /api/v1/items/:id/stats | 获取物品统计 |
| POST | /api/v1/items/batch | 批量获取物品 |

---

## 注意事项

1. **向量存储**: 物品向量存储在 Milvus，元数据存储在 PostgreSQL
2. **批量操作**: 使用批量接口减少网络开销
3. **缓存策略**: 物品信息缓存 1 小时
4. **相似搜索**: 使用 Milvus 的 ANN 搜索

## 输出要求

请输出完整的可运行代码，包含：
1. 所有 Go 文件
2. 详细的中文注释
3. 单元测试
4. API 文档

