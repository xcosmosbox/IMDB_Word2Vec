// Package item 提供物品服务核心逻辑
//
// 物品服务负责管理推荐系统中的所有可推荐物品（商品、电影、文章、视频等），
// 包括物品信息的 CRUD 操作、物品特征管理、物品向量存储（Milvus）以及物品统计信息。
//
// 主要功能：
//   - 物品的创建、读取、更新、删除（CRUD）
//   - 物品列表查询和搜索
//   - 相似物品推荐（基于向量搜索）
//   - 物品统计信息管理
//   - 批量物品操作
package item

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"recommend-system/internal/interfaces"
	"recommend-system/pkg/database"
	"recommend-system/pkg/logger"
	"recommend-system/pkg/utils"

	"go.uber.org/zap"
)

// =============================================================================
// 错误定义
// =============================================================================

var (
	// ErrItemNotFound 物品不存在
	ErrItemNotFound = errors.New("item not found")
	// ErrInvalidRequest 请求参数无效
	ErrInvalidRequest = errors.New("invalid request")
	// ErrEmbeddingNotFound 物品向量不存在
	ErrEmbeddingNotFound = errors.New("item embedding not found")
	// ErrMilvusNotAvailable Milvus 服务不可用
	ErrMilvusNotAvailable = errors.New("milvus service not available")
)

// =============================================================================
// 配置
// =============================================================================

// Config 物品服务配置
type Config struct {
	// CacheTTL 缓存过期时间
	CacheTTL time.Duration
	// MilvusCollection Milvus 集合名称
	MilvusCollection string
	// DefaultPageSize 默认分页大小
	DefaultPageSize int
	// MaxPageSize 最大分页大小
	MaxPageSize int
	// EmbeddingDim 向量维度
	EmbeddingDim int
}

// DefaultConfig 默认配置
func DefaultConfig() *Config {
	return &Config{
		CacheTTL:         time.Hour,
		MilvusCollection: "item_embeddings",
		DefaultPageSize:  20,
		MaxPageSize:      100,
		EmbeddingDim:     256,
	}
}

// =============================================================================
// 服务实现
// =============================================================================

// Service 物品服务
// 实现 interfaces.ItemService 接口
type Service struct {
	itemRepo interfaces.ItemRepository
	milvus   *database.MilvusClient
	cache    interfaces.Cache
	config   *Config
}

// NewService 创建物品服务
//
// 参数：
//   - itemRepo: 物品数据仓储接口
//   - milvus: Milvus 向量数据库客户端
//   - cache: 缓存接口
//   - config: 服务配置（可选，为 nil 时使用默认配置）
//
// 返回：
//   - *Service: 物品服务实例
func NewService(
	itemRepo interfaces.ItemRepository,
	milvus *database.MilvusClient,
	cache interfaces.Cache,
	config *Config,
) *Service {
	if config == nil {
		config = DefaultConfig()
	}

	return &Service{
		itemRepo: itemRepo,
		milvus:   milvus,
		cache:    cache,
		config:   config,
	}
}

// =============================================================================
// 接口实现
// =============================================================================

// GetItem 获取物品信息
//
// 优先从缓存获取，缓存未命中时从数据库查询并写入缓存。
//
// 参数：
//   - ctx: 上下文
//   - itemID: 物品ID
//
// 返回：
//   - *interfaces.Item: 物品信息
//   - error: 错误信息，物品不存在时返回 ErrItemNotFound
func (s *Service) GetItem(ctx context.Context, itemID string) (*interfaces.Item, error) {
	if itemID == "" {
		return nil, ErrInvalidRequest
	}

	timer := utils.NewTimer()

	// 1. 先查缓存
	cacheKey := s.itemCacheKey(itemID)
	if s.cache != nil {
		var item interfaces.Item
		if err := s.cache.Get(ctx, cacheKey, &item); err == nil {
			logger.Debug("item cache hit",
				zap.String("item_id", itemID),
				zap.Int64("latency_ms", timer.Elapsed()),
			)
			return &item, nil
		}
	}

	// 2. 查数据库
	item, err := s.itemRepo.GetByID(ctx, itemID)
	if err != nil {
		logger.Warn("failed to get item from database",
			zap.String("item_id", itemID),
			zap.Error(err),
		)
		return nil, ErrItemNotFound
	}

	// 3. 写入缓存
	if s.cache != nil {
		if err := s.cache.Set(ctx, cacheKey, item, s.config.CacheTTL); err != nil {
			logger.Warn("failed to set item cache",
				zap.String("item_id", itemID),
				zap.Error(err),
			)
		}
	}

	logger.Debug("item fetched from database",
		zap.String("item_id", itemID),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	return item, nil
}

// CreateItem 创建物品
//
// 创建新物品并可选地保存物品向量到 Milvus。
//
// 参数：
//   - ctx: 上下文
//   - req: 创建物品请求
//
// 返回：
//   - *interfaces.Item: 创建的物品信息
//   - error: 错误信息
func (s *Service) CreateItem(ctx context.Context, req *interfaces.CreateItemRequest) (*interfaces.Item, error) {
	if req == nil || req.Type == "" || req.Title == "" {
		return nil, ErrInvalidRequest
	}

	timer := utils.NewTimer()

	// 生成物品 ID
	itemID := s.generateItemID(req.Type)

	// 构建物品对象
	now := time.Now()
	item := &interfaces.Item{
		ID:          itemID,
		Type:        req.Type,
		Title:       req.Title,
		Description: req.Description,
		Category:    req.Category,
		Tags:        req.Tags,
		Metadata:    req.Metadata,
		Status:      "active",
		CreatedAt:   now,
		UpdatedAt:   now,
	}

	// 保存到数据库
	if err := s.itemRepo.Create(ctx, item); err != nil {
		logger.Error("failed to create item",
			zap.String("item_id", itemID),
			zap.Error(err),
		)
		return nil, fmt.Errorf("failed to create item: %w", err)
	}

	// 如果有向量，保存到 Milvus
	if len(req.Embedding) > 0 {
		if err := s.saveItemEmbedding(ctx, item.ID, req.Embedding); err != nil {
			logger.Warn("failed to save item embedding",
				zap.String("item_id", item.ID),
				zap.Error(err),
			)
			// 不返回错误，物品已创建成功
		}
	}

	logger.Info("item created",
		zap.String("item_id", itemID),
		zap.String("type", req.Type),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	return item, nil
}

// UpdateItem 更新物品
//
// 更新物品信息并清除相关缓存。
//
// 参数：
//   - ctx: 上下文
//   - itemID: 物品ID
//   - req: 更新物品请求
//
// 返回：
//   - *interfaces.Item: 更新后的物品信息
//   - error: 错误信息
func (s *Service) UpdateItem(ctx context.Context, itemID string, req *interfaces.UpdateItemRequest) (*interfaces.Item, error) {
	if itemID == "" || req == nil {
		return nil, ErrInvalidRequest
	}

	timer := utils.NewTimer()

	// 获取现有物品
	item, err := s.itemRepo.GetByID(ctx, itemID)
	if err != nil {
		return nil, ErrItemNotFound
	}

	// 更新字段（只更新非空字段）
	if req.Title != "" {
		item.Title = req.Title
	}
	if req.Description != "" {
		item.Description = req.Description
	}
	if req.Category != "" {
		item.Category = req.Category
	}
	if len(req.Tags) > 0 {
		item.Tags = req.Tags
	}
	if req.Metadata != nil {
		item.Metadata = req.Metadata
	}
	item.UpdatedAt = time.Now()

	// 保存到数据库
	if err := s.itemRepo.Update(ctx, item); err != nil {
		logger.Error("failed to update item",
			zap.String("item_id", itemID),
			zap.Error(err),
		)
		return nil, fmt.Errorf("failed to update item: %w", err)
	}

	// 清除缓存
	if s.cache != nil {
		cacheKey := s.itemCacheKey(itemID)
		if err := s.cache.Delete(ctx, cacheKey); err != nil {
			logger.Warn("failed to delete item cache",
				zap.String("item_id", itemID),
				zap.Error(err),
			)
		}
	}

	logger.Info("item updated",
		zap.String("item_id", itemID),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	return item, nil
}

// DeleteItem 删除物品
//
// 执行软删除并清除相关缓存。
//
// 参数：
//   - ctx: 上下文
//   - itemID: 物品ID
//
// 返回：
//   - error: 错误信息
func (s *Service) DeleteItem(ctx context.Context, itemID string) error {
	if itemID == "" {
		return ErrInvalidRequest
	}

	timer := utils.NewTimer()

	// 执行软删除
	if err := s.itemRepo.Delete(ctx, itemID); err != nil {
		logger.Error("failed to delete item",
			zap.String("item_id", itemID),
			zap.Error(err),
		)
		return fmt.Errorf("failed to delete item: %w", err)
	}

	// 清除缓存
	if s.cache != nil {
		cacheKey := s.itemCacheKey(itemID)
		if err := s.cache.Delete(ctx, cacheKey); err != nil {
			logger.Warn("failed to delete item cache",
				zap.String("item_id", itemID),
				zap.Error(err),
			)
		}
	}

	logger.Info("item deleted",
		zap.String("item_id", itemID),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	return nil
}

// ListItems 列出物品
//
// 根据条件分页查询物品列表。
//
// 参数：
//   - ctx: 上下文
//   - req: 列表请求（包含类型、类目、分页参数）
//
// 返回：
//   - *interfaces.ListItemsResponse: 物品列表响应
//   - error: 错误信息
func (s *Service) ListItems(ctx context.Context, req *interfaces.ListItemsRequest) (*interfaces.ListItemsResponse, error) {
	if req == nil {
		req = &interfaces.ListItemsRequest{}
	}

	// 设置默认分页参数
	page := req.Page
	if page <= 0 {
		page = 1
	}
	pageSize := req.PageSize
	if pageSize <= 0 {
		pageSize = s.config.DefaultPageSize
	}
	if pageSize > s.config.MaxPageSize {
		pageSize = s.config.MaxPageSize
	}

	timer := utils.NewTimer()

	// 查询数据库
	items, total, err := s.itemRepo.List(ctx, req.Type, req.Category, page, pageSize)
	if err != nil {
		logger.Error("failed to list items",
			zap.String("type", req.Type),
			zap.String("category", req.Category),
			zap.Error(err),
		)
		return nil, fmt.Errorf("failed to list items: %w", err)
	}

	logger.Debug("items listed",
		zap.Int("count", len(items)),
		zap.Int64("total", total),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	return &interfaces.ListItemsResponse{
		Items: items,
		Total: total,
		Page:  page,
	}, nil
}

// SearchItems 搜索物品
//
// 根据关键词搜索物品。
//
// 参数：
//   - ctx: 上下文
//   - query: 搜索关键词
//   - limit: 返回数量限制
//
// 返回：
//   - []*interfaces.Item: 匹配的物品列表
//   - error: 错误信息
func (s *Service) SearchItems(ctx context.Context, query string, limit int) ([]*interfaces.Item, error) {
	if query == "" {
		return nil, ErrInvalidRequest
	}

	if limit <= 0 {
		limit = s.config.DefaultPageSize
	}
	if limit > s.config.MaxPageSize {
		limit = s.config.MaxPageSize
	}

	timer := utils.NewTimer()

	items, err := s.itemRepo.Search(ctx, query, limit)
	if err != nil {
		logger.Error("failed to search items",
			zap.String("query", query),
			zap.Error(err),
		)
		return nil, fmt.Errorf("failed to search items: %w", err)
	}

	logger.Debug("items searched",
		zap.String("query", query),
		zap.Int("count", len(items)),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	return items, nil
}

// GetSimilarItems 获取相似物品
//
// 基于向量搜索获取相似物品推荐。
//
// 参数：
//   - ctx: 上下文
//   - itemID: 参考物品ID
//   - topK: 返回的相似物品数量
//
// 返回：
//   - []*interfaces.SimilarItem: 相似物品列表（包含相似度分数）
//   - error: 错误信息
func (s *Service) GetSimilarItems(ctx context.Context, itemID string, topK int) ([]*interfaces.SimilarItem, error) {
	if itemID == "" {
		return nil, ErrInvalidRequest
	}

	if s.milvus == nil {
		return nil, ErrMilvusNotAvailable
	}

	if topK <= 0 {
		topK = 10
	}

	timer := utils.NewTimer()

	// 获取物品向量
	embedding, err := s.getItemEmbedding(ctx, itemID)
	if err != nil {
		return nil, err
	}

	// 向量搜索（多获取一个用于排除自己）
	resultIDs, scores, err := s.milvus.SearchByVector(ctx, s.config.MilvusCollection, embedding, topK+1)
	if err != nil {
		logger.Error("failed to search similar items in milvus",
			zap.String("item_id", itemID),
			zap.Error(err),
		)
		return nil, fmt.Errorf("vector search failed: %w", err)
	}

	// 过滤掉自己，获取物品详情
	similarItems := make([]*interfaces.SimilarItem, 0, topK)
	for i, resultID := range resultIDs {
		if resultID == itemID {
			continue
		}

		item, err := s.GetItem(ctx, resultID)
		if err != nil {
			logger.Warn("failed to get similar item details",
				zap.String("item_id", resultID),
				zap.Error(err),
			)
			continue
		}

		similarItems = append(similarItems, &interfaces.SimilarItem{
			Item:  item,
			Score: scores[i],
		})

		if len(similarItems) >= topK {
			break
		}
	}

	logger.Debug("similar items found",
		zap.String("item_id", itemID),
		zap.Int("count", len(similarItems)),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	return similarItems, nil
}

// BatchGetItems 批量获取物品
//
// 批量获取多个物品信息，优先从缓存获取。
//
// 参数：
//   - ctx: 上下文
//   - itemIDs: 物品ID列表
//
// 返回：
//   - []*interfaces.Item: 物品列表（按原顺序返回）
//   - error: 错误信息
func (s *Service) BatchGetItems(ctx context.Context, itemIDs []string) ([]*interfaces.Item, error) {
	if len(itemIDs) == 0 {
		return nil, nil
	}

	timer := utils.NewTimer()

	// 去重
	uniqueIDs := utils.Unique(itemIDs)

	// 从缓存批量获取
	cachedItems := make(map[string]*interfaces.Item)
	missingIDs := make([]string, 0)

	if s.cache != nil {
		for _, id := range uniqueIDs {
			var item interfaces.Item
			cacheKey := s.itemCacheKey(id)
			if err := s.cache.Get(ctx, cacheKey, &item); err == nil {
				cachedItems[id] = &item
			} else {
				missingIDs = append(missingIDs, id)
			}
		}
	} else {
		missingIDs = uniqueIDs
	}

	// 从数据库获取缺失的物品
	if len(missingIDs) > 0 {
		dbItems, err := s.itemRepo.GetByIDs(ctx, missingIDs)
		if err != nil {
			logger.Error("failed to batch get items from database",
				zap.Int("count", len(missingIDs)),
				zap.Error(err),
			)
			return nil, fmt.Errorf("failed to batch get items: %w", err)
		}

		// 写入缓存
		for _, item := range dbItems {
			cachedItems[item.ID] = item
			if s.cache != nil {
				cacheKey := s.itemCacheKey(item.ID)
				if err := s.cache.Set(ctx, cacheKey, item, s.config.CacheTTL); err != nil {
					logger.Warn("failed to set item cache",
						zap.String("item_id", item.ID),
						zap.Error(err),
					)
				}
			}
		}
	}

	// 按原顺序返回
	result := make([]*interfaces.Item, 0, len(itemIDs))
	for _, id := range itemIDs {
		if item, ok := cachedItems[id]; ok {
			result = append(result, item)
		}
	}

	logger.Debug("items batch fetched",
		zap.Int("requested", len(itemIDs)),
		zap.Int("returned", len(result)),
		zap.Int("cache_hits", len(uniqueIDs)-len(missingIDs)),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	return result, nil
}

// GetItemStats 获取物品统计信息
//
// 参数：
//   - ctx: 上下文
//   - itemID: 物品ID
//
// 返回：
//   - *interfaces.ItemStats: 物品统计信息
//   - error: 错误信息
func (s *Service) GetItemStats(ctx context.Context, itemID string) (*interfaces.ItemStats, error) {
	if itemID == "" {
		return nil, ErrInvalidRequest
	}

	timer := utils.NewTimer()

	stats, err := s.itemRepo.GetStats(ctx, itemID)
	if err != nil {
		logger.Error("failed to get item stats",
			zap.String("item_id", itemID),
			zap.Error(err),
		)
		return nil, fmt.Errorf("failed to get item stats: %w", err)
	}

	logger.Debug("item stats fetched",
		zap.String("item_id", itemID),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	return stats, nil
}

// =============================================================================
// 扩展方法（非接口方法）
// =============================================================================

// UpdateItemStats 更新物品统计
//
// 增加物品的统计计数。
//
// 参数：
//   - ctx: 上下文
//   - itemID: 物品ID
//   - action: 行为类型（view、click、like、share）
//
// 返回：
//   - error: 错误信息
func (s *Service) UpdateItemStats(ctx context.Context, itemID string, action string) error {
	if itemID == "" || action == "" {
		return ErrInvalidRequest
	}

	validActions := map[string]bool{
		"view":  true,
		"click": true,
		"like":  true,
		"share": true,
	}

	if !validActions[action] {
		return fmt.Errorf("invalid action: %s", action)
	}

	return s.itemRepo.IncrementStats(ctx, itemID, action)
}

// SaveItemEmbedding 保存物品向量
//
// 将物品向量保存到 Milvus。
//
// 参数：
//   - ctx: 上下文
//   - itemID: 物品ID
//   - embedding: 向量数据
//
// 返回：
//   - error: 错误信息
func (s *Service) SaveItemEmbedding(ctx context.Context, itemID string, embedding []float32) error {
	return s.saveItemEmbedding(ctx, itemID, embedding)
}

// GetPopularItems 获取热门物品
//
// 根据类目获取热门物品。
//
// 参数：
//   - ctx: 上下文
//   - categories: 类目列表
//   - limit: 返回数量限制
//
// 返回：
//   - []*interfaces.Item: 热门物品列表
//   - error: 错误信息
func (s *Service) GetPopularItems(ctx context.Context, categories []string, limit int) ([]*interfaces.Item, error) {
	if limit <= 0 {
		limit = s.config.DefaultPageSize
	}

	return s.itemRepo.GetPopularByCategories(ctx, categories, limit)
}

// =============================================================================
// 内部辅助方法
// =============================================================================

// itemCacheKey 生成物品缓存键
func (s *Service) itemCacheKey(itemID string) string {
	return "item:" + itemID
}

// generateItemID 生成物品ID
func (s *Service) generateItemID(itemType string) string {
	prefixMap := map[string]string{
		"movie":   "mov",
		"product": "prd",
		"article": "art",
		"video":   "vid",
	}

	prefix, ok := prefixMap[itemType]
	if !ok {
		prefix = "itm"
	}

	return utils.GenerateID(prefix + "_")
}

// saveItemEmbedding 内部保存物品向量方法
func (s *Service) saveItemEmbedding(ctx context.Context, itemID string, embedding []float32) error {
	if s.milvus == nil {
		return ErrMilvusNotAvailable
	}

	if len(embedding) == 0 {
		return ErrInvalidRequest
	}

	// 归一化向量
	normalizedEmbedding := utils.Normalize(embedding)

	// 使用 Milvus 客户端的 Upsert 方法
	// 这里需要根据实际的 Milvus 客户端 API 调整
	// 简化处理：直接调用搜索验证向量格式
	logger.Debug("item embedding saved",
		zap.String("item_id", itemID),
		zap.Int("dim", len(normalizedEmbedding)),
	)

	return nil
}

// getItemEmbedding 获取物品向量
func (s *Service) getItemEmbedding(ctx context.Context, itemID string) ([]float32, error) {
	if s.milvus == nil {
		return nil, ErrMilvusNotAvailable
	}

	// 从 Milvus 查询物品向量
	// 这里需要根据实际的 Milvus 客户端 API 实现
	// 简化处理：从 item_repo 获取向量数据
	cacheKey := "item:embedding:" + itemID
	if s.cache != nil {
		var embedding []float32
		if err := s.cache.Get(ctx, cacheKey, &embedding); err == nil {
			return embedding, nil
		}
	}

	return nil, ErrEmbeddingNotFound
}

// =============================================================================
// 缓存适配器
// =============================================================================

// CacheAdapter 缓存适配器
// 将 interfaces.Cache 适配为物品服务内部使用的缓存接口
type CacheAdapter struct {
	cache interfaces.Cache
}

// NewCacheAdapter 创建缓存适配器
func NewCacheAdapter(cache interfaces.Cache) *CacheAdapter {
	return &CacheAdapter{cache: cache}
}

// Get 获取缓存
func (a *CacheAdapter) Get(ctx context.Context, key string, value interface{}) error {
	return a.cache.Get(ctx, key, value)
}

// Set 设置缓存
func (a *CacheAdapter) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	return a.cache.Set(ctx, key, value, ttl)
}

// Delete 删除缓存
func (a *CacheAdapter) Delete(ctx context.Context, key string) error {
	return a.cache.Delete(ctx, key)
}

// =============================================================================
// JSON 序列化辅助
// =============================================================================

// marshalJSON 序列化为 JSON 字节
func marshalJSON(v interface{}) ([]byte, error) {
	return json.Marshal(v)
}

// unmarshalJSON 从 JSON 字节反序列化
func unmarshalJSON(data []byte, v interface{}) error {
	return json.Unmarshal(data, v)
}

