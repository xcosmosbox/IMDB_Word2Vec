// Package coldstart 提供冷启动服务实现
//
// 冷启动服务负责处理新用户和新物品的推荐问题。
// 当用户或物品缺乏历史数据时，利用大语言模型（LLM）生成语义先验，
// 结合跨域知识迁移和快速适应策略，实现高质量的冷启动推荐。
//
// 核心功能：
//   - 新用户冷启动：基于用户属性生成初始偏好画像
//   - 新物品冷启动：基于物品内容生成特征和嵌入向量
//   - 冷启动推荐：为缺乏历史的用户提供个性化推荐
//   - 推荐解释：生成可解释的推荐理由
//
// 使用方式：
//
//	service := coldstart.NewService(cfg, llmClient, userRepo, itemRepo, cache, logger)
//	result, err := service.HandleNewUser(ctx, user)
package coldstart

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"recommend-system/internal/interfaces"
	"recommend-system/pkg/logger"

	"go.uber.org/zap"
)

// =============================================================================
// 服务配置
// =============================================================================

// Config 冷启动服务配置
type Config struct {
	// CacheTTL 缓存过期时间
	CacheTTL time.Duration `json:"cache_ttl" yaml:"cache_ttl"`

	// MaxRecommendations 最大推荐数量
	MaxRecommendations int `json:"max_recommendations" yaml:"max_recommendations"`

	// LLMTimeout LLM 请求超时时间
	LLMTimeout time.Duration `json:"llm_timeout" yaml:"llm_timeout"`

	// EnableLLMFallback 是否启用 LLM 降级
	EnableLLMFallback bool `json:"enable_llm_fallback" yaml:"enable_llm_fallback"`

	// DefaultCategories 默认推荐类别
	DefaultCategories []string `json:"default_categories" yaml:"default_categories"`

	// PopularItemsLimit 热门物品限制数
	PopularItemsLimit int `json:"popular_items_limit" yaml:"popular_items_limit"`
}

// DefaultConfig 返回默认配置
func DefaultConfig() Config {
	return Config{
		CacheTTL:           24 * time.Hour,
		MaxRecommendations: 50,
		LLMTimeout:         30 * time.Second,
		EnableLLMFallback:  true,
		DefaultCategories:  []string{"热门", "推荐", "新品"},
		PopularItemsLimit:  100,
	}
}

// =============================================================================
// 服务实现
// =============================================================================

// Service 冷启动服务
// 实现 interfaces.ColdStartService 接口
type Service struct {
	cfg      Config
	llm      interfaces.LLMClient
	userRepo interfaces.UserRepository
	itemRepo interfaces.ItemRepository
	cache    interfaces.Cache
	logger   *zap.Logger
}

// NewService 创建冷启动服务
//
// 参数：
//   - cfg: 服务配置
//   - llm: LLM 客户端
//   - userRepo: 用户仓库
//   - itemRepo: 物品仓库
//   - cache: 缓存接口
//   - log: 日志器
//
// 返回：
//   - *Service: 服务实例
func NewService(
	cfg Config,
	llm interfaces.LLMClient,
	userRepo interfaces.UserRepository,
	itemRepo interfaces.ItemRepository,
	cache interfaces.Cache,
	log *zap.Logger,
) *Service {
	if log == nil {
		log = zap.NewNop()
	}

	return &Service{
		cfg:      cfg,
		llm:      llm,
		userRepo: userRepo,
		itemRepo: itemRepo,
		cache:    cache,
		logger:   log,
	}
}

// =============================================================================
// ColdStartService 接口实现
// =============================================================================

// HandleNewUser 处理新用户冷启动
//
// 基于用户属性（年龄、性别等）使用 LLM 分析生成初始偏好画像，
// 并据此获取初始推荐列表。
//
// 参数：
//   - ctx: 上下文
//   - user: 用户信息
//
// 返回：
//   - *interfaces.ColdStartResult: 冷启动结果
//   - error: 如果失败则返回错误
func (s *Service) HandleNewUser(ctx context.Context, user *interfaces.User) (*interfaces.ColdStartResult, error) {
	if user == nil {
		return nil, fmt.Errorf("user cannot be nil")
	}

	s.logger.Info("handling new user cold start",
		zap.String("user_id", user.ID),
		zap.Int("age", user.Age),
		zap.String("gender", user.Gender))

	// 1. 构建用户偏好分析提示词
	prompt := s.buildUserPreferencePrompt(user)

	// 2. 使用 LLM 分析用户偏好
	llmCtx, cancel := context.WithTimeout(ctx, s.cfg.LLMTimeout)
	defer cancel()

	response, err := s.llm.Chat(llmCtx, []interfaces.Message{
		{
			Role:    "system",
			Content: userPreferenceSystemPrompt,
		},
		{
			Role:    "user",
			Content: prompt,
		},
	}, interfaces.WithMaxTokens(512), interfaces.WithTemperature(0.3))

	var preferences map[string]interface{}

	if err != nil {
		s.logger.Warn("LLM cold start analysis failed, using fallback",
			zap.String("user_id", user.ID),
			zap.Error(err))

		if s.cfg.EnableLLMFallback {
			return s.fallbackUserColdStart(ctx, user)
		}
		return nil, fmt.Errorf("LLM analysis failed: %w", err)
	}

	// 3. 解析 LLM 响应
	preferences, err = s.parseUserPreferences(response)
	if err != nil {
		s.logger.Warn("failed to parse LLM response, using fallback",
			zap.String("user_id", user.ID),
			zap.String("response", response),
			zap.Error(err))

		if s.cfg.EnableLLMFallback {
			return s.fallbackUserColdStart(ctx, user)
		}
		return nil, fmt.Errorf("failed to parse preferences: %w", err)
	}

	// 4. 基于偏好获取初始推荐
	recommendations, err := s.getInitialRecommendations(ctx, preferences)
	if err != nil {
		s.logger.Warn("failed to get initial recommendations",
			zap.String("user_id", user.ID),
			zap.Error(err))
		recommendations = []string{}
	}

	// 5. 构建冷启动结果
	result := &interfaces.ColdStartResult{
		UserID:          user.ID,
		Preferences:     preferences,
		Recommendations: recommendations,
		Strategy:        "llm_based",
		CreatedAt:       time.Now(),
	}

	// 6. 缓存结果
	if s.cache != nil {
		cacheKey := fmt.Sprintf("coldstart:user:%s", user.ID)
		if cacheErr := s.cache.Set(ctx, cacheKey, result, s.cfg.CacheTTL); cacheErr != nil {
			s.logger.Warn("failed to cache cold start result",
				zap.String("user_id", user.ID),
				zap.Error(cacheErr))
		}
	}

	s.logger.Info("user cold start completed",
		zap.String("user_id", user.ID),
		zap.String("strategy", result.Strategy),
		zap.Int("recommendations_count", len(recommendations)))

	return result, nil
}

// HandleNewItem 处理新物品冷启动
//
// 使用 LLM 分析物品内容，生成特征描述和嵌入向量，
// 并找到相似的已有物品用于冷启动推荐。
//
// 参数：
//   - ctx: 上下文
//   - item: 物品信息
//
// 返回：
//   - *interfaces.ItemColdStartResult: 物品冷启动结果
//   - error: 如果失败则返回错误
func (s *Service) HandleNewItem(ctx context.Context, item *interfaces.Item) (*interfaces.ItemColdStartResult, error) {
	if item == nil {
		return nil, fmt.Errorf("item cannot be nil")
	}

	s.logger.Info("handling new item cold start",
		zap.String("item_id", item.ID),
		zap.String("title", item.Title),
		zap.String("category", item.Category))

	// 1. 构建物品分析提示词
	prompt := s.buildItemAnalysisPrompt(item)

	// 2. 使用 LLM 分析物品特征
	llmCtx, cancel := context.WithTimeout(ctx, s.cfg.LLMTimeout)
	defer cancel()

	response, err := s.llm.Chat(llmCtx, []interfaces.Message{
		{
			Role:    "system",
			Content: itemAnalysisSystemPrompt,
		},
		{
			Role:    "user",
			Content: prompt,
		},
	}, interfaces.WithMaxTokens(512), interfaces.WithTemperature(0.3))

	var features map[string]interface{}

	if err != nil {
		s.logger.Warn("LLM item analysis failed, using fallback",
			zap.String("item_id", item.ID),
			zap.Error(err))

		if s.cfg.EnableLLMFallback {
			return s.fallbackItemColdStart(ctx, item)
		}
		return nil, fmt.Errorf("LLM analysis failed: %w", err)
	}

	// 3. 解析 LLM 响应
	features, err = s.parseItemFeatures(response)
	if err != nil {
		s.logger.Warn("failed to parse item features, using fallback",
			zap.String("item_id", item.ID),
			zap.String("response", response),
			zap.Error(err))

		if s.cfg.EnableLLMFallback {
			return s.fallbackItemColdStart(ctx, item)
		}
		return nil, fmt.Errorf("failed to parse features: %w", err)
	}

	// 4. 生成物品嵌入向量
	embedText := item.Title
	if item.Description != "" {
		embedText += " " + item.Description
	}

	var embedding []float32
	embedding, err = s.llm.Embed(ctx, embedText)
	if err != nil {
		s.logger.Warn("failed to generate item embedding",
			zap.String("item_id", item.ID),
			zap.Error(err))
		// 嵌入失败不影响整体流程
	}

	// 5. 查找相似物品
	similarItems := s.findSimilarItems(ctx, item, embedding)

	// 6. 构建冷启动结果
	result := &interfaces.ItemColdStartResult{
		ItemID:       item.ID,
		Features:     features,
		Embedding:    embedding,
		SimilarItems: similarItems,
		Strategy:     "llm_based",
		CreatedAt:    time.Now(),
	}

	// 7. 缓存结果
	if s.cache != nil {
		cacheKey := fmt.Sprintf("coldstart:item:%s", item.ID)
		if cacheErr := s.cache.Set(ctx, cacheKey, result, s.cfg.CacheTTL); cacheErr != nil {
			s.logger.Warn("failed to cache item cold start result",
				zap.String("item_id", item.ID),
				zap.Error(cacheErr))
		}
	}

	s.logger.Info("item cold start completed",
		zap.String("item_id", item.ID),
		zap.String("strategy", result.Strategy),
		zap.Int("similar_items_count", len(similarItems)))

	return result, nil
}

// GetColdStartRecommendations 获取冷启动推荐
//
// 为新用户或缺乏历史的用户获取推荐列表。
// 首先尝试从缓存获取已有的冷启动结果，如果没有则执行冷启动流程。
//
// 参数：
//   - ctx: 上下文
//   - userID: 用户 ID
//   - limit: 推荐数量限制
//
// 返回：
//   - []*interfaces.Item: 推荐物品列表
//   - error: 如果失败则返回错误
func (s *Service) GetColdStartRecommendations(ctx context.Context, userID string, limit int) ([]*interfaces.Item, error) {
	if userID == "" {
		return nil, fmt.Errorf("user ID cannot be empty")
	}

	if limit <= 0 {
		limit = 20
	}
	if limit > s.cfg.MaxRecommendations {
		limit = s.cfg.MaxRecommendations
	}

	s.logger.Debug("getting cold start recommendations",
		zap.String("user_id", userID),
		zap.Int("limit", limit))

	// 1. 尝试从缓存获取冷启动结果
	var cachedResult interfaces.ColdStartResult
	cacheKey := fmt.Sprintf("coldstart:user:%s", userID)

	if s.cache != nil {
		if err := s.cache.Get(ctx, cacheKey, &cachedResult); err == nil {
			s.logger.Debug("found cached cold start result",
				zap.String("user_id", userID))
			return s.expandRecommendations(ctx, cachedResult.Recommendations, limit)
		}
	}

	// 2. 缓存未命中，获取用户信息
	user, err := s.userRepo.GetByID(ctx, userID)
	if err != nil {
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	// 3. 执行冷启动
	result, err := s.HandleNewUser(ctx, user)
	if err != nil {
		return nil, fmt.Errorf("cold start failed: %w", err)
	}

	// 4. 扩展推荐列表
	return s.expandRecommendations(ctx, result.Recommendations, limit)
}

// ExplainRecommendation 生成推荐解释
//
// 使用 LLM 生成针对特定用户和物品的个性化推荐解释。
//
// 参数：
//   - ctx: 上下文
//   - userID: 用户 ID
//   - itemID: 物品 ID
//
// 返回：
//   - string: 推荐解释文本
//   - error: 如果失败则返回错误
func (s *Service) ExplainRecommendation(ctx context.Context, userID, itemID string) (string, error) {
	if userID == "" || itemID == "" {
		return "", fmt.Errorf("user ID and item ID cannot be empty")
	}

	s.logger.Debug("generating recommendation explanation",
		zap.String("user_id", userID),
		zap.String("item_id", itemID))

	// 1. 获取用户和物品信息
	user, err := s.userRepo.GetByID(ctx, userID)
	if err != nil {
		return "", fmt.Errorf("failed to get user: %w", err)
	}

	item, err := s.itemRepo.GetByID(ctx, itemID)
	if err != nil {
		return "", fmt.Errorf("failed to get item: %w", err)
	}

	// 2. 构建解释生成提示词
	prompt := s.buildExplanationPrompt(user, item)

	// 3. 使用 LLM 生成解释
	llmCtx, cancel := context.WithTimeout(ctx, s.cfg.LLMTimeout)
	defer cancel()

	explanation, err := s.llm.Complete(llmCtx, prompt, interfaces.WithMaxTokens(100), interfaces.WithTemperature(0.5))
	if err != nil {
		s.logger.Warn("failed to generate recommendation explanation",
			zap.String("user_id", userID),
			zap.String("item_id", itemID),
			zap.Error(err))
		// 返回默认解释
		return s.getDefaultExplanation(user, item), nil
	}

	// 4. 清理和返回解释
	explanation = strings.TrimSpace(explanation)
	if explanation == "" {
		return s.getDefaultExplanation(user, item), nil
	}

	return explanation, nil
}

// =============================================================================
// 辅助方法 - 提示词构建
// =============================================================================

// userPreferenceSystemPrompt 用户偏好分析系统提示词
const userPreferenceSystemPrompt = `你是一个推荐系统助手，负责分析用户偏好。
请根据用户的基本信息，推测其可能的兴趣偏好。
请以 JSON 格式返回分析结果，格式如下：
{
    "preferred_categories": ["类别1", "类别2", "类别3"],
    "preferred_tags": ["标签1", "标签2", "标签3"],
    "content_preference": "short/medium/long",
    "price_sensitivity": "low/medium/high",
    "style_preference": ["风格1", "风格2"]
}
只返回 JSON，不要有其他内容。`

// itemAnalysisSystemPrompt 物品分析系统提示词
const itemAnalysisSystemPrompt = `你是一个内容分析助手，负责分析物品特征。
请根据物品信息，提取其核心特征和目标受众。
请以 JSON 格式返回分析结果，格式如下：
{
    "main_category": "主类别",
    "sub_categories": ["子类别1", "子类别2"],
    "target_audience": ["目标用户群1", "目标用户群2"],
    "content_type": "类型描述",
    "quality_score": 0.8,
    "keywords": ["关键词1", "关键词2", "关键词3"]
}
只返回 JSON，不要有其他内容。`

// buildUserPreferencePrompt 构建用户偏好分析提示词
func (s *Service) buildUserPreferencePrompt(user *interfaces.User) string {
	var builder strings.Builder

	builder.WriteString("请分析以下用户的可能偏好：\n\n")

	builder.WriteString(fmt.Sprintf("- 用户ID: %s\n", user.ID))

	if user.Age > 0 {
		builder.WriteString(fmt.Sprintf("- 年龄: %d岁\n", user.Age))
	}

	if user.Gender != "" {
		builder.WriteString(fmt.Sprintf("- 性别: %s\n", user.Gender))
	}

	if len(user.Metadata) > 0 {
		builder.WriteString("- 其他信息:\n")
		for k, v := range user.Metadata {
			builder.WriteString(fmt.Sprintf("  - %s: %s\n", k, v))
		}
	}

	builder.WriteString("\n请返回 JSON 格式的偏好分析结果。")

	return builder.String()
}

// buildItemAnalysisPrompt 构建物品分析提示词
func (s *Service) buildItemAnalysisPrompt(item *interfaces.Item) string {
	var builder strings.Builder

	builder.WriteString("请分析以下物品的特征：\n\n")

	builder.WriteString(fmt.Sprintf("- 标题: %s\n", item.Title))

	if item.Category != "" {
		builder.WriteString(fmt.Sprintf("- 类别: %s\n", item.Category))
	}

	if item.Description != "" {
		desc := item.Description
		if len(desc) > 500 {
			desc = desc[:500] + "..."
		}
		builder.WriteString(fmt.Sprintf("- 描述: %s\n", desc))
	}

	if len(item.Tags) > 0 {
		builder.WriteString(fmt.Sprintf("- 标签: %s\n", strings.Join(item.Tags, ", ")))
	}

	if item.Type != "" {
		builder.WriteString(fmt.Sprintf("- 类型: %s\n", item.Type))
	}

	builder.WriteString("\n请返回 JSON 格式的特征分析结果。")

	return builder.String()
}

// buildExplanationPrompt 构建推荐解释提示词
func (s *Service) buildExplanationPrompt(user *interfaces.User, item *interfaces.Item) string {
	var builder strings.Builder

	builder.WriteString("请用一句简洁的话解释为什么向这位用户推荐这个物品。\n\n")

	builder.WriteString("用户信息：\n")
	if user.Age > 0 {
		builder.WriteString(fmt.Sprintf("- 年龄：%d岁\n", user.Age))
	}
	if user.Gender != "" {
		builder.WriteString(fmt.Sprintf("- 性别：%s\n", user.Gender))
	}

	builder.WriteString("\n推荐物品：\n")
	builder.WriteString(fmt.Sprintf("- 标题：%s\n", item.Title))
	if item.Category != "" {
		builder.WriteString(fmt.Sprintf("- 类别：%s\n", item.Category))
	}
	if item.Description != "" {
		desc := item.Description
		if len(desc) > 200 {
			desc = desc[:200] + "..."
		}
		builder.WriteString(fmt.Sprintf("- 描述：%s\n", desc))
	}

	builder.WriteString("\n请用一句话回答，不超过50个字。")

	return builder.String()
}

// =============================================================================
// 辅助方法 - 响应解析
// =============================================================================

// parseUserPreferences 解析用户偏好 JSON
func (s *Service) parseUserPreferences(response string) (map[string]interface{}, error) {
	// 清理响应，提取 JSON 部分
	response = extractJSON(response)

	var preferences map[string]interface{}
	if err := json.Unmarshal([]byte(response), &preferences); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	return preferences, nil
}

// parseItemFeatures 解析物品特征 JSON
func (s *Service) parseItemFeatures(response string) (map[string]interface{}, error) {
	response = extractJSON(response)

	var features map[string]interface{}
	if err := json.Unmarshal([]byte(response), &features); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	return features, nil
}

// extractJSON 从响应中提取 JSON 部分
func extractJSON(response string) string {
	response = strings.TrimSpace(response)

	// 查找 JSON 开始和结束位置
	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")

	if start != -1 && end != -1 && end > start {
		return response[start : end+1]
	}

	return response
}

// =============================================================================
// 辅助方法 - 推荐获取
// =============================================================================

// getInitialRecommendations 基于偏好获取初始推荐
func (s *Service) getInitialRecommendations(ctx context.Context, preferences map[string]interface{}) ([]string, error) {
	// 从偏好中提取类别
	categories := s.extractCategories(preferences)

	if len(categories) == 0 {
		categories = s.cfg.DefaultCategories
	}

	// 获取热门物品
	items, err := s.itemRepo.GetPopularByCategories(ctx, categories, s.cfg.PopularItemsLimit)
	if err != nil {
		return nil, err
	}

	// 提取物品 ID
	ids := make([]string, len(items))
	for i, item := range items {
		ids[i] = item.ID
	}

	return ids, nil
}

// extractCategories 从偏好中提取类别
func (s *Service) extractCategories(preferences map[string]interface{}) []string {
	var categories []string

	// 尝试从 preferred_categories 提取
	if cats, ok := preferences["preferred_categories"]; ok {
		if catList, ok := cats.([]interface{}); ok {
			for _, cat := range catList {
				if catStr, ok := cat.(string); ok {
					categories = append(categories, catStr)
				}
			}
		}
	}

	return categories
}

// findSimilarItems 查找相似物品
func (s *Service) findSimilarItems(ctx context.Context, item *interfaces.Item, embedding []float32) []string {
	// 如果有嵌入向量，可以使用向量搜索（需要向量数据库支持）
	// 这里简化处理，基于类别查找

	if item.Category == "" {
		return nil
	}

	items, err := s.itemRepo.GetPopularByCategories(ctx, []string{item.Category}, 10)
	if err != nil {
		s.logger.Warn("failed to find similar items",
			zap.String("item_id", item.ID),
			zap.Error(err))
		return nil
	}

	// 过滤掉自身
	var similarIDs []string
	for _, i := range items {
		if i.ID != item.ID {
			similarIDs = append(similarIDs, i.ID)
		}
	}

	return similarIDs
}

// expandRecommendations 扩展推荐列表为物品对象
func (s *Service) expandRecommendations(ctx context.Context, itemIDs []string, limit int) ([]*interfaces.Item, error) {
	if len(itemIDs) == 0 {
		return []*interfaces.Item{}, nil
	}

	// 限制数量
	if len(itemIDs) > limit {
		itemIDs = itemIDs[:limit]
	}

	// 批量获取物品
	items, err := s.itemRepo.GetByIDs(ctx, itemIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to get items: %w", err)
	}

	return items, nil
}

// =============================================================================
// 辅助方法 - 降级策略
// =============================================================================

// fallbackUserColdStart 用户冷启动降级策略
// 基于人口统计学特征生成默认推荐
func (s *Service) fallbackUserColdStart(ctx context.Context, user *interfaces.User) (*interfaces.ColdStartResult, error) {
	s.logger.Info("using fallback cold start for user",
		zap.String("user_id", user.ID))

	// 基于年龄生成默认偏好
	preferences := s.generateDefaultPreferences(user)

	// 获取热门物品
	categories := s.extractCategories(preferences)
	if len(categories) == 0 {
		categories = s.cfg.DefaultCategories
	}

	items, err := s.itemRepo.GetPopularByCategories(ctx, categories, s.cfg.PopularItemsLimit)
	if err != nil {
		// 如果获取失败，返回空推荐
		items = []*interfaces.Item{}
	}

	// 提取物品 ID
	recommendations := make([]string, len(items))
	for i, item := range items {
		recommendations[i] = item.ID
	}

	return &interfaces.ColdStartResult{
		UserID:          user.ID,
		Preferences:     preferences,
		Recommendations: recommendations,
		Strategy:        "demographic_fallback",
		CreatedAt:       time.Now(),
	}, nil
}

// fallbackItemColdStart 物品冷启动降级策略
func (s *Service) fallbackItemColdStart(ctx context.Context, item *interfaces.Item) (*interfaces.ItemColdStartResult, error) {
	s.logger.Info("using fallback cold start for item",
		zap.String("item_id", item.ID))

	// 生成基本特征
	features := map[string]interface{}{
		"main_category":  item.Category,
		"sub_categories": item.Tags,
		"content_type":   item.Type,
	}

	// 查找同类别的相似物品
	var similarItems []string
	if item.Category != "" {
		items, err := s.itemRepo.GetPopularByCategories(ctx, []string{item.Category}, 10)
		if err == nil {
			for _, i := range items {
				if i.ID != item.ID {
					similarItems = append(similarItems, i.ID)
				}
			}
		}
	}

	return &interfaces.ItemColdStartResult{
		ItemID:       item.ID,
		Features:     features,
		Embedding:    nil,
		SimilarItems: similarItems,
		Strategy:     "fallback",
		CreatedAt:    time.Now(),
	}, nil
}

// generateDefaultPreferences 基于用户属性生成默认偏好
func (s *Service) generateDefaultPreferences(user *interfaces.User) map[string]interface{} {
	preferences := make(map[string]interface{})

	// 基于年龄生成偏好类别
	var categories []string
	switch {
	case user.Age > 0 && user.Age < 18:
		categories = []string{"动漫", "游戏", "学习"}
	case user.Age >= 18 && user.Age < 25:
		categories = []string{"科技", "游戏", "娱乐", "时尚"}
	case user.Age >= 25 && user.Age < 35:
		categories = []string{"商业", "科技", "生活", "理财"}
	case user.Age >= 35 && user.Age < 50:
		categories = []string{"商业", "新闻", "健康", "家庭"}
	case user.Age >= 50:
		categories = []string{"健康", "新闻", "文化", "养生"}
	default:
		categories = s.cfg.DefaultCategories
	}

	preferences["preferred_categories"] = categories

	// 基于性别调整（可选）
	if user.Gender == "female" {
		preferences["preferred_tags"] = []string{"时尚", "美妆", "生活"}
	} else if user.Gender == "male" {
		preferences["preferred_tags"] = []string{"科技", "运动", "汽车"}
	}

	preferences["content_preference"] = "medium"
	preferences["price_sensitivity"] = "medium"

	return preferences
}

// getDefaultExplanation 获取默认推荐解释
func (s *Service) getDefaultExplanation(user *interfaces.User, item *interfaces.Item) string {
	if item.Category != "" {
		return fmt.Sprintf("根据您的偏好，为您推荐这款%s类内容", item.Category)
	}
	return "根据您的偏好为您推荐"
}

// =============================================================================
// 接口验证
// =============================================================================

// 确保 Service 实现了 ColdStartService 接口
var _ interfaces.ColdStartService = (*Service)(nil)

// =============================================================================
// 日志包装器（兼容不同日志实现）
// =============================================================================

// NewServiceWithLogger 使用 logger 包创建服务（向后兼容）
func NewServiceWithLogger(
	cfg Config,
	llm interfaces.LLMClient,
	userRepo interfaces.UserRepository,
	itemRepo interfaces.ItemRepository,
	cache interfaces.Cache,
) *Service {
	return NewService(cfg, llm, userRepo, itemRepo, cache, logger.Logger)
}

