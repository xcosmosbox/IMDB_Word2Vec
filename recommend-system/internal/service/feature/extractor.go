package feature

import (
	"context"
	"sort"
	"time"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// 特征提取器
// =============================================================================

// FeatureExtractor 特征提取器
// 负责从原始数据中提取用户特征、物品特征、交叉特征和上下文特征
type FeatureExtractor struct {
	userRepo interfaces.UserRepository // 用户数据仓库
	itemRepo interfaces.ItemRepository // 物品数据仓库
}

// NewFeatureExtractor 创建特征提取器实例
func NewFeatureExtractor(userRepo interfaces.UserRepository, itemRepo interfaces.ItemRepository) *FeatureExtractor {
	return &FeatureExtractor{
		userRepo: userRepo,
		itemRepo: itemRepo,
	}
}

// =============================================================================
// 用户特征提取
// =============================================================================

// ExtractUserFeatures 提取用户特征
// 从用户基本信息和行为数据中提取人口统计特征、行为特征和偏好特征
func (e *FeatureExtractor) ExtractUserFeatures(ctx context.Context, userID string) (*InternalUserFeatures, error) {
	// 获取用户基本信息
	user, err := e.userRepo.GetByID(ctx, userID)
	if err != nil {
		return nil, ErrUserNotFound
	}

	// 获取用户行为历史（最近1000条）
	behaviors, err := e.userRepo.GetBehaviors(ctx, userID, 1000)
	if err != nil {
		// 用户可能没有行为记录，使用空列表继续
		behaviors = []*interfaces.UserBehavior{}
	}

	// 提取人口统计特征
	demographics := e.extractDemographics(user)

	// 提取行为特征
	behaviorFeatures := e.extractBehaviorFeatures(behaviors)

	// 提取偏好特征
	preferenceFeatures := e.extractPreferenceFeatures(ctx, behaviors)

	return &InternalUserFeatures{
		UserID:       userID,
		Demographics: demographics,
		Behavior:     behaviorFeatures,
		Preferences:  preferenceFeatures,
		LastUpdated:  time.Now(),
	}, nil
}

// extractDemographics 提取人口统计特征
func (e *FeatureExtractor) extractDemographics(user *interfaces.User) DemographicFeatures {
	demographics := DemographicFeatures{
		Age:    user.Age,
		Gender: user.Gender,
	}

	// 从元数据中提取额外信息
	if user.Metadata != nil {
		if location, ok := user.Metadata["location"]; ok {
			demographics.Location = location
		}
		if device, ok := user.Metadata["device"]; ok {
			demographics.Device = device
		}
	}

	return demographics
}

// extractBehaviorFeatures 从行为列表提取行为特征
func (e *FeatureExtractor) extractBehaviorFeatures(behaviors []*interfaces.UserBehavior) BehaviorFeatures {
	features := BehaviorFeatures{
		PreferredHours: make([]int, 0),
	}

	if len(behaviors) == 0 {
		return features
	}

	// 统计行为类型
	hourCounts := make(map[int]int)
	activeDays := make(map[string]bool)
	var totalDuration float64
	var sessionCount int

	for _, b := range behaviors {
		// 统计行为类型
		switch b.Action {
		case "view":
			features.TotalViews++
		case "click":
			features.TotalClicks++
		case "buy", "purchase":
			features.TotalPurchases++
		}

		// 统计活跃时段
		hour := b.Timestamp.Hour()
		hourCounts[hour]++

		// 统计活跃天数
		day := b.Timestamp.Format("2006-01-02")
		activeDays[day] = true

		// 统计会话时长（从上下文中获取）
		if b.Context != nil {
			if durationStr, ok := b.Context["duration"]; ok {
				// 简单处理，假设 duration 是秒数字符串
				var duration float64
				if _, err := time.ParseDuration(durationStr + "s"); err == nil {
					totalDuration += duration
					sessionCount++
				}
			}
		}
	}

	features.ActiveDays = len(activeDays)

	// 计算平均会话时长
	if sessionCount > 0 {
		features.AvgSessionTime = totalDuration / float64(sessionCount)
	}

	// 获取最后活跃时段
	if len(behaviors) > 0 {
		features.LastActiveHour = behaviors[0].Timestamp.Hour()
	}

	// 找出偏好时段（出现次数 > 平均值的时段）
	if len(behaviors) > 0 {
		avgCount := float64(len(behaviors)) / 24.0
		for hour, count := range hourCounts {
			if float64(count) > avgCount {
				features.PreferredHours = append(features.PreferredHours, hour)
			}
		}
		// 排序偏好时段
		sort.Ints(features.PreferredHours)
	}

	return features
}

// extractPreferenceFeatures 从行为列表提取偏好特征
func (e *FeatureExtractor) extractPreferenceFeatures(ctx context.Context, behaviors []*interfaces.UserBehavior) PreferenceFeatures {
	features := PreferenceFeatures{
		TopCategories: make([]CategoryScore, 0),
		TopTags:       make([]TagScore, 0),
		PriceRange:    [2]float64{0, 0},
		ContentLength: "medium",
	}

	if len(behaviors) == 0 {
		return features
	}

	// 统计类目偏好
	categoryScores := make(map[string]float64)
	tagScores := make(map[string]float64)
	var minPrice, maxPrice float64
	priceCount := 0

	// 收集所有物品 ID
	itemIDs := make([]string, 0, len(behaviors))
	for _, b := range behaviors {
		itemIDs = append(itemIDs, b.ItemID)
	}

	// 批量获取物品信息
	items, err := e.itemRepo.GetByIDs(ctx, itemIDs)
	if err != nil {
		return features
	}

	// 构建物品映射
	itemMap := make(map[string]*interfaces.Item)
	for _, item := range items {
		itemMap[item.ID] = item
	}

	// 遍历行为计算偏好
	for _, b := range behaviors {
		item, exists := itemMap[b.ItemID]
		if !exists {
			continue
		}

		// 根据行为类型赋予不同权重
		weight := e.getActionWeight(b.Action)

		// 统计类目
		if item.Category != "" {
			categoryScores[item.Category] += weight
		}

		// 统计标签
		for _, tag := range item.Tags {
			tagScores[tag] += weight
		}

		// 统计价格区间（从元数据中获取）
		if item.Metadata != nil {
			if priceVal, ok := item.Metadata["price"]; ok {
				if price, ok := priceVal.(float64); ok {
					if priceCount == 0 {
						minPrice = price
						maxPrice = price
					} else {
						if price < minPrice {
							minPrice = price
						}
						if price > maxPrice {
							maxPrice = price
						}
					}
					priceCount++
				}
			}
		}
	}

	// 转换类目偏好为排序列表
	features.TopCategories = e.convertToSortedCategories(categoryScores, 10)

	// 转换标签偏好为排序列表
	features.TopTags = e.convertToSortedTags(tagScores, 20)

	// 设置价格区间
	features.PriceRange = [2]float64{minPrice, maxPrice}

	return features
}

// getActionWeight 获取行为权重
func (e *FeatureExtractor) getActionWeight(action string) float64 {
	weights := map[string]float64{
		"view":     1.0,
		"click":    2.0,
		"like":     3.0,
		"favorite": 4.0,
		"share":    4.0,
		"comment":  3.5,
		"buy":      5.0,
		"purchase": 5.0,
		"rate":     4.0,
	}

	if weight, ok := weights[action]; ok {
		return weight
	}
	return 1.0
}

// convertToSortedCategories 将类目分数转换为排序列表
func (e *FeatureExtractor) convertToSortedCategories(scores map[string]float64, limit int) []CategoryScore {
	result := make([]CategoryScore, 0, len(scores))
	for category, score := range scores {
		result = append(result, CategoryScore{
			Category: category,
			Score:    score,
		})
	}

	// 按分数降序排序
	sort.Slice(result, func(i, j int) bool {
		return result[i].Score > result[j].Score
	})

	// 限制返回数量
	if len(result) > limit {
		result = result[:limit]
	}

	return result
}

// convertToSortedTags 将标签分数转换为排序列表
func (e *FeatureExtractor) convertToSortedTags(scores map[string]float64, limit int) []TagScore {
	result := make([]TagScore, 0, len(scores))
	for tag, score := range scores {
		result = append(result, TagScore{
			Tag:   tag,
			Score: score,
		})
	}

	// 按分数降序排序
	sort.Slice(result, func(i, j int) bool {
		return result[i].Score > result[j].Score
	})

	// 限制返回数量
	if len(result) > limit {
		result = result[:limit]
	}

	return result
}

// =============================================================================
// 物品特征提取
// =============================================================================

// ExtractItemFeatures 提取物品特征
// 从物品基本信息和统计数据中提取内容特征和统计特征
func (e *FeatureExtractor) ExtractItemFeatures(ctx context.Context, itemID string) (*InternalItemFeatures, error) {
	// 获取物品基本信息
	item, err := e.itemRepo.GetByID(ctx, itemID)
	if err != nil {
		return nil, ErrItemNotFound
	}

	// 获取物品统计数据
	stats, err := e.itemRepo.GetStats(ctx, itemID)
	if err != nil {
		// 使用默认统计数据
		stats = &interfaces.ItemStats{}
	}

	// 提取内容特征
	contentFeatures := e.extractContentFeatures(item)

	// 提取统计特征
	statisticFeatures := e.extractStatisticFeatures(stats)

	return &InternalItemFeatures{
		ItemID:      itemID,
		Type:        item.Type,
		Content:     contentFeatures,
		Statistics:  statisticFeatures,
		LastUpdated: time.Now(),
	}, nil
}

// extractContentFeatures 提取内容特征
func (e *FeatureExtractor) extractContentFeatures(item *interfaces.Item) ContentFeatures {
	features := ContentFeatures{
		Category: item.Category,
		Tags:     item.Tags,
	}

	// 从元数据中提取额外信息
	if item.Metadata != nil {
		if subCat, ok := item.Metadata["sub_category"].(string); ok {
			features.SubCategory = subCat
		}
		if price, ok := item.Metadata["price"].(float64); ok {
			features.Price = price
		}
		if duration, ok := item.Metadata["duration"].(float64); ok {
			features.Duration = int(duration)
		}
		if wordCount, ok := item.Metadata["word_count"].(float64); ok {
			features.WordCount = int(wordCount)
		}
		if releaseDate, ok := item.Metadata["release_date"].(string); ok {
			features.ReleaseDate = releaseDate
		}
	}

	return features
}

// extractStatisticFeatures 提取统计特征
func (e *FeatureExtractor) extractStatisticFeatures(stats *interfaces.ItemStats) StatisticFeatures {
	features := StatisticFeatures{
		ViewCount:  stats.ViewCount,
		ClickCount: stats.ClickCount,
		LikeCount:  stats.LikeCount,
		ShareCount: stats.ShareCount,
		AvgRating:  stats.AvgRating,
	}

	// 计算点击率
	features.CTR = e.calculateCTR(stats)

	return features
}

// calculateCTR 计算点击率
func (e *FeatureExtractor) calculateCTR(stats *interfaces.ItemStats) float64 {
	if stats.ViewCount == 0 {
		return 0
	}
	return float64(stats.ClickCount) / float64(stats.ViewCount)
}

// =============================================================================
// 交叉特征提取
// =============================================================================

// ExtractCrossFeatures 提取用户-物品交叉特征
func (e *FeatureExtractor) ExtractCrossFeatures(ctx context.Context, userID, itemID string) (*CrossFeatures, error) {
	// 查询用户对该物品的历史交互
	interactions, err := e.userRepo.GetUserItemInteractions(ctx, userID, itemID)
	if err != nil {
		return nil, err
	}

	cross := &CrossFeatures{
		UserID:       userID,
		ItemID:       itemID,
		Interactions: len(interactions),
	}

	if len(interactions) > 0 {
		// 获取最近的交互记录
		lastInteraction := interactions[0]
		cross.LastAction = lastInteraction.Action
		cross.LastTime = lastInteraction.Timestamp
	}

	return cross, nil
}

// =============================================================================
// 上下文特征提取
// =============================================================================

// ExtractContextFeatures 提取请求上下文特征
func (e *FeatureExtractor) ExtractContextFeatures(ctx context.Context, req *ContextRequest) *ContextFeatures {
	now := time.Now()

	features := &ContextFeatures{
		Timestamp: now,
		Hour:      now.Hour(),
		DayOfWeek: int(now.Weekday()),
		IsWeekend: now.Weekday() == time.Saturday || now.Weekday() == time.Sunday,
	}

	if req != nil {
		features.Device = req.Device
		features.OS = req.OS
		features.Location = req.Location
		features.PageContext = req.PageContext
	}

	return features
}

// =============================================================================
// 批量特征提取
// =============================================================================

// BatchExtractUserFeatures 批量提取用户特征
func (e *FeatureExtractor) BatchExtractUserFeatures(ctx context.Context, userIDs []string) (map[string]*InternalUserFeatures, error) {
	result := make(map[string]*InternalUserFeatures)

	for _, userID := range userIDs {
		features, err := e.ExtractUserFeatures(ctx, userID)
		if err != nil {
			continue // 跳过失败的用户
		}
		result[userID] = features
	}

	return result, nil
}

// BatchExtractItemFeatures 批量提取物品特征
func (e *FeatureExtractor) BatchExtractItemFeatures(ctx context.Context, itemIDs []string) (map[string]*InternalItemFeatures, error) {
	result := make(map[string]*InternalItemFeatures)

	for _, itemID := range itemIDs {
		features, err := e.ExtractItemFeatures(ctx, itemID)
		if err != nil {
			continue // 跳过失败的物品
		}
		result[itemID] = features
	}

	return result, nil
}

