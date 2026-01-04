package feature

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// 特征存储
// =============================================================================

// FeatureStore 特征存储器
// 负责用户特征和物品特征的缓存存取
type FeatureStore struct {
	cache          interfaces.Cache // 缓存接口
	userFeatureTTL time.Duration    // 用户特征缓存过期时间
	itemFeatureTTL time.Duration    // 物品特征缓存过期时间
}

// NewFeatureStore 创建特征存储实例
func NewFeatureStore(cache interfaces.Cache, userTTL, itemTTL time.Duration) *FeatureStore {
	return &FeatureStore{
		cache:          cache,
		userFeatureTTL: userTTL,
		itemFeatureTTL: itemTTL,
	}
}

// NewFeatureStoreWithConfig 使用配置创建特征存储实例
func NewFeatureStoreWithConfig(cache interfaces.Cache, config *FeatureCacheConfig) *FeatureStore {
	if config == nil {
		config = DefaultFeatureCacheConfig()
	}
	return &FeatureStore{
		cache:          cache,
		userFeatureTTL: config.UserFeatureTTL,
		itemFeatureTTL: config.ItemFeatureTTL,
	}
}

// =============================================================================
// 缓存键生成
// =============================================================================

// userFeatureKey 生成用户特征缓存键
func userFeatureKey(userID string) string {
	return fmt.Sprintf("feature:user:%s", userID)
}

// itemFeatureKey 生成物品特征缓存键
func itemFeatureKey(itemID string) string {
	return fmt.Sprintf("feature:item:%s", itemID)
}

// crossFeatureKey 生成交叉特征缓存键
func crossFeatureKey(userID, itemID string) string {
	return fmt.Sprintf("feature:cross:%s:%s", userID, itemID)
}

// =============================================================================
// 用户特征存取
// =============================================================================

// SaveUserFeatures 保存用户特征到缓存
func (s *FeatureStore) SaveUserFeatures(ctx context.Context, features *InternalUserFeatures) error {
	if features == nil {
		return fmt.Errorf("features cannot be nil")
	}

	key := userFeatureKey(features.UserID)
	return s.cache.Set(ctx, key, features, s.userFeatureTTL)
}

// GetUserFeatures 从缓存获取用户特征
func (s *FeatureStore) GetUserFeatures(ctx context.Context, userID string) (*InternalUserFeatures, error) {
	key := userFeatureKey(userID)

	var features InternalUserFeatures
	if err := s.cache.Get(ctx, key, &features); err != nil {
		return nil, err
	}

	return &features, nil
}

// SaveInterfaceUserFeatures 保存接口类型的用户特征
func (s *FeatureStore) SaveInterfaceUserFeatures(ctx context.Context, features *interfaces.UserFeatures) error {
	if features == nil {
		return fmt.Errorf("features cannot be nil")
	}

	key := userFeatureKey(features.UserID)
	return s.cache.Set(ctx, key, features, s.userFeatureTTL)
}

// GetInterfaceUserFeatures 获取接口类型的用户特征
func (s *FeatureStore) GetInterfaceUserFeatures(ctx context.Context, userID string) (*interfaces.UserFeatures, error) {
	key := userFeatureKey(userID)

	var features interfaces.UserFeatures
	if err := s.cache.Get(ctx, key, &features); err != nil {
		return nil, err
	}

	return &features, nil
}

// =============================================================================
// 物品特征存取
// =============================================================================

// SaveItemFeatures 保存物品特征到缓存
func (s *FeatureStore) SaveItemFeatures(ctx context.Context, features *InternalItemFeatures) error {
	if features == nil {
		return fmt.Errorf("features cannot be nil")
	}

	key := itemFeatureKey(features.ItemID)
	return s.cache.Set(ctx, key, features, s.itemFeatureTTL)
}

// GetItemFeatures 从缓存获取物品特征
func (s *FeatureStore) GetItemFeatures(ctx context.Context, itemID string) (*InternalItemFeatures, error) {
	key := itemFeatureKey(itemID)

	var features InternalItemFeatures
	if err := s.cache.Get(ctx, key, &features); err != nil {
		return nil, err
	}

	return &features, nil
}

// SaveInterfaceItemFeatures 保存接口类型的物品特征
func (s *FeatureStore) SaveInterfaceItemFeatures(ctx context.Context, features *interfaces.ItemFeatures) error {
	if features == nil {
		return fmt.Errorf("features cannot be nil")
	}

	key := itemFeatureKey(features.ItemID)
	return s.cache.Set(ctx, key, features, s.itemFeatureTTL)
}

// GetInterfaceItemFeatures 获取接口类型的物品特征
func (s *FeatureStore) GetInterfaceItemFeatures(ctx context.Context, itemID string) (*interfaces.ItemFeatures, error) {
	key := itemFeatureKey(itemID)

	var features interfaces.ItemFeatures
	if err := s.cache.Get(ctx, key, &features); err != nil {
		return nil, err
	}

	return &features, nil
}

// =============================================================================
// 批量操作
// =============================================================================

// BatchGetUserFeatures 批量获取用户特征
func (s *FeatureStore) BatchGetUserFeatures(ctx context.Context, userIDs []string) (map[string]*InternalUserFeatures, []string, error) {
	found := make(map[string]*InternalUserFeatures)
	missing := make([]string, 0)

	for _, userID := range userIDs {
		features, err := s.GetUserFeatures(ctx, userID)
		if err != nil {
			missing = append(missing, userID)
		} else {
			found[userID] = features
		}
	}

	return found, missing, nil
}

// BatchGetItemFeatures 批量获取物品特征
func (s *FeatureStore) BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]*InternalItemFeatures, []string, error) {
	found := make(map[string]*InternalItemFeatures)
	missing := make([]string, 0)

	for _, itemID := range itemIDs {
		features, err := s.GetItemFeatures(ctx, itemID)
		if err != nil {
			missing = append(missing, itemID)
		} else {
			found[itemID] = features
		}
	}

	return found, missing, nil
}

// BatchSaveUserFeatures 批量保存用户特征
func (s *FeatureStore) BatchSaveUserFeatures(ctx context.Context, features map[string]*InternalUserFeatures) error {
	for _, f := range features {
		if err := s.SaveUserFeatures(ctx, f); err != nil {
			// 记录错误但继续处理其他特征
			continue
		}
	}
	return nil
}

// BatchSaveItemFeatures 批量保存物品特征
func (s *FeatureStore) BatchSaveItemFeatures(ctx context.Context, features map[string]*InternalItemFeatures) error {
	for _, f := range features {
		if err := s.SaveItemFeatures(ctx, f); err != nil {
			// 记录错误但继续处理其他特征
			continue
		}
	}
	return nil
}

// =============================================================================
// 缓存失效
// =============================================================================

// InvalidateUserFeatures 使用户特征缓存失效
func (s *FeatureStore) InvalidateUserFeatures(ctx context.Context, userID string) error {
	key := userFeatureKey(userID)
	return s.cache.Delete(ctx, key)
}

// InvalidateItemFeatures 使物品特征缓存失效
func (s *FeatureStore) InvalidateItemFeatures(ctx context.Context, itemID string) error {
	key := itemFeatureKey(itemID)
	return s.cache.Delete(ctx, key)
}

// InvalidateCrossFeatures 使交叉特征缓存失效
func (s *FeatureStore) InvalidateCrossFeatures(ctx context.Context, userID, itemID string) error {
	key := crossFeatureKey(userID, itemID)
	return s.cache.Delete(ctx, key)
}

// =============================================================================
// 缓存检查
// =============================================================================

// ExistsUserFeatures 检查用户特征缓存是否存在
func (s *FeatureStore) ExistsUserFeatures(ctx context.Context, userID string) (bool, error) {
	key := userFeatureKey(userID)
	return s.cache.Exists(ctx, key)
}

// ExistsItemFeatures 检查物品特征缓存是否存在
func (s *FeatureStore) ExistsItemFeatures(ctx context.Context, itemID string) (bool, error) {
	key := itemFeatureKey(itemID)
	return s.cache.Exists(ctx, key)
}

// =============================================================================
// 特征序列化工具
// =============================================================================

// serializeFeatures 将特征序列化为 JSON 字节
func serializeFeatures(features interface{}) ([]byte, error) {
	return json.Marshal(features)
}

// deserializeUserFeatures 反序列化用户特征
func deserializeUserFeatures(data []byte) (*InternalUserFeatures, error) {
	var features InternalUserFeatures
	if err := json.Unmarshal(data, &features); err != nil {
		return nil, err
	}
	return &features, nil
}

// deserializeItemFeatures 反序列化物品特征
func deserializeItemFeatures(data []byte) (*InternalItemFeatures, error) {
	var features InternalItemFeatures
	if err := json.Unmarshal(data, &features); err != nil {
		return nil, err
	}
	return &features, nil
}

// =============================================================================
// 特征转换工具
// =============================================================================

// ConvertToInterfaceUserFeatures 将内部用户特征转换为接口类型
func ConvertToInterfaceUserFeatures(internal *InternalUserFeatures) *interfaces.UserFeatures {
	if internal == nil {
		return nil
	}

	return &interfaces.UserFeatures{
		UserID: internal.UserID,
		Demographics: map[string]interface{}{
			"age":      internal.Demographics.Age,
			"gender":   internal.Demographics.Gender,
			"location": internal.Demographics.Location,
			"device":   internal.Demographics.Device,
		},
		Behavior: map[string]interface{}{
			"total_views":      internal.Behavior.TotalViews,
			"total_clicks":     internal.Behavior.TotalClicks,
			"total_purchases":  internal.Behavior.TotalPurchases,
			"avg_session_time": internal.Behavior.AvgSessionTime,
			"active_days":      internal.Behavior.ActiveDays,
			"last_active_hour": internal.Behavior.LastActiveHour,
			"preferred_hours":  internal.Behavior.PreferredHours,
		},
		Preferences: map[string]interface{}{
			"top_categories": internal.Preferences.TopCategories,
			"top_tags":       internal.Preferences.TopTags,
			"price_range":    internal.Preferences.PriceRange,
			"content_length": internal.Preferences.ContentLength,
		},
		Embedding:   internal.Embedding,
		LastUpdated: internal.LastUpdated,
	}
}

// ConvertToInterfaceItemFeatures 将内部物品特征转换为接口类型
func ConvertToInterfaceItemFeatures(internal *InternalItemFeatures) *interfaces.ItemFeatures {
	if internal == nil {
		return nil
	}

	return &interfaces.ItemFeatures{
		ItemID: internal.ItemID,
		Type:   internal.Type,
		Content: map[string]interface{}{
			"category":     internal.Content.Category,
			"sub_category": internal.Content.SubCategory,
			"tags":         internal.Content.Tags,
			"price":        internal.Content.Price,
			"duration":     internal.Content.Duration,
			"word_count":   internal.Content.WordCount,
			"release_date": internal.Content.ReleaseDate,
		},
		Statistics: map[string]interface{}{
			"view_count":    internal.Statistics.ViewCount,
			"click_count":   internal.Statistics.ClickCount,
			"like_count":    internal.Statistics.LikeCount,
			"share_count":   internal.Statistics.ShareCount,
			"comment_count": internal.Statistics.CommentCount,
			"avg_rating":    internal.Statistics.AvgRating,
			"ctr":           internal.Statistics.CTR,
			"cvr":           internal.Statistics.CVR,
		},
		Embedding:   internal.Embedding,
		SemanticID:  internal.SemanticID,
		LastUpdated: internal.LastUpdated,
	}
}

// ConvertFromInterfaceUserFeatures 将接口类型转换为内部用户特征
func ConvertFromInterfaceUserFeatures(intf *interfaces.UserFeatures) *InternalUserFeatures {
	if intf == nil {
		return nil
	}

	internal := &InternalUserFeatures{
		UserID:      intf.UserID,
		Embedding:   intf.Embedding,
		LastUpdated: intf.LastUpdated,
	}

	// 解析人口统计特征
	if intf.Demographics != nil {
		if age, ok := intf.Demographics["age"].(float64); ok {
			internal.Demographics.Age = int(age)
		}
		if gender, ok := intf.Demographics["gender"].(string); ok {
			internal.Demographics.Gender = gender
		}
		if location, ok := intf.Demographics["location"].(string); ok {
			internal.Demographics.Location = location
		}
		if device, ok := intf.Demographics["device"].(string); ok {
			internal.Demographics.Device = device
		}
	}

	// 解析行为特征
	if intf.Behavior != nil {
		if totalViews, ok := intf.Behavior["total_views"].(float64); ok {
			internal.Behavior.TotalViews = int64(totalViews)
		}
		if totalClicks, ok := intf.Behavior["total_clicks"].(float64); ok {
			internal.Behavior.TotalClicks = int64(totalClicks)
		}
		if totalPurchases, ok := intf.Behavior["total_purchases"].(float64); ok {
			internal.Behavior.TotalPurchases = int64(totalPurchases)
		}
		if avgSessionTime, ok := intf.Behavior["avg_session_time"].(float64); ok {
			internal.Behavior.AvgSessionTime = avgSessionTime
		}
		if activeDays, ok := intf.Behavior["active_days"].(float64); ok {
			internal.Behavior.ActiveDays = int(activeDays)
		}
		if lastActiveHour, ok := intf.Behavior["last_active_hour"].(float64); ok {
			internal.Behavior.LastActiveHour = int(lastActiveHour)
		}
	}

	return internal
}

// ConvertFromInterfaceItemFeatures 将接口类型转换为内部物品特征
func ConvertFromInterfaceItemFeatures(intf *interfaces.ItemFeatures) *InternalItemFeatures {
	if intf == nil {
		return nil
	}

	internal := &InternalItemFeatures{
		ItemID:      intf.ItemID,
		Type:        intf.Type,
		Embedding:   intf.Embedding,
		SemanticID:  intf.SemanticID,
		LastUpdated: intf.LastUpdated,
	}

	// 解析内容特征
	if intf.Content != nil {
		if category, ok := intf.Content["category"].(string); ok {
			internal.Content.Category = category
		}
		if subCategory, ok := intf.Content["sub_category"].(string); ok {
			internal.Content.SubCategory = subCategory
		}
		if tags, ok := intf.Content["tags"].([]interface{}); ok {
			internal.Content.Tags = make([]string, 0, len(tags))
			for _, t := range tags {
				if tag, ok := t.(string); ok {
					internal.Content.Tags = append(internal.Content.Tags, tag)
				}
			}
		}
		if price, ok := intf.Content["price"].(float64); ok {
			internal.Content.Price = price
		}
		if duration, ok := intf.Content["duration"].(float64); ok {
			internal.Content.Duration = int(duration)
		}
	}

	// 解析统计特征
	if intf.Statistics != nil {
		if viewCount, ok := intf.Statistics["view_count"].(float64); ok {
			internal.Statistics.ViewCount = int64(viewCount)
		}
		if clickCount, ok := intf.Statistics["click_count"].(float64); ok {
			internal.Statistics.ClickCount = int64(clickCount)
		}
		if likeCount, ok := intf.Statistics["like_count"].(float64); ok {
			internal.Statistics.LikeCount = int64(likeCount)
		}
		if shareCount, ok := intf.Statistics["share_count"].(float64); ok {
			internal.Statistics.ShareCount = int64(shareCount)
		}
		if avgRating, ok := intf.Statistics["avg_rating"].(float64); ok {
			internal.Statistics.AvgRating = avgRating
		}
		if ctr, ok := intf.Statistics["ctr"].(float64); ok {
			internal.Statistics.CTR = ctr
		}
	}

	return internal
}

