// Package fixtures 提供测试用的固定数据
//
// 本包包含各种测试场景所需的预定义数据，
// 包括用户、物品、行为等实体的测试样本。
package fixtures

import (
	"time"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// 时间常量
// =============================================================================

var (
	// BaseTime 基准时间（用于测试数据的时间戳）
	BaseTime = time.Date(2025, 1, 4, 10, 0, 0, 0, time.UTC)
)

// =============================================================================
// 测试用户数据
// =============================================================================

// TestUsers 预定义测试用户列表
var TestUsers = []*interfaces.User{
	{
		ID:        "user_001",
		Name:      "Alice",
		Email:     "alice@example.com",
		Age:       25,
		Gender:    "female",
		Metadata:  map[string]string{"source": "organic", "level": "premium"},
		CreatedAt: BaseTime.Add(-30 * 24 * time.Hour),
		UpdatedAt: BaseTime,
	},
	{
		ID:        "user_002",
		Name:      "Bob",
		Email:     "bob@example.com",
		Age:       35,
		Gender:    "male",
		Metadata:  map[string]string{"source": "referral", "level": "basic"},
		CreatedAt: BaseTime.Add(-60 * 24 * time.Hour),
		UpdatedAt: BaseTime,
	},
	{
		ID:        "user_003",
		Name:      "Charlie",
		Email:     "charlie@example.com",
		Age:       45,
		Gender:    "male",
		Metadata:  map[string]string{"source": "ads", "level": "premium"},
		CreatedAt: BaseTime.Add(-90 * 24 * time.Hour),
		UpdatedAt: BaseTime,
	},
	{
		ID:        "user_004",
		Name:      "Diana",
		Email:     "diana@example.com",
		Age:       28,
		Gender:    "female",
		Metadata:  map[string]string{"source": "organic", "level": "basic"},
		CreatedAt: BaseTime.Add(-15 * 24 * time.Hour),
		UpdatedAt: BaseTime,
	},
	{
		ID:        "user_new",
		Name:      "NewUser",
		Email:     "newuser@example.com",
		Age:       20,
		Gender:    "unknown",
		Metadata:  map[string]string{},
		CreatedAt: BaseTime,
		UpdatedAt: BaseTime,
	},
}

// GetTestUser 根据 ID 获取测试用户
func GetTestUser(id string) *interfaces.User {
	for _, u := range TestUsers {
		if u.ID == id {
			// 返回副本
			userCopy := *u
			if u.Metadata != nil {
				userCopy.Metadata = make(map[string]string)
				for k, v := range u.Metadata {
					userCopy.Metadata[k] = v
				}
			}
			return &userCopy
		}
	}
	return nil
}

// GetAllTestUsers 获取所有测试用户的副本
func GetAllTestUsers() []*interfaces.User {
	result := make([]*interfaces.User, len(TestUsers))
	for i, u := range TestUsers {
		result[i] = GetTestUser(u.ID)
	}
	return result
}

// =============================================================================
// 测试物品数据
// =============================================================================

// TestItems 预定义测试物品列表
var TestItems = []*interfaces.Item{
	{
		ID:          "item_001",
		Type:        "movie",
		Title:       "The Matrix",
		Description: "A computer hacker learns the truth about reality",
		Category:    "action",
		Tags:        []string{"sci-fi", "action", "classic"},
		Status:      "active",
		CreatedAt:   BaseTime.Add(-365 * 24 * time.Hour),
		UpdatedAt:   BaseTime,
	},
	{
		ID:          "item_002",
		Type:        "movie",
		Title:       "Inception",
		Description: "A thief who steals corporate secrets through dreams",
		Category:    "thriller",
		Tags:        []string{"sci-fi", "thriller", "mind-bending"},
		Status:      "active",
		CreatedAt:   BaseTime.Add(-200 * 24 * time.Hour),
		UpdatedAt:   BaseTime,
	},
	{
		ID:          "item_003",
		Type:        "product",
		Title:       "iPhone 15",
		Description: "Latest Apple smartphone with advanced features",
		Category:    "electronics",
		Tags:        []string{"phone", "apple", "premium"},
		Status:      "active",
		CreatedAt:   BaseTime.Add(-30 * 24 * time.Hour),
		UpdatedAt:   BaseTime,
	},
	{
		ID:          "item_004",
		Type:        "video",
		Title:       "How to Code in Go",
		Description: "A comprehensive tutorial on Go programming",
		Category:    "education",
		Tags:        []string{"programming", "golang", "tutorial"},
		Status:      "active",
		CreatedAt:   BaseTime.Add(-7 * 24 * time.Hour),
		UpdatedAt:   BaseTime,
	},
	{
		ID:          "item_005",
		Type:        "article",
		Title:       "Machine Learning Basics",
		Description: "Introduction to machine learning concepts",
		Category:    "technology",
		Tags:        []string{"ml", "ai", "beginner"},
		Status:      "active",
		CreatedAt:   BaseTime.Add(-14 * 24 * time.Hour),
		UpdatedAt:   BaseTime,
	},
	{
		ID:          "item_006",
		Type:        "movie",
		Title:       "The Shawshank Redemption",
		Description: "Two imprisoned men bond over years",
		Category:    "drama",
		Tags:        []string{"drama", "classic", "prison"},
		Status:      "active",
		CreatedAt:   BaseTime.Add(-500 * 24 * time.Hour),
		UpdatedAt:   BaseTime,
	},
	{
		ID:          "item_007",
		Type:        "product",
		Title:       "MacBook Pro",
		Description: "Professional laptop for developers",
		Category:    "electronics",
		Tags:        []string{"laptop", "apple", "professional"},
		Status:      "active",
		CreatedAt:   BaseTime.Add(-60 * 24 * time.Hour),
		UpdatedAt:   BaseTime,
	},
	{
		ID:          "item_008",
		Type:        "video",
		Title:       "Kubernetes Tutorial",
		Description: "Learn Kubernetes from scratch",
		Category:    "education",
		Tags:        []string{"kubernetes", "devops", "cloud"},
		Status:      "active",
		CreatedAt:   BaseTime.Add(-21 * 24 * time.Hour),
		UpdatedAt:   BaseTime,
	},
	{
		ID:          "item_inactive",
		Type:        "movie",
		Title:       "Deleted Movie",
		Description: "This movie has been removed",
		Category:    "unknown",
		Tags:        []string{},
		Status:      "inactive",
		CreatedAt:   BaseTime.Add(-100 * 24 * time.Hour),
		UpdatedAt:   BaseTime,
	},
}

// GetTestItem 根据 ID 获取测试物品
func GetTestItem(id string) *interfaces.Item {
	for _, item := range TestItems {
		if item.ID == id {
			// 返回副本
			itemCopy := *item
			if item.Tags != nil {
				itemCopy.Tags = make([]string, len(item.Tags))
				copy(itemCopy.Tags, item.Tags)
			}
			return &itemCopy
		}
	}
	return nil
}

// GetAllTestItems 获取所有测试物品的副本
func GetAllTestItems() []*interfaces.Item {
	result := make([]*interfaces.Item, len(TestItems))
	for i, item := range TestItems {
		result[i] = GetTestItem(item.ID)
	}
	return result
}

// GetTestItemsByType 按类型获取测试物品
func GetTestItemsByType(itemType string) []*interfaces.Item {
	var result []*interfaces.Item
	for _, item := range TestItems {
		if item.Type == itemType {
			result = append(result, GetTestItem(item.ID))
		}
	}
	return result
}

// GetTestItemsByCategory 按类目获取测试物品
func GetTestItemsByCategory(category string) []*interfaces.Item {
	var result []*interfaces.Item
	for _, item := range TestItems {
		if item.Category == category {
			result = append(result, GetTestItem(item.ID))
		}
	}
	return result
}

// =============================================================================
// 测试物品统计数据
// =============================================================================

// TestItemStats 预定义测试物品统计列表
var TestItemStats = []*interfaces.ItemStats{
	{
		ItemID:     "item_001",
		ViewCount:  10000,
		ClickCount: 5000,
		LikeCount:  2000,
		ShareCount: 500,
		AvgRating:  4.8,
	},
	{
		ItemID:     "item_002",
		ViewCount:  8000,
		ClickCount: 4000,
		LikeCount:  1500,
		ShareCount: 300,
		AvgRating:  4.6,
	},
	{
		ItemID:     "item_003",
		ViewCount:  15000,
		ClickCount: 8000,
		LikeCount:  3000,
		ShareCount: 1000,
		AvgRating:  4.5,
	},
	{
		ItemID:     "item_004",
		ViewCount:  5000,
		ClickCount: 2500,
		LikeCount:  1000,
		ShareCount: 200,
		AvgRating:  4.7,
	},
	{
		ItemID:     "item_005",
		ViewCount:  3000,
		ClickCount: 1500,
		LikeCount:  600,
		ShareCount: 100,
		AvgRating:  4.4,
	},
}

// GetTestItemStats 根据物品 ID 获取统计
func GetTestItemStats(itemID string) *interfaces.ItemStats {
	for _, stats := range TestItemStats {
		if stats.ItemID == itemID {
			statsCopy := *stats
			return &statsCopy
		}
	}
	return nil
}

// =============================================================================
// 测试用户行为数据
// =============================================================================

// TestBehaviors 预定义测试行为列表
var TestBehaviors = []*interfaces.UserBehavior{
	// user_001 的行为
	{
		UserID:    "user_001",
		ItemID:    "item_001",
		Action:    "view",
		Timestamp: BaseTime.Add(-7 * 24 * time.Hour),
		Context:   map[string]string{"device": "mobile", "source": "home"},
	},
	{
		UserID:    "user_001",
		ItemID:    "item_001",
		Action:    "like",
		Timestamp: BaseTime.Add(-6 * 24 * time.Hour),
		Context:   map[string]string{"device": "mobile"},
	},
	{
		UserID:    "user_001",
		ItemID:    "item_002",
		Action:    "view",
		Timestamp: BaseTime.Add(-5 * 24 * time.Hour),
		Context:   map[string]string{"device": "desktop", "source": "search"},
	},
	{
		UserID:    "user_001",
		ItemID:    "item_003",
		Action:    "click",
		Timestamp: BaseTime.Add(-4 * 24 * time.Hour),
		Context:   map[string]string{"device": "mobile", "source": "recommend"},
	},
	{
		UserID:    "user_001",
		ItemID:    "item_003",
		Action:    "purchase",
		Timestamp: BaseTime.Add(-3 * 24 * time.Hour),
		Context:   map[string]string{"device": "mobile"},
	},
	// user_002 的行为
	{
		UserID:    "user_002",
		ItemID:    "item_004",
		Action:    "view",
		Timestamp: BaseTime.Add(-10 * 24 * time.Hour),
		Context:   map[string]string{"device": "desktop"},
	},
	{
		UserID:    "user_002",
		ItemID:    "item_005",
		Action:    "view",
		Timestamp: BaseTime.Add(-8 * 24 * time.Hour),
		Context:   map[string]string{"device": "desktop"},
	},
	{
		UserID:    "user_002",
		ItemID:    "item_004",
		Action:    "like",
		Timestamp: BaseTime.Add(-7 * 24 * time.Hour),
		Context:   map[string]string{"device": "desktop"},
	},
	// user_003 的行为
	{
		UserID:    "user_003",
		ItemID:    "item_001",
		Action:    "view",
		Timestamp: BaseTime.Add(-15 * 24 * time.Hour),
		Context:   map[string]string{"device": "tablet"},
	},
	{
		UserID:    "user_003",
		ItemID:    "item_006",
		Action:    "view",
		Timestamp: BaseTime.Add(-14 * 24 * time.Hour),
		Context:   map[string]string{"device": "tablet"},
	},
	{
		UserID:    "user_003",
		ItemID:    "item_006",
		Action:    "like",
		Timestamp: BaseTime.Add(-13 * 24 * time.Hour),
		Context:   map[string]string{"device": "tablet"},
	},
	{
		UserID:    "user_003",
		ItemID:    "item_006",
		Action:    "share",
		Timestamp: BaseTime.Add(-12 * 24 * time.Hour),
		Context:   map[string]string{"device": "tablet", "platform": "twitter"},
	},
}

// GetBehaviorsForUser 获取指定用户的行为列表
func GetBehaviorsForUser(userID string) []*interfaces.UserBehavior {
	var result []*interfaces.UserBehavior
	for _, b := range TestBehaviors {
		if b.UserID == userID {
			bCopy := *b
			if b.Context != nil {
				bCopy.Context = make(map[string]string)
				for k, v := range b.Context {
					bCopy.Context[k] = v
				}
			}
			result = append(result, &bCopy)
		}
	}
	return result
}

// GetBehaviorsForItem 获取指定物品的行为列表
func GetBehaviorsForItem(itemID string) []*interfaces.UserBehavior {
	var result []*interfaces.UserBehavior
	for _, b := range TestBehaviors {
		if b.ItemID == itemID {
			bCopy := *b
			if b.Context != nil {
				bCopy.Context = make(map[string]string)
				for k, v := range b.Context {
					bCopy.Context[k] = v
				}
			}
			result = append(result, &bCopy)
		}
	}
	return result
}

// =============================================================================
// 测试推荐请求数据
// =============================================================================

// TestRecommendRequests 预定义测试推荐请求
var TestRecommendRequests = []*interfaces.RecommendRequest{
	{
		UserID:       "user_001",
		Limit:        10,
		ExcludeItems: []string{"item_001"},
		Scene:        "home",
		Context:      map[string]string{"device": "mobile"},
	},
	{
		UserID:       "user_002",
		Limit:        20,
		ExcludeItems: []string{},
		Scene:        "detail",
		Context:      map[string]string{"device": "desktop", "ref_item": "item_004"},
	},
	{
		UserID:       "user_new",
		Limit:        10,
		ExcludeItems: []string{},
		Scene:        "home",
		Context:      map[string]string{"device": "mobile", "is_new_user": "true"},
	},
}

// =============================================================================
// 测试用户画像数据
// =============================================================================

// TestUserProfiles 预定义测试用户画像
var TestUserProfiles = []*interfaces.UserProfile{
	{
		User:           GetTestUser("user_001"),
		TotalActions:   5,
		PreferredTypes: map[string]int{"view": 2, "like": 1, "click": 1, "purchase": 1},
		ActiveHours:    map[int]int{10: 2, 14: 2, 20: 1},
		LastActive:     BaseTime.Add(-3 * 24 * time.Hour),
	},
	{
		User:           GetTestUser("user_002"),
		TotalActions:   3,
		PreferredTypes: map[string]int{"view": 2, "like": 1},
		ActiveHours:    map[int]int{9: 1, 15: 1, 16: 1},
		LastActive:     BaseTime.Add(-7 * 24 * time.Hour),
	},
}

// GetTestUserProfile 根据用户 ID 获取测试画像
func GetTestUserProfile(userID string) *interfaces.UserProfile {
	for _, profile := range TestUserProfiles {
		if profile.User != nil && profile.User.ID == userID {
			// 返回副本
			profileCopy := *profile
			profileCopy.User = GetTestUser(userID)
			if profile.PreferredTypes != nil {
				profileCopy.PreferredTypes = make(map[string]int)
				for k, v := range profile.PreferredTypes {
					profileCopy.PreferredTypes[k] = v
				}
			}
			if profile.ActiveHours != nil {
				profileCopy.ActiveHours = make(map[int]int)
				for k, v := range profile.ActiveHours {
					profileCopy.ActiveHours[k] = v
				}
			}
			return &profileCopy
		}
	}
	return nil
}

// =============================================================================
// 测试特征数据
// =============================================================================

// TestUserFeatures 预定义测试用户特征
var TestUserFeatures = []*interfaces.UserFeatures{
	{
		UserID: "user_001",
		Demographics: map[string]interface{}{
			"age_range": "25-34",
			"gender":    "female",
			"city":      "Beijing",
		},
		Behavior: map[string]interface{}{
			"total_views":    100,
			"total_clicks":   50,
			"avg_session":    300.0,
		},
		Preferences: map[string]interface{}{
			"categories": []string{"movie", "electronics"},
			"tags":       []string{"sci-fi", "action"},
		},
		Embedding:   generateMockEmbedding(128),
		LastUpdated: BaseTime,
	},
	{
		UserID: "user_002",
		Demographics: map[string]interface{}{
			"age_range": "35-44",
			"gender":    "male",
			"city":      "Shanghai",
		},
		Behavior: map[string]interface{}{
			"total_views":    50,
			"total_clicks":   25,
			"avg_session":    200.0,
		},
		Preferences: map[string]interface{}{
			"categories": []string{"video", "article"},
			"tags":       []string{"programming", "technology"},
		},
		Embedding:   generateMockEmbedding(128),
		LastUpdated: BaseTime,
	},
}

// TestItemFeatures 预定义测试物品特征
var TestItemFeatures = []*interfaces.ItemFeatures{
	{
		ItemID: "item_001",
		Type:   "movie",
		Content: map[string]interface{}{
			"title":    "The Matrix",
			"genre":    "action",
			"year":     1999,
		},
		Statistics: map[string]interface{}{
			"view_count":  10000,
			"like_count":  2000,
			"avg_rating":  4.8,
		},
		Embedding:   generateMockEmbedding(128),
		SemanticID:  [3]int{1, 2, 3},
		LastUpdated: BaseTime,
	},
	{
		ItemID: "item_002",
		Type:   "movie",
		Content: map[string]interface{}{
			"title":    "Inception",
			"genre":    "thriller",
			"year":     2010,
		},
		Statistics: map[string]interface{}{
			"view_count":  8000,
			"like_count":  1500,
			"avg_rating":  4.6,
		},
		Embedding:   generateMockEmbedding(128),
		SemanticID:  [3]int{1, 3, 5},
		LastUpdated: BaseTime,
	},
}

// GetTestUserFeatures 获取测试用户特征
func GetTestUserFeatures(userID string) *interfaces.UserFeatures {
	for _, f := range TestUserFeatures {
		if f.UserID == userID {
			return f
		}
	}
	return nil
}

// GetTestItemFeatures 获取测试物品特征
func GetTestItemFeatures(itemID string) *interfaces.ItemFeatures {
	for _, f := range TestItemFeatures {
		if f.ItemID == itemID {
			return f
		}
	}
	return nil
}

// =============================================================================
// 辅助函数
// =============================================================================

// generateMockEmbedding 生成模拟嵌入向量
func generateMockEmbedding(dim int) []float32 {
	embedding := make([]float32, dim)
	for i := 0; i < dim; i++ {
		embedding[i] = float32(i) / float32(dim)
	}
	return embedding
}

// CreateTestUser 创建自定义测试用户
func CreateTestUser(id, name, email string, age int, gender string) *interfaces.User {
	return &interfaces.User{
		ID:        id,
		Name:      name,
		Email:     email,
		Age:       age,
		Gender:    gender,
		Metadata:  make(map[string]string),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
}

// CreateTestItem 创建自定义测试物品
func CreateTestItem(id, itemType, title, category string, tags []string) *interfaces.Item {
	return &interfaces.Item{
		ID:          id,
		Type:        itemType,
		Title:       title,
		Description: "Test item: " + title,
		Category:    category,
		Tags:        tags,
		Status:      "active",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
}

// CreateTestBehavior 创建自定义测试行为
func CreateTestBehavior(userID, itemID, action string) *interfaces.UserBehavior {
	return &interfaces.UserBehavior{
		UserID:    userID,
		ItemID:    itemID,
		Action:    action,
		Timestamp: time.Now(),
		Context:   make(map[string]string),
	}
}

