package feature

import (
	"testing"
	"time"
)

// =============================================================================
// 类型定义测试
// =============================================================================

func TestFeatureType(t *testing.T) {
	tests := []struct {
		name     string
		ft       FeatureType
		expected string
	}{
		{"User type", FeatureTypeUser, "user"},
		{"Item type", FeatureTypeItem, "item"},
		{"Cross type", FeatureTypeCross, "cross"},
		{"Context type", FeatureTypeContext, "context"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if string(tt.ft) != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, string(tt.ft))
			}
		})
	}
}

func TestDemographicFeatures(t *testing.T) {
	demo := DemographicFeatures{
		Age:      25,
		Gender:   "male",
		Location: "Beijing",
		Device:   "mobile",
	}

	if demo.Age != 25 {
		t.Errorf("expected age 25, got %d", demo.Age)
	}
	if demo.Gender != "male" {
		t.Errorf("expected gender male, got %s", demo.Gender)
	}
	if demo.Location != "Beijing" {
		t.Errorf("expected location Beijing, got %s", demo.Location)
	}
	if demo.Device != "mobile" {
		t.Errorf("expected device mobile, got %s", demo.Device)
	}
}

func TestBehaviorFeatures(t *testing.T) {
	behavior := BehaviorFeatures{
		TotalViews:     100,
		TotalClicks:    50,
		TotalPurchases: 10,
		AvgSessionTime: 120.5,
		ActiveDays:     30,
		LastActiveHour: 14,
		PreferredHours: []int{10, 11, 14, 15, 20, 21},
	}

	if behavior.TotalViews != 100 {
		t.Errorf("expected TotalViews 100, got %d", behavior.TotalViews)
	}
	if behavior.TotalClicks != 50 {
		t.Errorf("expected TotalClicks 50, got %d", behavior.TotalClicks)
	}
	if behavior.TotalPurchases != 10 {
		t.Errorf("expected TotalPurchases 10, got %d", behavior.TotalPurchases)
	}
	if behavior.AvgSessionTime != 120.5 {
		t.Errorf("expected AvgSessionTime 120.5, got %f", behavior.AvgSessionTime)
	}
	if len(behavior.PreferredHours) != 6 {
		t.Errorf("expected 6 preferred hours, got %d", len(behavior.PreferredHours))
	}
}

func TestPreferenceFeatures(t *testing.T) {
	prefs := PreferenceFeatures{
		TopCategories: []CategoryScore{
			{Category: "electronics", Score: 0.8},
			{Category: "books", Score: 0.6},
		},
		TopTags: []TagScore{
			{Tag: "smartphone", Score: 0.9},
			{Tag: "novel", Score: 0.5},
		},
		PriceRange:    [2]float64{100.0, 5000.0},
		ContentLength: "medium",
	}

	if len(prefs.TopCategories) != 2 {
		t.Errorf("expected 2 top categories, got %d", len(prefs.TopCategories))
	}
	if prefs.TopCategories[0].Category != "electronics" {
		t.Errorf("expected first category electronics, got %s", prefs.TopCategories[0].Category)
	}
	if prefs.PriceRange[0] != 100.0 {
		t.Errorf("expected min price 100.0, got %f", prefs.PriceRange[0])
	}
	if prefs.ContentLength != "medium" {
		t.Errorf("expected content length medium, got %s", prefs.ContentLength)
	}
}

func TestContentFeatures(t *testing.T) {
	content := ContentFeatures{
		Category:    "movie",
		SubCategory: "action",
		Tags:        []string{"thriller", "sci-fi"},
		Price:       0,
		Duration:    7200,
		ReleaseDate: "2024-01-15",
	}

	if content.Category != "movie" {
		t.Errorf("expected category movie, got %s", content.Category)
	}
	if content.Duration != 7200 {
		t.Errorf("expected duration 7200, got %d", content.Duration)
	}
	if len(content.Tags) != 2 {
		t.Errorf("expected 2 tags, got %d", len(content.Tags))
	}
}

func TestStatisticFeatures(t *testing.T) {
	stats := StatisticFeatures{
		ViewCount:    10000,
		ClickCount:   5000,
		LikeCount:    2000,
		ShareCount:   500,
		CommentCount: 300,
		AvgRating:    4.5,
		CTR:          0.5,
		CVR:          0.1,
	}

	if stats.ViewCount != 10000 {
		t.Errorf("expected view count 10000, got %d", stats.ViewCount)
	}
	if stats.CTR != 0.5 {
		t.Errorf("expected CTR 0.5, got %f", stats.CTR)
	}
	if stats.AvgRating != 4.5 {
		t.Errorf("expected avg rating 4.5, got %f", stats.AvgRating)
	}
}

func TestCrossFeatures(t *testing.T) {
	now := time.Now()
	cross := CrossFeatures{
		UserID:       "user123",
		ItemID:       "item456",
		Interactions: 5,
		LastAction:   "click",
		LastTime:     now,
		Similarity:   0.85,
	}

	if cross.UserID != "user123" {
		t.Errorf("expected user ID user123, got %s", cross.UserID)
	}
	if cross.ItemID != "item456" {
		t.Errorf("expected item ID item456, got %s", cross.ItemID)
	}
	if cross.Interactions != 5 {
		t.Errorf("expected 5 interactions, got %d", cross.Interactions)
	}
	if cross.Similarity != 0.85 {
		t.Errorf("expected similarity 0.85, got %f", cross.Similarity)
	}
}

func TestContextFeatures(t *testing.T) {
	now := time.Now()
	ctx := ContextFeatures{
		Timestamp:   now,
		Hour:        14,
		DayOfWeek:   3, // Wednesday
		IsWeekend:   false,
		Device:      "mobile",
		OS:          "iOS",
		Location:    "Shanghai",
		PageContext: "home",
	}

	if ctx.Hour != 14 {
		t.Errorf("expected hour 14, got %d", ctx.Hour)
	}
	if ctx.IsWeekend {
		t.Errorf("expected not weekend")
	}
	if ctx.PageContext != "home" {
		t.Errorf("expected page context home, got %s", ctx.PageContext)
	}
}

func TestInternalUserFeatures(t *testing.T) {
	now := time.Now()
	features := InternalUserFeatures{
		UserID: "user123",
		Demographics: DemographicFeatures{
			Age:    30,
			Gender: "female",
		},
		Behavior: BehaviorFeatures{
			TotalViews: 500,
		},
		Preferences: PreferenceFeatures{
			ContentLength: "long",
		},
		Embedding:   []float32{0.1, 0.2, 0.3},
		LastUpdated: now,
	}

	if features.UserID != "user123" {
		t.Errorf("expected user ID user123, got %s", features.UserID)
	}
	if features.Demographics.Age != 30 {
		t.Errorf("expected age 30, got %d", features.Demographics.Age)
	}
	if len(features.Embedding) != 3 {
		t.Errorf("expected 3 embedding dims, got %d", len(features.Embedding))
	}
}

func TestInternalItemFeatures(t *testing.T) {
	now := time.Now()
	features := InternalItemFeatures{
		ItemID: "item456",
		Type:   "movie",
		Content: ContentFeatures{
			Category: "action",
		},
		Statistics: StatisticFeatures{
			ViewCount: 1000,
		},
		Embedding:   []float32{0.4, 0.5, 0.6},
		SemanticID:  [3]int{100, 200, 300},
		LastUpdated: now,
	}

	if features.ItemID != "item456" {
		t.Errorf("expected item ID item456, got %s", features.ItemID)
	}
	if features.SemanticID[0] != 100 {
		t.Errorf("expected semantic L1 100, got %d", features.SemanticID[0])
	}
}

func TestDefaultFeatureCacheConfig(t *testing.T) {
	config := DefaultFeatureCacheConfig()

	if config.UserFeatureTTL != 30*time.Minute {
		t.Errorf("expected user feature TTL 30m, got %v", config.UserFeatureTTL)
	}
	if config.ItemFeatureTTL != 60*time.Minute {
		t.Errorf("expected item feature TTL 60m, got %v", config.ItemFeatureTTL)
	}
}

func TestFeatureError(t *testing.T) {
	err := &FeatureError{
		Code:    "TEST_ERROR",
		Message: "test error message",
	}

	if err.Error() != "test error message" {
		t.Errorf("expected error message 'test error message', got '%s'", err.Error())
	}

	// 测试预定义错误
	if ErrUserNotFound.Code != "USER_NOT_FOUND" {
		t.Errorf("expected code USER_NOT_FOUND, got %s", ErrUserNotFound.Code)
	}
	if ErrItemNotFound.Code != "ITEM_NOT_FOUND" {
		t.Errorf("expected code ITEM_NOT_FOUND, got %s", ErrItemNotFound.Code)
	}
	if ErrFeatureNotFound.Code != "FEATURE_NOT_FOUND" {
		t.Errorf("expected code FEATURE_NOT_FOUND, got %s", ErrFeatureNotFound.Code)
	}
}

func TestTokenConstants(t *testing.T) {
	// 测试 Token 类型常量
	if TokenTypeUser != 0 {
		t.Errorf("expected TokenTypeUser 0, got %d", TokenTypeUser)
	}
	if TokenTypeItem != 1 {
		t.Errorf("expected TokenTypeItem 1, got %d", TokenTypeItem)
	}
	if TokenTypeAction != 2 {
		t.Errorf("expected TokenTypeAction 2, got %d", TokenTypeAction)
	}
	if TokenTypeContext != 3 {
		t.Errorf("expected TokenTypeContext 3, got %d", TokenTypeContext)
	}

	// 测试特殊 Token ID
	if TokenIDCLS != 1 {
		t.Errorf("expected TokenIDCLS 1, got %d", TokenIDCLS)
	}
	if TokenIDSEP != 2 {
		t.Errorf("expected TokenIDSEP 2, got %d", TokenIDSEP)
	}
	if TokenIDPAD != 0 {
		t.Errorf("expected TokenIDPAD 0, got %d", TokenIDPAD)
	}
}

func TestTokenBases(t *testing.T) {
	if AgeTokenBase != 1000 {
		t.Errorf("expected AgeTokenBase 1000, got %d", AgeTokenBase)
	}
	if GenderTokenBase != 1100 {
		t.Errorf("expected GenderTokenBase 1100, got %d", GenderTokenBase)
	}
	if HourTokenBase != 2000 {
		t.Errorf("expected HourTokenBase 2000, got %d", HourTokenBase)
	}
}

