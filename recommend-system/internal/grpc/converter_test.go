// Package grpc 类型转换器单元测试
package grpc

import (
	"testing"
	"time"

	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	"recommend-system/internal/interfaces"
	itemv1 "recommend-system/proto/item/v1"
	recommendv1 "recommend-system/proto/recommend/v1"
	userv1 "recommend-system/proto/user/v1"
)

// =============================================================================
// 用户转换测试
// =============================================================================

func TestUserToProto(t *testing.T) {
	now := time.Now()
	user := &interfaces.User{
		ID:        "user123",
		Name:      "Test User",
		Email:     "test@example.com",
		Age:       25,
		Gender:    "male",
		Metadata:  map[string]string{"key": "value"},
		CreatedAt: now,
		UpdatedAt: now,
	}

	proto := UserToProto(user)

	if proto.Id != user.ID {
		t.Errorf("expected ID %s, got %s", user.ID, proto.Id)
	}
	if proto.Name != user.Name {
		t.Errorf("expected Name %s, got %s", user.Name, proto.Name)
	}
	if proto.Email != user.Email {
		t.Errorf("expected Email %s, got %s", user.Email, proto.Email)
	}
	if proto.Age != int32(user.Age) {
		t.Errorf("expected Age %d, got %d", user.Age, proto.Age)
	}
	if proto.Gender != user.Gender {
		t.Errorf("expected Gender %s, got %s", user.Gender, proto.Gender)
	}
	if proto.Metadata["key"] != "value" {
		t.Error("expected metadata key to equal value")
	}
}

func TestUserToProtoNil(t *testing.T) {
	proto := UserToProto(nil)
	if proto != nil {
		t.Error("expected nil result for nil input")
	}
}

func TestProtoToUser(t *testing.T) {
	now := time.Now()
	proto := &userv1.User{
		Id:        "user123",
		Name:      "Test User",
		Email:     "test@example.com",
		Age:       25,
		Gender:    "male",
		Metadata:  map[string]string{"key": "value"},
		CreatedAt: timestamppb.New(now),
		UpdatedAt: timestamppb.New(now),
	}

	user := ProtoToUser(proto)

	if user.ID != proto.Id {
		t.Errorf("expected ID %s, got %s", proto.Id, user.ID)
	}
	if user.Name != proto.Name {
		t.Errorf("expected Name %s, got %s", proto.Name, user.Name)
	}
	if user.Email != proto.Email {
		t.Errorf("expected Email %s, got %s", proto.Email, user.Email)
	}
	if user.Age != int(proto.Age) {
		t.Errorf("expected Age %d, got %d", proto.Age, user.Age)
	}
}

func TestProtoToUserNil(t *testing.T) {
	user := ProtoToUser(nil)
	if user != nil {
		t.Error("expected nil result for nil input")
	}
}

func TestUserBehaviorToProto(t *testing.T) {
	now := time.Now()
	behavior := &interfaces.UserBehavior{
		UserID:    "user123",
		ItemID:    "item456",
		Action:    "click",
		Timestamp: now,
		Context:   map[string]string{"device": "mobile"},
	}

	proto := UserBehaviorToProto(behavior)

	if proto.UserId != behavior.UserID {
		t.Errorf("expected UserID %s, got %s", behavior.UserID, proto.UserId)
	}
	if proto.ItemId != behavior.ItemID {
		t.Errorf("expected ItemID %s, got %s", behavior.ItemID, proto.ItemId)
	}
	if proto.Action != behavior.Action {
		t.Errorf("expected Action %s, got %s", behavior.Action, proto.Action)
	}
}

func TestProtoToUserBehavior(t *testing.T) {
	now := time.Now()
	proto := &userv1.UserBehavior{
		UserId:    "user123",
		ItemId:    "item456",
		Action:    "click",
		Timestamp: timestamppb.New(now),
		Context:   map[string]string{"device": "mobile"},
	}

	behavior := ProtoToUserBehavior(proto)

	if behavior.UserID != proto.UserId {
		t.Errorf("expected UserID %s, got %s", proto.UserId, behavior.UserID)
	}
	if behavior.ItemID != proto.ItemId {
		t.Errorf("expected ItemID %s, got %s", proto.ItemId, behavior.ItemID)
	}
}

// =============================================================================
// 物品转换测试
// =============================================================================

func TestItemToProto(t *testing.T) {
	now := time.Now()
	item := &interfaces.Item{
		ID:          "item123",
		Type:        "movie",
		Title:       "Test Movie",
		Description: "A test movie",
		Category:    "action",
		Tags:        []string{"tag1", "tag2"},
		Metadata:    map[string]interface{}{"rating": 8.5},
		Status:      "active",
		CreatedAt:   now,
		UpdatedAt:   now,
	}

	proto := ItemToProto(item)

	if proto.Id != item.ID {
		t.Errorf("expected ID %s, got %s", item.ID, proto.Id)
	}
	if proto.Type != item.Type {
		t.Errorf("expected Type %s, got %s", item.Type, proto.Type)
	}
	if proto.Title != item.Title {
		t.Errorf("expected Title %s, got %s", item.Title, proto.Title)
	}
	if proto.Category != item.Category {
		t.Errorf("expected Category %s, got %s", item.Category, proto.Category)
	}
	if len(proto.Tags) != 2 {
		t.Errorf("expected 2 tags, got %d", len(proto.Tags))
	}
}

func TestItemToProtoNil(t *testing.T) {
	proto := ItemToProto(nil)
	if proto != nil {
		t.Error("expected nil result for nil input")
	}
}

func TestProtoToItem(t *testing.T) {
	now := time.Now()
	metadata, _ := structpb.NewStruct(map[string]interface{}{"rating": 8.5})
	proto := &itemv1.Item{
		Id:          "item123",
		Type:        "movie",
		Title:       "Test Movie",
		Description: "A test movie",
		Category:    "action",
		Tags:        []string{"tag1", "tag2"},
		Metadata:    metadata,
		Status:      "active",
		CreatedAt:   timestamppb.New(now),
		UpdatedAt:   timestamppb.New(now),
	}

	item := ProtoToItem(proto)

	if item.ID != proto.Id {
		t.Errorf("expected ID %s, got %s", proto.Id, item.ID)
	}
	if item.Type != proto.Type {
		t.Errorf("expected Type %s, got %s", proto.Type, item.Type)
	}
	if item.Title != proto.Title {
		t.Errorf("expected Title %s, got %s", proto.Title, item.Title)
	}
}

func TestProtoToItemNil(t *testing.T) {
	item := ProtoToItem(nil)
	if item != nil {
		t.Error("expected nil result for nil input")
	}
}

func TestItemStatsToProto(t *testing.T) {
	stats := &interfaces.ItemStats{
		ItemID:     "item123",
		ViewCount:  1000,
		ClickCount: 100,
		LikeCount:  50,
		ShareCount: 10,
		AvgRating:  4.5,
	}

	proto := ItemStatsToProto(stats)

	if proto.ItemId != stats.ItemID {
		t.Errorf("expected ItemID %s, got %s", stats.ItemID, proto.ItemId)
	}
	if proto.ViewCount != stats.ViewCount {
		t.Errorf("expected ViewCount %d, got %d", stats.ViewCount, proto.ViewCount)
	}
	if proto.AvgRating != stats.AvgRating {
		t.Errorf("expected AvgRating %f, got %f", stats.AvgRating, proto.AvgRating)
	}
}

func TestProtoToItemStats(t *testing.T) {
	proto := &itemv1.ItemStats{
		ItemId:     "item123",
		ViewCount:  1000,
		ClickCount: 100,
		LikeCount:  50,
		ShareCount: 10,
		AvgRating:  4.5,
	}

	stats := ProtoToItemStats(proto)

	if stats.ItemID != proto.ItemId {
		t.Errorf("expected ItemID %s, got %s", proto.ItemId, stats.ItemID)
	}
	if stats.ViewCount != proto.ViewCount {
		t.Errorf("expected ViewCount %d, got %d", proto.ViewCount, stats.ViewCount)
	}
}

// =============================================================================
// 推荐转换测试
// =============================================================================

func TestRecommendationToProto(t *testing.T) {
	rec := &interfaces.Recommendation{
		ItemID:     "item123",
		Score:      0.95,
		Reason:     "Based on your history",
		SemanticID: [3]int{1, 2, 3},
	}

	proto := RecommendationToProto(rec)

	if proto.ItemId != rec.ItemID {
		t.Errorf("expected ItemID %s, got %s", rec.ItemID, proto.ItemId)
	}
	if proto.Score != rec.Score {
		t.Errorf("expected Score %f, got %f", rec.Score, proto.Score)
	}
	if proto.Reason != rec.Reason {
		t.Errorf("expected Reason %s, got %s", rec.Reason, proto.Reason)
	}
	if proto.SemanticId.L1 != 1 || proto.SemanticId.L2 != 2 || proto.SemanticId.L3 != 3 {
		t.Error("semantic ID mismatch")
	}
}

func TestRecommendationToProtoNil(t *testing.T) {
	proto := RecommendationToProto(nil)
	if proto != nil {
		t.Error("expected nil result for nil input")
	}
}

func TestProtoToRecommendation(t *testing.T) {
	proto := &recommendv1.Recommendation{
		ItemId: "item123",
		Score:  0.95,
		Reason: "Based on your history",
		SemanticId: &recommendv1.SemanticID{
			L1: 1,
			L2: 2,
			L3: 3,
		},
	}

	rec := ProtoToRecommendation(proto)

	if rec.ItemID != proto.ItemId {
		t.Errorf("expected ItemID %s, got %s", proto.ItemId, rec.ItemID)
	}
	if rec.Score != proto.Score {
		t.Errorf("expected Score %f, got %f", proto.Score, rec.Score)
	}
	if rec.SemanticID[0] != 1 || rec.SemanticID[1] != 2 || rec.SemanticID[2] != 3 {
		t.Error("semantic ID mismatch")
	}
}

func TestProtoToRecommendationNil(t *testing.T) {
	rec := ProtoToRecommendation(nil)
	if rec != nil {
		t.Error("expected nil result for nil input")
	}
}

func TestProtoToRecommendationNilSemanticID(t *testing.T) {
	proto := &recommendv1.Recommendation{
		ItemId:     "item123",
		Score:      0.95,
		SemanticId: nil,
	}

	rec := ProtoToRecommendation(proto)

	// 应该有默认的零值语义ID
	if rec.SemanticID[0] != 0 || rec.SemanticID[1] != 0 || rec.SemanticID[2] != 0 {
		t.Error("expected zero semantic ID for nil input")
	}
}

func TestFeedbackToProto(t *testing.T) {
	now := time.Now()
	feedback := &interfaces.Feedback{
		UserID:    "user123",
		ItemID:    "item456",
		Action:    "like",
		RequestID: "req789",
		Timestamp: now,
	}

	proto := FeedbackToProto(feedback)

	if proto.UserId != feedback.UserID {
		t.Errorf("expected UserID %s, got %s", feedback.UserID, proto.UserId)
	}
	if proto.ItemId != feedback.ItemID {
		t.Errorf("expected ItemID %s, got %s", feedback.ItemID, proto.ItemId)
	}
	if proto.Action != feedback.Action {
		t.Errorf("expected Action %s, got %s", feedback.Action, proto.Action)
	}
}

func TestProtoToFeedback(t *testing.T) {
	now := time.Now()
	proto := &recommendv1.SubmitFeedbackRequest{
		UserId:    "user123",
		ItemId:    "item456",
		Action:    "like",
		RequestId: "req789",
		Timestamp: timestamppb.New(now),
	}

	feedback := ProtoToFeedback(proto)

	if feedback.UserID != proto.UserId {
		t.Errorf("expected UserID %s, got %s", proto.UserId, feedback.UserID)
	}
	if feedback.ItemID != proto.ItemId {
		t.Errorf("expected ItemID %s, got %s", proto.ItemId, feedback.ItemID)
	}
}

// =============================================================================
// 批量转换测试
// =============================================================================

func TestUsersToProto(t *testing.T) {
	now := time.Now()
	users := []*interfaces.User{
		{ID: "user1", Name: "User 1", CreatedAt: now, UpdatedAt: now},
		{ID: "user2", Name: "User 2", CreatedAt: now, UpdatedAt: now},
	}

	protos := UsersToProto(users)

	if len(protos) != 2 {
		t.Errorf("expected 2 users, got %d", len(protos))
	}
	if protos[0].Id != "user1" {
		t.Errorf("expected first user ID user1, got %s", protos[0].Id)
	}
}

func TestProtoToUsers(t *testing.T) {
	now := time.Now()
	protos := []*userv1.User{
		{Id: "user1", Name: "User 1", CreatedAt: timestamppb.New(now), UpdatedAt: timestamppb.New(now)},
		{Id: "user2", Name: "User 2", CreatedAt: timestamppb.New(now), UpdatedAt: timestamppb.New(now)},
	}

	users := ProtoToUsers(protos)

	if len(users) != 2 {
		t.Errorf("expected 2 users, got %d", len(users))
	}
	if users[0].ID != "user1" {
		t.Errorf("expected first user ID user1, got %s", users[0].ID)
	}
}

func TestItemsToProto(t *testing.T) {
	now := time.Now()
	items := []*interfaces.Item{
		{ID: "item1", Title: "Item 1", CreatedAt: now, UpdatedAt: now},
		{ID: "item2", Title: "Item 2", CreatedAt: now, UpdatedAt: now},
	}

	protos := ItemsToProto(items)

	if len(protos) != 2 {
		t.Errorf("expected 2 items, got %d", len(protos))
	}
}

func TestProtoToItems(t *testing.T) {
	now := time.Now()
	protos := []*itemv1.Item{
		{Id: "item1", Title: "Item 1", CreatedAt: timestamppb.New(now), UpdatedAt: timestamppb.New(now)},
		{Id: "item2", Title: "Item 2", CreatedAt: timestamppb.New(now), UpdatedAt: timestamppb.New(now)},
	}

	items := ProtoToItems(protos)

	if len(items) != 2 {
		t.Errorf("expected 2 items, got %d", len(items))
	}
}

func TestRecommendationsToProto(t *testing.T) {
	recs := []*interfaces.Recommendation{
		{ItemID: "item1", Score: 0.9},
		{ItemID: "item2", Score: 0.8},
	}

	protos := RecommendationsToProto(recs)

	if len(protos) != 2 {
		t.Errorf("expected 2 recommendations, got %d", len(protos))
	}
}

func TestProtoToRecommendations(t *testing.T) {
	protos := []*recommendv1.Recommendation{
		{ItemId: "item1", Score: 0.9},
		{ItemId: "item2", Score: 0.8},
	}

	recs := ProtoToRecommendations(protos)

	if len(recs) != 2 {
		t.Errorf("expected 2 recommendations, got %d", len(recs))
	}
}

// =============================================================================
// 时间转换测试
// =============================================================================

func TestTimeToProto(t *testing.T) {
	now := time.Now()
	proto := TimeToProto(now)

	if proto == nil {
		t.Error("expected non-nil proto timestamp")
	}

	// 验证时间一致
	protoTime := proto.AsTime()
	if !protoTime.Equal(now) {
		t.Errorf("time mismatch: expected %v, got %v", now, protoTime)
	}
}

func TestProtoToTime(t *testing.T) {
	now := time.Now()
	proto := timestamppb.New(now)

	result := ProtoToTime(proto)

	if !result.Equal(now) {
		t.Errorf("time mismatch: expected %v, got %v", now, result)
	}
}

func TestProtoToTimeNil(t *testing.T) {
	result := ProtoToTime(nil)

	if !result.IsZero() {
		t.Error("expected zero time for nil input")
	}
}

// =============================================================================
// 请求/响应转换测试
// =============================================================================

func TestRecommendRequestToProto(t *testing.T) {
	req := &interfaces.RecommendRequest{
		UserID:       "user123",
		Limit:        20,
		ExcludeItems: []string{"item1", "item2"},
		Scene:        "home",
		Context:      map[string]string{"device": "mobile"},
	}

	proto := RecommendRequestToProto(req)

	if proto.UserId != req.UserID {
		t.Errorf("expected UserID %s, got %s", req.UserID, proto.UserId)
	}
	if proto.Limit != int32(req.Limit) {
		t.Errorf("expected Limit %d, got %d", req.Limit, proto.Limit)
	}
	if proto.Scene != req.Scene {
		t.Errorf("expected Scene %s, got %s", req.Scene, proto.Scene)
	}
	if len(proto.ExcludeItems) != 2 {
		t.Errorf("expected 2 exclude items, got %d", len(proto.ExcludeItems))
	}
}

func TestProtoToRecommendRequest(t *testing.T) {
	proto := &recommendv1.GetRecommendationsRequest{
		UserId:       "user123",
		Limit:        20,
		ExcludeItems: []string{"item1", "item2"},
		Scene:        "home",
		Context: &recommendv1.Context{
			Extra: map[string]string{"device": "mobile"},
		},
	}

	req := ProtoToRecommendRequest(proto)

	if req.UserID != proto.UserId {
		t.Errorf("expected UserID %s, got %s", proto.UserId, req.UserID)
	}
	if req.Limit != int(proto.Limit) {
		t.Errorf("expected Limit %d, got %d", proto.Limit, req.Limit)
	}
}

func TestRecommendResponseToProto(t *testing.T) {
	resp := &interfaces.RecommendResponse{
		Recommendations: []*interfaces.Recommendation{
			{ItemID: "item1", Score: 0.9},
		},
		RequestID: "req123",
		Strategy:  "collaborative",
	}

	proto := RecommendResponseToProto(resp)

	if proto.RequestId != resp.RequestID {
		t.Errorf("expected RequestID %s, got %s", resp.RequestID, proto.RequestId)
	}
	if proto.Strategy != resp.Strategy {
		t.Errorf("expected Strategy %s, got %s", resp.Strategy, proto.Strategy)
	}
	if len(proto.Recommendations) != 1 {
		t.Errorf("expected 1 recommendation, got %d", len(proto.Recommendations))
	}
}

func TestProtoToRecommendResponse(t *testing.T) {
	proto := &recommendv1.GetRecommendationsResponse{
		Recommendations: []*recommendv1.Recommendation{
			{ItemId: "item1", Score: 0.9},
		},
		RequestId: "req123",
		Strategy:  "collaborative",
	}

	resp := ProtoToRecommendResponse(proto)

	if resp.RequestID != proto.RequestId {
		t.Errorf("expected RequestID %s, got %s", proto.RequestId, resp.RequestID)
	}
	if len(resp.Recommendations) != 1 {
		t.Errorf("expected 1 recommendation, got %d", len(resp.Recommendations))
	}
}

// =============================================================================
// 基准测试
// =============================================================================

func BenchmarkUserToProto(b *testing.B) {
	now := time.Now()
	user := &interfaces.User{
		ID:        "user123",
		Name:      "Test User",
		Email:     "test@example.com",
		Age:       25,
		Gender:    "male",
		Metadata:  map[string]string{"key": "value"},
		CreatedAt: now,
		UpdatedAt: now,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		UserToProto(user)
	}
}

func BenchmarkProtoToUser(b *testing.B) {
	now := time.Now()
	proto := &userv1.User{
		Id:        "user123",
		Name:      "Test User",
		Email:     "test@example.com",
		Age:       25,
		Gender:    "male",
		Metadata:  map[string]string{"key": "value"},
		CreatedAt: timestamppb.New(now),
		UpdatedAt: timestamppb.New(now),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ProtoToUser(proto)
	}
}

func BenchmarkItemToProto(b *testing.B) {
	now := time.Now()
	item := &interfaces.Item{
		ID:          "item123",
		Type:        "movie",
		Title:       "Test Movie",
		Description: "A test movie",
		Category:    "action",
		Tags:        []string{"tag1", "tag2"},
		Metadata:    map[string]interface{}{"rating": 8.5},
		Status:      "active",
		CreatedAt:   now,
		UpdatedAt:   now,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ItemToProto(item)
	}
}

func BenchmarkRecommendationsToProto(b *testing.B) {
	recs := make([]*interfaces.Recommendation, 100)
	for i := 0; i < 100; i++ {
		recs[i] = &interfaces.Recommendation{
			ItemID:     "item" + string(rune(i)),
			Score:      float32(i) / 100.0,
			SemanticID: [3]int{i % 10, i % 100, i},
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		RecommendationsToProto(recs)
	}
}

