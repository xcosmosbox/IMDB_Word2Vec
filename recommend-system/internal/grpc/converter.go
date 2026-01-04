// Package grpc 提供 Proto 消息与 interfaces 类型之间的转换
//
// 该文件实现了 gRPC Proto 消息与 interfaces.go 中定义的
// Go 结构体之间的双向转换，确保服务间通信的类型一致性。
package grpc

import (
	"time"

	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	"recommend-system/internal/interfaces"
	itemv1 "recommend-system/proto/item/v1"
	recommendv1 "recommend-system/proto/recommend/v1"
	userv1 "recommend-system/proto/user/v1"
)

// =============================================================================
// 用户相关转换
// =============================================================================

// UserToProto 将 interfaces.User 转换为 Proto User
func UserToProto(u *interfaces.User) *userv1.User {
	if u == nil {
		return nil
	}
	return &userv1.User{
		Id:        u.ID,
		Name:      u.Name,
		Email:     u.Email,
		Age:       int32(u.Age),
		Gender:    u.Gender,
		Metadata:  u.Metadata,
		CreatedAt: timestamppb.New(u.CreatedAt),
		UpdatedAt: timestamppb.New(u.UpdatedAt),
	}
}

// ProtoToUser 将 Proto User 转换为 interfaces.User
func ProtoToUser(u *userv1.User) *interfaces.User {
	if u == nil {
		return nil
	}
	return &interfaces.User{
		ID:        u.Id,
		Name:      u.Name,
		Email:     u.Email,
		Age:       int(u.Age),
		Gender:    u.Gender,
		Metadata:  u.Metadata,
		CreatedAt: u.CreatedAt.AsTime(),
		UpdatedAt: u.UpdatedAt.AsTime(),
	}
}

// UserBehaviorToProto 将 interfaces.UserBehavior 转换为 Proto UserBehavior
func UserBehaviorToProto(b *interfaces.UserBehavior) *userv1.UserBehavior {
	if b == nil {
		return nil
	}
	return &userv1.UserBehavior{
		UserId:    b.UserID,
		ItemId:    b.ItemID,
		Action:    b.Action,
		Timestamp: timestamppb.New(b.Timestamp),
		Context:   b.Context,
	}
}

// ProtoToUserBehavior 将 Proto UserBehavior 转换为 interfaces.UserBehavior
func ProtoToUserBehavior(b *userv1.UserBehavior) *interfaces.UserBehavior {
	if b == nil {
		return nil
	}
	return &interfaces.UserBehavior{
		UserID:    b.UserId,
		ItemID:    b.ItemId,
		Action:    b.Action,
		Timestamp: b.Timestamp.AsTime(),
		Context:   b.Context,
	}
}

// UserProfileToProto 将 interfaces.UserProfile 转换为 Proto UserProfile
func UserProfileToProto(p *interfaces.UserProfile) *userv1.UserProfile {
	if p == nil {
		return nil
	}

	// 转换偏好类目
	categories := make([]*userv1.CategoryScore, 0, len(p.PreferredTypes))
	for cat, count := range p.PreferredTypes {
		categories = append(categories, &userv1.CategoryScore{
			Category: cat,
			Score:    float32(count),
			Count:    int32(count),
		})
	}

	// 转换活跃时段
	activeHours := make([]int32, 24)
	for hour, count := range p.ActiveHours {
		if hour >= 0 && hour < 24 {
			activeHours[hour] = int32(count)
		}
	}

	return &userv1.UserProfile{
		User:                UserToProto(p.User),
		TotalActions:        int32(p.TotalActions),
		PreferredCategories: categories,
		ActiveHours:         activeHours,
		LastActive:          timestamppb.New(p.LastActive),
	}
}

// ProtoToUserProfile 将 Proto UserProfile 转换为 interfaces.UserProfile
func ProtoToUserProfile(p *userv1.UserProfile) *interfaces.UserProfile {
	if p == nil {
		return nil
	}

	// 转换偏好类目
	preferredTypes := make(map[string]int)
	for _, cat := range p.PreferredCategories {
		preferredTypes[cat.Category] = int(cat.Count)
	}

	// 转换活跃时段
	activeHours := make(map[int]int)
	for hour, count := range p.ActiveHours {
		if count > 0 {
			activeHours[hour] = int(count)
		}
	}

	return &interfaces.UserProfile{
		User:           ProtoToUser(p.User),
		TotalActions:   int(p.TotalActions),
		PreferredTypes: preferredTypes,
		ActiveHours:    activeHours,
		LastActive:     p.LastActive.AsTime(),
	}
}

// CreateUserRequestToProto 将 interfaces.CreateUserRequest 转换为 Proto
func CreateUserRequestToProto(req *interfaces.CreateUserRequest) *userv1.CreateUserRequest {
	if req == nil {
		return nil
	}
	return &userv1.CreateUserRequest{
		Name:   req.Name,
		Email:  req.Email,
		Age:    int32(req.Age),
		Gender: req.Gender,
	}
}

// ProtoToCreateUserRequest 将 Proto 转换为 interfaces.CreateUserRequest
func ProtoToCreateUserRequest(req *userv1.CreateUserRequest) *interfaces.CreateUserRequest {
	if req == nil {
		return nil
	}
	return &interfaces.CreateUserRequest{
		Name:   req.Name,
		Email:  req.Email,
		Age:    int(req.Age),
		Gender: req.Gender,
	}
}

// =============================================================================
// 物品相关转换
// =============================================================================

// ItemToProto 将 interfaces.Item 转换为 Proto Item
func ItemToProto(item *interfaces.Item) *itemv1.Item {
	if item == nil {
		return nil
	}

	// 转换 metadata
	var metadata *structpb.Struct
	if item.Metadata != nil {
		metadata, _ = structpb.NewStruct(item.Metadata)
	}

	return &itemv1.Item{
		Id:          item.ID,
		Type:        item.Type,
		Title:       item.Title,
		Description: item.Description,
		Category:    item.Category,
		Tags:        item.Tags,
		Metadata:    metadata,
		Status:      item.Status,
		CreatedAt:   timestamppb.New(item.CreatedAt),
		UpdatedAt:   timestamppb.New(item.UpdatedAt),
	}
}

// ProtoToItem 将 Proto Item 转换为 interfaces.Item
func ProtoToItem(item *itemv1.Item) *interfaces.Item {
	if item == nil {
		return nil
	}

	// 转换 metadata
	var metadata map[string]interface{}
	if item.Metadata != nil {
		metadata = item.Metadata.AsMap()
	}

	return &interfaces.Item{
		ID:          item.Id,
		Type:        item.Type,
		Title:       item.Title,
		Description: item.Description,
		Category:    item.Category,
		Tags:        item.Tags,
		Metadata:    metadata,
		Status:      item.Status,
		CreatedAt:   item.CreatedAt.AsTime(),
		UpdatedAt:   item.UpdatedAt.AsTime(),
	}
}

// ItemStatsToProto 将 interfaces.ItemStats 转换为 Proto ItemStats
func ItemStatsToProto(stats *interfaces.ItemStats) *itemv1.ItemStats {
	if stats == nil {
		return nil
	}
	return &itemv1.ItemStats{
		ItemId:     stats.ItemID,
		ViewCount:  stats.ViewCount,
		ClickCount: stats.ClickCount,
		LikeCount:  stats.LikeCount,
		ShareCount: stats.ShareCount,
		AvgRating:  stats.AvgRating,
	}
}

// ProtoToItemStats 将 Proto ItemStats 转换为 interfaces.ItemStats
func ProtoToItemStats(stats *itemv1.ItemStats) *interfaces.ItemStats {
	if stats == nil {
		return nil
	}
	return &interfaces.ItemStats{
		ItemID:     stats.ItemId,
		ViewCount:  stats.ViewCount,
		ClickCount: stats.ClickCount,
		LikeCount:  stats.LikeCount,
		ShareCount: stats.ShareCount,
		AvgRating:  stats.AvgRating,
	}
}

// SimilarItemToProto 将 interfaces.SimilarItem 转换为 Proto SimilarItem
func SimilarItemToProto(s *interfaces.SimilarItem) *itemv1.SimilarItem {
	if s == nil {
		return nil
	}
	return &itemv1.SimilarItem{
		Item:  ItemToProto(s.Item),
		Score: s.Score,
	}
}

// ProtoToSimilarItem 将 Proto SimilarItem 转换为 interfaces.SimilarItem
func ProtoToSimilarItem(s *itemv1.SimilarItem) *interfaces.SimilarItem {
	if s == nil {
		return nil
	}
	return &interfaces.SimilarItem{
		Item:  ProtoToItem(s.Item),
		Score: s.Score,
	}
}

// CreateItemRequestToProto 将 interfaces.CreateItemRequest 转换为 Proto
func CreateItemRequestToProto(req *interfaces.CreateItemRequest) *itemv1.CreateItemRequest {
	if req == nil {
		return nil
	}

	var metadata *structpb.Struct
	if req.Metadata != nil {
		metadata, _ = structpb.NewStruct(req.Metadata)
	}

	return &itemv1.CreateItemRequest{
		Type:        req.Type,
		Title:       req.Title,
		Description: req.Description,
		Category:    req.Category,
		Tags:        req.Tags,
		Metadata:    metadata,
		Embedding:   req.Embedding,
	}
}

// ProtoToCreateItemRequest 将 Proto 转换为 interfaces.CreateItemRequest
func ProtoToCreateItemRequest(req *itemv1.CreateItemRequest) *interfaces.CreateItemRequest {
	if req == nil {
		return nil
	}

	var metadata map[string]interface{}
	if req.Metadata != nil {
		metadata = req.Metadata.AsMap()
	}

	return &interfaces.CreateItemRequest{
		Type:        req.Type,
		Title:       req.Title,
		Description: req.Description,
		Category:    req.Category,
		Tags:        req.Tags,
		Metadata:    metadata,
		Embedding:   req.Embedding,
	}
}

// ListItemsRequestToProto 将 interfaces.ListItemsRequest 转换为 Proto
func ListItemsRequestToProto(req *interfaces.ListItemsRequest) *itemv1.ListItemsRequest {
	if req == nil {
		return nil
	}
	return &itemv1.ListItemsRequest{
		Type:     req.Type,
		Category: req.Category,
		Page:     int32(req.Page),
		PageSize: int32(req.PageSize),
	}
}

// ProtoToListItemsRequest 将 Proto 转换为 interfaces.ListItemsRequest
func ProtoToListItemsRequest(req *itemv1.ListItemsRequest) *interfaces.ListItemsRequest {
	if req == nil {
		return nil
	}
	return &interfaces.ListItemsRequest{
		Type:     req.Type,
		Category: req.Category,
		Page:     int(req.Page),
		PageSize: int(req.PageSize),
	}
}

// =============================================================================
// 推荐相关转换
// =============================================================================

// RecommendationToProto 将 interfaces.Recommendation 转换为 Proto Recommendation
func RecommendationToProto(r *interfaces.Recommendation) *recommendv1.Recommendation {
	if r == nil {
		return nil
	}
	return &recommendv1.Recommendation{
		ItemId: r.ItemID,
		Score:  r.Score,
		Reason: r.Reason,
		SemanticId: &recommendv1.SemanticID{
			L1: int32(r.SemanticID[0]),
			L2: int32(r.SemanticID[1]),
			L3: int32(r.SemanticID[2]),
		},
	}
}

// ProtoToRecommendation 将 Proto Recommendation 转换为 interfaces.Recommendation
func ProtoToRecommendation(r *recommendv1.Recommendation) *interfaces.Recommendation {
	if r == nil {
		return nil
	}
	semanticID := [3]int{}
	if r.SemanticId != nil {
		semanticID = [3]int{
			int(r.SemanticId.L1),
			int(r.SemanticId.L2),
			int(r.SemanticId.L3),
		}
	}
	return &interfaces.Recommendation{
		ItemID:     r.ItemId,
		Score:      r.Score,
		Reason:     r.Reason,
		SemanticID: semanticID,
	}
}

// RecommendRequestToProto 将 interfaces.RecommendRequest 转换为 Proto
func RecommendRequestToProto(req *interfaces.RecommendRequest) *recommendv1.GetRecommendationsRequest {
	if req == nil {
		return nil
	}

	var ctx *recommendv1.Context
	if req.Context != nil {
		ctx = &recommendv1.Context{
			Extra: req.Context,
		}
	}

	return &recommendv1.GetRecommendationsRequest{
		UserId:       req.UserID,
		Limit:        int32(req.Limit),
		Context:      ctx,
		ExcludeItems: req.ExcludeItems,
		Scene:        req.Scene,
	}
}

// ProtoToRecommendRequest 将 Proto 转换为 interfaces.RecommendRequest
func ProtoToRecommendRequest(req *recommendv1.GetRecommendationsRequest) *interfaces.RecommendRequest {
	if req == nil {
		return nil
	}

	var context map[string]string
	if req.Context != nil {
		context = req.Context.Extra
	}

	return &interfaces.RecommendRequest{
		UserID:       req.UserId,
		Limit:        int(req.Limit),
		ExcludeItems: req.ExcludeItems,
		Scene:        req.Scene,
		Context:      context,
	}
}

// RecommendResponseToProto 将 interfaces.RecommendResponse 转换为 Proto
func RecommendResponseToProto(resp *interfaces.RecommendResponse) *recommendv1.GetRecommendationsResponse {
	if resp == nil {
		return nil
	}

	recommendations := make([]*recommendv1.Recommendation, len(resp.Recommendations))
	for i, r := range resp.Recommendations {
		recommendations[i] = RecommendationToProto(r)
	}

	return &recommendv1.GetRecommendationsResponse{
		Recommendations: recommendations,
		RequestId:       resp.RequestID,
		Strategy:        resp.Strategy,
	}
}

// ProtoToRecommendResponse 将 Proto 转换为 interfaces.RecommendResponse
func ProtoToRecommendResponse(resp *recommendv1.GetRecommendationsResponse) *interfaces.RecommendResponse {
	if resp == nil {
		return nil
	}

	recommendations := make([]*interfaces.Recommendation, len(resp.Recommendations))
	for i, r := range resp.Recommendations {
		recommendations[i] = ProtoToRecommendation(r)
	}

	return &interfaces.RecommendResponse{
		Recommendations: recommendations,
		RequestID:       resp.RequestId,
		Strategy:        resp.Strategy,
	}
}

// FeedbackToProto 将 interfaces.Feedback 转换为 Proto SubmitFeedbackRequest
func FeedbackToProto(f *interfaces.Feedback) *recommendv1.SubmitFeedbackRequest {
	if f == nil {
		return nil
	}
	return &recommendv1.SubmitFeedbackRequest{
		UserId:    f.UserID,
		ItemId:    f.ItemID,
		Action:    f.Action,
		RequestId: f.RequestID,
		Timestamp: timestamppb.New(f.Timestamp),
	}
}

// ProtoToFeedback 将 Proto SubmitFeedbackRequest 转换为 interfaces.Feedback
func ProtoToFeedback(req *recommendv1.SubmitFeedbackRequest) *interfaces.Feedback {
	if req == nil {
		return nil
	}
	return &interfaces.Feedback{
		UserID:    req.UserId,
		ItemID:    req.ItemId,
		Action:    req.Action,
		RequestID: req.RequestId,
		Timestamp: req.Timestamp.AsTime(),
	}
}

// =============================================================================
// 批量转换辅助函数
// =============================================================================

// UsersToProto 批量转换用户
func UsersToProto(users []*interfaces.User) []*userv1.User {
	result := make([]*userv1.User, len(users))
	for i, u := range users {
		result[i] = UserToProto(u)
	}
	return result
}

// ProtoToUsers 批量转换用户
func ProtoToUsers(users []*userv1.User) []*interfaces.User {
	result := make([]*interfaces.User, len(users))
	for i, u := range users {
		result[i] = ProtoToUser(u)
	}
	return result
}

// ItemsToProto 批量转换物品
func ItemsToProto(items []*interfaces.Item) []*itemv1.Item {
	result := make([]*itemv1.Item, len(items))
	for i, item := range items {
		result[i] = ItemToProto(item)
	}
	return result
}

// ProtoToItems 批量转换物品
func ProtoToItems(items []*itemv1.Item) []*interfaces.Item {
	result := make([]*interfaces.Item, len(items))
	for i, item := range items {
		result[i] = ProtoToItem(item)
	}
	return result
}

// UserBehaviorsToProto 批量转换用户行为
func UserBehaviorsToProto(behaviors []*interfaces.UserBehavior) []*userv1.UserBehavior {
	result := make([]*userv1.UserBehavior, len(behaviors))
	for i, b := range behaviors {
		result[i] = UserBehaviorToProto(b)
	}
	return result
}

// ProtoToUserBehaviors 批量转换用户行为
func ProtoToUserBehaviors(behaviors []*userv1.UserBehavior) []*interfaces.UserBehavior {
	result := make([]*interfaces.UserBehavior, len(behaviors))
	for i, b := range behaviors {
		result[i] = ProtoToUserBehavior(b)
	}
	return result
}

// RecommendationsToProto 批量转换推荐
func RecommendationsToProto(recs []*interfaces.Recommendation) []*recommendv1.Recommendation {
	result := make([]*recommendv1.Recommendation, len(recs))
	for i, r := range recs {
		result[i] = RecommendationToProto(r)
	}
	return result
}

// ProtoToRecommendations 批量转换推荐
func ProtoToRecommendations(recs []*recommendv1.Recommendation) []*interfaces.Recommendation {
	result := make([]*interfaces.Recommendation, len(recs))
	for i, r := range recs {
		result[i] = ProtoToRecommendation(r)
	}
	return result
}

// =============================================================================
// 时间转换辅助函数
// =============================================================================

// TimeToProto 将 time.Time 转换为 Protobuf Timestamp
func TimeToProto(t time.Time) *timestamppb.Timestamp {
	return timestamppb.New(t)
}

// ProtoToTime 将 Protobuf Timestamp 转换为 time.Time
func ProtoToTime(ts *timestamppb.Timestamp) time.Time {
	if ts == nil {
		return time.Time{}
	}
	return ts.AsTime()
}

