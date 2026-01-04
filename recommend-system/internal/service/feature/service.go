package feature

import (
	"context"
	"time"

	"go.uber.org/zap"

	"recommend-system/internal/interfaces"
	"recommend-system/pkg/logger"
)

// =============================================================================
// 特征服务主体
// =============================================================================

// Service 特征服务
// 实现 interfaces.FeatureService 接口
// 负责特征的提取、缓存、序列化等核心功能
type Service struct {
	extractor *FeatureExtractor // 特征提取器
	store     *FeatureStore     // 特征存储
	userRepo  interfaces.UserRepository
	itemRepo  interfaces.ItemRepository
}

// NewService 创建特征服务实例
func NewService(
	userRepo interfaces.UserRepository,
	itemRepo interfaces.ItemRepository,
	cache interfaces.Cache,
) *Service {
	config := DefaultFeatureCacheConfig()

	return &Service{
		extractor: NewFeatureExtractor(userRepo, itemRepo),
		store:     NewFeatureStoreWithConfig(cache, config),
		userRepo:  userRepo,
		itemRepo:  itemRepo,
	}
}

// NewServiceWithConfig 使用自定义配置创建特征服务实例
func NewServiceWithConfig(
	userRepo interfaces.UserRepository,
	itemRepo interfaces.ItemRepository,
	cache interfaces.Cache,
	config *FeatureCacheConfig,
) *Service {
	return &Service{
		extractor: NewFeatureExtractor(userRepo, itemRepo),
		store:     NewFeatureStoreWithConfig(cache, config),
		userRepo:  userRepo,
		itemRepo:  itemRepo,
	}
}

// =============================================================================
// interfaces.FeatureService 接口实现
// =============================================================================

// GetUserFeatures 获取用户特征（缓存优先）
// 实现 interfaces.FeatureService.GetUserFeatures
func (s *Service) GetUserFeatures(ctx context.Context, userID string) (*interfaces.UserFeatures, error) {
	// 1. 先查缓存
	cached, err := s.store.GetInterfaceUserFeatures(ctx, userID)
	if err == nil && cached != nil {
		logger.Debug("user features cache hit",
			zap.String("user_id", userID))
		return cached, nil
	}

	// 2. 缓存未命中，提取特征
	logger.Debug("user features cache miss, extracting",
		zap.String("user_id", userID))

	internalFeatures, err := s.extractor.ExtractUserFeatures(ctx, userID)
	if err != nil {
		logger.Warn("failed to extract user features",
			zap.String("user_id", userID),
			zap.Error(err))
		return nil, err
	}

	// 3. 转换为接口类型
	features := ConvertToInterfaceUserFeatures(internalFeatures)

	// 4. 保存到缓存（异步）
	go func() {
		if err := s.store.SaveInterfaceUserFeatures(context.Background(), features); err != nil {
			logger.Warn("failed to cache user features",
				zap.String("user_id", userID),
				zap.Error(err))
		}
	}()

	return features, nil
}

// GetItemFeatures 获取物品特征（缓存优先）
// 实现 interfaces.FeatureService.GetItemFeatures
func (s *Service) GetItemFeatures(ctx context.Context, itemID string) (*interfaces.ItemFeatures, error) {
	// 1. 先查缓存
	cached, err := s.store.GetInterfaceItemFeatures(ctx, itemID)
	if err == nil && cached != nil {
		logger.Debug("item features cache hit",
			zap.String("item_id", itemID))
		return cached, nil
	}

	// 2. 缓存未命中，提取特征
	logger.Debug("item features cache miss, extracting",
		zap.String("item_id", itemID))

	internalFeatures, err := s.extractor.ExtractItemFeatures(ctx, itemID)
	if err != nil {
		logger.Warn("failed to extract item features",
			zap.String("item_id", itemID),
			zap.Error(err))
		return nil, err
	}

	// 3. 转换为接口类型
	features := ConvertToInterfaceItemFeatures(internalFeatures)

	// 4. 保存到缓存（异步）
	go func() {
		if err := s.store.SaveInterfaceItemFeatures(context.Background(), features); err != nil {
			logger.Warn("failed to cache item features",
				zap.String("item_id", itemID),
				zap.Error(err))
		}
	}()

	return features, nil
}

// GetFeatureVector 获取完整特征向量（用于模型推理）
// 实现 interfaces.FeatureService.GetFeatureVector
func (s *Service) GetFeatureVector(ctx context.Context, req *interfaces.FeatureVectorRequest) (*interfaces.FeatureVector, error) {
	if req == nil || req.UserID == "" {
		return nil, ErrUserNotFound
	}

	vector := &interfaces.FeatureVector{}

	// 1. 获取用户特征（必需）
	userFeatures, err := s.GetUserFeatures(ctx, req.UserID)
	if err != nil {
		logger.Warn("failed to get user features for vector",
			zap.String("user_id", req.UserID),
			zap.Error(err))
		return nil, err
	}
	vector.UserFeatures = userFeatures

	// 2. 获取物品特征（可选，用于排序场景）
	if req.ItemID != "" {
		itemFeatures, err := s.GetItemFeatures(ctx, req.ItemID)
		if err == nil {
			vector.ItemFeatures = itemFeatures
		}
	}

	// 3. 构建上下文请求
	contextReq := &ContextRequest{}
	if req.Context != nil {
		contextReq.Device = req.Context["device"]
		contextReq.OS = req.Context["os"]
		contextReq.Location = req.Context["location"]
		contextReq.PageContext = req.Context["page_context"]
	}

	// 4. 序列化为 Token IDs（用于 UGT 模型）
	s.serializeToTokens(vector, contextReq)

	return vector, nil
}

// BatchGetFeatureVectors 批量获取特征向量
// 实现 interfaces.FeatureService.BatchGetFeatureVectors
func (s *Service) BatchGetFeatureVectors(ctx context.Context, reqs []*interfaces.FeatureVectorRequest) ([]*interfaces.FeatureVector, error) {
	if len(reqs) == 0 {
		return []*interfaces.FeatureVector{}, nil
	}

	vectors := make([]*interfaces.FeatureVector, 0, len(reqs))

	for _, req := range reqs {
		vector, err := s.GetFeatureVector(ctx, req)
		if err != nil {
			logger.Warn("failed to get feature vector in batch",
				zap.String("user_id", req.UserID),
				zap.Error(err))
			continue
		}
		vectors = append(vectors, vector)
	}

	return vectors, nil
}

// RefreshUserFeatures 刷新用户特征
// 实现 interfaces.FeatureService.RefreshUserFeatures
func (s *Service) RefreshUserFeatures(ctx context.Context, userID string) error {
	// 1. 使缓存失效
	if err := s.store.InvalidateUserFeatures(ctx, userID); err != nil {
		logger.Warn("failed to invalidate user features cache",
			zap.String("user_id", userID),
			zap.Error(err))
	}

	// 2. 重新提取并缓存
	_, err := s.GetUserFeatures(ctx, userID)
	if err != nil {
		logger.Warn("failed to refresh user features",
			zap.String("user_id", userID),
			zap.Error(err))
		return err
	}

	logger.Info("user features refreshed",
		zap.String("user_id", userID))

	return nil
}

// RefreshItemFeatures 刷新物品特征
// 实现 interfaces.FeatureService.RefreshItemFeatures
func (s *Service) RefreshItemFeatures(ctx context.Context, itemID string) error {
	// 1. 使缓存失效
	if err := s.store.InvalidateItemFeatures(ctx, itemID); err != nil {
		logger.Warn("failed to invalidate item features cache",
			zap.String("item_id", itemID),
			zap.Error(err))
	}

	// 2. 重新提取并缓存
	_, err := s.GetItemFeatures(ctx, itemID)
	if err != nil {
		logger.Warn("failed to refresh item features",
			zap.String("item_id", itemID),
			zap.Error(err))
		return err
	}

	logger.Info("item features refreshed",
		zap.String("item_id", itemID))

	return nil
}

// =============================================================================
// Token 序列化
// =============================================================================

// serializeToTokens 将特征向量序列化为 Token IDs
func (s *Service) serializeToTokens(vector *interfaces.FeatureVector, contextReq *ContextRequest) {
	// Token 类型: 0=USER, 1=ITEM, 2=ACTION, 3=CONTEXT
	tokenIDs := make([]int64, 0, 16)
	tokenTypes := make([]int, 0, 16)

	// [CLS] Token
	tokenIDs = append(tokenIDs, TokenIDCLS)
	tokenTypes = append(tokenTypes, TokenTypeContext)

	// 用户特征 Tokens
	if vector.UserFeatures != nil {
		s.addUserTokens(&tokenIDs, &tokenTypes, vector.UserFeatures)
	}

	// 物品特征 Tokens
	if vector.ItemFeatures != nil {
		s.addItemTokens(&tokenIDs, &tokenTypes, vector.ItemFeatures)
	}

	// 上下文 Tokens
	if contextReq != nil {
		s.addContextTokens(&tokenIDs, &tokenTypes, contextReq)
	}

	// [SEP] Token
	tokenIDs = append(tokenIDs, TokenIDSEP)
	tokenTypes = append(tokenTypes, TokenTypeContext)

	// 位置编码
	positions := make([]int, len(tokenIDs))
	for i := range positions {
		positions[i] = i
	}

	vector.TokenIDs = tokenIDs
	vector.TokenTypes = tokenTypes
	vector.Positions = positions
}

// addUserTokens 添加用户特征 Tokens
func (s *Service) addUserTokens(tokenIDs *[]int64, tokenTypes *[]int, features *interfaces.UserFeatures) {
	if features.Demographics == nil {
		return
	}

	// 年龄分桶 Token
	if ageVal, ok := features.Demographics["age"]; ok {
		var age int
		switch v := ageVal.(type) {
		case int:
			age = v
		case float64:
			age = int(v)
		}
		if age > 0 {
			ageToken := s.getAgeToken(age)
			*tokenIDs = append(*tokenIDs, ageToken)
			*tokenTypes = append(*tokenTypes, TokenTypeUser)
		}
	}

	// 性别 Token
	if genderVal, ok := features.Demographics["gender"]; ok {
		if gender, ok := genderVal.(string); ok && gender != "" {
			genderToken := s.getGenderToken(gender)
			*tokenIDs = append(*tokenIDs, genderToken)
			*tokenTypes = append(*tokenTypes, TokenTypeUser)
		}
	}
}

// addItemTokens 添加物品特征 Tokens
func (s *Service) addItemTokens(tokenIDs *[]int64, tokenTypes *[]int, features *interfaces.ItemFeatures) {
	// 添加 Semantic ID (L1, L2, L3)
	for _, level := range features.SemanticID {
		if level > 0 {
			*tokenIDs = append(*tokenIDs, int64(level))
			*tokenTypes = append(*tokenTypes, TokenTypeItem)
		}
	}
}

// addContextTokens 添加上下文特征 Tokens
func (s *Service) addContextTokens(tokenIDs *[]int64, tokenTypes *[]int, req *ContextRequest) {
	// 添加时间段 Token
	now := time.Now()
	hourToken := s.getHourToken(now.Hour())
	*tokenIDs = append(*tokenIDs, hourToken)
	*tokenTypes = append(*tokenTypes, TokenTypeContext)
}

// =============================================================================
// Token 映射辅助函数
// =============================================================================

// getAgeToken 获取年龄分桶 Token
// 年龄分桶: 0-17, 18-25, 26-35, 36-45, 46-55, 56+
func (s *Service) getAgeToken(age int) int64 {
	switch {
	case age < 18:
		return AgeTokenBase + 0
	case age < 26:
		return AgeTokenBase + 1
	case age < 36:
		return AgeTokenBase + 2
	case age < 46:
		return AgeTokenBase + 3
	case age < 56:
		return AgeTokenBase + 4
	default:
		return AgeTokenBase + 5
	}
}

// getGenderToken 获取性别 Token
func (s *Service) getGenderToken(gender string) int64 {
	switch gender {
	case "male", "m", "M":
		return GenderTokenBase + 0
	case "female", "f", "F":
		return GenderTokenBase + 1
	default:
		return GenderTokenBase + 2
	}
}

// getHourToken 获取时间段 Token
// 时间分桶: night(0-6), morning(6-12), afternoon(12-18), evening(18-24)
func (s *Service) getHourToken(hour int) int64 {
	switch {
	case hour < 6:
		return HourTokenBase + 0 // night
	case hour < 12:
		return HourTokenBase + 1 // morning
	case hour < 18:
		return HourTokenBase + 2 // afternoon
	default:
		return HourTokenBase + 3 // evening
	}
}

// =============================================================================
// 扩展方法
// =============================================================================

// GetInternalUserFeatures 获取内部用户特征（带详细结构）
func (s *Service) GetInternalUserFeatures(ctx context.Context, userID string) (*InternalUserFeatures, error) {
	// 先查缓存
	cached, err := s.store.GetUserFeatures(ctx, userID)
	if err == nil && cached != nil {
		return cached, nil
	}

	// 提取特征
	features, err := s.extractor.ExtractUserFeatures(ctx, userID)
	if err != nil {
		return nil, err
	}

	// 保存到缓存
	go func() {
		_ = s.store.SaveUserFeatures(context.Background(), features)
	}()

	return features, nil
}

// GetInternalItemFeatures 获取内部物品特征（带详细结构）
func (s *Service) GetInternalItemFeatures(ctx context.Context, itemID string) (*InternalItemFeatures, error) {
	// 先查缓存
	cached, err := s.store.GetItemFeatures(ctx, itemID)
	if err == nil && cached != nil {
		return cached, nil
	}

	// 提取特征
	features, err := s.extractor.ExtractItemFeatures(ctx, itemID)
	if err != nil {
		return nil, err
	}

	// 保存到缓存
	go func() {
		_ = s.store.SaveItemFeatures(context.Background(), features)
	}()

	return features, nil
}

// GetCrossFeatures 获取交叉特征
func (s *Service) GetCrossFeatures(ctx context.Context, userID, itemID string) (*CrossFeatures, error) {
	return s.extractor.ExtractCrossFeatures(ctx, userID, itemID)
}

// GetContextFeatures 获取上下文特征
func (s *Service) GetContextFeatures(ctx context.Context, req *ContextRequest) *ContextFeatures {
	return s.extractor.ExtractContextFeatures(ctx, req)
}

// BatchRefreshUserFeatures 批量刷新用户特征
func (s *Service) BatchRefreshUserFeatures(ctx context.Context, userIDs []string) error {
	for _, userID := range userIDs {
		if err := s.RefreshUserFeatures(ctx, userID); err != nil {
			logger.Warn("failed to refresh user features in batch",
				zap.String("user_id", userID),
				zap.Error(err))
		}
	}
	return nil
}

// BatchRefreshItemFeatures 批量刷新物品特征
func (s *Service) BatchRefreshItemFeatures(ctx context.Context, itemIDs []string) error {
	for _, itemID := range itemIDs {
		if err := s.RefreshItemFeatures(ctx, itemID); err != nil {
			logger.Warn("failed to refresh item features in batch",
				zap.String("item_id", itemID),
				zap.Error(err))
		}
	}
	return nil
}

// =============================================================================
// 编译时接口检查
// =============================================================================

// 确保 Service 实现了 interfaces.FeatureService 接口
var _ interfaces.FeatureService = (*Service)(nil)

