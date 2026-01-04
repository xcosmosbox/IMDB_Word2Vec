// Package recommend 提供推荐服务核心逻辑
package recommend

import (
	"context"
	"fmt"
	"sort"
	"time"

	"recommend-system/internal/cache"
	"recommend-system/internal/inference"
	"recommend-system/internal/model"
	"recommend-system/internal/repository"
	"recommend-system/pkg/database"
	"recommend-system/pkg/logger"
	"recommend-system/pkg/utils"
	"go.uber.org/zap"
)

// Service 推荐服务
type Service struct {
	userRepo      *repository.UserRepository
	itemRepo      *repository.ItemRepository
	recommendRepo *repository.RecommendRepository
	inferClient   inference.Client
	milvus        *database.MilvusClient
	cache         *cache.MultiLevelCache
	config        *Config
}

// Config 推荐服务配置
type Config struct {
	DefaultSize          int
	MaxSize              int
	CandidateSize        int
	ExposureFilterHours  int
	ColdStartThreshold   int
	ModelVersion         string
	MilvusCollection     string
}

// NewService 创建推荐服务
func NewService(
	userRepo *repository.UserRepository,
	itemRepo *repository.ItemRepository,
	recommendRepo *repository.RecommendRepository,
	inferClient inference.Client,
	milvus *database.MilvusClient,
	cache *cache.MultiLevelCache,
	config *Config,
) *Service {
	return &Service{
		userRepo:      userRepo,
		itemRepo:      itemRepo,
		recommendRepo: recommendRepo,
		inferClient:   inferClient,
		milvus:        milvus,
		cache:         cache,
		config:        config,
	}
}

// Recommend 生成推荐
func (s *Service) Recommend(ctx context.Context, req *model.RecommendRequest) (*model.RecommendResponse, error) {
	timer := utils.NewTimer()
	requestID := utils.GenerateRequestID()

	logger.Info("recommend request",
		zap.String("request_id", requestID),
		zap.String("user_id", req.UserID),
		zap.Int("size", req.Size),
	)

	// 设置默认值
	if req.Size <= 0 {
		req.Size = s.config.DefaultSize
	}
	if req.Size > s.config.MaxSize {
		req.Size = s.config.MaxSize
	}

	// 获取用户行为序列
	userSeq, err := s.userRepo.GetUserSequence(ctx, req.UserID, 100)
	if err != nil {
		logger.Warn("failed to get user sequence", zap.Error(err))
		userSeq = &model.UserSequence{UserID: req.UserID}
	}

	// 判断冷启动
	isColdStart := userSeq.Length < s.config.ColdStartThreshold

	var candidates []model.ItemCandidate

	if isColdStart {
		// 冷启动处理
		candidates, err = s.handleColdStart(ctx, req)
	} else {
		// 正常推荐流程
		candidates, err = s.generateCandidates(ctx, req, userSeq)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to generate candidates: %w", err)
	}

	// 过滤已曝光物品
	candidates = s.filterExposed(ctx, req.UserID, candidates, req.Exclude)

	// 排序和截断
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Score > candidates[j].Score
	})

	if len(candidates) > req.Size {
		candidates = candidates[:req.Size]
	}

	// 构建响应
	items := make([]model.RecommendItem, len(candidates))
	for i, c := range candidates {
		items[i] = model.RecommendItem{
			ItemID:   c.ItemID,
			Score:    c.Score,
			Source:   c.Source,
			Reason:   c.Reason,
			Position: i + 1,
		}
	}

	// 填充物品详情
	items = s.enrichItems(ctx, items)

	response := &model.RecommendResponse{
		RequestID:    requestID,
		UserID:       req.UserID,
		Items:        items,
		GeneratedAt:  time.Now(),
		ModelVersion: s.config.ModelVersion,
	}

	// 调试信息
	if req.Debug {
		response.DebugInfo = &model.DebugInfo{
			TotalTime:      timer.Elapsed(),
			CandidateCount: len(candidates),
		}
	}

	// 异步保存推荐日志
	go s.saveRecommendLog(ctx, response)

	logger.Info("recommend completed",
		zap.String("request_id", requestID),
		zap.Int("result_count", len(items)),
		zap.Int64("latency_ms", timer.Elapsed()),
	)

	return response, nil
}

// generateCandidates 生成候选集
func (s *Service) generateCandidates(ctx context.Context, req *model.RecommendRequest, userSeq *model.UserSequence) ([]model.ItemCandidate, error) {
	var candidates []model.ItemCandidate

	// 1. UGT 模型生成
	ugtCandidates, err := s.generateFromModel(ctx, userSeq, req)
	if err != nil {
		logger.Warn("ugt generation failed, fallback to retrieval", zap.Error(err))
	} else {
		candidates = append(candidates, ugtCandidates...)
	}

	// 2. 向量召回补充
	if len(candidates) < s.config.CandidateSize {
		retrievalCandidates, err := s.retrievalFromMilvus(ctx, req.UserID, s.config.CandidateSize-len(candidates))
		if err != nil {
			logger.Warn("retrieval failed", zap.Error(err))
		} else {
			candidates = append(candidates, retrievalCandidates...)
		}
	}

	// 3. 热门补充
	if len(candidates) < s.config.CandidateSize/2 {
		popularCandidates, err := s.getPopularItems(ctx, req.ItemTypes, 50)
		if err != nil {
			logger.Warn("popular items failed", zap.Error(err))
		} else {
			candidates = append(candidates, popularCandidates...)
		}
	}

	return candidates, nil
}

// generateFromModel UGT 模型生成推荐
func (s *Service) generateFromModel(ctx context.Context, userSeq *model.UserSequence, req *model.RecommendRequest) ([]model.ItemCandidate, error) {
	// 构建模型输入
	tokens := userSeq.ToTokens()

	// 简化的 tokenize (实际需要完整的 tokenizer)
	inputIDs := make([]int64, len(tokens))
	for i := range tokens {
		inputIDs[i] = int64(i + 1) // 简化处理
	}

	attentionMask := make([]int64, len(inputIDs))
	for i := range attentionMask {
		attentionMask[i] = 1
	}

	input := &model.ModelInput{
		UserID:        userSeq.UserID,
		InputIDs:      inputIDs,
		AttentionMask: attentionMask,
		TargetLength:  req.Size * 3, // 生成更多用于排序
	}

	// 调用推理服务
	output, err := s.inferClient.Infer(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("model inference failed: %w", err)
	}

	// 解析生成结果
	candidates := make([]model.ItemCandidate, 0)
	if len(output.GeneratedIDs) > 0 {
		for i, id := range output.GeneratedIDs[0] {
			var score float64 = 1.0
			if len(output.Logits) > 0 && i < len(output.Logits[0]) {
				score = float64(output.Logits[0][i])
			}

			candidates = append(candidates, model.ItemCandidate{
				ItemID: fmt.Sprintf("item_%d", id), // 需要 ID 映射
				Score:  score,
				Source: "ugt",
			})
		}
	}

	return candidates, nil
}

// retrievalFromMilvus 从 Milvus 向量召回
func (s *Service) retrievalFromMilvus(ctx context.Context, userID string, topK int) ([]model.ItemCandidate, error) {
	// 获取用户嵌入 (简化：使用最近交互物品的平均嵌入)
	userSeq, err := s.userRepo.GetUserSequence(ctx, userID, 10)
	if err != nil || userSeq.Length == 0 {
		return nil, nil
	}

	// 获取最近物品的嵌入
	embeddings, err := s.itemRepo.BatchGetEmbeddings(ctx, userSeq.ItemIDs)
	if err != nil || len(embeddings) == 0 {
		return nil, err
	}

	// 计算平均嵌入
	var avgEmbedding []float32
	for _, emb := range embeddings {
		if avgEmbedding == nil {
			avgEmbedding = make([]float32, len(emb.Embedding))
		}
		for i, v := range emb.Embedding {
			avgEmbedding[i] += v
		}
	}
	for i := range avgEmbedding {
		avgEmbedding[i] /= float32(len(embeddings))
	}

	// 归一化
	avgEmbedding = utils.Normalize(avgEmbedding)

	// 向量搜索
	itemIDs, scores, err := s.milvus.SearchByVector(ctx, s.config.MilvusCollection, avgEmbedding, topK)
	if err != nil {
		return nil, err
	}

	// 构建候选
	candidates := make([]model.ItemCandidate, len(itemIDs))
	for i, itemID := range itemIDs {
		candidates[i] = model.ItemCandidate{
			ItemID: itemID,
			Score:  float64(scores[i]),
			Source: "retrieval",
		}
	}

	return candidates, nil
}

// handleColdStart 处理冷启动用户
func (s *Service) handleColdStart(ctx context.Context, req *model.RecommendRequest) ([]model.ItemCandidate, error) {
	logger.Info("cold start user", zap.String("user_id", req.UserID))

	// 策略1：热门物品
	candidates, err := s.getPopularItems(ctx, req.ItemTypes, req.Size)
	if err != nil {
		return nil, err
	}

	// 标记来源
	for i := range candidates {
		candidates[i].Source = "cold_start"
		candidates[i].Reason = "热门推荐"
	}

	return candidates, nil
}

// getPopularItems 获取热门物品
func (s *Service) getPopularItems(ctx context.Context, itemTypes []model.ItemType, limit int) ([]model.ItemCandidate, error) {
	var candidates []model.ItemCandidate

	for _, itemType := range itemTypes {
		items, err := s.itemRepo.GetPopular(ctx, itemType, limit)
		if err != nil {
			continue
		}

		for i, item := range items {
			candidates = append(candidates, model.ItemCandidate{
				ItemID: item.ID,
				Score:  1.0 - float64(i)*0.01, // 按热度递减
				Source: "popular",
			})
		}
	}

	// 如果没有指定类型，获取所有类型热门
	if len(itemTypes) == 0 {
		for _, itemType := range []model.ItemType{model.ItemTypeMovie, model.ItemTypeVideo, model.ItemTypeProduct} {
			items, err := s.itemRepo.GetPopular(ctx, itemType, limit/3)
			if err != nil {
				continue
			}

			for i, item := range items {
				candidates = append(candidates, model.ItemCandidate{
					ItemID: item.ID,
					Score:  1.0 - float64(i)*0.01,
					Source: "popular",
				})
			}
		}
	}

	return candidates, nil
}

// filterExposed 过滤已曝光物品
func (s *Service) filterExposed(ctx context.Context, userID string, candidates []model.ItemCandidate, exclude []string) []model.ItemCandidate {
	// 获取已曝光物品
	exposed, err := s.recommendRepo.GetExposedItems(ctx, userID, s.config.ExposureFilterHours)
	if err != nil {
		logger.Warn("failed to get exposed items", zap.Error(err))
	}

	// 合并排除列表
	excludeSet := make(map[string]bool)
	for _, id := range exclude {
		excludeSet[id] = true
	}
	for _, id := range exposed {
		excludeSet[id] = true
	}

	// 过滤
	filtered := make([]model.ItemCandidate, 0, len(candidates))
	for _, c := range candidates {
		if !excludeSet[c.ItemID] {
			filtered = append(filtered, c)
		}
	}

	return filtered
}

// enrichItems 填充物品详情
func (s *Service) enrichItems(ctx context.Context, items []model.RecommendItem) []model.RecommendItem {
	if len(items) == 0 {
		return items
	}

	// 收集物品 ID
	itemIDs := make([]string, len(items))
	for i, item := range items {
		itemIDs[i] = item.ItemID
	}

	// 批量获取物品详情
	itemDetails, err := s.itemRepo.BatchGetByIDs(ctx, itemIDs)
	if err != nil {
		logger.Warn("failed to get item details", zap.Error(err))
		return items
	}

	// 构建映射
	detailMap := make(map[string]*model.Item)
	for _, item := range itemDetails {
		detailMap[item.ID] = item
	}

	// 填充详情
	for i := range items {
		if detail, ok := detailMap[items[i].ItemID]; ok {
			items[i].ItemType = detail.Type
			items[i].Title = detail.Title
			items[i].CoverURL = detail.CoverURL
		}
	}

	return items
}

// saveRecommendLog 保存推荐日志
func (s *Service) saveRecommendLog(ctx context.Context, resp *model.RecommendResponse) {
	itemIDs := make([]string, len(resp.Items))
	scores := make([]float64, len(resp.Items))
	sources := make([]string, len(resp.Items))

	for i, item := range resp.Items {
		itemIDs[i] = item.ItemID
		scores[i] = item.Score
		sources[i] = item.Source
	}

	log := &model.RecommendLog{
		RequestID:    resp.RequestID,
		UserID:       resp.UserID,
		ItemIDs:      itemIDs,
		Scores:       scores,
		Sources:      sources,
		ModelVersion: resp.ModelVersion,
		Timestamp:    resp.GeneratedAt,
	}

	if err := s.recommendRepo.SaveLog(ctx, log); err != nil {
		logger.Warn("failed to save recommend log", zap.Error(err))
	}
}

// Similar 获取相似推荐
func (s *Service) Similar(ctx context.Context, req *model.SimilarRequest) (*model.SimilarResponse, error) {
	requestID := utils.GenerateRequestID()

	// 获取参考物品的嵌入
	emb, err := s.itemRepo.GetEmbedding(ctx, req.ItemID)
	if err != nil {
		return nil, fmt.Errorf("item embedding not found: %w", err)
	}

	// 向量搜索
	size := req.Size
	if size <= 0 {
		size = 20
	}

	itemIDs, scores, err := s.milvus.SearchByVector(ctx, s.config.MilvusCollection, emb.Embedding, size+1) // +1 排除自己
	if err != nil {
		return nil, fmt.Errorf("vector search failed: %w", err)
	}

	// 过滤自己和排除列表
	excludeSet := make(map[string]bool)
	excludeSet[req.ItemID] = true
	for _, id := range req.Exclude {
		excludeSet[id] = true
	}

	items := make([]model.RecommendItem, 0, size)
	for i, itemID := range itemIDs {
		if excludeSet[itemID] {
			continue
		}
		items = append(items, model.RecommendItem{
			ItemID:   itemID,
			Score:    float64(scores[i]),
			Source:   "similar",
			Position: len(items) + 1,
		})
		if len(items) >= size {
			break
		}
	}

	// 填充详情
	items = s.enrichItems(ctx, items)

	return &model.SimilarResponse{
		RequestID: requestID,
		RefItemID: req.ItemID,
		Items:     items,
	}, nil
}

