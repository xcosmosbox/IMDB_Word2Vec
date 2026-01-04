package model

import (
	"time"
)

// RecommendRequest 推荐请求
type RecommendRequest struct {
	UserID      string            `json:"user_id" binding:"required"`
	Context     *RequestContext   `json:"context,omitempty"`
	Size        int               `json:"size"`         // 请求数量，默认 20
	ItemTypes   []ItemType        `json:"item_types"`   // 物品类型过滤
	Categories  []string          `json:"categories"`   // 类目过滤
	Exclude     []string          `json:"exclude"`      // 排除物品 ID
	Debug       bool              `json:"debug"`        // 调试模式
}

// RequestContext 请求上下文
type RequestContext struct {
	DeviceType  string  `json:"device_type"`   // mobile, desktop
	Platform    string  `json:"platform"`      // ios, android, web
	Location    string  `json:"location"`      // 地理位置
	SessionID   string  `json:"session_id"`
	PageType    string  `json:"page_type"`     // home, detail, category
	RefItemID   string  `json:"ref_item_id"`   // 参考物品（相似推荐时）
	SearchQuery string  `json:"search_query"`  // 搜索词
	Timestamp   int64   `json:"timestamp"`     // 请求时间戳
}

// RecommendResponse 推荐响应
type RecommendResponse struct {
	RequestID    string            `json:"request_id"`
	UserID       string            `json:"user_id"`
	Items        []RecommendItem   `json:"items"`
	TraceID      string            `json:"trace_id,omitempty"`
	DebugInfo    *DebugInfo        `json:"debug_info,omitempty"`
	GeneratedAt  time.Time         `json:"generated_at"`
	ModelVersion string            `json:"model_version"`
}

// RecommendItem 推荐物品
type RecommendItem struct {
	ItemID      string   `json:"item_id"`
	ItemType    ItemType `json:"item_type"`
	Title       string   `json:"title"`
	CoverURL    string   `json:"cover_url,omitempty"`
	Score       float64  `json:"score"`
	Reason      string   `json:"reason,omitempty"`    // 推荐理由
	Position    int      `json:"position"`
	Source      string   `json:"source"`              // 来源：ugt, retrieval, popular, cold_start
}

// DebugInfo 调试信息
type DebugInfo struct {
	RetrievalTime    int64             `json:"retrieval_time_ms"`
	RankingTime      int64             `json:"ranking_time_ms"`
	TotalTime        int64             `json:"total_time_ms"`
	CandidateCount   int               `json:"candidate_count"`
	FilteredCount    int               `json:"filtered_count"`
	Sources          map[string]int    `json:"sources"`          // 各来源数量
	ModelScores      map[string]float64 `json:"model_scores"`     // 模型分数分布
	FeatureStats     map[string]string `json:"feature_stats"`
}

// SimilarRequest 相似推荐请求
type SimilarRequest struct {
	ItemID    string   `json:"item_id" binding:"required"`
	Size      int      `json:"size"`
	UserID    string   `json:"user_id,omitempty"`     // 可选，用于个性化
	Exclude   []string `json:"exclude"`
}

// SimilarResponse 相似推荐响应
type SimilarResponse struct {
	RequestID string          `json:"request_id"`
	RefItemID string          `json:"ref_item_id"`
	Items     []RecommendItem `json:"items"`
	TraceID   string          `json:"trace_id,omitempty"`
}

// FeedbackRequest 反馈请求
type FeedbackRequest struct {
	UserID     string     `json:"user_id" binding:"required"`
	ItemID     string     `json:"item_id" binding:"required"`
	Action     ActionType `json:"action" binding:"required"`
	Value      float64    `json:"value,omitempty"`   // 评分等数值
	RequestID  string     `json:"request_id"`        // 关联的推荐请求
	Position   int        `json:"position"`          // 物品位置
	Duration   float64    `json:"duration"`          // 停留时长
	Context    *BehaviorCtx `json:"context,omitempty"`
}

// FeedbackResponse 反馈响应
type FeedbackResponse struct {
	Success   bool   `json:"success"`
	Message   string `json:"message,omitempty"`
	EventID   string `json:"event_id"`
}

// RecommendLog 推荐日志 (用于效果追踪)
type RecommendLog struct {
	ID          int64     `json:"id" db:"id"`
	RequestID   string    `json:"request_id" db:"request_id"`
	UserID      string    `json:"user_id" db:"user_id"`
	ItemIDs     []string  `json:"item_ids" db:"item_ids"`
	Scores      []float64 `json:"scores" db:"scores"`
	Sources     []string  `json:"sources" db:"sources"`
	ModelVersion string   `json:"model_version" db:"model_version"`
	Context     string    `json:"context" db:"context"`
	Timestamp   time.Time `json:"timestamp" db:"timestamp"`
}

// ModelInput UGT 模型输入
type ModelInput struct {
	UserID        string    `json:"user_id"`
	InputIDs      []int64   `json:"input_ids"`       // Token ID 序列
	AttentionMask []int64   `json:"attention_mask"`  // 注意力掩码
	TokenTypes    []int64   `json:"token_types"`     // Token 类型标记
	Positions     []int64   `json:"positions"`       // 位置编码
	ContextIDs    []int64   `json:"context_ids"`     // 上下文 Token
	TargetLength  int       `json:"target_length"`   // 目标生成长度
}

// ModelOutput UGT 模型输出
type ModelOutput struct {
	GeneratedIDs  [][]int64   `json:"generated_ids"`   // 生成的 Token ID
	Logits        [][]float32 `json:"logits"`          // 原始 logits
	Probabilities [][]float32 `json:"probabilities"`   // 概率分布
	BeamScores    []float32   `json:"beam_scores"`     // Beam 分数
}

// GenerationConfig 生成配置
type GenerationConfig struct {
	MaxLength     int     `json:"max_length"`
	MinLength     int     `json:"min_length"`
	NumBeams      int     `json:"num_beams"`        // Beam Search 数量
	TopK          int     `json:"top_k"`            // Top-K 采样
	TopP          float64 `json:"top_p"`            // Top-P (Nucleus) 采样
	Temperature   float64 `json:"temperature"`      // 温度参数
	DoSample      bool    `json:"do_sample"`        // 是否采样
	RepetitionPenalty float64 `json:"repetition_penalty"` // 重复惩罚
}

// DefaultGenerationConfig 默认生成配置
func DefaultGenerationConfig() *GenerationConfig {
	return &GenerationConfig{
		MaxLength:        100,
		MinLength:        10,
		NumBeams:         4,
		TopK:             50,
		TopP:             0.9,
		Temperature:      1.0,
		DoSample:         false,
		RepetitionPenalty: 1.2,
	}
}

