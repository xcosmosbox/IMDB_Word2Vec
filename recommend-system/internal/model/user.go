// Package model 定义数据模型
package model

import (
	"encoding/json"
	"time"
)

// User 用户模型
type User struct {
	ID          string            `json:"id" db:"id"`
	Username    string            `json:"username" db:"username"`
	Email       string            `json:"email" db:"email"`
	Phone       string            `json:"phone,omitempty" db:"phone"`
	AvatarURL   string            `json:"avatar_url,omitempty" db:"avatar_url"`
	Status      UserStatus        `json:"status" db:"status"`
	Preferences map[string]string `json:"preferences,omitempty" db:"preferences"`
	Tags        []string          `json:"tags,omitempty" db:"tags"`
	CreatedAt   time.Time         `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at" db:"updated_at"`
	LastLoginAt *time.Time        `json:"last_login_at,omitempty" db:"last_login_at"`
}

// UserStatus 用户状态
type UserStatus int

const (
	UserStatusActive   UserStatus = 1 // 活跃
	UserStatusInactive UserStatus = 2 // 未激活
	UserStatusBlocked  UserStatus = 3 // 封禁
	UserStatusDeleted  UserStatus = 4 // 删除
)

// UserBehavior 用户行为记录
type UserBehavior struct {
	ID        int64        `json:"id" db:"id"`
	UserID    string       `json:"user_id" db:"user_id"`
	ItemID    string       `json:"item_id" db:"item_id"`
	ItemType  string       `json:"item_type" db:"item_type"` // movie, video, product, article
	Action    ActionType   `json:"action" db:"action"`
	Value     float64      `json:"value,omitempty" db:"value"` // 评分、观看时长等
	Context   *BehaviorCtx `json:"context,omitempty" db:"context"`
	Timestamp time.Time    `json:"timestamp" db:"timestamp"`
}

// ActionType 行为类型
type ActionType string

const (
	ActionView     ActionType = "view"     // 浏览
	ActionClick    ActionType = "click"    // 点击
	ActionLike     ActionType = "like"     // 点赞
	ActionDislike  ActionType = "dislike"  // 点踩
	ActionFavorite ActionType = "favorite" // 收藏
	ActionShare    ActionType = "share"    // 分享
	ActionComment  ActionType = "comment"  // 评论
	ActionPurchase ActionType = "purchase" // 购买
	ActionRate     ActionType = "rate"     // 评分
	ActionPlay     ActionType = "play"     // 播放
	ActionComplete ActionType = "complete" // 完成观看/阅读
	ActionSearch   ActionType = "search"   // 搜索
)

// BehaviorCtx 行为上下文
type BehaviorCtx struct {
	DeviceType string  `json:"device_type,omitempty"` // mobile, desktop, tablet
	Platform   string  `json:"platform,omitempty"`    // ios, android, web
	Location   string  `json:"location,omitempty"`    // 地理位置
	SessionID  string  `json:"session_id,omitempty"`  // 会话 ID
	Source     string  `json:"source,omitempty"`      // 来源：recommend, search, home
	Position   int     `json:"position,omitempty"`    // 推荐列表位置
	Duration   float64 `json:"duration,omitempty"`    // 停留时长（秒）
}

// Scan 从数据库读取 JSON
func (c *BehaviorCtx) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	bytes, ok := value.([]byte)
	if !ok {
		return nil
	}
	return json.Unmarshal(bytes, c)
}

// UserProfile 用户画像
type UserProfile struct {
	UserID            string            `json:"user_id"`
	Demographics      *Demographics     `json:"demographics,omitempty"`
	Interests         []Interest        `json:"interests"`
	BehaviorStats     *BehaviorStats    `json:"behavior_stats"`
	ContentPrefs      map[string]float64 `json:"content_preferences"`
	RecentItems       []string          `json:"recent_items"`        // 最近交互物品
	LongTermInterests []string          `json:"long_term_interests"` // 长期兴趣
	UpdatedAt         time.Time         `json:"updated_at"`
}

// Demographics 人口统计信息
type Demographics struct {
	AgeRange string `json:"age_range,omitempty"` // 18-24, 25-34, etc.
	Gender   string `json:"gender,omitempty"`
	City     string `json:"city,omitempty"`
	Country  string `json:"country,omitempty"`
}

// Interest 兴趣标签
type Interest struct {
	Category   string    `json:"category"`   // 一级类目
	SubCat     string    `json:"sub_cat"`    // 二级类目
	Weight     float64   `json:"weight"`     // 权重 0-1
	Source     string    `json:"source"`     // 来源：explicit, implicit
	UpdatedAt  time.Time `json:"updated_at"`
}

// BehaviorStats 行为统计
type BehaviorStats struct {
	TotalViews     int64   `json:"total_views"`
	TotalClicks    int64   `json:"total_clicks"`
	TotalPurchases int64   `json:"total_purchases"`
	AvgRating      float64 `json:"avg_rating"`
	ActiveDays     int     `json:"active_days"`
	LastActiveAt   time.Time `json:"last_active_at"`
}

// UserSequence 用户行为序列 (用于模型输入)
type UserSequence struct {
	UserID      string      `json:"user_id"`
	ItemIDs     []string    `json:"item_ids"`     // 物品 ID 序列
	Actions     []string    `json:"actions"`      // 行为类型序列
	Timestamps  []int64     `json:"timestamps"`   // 时间戳序列
	SemanticIDs [][]int     `json:"semantic_ids"` // 语义 ID 序列
	Length      int         `json:"length"`       // 序列长度
}

// ToTokens 转换为 Token 序列 (用于模型输入)
func (s *UserSequence) ToTokens() []string {
	tokens := make([]string, 0, len(s.ItemIDs)*3)
	
	for i, itemID := range s.ItemIDs {
		// 添加行为 Token
		tokens = append(tokens, "ACTION_"+s.Actions[i])
		// 添加物品 Token
		tokens = append(tokens, "ITEM_"+itemID)
		// 添加时间 Token (按小时分桶)
		hour := time.Unix(s.Timestamps[i], 0).Hour()
		tokens = append(tokens, timeToken(hour))
	}
	
	return tokens
}

// timeToken 生成时间 Token
func timeToken(hour int) string {
	switch {
	case hour >= 0 && hour < 6:
		return "TIME_NIGHT"
	case hour >= 6 && hour < 12:
		return "TIME_MORNING"
	case hour >= 12 && hour < 18:
		return "TIME_AFTERNOON"
	default:
		return "TIME_EVENING"
	}
}

