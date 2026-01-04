package model

import (
	"encoding/json"
	"time"
)

// Item 物品基础模型
type Item struct {
	ID          string            `json:"id" db:"id"`
	Type        ItemType          `json:"type" db:"type"`
	Title       string            `json:"title" db:"title"`
	Description string            `json:"description,omitempty" db:"description"`
	CoverURL    string            `json:"cover_url,omitempty" db:"cover_url"`
	Category    string            `json:"category" db:"category"`
	SubCategory string            `json:"sub_category,omitempty" db:"sub_category"`
	Tags        []string          `json:"tags,omitempty" db:"tags"`
	Attributes  map[string]string `json:"attributes,omitempty" db:"attributes"`
	Status      ItemStatus        `json:"status" db:"status"`
	CreatedAt   time.Time         `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at" db:"updated_at"`
	PublishedAt *time.Time        `json:"published_at,omitempty" db:"published_at"`
}

// ItemType 物品类型
type ItemType string

const (
	ItemTypeMovie   ItemType = "movie"
	ItemTypeVideo   ItemType = "video"
	ItemTypeProduct ItemType = "product"
	ItemTypeArticle ItemType = "article"
	ItemTypeMusic   ItemType = "music"
	ItemTypeBook    ItemType = "book"
)

// ItemStatus 物品状态
type ItemStatus int

const (
	ItemStatusDraft     ItemStatus = 0 // 草稿
	ItemStatusPublished ItemStatus = 1 // 已发布
	ItemStatusOffline   ItemStatus = 2 // 已下线
	ItemStatusDeleted   ItemStatus = 3 // 已删除
)

// ItemStats 物品统计信息
type ItemStats struct {
	ItemID       string    `json:"item_id" db:"item_id"`
	ViewCount    int64     `json:"view_count" db:"view_count"`
	ClickCount   int64     `json:"click_count" db:"click_count"`
	LikeCount    int64     `json:"like_count" db:"like_count"`
	ShareCount   int64     `json:"share_count" db:"share_count"`
	CommentCount int64     `json:"comment_count" db:"comment_count"`
	RatingSum    float64   `json:"rating_sum" db:"rating_sum"`
	RatingCount  int64     `json:"rating_count" db:"rating_count"`
	AvgRating    float64   `json:"avg_rating" db:"avg_rating"`
	UpdatedAt    time.Time `json:"updated_at" db:"updated_at"`
}

// ItemEmbedding 物品嵌入向量
type ItemEmbedding struct {
	ItemID      string    `json:"item_id"`
	Embedding   []float32 `json:"embedding"`
	SemanticID  []int     `json:"semantic_id"`  // 层次化语义 ID
	SemanticL1  int       `json:"semantic_l1"`  // 第一层语义 ID
	SemanticL2  int       `json:"semantic_l2"`  // 第二层语义 ID
	SemanticL3  int       `json:"semantic_l3"`  // 第三层语义 ID
	ModelVersion string   `json:"model_version"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// Movie 电影模型
type Movie struct {
	Item
	Director    string    `json:"director" db:"director"`
	Actors      []string  `json:"actors" db:"actors"`
	Genres      []string  `json:"genres" db:"genres"`
	ReleaseDate string    `json:"release_date" db:"release_date"`
	Duration    int       `json:"duration" db:"duration"` // 分钟
	Rating      float64   `json:"rating" db:"rating"`
	Country     string    `json:"country" db:"country"`
	Language    string    `json:"language" db:"language"`
}

// Video 视频模型
type Video struct {
	Item
	AuthorID    string   `json:"author_id" db:"author_id"`
	AuthorName  string   `json:"author_name" db:"author_name"`
	Duration    int      `json:"duration" db:"duration"` // 秒
	Resolution  string   `json:"resolution" db:"resolution"`
	PlayURL     string   `json:"play_url" db:"play_url"`
}

// Product 商品模型
type Product struct {
	Item
	Price       float64  `json:"price" db:"price"`
	OrigPrice   float64  `json:"orig_price,omitempty" db:"orig_price"`
	Currency    string   `json:"currency" db:"currency"`
	Stock       int      `json:"stock" db:"stock"`
	Sales       int      `json:"sales" db:"sales"`
	Brand       string   `json:"brand,omitempty" db:"brand"`
	ShopID      string   `json:"shop_id" db:"shop_id"`
	ShopName    string   `json:"shop_name" db:"shop_name"`
}

// Article 文章模型
type Article struct {
	Item
	AuthorID    string   `json:"author_id" db:"author_id"`
	AuthorName  string   `json:"author_name" db:"author_name"`
	Content     string   `json:"content,omitempty" db:"content"`
	WordCount   int      `json:"word_count" db:"word_count"`
	ReadTime    int      `json:"read_time" db:"read_time"` // 预估阅读时间（分钟）
}

// ItemFeatures 物品特征 (用于模型输入)
type ItemFeatures struct {
	ItemID       string            `json:"item_id"`
	Type         ItemType          `json:"type"`
	Category     string            `json:"category"`
	SubCategory  string            `json:"sub_category"`
	Tags         []string          `json:"tags"`
	NumericFeats map[string]float64 `json:"numeric_features"` // 数值特征
	TextFeats    map[string]string  `json:"text_features"`    // 文本特征
	Popularity   float64           `json:"popularity"`        // 流行度分数
	Freshness    float64           `json:"freshness"`         // 新鲜度分数
}

// Scan 从数据库读取 JSON
func (f *ItemFeatures) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	bytes, ok := value.([]byte)
	if !ok {
		return nil
	}
	return json.Unmarshal(bytes, f)
}

// ItemCandidate 候选物品 (用于推荐结果)
type ItemCandidate struct {
	ItemID      string            `json:"item_id"`
	Score       float64           `json:"score"`
	Source      string            `json:"source"` // retrieval, llm, popular, similar
	Reason      string            `json:"reason,omitempty"`
	Features    *ItemFeatures     `json:"features,omitempty"`
	Position    int               `json:"position"`
	DebugInfo   map[string]string `json:"debug_info,omitempty"`
}

// ToTokenSequence 转换为 Token 序列
func (i *Item) ToTokenSequence() []string {
	tokens := make([]string, 0, 10)
	
	// 类型 Token
	tokens = append(tokens, "TYPE_"+string(i.Type))
	
	// 类目 Token
	tokens = append(tokens, "CAT_"+i.Category)
	if i.SubCategory != "" {
		tokens = append(tokens, "SUBCAT_"+i.SubCategory)
	}
	
	// 标签 Tokens
	for _, tag := range i.Tags {
		tokens = append(tokens, "TAG_"+tag)
	}
	
	return tokens
}

