package repository

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"recommend-system/internal/model"
	"recommend-system/pkg/database"
)

// ItemRepository 物品数据仓储
type ItemRepository struct {
	db *database.PostgresDB
}

// NewItemRepository 创建物品仓储
func NewItemRepository(db *database.PostgresDB) *ItemRepository {
	return &ItemRepository{db: db}
}

// GetByID 根据 ID 获取物品
func (r *ItemRepository) GetByID(ctx context.Context, itemID string) (*model.Item, error) {
	query := `
		SELECT id, type, title, description, cover_url, category, sub_category,
		       tags, attributes, status, created_at, updated_at, published_at
		FROM items
		WHERE id = $1 AND status != $2
	`

	var item model.Item
	var tags, attributes []byte

	err := r.db.QueryRow(ctx, query, itemID, model.ItemStatusDeleted).Scan(
		&item.ID, &item.Type, &item.Title, &item.Description, &item.CoverURL,
		&item.Category, &item.SubCategory, &tags, &attributes,
		&item.Status, &item.CreatedAt, &item.UpdatedAt, &item.PublishedAt,
	)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, fmt.Errorf("item not found: %s", itemID)
		}
		return nil, fmt.Errorf("failed to get item: %w", err)
	}

	if len(tags) > 0 {
		json.Unmarshal(tags, &item.Tags)
	}
	if len(attributes) > 0 {
		json.Unmarshal(attributes, &item.Attributes)
	}

	return &item, nil
}

// BatchGetByIDs 批量获取物品
func (r *ItemRepository) BatchGetByIDs(ctx context.Context, itemIDs []string) ([]*model.Item, error) {
	if len(itemIDs) == 0 {
		return nil, nil
	}

	query := `
		SELECT id, type, title, description, cover_url, category, sub_category,
		       tags, attributes, status, created_at, updated_at, published_at
		FROM items
		WHERE id = ANY($1) AND status = $2
	`

	rows, err := r.db.Query(ctx, query, itemIDs, model.ItemStatusPublished)
	if err != nil {
		return nil, fmt.Errorf("failed to batch get items: %w", err)
	}
	defer rows.Close()

	var items []*model.Item
	for rows.Next() {
		var item model.Item
		var tags, attributes []byte

		if err := rows.Scan(
			&item.ID, &item.Type, &item.Title, &item.Description, &item.CoverURL,
			&item.Category, &item.SubCategory, &tags, &attributes,
			&item.Status, &item.CreatedAt, &item.UpdatedAt, &item.PublishedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan item: %w", err)
		}

		if len(tags) > 0 {
			json.Unmarshal(tags, &item.Tags)
		}
		if len(attributes) > 0 {
			json.Unmarshal(attributes, &item.Attributes)
		}

		items = append(items, &item)
	}

	return items, nil
}

// Create 创建物品
func (r *ItemRepository) Create(ctx context.Context, item *model.Item) error {
	query := `
		INSERT INTO items (id, type, title, description, cover_url, category, sub_category,
		                   tags, attributes, status, created_at, updated_at, published_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
	`

	tags, _ := json.Marshal(item.Tags)
	attributes, _ := json.Marshal(item.Attributes)

	return r.db.Exec(ctx, query,
		item.ID, item.Type, item.Title, item.Description, item.CoverURL,
		item.Category, item.SubCategory, tags, attributes,
		item.Status, item.CreatedAt, item.UpdatedAt, item.PublishedAt,
	)
}

// Update 更新物品
func (r *ItemRepository) Update(ctx context.Context, item *model.Item) error {
	query := `
		UPDATE items
		SET type = $2, title = $3, description = $4, cover_url = $5,
		    category = $6, sub_category = $7, tags = $8, attributes = $9,
		    status = $10, updated_at = $11
		WHERE id = $1
	`

	tags, _ := json.Marshal(item.Tags)
	attributes, _ := json.Marshal(item.Attributes)

	return r.db.Exec(ctx, query,
		item.ID, item.Type, item.Title, item.Description, item.CoverURL,
		item.Category, item.SubCategory, tags, attributes,
		item.Status, time.Now(),
	)
}

// GetStats 获取物品统计
func (r *ItemRepository) GetStats(ctx context.Context, itemID string) (*model.ItemStats, error) {
	query := `
		SELECT item_id, view_count, click_count, like_count, share_count,
		       comment_count, rating_sum, rating_count, avg_rating, updated_at
		FROM item_stats
		WHERE item_id = $1
	`

	var stats model.ItemStats
	err := r.db.QueryRow(ctx, query, itemID).Scan(
		&stats.ItemID, &stats.ViewCount, &stats.ClickCount, &stats.LikeCount,
		&stats.ShareCount, &stats.CommentCount, &stats.RatingSum,
		&stats.RatingCount, &stats.AvgRating, &stats.UpdatedAt,
	)
	if err != nil {
		if err == pgx.ErrNoRows {
			return &model.ItemStats{ItemID: itemID}, nil
		}
		return nil, fmt.Errorf("failed to get item stats: %w", err)
	}

	return &stats, nil
}

// IncrementStats 增加物品统计
func (r *ItemRepository) IncrementStats(ctx context.Context, itemID string, field string, delta int64) error {
	query := fmt.Sprintf(`
		INSERT INTO item_stats (item_id, %s, updated_at)
		VALUES ($1, $2, $3)
		ON CONFLICT (item_id) DO UPDATE SET
			%s = item_stats.%s + $2,
			updated_at = $3
	`, field, field, field)

	return r.db.Exec(ctx, query, itemID, delta, time.Now())
}

// GetEmbedding 获取物品嵌入
func (r *ItemRepository) GetEmbedding(ctx context.Context, itemID string) (*model.ItemEmbedding, error) {
	query := `
		SELECT item_id, embedding, semantic_id, semantic_l1, semantic_l2, semantic_l3,
		       model_version, updated_at
		FROM item_embeddings
		WHERE item_id = $1
	`

	var emb model.ItemEmbedding
	var embeddingData, semanticIDData []byte

	err := r.db.QueryRow(ctx, query, itemID).Scan(
		&emb.ItemID, &embeddingData, &semanticIDData,
		&emb.SemanticL1, &emb.SemanticL2, &emb.SemanticL3,
		&emb.ModelVersion, &emb.UpdatedAt,
	)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, fmt.Errorf("embedding not found: %s", itemID)
		}
		return nil, fmt.Errorf("failed to get embedding: %w", err)
	}

	if len(embeddingData) > 0 {
		json.Unmarshal(embeddingData, &emb.Embedding)
	}
	if len(semanticIDData) > 0 {
		json.Unmarshal(semanticIDData, &emb.SemanticID)
	}

	return &emb, nil
}

// BatchGetEmbeddings 批量获取物品嵌入
func (r *ItemRepository) BatchGetEmbeddings(ctx context.Context, itemIDs []string) (map[string]*model.ItemEmbedding, error) {
	if len(itemIDs) == 0 {
		return nil, nil
	}

	query := `
		SELECT item_id, embedding, semantic_id, semantic_l1, semantic_l2, semantic_l3,
		       model_version, updated_at
		FROM item_embeddings
		WHERE item_id = ANY($1)
	`

	rows, err := r.db.Query(ctx, query, itemIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to batch get embeddings: %w", err)
	}
	defer rows.Close()

	result := make(map[string]*model.ItemEmbedding)
	for rows.Next() {
		var emb model.ItemEmbedding
		var embeddingData, semanticIDData []byte

		if err := rows.Scan(
			&emb.ItemID, &embeddingData, &semanticIDData,
			&emb.SemanticL1, &emb.SemanticL2, &emb.SemanticL3,
			&emb.ModelVersion, &emb.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan embedding: %w", err)
		}

		if len(embeddingData) > 0 {
			json.Unmarshal(embeddingData, &emb.Embedding)
		}
		if len(semanticIDData) > 0 {
			json.Unmarshal(semanticIDData, &emb.SemanticID)
		}

		result[emb.ItemID] = &emb
	}

	return result, nil
}

// UpsertEmbedding 更新或插入物品嵌入
func (r *ItemRepository) UpsertEmbedding(ctx context.Context, emb *model.ItemEmbedding) error {
	query := `
		INSERT INTO item_embeddings (item_id, embedding, semantic_id, semantic_l1, 
		                              semantic_l2, semantic_l3, model_version, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		ON CONFLICT (item_id) DO UPDATE SET
			embedding = EXCLUDED.embedding,
			semantic_id = EXCLUDED.semantic_id,
			semantic_l1 = EXCLUDED.semantic_l1,
			semantic_l2 = EXCLUDED.semantic_l2,
			semantic_l3 = EXCLUDED.semantic_l3,
			model_version = EXCLUDED.model_version,
			updated_at = EXCLUDED.updated_at
	`

	embeddingData, _ := json.Marshal(emb.Embedding)
	semanticIDData, _ := json.Marshal(emb.SemanticID)

	return r.db.Exec(ctx, query,
		emb.ItemID, embeddingData, semanticIDData,
		emb.SemanticL1, emb.SemanticL2, emb.SemanticL3,
		emb.ModelVersion, time.Now(),
	)
}

// GetByCategory 根据类目获取物品
func (r *ItemRepository) GetByCategory(ctx context.Context, category string, limit int) ([]*model.Item, error) {
	query := `
		SELECT id, type, title, description, cover_url, category, sub_category,
		       tags, attributes, status, created_at, updated_at, published_at
		FROM items
		WHERE category = $1 AND status = $2
		ORDER BY updated_at DESC
		LIMIT $3
	`

	rows, err := r.db.Query(ctx, query, category, model.ItemStatusPublished, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to get items by category: %w", err)
	}
	defer rows.Close()

	var items []*model.Item
	for rows.Next() {
		var item model.Item
		var tags, attributes []byte

		if err := rows.Scan(
			&item.ID, &item.Type, &item.Title, &item.Description, &item.CoverURL,
			&item.Category, &item.SubCategory, &tags, &attributes,
			&item.Status, &item.CreatedAt, &item.UpdatedAt, &item.PublishedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan item: %w", err)
		}

		if len(tags) > 0 {
			json.Unmarshal(tags, &item.Tags)
		}
		if len(attributes) > 0 {
			json.Unmarshal(attributes, &item.Attributes)
		}

		items = append(items, &item)
	}

	return items, nil
}

// GetPopular 获取热门物品
func (r *ItemRepository) GetPopular(ctx context.Context, itemType model.ItemType, limit int) ([]*model.Item, error) {
	query := `
		SELECT i.id, i.type, i.title, i.description, i.cover_url, i.category, i.sub_category,
		       i.tags, i.attributes, i.status, i.created_at, i.updated_at, i.published_at
		FROM items i
		LEFT JOIN item_stats s ON i.id = s.item_id
		WHERE i.type = $1 AND i.status = $2
		ORDER BY COALESCE(s.view_count, 0) DESC, i.updated_at DESC
		LIMIT $3
	`

	rows, err := r.db.Query(ctx, query, itemType, model.ItemStatusPublished, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to get popular items: %w", err)
	}
	defer rows.Close()

	var items []*model.Item
	for rows.Next() {
		var item model.Item
		var tags, attributes []byte

		if err := rows.Scan(
			&item.ID, &item.Type, &item.Title, &item.Description, &item.CoverURL,
			&item.Category, &item.SubCategory, &tags, &attributes,
			&item.Status, &item.CreatedAt, &item.UpdatedAt, &item.PublishedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan item: %w", err)
		}

		if len(tags) > 0 {
			json.Unmarshal(tags, &item.Tags)
		}
		if len(attributes) > 0 {
			json.Unmarshal(attributes, &item.Attributes)
		}

		items = append(items, &item)
	}

	return items, nil
}

