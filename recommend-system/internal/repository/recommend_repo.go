package repository

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"recommend-system/internal/model"
	"recommend-system/pkg/database"
)

// RecommendRepository 推荐日志仓储
type RecommendRepository struct {
	db *database.PostgresDB
}

// NewRecommendRepository 创建推荐日志仓储
func NewRecommendRepository(db *database.PostgresDB) *RecommendRepository {
	return &RecommendRepository{db: db}
}

// SaveLog 保存推荐日志
func (r *RecommendRepository) SaveLog(ctx context.Context, log *model.RecommendLog) error {
	query := `
		INSERT INTO recommend_logs (request_id, user_id, item_ids, scores, sources, 
		                            model_version, context, timestamp)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`

	itemIDs, _ := json.Marshal(log.ItemIDs)
	scores, _ := json.Marshal(log.Scores)
	sources, _ := json.Marshal(log.Sources)

	return r.db.Exec(ctx, query,
		log.RequestID, log.UserID, itemIDs, scores, sources,
		log.ModelVersion, log.Context, log.Timestamp,
	)
}

// GetLog 获取推荐日志
func (r *RecommendRepository) GetLog(ctx context.Context, requestID string) (*model.RecommendLog, error) {
	query := `
		SELECT id, request_id, user_id, item_ids, scores, sources, 
		       model_version, context, timestamp
		FROM recommend_logs
		WHERE request_id = $1
	`

	var log model.RecommendLog
	var itemIDs, scores, sources []byte

	err := r.db.QueryRow(ctx, query, requestID).Scan(
		&log.ID, &log.RequestID, &log.UserID, &itemIDs, &scores, &sources,
		&log.ModelVersion, &log.Context, &log.Timestamp,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to get recommend log: %w", err)
	}

	json.Unmarshal(itemIDs, &log.ItemIDs)
	json.Unmarshal(scores, &log.Scores)
	json.Unmarshal(sources, &log.Sources)

	return &log, nil
}

// GetUserLogs 获取用户的推荐日志
func (r *RecommendRepository) GetUserLogs(ctx context.Context, userID string, limit int) ([]*model.RecommendLog, error) {
	query := `
		SELECT id, request_id, user_id, item_ids, scores, sources, 
		       model_version, context, timestamp
		FROM recommend_logs
		WHERE user_id = $1
		ORDER BY timestamp DESC
		LIMIT $2
	`

	rows, err := r.db.Query(ctx, query, userID, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to get user recommend logs: %w", err)
	}
	defer rows.Close()

	var logs []*model.RecommendLog
	for rows.Next() {
		var log model.RecommendLog
		var itemIDs, scores, sources []byte

		if err := rows.Scan(
			&log.ID, &log.RequestID, &log.UserID, &itemIDs, &scores, &sources,
			&log.ModelVersion, &log.Context, &log.Timestamp,
		); err != nil {
			return nil, fmt.Errorf("failed to scan recommend log: %w", err)
		}

		json.Unmarshal(itemIDs, &log.ItemIDs)
		json.Unmarshal(scores, &log.Scores)
		json.Unmarshal(sources, &log.Sources)

		logs = append(logs, &log)
	}

	return logs, nil
}

// GetExposedItems 获取用户已曝光物品
func (r *RecommendRepository) GetExposedItems(ctx context.Context, userID string, hours int) ([]string, error) {
	query := `
		SELECT DISTINCT unnest(item_ids) as item_id
		FROM recommend_logs
		WHERE user_id = $1 AND timestamp > $2
	`

	since := time.Now().Add(-time.Duration(hours) * time.Hour)
	rows, err := r.db.Query(ctx, query, userID, since)
	if err != nil {
		return nil, fmt.Errorf("failed to get exposed items: %w", err)
	}
	defer rows.Close()

	var items []string
	for rows.Next() {
		var itemID string
		if err := rows.Scan(&itemID); err != nil {
			return nil, fmt.Errorf("failed to scan item id: %w", err)
		}
		items = append(items, itemID)
	}

	return items, nil
}

// CleanOldLogs 清理旧日志
func (r *RecommendRepository) CleanOldLogs(ctx context.Context, days int) (int64, error) {
	query := `DELETE FROM recommend_logs WHERE timestamp < $1`
	
	cutoff := time.Now().AddDate(0, 0, -days)
	result, err := r.db.Pool().Exec(ctx, query, cutoff)
	if err != nil {
		return 0, fmt.Errorf("failed to clean old logs: %w", err)
	}

	return result.RowsAffected(), nil
}

