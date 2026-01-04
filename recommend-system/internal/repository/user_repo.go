// Package repository 提供数据访问层
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

// UserRepository 用户数据仓储
type UserRepository struct {
	db *database.PostgresDB
}

// NewUserRepository 创建用户仓储
func NewUserRepository(db *database.PostgresDB) *UserRepository {
	return &UserRepository{db: db}
}

// GetByID 根据 ID 获取用户
func (r *UserRepository) GetByID(ctx context.Context, userID string) (*model.User, error) {
	query := `
		SELECT id, username, email, phone, avatar_url, status, 
		       preferences, tags, created_at, updated_at, last_login_at
		FROM users
		WHERE id = $1 AND status != $2
	`

	var user model.User
	var preferences, tags []byte

	err := r.db.QueryRow(ctx, query, userID, model.UserStatusDeleted).Scan(
		&user.ID, &user.Username, &user.Email, &user.Phone, &user.AvatarURL,
		&user.Status, &preferences, &tags, &user.CreatedAt, &user.UpdatedAt,
		&user.LastLoginAt,
	)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, fmt.Errorf("user not found: %s", userID)
		}
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	// 解析 JSON 字段
	if len(preferences) > 0 {
		json.Unmarshal(preferences, &user.Preferences)
	}
	if len(tags) > 0 {
		json.Unmarshal(tags, &user.Tags)
	}

	return &user, nil
}

// Create 创建用户
func (r *UserRepository) Create(ctx context.Context, user *model.User) error {
	query := `
		INSERT INTO users (id, username, email, phone, avatar_url, status, preferences, tags, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
	`

	preferences, _ := json.Marshal(user.Preferences)
	tags, _ := json.Marshal(user.Tags)

	return r.db.Exec(ctx, query,
		user.ID, user.Username, user.Email, user.Phone, user.AvatarURL,
		user.Status, preferences, tags, user.CreatedAt, user.UpdatedAt,
	)
}

// Update 更新用户
func (r *UserRepository) Update(ctx context.Context, user *model.User) error {
	query := `
		UPDATE users
		SET username = $2, email = $3, phone = $4, avatar_url = $5,
		    status = $6, preferences = $7, tags = $8, updated_at = $9
		WHERE id = $1
	`

	preferences, _ := json.Marshal(user.Preferences)
	tags, _ := json.Marshal(user.Tags)

	return r.db.Exec(ctx, query,
		user.ID, user.Username, user.Email, user.Phone, user.AvatarURL,
		user.Status, preferences, tags, time.Now(),
	)
}

// UpdateLastLogin 更新最后登录时间
func (r *UserRepository) UpdateLastLogin(ctx context.Context, userID string) error {
	query := `UPDATE users SET last_login_at = $2 WHERE id = $1`
	return r.db.Exec(ctx, query, userID, time.Now())
}

// GetBehaviors 获取用户行为记录
func (r *UserRepository) GetBehaviors(ctx context.Context, userID string, limit int) ([]model.UserBehavior, error) {
	query := `
		SELECT id, user_id, item_id, item_type, action, value, context, timestamp
		FROM user_behaviors
		WHERE user_id = $1
		ORDER BY timestamp DESC
		LIMIT $2
	`

	rows, err := r.db.Query(ctx, query, userID, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to get behaviors: %w", err)
	}
	defer rows.Close()

	var behaviors []model.UserBehavior
	for rows.Next() {
		var b model.UserBehavior
		var contextData []byte

		if err := rows.Scan(
			&b.ID, &b.UserID, &b.ItemID, &b.ItemType, &b.Action,
			&b.Value, &contextData, &b.Timestamp,
		); err != nil {
			return nil, fmt.Errorf("failed to scan behavior: %w", err)
		}

		if len(contextData) > 0 {
			b.Context = &model.BehaviorCtx{}
			json.Unmarshal(contextData, b.Context)
		}

		behaviors = append(behaviors, b)
	}

	return behaviors, nil
}

// AddBehavior 添加用户行为
func (r *UserRepository) AddBehavior(ctx context.Context, behavior *model.UserBehavior) error {
	query := `
		INSERT INTO user_behaviors (user_id, item_id, item_type, action, value, context, timestamp)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
	`

	contextData, _ := json.Marshal(behavior.Context)

	return r.db.Exec(ctx, query,
		behavior.UserID, behavior.ItemID, behavior.ItemType, behavior.Action,
		behavior.Value, contextData, behavior.Timestamp,
	)
}

// GetUserSequence 获取用户行为序列 (用于模型输入)
func (r *UserRepository) GetUserSequence(ctx context.Context, userID string, maxLen int) (*model.UserSequence, error) {
	behaviors, err := r.GetBehaviors(ctx, userID, maxLen)
	if err != nil {
		return nil, err
	}

	seq := &model.UserSequence{
		UserID:     userID,
		ItemIDs:    make([]string, 0, len(behaviors)),
		Actions:    make([]string, 0, len(behaviors)),
		Timestamps: make([]int64, 0, len(behaviors)),
		Length:     len(behaviors),
	}

	// 按时间正序
	for i := len(behaviors) - 1; i >= 0; i-- {
		b := behaviors[i]
		seq.ItemIDs = append(seq.ItemIDs, b.ItemID)
		seq.Actions = append(seq.Actions, string(b.Action))
		seq.Timestamps = append(seq.Timestamps, b.Timestamp.Unix())
	}

	return seq, nil
}

// GetProfile 获取用户画像
func (r *UserRepository) GetProfile(ctx context.Context, userID string) (*model.UserProfile, error) {
	query := `
		SELECT user_id, demographics, interests, behavior_stats, 
		       content_preferences, recent_items, long_term_interests, updated_at
		FROM user_profiles
		WHERE user_id = $1
	`

	var profile model.UserProfile
	var demographics, interests, behaviorStats, contentPrefs, recentItems, longTermInterests []byte

	err := r.db.QueryRow(ctx, query, userID).Scan(
		&profile.UserID, &demographics, &interests, &behaviorStats,
		&contentPrefs, &recentItems, &longTermInterests, &profile.UpdatedAt,
	)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, fmt.Errorf("profile not found: %s", userID)
		}
		return nil, fmt.Errorf("failed to get profile: %w", err)
	}

	// 解析 JSON 字段
	if len(demographics) > 0 {
		profile.Demographics = &model.Demographics{}
		json.Unmarshal(demographics, profile.Demographics)
	}
	if len(interests) > 0 {
		json.Unmarshal(interests, &profile.Interests)
	}
	if len(behaviorStats) > 0 {
		profile.BehaviorStats = &model.BehaviorStats{}
		json.Unmarshal(behaviorStats, profile.BehaviorStats)
	}
	if len(contentPrefs) > 0 {
		json.Unmarshal(contentPrefs, &profile.ContentPrefs)
	}
	if len(recentItems) > 0 {
		json.Unmarshal(recentItems, &profile.RecentItems)
	}
	if len(longTermInterests) > 0 {
		json.Unmarshal(longTermInterests, &profile.LongTermInterests)
	}

	return &profile, nil
}

// UpsertProfile 更新或插入用户画像
func (r *UserRepository) UpsertProfile(ctx context.Context, profile *model.UserProfile) error {
	query := `
		INSERT INTO user_profiles (user_id, demographics, interests, behavior_stats, 
		                           content_preferences, recent_items, long_term_interests, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		ON CONFLICT (user_id) DO UPDATE SET
			demographics = EXCLUDED.demographics,
			interests = EXCLUDED.interests,
			behavior_stats = EXCLUDED.behavior_stats,
			content_preferences = EXCLUDED.content_preferences,
			recent_items = EXCLUDED.recent_items,
			long_term_interests = EXCLUDED.long_term_interests,
			updated_at = EXCLUDED.updated_at
	`

	demographics, _ := json.Marshal(profile.Demographics)
	interests, _ := json.Marshal(profile.Interests)
	behaviorStats, _ := json.Marshal(profile.BehaviorStats)
	contentPrefs, _ := json.Marshal(profile.ContentPrefs)
	recentItems, _ := json.Marshal(profile.RecentItems)
	longTermInterests, _ := json.Marshal(profile.LongTermInterests)

	return r.db.Exec(ctx, query,
		profile.UserID, demographics, interests, behaviorStats,
		contentPrefs, recentItems, longTermInterests, time.Now(),
	)
}

