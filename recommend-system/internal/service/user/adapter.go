// Package user 提供用户服务的适配器
//
// 适配器用于将底层实现转换为接口定义，实现依赖注入和解耦。
package user

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"recommend-system/internal/cache"
	"recommend-system/internal/interfaces"
	"recommend-system/internal/model"
	"recommend-system/internal/repository"
)

// =============================================================================
// 用户仓库适配器
// =============================================================================

// RepositoryAdapter 用户仓库适配器
//
// 将 repository.UserRepository 适配到 interfaces.UserRepository
// 负责数据模型的转换
type RepositoryAdapter struct {
	repo *repository.UserRepository
}

// NewRepositoryAdapter 创建仓库适配器
func NewRepositoryAdapter(repo *repository.UserRepository) *RepositoryAdapter {
	return &RepositoryAdapter{repo: repo}
}

// GetByID 根据 ID 获取用户
func (a *RepositoryAdapter) GetByID(ctx context.Context, userID string) (*interfaces.User, error) {
	modelUser, err := a.repo.GetByID(ctx, userID)
	if err != nil {
		return nil, err
	}
	return convertModelToInterfaceUser(modelUser), nil
}

// GetByIDs 批量获取用户
func (a *RepositoryAdapter) GetByIDs(ctx context.Context, userIDs []string) ([]*interfaces.User, error) {
	users := make([]*interfaces.User, 0, len(userIDs))
	for _, id := range userIDs {
		user, err := a.GetByID(ctx, id)
		if err != nil {
			continue // 跳过找不到的用户
		}
		users = append(users, user)
	}
	return users, nil
}

// Create 创建用户
func (a *RepositoryAdapter) Create(ctx context.Context, user *interfaces.User) error {
	modelUser := convertInterfaceToModelUser(user)
	return a.repo.Create(ctx, modelUser)
}

// Update 更新用户
func (a *RepositoryAdapter) Update(ctx context.Context, user *interfaces.User) error {
	modelUser := convertInterfaceToModelUser(user)
	return a.repo.Update(ctx, modelUser)
}

// Delete 删除用户
func (a *RepositoryAdapter) Delete(ctx context.Context, userID string) error {
	// 软删除：更新状态为已删除
	modelUser, err := a.repo.GetByID(ctx, userID)
	if err != nil {
		return err
	}
	modelUser.Status = model.UserStatusDeleted
	return a.repo.Update(ctx, modelUser)
}

// GetBehaviors 获取用户行为
func (a *RepositoryAdapter) GetBehaviors(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error) {
	modelBehaviors, err := a.repo.GetBehaviors(ctx, userID, limit)
	if err != nil {
		return nil, err
	}

	behaviors := make([]*interfaces.UserBehavior, len(modelBehaviors))
	for i, mb := range modelBehaviors {
		behaviors[i] = convertModelToInterfaceBehavior(&mb)
	}
	return behaviors, nil
}

// AddBehavior 添加用户行为
func (a *RepositoryAdapter) AddBehavior(ctx context.Context, behavior *interfaces.UserBehavior) error {
	modelBehavior := convertInterfaceToModelBehavior(behavior)
	return a.repo.AddBehavior(ctx, modelBehavior)
}

// GetUserItemInteractions 获取用户与物品的交互记录
func (a *RepositoryAdapter) GetUserItemInteractions(ctx context.Context, userID, itemID string) ([]*interfaces.UserBehavior, error) {
	// 获取所有行为然后过滤
	allBehaviors, err := a.repo.GetBehaviors(ctx, userID, 1000)
	if err != nil {
		return nil, err
	}

	interactions := make([]*interfaces.UserBehavior, 0)
	for _, mb := range allBehaviors {
		if mb.ItemID == itemID {
			interactions = append(interactions, convertModelToInterfaceBehavior(&mb))
		}
	}
	return interactions, nil
}

// =============================================================================
// 缓存适配器
// =============================================================================

// CacheAdapter 缓存适配器
//
// 将 cache.MultiLevelCache 适配到 interfaces.Cache
type CacheAdapter struct {
	cache *cache.MultiLevelCache
}

// NewCacheAdapter 创建缓存适配器
func NewCacheAdapter(c *cache.MultiLevelCache) *CacheAdapter {
	return &CacheAdapter{cache: c}
}

// Get 获取缓存值
func (a *CacheAdapter) Get(ctx context.Context, key string, value interface{}) error {
	data, err := a.cache.Get(ctx, key)
	if err != nil {
		return err
	}
	if len(data) == 0 {
		return fmt.Errorf("cache miss: %s", key)
	}
	return json.Unmarshal(data, value)
}

// Set 设置缓存值
func (a *CacheAdapter) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return a.cache.Set(ctx, key, data, ttl)
}

// Delete 删除缓存
func (a *CacheAdapter) Delete(ctx context.Context, key string) error {
	return a.cache.Delete(ctx, key)
}

// Exists 检查缓存是否存在
func (a *CacheAdapter) Exists(ctx context.Context, key string) (bool, error) {
	return a.cache.Exists(ctx, key)
}

// MGet 批量获取缓存
func (a *CacheAdapter) MGet(ctx context.Context, keys []string) ([]interface{}, error) {
	results := make([]interface{}, len(keys))
	for i, key := range keys {
		data, err := a.cache.Get(ctx, key)
		if err != nil {
			results[i] = nil
			continue
		}
		results[i] = data
	}
	return results, nil
}

// MSet 批量设置缓存
func (a *CacheAdapter) MSet(ctx context.Context, kvs map[string]interface{}, ttl time.Duration) error {
	for key, value := range kvs {
		if err := a.Set(ctx, key, value, ttl); err != nil {
			return err
		}
	}
	return nil
}

// =============================================================================
// 模型转换函数
// =============================================================================

// convertModelToInterfaceUser 将 model.User 转换为 interfaces.User
func convertModelToInterfaceUser(m *model.User) *interfaces.User {
	if m == nil {
		return nil
	}

	user := &interfaces.User{
		ID:        m.ID,
		Name:      m.Username,
		Email:     m.Email,
		Metadata:  m.Preferences,
		CreatedAt: m.CreatedAt,
		UpdatedAt: m.UpdatedAt,
	}

	// 处理可选的最后登录时间
	if m.LastLoginAt != nil {
		// 可以添加到 metadata 中
	}

	return user
}

// convertInterfaceToModelUser 将 interfaces.User 转换为 model.User
func convertInterfaceToModelUser(u *interfaces.User) *model.User {
	if u == nil {
		return nil
	}

	return &model.User{
		ID:          u.ID,
		Username:    u.Name,
		Email:       u.Email,
		Status:      model.UserStatusActive,
		Preferences: u.Metadata,
		CreatedAt:   u.CreatedAt,
		UpdatedAt:   u.UpdatedAt,
	}
}

// convertModelToInterfaceBehavior 将 model.UserBehavior 转换为 interfaces.UserBehavior
func convertModelToInterfaceBehavior(m *model.UserBehavior) *interfaces.UserBehavior {
	if m == nil {
		return nil
	}

	behavior := &interfaces.UserBehavior{
		UserID:    m.UserID,
		ItemID:    m.ItemID,
		Action:    string(m.Action),
		Timestamp: m.Timestamp,
	}

	// 转换上下文
	if m.Context != nil {
		behavior.Context = map[string]string{
			"device_type": m.Context.DeviceType,
			"platform":    m.Context.Platform,
			"location":    m.Context.Location,
			"session_id":  m.Context.SessionID,
			"source":      m.Context.Source,
		}
	}

	return behavior
}

// convertInterfaceToModelBehavior 将 interfaces.UserBehavior 转换为 model.UserBehavior
func convertInterfaceToModelBehavior(b *interfaces.UserBehavior) *model.UserBehavior {
	if b == nil {
		return nil
	}

	behavior := &model.UserBehavior{
		UserID:    b.UserID,
		ItemID:    b.ItemID,
		Action:    model.ActionType(b.Action),
		Timestamp: b.Timestamp,
	}

	// 转换上下文
	if b.Context != nil {
		behavior.Context = &model.BehaviorCtx{
			DeviceType: b.Context["device_type"],
			Platform:   b.Context["platform"],
			Location:   b.Context["location"],
			SessionID:  b.Context["session_id"],
			Source:     b.Context["source"],
		}
	}

	return behavior
}

// =============================================================================
// 接口编译时检查
// =============================================================================

var _ interfaces.UserRepository = (*RepositoryAdapter)(nil)
var _ interfaces.Cache = (*CacheAdapter)(nil)

