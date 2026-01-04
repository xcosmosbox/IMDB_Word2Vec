// Package user 提供用户服务的测试 Mock
package user

import (
	"context"
	"fmt"
	"sync"
	"time"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// Mock 用户仓库
// =============================================================================

// MockUserRepository Mock 用户仓库实现
type MockUserRepository struct {
	mu        sync.RWMutex
	users     map[string]*interfaces.User
	behaviors map[string][]*interfaces.UserBehavior

	// 用于测试的回调函数
	GetByIDFunc     func(ctx context.Context, userID string) (*interfaces.User, error)
	CreateFunc      func(ctx context.Context, user *interfaces.User) error
	UpdateFunc      func(ctx context.Context, user *interfaces.User) error
	DeleteFunc      func(ctx context.Context, userID string) error
	GetBehaviorsFunc func(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error)
	AddBehaviorFunc  func(ctx context.Context, behavior *interfaces.UserBehavior) error
}

// NewMockUserRepository 创建 Mock 用户仓库
func NewMockUserRepository() *MockUserRepository {
	return &MockUserRepository{
		users:     make(map[string]*interfaces.User),
		behaviors: make(map[string][]*interfaces.UserBehavior),
	}
}

// GetByID 根据 ID 获取用户
func (m *MockUserRepository) GetByID(ctx context.Context, userID string) (*interfaces.User, error) {
	if m.GetByIDFunc != nil {
		return m.GetByIDFunc(ctx, userID)
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	user, exists := m.users[userID]
	if !exists {
		return nil, fmt.Errorf("user not found: %s", userID)
	}
	return user, nil
}

// GetByIDs 批量获取用户
func (m *MockUserRepository) GetByIDs(ctx context.Context, userIDs []string) ([]*interfaces.User, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	users := make([]*interfaces.User, 0, len(userIDs))
	for _, id := range userIDs {
		if user, exists := m.users[id]; exists {
			users = append(users, user)
		}
	}
	return users, nil
}

// Create 创建用户
func (m *MockUserRepository) Create(ctx context.Context, user *interfaces.User) error {
	if m.CreateFunc != nil {
		return m.CreateFunc(ctx, user)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.users[user.ID]; exists {
		return fmt.Errorf("user already exists: %s", user.ID)
	}
	m.users[user.ID] = user
	return nil
}

// Update 更新用户
func (m *MockUserRepository) Update(ctx context.Context, user *interfaces.User) error {
	if m.UpdateFunc != nil {
		return m.UpdateFunc(ctx, user)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.users[user.ID]; !exists {
		return fmt.Errorf("user not found: %s", user.ID)
	}
	m.users[user.ID] = user
	return nil
}

// Delete 删除用户
func (m *MockUserRepository) Delete(ctx context.Context, userID string) error {
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, userID)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.users[userID]; !exists {
		return fmt.Errorf("user not found: %s", userID)
	}
	delete(m.users, userID)
	return nil
}

// GetBehaviors 获取用户行为
func (m *MockUserRepository) GetBehaviors(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error) {
	if m.GetBehaviorsFunc != nil {
		return m.GetBehaviorsFunc(ctx, userID, limit)
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	behaviors, exists := m.behaviors[userID]
	if !exists {
		return []*interfaces.UserBehavior{}, nil
	}

	if limit > 0 && limit < len(behaviors) {
		return behaviors[:limit], nil
	}
	return behaviors, nil
}

// AddBehavior 添加用户行为
func (m *MockUserRepository) AddBehavior(ctx context.Context, behavior *interfaces.UserBehavior) error {
	if m.AddBehaviorFunc != nil {
		return m.AddBehaviorFunc(ctx, behavior)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.behaviors[behavior.UserID] = append(m.behaviors[behavior.UserID], behavior)
	return nil
}

// GetUserItemInteractions 获取用户与物品的交互记录
func (m *MockUserRepository) GetUserItemInteractions(ctx context.Context, userID, itemID string) ([]*interfaces.UserBehavior, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	behaviors, exists := m.behaviors[userID]
	if !exists {
		return []*interfaces.UserBehavior{}, nil
	}

	interactions := make([]*interfaces.UserBehavior, 0)
	for _, b := range behaviors {
		if b.ItemID == itemID {
			interactions = append(interactions, b)
		}
	}
	return interactions, nil
}

// AddUser 添加测试用户
func (m *MockUserRepository) AddUser(user *interfaces.User) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.users[user.ID] = user
}

// AddTestBehavior 添加测试行为
func (m *MockUserRepository) AddTestBehavior(behavior *interfaces.UserBehavior) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.behaviors[behavior.UserID] = append(m.behaviors[behavior.UserID], behavior)
}

// Clear 清除所有数据
func (m *MockUserRepository) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.users = make(map[string]*interfaces.User)
	m.behaviors = make(map[string][]*interfaces.UserBehavior)
}

// =============================================================================
// Mock 缓存
// =============================================================================

// MockCache Mock 缓存实现
type MockCache struct {
	mu   sync.RWMutex
	data map[string]cacheEntry

	// 用于测试的回调函数
	GetFunc    func(ctx context.Context, key string, value interface{}) error
	SetFunc    func(ctx context.Context, key string, value interface{}, ttl time.Duration) error
	DeleteFunc func(ctx context.Context, key string) error

	// 统计信息
	GetCalls    int
	SetCalls    int
	DeleteCalls int
}

type cacheEntry struct {
	value     interface{}
	expiresAt time.Time
}

// NewMockCache 创建 Mock 缓存
func NewMockCache() *MockCache {
	return &MockCache{
		data: make(map[string]cacheEntry),
	}
}

// Get 获取缓存值
func (m *MockCache) Get(ctx context.Context, key string, value interface{}) error {
	m.mu.Lock()
	m.GetCalls++
	m.mu.Unlock()

	if m.GetFunc != nil {
		return m.GetFunc(ctx, key, value)
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	entry, exists := m.data[key]
	if !exists {
		return fmt.Errorf("cache miss: %s", key)
	}

	if time.Now().After(entry.expiresAt) {
		return fmt.Errorf("cache expired: %s", key)
	}

	// 简单的类型断言处理
	switch v := value.(type) {
	case *interfaces.User:
		if user, ok := entry.value.(*interfaces.User); ok {
			*v = *user
		}
	case *interfaces.UserProfile:
		if profile, ok := entry.value.(*interfaces.UserProfile); ok {
			*v = *profile
		}
	case *[]*interfaces.UserBehavior:
		if behaviors, ok := entry.value.([]*interfaces.UserBehavior); ok {
			*v = behaviors
		}
	}

	return nil
}

// Set 设置缓存值
func (m *MockCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.SetCalls++

	if m.SetFunc != nil {
		return m.SetFunc(ctx, key, value, ttl)
	}

	m.data[key] = cacheEntry{
		value:     value,
		expiresAt: time.Now().Add(ttl),
	}
	return nil
}

// Delete 删除缓存
func (m *MockCache) Delete(ctx context.Context, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.DeleteCalls++

	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, key)
	}

	delete(m.data, key)
	return nil
}

// Exists 检查缓存是否存在
func (m *MockCache) Exists(ctx context.Context, key string) (bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	entry, exists := m.data[key]
	if !exists {
		return false, nil
	}
	return time.Now().Before(entry.expiresAt), nil
}

// MGet 批量获取
func (m *MockCache) MGet(ctx context.Context, keys []string) ([]interface{}, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	results := make([]interface{}, len(keys))
	for i, key := range keys {
		entry, exists := m.data[key]
		if exists && time.Now().Before(entry.expiresAt) {
			results[i] = entry.value
		}
	}
	return results, nil
}

// MSet 批量设置
func (m *MockCache) MSet(ctx context.Context, kvs map[string]interface{}, ttl time.Duration) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	for key, value := range kvs {
		m.data[key] = cacheEntry{
			value:     value,
			expiresAt: time.Now().Add(ttl),
		}
	}
	return nil
}

// SetCacheValue 设置缓存值（测试辅助方法）
func (m *MockCache) SetCacheValue(key string, value interface{}, ttl time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.data[key] = cacheEntry{
		value:     value,
		expiresAt: time.Now().Add(ttl),
	}
}

// Clear 清除所有缓存
func (m *MockCache) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.data = make(map[string]cacheEntry)
	m.GetCalls = 0
	m.SetCalls = 0
	m.DeleteCalls = 0
}

// GetStats 获取统计信息
func (m *MockCache) GetStats() (getCalls, setCalls, deleteCalls int) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.GetCalls, m.SetCalls, m.DeleteCalls
}

// =============================================================================
// 接口编译时检查
// =============================================================================

var _ interfaces.UserRepository = (*MockUserRepository)(nil)
var _ interfaces.Cache = (*MockCache)(nil)

