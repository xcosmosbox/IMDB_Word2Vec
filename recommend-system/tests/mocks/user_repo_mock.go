package mocks

import (
	"context"
	"sync"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// MockUserRepository - 用户仓库 Mock 实现
// =============================================================================

// MockUserRepository Mock 用户仓库
//
// 实现 interfaces.UserRepository 接口
// 使用内存存储模拟数据库操作，支持并发安全访问
type MockUserRepository struct {
	mu        sync.RWMutex
	users     map[string]*interfaces.User
	behaviors map[string][]*interfaces.UserBehavior

	// 调用计数器（用于验证调用次数）
	GetByIDCalls    int
	GetByIDsCalls   int
	CreateCalls     int
	UpdateCalls     int
	DeleteCalls     int
	GetBehaviorsCalls int
	AddBehaviorCalls  int

	// 可配置的错误（用于模拟错误场景）
	GetByIDError    error
	CreateError     error
	UpdateError     error
	DeleteError     error
	AddBehaviorError error
}

// NewMockUserRepository 创建 Mock 用户仓库实例
func NewMockUserRepository() *MockUserRepository {
	return &MockUserRepository{
		users:     make(map[string]*interfaces.User),
		behaviors: make(map[string][]*interfaces.UserBehavior),
	}
}

// GetByID 根据ID获取用户
//
// 实现 interfaces.UserRepository.GetByID
func (m *MockUserRepository) GetByID(ctx context.Context, userID string) (*interfaces.User, error) {
	m.mu.Lock()
	m.GetByIDCalls++
	m.mu.Unlock()

	// 检查是否有预设错误
	if m.GetByIDError != nil {
		return nil, m.GetByIDError
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	user, ok := m.users[userID]
	if !ok {
		return nil, ErrNotFound
	}

	// 返回副本，避免外部修改影响内部状态
	userCopy := *user
	return &userCopy, nil
}

// GetByIDs 批量获取用户
//
// 实现 interfaces.UserRepository.GetByIDs
func (m *MockUserRepository) GetByIDs(ctx context.Context, userIDs []string) ([]*interfaces.User, error) {
	m.mu.Lock()
	m.GetByIDsCalls++
	m.mu.Unlock()

	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]*interfaces.User, 0, len(userIDs))
	for _, id := range userIDs {
		if user, ok := m.users[id]; ok {
			userCopy := *user
			result = append(result, &userCopy)
		}
	}

	return result, nil
}

// Create 创建用户
//
// 实现 interfaces.UserRepository.Create
func (m *MockUserRepository) Create(ctx context.Context, user *interfaces.User) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.CreateCalls++

	// 检查是否有预设错误
	if m.CreateError != nil {
		return m.CreateError
	}

	// 检查是否已存在
	if _, exists := m.users[user.ID]; exists {
		return ErrDuplicate
	}

	// 保存副本
	userCopy := *user
	m.users[user.ID] = &userCopy

	return nil
}

// Update 更新用户
//
// 实现 interfaces.UserRepository.Update
func (m *MockUserRepository) Update(ctx context.Context, user *interfaces.User) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.UpdateCalls++

	// 检查是否有预设错误
	if m.UpdateError != nil {
		return m.UpdateError
	}

	// 检查是否存在
	if _, exists := m.users[user.ID]; !exists {
		return ErrNotFound
	}

	// 保存副本
	userCopy := *user
	m.users[user.ID] = &userCopy

	return nil
}

// Delete 删除用户
//
// 实现 interfaces.UserRepository.Delete
func (m *MockUserRepository) Delete(ctx context.Context, userID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.DeleteCalls++

	// 检查是否有预设错误
	if m.DeleteError != nil {
		return m.DeleteError
	}

	// 删除用户和相关行为
	delete(m.users, userID)
	delete(m.behaviors, userID)

	return nil
}

// GetBehaviors 获取用户行为历史
//
// 实现 interfaces.UserRepository.GetBehaviors
func (m *MockUserRepository) GetBehaviors(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error) {
	m.mu.Lock()
	m.GetBehaviorsCalls++
	m.mu.Unlock()

	m.mu.RLock()
	defer m.mu.RUnlock()

	behaviors, ok := m.behaviors[userID]
	if !ok {
		return []*interfaces.UserBehavior{}, nil
	}

	// 限制返回数量
	if limit > 0 && len(behaviors) > limit {
		behaviors = behaviors[:limit]
	}

	// 返回副本
	result := make([]*interfaces.UserBehavior, len(behaviors))
	for i, b := range behaviors {
		bCopy := *b
		result[i] = &bCopy
	}

	return result, nil
}

// AddBehavior 添加用户行为
//
// 实现 interfaces.UserRepository.AddBehavior
func (m *MockUserRepository) AddBehavior(ctx context.Context, behavior *interfaces.UserBehavior) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.AddBehaviorCalls++

	// 检查是否有预设错误
	if m.AddBehaviorError != nil {
		return m.AddBehaviorError
	}

	// 保存副本
	bCopy := *behavior
	m.behaviors[behavior.UserID] = append(m.behaviors[behavior.UserID], &bCopy)

	return nil
}

// GetUserItemInteractions 获取用户与物品的交互记录
//
// 实现 interfaces.UserRepository.GetUserItemInteractions
func (m *MockUserRepository) GetUserItemInteractions(ctx context.Context, userID, itemID string) ([]*interfaces.UserBehavior, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	behaviors, ok := m.behaviors[userID]
	if !ok {
		return []*interfaces.UserBehavior{}, nil
	}

	result := make([]*interfaces.UserBehavior, 0)
	for _, b := range behaviors {
		if b.ItemID == itemID {
			bCopy := *b
			result = append(result, &bCopy)
		}
	}

	return result, nil
}

// =============================================================================
// 测试辅助方法
// =============================================================================

// SetUser 设置用户数据（测试用）
//
// 直接设置内存中的用户数据，用于测试前的数据准备
func (m *MockUserRepository) SetUser(user *interfaces.User) {
	m.mu.Lock()
	defer m.mu.Unlock()

	userCopy := *user
	m.users[user.ID] = &userCopy
}

// SetUsers 批量设置用户数据（测试用）
func (m *MockUserRepository) SetUsers(users []*interfaces.User) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, user := range users {
		userCopy := *user
		m.users[user.ID] = &userCopy
	}
}

// SetBehaviors 设置用户行为数据（测试用）
//
// 直接设置内存中的行为数据，用于测试前的数据准备
func (m *MockUserRepository) SetBehaviors(userID string, behaviors []*interfaces.UserBehavior) {
	m.mu.Lock()
	defer m.mu.Unlock()

	copies := make([]*interfaces.UserBehavior, len(behaviors))
	for i, b := range behaviors {
		bCopy := *b
		copies[i] = &bCopy
	}
	m.behaviors[userID] = copies
}

// Reset 重置所有状态（测试用）
//
// 清空所有数据和计数器
func (m *MockUserRepository) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.users = make(map[string]*interfaces.User)
	m.behaviors = make(map[string][]*interfaces.UserBehavior)
	m.GetByIDCalls = 0
	m.GetByIDsCalls = 0
	m.CreateCalls = 0
	m.UpdateCalls = 0
	m.DeleteCalls = 0
	m.GetBehaviorsCalls = 0
	m.AddBehaviorCalls = 0
	m.GetByIDError = nil
	m.CreateError = nil
	m.UpdateError = nil
	m.DeleteError = nil
	m.AddBehaviorError = nil
}

// GetUserCount 获取用户数量（测试用）
func (m *MockUserRepository) GetUserCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return len(m.users)
}

// GetBehaviorCount 获取指定用户的行为数量（测试用）
func (m *MockUserRepository) GetBehaviorCount(userID string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return len(m.behaviors[userID])
}

