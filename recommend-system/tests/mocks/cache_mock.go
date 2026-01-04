package mocks

import (
	"context"
	"encoding/json"
	"sync"
	"time"
)

// =============================================================================
// MockCache - 缓存 Mock 实现
// =============================================================================

// cacheEntry 缓存条目
type cacheEntry struct {
	data      []byte
	expiresAt time.Time
}

// MockCache Mock 缓存
//
// 实现 interfaces.Cache 接口
// 使用内存存储模拟 Redis 等缓存服务
type MockCache struct {
	mu   sync.RWMutex
	data map[string]*cacheEntry

	// 调用计数器
	GetCalls    int
	SetCalls    int
	DeleteCalls int
	ExistsCalls int
	MGetCalls   int
	MSetCalls   int

	// 可配置的错误
	GetError    error
	SetError    error
	DeleteError error
	MGetError   error
	MSetError   error

	// 模拟延迟
	Latency time.Duration
}

// NewMockCache 创建 Mock 缓存实例
func NewMockCache() *MockCache {
	return &MockCache{
		data: make(map[string]*cacheEntry),
	}
}

// Get 获取缓存值
//
// 实现 interfaces.Cache.Get
func (m *MockCache) Get(ctx context.Context, key string, value interface{}) error {
	m.mu.Lock()
	m.GetCalls++
	m.mu.Unlock()

	// 模拟延迟
	if m.Latency > 0 {
		time.Sleep(m.Latency)
	}

	if m.GetError != nil {
		return m.GetError
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	entry, ok := m.data[key]
	if !ok {
		return ErrCacheMiss
	}

	// 检查是否过期
	if !entry.expiresAt.IsZero() && time.Now().After(entry.expiresAt) {
		return ErrCacheMiss
	}

	// 反序列化
	return json.Unmarshal(entry.data, value)
}

// Set 设置缓存值
//
// 实现 interfaces.Cache.Set
func (m *MockCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.SetCalls++

	// 模拟延迟
	if m.Latency > 0 {
		time.Sleep(m.Latency)
	}

	if m.SetError != nil {
		return m.SetError
	}

	// 序列化
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}

	var expiresAt time.Time
	if ttl > 0 {
		expiresAt = time.Now().Add(ttl)
	}

	m.data[key] = &cacheEntry{
		data:      data,
		expiresAt: expiresAt,
	}

	return nil
}

// Delete 删除缓存
//
// 实现 interfaces.Cache.Delete
func (m *MockCache) Delete(ctx context.Context, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.DeleteCalls++

	if m.DeleteError != nil {
		return m.DeleteError
	}

	delete(m.data, key)
	return nil
}

// Exists 检查缓存是否存在
//
// 实现 interfaces.Cache.Exists
func (m *MockCache) Exists(ctx context.Context, key string) (bool, error) {
	m.mu.Lock()
	m.ExistsCalls++
	m.mu.Unlock()

	m.mu.RLock()
	defer m.mu.RUnlock()

	entry, ok := m.data[key]
	if !ok {
		return false, nil
	}

	// 检查是否过期
	if !entry.expiresAt.IsZero() && time.Now().After(entry.expiresAt) {
		return false, nil
	}

	return true, nil
}

// MGet 批量获取缓存
//
// 实现 interfaces.Cache.MGet
func (m *MockCache) MGet(ctx context.Context, keys []string) ([]interface{}, error) {
	m.mu.Lock()
	m.MGetCalls++
	m.mu.Unlock()

	if m.MGetError != nil {
		return nil, m.MGetError
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	results := make([]interface{}, len(keys))
	for i, key := range keys {
		entry, ok := m.data[key]
		if !ok {
			results[i] = nil
			continue
		}

		// 检查是否过期
		if !entry.expiresAt.IsZero() && time.Now().After(entry.expiresAt) {
			results[i] = nil
			continue
		}

		var value interface{}
		if err := json.Unmarshal(entry.data, &value); err != nil {
			results[i] = nil
			continue
		}
		results[i] = value
	}

	return results, nil
}

// MSet 批量设置缓存
//
// 实现 interfaces.Cache.MSet
func (m *MockCache) MSet(ctx context.Context, kvs map[string]interface{}, ttl time.Duration) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.MSetCalls++

	if m.MSetError != nil {
		return m.MSetError
	}

	var expiresAt time.Time
	if ttl > 0 {
		expiresAt = time.Now().Add(ttl)
	}

	for key, value := range kvs {
		data, err := json.Marshal(value)
		if err != nil {
			continue
		}

		m.data[key] = &cacheEntry{
			data:      data,
			expiresAt: expiresAt,
		}
	}

	return nil
}

// =============================================================================
// 测试辅助方法
// =============================================================================

// Clear 清空所有缓存（测试用）
func (m *MockCache) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.data = make(map[string]*cacheEntry)
}

// Reset 重置所有状态（测试用）
func (m *MockCache) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.data = make(map[string]*cacheEntry)
	m.GetCalls = 0
	m.SetCalls = 0
	m.DeleteCalls = 0
	m.ExistsCalls = 0
	m.MGetCalls = 0
	m.MSetCalls = 0
	m.GetError = nil
	m.SetError = nil
	m.DeleteError = nil
	m.MGetError = nil
	m.MSetError = nil
	m.Latency = 0
}

// KeyExists 检查 key 是否存在（不增加计数器，仅测试用）
func (m *MockCache) KeyExists(key string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	entry, ok := m.data[key]
	if !ok {
		return false
	}

	// 检查是否过期
	if !entry.expiresAt.IsZero() && time.Now().After(entry.expiresAt) {
		return false
	}

	return true
}

// GetRawData 获取原始数据（测试用）
func (m *MockCache) GetRawData(key string) ([]byte, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	entry, ok := m.data[key]
	if !ok {
		return nil, false
	}

	return entry.data, true
}

// GetKeyCount 获取缓存键数量（测试用）
func (m *MockCache) GetKeyCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	count := 0
	now := time.Now()
	for _, entry := range m.data {
		if entry.expiresAt.IsZero() || now.Before(entry.expiresAt) {
			count++
		}
	}
	return count
}

// SetDirect 直接设置值（不计数，测试用）
func (m *MockCache) SetDirect(key string, value interface{}, ttl time.Duration) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	data, err := json.Marshal(value)
	if err != nil {
		return err
	}

	var expiresAt time.Time
	if ttl > 0 {
		expiresAt = time.Now().Add(ttl)
	}

	m.data[key] = &cacheEntry{
		data:      data,
		expiresAt: expiresAt,
	}

	return nil
}

