// Package cache 提供多级缓存管理
package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"recommend-system/pkg/database"
)

// Cache 缓存接口
type Cache interface {
	Get(ctx context.Context, key string) ([]byte, error)
	Set(ctx context.Context, key string, value []byte, ttl time.Duration) error
	Delete(ctx context.Context, key string) error
	Exists(ctx context.Context, key string) (bool, error)
}

// MultiLevelCache 多级缓存
type MultiLevelCache struct {
	local  *LocalCache
	redis  *database.RedisClient
	prefix string
}

// NewMultiLevelCache 创建多级缓存
func NewMultiLevelCache(redis *database.RedisClient, prefix string, localSize int, localTTL time.Duration) *MultiLevelCache {
	return &MultiLevelCache{
		local:  NewLocalCache(localSize, localTTL),
		redis:  redis,
		prefix: prefix,
	}
}

// Get 获取缓存 (L1 -> L2)
func (m *MultiLevelCache) Get(ctx context.Context, key string) ([]byte, error) {
	fullKey := m.prefix + key

	// L1: 本地缓存
	if data, found := m.local.Get(fullKey); found {
		return data, nil
	}

	// L2: Redis
	data, err := m.redis.Get(ctx, fullKey)
	if err != nil {
		return nil, err
	}

	// 回填 L1
	m.local.Set(fullKey, []byte(data))

	return []byte(data), nil
}

// Set 设置缓存 (同时写入 L1 和 L2)
func (m *MultiLevelCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	fullKey := m.prefix + key

	// L1
	m.local.Set(fullKey, value)

	// L2
	return m.redis.Set(ctx, fullKey, value, ttl)
}

// Delete 删除缓存
func (m *MultiLevelCache) Delete(ctx context.Context, key string) error {
	fullKey := m.prefix + key

	// L1
	m.local.Delete(fullKey)

	// L2
	return m.redis.Del(ctx, fullKey)
}

// Exists 检查缓存是否存在
func (m *MultiLevelCache) Exists(ctx context.Context, key string) (bool, error) {
	fullKey := m.prefix + key

	// L1
	if m.local.Exists(fullKey) {
		return true, nil
	}

	// L2
	count, err := m.redis.Exists(ctx, fullKey)
	return count > 0, err
}

// GetOrSet 获取或设置缓存
func (m *MultiLevelCache) GetOrSet(ctx context.Context, key string, ttl time.Duration, loader func() ([]byte, error)) ([]byte, error) {
	// 尝试获取
	data, err := m.Get(ctx, key)
	if err == nil && len(data) > 0 {
		return data, nil
	}

	// 加载数据
	data, err = loader()
	if err != nil {
		return nil, err
	}

	// 设置缓存
	if err := m.Set(ctx, key, data, ttl); err != nil {
		// 日志警告但不返回错误
	}

	return data, nil
}

// LocalCache 本地缓存 (LRU)
type LocalCache struct {
	mu       sync.RWMutex
	items    map[string]*cacheItem
	maxSize  int
	defaultTTL time.Duration
}

type cacheItem struct {
	data      []byte
	expiresAt time.Time
}

// NewLocalCache 创建本地缓存
func NewLocalCache(maxSize int, defaultTTL time.Duration) *LocalCache {
	c := &LocalCache{
		items:      make(map[string]*cacheItem),
		maxSize:    maxSize,
		defaultTTL: defaultTTL,
	}

	// 启动清理 goroutine
	go c.cleanup()

	return c
}

// Get 获取缓存
func (c *LocalCache) Get(key string) ([]byte, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	item, exists := c.items[key]
	if !exists {
		return nil, false
	}

	if time.Now().After(item.expiresAt) {
		return nil, false
	}

	return item.data, true
}

// Set 设置缓存
func (c *LocalCache) Set(key string, data []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// 简单的大小控制
	if len(c.items) >= c.maxSize {
		c.evict()
	}

	c.items[key] = &cacheItem{
		data:      data,
		expiresAt: time.Now().Add(c.defaultTTL),
	}
}

// Delete 删除缓存
func (c *LocalCache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.items, key)
}

// Exists 检查是否存在
func (c *LocalCache) Exists(key string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	item, exists := c.items[key]
	if !exists {
		return false
	}

	return time.Now().Before(item.expiresAt)
}

// evict 淘汰旧数据
func (c *LocalCache) evict() {
	// 简单策略：删除 10% 最旧的
	toDelete := c.maxSize / 10
	if toDelete < 1 {
		toDelete = 1
	}

	for key := range c.items {
		delete(c.items, key)
		toDelete--
		if toDelete <= 0 {
			break
		}
	}
}

// cleanup 定期清理过期数据
func (c *LocalCache) cleanup() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		c.mu.Lock()
		now := time.Now()
		for key, item := range c.items {
			if now.After(item.expiresAt) {
				delete(c.items, key)
			}
		}
		c.mu.Unlock()
	}
}

// UserProfileCache 用户画像缓存
type UserProfileCache struct {
	cache *MultiLevelCache
	ttl   time.Duration
}

// NewUserProfileCache 创建用户画像缓存
func NewUserProfileCache(cache *MultiLevelCache, ttl time.Duration) *UserProfileCache {
	return &UserProfileCache{
		cache: cache,
		ttl:   ttl,
	}
}

// Get 获取用户画像
func (c *UserProfileCache) Get(ctx context.Context, userID string) (map[string]interface{}, error) {
	key := fmt.Sprintf("user:profile:%s", userID)
	data, err := c.cache.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	var profile map[string]interface{}
	if err := json.Unmarshal(data, &profile); err != nil {
		return nil, err
	}

	return profile, nil
}

// Set 设置用户画像
func (c *UserProfileCache) Set(ctx context.Context, userID string, profile map[string]interface{}) error {
	key := fmt.Sprintf("user:profile:%s", userID)
	data, err := json.Marshal(profile)
	if err != nil {
		return err
	}

	return c.cache.Set(ctx, key, data, c.ttl)
}

// ItemEmbeddingCache 物品嵌入缓存
type ItemEmbeddingCache struct {
	cache *MultiLevelCache
	ttl   time.Duration
}

// NewItemEmbeddingCache 创建物品嵌入缓存
func NewItemEmbeddingCache(cache *MultiLevelCache, ttl time.Duration) *ItemEmbeddingCache {
	return &ItemEmbeddingCache{
		cache: cache,
		ttl:   ttl,
	}
}

// Get 获取物品嵌入
func (c *ItemEmbeddingCache) Get(ctx context.Context, itemID string) ([]float32, error) {
	key := fmt.Sprintf("item:embedding:%s", itemID)
	data, err := c.cache.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	var embedding []float32
	if err := json.Unmarshal(data, &embedding); err != nil {
		return nil, err
	}

	return embedding, nil
}

// Set 设置物品嵌入
func (c *ItemEmbeddingCache) Set(ctx context.Context, itemID string, embedding []float32) error {
	key := fmt.Sprintf("item:embedding:%s", itemID)
	data, err := json.Marshal(embedding)
	if err != nil {
		return err
	}

	return c.cache.Set(ctx, key, data, c.ttl)
}

// BatchGet 批量获取物品嵌入
func (c *ItemEmbeddingCache) BatchGet(ctx context.Context, itemIDs []string) (map[string][]float32, []string, error) {
	found := make(map[string][]float32)
	missing := make([]string, 0)

	for _, itemID := range itemIDs {
		embedding, err := c.Get(ctx, itemID)
		if err != nil {
			missing = append(missing, itemID)
		} else {
			found[itemID] = embedding
		}
	}

	return found, missing, nil
}

