package database

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
	"recommend-system/pkg/logger"
	"go.uber.org/zap"
)

// RedisClient Redis 客户端封装
type RedisClient struct {
	client redis.UniversalClient
}

// RedisConfig Redis 配置
type RedisConfig struct {
	Addrs        []string
	Password     string
	DB           int
	PoolSize     int
	MinIdleConns int
	DialTimeout  time.Duration
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
	ClusterMode  bool
}

// NewRedisClient 创建 Redis 客户端
func NewRedisClient(cfg *RedisConfig) (*RedisClient, error) {
	var client redis.UniversalClient

	if cfg.ClusterMode {
		// 集群模式
		client = redis.NewClusterClient(&redis.ClusterOptions{
			Addrs:        cfg.Addrs,
			Password:     cfg.Password,
			PoolSize:     cfg.PoolSize,
			MinIdleConns: cfg.MinIdleConns,
			DialTimeout:  cfg.DialTimeout,
			ReadTimeout:  cfg.ReadTimeout,
			WriteTimeout: cfg.WriteTimeout,
		})
	} else {
		// 单机模式
		client = redis.NewClient(&redis.Options{
			Addr:         cfg.Addrs[0],
			Password:     cfg.Password,
			DB:           cfg.DB,
			PoolSize:     cfg.PoolSize,
			MinIdleConns: cfg.MinIdleConns,
			DialTimeout:  cfg.DialTimeout,
			ReadTimeout:  cfg.ReadTimeout,
			WriteTimeout: cfg.WriteTimeout,
		})
	}

	// 测试连接
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to ping redis: %w", err)
	}

	mode := "standalone"
	if cfg.ClusterMode {
		mode = "cluster"
	}

	logger.Info("Redis connected",
		zap.Strings("addrs", cfg.Addrs),
		zap.String("mode", mode),
	)

	return &RedisClient{client: client}, nil
}

// Client 返回原始客户端
func (r *RedisClient) Client() redis.UniversalClient {
	return r.client
}

// Close 关闭连接
func (r *RedisClient) Close() error {
	if r.client != nil {
		logger.Info("Redis connection closed")
		return r.client.Close()
	}
	return nil
}

// Ping 测试连接
func (r *RedisClient) Ping(ctx context.Context) error {
	return r.client.Ping(ctx).Err()
}

// ========== String 操作 ==========

// Get 获取字符串值
func (r *RedisClient) Get(ctx context.Context, key string) (string, error) {
	return r.client.Get(ctx, key).Result()
}

// Set 设置字符串值
func (r *RedisClient) Set(ctx context.Context, key string, value interface{}, expiration time.Duration) error {
	return r.client.Set(ctx, key, value, expiration).Err()
}

// SetNX 仅当 key 不存在时设置
func (r *RedisClient) SetNX(ctx context.Context, key string, value interface{}, expiration time.Duration) (bool, error) {
	return r.client.SetNX(ctx, key, value, expiration).Result()
}

// Del 删除 key
func (r *RedisClient) Del(ctx context.Context, keys ...string) error {
	return r.client.Del(ctx, keys...).Err()
}

// Exists 检查 key 是否存在
func (r *RedisClient) Exists(ctx context.Context, keys ...string) (int64, error) {
	return r.client.Exists(ctx, keys...).Result()
}

// Expire 设置过期时间
func (r *RedisClient) Expire(ctx context.Context, key string, expiration time.Duration) error {
	return r.client.Expire(ctx, key, expiration).Err()
}

// TTL 获取 key 的剩余过期时间
func (r *RedisClient) TTL(ctx context.Context, key string) (time.Duration, error) {
	return r.client.TTL(ctx, key).Result()
}

// ========== Hash 操作 ==========

// HGet 获取 hash 字段值
func (r *RedisClient) HGet(ctx context.Context, key, field string) (string, error) {
	return r.client.HGet(ctx, key, field).Result()
}

// HSet 设置 hash 字段
func (r *RedisClient) HSet(ctx context.Context, key string, values ...interface{}) error {
	return r.client.HSet(ctx, key, values...).Err()
}

// HGetAll 获取所有 hash 字段
func (r *RedisClient) HGetAll(ctx context.Context, key string) (map[string]string, error) {
	return r.client.HGetAll(ctx, key).Result()
}

// HMSet 批量设置 hash 字段
func (r *RedisClient) HMSet(ctx context.Context, key string, values ...interface{}) error {
	return r.client.HMSet(ctx, key, values...).Err()
}

// HDel 删除 hash 字段
func (r *RedisClient) HDel(ctx context.Context, key string, fields ...string) error {
	return r.client.HDel(ctx, key, fields...).Err()
}

// ========== List 操作 ==========

// LPush 左侧推入列表
func (r *RedisClient) LPush(ctx context.Context, key string, values ...interface{}) error {
	return r.client.LPush(ctx, key, values...).Err()
}

// RPush 右侧推入列表
func (r *RedisClient) RPush(ctx context.Context, key string, values ...interface{}) error {
	return r.client.RPush(ctx, key, values...).Err()
}

// LRange 获取列表范围
func (r *RedisClient) LRange(ctx context.Context, key string, start, stop int64) ([]string, error) {
	return r.client.LRange(ctx, key, start, stop).Result()
}

// LTrim 裁剪列表
func (r *RedisClient) LTrim(ctx context.Context, key string, start, stop int64) error {
	return r.client.LTrim(ctx, key, start, stop).Err()
}

// LLen 获取列表长度
func (r *RedisClient) LLen(ctx context.Context, key string) (int64, error) {
	return r.client.LLen(ctx, key).Result()
}

// ========== Sorted Set 操作 ==========

// ZAdd 添加有序集合成员
func (r *RedisClient) ZAdd(ctx context.Context, key string, members ...redis.Z) error {
	return r.client.ZAdd(ctx, key, members...).Err()
}

// ZRange 获取有序集合范围
func (r *RedisClient) ZRange(ctx context.Context, key string, start, stop int64) ([]string, error) {
	return r.client.ZRange(ctx, key, start, stop).Result()
}

// ZRevRange 倒序获取有序集合范围
func (r *RedisClient) ZRevRange(ctx context.Context, key string, start, stop int64) ([]string, error) {
	return r.client.ZRevRange(ctx, key, start, stop).Result()
}

// ZRevRangeWithScores 倒序获取有序集合范围及分数
func (r *RedisClient) ZRevRangeWithScores(ctx context.Context, key string, start, stop int64) ([]redis.Z, error) {
	return r.client.ZRevRangeWithScores(ctx, key, start, stop).Result()
}

// ZRem 删除有序集合成员
func (r *RedisClient) ZRem(ctx context.Context, key string, members ...interface{}) error {
	return r.client.ZRem(ctx, key, members...).Err()
}

// ZCard 获取有序集合成员数量
func (r *RedisClient) ZCard(ctx context.Context, key string) (int64, error) {
	return r.client.ZCard(ctx, key).Result()
}

// ========== Pipeline 操作 ==========

// Pipeline 创建 Pipeline
func (r *RedisClient) Pipeline() redis.Pipeliner {
	return r.client.Pipeline()
}

// TxPipeline 创建事务 Pipeline
func (r *RedisClient) TxPipeline() redis.Pipeliner {
	return r.client.TxPipeline()
}

// ========== 分布式锁 ==========

// Lock 获取分布式锁
func (r *RedisClient) Lock(ctx context.Context, key string, value interface{}, expiration time.Duration) (bool, error) {
	return r.client.SetNX(ctx, key, value, expiration).Result()
}

// Unlock 释放分布式锁 (使用 Lua 脚本保证原子性)
func (r *RedisClient) Unlock(ctx context.Context, key string, value interface{}) (bool, error) {
	script := `
		if redis.call("get", KEYS[1]) == ARGV[1] then
			return redis.call("del", KEYS[1])
		else
			return 0
		end
	`
	result, err := r.client.Eval(ctx, script, []string{key}, value).Result()
	if err != nil {
		return false, err
	}
	return result.(int64) == 1, nil
}

