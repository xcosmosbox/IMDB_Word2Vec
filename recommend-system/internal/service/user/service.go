// Package user 提供用户服务的业务逻辑实现
//
// 用户服务是推荐系统的核心服务之一，负责：
// - 用户信息的 CRUD 操作
// - 用户行为记录与查询
// - 用户画像管理
//
// 本模块实现了 interfaces.UserService 接口，
// 通过依赖注入实现与存储层和缓存层的解耦。
package user

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"go.uber.org/zap"

	"recommend-system/internal/interfaces"
	"recommend-system/pkg/logger"
)

// =============================================================================
// 常量定义
// =============================================================================

const (
	// 缓存 Key 前缀
	cacheKeyPrefixUser     = "user:"
	cacheKeyPrefixProfile  = "user:profile:"
	cacheKeyPrefixBehavior = "user:behavior:"

	// 缓存过期时间
	userCacheTTL     = 30 * time.Minute
	profileCacheTTL  = 15 * time.Minute
	behaviorCacheTTL = 5 * time.Minute

	// 默认限制
	defaultBehaviorLimit = 100
	maxBehaviorLimit     = 1000
)

// =============================================================================
// 错误定义
// =============================================================================

var (
	// ErrUserNotFound 用户不存在
	ErrUserNotFound = fmt.Errorf("user not found")
	// ErrInvalidRequest 无效请求
	ErrInvalidRequest = fmt.Errorf("invalid request")
	// ErrUserAlreadyExists 用户已存在
	ErrUserAlreadyExists = fmt.Errorf("user already exists")
)

// =============================================================================
// Service 实现
// =============================================================================

// Service 用户服务实现
//
// 实现 interfaces.UserService 接口，提供用户相关的所有业务逻辑。
// 使用依赖注入的方式注入 Repository 和 Cache，便于测试和解耦。
type Service struct {
	userRepo interfaces.UserRepository // 用户数据仓库
	cache    interfaces.Cache          // 缓存服务
	logger   *zap.Logger               // 日志记录器
}

// NewService 创建用户服务实例
//
// 参数:
//   - userRepo: 用户数据仓库接口实现
//   - cache: 缓存接口实现
//   - log: 可选的日志记录器，如果为 nil 则使用全局 logger
//
// 返回:
//   - *Service: 用户服务实例
func NewService(userRepo interfaces.UserRepository, cache interfaces.Cache, log *zap.Logger) *Service {
	if log == nil {
		log = logger.Logger
	}
	if log == nil {
		log = zap.NewNop()
	}

	return &Service{
		userRepo: userRepo,
		cache:    cache,
		logger:   log.Named("user-service"),
	}
}

// =============================================================================
// UserService 接口实现
// =============================================================================

// GetUser 获取用户信息
//
// 实现 interfaces.UserService.GetUser 方法
// 首先尝试从缓存获取，缓存未命中则查询数据库并回填缓存
//
// 参数:
//   - ctx: 上下文，用于超时控制和取消
//   - userID: 用户唯一标识
//
// 返回:
//   - *interfaces.User: 用户信息
//   - error: 错误信息，用户不存在时返回 ErrUserNotFound
func (s *Service) GetUser(ctx context.Context, userID string) (*interfaces.User, error) {
	// 参数校验
	if userID == "" {
		return nil, ErrInvalidRequest
	}

	// 1. 尝试从缓存获取
	cacheKey := cacheKeyPrefixUser + userID
	var user interfaces.User
	err := s.cache.Get(ctx, cacheKey, &user)
	if err == nil {
		s.logger.Debug("cache hit for user",
			zap.String("user_id", userID),
		)
		return &user, nil
	}

	// 2. 缓存未命中，查询数据库
	s.logger.Debug("cache miss, querying database",
		zap.String("user_id", userID),
	)

	userPtr, err := s.userRepo.GetByID(ctx, userID)
	if err != nil {
		s.logger.Error("failed to get user from database",
			zap.String("user_id", userID),
			zap.Error(err),
		)
		return nil, fmt.Errorf("%w: %s", ErrUserNotFound, userID)
	}

	// 3. 写入缓存（忽略缓存写入错误，不影响主流程）
	if cacheErr := s.cache.Set(ctx, cacheKey, userPtr, userCacheTTL); cacheErr != nil {
		s.logger.Warn("failed to set user cache",
			zap.String("user_id", userID),
			zap.Error(cacheErr),
		)
	}

	return userPtr, nil
}

// CreateUser 创建用户
//
// 实现 interfaces.UserService.CreateUser 方法
// 生成唯一用户ID，创建用户记录
//
// 参数:
//   - ctx: 上下文
//   - req: 创建用户请求，包含用户基本信息
//
// 返回:
//   - *interfaces.User: 创建成功的用户信息
//   - error: 错误信息
func (s *Service) CreateUser(ctx context.Context, req *interfaces.CreateUserRequest) (*interfaces.User, error) {
	// 参数校验
	if req == nil {
		return nil, ErrInvalidRequest
	}
	if req.Name == "" || req.Email == "" {
		return nil, fmt.Errorf("%w: name and email are required", ErrInvalidRequest)
	}

	// 生成唯一用户 ID
	userID, err := generateUserID()
	if err != nil {
		s.logger.Error("failed to generate user ID", zap.Error(err))
		return nil, fmt.Errorf("failed to generate user ID: %w", err)
	}

	// 构建用户对象
	now := time.Now()
	user := &interfaces.User{
		ID:        userID,
		Name:      req.Name,
		Email:     req.Email,
		Age:       req.Age,
		Gender:    req.Gender,
		Metadata:  make(map[string]string),
		CreatedAt: now,
		UpdatedAt: now,
	}

	// 保存到数据库
	if err := s.userRepo.Create(ctx, user); err != nil {
		s.logger.Error("failed to create user",
			zap.String("email", req.Email),
			zap.Error(err),
		)
		return nil, fmt.Errorf("failed to create user: %w", err)
	}

	s.logger.Info("user created successfully",
		zap.String("user_id", userID),
		zap.String("email", req.Email),
	)

	return user, nil
}

// UpdateUser 更新用户信息
//
// 实现 interfaces.UserService.UpdateUser 方法
// 只更新请求中非空的字段，更新后清除缓存
//
// 参数:
//   - ctx: 上下文
//   - userID: 用户唯一标识
//   - req: 更新请求，包含要更新的字段
//
// 返回:
//   - *interfaces.User: 更新后的用户信息
//   - error: 错误信息
func (s *Service) UpdateUser(ctx context.Context, userID string, req *interfaces.UpdateUserRequest) (*interfaces.User, error) {
	// 参数校验
	if userID == "" || req == nil {
		return nil, ErrInvalidRequest
	}

	// 获取现有用户信息
	user, err := s.userRepo.GetByID(ctx, userID)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrUserNotFound, userID)
	}

	// 更新非空字段
	if req.Name != "" {
		user.Name = req.Name
	}
	if req.Email != "" {
		user.Email = req.Email
	}
	if req.Age > 0 {
		user.Age = req.Age
	}
	if req.Gender != "" {
		user.Gender = req.Gender
	}
	user.UpdatedAt = time.Now()

	// 保存更新
	if err := s.userRepo.Update(ctx, user); err != nil {
		s.logger.Error("failed to update user",
			zap.String("user_id", userID),
			zap.Error(err),
		)
		return nil, fmt.Errorf("failed to update user: %w", err)
	}

	// 清除缓存
	cacheKey := cacheKeyPrefixUser + userID
	if err := s.cache.Delete(ctx, cacheKey); err != nil {
		s.logger.Warn("failed to delete user cache",
			zap.String("user_id", userID),
			zap.Error(err),
		)
	}

	s.logger.Info("user updated successfully",
		zap.String("user_id", userID),
	)

	return user, nil
}

// DeleteUser 删除用户
//
// 实现 interfaces.UserService.DeleteUser 方法
// 删除用户记录并清除相关缓存
//
// 参数:
//   - ctx: 上下文
//   - userID: 用户唯一标识
//
// 返回:
//   - error: 错误信息
func (s *Service) DeleteUser(ctx context.Context, userID string) error {
	// 参数校验
	if userID == "" {
		return ErrInvalidRequest
	}

	// 删除数据库记录
	if err := s.userRepo.Delete(ctx, userID); err != nil {
		s.logger.Error("failed to delete user",
			zap.String("user_id", userID),
			zap.Error(err),
		)
		return fmt.Errorf("failed to delete user: %w", err)
	}

	// 清除所有相关缓存
	s.clearUserCaches(ctx, userID)

	s.logger.Info("user deleted successfully",
		zap.String("user_id", userID),
	)

	return nil
}

// RecordBehavior 记录用户行为
//
// 实现 interfaces.UserService.RecordBehavior 方法
// 记录用户与物品的交互行为，用于后续推荐
//
// 参数:
//   - ctx: 上下文
//   - req: 行为记录请求
//
// 返回:
//   - error: 错误信息
func (s *Service) RecordBehavior(ctx context.Context, req *interfaces.RecordBehaviorRequest) error {
	// 参数校验
	if req == nil || req.UserID == "" || req.ItemID == "" || req.Action == "" {
		return fmt.Errorf("%w: user_id, item_id, and action are required", ErrInvalidRequest)
	}

	// 构建行为记录
	behavior := &interfaces.UserBehavior{
		UserID:    req.UserID,
		ItemID:    req.ItemID,
		Action:    req.Action,
		Timestamp: time.Now(),
		Context:   req.Context,
	}

	// 保存到数据库
	if err := s.userRepo.AddBehavior(ctx, behavior); err != nil {
		s.logger.Error("failed to record behavior",
			zap.String("user_id", req.UserID),
			zap.String("item_id", req.ItemID),
			zap.String("action", req.Action),
			zap.Error(err),
		)
		return fmt.Errorf("failed to record behavior: %w", err)
	}

	// 清除行为缓存（如果有）
	behaviorCacheKey := cacheKeyPrefixBehavior + req.UserID
	_ = s.cache.Delete(ctx, behaviorCacheKey)

	s.logger.Debug("behavior recorded",
		zap.String("user_id", req.UserID),
		zap.String("item_id", req.ItemID),
		zap.String("action", req.Action),
	)

	return nil
}

// GetUserBehaviors 获取用户行为历史
//
// 实现 interfaces.UserService.GetUserBehaviors 方法
// 获取指定用户的历史行为记录，按时间倒序排列
//
// 参数:
//   - ctx: 上下文
//   - userID: 用户唯一标识
//   - limit: 返回的最大记录数
//
// 返回:
//   - []*interfaces.UserBehavior: 行为记录列表
//   - error: 错误信息
func (s *Service) GetUserBehaviors(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error) {
	// 参数校验
	if userID == "" {
		return nil, ErrInvalidRequest
	}

	// 限制范围
	if limit <= 0 {
		limit = defaultBehaviorLimit
	}
	if limit > maxBehaviorLimit {
		limit = maxBehaviorLimit
	}

	// 尝试从缓存获取
	cacheKey := fmt.Sprintf("%s%s:%d", cacheKeyPrefixBehavior, userID, limit)
	var behaviors []*interfaces.UserBehavior
	if err := s.cache.Get(ctx, cacheKey, &behaviors); err == nil && len(behaviors) > 0 {
		s.logger.Debug("cache hit for user behaviors",
			zap.String("user_id", userID),
			zap.Int("count", len(behaviors)),
		)
		return behaviors, nil
	}

	// 从数据库获取
	behaviors, err := s.userRepo.GetBehaviors(ctx, userID, limit)
	if err != nil {
		s.logger.Error("failed to get user behaviors",
			zap.String("user_id", userID),
			zap.Error(err),
		)
		return nil, fmt.Errorf("failed to get behaviors: %w", err)
	}

	// 缓存结果
	if len(behaviors) > 0 {
		if cacheErr := s.cache.Set(ctx, cacheKey, behaviors, behaviorCacheTTL); cacheErr != nil {
			s.logger.Warn("failed to cache behaviors",
				zap.String("user_id", userID),
				zap.Error(cacheErr),
			)
		}
	}

	return behaviors, nil
}

// GetUserProfile 获取用户画像
//
// 实现 interfaces.UserService.GetUserProfile 方法
// 综合用户基本信息和行为统计，生成用户画像
//
// 参数:
//   - ctx: 上下文
//   - userID: 用户唯一标识
//
// 返回:
//   - *interfaces.UserProfile: 用户画像信息
//   - error: 错误信息
func (s *Service) GetUserProfile(ctx context.Context, userID string) (*interfaces.UserProfile, error) {
	// 参数校验
	if userID == "" {
		return nil, ErrInvalidRequest
	}

	// 尝试从缓存获取
	cacheKey := cacheKeyPrefixProfile + userID
	var profile interfaces.UserProfile
	if err := s.cache.Get(ctx, cacheKey, &profile); err == nil {
		s.logger.Debug("cache hit for user profile",
			zap.String("user_id", userID),
		)
		return &profile, nil
	}

	// 获取用户基本信息
	user, err := s.GetUser(ctx, userID)
	if err != nil {
		return nil, err
	}

	// 获取用户行为历史用于统计
	behaviors, err := s.userRepo.GetBehaviors(ctx, userID, maxBehaviorLimit)
	if err != nil {
		s.logger.Warn("failed to get behaviors for profile",
			zap.String("user_id", userID),
			zap.Error(err),
		)
		behaviors = []*interfaces.UserBehavior{}
	}

	// 构建用户画像
	profile = interfaces.UserProfile{
		User:           user,
		TotalActions:   len(behaviors),
		PreferredTypes: s.calculatePreferredTypes(behaviors),
		ActiveHours:    s.calculateActiveHours(behaviors),
		LastActive:     s.getLastActiveTime(behaviors),
	}

	// 缓存画像
	if cacheErr := s.cache.Set(ctx, cacheKey, &profile, profileCacheTTL); cacheErr != nil {
		s.logger.Warn("failed to cache user profile",
			zap.String("user_id", userID),
			zap.Error(cacheErr),
		)
	}

	return &profile, nil
}

// =============================================================================
// 辅助方法
// =============================================================================

// clearUserCaches 清除用户相关的所有缓存
func (s *Service) clearUserCaches(ctx context.Context, userID string) {
	keys := []string{
		cacheKeyPrefixUser + userID,
		cacheKeyPrefixProfile + userID,
		cacheKeyPrefixBehavior + userID,
	}

	for _, key := range keys {
		if err := s.cache.Delete(ctx, key); err != nil {
			s.logger.Warn("failed to delete cache",
				zap.String("key", key),
				zap.Error(err),
			)
		}
	}
}

// calculatePreferredTypes 计算用户偏好类型
//
// 根据用户行为统计各类型的交互次数
func (s *Service) calculatePreferredTypes(behaviors []*interfaces.UserBehavior) map[string]int {
	types := make(map[string]int)

	for _, b := range behaviors {
		// 统计各行为类型的次数
		types[b.Action]++
	}

	return types
}

// calculateActiveHours 计算用户活跃时段
//
// 统计用户在各小时段的活跃次数
func (s *Service) calculateActiveHours(behaviors []*interfaces.UserBehavior) map[int]int {
	hours := make(map[int]int)

	for _, b := range behaviors {
		hour := b.Timestamp.Hour()
		hours[hour]++
	}

	return hours
}

// getLastActiveTime 获取用户最后活跃时间
//
// 返回最近一次行为的时间戳
func (s *Service) getLastActiveTime(behaviors []*interfaces.UserBehavior) time.Time {
	if len(behaviors) == 0 {
		return time.Time{}
	}

	// 假设行为列表已按时间倒序排列
	return behaviors[0].Timestamp
}

// =============================================================================
// 工具函数
// =============================================================================

// generateUserID 生成唯一用户 ID
//
// 格式: u_时间戳_随机字符串
// 例如: u_20250104150405_a1b2c3d4
func generateUserID() (string, error) {
	// 生成 8 字节随机数
	randomBytes := make([]byte, 4)
	if _, err := rand.Read(randomBytes); err != nil {
		return "", err
	}

	// 格式化 ID
	timestamp := time.Now().Format("20060102150405")
	randomStr := hex.EncodeToString(randomBytes)

	return fmt.Sprintf("u_%s_%s", timestamp, randomStr), nil
}

// =============================================================================
// 缓存辅助类型
// =============================================================================

// CachedUser 用于缓存序列化的用户数据
type CachedUser struct {
	Data      []byte    `json:"data"`
	CachedAt  time.Time `json:"cached_at"`
	ExpiresAt time.Time `json:"expires_at"`
}

// MarshalUser 序列化用户数据用于缓存
func MarshalUser(user *interfaces.User) ([]byte, error) {
	return json.Marshal(user)
}

// UnmarshalUser 反序列化缓存的用户数据
func UnmarshalUser(data []byte) (*interfaces.User, error) {
	var user interfaces.User
	if err := json.Unmarshal(data, &user); err != nil {
		return nil, err
	}
	return &user, nil
}

