package middleware

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

// RateLimitConfig 限流配置
type RateLimitConfig struct {
	// 全局限流
	GlobalRate  float64 // 每秒请求数
	GlobalBurst int     // 突发容量

	// 用户级限流
	UserRate  float64
	UserBurst int

	// IP 级限流
	IPRate  float64
	IPBurst int

	// 清理间隔
	CleanupInterval time.Duration
}

// DefaultRateLimitConfig 默认限流配置
func DefaultRateLimitConfig() *RateLimitConfig {
	return &RateLimitConfig{
		GlobalRate:      10000,        // 全局 10000 QPS
		GlobalBurst:     20000,
		UserRate:        100,          // 每用户 100 QPS
		UserBurst:       200,
		IPRate:          50,           // 每 IP 50 QPS
		IPBurst:         100,
		CleanupInterval: 5 * time.Minute,
	}
}

// RateLimiter 限流器
type RateLimiter struct {
	cfg           *RateLimitConfig
	globalLimiter *rate.Limiter
	userLimiters  sync.Map // user_id -> *rate.Limiter
	ipLimiters    sync.Map // ip -> *rate.Limiter
}

// NewRateLimiter 创建限流器
func NewRateLimiter(cfg *RateLimitConfig) *RateLimiter {
	rl := &RateLimiter{
		cfg:           cfg,
		globalLimiter: rate.NewLimiter(rate.Limit(cfg.GlobalRate), cfg.GlobalBurst),
	}

	// 启动清理 goroutine
	go rl.cleanup()

	return rl
}

// cleanup 定期清理过期的限流器
func (rl *RateLimiter) cleanup() {
	ticker := time.NewTicker(rl.cfg.CleanupInterval)
	defer ticker.Stop()

	for range ticker.C {
		// 简单策略：清理所有用户和 IP 限流器
		// 实际项目可以使用 LRU 或 TTL
		rl.userLimiters = sync.Map{}
		rl.ipLimiters = sync.Map{}
	}
}

// getUserLimiter 获取用户限流器
func (rl *RateLimiter) getUserLimiter(userID string) *rate.Limiter {
	if limiter, ok := rl.userLimiters.Load(userID); ok {
		return limiter.(*rate.Limiter)
	}

	newLimiter := rate.NewLimiter(rate.Limit(rl.cfg.UserRate), rl.cfg.UserBurst)
	actual, _ := rl.userLimiters.LoadOrStore(userID, newLimiter)
	return actual.(*rate.Limiter)
}

// getIPLimiter 获取 IP 限流器
func (rl *RateLimiter) getIPLimiter(ip string) *rate.Limiter {
	if limiter, ok := rl.ipLimiters.Load(ip); ok {
		return limiter.(*rate.Limiter)
	}

	newLimiter := rate.NewLimiter(rate.Limit(rl.cfg.IPRate), rl.cfg.IPBurst)
	actual, _ := rl.ipLimiters.LoadOrStore(ip, newLimiter)
	return actual.(*rate.Limiter)
}

// Allow 检查是否允许请求
func (rl *RateLimiter) Allow(ctx context.Context, userID, ip string) error {
	// 全局限流
	if !rl.globalLimiter.Allow() {
		return fmt.Errorf("global rate limit exceeded")
	}

	// 用户限流
	if userID != "" {
		userLimiter := rl.getUserLimiter(userID)
		if !userLimiter.Allow() {
			return fmt.Errorf("user rate limit exceeded")
		}
	}

	// IP 限流
	if ip != "" {
		ipLimiter := rl.getIPLimiter(ip)
		if !ipLimiter.Allow() {
			return fmt.Errorf("ip rate limit exceeded")
		}
	}

	return nil
}

// RateLimit 限流中间件
func RateLimit(rl *RateLimiter) gin.HandlerFunc {
	return func(c *gin.Context) {
		userID := GetUserID(c)
		ip := c.ClientIP()

		if err := rl.Allow(c.Request.Context(), userID, ip); err != nil {
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"code":    429,
				"message": "rate limit exceeded",
				"error":   err.Error(),
			})
			return
		}

		c.Next()
	}
}

// AdaptiveRateLimiter 自适应限流器
type AdaptiveRateLimiter struct {
	baseLimiter   *rate.Limiter
	currentRate   float64
	minRate       float64
	maxRate       float64
	adjustFactor  float64
	mu            sync.RWMutex
	successCount  int64
	failureCount  int64
	windowSize    time.Duration
	lastAdjust    time.Time
}

// NewAdaptiveRateLimiter 创建自适应限流器
func NewAdaptiveRateLimiter(minRate, maxRate, initialRate float64, burst int) *AdaptiveRateLimiter {
	arl := &AdaptiveRateLimiter{
		baseLimiter:  rate.NewLimiter(rate.Limit(initialRate), burst),
		currentRate:  initialRate,
		minRate:      minRate,
		maxRate:      maxRate,
		adjustFactor: 0.1,
		windowSize:   time.Minute,
		lastAdjust:   time.Now(),
	}

	go arl.adjustLoop()

	return arl
}

// adjustLoop 自适应调整循环
func (arl *AdaptiveRateLimiter) adjustLoop() {
	ticker := time.NewTicker(arl.windowSize)
	defer ticker.Stop()

	for range ticker.C {
		arl.adjust()
	}
}

// adjust 调整限流速率
func (arl *AdaptiveRateLimiter) adjust() {
	arl.mu.Lock()
	defer arl.mu.Unlock()

	total := arl.successCount + arl.failureCount
	if total == 0 {
		return
	}

	// 计算成功率
	successRate := float64(arl.successCount) / float64(total)

	// 根据成功率调整
	if successRate > 0.95 {
		// 成功率高，增加速率
		arl.currentRate = min(arl.currentRate*(1+arl.adjustFactor), arl.maxRate)
	} else if successRate < 0.8 {
		// 成功率低，降低速率
		arl.currentRate = max(arl.currentRate*(1-arl.adjustFactor), arl.minRate)
	}

	// 更新限流器
	arl.baseLimiter.SetLimit(rate.Limit(arl.currentRate))

	// 重置计数器
	arl.successCount = 0
	arl.failureCount = 0
	arl.lastAdjust = time.Now()
}

// Allow 检查是否允许
func (arl *AdaptiveRateLimiter) Allow() bool {
	return arl.baseLimiter.Allow()
}

// RecordSuccess 记录成功
func (arl *AdaptiveRateLimiter) RecordSuccess() {
	arl.mu.Lock()
	arl.successCount++
	arl.mu.Unlock()
}

// RecordFailure 记录失败
func (arl *AdaptiveRateLimiter) RecordFailure() {
	arl.mu.Lock()
	arl.failureCount++
	arl.mu.Unlock()
}

// CurrentRate 获取当前速率
func (arl *AdaptiveRateLimiter) CurrentRate() float64 {
	arl.mu.RLock()
	defer arl.mu.RUnlock()
	return arl.currentRate
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

