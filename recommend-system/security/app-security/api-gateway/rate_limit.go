package security

import (
	"context"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

// RateLimiter interface allows switching between local and Redis implementations
type RateLimiter interface {
	Allow(key string) bool
}

// LocalRateLimiter uses in-memory map of rate.Limiters
type LocalRateLimiter struct {
	visitors map[string]*rate.Limiter
	mu       sync.Mutex
	rate     rate.Limit
	burst    int
}

func NewLocalRateLimiter(rps float64, burst int) *LocalRateLimiter {
	return &LocalRateLimiter{
		visitors: make(map[string]*rate.Limiter),
		rate:     rate.Limit(rps),
		burst:    burst,
	}
}

func (l *LocalRateLimiter) Allow(key string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()

	limiter, exists := l.visitors[key]
	if !exists {
		limiter = rate.NewLimiter(l.rate, l.burst)
		l.visitors[key] = limiter
	}

	return limiter.Allow()
}

// RateLimitMiddleware creates a Gin middleware
func RateLimitMiddleware(limiter RateLimiter) gin.HandlerFunc {
	return func(c *gin.Context) {
		ip := c.ClientIP()
		if !limiter.Allow(ip) {
			c.Header("X-RateLimit-Limit", "100")
			c.Header("X-RateLimit-Remaining", "0")
			c.Header("Retry-After", "60")
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{"error": "Too many requests"})
			return
		}
		c.Next()
	}
}

