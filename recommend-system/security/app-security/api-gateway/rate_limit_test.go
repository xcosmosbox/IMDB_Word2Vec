package security

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestRateLimitMiddleware(t *testing.T) {
	// Allow 1 request per second, burst 1
	limiter := NewLocalRateLimiter(1, 1)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(RateLimitMiddleware(limiter))
	r.GET("/", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	// First request should pass
	req1, _ := http.NewRequest("GET", "/", nil)
	w1 := httptest.NewRecorder()
	r.ServeHTTP(w1, req1)
	if w1.Code != http.StatusOK {
		t.Errorf("First request: expected 200, got %d", w1.Code)
	}

	// Second request immediately after should fail (burst consumed)
	req2, _ := http.NewRequest("GET", "/", nil)
	w2 := httptest.NewRecorder()
	r.ServeHTTP(w2, req2)
	if w2.Code != http.StatusTooManyRequests {
		t.Errorf("Second request: expected 429, got %d", w2.Code)
	}
}

