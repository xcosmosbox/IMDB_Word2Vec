package middleware

import (
	"crypto/rand"
	"crypto/rsa"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"recommend-system/security/iam/auth-service/core"
)

func TestAuthMiddleware(t *testing.T) {
	// Setup Token Service
	privateKey, _ := rsa.GenerateKey(rand.Reader, 2048)
	tokenService := core.NewRSATokenService(privateKey, &privateKey.PublicKey)

	// Setup Gin
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(AuthMiddleware(tokenService))
	r.GET("/test", func(c *gin.Context) {
		claims, _ := c.Get("claims")
		c.JSON(http.StatusOK, claims)
	})

	t.Run("NoHeader", func(t *testing.T) {
		req, _ := http.NewRequest("GET", "/test", nil)
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)

		if w.Code != http.StatusUnauthorized {
			t.Errorf("Expected 401, got %d", w.Code)
		}
	})

	t.Run("InvalidHeader", func(t *testing.T) {
		req, _ := http.NewRequest("GET", "/test", nil)
		req.Header.Set("Authorization", "Basic xyz")
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)

		if w.Code != http.StatusUnauthorized {
			t.Errorf("Expected 401, got %d", w.Code)
		}
	})

	t.Run("ValidToken", func(t *testing.T) {
		user := &core.User{ID: "1", Username: "user", Roles: []string{"user"}}
		tokens, _ := tokenService.GenerateTokens(user)

		req, _ := http.NewRequest("GET", "/test", nil)
		req.Header.Set("Authorization", "Bearer "+tokens.AccessToken)
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected 200, got %d", w.Code)
		}
	})
}

