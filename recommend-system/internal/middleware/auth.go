// Package middleware 提供 HTTP 中间件
package middleware

import (
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// AuthConfig 认证配置
type AuthConfig struct {
	JWTSecret     string
	TokenHeader   string
	TokenPrefix   string
	ExcludePaths  []string
	EnableAPIKey  bool
	APIKeyHeader  string
	ValidAPIKeys  map[string]string // key -> user_id
}

// DefaultAuthConfig 默认认证配置
func DefaultAuthConfig() *AuthConfig {
	return &AuthConfig{
		TokenHeader:  "Authorization",
		TokenPrefix:  "Bearer ",
		ExcludePaths: []string{"/health", "/metrics", "/api/v1/auth/login"},
		EnableAPIKey: true,
		APIKeyHeader: "X-API-Key",
		ValidAPIKeys: make(map[string]string),
	}
}

// Claims JWT 声明
type Claims struct {
	UserID    string   `json:"user_id"`
	Username  string   `json:"username"`
	Roles     []string `json:"roles"`
	ExpiresAt int64    `json:"exp"`
	IssuedAt  int64    `json:"iat"`
}

// Validate 验证 Claims
func (c *Claims) Validate() error {
	if c.UserID == "" {
		return errors.New("user_id is required")
	}
	if c.ExpiresAt < time.Now().Unix() {
		return errors.New("token has expired")
	}
	return nil
}

// Auth 认证中间件
func Auth(cfg *AuthConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 检查是否是排除路径
		path := c.Request.URL.Path
		for _, excludePath := range cfg.ExcludePaths {
			if strings.HasPrefix(path, excludePath) {
				c.Next()
				return
			}
		}

		// 尝试 API Key 认证
		if cfg.EnableAPIKey {
			apiKey := c.GetHeader(cfg.APIKeyHeader)
			if apiKey != "" {
				if userID, ok := cfg.ValidAPIKeys[apiKey]; ok {
					c.Set("user_id", userID)
					c.Set("auth_type", "api_key")
					c.Next()
					return
				}
			}
		}

		// JWT 认证
		tokenString := c.GetHeader(cfg.TokenHeader)
		if tokenString == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"code":    401,
				"message": "missing authorization header",
			})
			return
		}

		// 移除 Bearer 前缀
		if strings.HasPrefix(tokenString, cfg.TokenPrefix) {
			tokenString = strings.TrimPrefix(tokenString, cfg.TokenPrefix)
		}

		// 解析和验证 JWT (简化版，实际需要使用 jwt-go)
		claims, err := parseAndValidateToken(tokenString, cfg.JWTSecret)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"code":    401,
				"message": "invalid or expired token",
				"error":   err.Error(),
			})
			return
		}

		// 设置用户信息到上下文
		c.Set("user_id", claims.UserID)
		c.Set("username", claims.Username)
		c.Set("roles", claims.Roles)
		c.Set("auth_type", "jwt")

		c.Next()
	}
}

// parseAndValidateToken 解析和验证 JWT (占位实现)
func parseAndValidateToken(tokenString, secret string) (*Claims, error) {
	// TODO: 使用 github.com/golang-jwt/jwt/v5 实现
	// 这里是占位实现，实际项目需要完整的 JWT 验证
	
	// 简单模拟
	if tokenString == "" {
		return nil, errors.New("empty token")
	}
	
	// 实际实现应该:
	// 1. 解析 JWT
	// 2. 验证签名
	// 3. 验证过期时间
	// 4. 返回 Claims
	
	return &Claims{
		UserID:    "user_001",
		Username:  "test_user",
		Roles:     []string{"user"},
		ExpiresAt: time.Now().Add(24 * time.Hour).Unix(),
		IssuedAt:  time.Now().Unix(),
	}, nil
}

// RequireRole 角色检查中间件
func RequireRole(roles ...string) gin.HandlerFunc {
	return func(c *gin.Context) {
		userRoles, exists := c.Get("roles")
		if !exists {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"code":    403,
				"message": "no roles found",
			})
			return
		}

		roleList, ok := userRoles.([]string)
		if !ok {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"code":    403,
				"message": "invalid roles format",
			})
			return
		}

		// 检查是否有所需角色
		hasRole := false
		for _, required := range roles {
			for _, userRole := range roleList {
				if required == userRole {
					hasRole = true
					break
				}
			}
			if hasRole {
				break
			}
		}

		if !hasRole {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"code":    403,
				"message": "insufficient permissions",
			})
			return
		}

		c.Next()
	}
}

// GetUserID 从上下文获取用户 ID
func GetUserID(c *gin.Context) string {
	if userID, exists := c.Get("user_id"); exists {
		if id, ok := userID.(string); ok {
			return id
		}
	}
	return ""
}

// GetUsername 从上下文获取用户名
func GetUsername(c *gin.Context) string {
	if username, exists := c.Get("username"); exists {
		if name, ok := username.(string); ok {
			return name
		}
	}
	return ""
}

// GetRoles 从上下文获取角色列表
func GetRoles(c *gin.Context) []string {
	if roles, exists := c.Get("roles"); exists {
		if roleList, ok := roles.([]string); ok {
			return roleList
		}
	}
	return nil
}

