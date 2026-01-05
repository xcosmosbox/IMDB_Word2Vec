package middleware

import (
	"context"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"recommend-system/security/iam/auth-service/core"
)

// AuthMiddleware creates a Gin middleware for authentication
func AuthMiddleware(tokenService core.TokenService) gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "Authorization header required"})
			return
		}

		parts := strings.Split(authHeader, " ")
		if len(parts) != 2 || parts[0] != "Bearer" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "Invalid authorization header format"})
			return
		}

		tokenString := parts[1]
		claims, err := tokenService.ValidateToken(tokenString)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "Invalid or expired token"})
			return
		}

		// Inject claims into context
		c.Set("claims", claims)
		c.Set("userID", claims.UserID)
		c.Set("roles", claims.Roles)

		// TODO: Call OPA for authorization here if needed, or in a separate Authorization middleware

		c.Next()
	}
}

