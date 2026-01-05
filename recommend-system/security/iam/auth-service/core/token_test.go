package core

import (
	"crypto/rand"
	"crypto/rsa"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

func TestRSATokenService(t *testing.T) {
	// Generate test keys
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("Failed to generate private key: %v", err)
	}
	publicKey := &privateKey.PublicKey

	service := NewRSATokenService(privateKey, publicKey)

	user := &User{
		ID:       "user-123",
		Username: "testuser",
		Roles:    []string{"admin"},
	}

	t.Run("GenerateTokens", func(t *testing.T) {
		tokens, err := service.GenerateTokens(user)
		if err != nil {
			t.Fatalf("GenerateTokens failed: %v", err)
		}

		if tokens.AccessToken == "" {
			t.Error("Access token is empty")
		}
		if tokens.RefreshToken == "" {
			t.Error("Refresh token is empty")
		}
		if tokens.ExpiresIn != 900 {
			t.Errorf("Expected expires in 900, got %d", tokens.ExpiresIn)
		}
	})

	t.Run("ValidateToken", func(t *testing.T) {
		tokens, _ := service.GenerateTokens(user)
		claims, err := service.ValidateToken(tokens.AccessToken)
		if err != nil {
			t.Fatalf("ValidateToken failed: %v", err)
		}

		if claims.UserID != user.ID {
			t.Errorf("Expected UserID %s, got %s", user.ID, claims.UserID)
		}
		if claims.Username != user.Username {
			t.Errorf("Expected Username %s, got %s", user.Username, claims.Username)
		}
	})
	
	t.Run("ExpiredToken", func(t *testing.T) {
		// Create an expired token manually
		now := time.Now().Add(-1 * time.Hour)
		claims := TokenClaims{
			UserID: user.ID,
			RegisteredClaims: jwt.RegisteredClaims{
				ExpiresAt: jwt.NewNumericDate(now),
			},
		}
		token, _ := jwt.NewWithClaims(jwt.SigningMethodRS256, claims).SignedString(privateKey)
		
		_, err := service.ValidateToken(token)
		if err == nil {
			t.Error("Expected error for expired token, got nil")
		}
	})
}

