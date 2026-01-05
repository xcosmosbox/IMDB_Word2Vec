package core

import (
	"crypto/rsa"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

// User defines the user structure for token generation
type User struct {
	ID       string
	Username string
	Roles    []string
}

// TokenPair contains access and refresh tokens
type TokenPair struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int64  `json:"expires_in"`
}

// TokenClaims defines JWT payload
type TokenClaims struct {
	UserID    string   `json:"sub"`
	Username  string   `json:"name"`
	Roles     []string `json:"roles"`
	Scope     []string `json:"scope"`
	SessionID string   `json:"sid"`
	jwt.RegisteredClaims
}

// TokenService interface
type TokenService interface {
	GenerateTokens(user *User) (*TokenPair, error)
	ValidateToken(tokenString string) (*TokenClaims, error)
}

type RSATokenService struct {
	privateKey *rsa.PrivateKey
	publicKey  *rsa.PublicKey
}

func NewRSATokenService(privateKey *rsa.PrivateKey, publicKey *rsa.PublicKey) *RSATokenService {
	return &RSATokenService{
		privateKey: privateKey,
		publicKey:  publicKey,
	}
}

func (s *RSATokenService) GenerateTokens(user *User) (*TokenPair, error) {
	now := time.Now()

	// Access Token (15 min)
	accessClaims := TokenClaims{
		UserID:   user.ID,
		Username: user.Username,
		Roles:    user.Roles,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(now.Add(15 * time.Minute)),
			IssuedAt:  jwt.NewNumericDate(now),
			NotBefore: jwt.NewNumericDate(now),
			Issuer:    "recommend-system-iam",
			ID:        uuid.New().String(),
		},
	}

	accessToken, err := jwt.NewWithClaims(jwt.SigningMethodRS256, accessClaims).SignedString(s.privateKey)
	if err != nil {
		return nil, err
	}

	// Refresh Token (7 days)
	refreshClaims := jwt.RegisteredClaims{
		Subject:   user.ID,
		ExpiresAt: jwt.NewNumericDate(now.Add(7 * 24 * time.Hour)),
		ID:        uuid.New().String(),
	}

	refreshToken, err := jwt.NewWithClaims(jwt.SigningMethodRS256, refreshClaims).SignedString(s.privateKey)
	if err != nil {
		return nil, err
	}

	return &TokenPair{
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		ExpiresIn:    900,
	}, nil
}

func (s *RSATokenService) ValidateToken(tokenString string) (*TokenClaims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &TokenClaims{}, func(token *jwt.Token) (interface{}, error) {
		return s.publicKey, nil
	})

	if err != nil {
		return nil, err
	}

	if claims, ok := token.Claims.(*TokenClaims); ok && token.Valid {
		return claims, nil
	}

	return nil, jwt.ErrTokenInvalidClaims
}

