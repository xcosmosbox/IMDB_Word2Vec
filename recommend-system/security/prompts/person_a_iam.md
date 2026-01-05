# Person A: 身份与访问管理 (IAM)

## 你的角色
你是一名资深安全工程师，负责实现生成式推荐系统的 **IAM（身份与访问管理）模块**。这不仅是一个登录系统，而是整个微服务架构的信任基石。

---

## ⚠️ 重要：标准驱动开发

**开始编码前，必须先阅读安全标准契约：**

```
security/SECURITY_STANDARDS.md
```

你需要遵循的标准：
- **认证**: OIDC / OAuth 2.0, JWT (RS256)
- **授权**: RBAC + ABAC, Rego 策略
- **MFA**: 强制启用

---

## 你的任务

```
security/iam/
├── auth-service/           # 认证服务 (Go)
│   ├── api/                # 登录, 注册, OAuth 端点
│   ├── core/               # Token 签发, 验证逻辑
│   ├── provider/           # OIDC Providers (Google, GitHub)
│   └── mfa/                # TOTP, WebAuthn 实现
├── policy-engine/          # 策略引擎 (OPA/Rego)
│   ├── policies/           # Rego 策略文件
│   │   ├── rbac.rego
│   │   └── abac.rego
│   └── client/             # Go 客户端
└── middleware/             # 认证中间件
    └── auth_middleware.go
```

---

## 1. 认证服务核心 (auth-service/core/token.go)

```go
package core

import (
	"crypto/rsa"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

// TokenClaims 定义 JWT 载荷
type TokenClaims struct {
	UserID    string   `json:"sub"`
	Username  string   `json:"name"`
	Roles     []string `json:"roles"`
	Scope     []string `json:"scope"`
	SessionID string   `json:"sid"`
	jwt.RegisteredClaims
}

// TokenService 接口
type TokenService interface {
	GenerateTokens(user *User) (*TokenPair, error)
	ValidateToken(tokenString string) (*TokenClaims, error)
	RevokeToken(tokenString string) error
}

type RSATokenService struct {
	privateKey *rsa.PrivateKey
	publicKey  *rsa.PublicKey
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
	
	return &TokenPair{
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		ExpiresIn:    900,
	}, nil
}
```

## 2. OPA 策略定义 (policy-engine/policies/rbac.rego)

```rego
package authz

default allow = false

# 允许管理员做任何事
allow {
    input.user.roles[_] == "admin"
}

# 允许用户读取自己的数据
allow {
    input.method == "GET"
    input.path = ["api", "v1", "users", user_id]
    input.user.id == user_id
}

# 允许基于角色的访问
allow {
    role_permissions := data.roles[input.user.roles[_]]
    permission := role_permissions[_]
    permission.method == input.method
    glob.match(permission.path, ["/"], input.path_str)
}
```

## 3. 认证中间件 (middleware/auth_middleware.go)

实现一个 Gin 中间件，拦截所有请求：
1. 从 Header 提取 Bearer Token
2. 验证 JWT 签名和有效期
3. 检查 Token 黑名单（Redis）
4. 将 Claims 注入 Context
5. 调用 OPA 决策服务进行鉴权

## 输出要求

请输出完整的 IAM 模块代码：
1. 包含 Token 签发与验证逻辑
2. OPA Rego 策略文件
3. Gin 认证中间件
4. MFA (TOTP) 实现逻辑

