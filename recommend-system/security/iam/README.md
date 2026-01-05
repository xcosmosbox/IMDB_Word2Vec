# Identity and Access Management (IAM)

## 概述
IAM 模块是推荐系统的核心安全组件，负责身份认证、鉴权和会话管理。

## 目录结构
- `auth-service/`: 认证服务核心代码
    - `core/`: Token 签发与验证 (JWT + RS256)
    - `mfa/`: 多因素认证 (TOTP)
- `middleware/`: HTTP 认证中间件 (Gin)
- `policy-engine/`: OPA 策略文件 (Rego)

## 功能特性
1. **JWT 认证**: 使用 RS256 非对称加密签发 Token，包含 Access Token (15min) 和 Refresh Token (7天)。
2. **RBAC 鉴权**: 基于 Open Policy Agent (OPA) 的 Rego 策略引擎。
3. **MFA**: 支持 TOTP (Time-based One-Time Password) 验证。
4. **Middleware**: Gin 中间件，自动解析 Token 并注入 User Context。

## 使用指南

### 1. 初始化 Token Service
```go
privateKey, _ := rsa.GenerateKey(rand.Reader, 2048)
tokenService := core.NewRSATokenService(privateKey, &privateKey.PublicKey)
```

### 2. 签发 Token
```go
user := &core.User{ID: "1", Username: "alice", Roles: []string{"admin"}}
tokens, err := tokenService.GenerateTokens(user)
```

### 3. 集成中间件
```go
r := gin.Default()
r.Use(middleware.AuthMiddleware(tokenService))
```

## 测试
```bash
go test ./...
```

