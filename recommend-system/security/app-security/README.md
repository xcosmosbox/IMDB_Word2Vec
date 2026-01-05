# Application Security (AppSec)

## 概述
AppSec 模块负责防御常见的 Web 攻击 (OWASP Top 10) 和 API 滥用，确保业务接口的安全。

## 目录结构
- `api-gateway/`: API 网关安全
    - `signature.go`: HMAC 请求签名验证
    - `rate_limit.go`: 令牌桶限流 (Token Bucket)
- `secure-headers/`: 安全响应头中间件
- `waf/`: WAF 规则配置
    - `rules/`: ModSecurity 规则文件

## 功能特性

### 1. API 签名 (HMAC)
- 验证请求来源的合法性和完整性。
- **算法**: HMAC-SHA256
- **校验内容**: Method + Path + Timestamp + Body
- **防重放**: 5分钟时间窗口校验

### 2. 限流 (Rate Limiting)
- 基于令牌桶算法，支持按 IP 或 UserID 限流。
- 默认策略：100 RPS / IP。

### 3. 安全响应头
- `Strict-Transport-Security`: 强制 HTTPS
- `Content-Security-Policy`: 防御 XSS
- `X-Frame-Options`: 防御 Clickjacking

### 4. WAF 集成
- 提供 ModSecurity 兼容的规则集，拦截 SQL 注入和 XSS 攻击。

## 使用指南

### 校验签名
```go
err := security.VerifyHMACSignature(method, path, ts, body, sig, secret)
```

### 启用限流与安全头
```go
r := gin.Default()
r.Use(headers.SecurityHeadersMiddleware())
r.Use(security.RateLimitMiddleware(NewLocalRateLimiter(100, 10)))
```

## 测试
```bash
go test ./...
```

