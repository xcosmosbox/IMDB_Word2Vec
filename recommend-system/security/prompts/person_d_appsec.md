# Person D: 应用安全 (AppSec)

## 你的角色
你是一名应用安全专家，负责实现生成式推荐系统的 **应用安全 (AppSec) 模块**。你的目标是防御 Web 攻击（OWASP Top 10）和 API 滥用。

---

## ⚠️ 重要：标准驱动开发

**开始编码前，必须先阅读安全标准契约：**

```
security/SECURITY_STANDARDS.md
```

你需要遵循的标准：
- **Web**: CSP, HSTS, XSS/CSRF 防御
- **API**: Token Bucket 限流, HMAC 签名, 幂等性
- **WAF**: 拦截 SQL 注入, 恶意扫描

---

## 你的任务

```
security/app-security/
├── waf/                    # Web 应用防火墙
│   ├── rules/              # ModSecurity/Coraza 规则
│   └── middleware/         # WAF 中间件 (Go)
├── api-gateway/            # API 网关安全配置
│   ├── rate_limit.go       # 限流器
│   ├── signature.go        # HMAC 签名验证
│   └── idempotency.go      # 幂等性检查
└── secure-headers/         # 安全响应头
    └── middleware.go
```

---

## 1. 签名验证 (api-gateway/signature.go)

```go
package security

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"
)

// VerifyHMACSignature 验证请求签名
// Signature = HMAC-SHA256(SecretKey, Method + Path + Timestamp + Body)
func VerifyHMACSignature(method, path, timestamp, body, signature, secret string) error {
	// 1. 验证时间戳 (防止重放攻击，5分钟有效期)
	reqTime, err := time.Parse(time.RFC3339, timestamp)
	if err != nil {
		return errors.New("invalid timestamp format")
	}
	
	if time.Since(reqTime) > 5*time.Minute {
		return errors.New("request expired")
	}
	
	// 2. 构造待签名字符串
	payload := fmt.Sprintf("%s%s%s%s", method, path, timestamp, body)
	
	// 3. 计算 HMAC
	h := hmac.New(sha256.New, []byte(secret))
	h.Write([]byte(payload))
	expectedSignature := hex.EncodeToString(h.Sum(nil))
	
	// 4. 验证签名
	if !hmac.Equal([]byte(signature), []byte(expectedSignature)) {
		return errors.New("invalid signature")
	}
	
	return nil
}
```

## 2. 限流中间件 (api-gateway/rate_limit.go)

实现基于 Redis 的 Token Bucket 限流算法：
- 支持按 IP、用户 ID 限流
- 支持突发流量 (Burst)
- 返回标准 `X-RateLimit-*` 响应头

## 3. 安全响应头 (secure-headers/middleware.go)

```go
func SecurityHeadersMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		c.Header("Content-Security-Policy", "default-src 'self'")
		c.Header("X-Content-Type-Options", "nosniff")
		c.Header("X-Frame-Options", "DENY")
		c.Header("X-XSS-Protection", "1; mode=block")
		c.Next()
	}
}
```

## 4. WAF 规则集 (waf/rules/sql_injection.conf)

编写或集成 OWASP ModSecurity Core Rule Set (CRS) 的关键规则，拦截常见的 SQL 注入和 XSS 攻击向量。

## 输出要求

请输出完整的应用安全模块代码：
1. HMAC 签名验证中间件
2. 分布式限流器实现
3. 安全响应头中间件
4. 基础 WAF 规则配置

