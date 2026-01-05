# Person C: 数据安全与隐私

## 你的角色
你是一名数据安全专家，负责实现生成式推荐系统的 **数据安全与隐私保护模块**。你需要确保敏感数据（PII）在存储、传输和使用过程中的安全。

---

## ⚠️ 重要：标准驱动开发

**开始编码前，必须先阅读安全标准契约：**

```
security/SECURITY_STANDARDS.md
```

你需要遵循的标准：
- **加密**: AES-256-GCM, TLS 1.3
- **KMS**: 密钥生命周期管理
- **PII**: 自动脱敏, 数据分级

---

## 你的任务

```
security/data-privacy/
├── crypto/                 # 加密库
│   ├── aes_gcm.go          # 对称加密实现
│   ├── envelope.go         # 信封加密 (KMS集成)
│   └── key_rotation.go     # 密钥轮换逻辑
├── masking/                # 脱敏库
│   ├── rules.go            # 脱敏规则 (手机号, 邮箱)
│   ├── scanner.go          # PII 扫描器
│   └── dynamic.go          # 动态脱敏 (读取时)
└── dlp/                    # 数据防泄露
    ├── inspector.go        # 敏感数据检测
    └── filter.go           # 流量过滤器
```

---

## 1. 对称加密 (crypto/aes_gcm.go)

```go
package crypto

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"io"
)

type CryptoService struct {
	key []byte
}

func NewCryptoService(keyBase64 string) (*CryptoService, error) {
	key, err := base64.StdEncoding.DecodeString(keyBase64)
	if err != nil {
		return nil, err
	}
	if len(key) != 32 {
		return nil, errors.New("key length must be 32 bytes")
	}
	return &CryptoService{key: key}, nil
}

func (s *CryptoService) Encrypt(plaintext []byte) (string, error) {
	block, err := aes.NewCipher(s.key)
	if err != nil {
		return "", err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", err
	}

	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return "", err
	}

	ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)
	return base64.StdEncoding.EncodeToString(ciphertext), nil
}
```

## 2. PII 脱敏 (masking/rules.go)

实现以下脱敏规则：
- **手机号**: 保留后4位 (`138****1234`)
- **邮箱**: 掩盖用户名部分 (`a***@example.com`)
- **身份证**: 掩盖生日 (`110101********1234`)
- **姓名**: 保留姓氏 (`张**`)

## 3. 信封加密 (crypto/envelope.go)

实现与 KMS (如 AWS KMS 或 HashiCorp Vault) 的集成：
1. 生成数据密钥 (Data Key)
2. 使用 KMS 主密钥 (CMK) 加密数据密钥
3. 使用明文数据密钥加密实际数据
4. 存储密文数据密钥和密文数据

## 输出要求

请输出完整的数据安全模块代码：
1. AES-256-GCM 加密封装
2. PII 脱敏工具库
3. 模拟 KMS 的信封加密实现
4. 敏感数据扫描器

