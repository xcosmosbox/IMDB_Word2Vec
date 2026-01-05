# Data Privacy

## 概述
数据隐私模块负责敏感数据 (PII) 的加密存储和脱敏展示，保障用户隐私安全。

## 目录结构
- `crypto/`: 加密库
    - `aes_gcm.go`: 本地 AES-256-GCM 加密实现
    - `envelope.go`: 信封加密逻辑，支持 KMS 集成
- `masking/`: 脱敏库
    - `rules.go`: 常用字段脱敏规则 (手机, 邮箱, 身份证)
    - `scanner.go`: 文本扫描与自动脱敏工具

## 功能特性

### 1. 强加密
- 使用 **AES-256-GCM** 进行数据加密。
- 采用 **信封加密** (Envelope Encryption) 机制：
    - 数据由数据密钥 (DEK) 加密。
    - DEK 由主密钥 (CMK, stored in KMS) 加密。
    - 数据库仅存储密文数据和密文 DEK。

### 2. 智能脱敏
- 支持对特定字段的静态脱敏。
- 支持对非结构化文本的正则扫描与自动脱敏。

## 使用指南

### 加密数据
```go
service, _ := crypto.NewCryptoService(keyBase64)
encrypted, _ := service.Encrypt([]byte("secret"))
```

### 信封加密 (推荐)
```go
kms := NewKMSClient(...)
envelope := crypto.NewEnvelopeCrypto(kms)
encData, encKey, _ := envelope.EncryptData([]byte("sensitive"))
```

### 数据脱敏
```go
maskedPhone := masking.MaskPhoneNumber("13812345678")
// Output: 138****5678
```

## 测试
```bash
go test ./...
```

