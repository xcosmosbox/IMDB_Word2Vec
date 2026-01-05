# 安全与合规标准契约 (Security Standards & Contracts)

本文档定义了推荐系统的全局安全标准、协议和接口。所有模块的开发必须严格遵循此契约。

---

## 1. 身份与访问 (IAM)

### 1.1 认证协议
- **标准**: OpenID Connect (OIDC) / OAuth 2.0
- **Token 格式**: JWT (RS256 签名)
- **Token 有效期**: Access Token (15min), Refresh Token (7days)
- **MFA**: 强制对管理员和高风险操作启用 MFA (TOTP/WebAuthn)

### 1.2 授权模型
- **模型**: RBAC (Role-Based) + ABAC (Attribute-Based) 混合模式
- **策略语言**: Rego (Open Policy Agent)
- **最小权限原则**: 所有服务间调用默认为 Deny

## 2. 数据安全

### 2.1 加密标准
- **传输层 (Data in Transit)**: TLS 1.3 (强制), 禁用弱密码套件
- **存储层 (Data in Rest)**: AES-256-GCM
- **密钥管理**: 必须使用 KMS (Key Management Service)，禁止硬编码密钥

### 2.2 隐私保护
- **PII 定义**: 姓名, 邮箱, 手机号, IP, 设备ID, 生物特征
- **脱敏规则**:
  - 手机号: 保留后4位 (`*******1234`)
  - 邮箱: 掩盖用户名部分 (`a***@example.com`)
  - 身份证: 掩盖出生日期 (`110101********1234`)
- **数据分级**: 
  - L1 (Public): 公开数据
  - L2 (Internal): 内部业务数据
  - L3 (Confidential): 用户敏感数据 (PII)
  - L4 (Critical): 核心密钥, 支付信息

## 3. 应用安全

### 3.1 Web 安全
- **Headers**:
  - `Strict-Transport-Security: max-age=31536000; includeSubDomains`
  - `Content-Security-Policy: default-src 'self'`
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
- **防御**:
  - SQL 注入: 强制使用参数化查询/ORM
  - XSS: 强制输出编码, 禁用 `unsafe-inline`
  - CSRF: 强制 SameSite Cookie + CSRF Token

### 3.2 API 安全
- **限流**: 基于 Token Bucket 算法, 默认 100 RPS/IP
- **签名**: 关键 API 请求必须包含 HMAC 签名
- **幂等性**: 写入接口必须支持 `Idempotency-Key`

## 4. AI 安全

### 4.1 模型输入/输出
- **Prompt 过滤**: 检测并拦截 Prompt Injection, Jailbreak 攻击
- **内容审核**: 过滤仇恨言论, 色情, 暴力内容
- **隐私泄露**: 输出层检测是否包含训练数据中的 PII

### 4.2 对抗防御
- **扰动检测**: 检测输入嵌入的对抗性扰动
- **鲁棒性测试**: 定期进行对抗样本测试

## 5. 基础设施安全

### 5.1 网络隔离
- **零信任**: 服务间通信必须通过 mTLS (Istio)
- **网络策略**: K8s NetworkPolicy 默认 Deny All, 显式 Allow 白名单

### 5.2 容器安全
- **镜像**: 必须扫描漏洞 (Trivy), 禁止使用 `latest` 标签
- **运行时**: 禁止 `privileged` 容器, 强制 `runAsNonRoot`

## 6. 合规与审计

### 6.1 审计日志
- **记录范围**: 所有登录, 权限变更, 敏感数据访问, 关键配置修改
- **日志格式**: CloudEvents 规范 (JSON)
- **保留策略**: 在线 90 天, 归档 3 年

### 6.2 合规标准
- **GDPR**: 支持用户数据导出, 删除 (Right to be Forgotten)
- **SOC2**: 满足访问控制, 变更管理, 监控告警要求

