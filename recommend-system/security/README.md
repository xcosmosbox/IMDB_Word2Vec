# 生成式推荐系统 - 安全架构层 (Security Layer)

## 项目概述
本项目实现了生成式推荐系统的全面安全架构，遵循“纵深防御 (Defense in Depth)”和“零信任 (Zero Trust)”原则。

## 模块导航

| 模块 | 目录 | 负责人 | 核心功能 |
|------|------|--------|----------|
| **IAM** | [iam/](./iam/README.md) | Person A | 身份认证 (JWT+RS256), 鉴权 (OPA/Rego), MFA |
| **Infrastructure** | [infrastructure/](./infrastructure/README.md) | Person B | K8s NetworkPolicies, Istio mTLS, 容器安全 |
| **Data Privacy** | [data-privacy/](./data-privacy/README.md) | Person C | 敏感数据加密 (AES-GCM), 信封加密, PII 脱敏 |
| **AppSec** | [app-security/](./app-security/README.md) | Person D | API 签名 (HMAC), 限流 (RateLimit), WAF |
| **AI Safety** | [ai-safety/](./ai-safety/README.md) | Person E | Prompt 注入防御, 内容审核, 对抗检测 |
| **Compliance** | [compliance/](./compliance/README.md) | Person F | 审计日志 (CloudEvents), GDPR 删除, 安全扫描 |

## 快速开始

### 前置依赖
- Go 1.21+
- Python 3.9+
- Kubernetes 1.24+ (可选, 用于部署基础设施配置)

### 运行测试
本项目包含完整的单元测试。

**Go 模块测试 (IAM, Data Privacy, AppSec, Compliance):**
```bash
# 在 recommend-system/security 目录下
go test ./iam/... ./data-privacy/... ./app-security/... ./compliance/...
```

**Python 模块测试 (AI Safety, GDPR Scripts):**
```bash
# 在 recommend-system/security 目录下
python -m unittest discover ai-safety/
python -m unittest discover compliance/gdpr/
```

## 安全标准
所有开发均遵循 [SECURITY_STANDARDS.md](./SECURITY_STANDARDS.md) 定义的契约：
- **加密**: AES-256-GCM / RSA-2048
- **认证**: OIDC / OAuth2
- **通信**: 全链路 TLS 1.3 / mTLS
- **隐私**: PII 默认脱敏

## 部署指南
各模块详细部署步骤请参考对应子目录下的 `README.md`。
- 基础设施层：使用 `kubectl apply -f infrastructure/...`
- 应用中间件：在业务代码中引入 `iam/middleware` 和 `app-security/api-gateway`。

