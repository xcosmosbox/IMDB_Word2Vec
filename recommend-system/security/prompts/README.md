# 安全与合规层开发任务分配

## 概述

本目录包含安全与合规层开发的 6 个独立任务，每个任务由一位安全工程师独立完成。

---

## ⚠️ 重要：标准驱动开发

所有开发者必须先阅读安全标准契约：

```
security/SECURITY_STANDARDS.md
```

该文档定义了：
- 认证与授权协议
- 加密与密钥管理标准
- 网络与容器安全基线
- 审计与合规要求

---

## 任务分配

| 角色 | 负责模块 | 提示词文件 | 技术重点 |
|------|----------|------------|----------|
| **Person A** | 身份与访问管理 (IAM) | `person_a_iam.md` | OAuth2, OPA, MFA |
| **Person B** | 基础设施安全 | `person_b_infrastructure.md` | K8s NetworkPolicy, Istio mTLS |
| **Person C** | 数据安全与隐私 | `person_c_data_privacy.md` | KMS, PII Masking, DLP |
| **Person D** | 应用安全 (AppSec) | `person_d_appsec.md` | WAF, Rate Limiting, Signing |
| **Person E** | AI 安全 | `person_e_ai_safety.md` | Prompt Injection, Content Safety |
| **Person F** | 合规与安全运营 | `person_f_compliance.md` | Audit Logs, GDPR, SIEM |

---

## 项目结构

```
security/
├── SECURITY_STANDARDS.md       # 核心契约
├── iam/                        # 身份管理 (Person A)
│   ├── auth-service/
│   └── policy-engine/
├── infrastructure/             # 基础设施安全 (Person B)
│   ├── network-policy/
│   └── istio-security/
├── data-privacy/               # 数据隐私 (Person C)
│   ├── crypto/
│   └── masking/
├── app-security/               # 应用安全 (Person D)
│   ├── waf/
│   └── api-gateway/
├── ai-safety/                  # AI 安全 (Person E)
│   ├── prompt-guard/
│   └── content-moderation/
├── compliance/                 # 合规与审计 (Person F)
│   ├── audit/
│   └── gdpr/
└── prompts/                    # 提示词文件
```

---

## 依赖关系

```
Person A (IAM) ───┬──► Person D (AppSec) ───► Person E (AI Safety)
                  │
                  ├──► Person C (Data Privacy)
                  │
                  └──► Person F (Compliance)

Person B (Infra) ────► (支撑所有层)
```

**建议开发顺序：**
1. **基础层**: Person B (Infra) + Person A (IAM)
2. **应用层**: Person D (AppSec) + Person C (Data Privacy)
3. **业务层**: Person E (AI Safety)
4. **运营层**: Person F (Compliance)

---

## 安全原则

1. **纵深防御 (Defense in Depth)**: 不依赖单一防线
2. **最小权限 (Least Privilege)**: 默认拒绝，显式允许
3. **安全左移 (Shift Left)**: 开发阶段集成安全检查
4. **自动化**: 安全策略即代码 (Policy as Code)

