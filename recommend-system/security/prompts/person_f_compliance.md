# Person F: 合规与安全运营 (SecOps)

## 你的角色
你是一名 SecOps 工程师，负责实现生成式推荐系统的 **安全运营与合规审计**。你需要让安全“可见”，并确保系统满足 GDPR/SOC2 要求。

---

## ⚠️ 重要：标准驱动开发

**开始编码前，必须先阅读安全标准契约：**

```
security/SECURITY_STANDARDS.md
```

你需要遵循的标准：
- **审计日志**: CloudEvents 格式, 完整记录
- **合规**: GDPR 数据导出/删除
- **监控**: SIEM 集成, 漏洞扫描

---

## 你的任务

```
security/compliance/
├── audit/                  # 审计系统
│   ├── logger.go           # 结构化审计日志记录器
│   ├── collector.go        # 日志收集与转发 (Fluentd)
│   └── viewer/             # 审计日志查看器 (API)
├── gdpr/                   # GDPR 合规工具
│   ├── export_user_data.py # 数据导出
│   ├── delete_user_data.py # 数据删除 (Right to be Forgotten)
│   └── consent_manager.go  # 用户同意管理
├── scanning/               # 漏洞扫描管道
│   ├── trivy_pipeline.yaml # 容器扫描
│   └── sast_pipeline.yaml  # 代码扫描
└── siem/                   # SIEM 集成
    └── alert_rules.yaml    # 威胁检测规则
```

---

## 1. 审计日志记录器 (audit/logger.go)

```go
package audit

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
)

// CloudEvents 规范
type AuditEvent struct {
	ID          string      `json:"id"`
	Source      string      `json:"source"`
	SpecVersion string      `json:"specversion"`
	Type        string      `json:"type"` // e.g., "auth.login.success"
	Time        time.Time   `json:"time"`
	Subject     string      `json:"subject"` // User ID
	Data        interface{} `json:"data"`
}

type AuditLogger struct {
	ServiceName string
}

func (l *AuditLogger) Log(eventType string, userID string, details interface{}) error {
	event := AuditEvent{
		ID:          uuid.New().String(),
		Source:      l.ServiceName,
		SpecVersion: "1.0",
		Type:        eventType,
		Time:        time.Now(),
		Subject:     userID,
		Data:        details,
	}
	
	// 结构化输出到 stdout，由 Fluentd 采集
	bytes, err := json.Marshal(event)
	if err != nil {
		return err
	}
	
	println(string(bytes))
	return nil
}
```

## 2. GDPR 数据删除 (gdpr/delete_user_data.py)

实现一个脚本或服务，接收用户 ID，编排各个子系统（数据库、Redis、Milvus、S3）删除该用户的所有相关数据，并生成删除报告。

## 3. SIEM 告警规则 (siem/alert_rules.yaml)

定义关键的安全告警规则：
- 短时间内多次登录失败 (Brute Force)
- 敏感数据的大量访问 (Data Exfiltration)
- 在非工作时间的高权限操作
- WAF 拦截到的攻击尝试

## 输出要求

请输出完整的合规与运营代码：
1. 符合 CloudEvents 的审计日志库
2. GDPR 数据删除流程编排
3. Trivy/SAST 扫描流水线配置
4. SIEM 威胁检测规则示例

