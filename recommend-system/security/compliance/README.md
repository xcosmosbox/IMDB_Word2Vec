# Compliance & SecOps

## 概述
合规与安全运营模块确保系统的可审计性，满足 GDPR 要求，并建立自动化的安全扫描与威胁检测机制。

## 目录结构
- `audit/`: 审计日志
    - `logger.go`: 符合 CloudEvents 规范的结构化日志记录器。
- `gdpr/`: GDPR 合规工具
    - `delete_user_data.py`: “被遗忘权”自动化删除脚本。
- `scanning/`: 漏洞扫描
    - `trivy_pipeline.yaml`: CI/CD 容器镜像扫描配置。
- `siem/`: 安全监控
    - `alert_rules.yaml`: Prometheus/Alertmanager 告警规则。

## 功能特性

### 1. 全链路审计
- 记录所有关键操作（登录、鉴权、数据访问）。
- 日志格式标准化 (JSON + CloudEvents)，便于 SIEM 采集分析。

### 2. GDPR 合规
- **数据删除**: 提供一键删除用户所有数据的工具（数据库、缓存、对象存储、向量库）。
- **数据导出**: (可扩展) 支持导出用户个人数据。

### 3. DevSecOps
- **镜像扫描**: 集成 Trivy，阻断包含 High/Critical 漏洞的镜像上线。
- **静态分析**: (SAST) 代码提交时自动扫描安全漏洞。

### 4. 威胁检测
- 实时监控异常行为：
    - 暴力破解 (Brute Force)
    - WAF 拦截峰值
    - 非工作时间的高权限操作

## 使用指南

### 记录审计日志
```go
logger := audit.NewAuditLogger("auth-service", os.Stdout)
logger.Log("user.login", "u123", map[string]interface{}{"ip": "1.2.3.4"})
```

### 执行 GDPR 删除
```bash
python3 gdpr/delete_user_data.py
```

## 测试
```bash
go test ./audit/...
python -m unittest discover gdpr/
```

