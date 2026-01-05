# 日志系统 (Logging System)

> 生成式推荐系统日志收集、存储、查询和可视化解决方案

## 📋 目录

- [概述](#概述)
- [架构设计](#架构设计)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [组件详解](#组件详解)
- [配置说明](#配置说明)
- [日志格式规范](#日志格式规范)
- [告警规则](#告警规则)
- [Grafana Dashboard](#grafana-dashboard)
- [测试](#测试)
- [运维指南](#运维指南)
- [故障排查](#故障排查)

---

## 概述

本日志系统基于 **Loki + Promtail/Fluentd + Grafana** 技术栈，为生成式推荐系统提供完整的日志解决方案。

### 核心特性

- 🚀 **高性能日志收集**：支持 Kubernetes 环境下的日志自动收集
- 📊 **结构化日志**：JSON 格式日志，便于查询和分析
- 🔍 **分布式追踪**：通过 trace_id 关联跨服务日志
- ⚠️ **智能告警**：基于日志内容的实时告警
- 📈 **可视化分析**：Grafana Dashboard 提供丰富的日志分析视图

### 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| Loki | 2.9.x | 日志聚合和存储 |
| Promtail | 2.9.x | 日志收集代理 (首选) |
| Fluentd | 1.16.x | 日志收集代理 (备选) |
| Grafana | 10.x | 日志可视化和查询 |

---

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    生成式推荐系统日志架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ recommend-  │  │   user-     │  │   ugt-      │  应用层     │
│  │  service    │  │  service    │  │ inference   │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Promtail / Fluentd (DaemonSet)              │   │
│  │                    日志收集代理                           │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      Loki                                │   │
│  │                   日志聚合存储                            │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌───────────┐      ┌───────────┐      ┌───────────┐          │
│  │  Grafana  │      │   Ruler   │      │ Alertmanager│         │
│  │   查询     │      │   告警    │      │   通知     │          │
│  └───────────┘      └───────────┘      └───────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
devops/logging/
├── README.md                    # 本文档
├── loki/                        # Loki 配置
│   ├── loki-config.yaml        # 生产环境配置
│   ├── local-config.yaml       # 本地开发配置
│   ├── deployment.yaml         # Kubernetes 部署清单
│   └── rules/
│       └── alerts.yaml         # 告警规则
├── promtail/                    # Promtail 配置 (首选)
│   ├── promtail-config.yaml    # Promtail 配置
│   └── daemonset.yaml          # DaemonSet 部署清单
├── fluentd/                     # Fluentd 配置 (备选)
│   ├── fluent.conf             # 主配置文件
│   ├── parsers.conf            # 解析器配置
│   └── daemonset.yaml          # DaemonSet 部署清单
├── grafana/
│   └── dashboards/
│       ├── logs.json           # 主日志 Dashboard
│       └── inference-logs.json # 推理日志 Dashboard
└── tests/                       # 单元测试
    ├── __init__.py
    ├── conftest.py
    ├── requirements.txt
    ├── test_config_validation.py
    ├── test_log_format.py
    └── test_integration.py
```

---

## 快速开始

### 1. 部署 Loki

```bash
# 创建命名空间
kubectl create namespace recommend-prod

# 部署 Loki
kubectl apply -f loki/deployment.yaml

# 验证部署
kubectl get pods -n recommend-prod -l app=loki
```

### 2. 部署 Promtail

```bash
# 部署 Promtail DaemonSet
kubectl apply -f promtail/daemonset.yaml

# 验证部署
kubectl get pods -n recommend-prod -l app=promtail
```

### 3. 配置 Grafana

```bash
# 添加 Loki 数据源
# URL: http://loki:3100
# 类型: Loki

# 导入 Dashboard
# 使用 grafana/dashboards/ 下的 JSON 文件
```

### 4. 验证日志收集

```bash
# 查看 Promtail 日志
kubectl logs -n recommend-prod -l app=promtail --tail=50

# 测试 Loki 查询
curl -G -s "http://loki:3100/loki/api/v1/labels" | jq
```

---

## 组件详解

### Loki

Loki 是一个水平可扩展、高可用的日志聚合系统。

#### 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `http_listen_port` | 3100 | HTTP API 端口 |
| `grpc_listen_port` | 9096 | gRPC 端口 |
| `retention_period` | 720h | 日志保留时间 (30 天) |
| `ingestion_rate_mb` | 16 | 摄入速率限制 (MB/s) |
| `max_entries_limit_per_query` | 5000 | 单次查询最大条目数 |

#### 存储架构

```
Loki 存储结构:
├── /loki/chunks/              # 日志块存储
├── /loki/boltdb-shipper-*/    # 索引存储
├── /loki/wal/                 # 预写日志
└── /loki/rules/               # 告警规则
```

### Promtail

Promtail 是 Loki 的日志收集代理。

#### 核心功能

1. **服务发现**：自动发现 Kubernetes Pod
2. **标签提取**：从 Pod 元数据提取标签
3. **日志解析**：JSON 解析和字段提取
4. **管道处理**：多阶段日志处理管道

#### 管道阶段

```yaml
pipeline_stages:
  - cri: {}              # 解析 CRI 格式
  - json:                # JSON 解析
      expressions:
        level: level
        message: message
        ...
  - timestamp:           # 时间戳解析
      source: timestamp
      format: RFC3339Nano
  - labels:              # 标签提取
      level:
      service:
  - match:               # 条件过滤
      selector: '{level="DEBUG"}'
      action: drop
```

### Fluentd (备选)

Fluentd 作为 Promtail 的备选方案，提供更丰富的日志处理能力。

#### 适用场景

- 需要复杂的日志转换
- 需要多目标输出
- 需要敏感数据脱敏

---

## 配置说明

### 接口契约

根据 `devops/interfaces.yaml` 定义的日志接口：

```yaml
logging:
  format:
    required_fields:
      - timestamp    # 时间戳 (RFC3339)
      - level        # 日志级别
      - service      # 服务名称
      - trace_id     # 追踪 ID
      - message      # 日志消息
    optional_fields:
      - user_id      # 用户 ID
      - request_id   # 请求 ID
      - duration_ms  # 耗时 (毫秒)
      - error_stack  # 错误堆栈
  
  levels:
    - DEBUG
    - INFO
    - WARN
    - ERROR
    - FATAL
  
  labels:
    - app          # 应用名称
    - env          # 环境
    - pod          # Pod 名称
    - namespace    # 命名空间
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOKI_URL` | `http://loki:3100` | Loki 服务地址 |
| `LOG_LEVEL` | `info` | 日志级别 |
| `RETENTION_PERIOD` | `720h` | 保留时间 |

---

## 日志格式规范

### 标准日志格式

所有服务必须输出 JSON 格式日志：

```json
{
  "timestamp": "2025-01-05T10:30:00.123456Z",
  "level": "INFO",
  "service": "recommend-service",
  "trace_id": "trace-abc-123",
  "message": "Received recommendation request",
  "user_id": "user_12345",
  "request_id": "req_67890",
  "duration_ms": 45.2
}
```

### Go 服务日志示例

```go
import "go.uber.org/zap"

logger, _ := zap.NewProduction()
logger.Info("Processing request",
    zap.String("trace_id", traceID),
    zap.String("user_id", userID),
    zap.Int64("duration_ms", duration),
)
```

### Python 服务日志示例

```python
import json
import logging
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": "ugt-inference",
            "trace_id": getattr(record, 'trace_id', ''),
            "message": record.getMessage(),
        })

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger()
logger.addHandler(handler)
```

---

## 告警规则

### 告警概览

| 告警名称 | 级别 | 条件 | 说明 |
|----------|------|------|------|
| HighErrorLogRate | warning | ERROR > 10/s | 错误日志过多 |
| CriticalErrorLogRate | critical | ERROR > 50/s | 严重错误 |
| FatalLogDetected | critical | FATAL 出现 | 致命错误 |
| PanicDetected | critical | panic 关键字 | 程序崩溃 |
| OutOfMemoryDetected | critical | OOM 关键字 | 内存溢出 |

### 自定义告警

在 `loki/rules/alerts.yaml` 中添加：

```yaml
groups:
  - name: custom-alerts
    rules:
      - alert: CustomAlert
        expr: |
          count_over_time({app="my-app", level="ERROR"} [5m]) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "自定义告警"
          description: "描述信息"
```

---

## Grafana Dashboard

### 主日志 Dashboard

**UID**: `recommend-logs`

功能：
- 日志量趋势图 (按级别)
- 错误日志详情
- 服务日志流
- 慢请求日志
- Trace ID 追踪
- 日志分布统计

### 推理日志 Dashboard

**UID**: `recommend-inference-logs`

功能：
- 推理成功/失败统计
- 延迟分位数
- Batch Size 分布
- 错误日志
- 慢推理分析

### 变量

| 变量 | 类型 | 说明 |
|------|------|------|
| `$namespace` | query | 命名空间筛选 |
| `$app` | query | 应用筛选 |
| `$level` | query | 日志级别筛选 |
| `$trace_id` | textbox | Trace ID 追踪 |

---

## 测试

### 运行测试

```bash
# 安装依赖
cd devops/logging/tests
pip install -r requirements.txt

# 运行所有测试
pytest -v

# 运行配置验证测试
pytest test_config_validation.py -v

# 运行日志格式测试
pytest test_log_format.py -v

# 运行集成测试
pytest test_integration.py -v

# 生成覆盖率报告
pytest --cov=. --cov-report=html
```

### 测试类别

| 测试文件 | 说明 |
|----------|------|
| `test_config_validation.py` | 配置文件验证 |
| `test_log_format.py` | 日志格式验证 |
| `test_integration.py` | 组件集成测试 |

---

## 运维指南

### 日常运维

#### 查看日志系统状态

```bash
# Loki 状态
kubectl get pods -n recommend-prod -l app=loki
curl http://loki:3100/ready

# Promtail 状态
kubectl get pods -n recommend-prod -l app=promtail
curl http://promtail:9080/ready
```

#### 日志查询

```bash
# 使用 LogCLI
logcli query '{app="recommend-service"}'

# 查询错误日志
logcli query '{level="ERROR"} | json'

# 按 trace_id 查询
logcli query '{trace_id="abc123"}'
```

### 性能调优

#### Loki 调优

```yaml
limits_config:
  ingestion_rate_mb: 32        # 增加摄入速率
  per_stream_rate_limit: 10MB  # 增加流速率限制
  max_query_parallelism: 64    # 增加查询并行度
```

#### Promtail 调优

```yaml
clients:
  - url: http://loki:3100/loki/api/v1/push
    batchsize: 2097152  # 增加批量大小 (2MB)
    batchwait: 2s       # 增加批量等待时间
```

### 扩容

```bash
# 增加 Loki 副本 (需要配置分布式存储)
kubectl scale deployment loki -n recommend-prod --replicas=3

# Promtail 自动在每个节点运行 (DaemonSet)
```

---

## 故障排查

### 常见问题

#### 1. 日志不显示

**检查步骤**：
1. 确认 Promtail Pod 运行正常
2. 检查 Promtail 日志
3. 验证 Loki 连接

```bash
kubectl logs -n recommend-prod -l app=promtail --tail=100
```

#### 2. 日志延迟

**可能原因**：
- 网络延迟
- Loki 过载
- 批量配置过大

**解决方案**：
```yaml
# 减少批量等待时间
clients:
  - url: http://loki:3100/loki/api/v1/push
    batchwait: 500ms
```

#### 3. 查询超时

**解决方案**：
```yaml
# 增加查询超时
querier:
  query_timeout: 10m
```

#### 4. 磁盘空间不足

**解决方案**：
```yaml
# 减少保留时间
table_manager:
  retention_period: 168h  # 7 天
```

### 日志

```bash
# Loki 日志
kubectl logs -n recommend-prod deployment/loki

# Promtail 日志
kubectl logs -n recommend-prod daemonset/promtail

# Fluentd 日志
kubectl logs -n recommend-prod daemonset/fluentd
```

---

## 参考资料

- [Loki 官方文档](https://grafana.com/docs/loki/latest/)
- [Promtail 配置](https://grafana.com/docs/loki/latest/clients/promtail/configuration/)
- [LogQL 查询语言](https://grafana.com/docs/loki/latest/logql/)
- [Fluentd 官方文档](https://docs.fluentd.org/)

---

## 联系方式

如有问题，请联系 DevOps 团队或在项目仓库提交 Issue。

