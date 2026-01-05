# 生成式推荐系统 - 监控告警系统

## 📋 概述

本目录包含生成式推荐系统的完整监控告警解决方案，基于 Prometheus + Grafana + AlertManager 技术栈构建，提供全方位的可观测性能力。

### 核心功能

- **指标收集**: Prometheus 自动发现和抓取服务指标
- **可视化**: Grafana 仪表板提供实时监控视图
- **告警通知**: AlertManager 多渠道告警通知
- **SLO 监控**: 基于 SLI/SLO 的服务质量监控

---

## 📁 目录结构

```
monitoring/
├── prometheus/                    # Prometheus 配置
│   ├── prometheus.yaml           # 主配置文件
│   ├── rules/                    # 规则文件
│   │   ├── alerting-rules.yaml   # 告警规则
│   │   └── recording-rules.yaml  # 记录规则
│   └── scrape-configs/           # 抓取配置
│       ├── kubernetes.yaml       # K8s 服务发现
│       └── custom.yaml           # 自定义服务
│
├── grafana/                       # Grafana 配置
│   ├── provisioning/             # 自动配置
│   │   ├── datasources/          # 数据源配置
│   │   │   └── prometheus.yaml
│   │   └── dashboards/           # 仪表板配置
│   │       └── default.yaml
│   └── dashboards/               # 仪表板定义
│       ├── overview.json         # 系统总览
│       ├── services.json         # 服务监控
│       ├── inference.json        # 推理监控
│       └── database.json         # 数据库监控
│
├── alertmanager/                  # AlertManager 配置
│   ├── alertmanager.yaml         # 主配置文件
│   └── templates/                # 告警模板
│       └── slack.tmpl            # Slack 模板
│
├── tests/                         # 单元测试
│   ├── test_prometheus_config.py
│   ├── test_alertmanager_config.py
│   ├── test_grafana_dashboards.py
│   ├── conftest.py
│   └── requirements.txt
│
└── README.md                      # 本文档
```

---

## 🚀 快速开始

### 1. 前置条件

- Kubernetes 集群 (1.20+)
- Helm 3.x
- kubectl 已配置

### 2. 部署 Prometheus Stack

```bash
# 添加 Helm 仓库
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# 创建命名空间
kubectl create namespace monitoring

# 部署 kube-prometheus-stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values prometheus/prometheus.yaml
```

### 3. 应用自定义配置

```bash
# 应用告警规则
kubectl apply -f prometheus/rules/ -n monitoring

# 应用 AlertManager 配置
kubectl create secret generic alertmanager-config \
  --from-file=alertmanager.yaml=alertmanager/alertmanager.yaml \
  -n monitoring

# 导入 Grafana 仪表板
kubectl create configmap grafana-dashboards \
  --from-file=grafana/dashboards/ \
  -n monitoring
```

### 4. 访问服务

```bash
# Prometheus UI
kubectl port-forward svc/prometheus-operated 9090:9090 -n monitoring

# Grafana UI (默认账号: admin/admin)
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring

# AlertManager UI
kubectl port-forward svc/alertmanager-operated 9093:9093 -n monitoring
```

---

## 📊 监控指标说明

### Go 后端服务指标

基于 `interfaces.yaml` 中定义的指标契约：

| 指标名称 | 类型 | 标签 | 说明 |
|---------|------|------|------|
| `http_requests_total` | Counter | service, method, path, status | HTTP 请求总数 |
| `http_request_duration_seconds` | Histogram | service, method, path | HTTP 请求延迟 |
| `grpc_requests_total` | Counter | service, method, status | gRPC 请求总数 |
| `cache_hit_ratio` | Gauge | cache_type | 缓存命中率 |
| `db_query_duration_seconds` | Histogram | query_type | 数据库查询延迟 |

### 推理服务指标

| 指标名称 | 类型 | 标签 | 说明 |
|---------|------|------|------|
| `inference_requests_total` | Counter | model, status | 推理请求总数 |
| `inference_latency_seconds` | Histogram | model, batch_size | 推理延迟 |
| `model_load_time_seconds` | Gauge | model | 模型加载时间 |
| `gpu_memory_usage_bytes` | Gauge | device | GPU 内存使用 |

### 服务端口契约

| 服务 | HTTP 端口 | gRPC 端口 | Metrics 端口 |
|------|----------|----------|-------------|
| recommend-service | 8080 | 9090 | 9091 |
| user-service | 8081 | 9091 | 9092 |
| item-service | 8082 | 9092 | 9093 |
| ugt-inference | - | 50051 | 9094 |

---

## 🚨 告警规则说明

### Critical 级别告警

这些告警需要立即响应，会触发 PagerDuty 和 Slack 通知：

| 告警名称 | 触发条件 | 持续时间 |
|---------|---------|---------|
| `ServiceDown` | `up == 0` | 1 分钟 |
| `HighErrorRate` | 5xx 错误率 > 5% | 5 分钟 |
| `HighLatency` | P99 延迟 > 500ms | 5 分钟 |
| `PostgresDown` | `pg_up == 0` | 1 分钟 |
| `RedisDown` | `redis_up == 0` | 1 分钟 |

### Warning 级别告警

这些告警需要关注但不紧急：

| 告警名称 | 触发条件 | 持续时间 |
|---------|---------|---------|
| `HighMemoryUsage` | 内存使用 > 80% | 10 分钟 |
| `HighCPUUsage` | CPU 使用 > 70% | 10 分钟 |
| `InferenceLatencyHigh` | 推理 P95 > 200ms | 5 分钟 |
| `GPUMemoryHigh` | GPU 内存 > 90% | 5 分钟 |
| `PodCrashLooping` | 1 小时内重启 > 3 次 | 5 分钟 |

### 告警路由

```
告警分发逻辑:
├── severity: critical
│   └── 接收者: critical-alerts (Slack + PagerDuty + Email)
│
├── team: ml
│   ├── severity: critical → ml-team-critical
│   └── severity: warning → ml-team-alerts
│
├── team: dba
│   ├── severity: critical → dba-team-critical
│   └── severity: warning → dba-team-alerts
│
├── 业务告警 (CTR, Cache)
│   └── 接收者: business-alerts
│
└── 默认
    └── 接收者: default-receiver
```

---

## 📈 Grafana 仪表板说明

### 1. 系统总览 (overview)

**UID**: `recommend-overview`

提供系统级别的核心指标概览：
- 在线服务数量
- 总体 QPS
- P99 延迟
- 错误率
- 推荐 CTR
- 缓存命中率

**使用场景**: 日常巡检、故障定位入口

### 2. 服务监控 (services)

**UID**: `recommend-services`

提供单个服务的详细监控：
- HTTP/gRPC 请求速率
- 延迟分位数
- 状态码分布
- 错误率趋势
- 缓存命中率
- 数据库查询延迟

**变量**: 
- `service`: 选择要查看的服务

### 3. 推理监控 (inference)

**UID**: `recommend-inference`

提供 ML 推理服务的专项监控：
- 推理 QPS 和错误率
- 推理延迟分位数
- 批处理大小分布
- GPU 利用率
- GPU 内存使用
- GPU 温度和功率
- 模型加载状态

**变量**:
- `model`: 选择要查看的模型

### 4. 数据库监控 (database)

**UID**: `recommend-database`

提供数据库层的监控：
- PostgreSQL: 连接数、缓存命中率、事务速率
- Redis: 内存使用、命中率、命令执行
- Milvus: 搜索延迟、操作速率

---

## 🔧 配置说明

### Prometheus 配置

#### 全局配置
```yaml
global:
  scrape_interval: 15s      # 抓取间隔
  evaluation_interval: 15s  # 规则评估间隔
  external_labels:
    cluster: 'recommend-prod'
    env: 'production'
```

#### 添加新的抓取目标

编辑 `scrape-configs/custom.yaml`:

```yaml
- job_name: 'my-new-service'
  static_configs:
    - targets: ['my-service:9091']
  metrics_path: /metrics
  relabel_configs:
    - source_labels: []
      target_label: service
      replacement: 'my-new-service'
```

### AlertManager 配置

#### 添加新的接收者

编辑 `alertmanager/alertmanager.yaml`:

```yaml
receivers:
  - name: 'my-team-alerts'
    slack_configs:
      - channel: '#my-team-alerts'
        send_resolved: true
```

#### 添加新的路由

```yaml
routes:
  - match:
      team: my-team
    receiver: 'my-team-alerts'
```

### 添加新的告警规则

编辑 `prometheus/rules/alerting-rules.yaml`:

```yaml
- alert: MyNewAlert
  expr: |
    my_metric > 100
  for: 5m
  labels:
    severity: warning
    team: my-team
  annotations:
    summary: "发现异常情况"
    description: "指标值 {{ $value }} 超过阈值"
```

---

## 🧪 测试

### 运行单元测试

```bash
cd monitoring

# 安装测试依赖
pip install -r tests/requirements.txt

# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_prometheus_config.py -v

# 生成覆盖率报告
pytest tests/ --cov=. --cov-report=html
```

### 测试内容

| 测试文件 | 测试内容 |
|---------|---------|
| `test_prometheus_config.py` | Prometheus 配置、告警规则、记录规则 |
| `test_alertmanager_config.py` | AlertManager 配置、路由、接收者、模板 |
| `test_grafana_dashboards.py` | 仪表板 JSON、面板、变量 |

---

## 📝 开发指南

### 添加新的仪表板

1. 在 Grafana UI 中创建仪表板
2. 导出 JSON
3. 保存到 `grafana/dashboards/` 目录
4. 添加必要的标签: `["recommend-system", "your-tag"]`
5. 编写对应的测试用例

### 添加新的告警规则

1. 确定告警的严重级别 (critical/warning/info)
2. 编写 PromQL 表达式
3. 添加到对应的规则组
4. 包含必要的标签和注解:
   - `severity`: 严重级别
   - `team`: 负责团队
   - `summary`: 简要描述
   - `description`: 详细描述
   - `runbook_url`: 处理手册链接 (可选)

### 记录规则命名约定

遵循 `namespace:metric:aggregation` 格式：

```yaml
# 好的命名
service:http_requests:rate5m
model:inference_latency_p99:rate5m
gpu:memory_usage_ratio

# 不好的命名
http_requests_rate
latency_p99
```

---

## 🔗 相关资源

### 文档链接

- [Prometheus 官方文档](https://prometheus.io/docs/)
- [Grafana 官方文档](https://grafana.com/docs/)
- [AlertManager 官方文档](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [PromQL 语法参考](https://prometheus.io/docs/prometheus/latest/querying/basics/)

### 项目相关

- 接口定义: `devops/interfaces.yaml`
- 系统架构: `docs/生成式推荐系统架构设计.md`

---

## ❓ 常见问题

### 1. 告警一直处于 Pending 状态

**原因**: 告警表达式结果为 true 但未超过 `for` 持续时间

**解决**: 等待持续时间结束，或检查指标数据是否正常

### 2. Grafana 仪表板显示 "No data"

**原因**: 
- 数据源配置错误
- 指标名称不匹配
- 服务未暴露指标

**解决**:
1. 检查 Prometheus 是否能抓取到目标
2. 在 Prometheus UI 验证 PromQL 表达式
3. 确认服务指标端点正常

### 3. AlertManager 没有发送通知

**原因**:
- Webhook URL 配置错误
- 告警被抑制
- 路由配置不正确

**解决**:
1. 检查 AlertManager 配置
2. 查看 AlertManager UI 中的告警状态
3. 验证抑制规则是否生效

---

## 📞 联系方式

如有问题，请联系:

- **平台团队**: platform@example.com
- **ML 团队**: ml@example.com
- **DBA 团队**: dba@example.com
- **Slack**: #recommend-monitoring

