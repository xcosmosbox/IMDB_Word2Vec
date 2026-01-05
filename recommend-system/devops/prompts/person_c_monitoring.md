# Person C: 监控告警系统

## 你的角色
你是一名 DevOps 工程师，负责实现生成式推荐系统的 **监控告警系统**，包括 Prometheus 规则、Grafana 仪表板、AlertManager 配置等。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读接口定义文件：**

```
devops/interfaces.yaml
```

你需要实现的契约：

```yaml
monitoring:
  metrics:
    go_services:
      - http_requests_total{service, method, path, status}
      - http_request_duration_seconds{service, method, path}
    inference:
      - inference_requests_total{model}
      - inference_latency_seconds{model, batch_size}
  
  alert_rules:
    critical:
      - ServiceDown
      - HighErrorRate
      - HighLatency
```

---

## 你的任务

```
devops/monitoring/
├── prometheus/
│   ├── prometheus.yaml
│   ├── rules/
│   │   ├── recording-rules.yaml
│   │   └── alerting-rules.yaml
│   └── scrape-configs/
│       ├── kubernetes.yaml
│       └── custom.yaml
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── prometheus.yaml
│   │   └── dashboards/
│   │       └── default.yaml
│   └── dashboards/
│       ├── overview.json
│       ├── services.json
│       ├── inference.json
│       └── database.json
└── alertmanager/
    ├── alertmanager.yaml
    └── templates/
        └── slack.tmpl
```

---

## 1. Prometheus 配置 (prometheus/prometheus.yaml)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'recommend-prod'
    env: 'production'

rule_files:
  - /etc/prometheus/rules/*.yaml

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

scrape_configs:
  # Kubernetes 服务发现
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: pod

  # Go 服务
  - job_name: 'recommend-service'
    static_configs:
      - targets: ['recommend-service:9091']
    metrics_path: /metrics

  # Python 推理服务
  - job_name: 'ugt-inference'
    static_configs:
      - targets: ['ugt-inference:9094']
    metrics_path: /metrics

  # Redis
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  # PostgreSQL
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  # Milvus
  - job_name: 'milvus'
    static_configs:
      - targets: ['milvus:9091']
```

---

## 2. 告警规则 (prometheus/rules/alerting-rules.yaml)

```yaml
groups:
  # ==========================================================================
  # 服务可用性告警
  # ==========================================================================
  - name: service-availability
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "服务 {{ $labels.job }} 不可用"
          description: "服务 {{ $labels.instance }} 已停止响应超过 1 分钟"
          runbook_url: "https://wiki.example.com/runbooks/service-down"

      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
          /
          sum(rate(http_requests_total[5m])) by (service)
          > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "服务 {{ $labels.service }} 错误率过高"
          description: "错误率 {{ $value | humanizePercentage }} 超过 5%"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.99, 
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)
          ) > 0.5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "服务 {{ $labels.service }} P99 延迟过高"
          description: "P99 延迟 {{ $value | humanizeDuration }} 超过 500ms"

  # ==========================================================================
  # 推理服务告警
  # ==========================================================================
  - name: inference-alerts
    rules:
      - alert: InferenceLatencyHigh
        expr: |
          histogram_quantile(0.95, 
            sum(rate(inference_latency_seconds_bucket[5m])) by (le, model)
          ) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "模型 {{ $labels.model }} 推理延迟过高"
          description: "P95 推理延迟 {{ $value | humanizeDuration }}"

      - alert: GPUMemoryHigh
        expr: gpu_memory_usage_bytes / gpu_memory_total_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU 内存使用率过高"
          description: "GPU {{ $labels.device }} 内存使用 {{ $value | humanizePercentage }}"

      - alert: ModelLoadFailed
        expr: increase(model_load_errors_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "模型加载失败"
          description: "模型 {{ $labels.model }} 加载失败"

  # ==========================================================================
  # 资源使用告警
  # ==========================================================================
  - name: resource-alerts
    rules:
      - alert: HighMemoryUsage
        expr: |
          container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "容器内存使用率过高"
          description: "Pod {{ $labels.pod }} 内存使用 {{ $value | humanizePercentage }}"

      - alert: HighCPUUsage
        expr: |
          sum(rate(container_cpu_usage_seconds_total[5m])) by (pod)
          /
          sum(container_spec_cpu_quota/container_spec_cpu_period) by (pod)
          > 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "容器 CPU 使用率过高"
          description: "Pod {{ $labels.pod }} CPU 使用 {{ $value | humanizePercentage }}"

      - alert: PodCrashLooping
        expr: increase(kube_pod_container_status_restarts_total[1h]) > 3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Pod 频繁重启"
          description: "Pod {{ $labels.pod }} 在过去 1 小时重启 {{ $value }} 次"

  # ==========================================================================
  # 数据库告警
  # ==========================================================================
  - name: database-alerts
    rules:
      - alert: PostgresDown
        expr: pg_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL 不可用"
          description: "PostgreSQL 实例 {{ $labels.instance }} 已停止响应"

      - alert: PostgresHighConnections
        expr: |
          pg_stat_database_numbackends / pg_settings_max_connections > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "PostgreSQL 连接数过高"
          description: "连接使用率 {{ $value | humanizePercentage }}"

      - alert: RedisDown
        expr: redis_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis 不可用"

      - alert: RedisHighMemory
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis 内存使用率过高"

  # ==========================================================================
  # 业务指标告警
  # ==========================================================================
  - name: business-alerts
    rules:
      - alert: LowRecommendationCTR
        expr: |
          sum(rate(recommendation_clicks_total[1h]))
          /
          sum(rate(recommendation_impressions_total[1h]))
          < 0.01
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "推荐点击率过低"
          description: "CTR {{ $value | humanizePercentage }} 低于 1%"

      - alert: HighCacheHitMiss
        expr: |
          sum(rate(cache_misses_total[5m]))
          /
          sum(rate(cache_requests_total[5m]))
          > 0.3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "缓存命中率过低"
          description: "缓存未命中率 {{ $value | humanizePercentage }}"
```

---

## 3. AlertManager 配置 (alertmanager/alertmanager.yaml)

```yaml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/xxx/xxx/xxx'

route:
  receiver: 'default'
  group_by: ['alertname', 'severity', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  
  routes:
    # Critical 告警 - 立即通知
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 10s
      repeat_interval: 1h
    
    # Warning 告警
    - match:
        severity: warning
      receiver: 'warning-alerts'
      repeat_interval: 4h
    
    # 业务告警 - 发送到业务频道
    - match_re:
        alertname: 'Low.*CTR|High.*CacheHitMiss'
      receiver: 'business-alerts'

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#alerts'
        send_resolved: true
        title: '{{ template "slack.title" . }}'
        text: '{{ template "slack.text" . }}'

  - name: 'critical-alerts'
    slack_configs:
      - channel: '#alerts-critical'
        send_resolved: true
        title: '🚨 {{ template "slack.title" . }}'
        text: '{{ template "slack.text" . }}'
    pagerduty_configs:
      - service_key: '<pagerduty-service-key>'
        severity: critical

  - name: 'warning-alerts'
    slack_configs:
      - channel: '#alerts-warning'
        send_resolved: true

  - name: 'business-alerts'
    slack_configs:
      - channel: '#business-metrics'
        send_resolved: true

inhibit_rules:
  # 如果服务 Down，抑制其他告警
  - source_match:
      alertname: 'ServiceDown'
    target_match_re:
      alertname: '.*'
    equal: ['service']
```

---

## 4. Grafana Dashboard - 系统总览 (grafana/dashboards/overview.json)

```json
{
  "dashboard": {
    "title": "推荐系统 - 总览",
    "uid": "recommend-overview",
    "timezone": "browser",
    "refresh": "30s",
    "panels": [
      {
        "title": "服务状态",
        "type": "stat",
        "gridPos": { "x": 0, "y": 0, "w": 6, "h": 4 },
        "targets": [
          {
            "expr": "sum(up{job=~\"recommend.*\"})",
            "legendFormat": "在线服务数"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                { "value": 0, "color": "red" },
                { "value": 3, "color": "yellow" },
                { "value": 5, "color": "green" }
              ]
            }
          }
        }
      },
      {
        "title": "请求速率 (QPS)",
        "type": "stat",
        "gridPos": { "x": 6, "y": 0, "w": 6, "h": 4 },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m]))",
            "legendFormat": "QPS"
          }
        ]
      },
      {
        "title": "P99 延迟",
        "type": "stat",
        "gridPos": { "x": 12, "y": 0, "w": 6, "h": 4 },
        "targets": [
          {
            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P99"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "thresholds": {
              "steps": [
                { "value": 0, "color": "green" },
                { "value": 0.2, "color": "yellow" },
                { "value": 0.5, "color": "red" }
              ]
            }
          }
        }
      },
      {
        "title": "错误率",
        "type": "stat",
        "gridPos": { "x": 18, "y": 0, "w": 6, "h": 4 },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))",
            "legendFormat": "Error Rate"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percentunit",
            "thresholds": {
              "steps": [
                { "value": 0, "color": "green" },
                { "value": 0.01, "color": "yellow" },
                { "value": 0.05, "color": "red" }
              ]
            }
          }
        }
      },
      {
        "title": "请求趋势",
        "type": "timeseries",
        "gridPos": { "x": 0, "y": 4, "w": 12, "h": 8 },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (service)",
            "legendFormat": "{{ service }}"
          }
        ]
      },
      {
        "title": "延迟分布",
        "type": "timeseries",
        "gridPos": { "x": 12, "y": 4, "w": 12, "h": 8 },
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.90, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P90"
          },
          {
            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P99"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s"
          }
        }
      }
    ]
  }
}
```

---

## 注意事项

1. 告警规则要有合理的阈值
2. 配置告警抑制避免告警风暴
3. Dashboard 要有清晰的层次
4. 记录规则提高查询性能
5. 配置合理的数据保留策略

## 输出要求

请输出完整的监控配置，包含：
1. Prometheus 完整配置
2. 告警规则
3. AlertManager 配置
4. Grafana Dashboard JSON

