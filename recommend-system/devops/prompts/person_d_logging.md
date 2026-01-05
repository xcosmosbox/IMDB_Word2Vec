# Person D: 日志系统

## 你的角色
你是一名 DevOps 工程师，负责实现生成式推荐系统的 **日志系统**，包括日志收集、存储、查询和可视化。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读接口定义文件：**

```
devops/interfaces.yaml
```

你需要实现的契约：

```yaml
logging:
  format:
    required_fields:
      - timestamp
      - level
      - service
      - trace_id
      - message
  
  levels:
    - DEBUG
    - INFO
    - WARN
    - ERROR
    - FATAL
```

---

## 你的任务

```
devops/logging/
├── loki/
│   ├── loki-config.yaml
│   ├── local-config.yaml
│   └── deployment.yaml
├── promtail/
│   ├── promtail-config.yaml
│   └── daemonset.yaml
├── fluentd/
│   ├── fluent.conf
│   ├── parsers.conf
│   └── daemonset.yaml
└── grafana/
    └── dashboards/
        └── logs.json
```

---

## 1. Loki 配置 (loki/loki-config.yaml)

```yaml
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2024-01-01
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    cache_ttl: 24h
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

compactor:
  working_directory: /loki/boltdb-shipper-compactor
  shared_store: filesystem

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
  max_entries_limit_per_query: 5000
  ingestion_rate_mb: 16
  ingestion_burst_size_mb: 32
  per_stream_rate_limit: 5MB
  per_stream_rate_limit_burst: 15MB

chunk_store_config:
  max_look_back_period: 720h

table_manager:
  retention_deletes_enabled: true
  retention_period: 720h

ruler:
  storage:
    type: local
    local:
      directory: /loki/rules
  rule_path: /loki/rules-temp
  alertmanager_url: http://alertmanager:9093
  ring:
    kvstore:
      store: inmemory
  enable_api: true

analytics:
  reporting_enabled: false
```

---

## 2. Promtail 配置 (promtail/promtail-config.yaml)

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push
    tenant_id: recommend

scrape_configs:
  # Kubernetes Pod 日志
  - job_name: kubernetes-pods
    kubernetes_sd_configs:
      - role: pod
    
    relabel_configs:
      # 只收集有标签的 Pod
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: .+
      
      # 设置命名空间标签
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: namespace
      
      # 设置 Pod 名称标签
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: pod
      
      # 设置容器名称标签
      - source_labels: [__meta_kubernetes_pod_container_name]
        action: replace
        target_label: container
      
      # 设置应用标签
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: replace
        target_label: app
      
      # 设置日志路径
      - source_labels: [__meta_kubernetes_pod_uid, __meta_kubernetes_pod_container_name]
        target_label: __path__
        separator: /
        replacement: /var/log/pods/*$1/$2/*.log
    
    pipeline_stages:
      # 解析 JSON 日志
      - json:
          expressions:
            level: level
            message: message
            service: service
            trace_id: trace_id
            timestamp: timestamp
            duration_ms: duration_ms
            user_id: user_id
            request_id: request_id
      
      # 设置时间戳
      - timestamp:
          source: timestamp
          format: RFC3339Nano
      
      # 设置日志级别标签
      - labels:
          level:
          service:
          trace_id:
      
      # 过滤 DEBUG 日志（生产环境）
      - match:
          selector: '{level="DEBUG"}'
          action: drop
          drop_counter_reason: debug_logs
      
      # 提取错误栈
      - multiline:
          firstline: '^\d{4}-\d{2}-\d{2}'
          max_wait_time: 3s
      
      # 指标提取
      - metrics:
          log_lines_total:
            type: Counter
            description: "Total log lines"
            config:
              match_all: true
              action: inc
          error_lines_total:
            type: Counter
            description: "Error log lines"
            source: level
            config:
              value: ERROR
              action: inc

  # 系统日志
  - job_name: system
    static_configs:
      - targets:
          - localhost
        labels:
          job: system
          __path__: /var/log/syslog

  # 应用特定日志
  - job_name: inference-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: inference
          __path__: /var/log/inference/*.log
    
    pipeline_stages:
      - json:
          expressions:
            model: model
            batch_size: batch_size
            latency_ms: latency_ms
            status: status
      - labels:
          model:
          status:
```

---

## 3. Fluentd 配置 (fluentd/fluent.conf)

```conf
# 系统设置
<system>
  log_level info
  workers 4
</system>

# 输入：Kubernetes 容器日志
<source>
  @type tail
  @id kubernetes-containers
  path /var/log/containers/*.log
  pos_file /var/log/fluentd-containers.log.pos
  tag kubernetes.*
  read_from_head true
  <parse>
    @type json
    time_key time
    time_format %Y-%m-%dT%H:%M:%S.%NZ
  </parse>
</source>

# 输入：应用日志
<source>
  @type tail
  @id application-logs
  path /var/log/apps/**/*.log
  pos_file /var/log/fluentd-apps.log.pos
  tag app.*
  <parse>
    @type json
    time_key timestamp
    time_format %Y-%m-%dT%H:%M:%S.%NZ
  </parse>
</source>

# 过滤：添加 Kubernetes 元数据
<filter kubernetes.**>
  @type kubernetes_metadata
  @id filter_kube_metadata
  kubernetes_url https://kubernetes.default.svc
  verify_ssl true
  ca_file /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
  bearer_token_file /var/run/secrets/kubernetes.io/serviceaccount/token
  skip_labels false
  skip_container_metadata false
  skip_master_url false
  skip_namespace_metadata false
</filter>

# 过滤：解析嵌套 JSON
<filter kubernetes.**>
  @type parser
  @id parse_log_field
  key_name log
  reserve_data true
  remove_key_name_field true
  <parse>
    @type json
  </parse>
</filter>

# 过滤：添加自定义字段
<filter **>
  @type record_transformer
  @id add_fields
  <record>
    cluster "recommend-prod"
    environment "production"
    fluentd_host "#{Socket.gethostname}"
  </record>
</filter>

# 过滤：日志级别过滤（生产环境过滤 DEBUG）
<filter **>
  @type grep
  <exclude>
    key level
    pattern /^DEBUG$/
  </exclude>
</filter>

# 输出：Loki
<match **>
  @type loki
  @id loki_output
  url "http://loki:3100"
  
  <label>
    app $.kubernetes.labels.app
    namespace $.kubernetes.namespace_name
    pod $.kubernetes.pod_name
    container $.kubernetes.container_name
    level $.level
    service $.service
  </label>
  
  <buffer>
    @type file
    path /var/log/fluentd-buffers/loki
    flush_mode interval
    flush_interval 5s
    flush_thread_count 4
    retry_type exponential_backoff
    retry_wait 1s
    retry_max_interval 60s
    retry_forever true
    overflow_action block
    chunk_limit_size 8MB
    total_limit_size 2GB
  </buffer>
</match>

# 输出：备份到 S3（可选）
<match backup.**>
  @type s3
  @id s3_backup
  
  aws_key_id "#{ENV['AWS_ACCESS_KEY_ID']}"
  aws_sec_key "#{ENV['AWS_SECRET_ACCESS_KEY']}"
  s3_bucket recommend-logs-backup
  s3_region us-east-1
  
  path logs/%Y/%m/%d/
  
  <buffer time>
    @type file
    path /var/log/fluentd-buffers/s3
    timekey 1h
    timekey_wait 10m
    chunk_limit_size 256m
  </buffer>
  
  <format>
    @type json
  </format>
</match>
```

---

## 4. Grafana 日志 Dashboard (grafana/dashboards/logs.json)

```json
{
  "dashboard": {
    "title": "推荐系统 - 日志分析",
    "uid": "recommend-logs",
    "timezone": "browser",
    "refresh": "10s",
    "panels": [
      {
        "title": "日志量趋势",
        "type": "timeseries",
        "gridPos": { "x": 0, "y": 0, "w": 24, "h": 6 },
        "targets": [
          {
            "expr": "sum(rate({app=~\"$app\"} | json [5m])) by (level)",
            "legendFormat": "{{ level }}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "custom": {
              "drawStyle": "bars",
              "fillOpacity": 50
            }
          },
          "overrides": [
            { "matcher": { "id": "byName", "options": "ERROR" }, "properties": [{ "id": "color", "value": { "fixedColor": "red" }}]},
            { "matcher": { "id": "byName", "options": "WARN" }, "properties": [{ "id": "color", "value": { "fixedColor": "yellow" }}]},
            { "matcher": { "id": "byName", "options": "INFO" }, "properties": [{ "id": "color", "value": { "fixedColor": "green" }}]}
          ]
        }
      },
      {
        "title": "错误日志",
        "type": "logs",
        "gridPos": { "x": 0, "y": 6, "w": 24, "h": 10 },
        "targets": [
          {
            "expr": "{app=~\"$app\", level=\"ERROR\"} | json",
            "legendFormat": ""
          }
        ],
        "options": {
          "showLabels": true,
          "showTime": true,
          "wrapLogMessage": true,
          "sortOrder": "Descending",
          "dedupStrategy": "none",
          "enableLogDetails": true
        }
      },
      {
        "title": "服务日志",
        "type": "logs",
        "gridPos": { "x": 0, "y": 16, "w": 24, "h": 10 },
        "targets": [
          {
            "expr": "{app=~\"$app\"} | json | line_format \"[{{.level}}] {{.service}} - {{.message}}\"",
            "legendFormat": ""
          }
        ],
        "options": {
          "showLabels": true,
          "showTime": true,
          "wrapLogMessage": false
        }
      },
      {
        "title": "慢请求日志 (>500ms)",
        "type": "logs",
        "gridPos": { "x": 0, "y": 26, "w": 24, "h": 8 },
        "targets": [
          {
            "expr": "{app=~\"$app\"} | json | duration_ms > 500",
            "legendFormat": ""
          }
        ]
      }
    ],
    "templating": {
      "list": [
        {
          "name": "app",
          "type": "query",
          "datasource": "Loki",
          "query": "label_values(app)",
          "multi": true,
          "includeAll": true
        }
      ]
    }
  }
}
```

---

## 5. Loki 告警规则 (loki/rules/alerts.yaml)

```yaml
groups:
  - name: log-alerts
    rules:
      - alert: HighErrorLogRate
        expr: |
          sum(rate({level="ERROR"}[5m])) by (app) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "应用 {{ $labels.app }} 错误日志过多"
          description: "错误日志速率 {{ $value | printf \"%.2f\" }} 条/秒"
      
      - alert: PanicDetected
        expr: |
          count_over_time({level="FATAL"} | json | message =~ "panic.*" [5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "检测到 Panic"
          description: "应用 {{ $labels.app }} 发生 Panic"
```

---

## 注意事项

1. 日志格式统一为 JSON
2. 配置合理的日志保留策略
3. 敏感信息脱敏处理
4. 配置日志采样（高流量场景）
5. 监控日志系统本身

## 输出要求

请输出完整的日志系统配置，包含：
1. Loki 完整配置
2. Promtail/Fluentd 配置
3. Grafana Dashboard
4. K8s 部署文件

