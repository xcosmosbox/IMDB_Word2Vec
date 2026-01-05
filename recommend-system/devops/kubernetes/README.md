# Kubernetes 配置文档

> 生成式推荐系统 Kubernetes 部署配置
> 
> 作者: Person B (DevOps Engineer)
> 
> 版本: 1.0.0

---

## 📋 目录

- [概述](#概述)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [基础配置详解](#基础配置详解)
- [环境配置](#环境配置)
- [Istio Service Mesh](#istio-service-mesh)
- [Ingress 配置](#ingress-配置)
- [安全配置](#安全配置)
- [监控与可观测性](#监控与可观测性)
- [故障排除](#故障排除)
- [最佳实践](#最佳实践)
- [测试](#测试)

---

## 概述

本目录包含生成式推荐系统的完整 Kubernetes 部署配置，采用 Kustomize 管理多环境配置，支持：

- ✅ **多环境部署**: 开发 (dev) 和生产 (prod) 环境
- ✅ **Service Mesh**: Istio 流量管理、安全认证
- ✅ **自动伸缩**: HPA 基于 CPU/内存/自定义指标
- ✅ **高可用**: PDB、反亲和性、跨可用区部署
- ✅ **金丝雀发布**: 灰度发布、A/B 测试
- ✅ **安全加固**: 网络策略、RBAC、安全上下文

### 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Internet                                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │    Istio Gateway /    │
                    │    NGINX Ingress      │
                    └───────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌───────────────┐
│   User App    │     │ Recommend Svc   │     │  Admin App    │
│   (Frontend)  │     │   (Backend)     │     │  (Frontend)   │
└───────────────┘     └────────┬────────┘     └───────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ User Service  │     │ Item Service  │     │ UGT Inference │
│               │     │               │     │    (GPU)      │
└───────────────┘     └───────────────┘     └───────────────┘
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  PostgreSQL   │     │    Milvus     │     │  Model Store  │
└───────────────┘     └───────────────┘     └───────────────┘
```

---

## 目录结构

```
kubernetes/
├── base/                           # 基础配置
│   ├── namespace.yaml              # 命名空间、RBAC、网络策略
│   ├── configmap.yaml              # 应用配置
│   ├── secret.yaml                 # 敏感配置
│   ├── deployment.yaml             # 部署配置
│   ├── service.yaml                # 服务定义
│   ├── hpa.yaml                    # 自动伸缩
│   ├── pdb.yaml                    # Pod 干扰预算
│   ├── pvc.yaml                    # 持久化存储
│   ├── kustomization.yaml          # Kustomize 配置
│   └── config/                     # 配置转换器
│       ├── label-transformer.yaml
│       └── annotation-transformer.yaml
│
├── overlays/                       # 环境覆盖配置
│   ├── dev/                        # 开发环境
│   │   ├── kustomization.yaml
│   │   ├── patches/
│   │   │   ├── deployment-resources.yaml
│   │   │   ├── hpa-scaling.yaml
│   │   │   └── configmap-dev.yaml
│   │   └── resources/
│   │       └── debug-tools.yaml
│   │
│   └── prod/                       # 生产环境
│       ├── kustomization.yaml
│       ├── canary.yaml             # 金丝雀发布配置
│       ├── patches/
│       │   ├── deployment-resources.yaml
│       │   ├── deployment-replicas.yaml
│       │   ├── hpa-scaling.yaml
│       │   ├── configmap-prod.yaml
│       │   └── security-context.yaml
│       └── resources/
│           ├── priority-class.yaml
│           └── network-policy-prod.yaml
│
├── istio/                          # Istio 配置
│   ├── gateway.yaml                # 入口网关
│   ├── virtual-service.yaml        # 虚拟服务路由
│   ├── destination-rule.yaml       # 目标规则、熔断
│   └── authorization-policy.yaml   # 授权策略
│
├── ingress/                        # Ingress 配置
│   ├── ingress.yaml                # Ingress 规则
│   └── certificate.yaml            # TLS 证书
│
└── tests/                          # 测试文件
    ├── validate.sh                 # Linux 验证脚本
    ├── validate.ps1                # Windows 验证脚本
    └── conftest/                   # OPA 策略测试
        └── policy/
            ├── deployment.rego
            ├── service.rego
            ├── hpa.rego
            └── security.rego
```

---

## 快速开始

### 前置条件

- Kubernetes 集群 (v1.25+)
- kubectl 已配置
- kustomize (v4.0+) 或 kubectl 内置 kustomize
- (可选) Istio (v1.18+)
- (可选) cert-manager (v1.12+)

### 部署到开发环境

```bash
# 预览生成的配置
kubectl kustomize overlays/dev

# 部署到开发环境
kubectl apply -k overlays/dev

# 验证部署状态
kubectl get pods -n recommend-dev
kubectl get svc -n recommend-dev
```

### 部署到生产环境

```bash
# 预览生成的配置
kubectl kustomize overlays/prod

# 部署到生产环境
kubectl apply -k overlays/prod

# 验证部署状态
kubectl get pods -n recommend-prod
kubectl get svc -n recommend-prod
```

### 部署 Istio 配置

```bash
# 确保 Istio 已安装
istioctl verify-install

# 部署 Istio 资源
kubectl apply -f istio/

# 验证
kubectl get gateway,virtualservice,destinationrule -n recommend-prod
```

---

## 基础配置详解

### 服务端口契约

根据 `interfaces.yaml` 定义的端口契约：

| 服务 | HTTP 端口 | gRPC 端口 | 指标端口 |
|------|-----------|-----------|----------|
| recommend-service | 8080 | 9090 | 9091 |
| user-service | 8081 | 9091 | 9092 |
| item-service | 8082 | 9092 | 9093 |
| ugt-inference | - | 50051 | 9094 |

### 资源配置

#### 开发环境资源限制

| 服务 | CPU 请求 | CPU 限制 | 内存请求 | 内存限制 |
|------|----------|----------|----------|----------|
| recommend-service | 50m | 500m | 128Mi | 512Mi |
| user-service | 50m | 300m | 128Mi | 256Mi |
| item-service | 50m | 300m | 128Mi | 256Mi |
| ugt-inference | 500m | 2000m | 2Gi | 8Gi |

#### 生产环境资源限制

| 服务 | CPU 请求 | CPU 限制 | 内存请求 | 内存限制 |
|------|----------|----------|----------|----------|
| recommend-service | 500m | 2000m | 1Gi | 4Gi |
| user-service | 200m | 1000m | 512Mi | 1Gi |
| item-service | 200m | 1000m | 512Mi | 2Gi |
| ugt-inference | 4000m | 8000m | 16Gi | 32Gi |

### 健康检查配置

所有服务都配置了三种探针：

```yaml
# 存活探针 - 检查服务是否存活
livenessProbe:
  httpGet:
    path: /health/live
    port: http
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 3

# 就绪探针 - 检查服务是否就绪
readinessProbe:
  httpGet:
    path: /health/ready
    port: http
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 3

# 启动探针 - 处理慢启动服务
startupProbe:
  httpGet:
    path: /health/live
    port: http
  initialDelaySeconds: 10
  periodSeconds: 5
  failureThreshold: 30
```

---

## 环境配置

### 开发环境特性

- **低资源配置**: 降低资源请求以节省成本
- **单副本部署**: 减少资源占用
- **详细日志**: 启用 debug 级别日志
- **全采样追踪**: 100% 采样率
- **调试工具**: 包含 Redis Commander、pgAdmin 等

### 生产环境特性

- **高资源配置**: 确保性能和稳定性
- **多副本部署**: 保证高可用
- **安全加固**: 严格的安全上下文和网络策略
- **优先级调度**: 使用 PriorityClass 确保关键服务
- **金丝雀发布**: 支持灰度发布

### 配置覆盖示例

```yaml
# overlays/prod/patches/configmap-prod.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  LOG_LEVEL: "info"           # 生产环境使用 info 级别
  TRACING_SAMPLE_RATE: "0.01" # 1% 采样率
  RATE_LIMIT_RPS: "10000"     # 更高的限流阈值
```

---

## Istio Service Mesh

### 流量管理

#### 金丝雀发布

```yaml
# 基于请求头的金丝雀路由
http:
  - match:
      - headers:
          x-canary:
            exact: "true"
    route:
      - destination:
          host: recommend-service
          subset: canary
```

#### 基于权重的灰度发布

```yaml
# 95/5 流量分配
- route:
    - destination:
        host: recommend-service
        subset: stable
      weight: 95
    - destination:
        host: recommend-service
        subset: canary
      weight: 5
```

### 熔断配置

```yaml
trafficPolicy:
  outlierDetection:
    consecutive5xxErrors: 5    # 连续 5 个 5xx 错误
    interval: 10s              # 检测间隔
    baseEjectionTime: 30s      # 基础驱逐时间
    maxEjectionPercent: 50     # 最大驱逐比例
```

### 连接池配置

```yaml
connectionPool:
  tcp:
    maxConnections: 100        # 最大 TCP 连接数
    connectTimeout: 5s
  http:
    http2MaxRequests: 1000     # 最大 HTTP/2 请求数
    maxRetries: 3              # 最大重试次数
```

---

## Ingress 配置

### NGINX Ingress

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/limit-rps: "1000"
    cert-manager.io/cluster-issuer: letsencrypt-prod
```

### TLS 证书管理

使用 cert-manager 自动管理 Let's Encrypt 证书：

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: recommend-api-cert
spec:
  secretName: recommend-api-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - api.recommend.example.com
  duration: 2160h      # 90 天
  renewBefore: 360h    # 提前 15 天续期
```

---

## 安全配置

### 安全上下文

```yaml
securityContext:
  runAsNonRoot: true          # 禁止 root 用户
  runAsUser: 1000             # 指定用户 ID
  readOnlyRootFilesystem: true # 只读根文件系统
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL                   # 删除所有 capabilities
```

### 网络策略

生产环境实施严格的网络隔离：

```yaml
# 默认拒绝所有入站流量
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
spec:
  podSelector: {}
  policyTypes:
    - Ingress
```

### RBAC 配置

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: recommend-role
rules:
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: ["get", "watch", "list"]
```

---

## 监控与可观测性

### Prometheus 指标

所有服务都暴露 Prometheus 指标：

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9091"
  prometheus.io/path: "/metrics"
```

### 关键指标

| 指标名称 | 描述 |
|----------|------|
| http_requests_total | HTTP 请求总数 |
| http_request_duration_seconds | HTTP 请求延迟 |
| grpc_requests_total | gRPC 请求总数 |
| inference_latency_seconds | 推理延迟 |
| cache_hit_ratio | 缓存命中率 |

### HPA 自定义指标

```yaml
metrics:
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
```

---

## 故障排除

### 常见问题

#### 1. Pod 无法启动

```bash
# 查看 Pod 状态
kubectl describe pod <pod-name> -n recommend-prod

# 查看容器日志
kubectl logs <pod-name> -n recommend-prod

# 检查事件
kubectl get events -n recommend-prod --sort-by='.lastTimestamp'
```

#### 2. 服务无法访问

```bash
# 检查 Service 端点
kubectl get endpoints <service-name> -n recommend-prod

# 测试服务连通性
kubectl run test-pod --rm -it --image=busybox -- wget -qO- http://recommend-service:8080/health
```

#### 3. HPA 不生效

```bash
# 查看 HPA 状态
kubectl get hpa -n recommend-prod

# 查看详细信息
kubectl describe hpa recommend-service-hpa -n recommend-prod

# 检查 metrics-server
kubectl top pods -n recommend-prod
```

#### 4. Istio 流量问题

```bash
# 检查 Istio 配置
istioctl analyze -n recommend-prod

# 查看 Envoy 代理配置
istioctl proxy-config routes <pod-name> -n recommend-prod

# 查看代理状态
istioctl proxy-status
```

### 日志聚合

```bash
# 查看所有服务日志
kubectl logs -l app.kubernetes.io/part-of=generative-recsys -n recommend-prod --all-containers

# 实时跟踪日志
kubectl logs -f -l app=recommend-service -n recommend-prod
```

---

## 最佳实践

### 1. 资源管理

- ✅ 始终设置资源请求和限制
- ✅ 使用 LimitRange 设置默认值
- ✅ 使用 ResourceQuota 限制命名空间资源

### 2. 高可用

- ✅ 配置 PodDisruptionBudget
- ✅ 使用反亲和性分布 Pod
- ✅ 跨可用区部署
- ✅ 设置合理的副本数

### 3. 安全

- ✅ 使用非 root 用户运行容器
- ✅ 只读根文件系统
- ✅ 删除不必要的 capabilities
- ✅ 使用网络策略隔离流量
- ✅ 加密传输 (mTLS)

### 4. 可观测性

- ✅ 配置健康检查探针
- ✅ 暴露 Prometheus 指标
- ✅ 集成链路追踪
- ✅ 结构化日志输出

### 5. 发布策略

- ✅ 使用滚动更新
- ✅ 配置 maxSurge 和 maxUnavailable
- ✅ 使用金丝雀发布验证新版本
- ✅ 设置合理的 terminationGracePeriodSeconds

---

## 测试

### 运行验证测试

**Linux/macOS:**

```bash
./tests/validate.sh
```

**Windows:**

```powershell
.\tests\validate.ps1
```

### 使用 Conftest 进行策略测试

```bash
# 安装 conftest
brew install conftest  # macOS
# 或
scoop install conftest  # Windows

# 运行策略测试
conftest test base/*.yaml -p tests/conftest/policy/
```

### 使用 kubeconform 验证

```bash
# 安装 kubeconform
brew install kubeconform

# 验证配置
kubeconform -strict base/*.yaml
kustomize build overlays/dev | kubeconform -strict
kustomize build overlays/prod | kubeconform -strict
```

---

## 参考资料

- [Kubernetes 官方文档](https://kubernetes.io/docs/)
- [Kustomize 文档](https://kustomize.io/)
- [Istio 文档](https://istio.io/latest/docs/)
- [cert-manager 文档](https://cert-manager.io/docs/)
- [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)

---

## 变更日志

### v1.0.0 (2025-01-05)

- 初始版本
- 完成 base 配置
- 完成 dev/prod overlays
- 完成 Istio 配置
- 完成 Ingress 配置
- 添加验证测试

---

## 贡献指南

1. 修改配置前先运行验证测试
2. 使用 `kubectl diff` 预览变更
3. 遵循 Kubernetes 命名规范
4. 更新相关文档

```bash
# 预览变更
kubectl diff -k overlays/prod

# 验证配置
./tests/validate.sh
```

---

## 联系方式

如有问题，请联系 DevOps 团队或提交 Issue。

