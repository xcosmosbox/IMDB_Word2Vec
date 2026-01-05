# Person B: 基础设施与网络安全

## 你的角色
你是一名资深云安全工程师，负责实现生成式推荐系统的 **基础设施与网络安全**。你的目标是构建一个“零信任”网络环境。

---

## ⚠️ 重要：标准驱动开发

**开始编码前，必须先阅读安全标准契约：**

```
security/SECURITY_STANDARDS.md
```

你需要遵循的标准：
- **零信任**: 服务间通信 mTLS (Istio)
- **网络隔离**: 默认 Deny All, 显式 Allow
- **容器安全**: 非特权容器, 镜像扫描

---

## 你的任务

```
security/infrastructure/
├── network-policy/         # K8s NetworkPolicies
│   ├── default-deny.yaml
│   ├── allow-dns.yaml
│   ├── allow-monitoring.yaml
│   └── service-rules/
│       ├── recommend-api.yaml
│       └── db-access.yaml
├── istio-security/         # Istio 安全配置
│   ├── peer-authentication.yaml
│   ├── authorization-policy.yaml
│   └── gateway-tls.yaml
├── cert-manager/           # 证书管理
│   ├── cluster-issuer.yaml
│   └── certificates.yaml
└── container-security/     # 容器安全策略
    ├── seccomp-profile.json
    └── kyverno-policies/
        ├── disallow-privileged.yaml
        └── require-run-as-nonroot.yaml
```

---

## 1. 默认拒绝策略 (network-policy/default-deny.yaml)

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: recommend-prod
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

## 2. Istio mTLS 配置 (istio-security/peer-authentication.yaml)

```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: recommend-prod
spec:
  mtls:
    mode: STRICT
```

## 3. Istio 授权策略 (istio-security/authorization-policy.yaml)

```yaml
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: recommend-service-policy
  namespace: recommend-prod
spec:
  selector:
    matchLabels:
      app: recommend-service
  action: ALLOW
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/recommend-prod/sa/api-gateway"]
    to:
    - operation:
        methods: ["POST"]
        paths: ["/api/v1/recommend"]
```

## 4. Kyverno 策略 (container-security/kyverno-policies/disallow-privileged.yaml)

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: disallow-privileged-containers
spec:
  validationFailureAction: enforce
  background: true
  rules:
  - name: validate-privileged
    match:
      resources:
        kinds:
        - Pod
    validate:
      message: "Privileged mode is not allowed."
      pattern:
        spec:
          containers:
          - =(securityContext):
              =(privileged): false
```

## 输出要求

请输出完整的安全基础设施配置：
1. 关键的 NetworkPolicy YAML
2. Istio mTLS 和 AuthorizationPolicy 配置
3. Cert-manager 证书配置
4. Kyverno 准入控制策略

