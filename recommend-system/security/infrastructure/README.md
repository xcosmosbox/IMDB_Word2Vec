# Infrastructure Security

## 概述
基础设施安全模块负责 Kubernetes 集群和 Istio 服务网格的安全配置，实现“零信任”架构。

## 目录结构
- `network-policy/`: K8s 网络策略 (NetworkPolicy)
    - `default-deny.yaml`: 默认拒绝所有 Ingress/Egress
- `istio-security/`: Istio 安全配置
    - `peer-authentication.yaml`: 强制开启 mTLS (STRICT)
    - `authorization-policy.yaml`: 基于身份的访问控制
- `cert-manager/`: 证书管理配置
- `container-security/`: 容器运行时安全
    - `kyverno-policies/`: Kyverno 准入控制策略

## 关键策略

### 1. 零信任网络
- **mTLS**: 所有服务间通信必须经过 Istio Sidecar 代理，并使用 mTLS 加密。
- **PeerAuthentication**: 设置为 STRICT 模式。

### 2. 网络隔离
- **Default Deny**: 所有 Pod 默认拒绝入站和出站流量。
- **Allowlist**: 仅显式允许的流量才能通过（通过 NetworkPolicy 配置）。

### 3. 容器安全
- **Kyverno**: 禁止特权容器 (Privileged Mode)。
- **Rootless**: 强制容器以非 Root 用户运行（策略待添加）。

## 部署
```bash
kubectl apply -f network-policy/
kubectl apply -f istio-security/
kubectl apply -f container-security/kyverno-policies/
```

