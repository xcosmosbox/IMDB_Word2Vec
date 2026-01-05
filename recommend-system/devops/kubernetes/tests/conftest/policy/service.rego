# OPA/Conftest 策略 - Service 验证

package main

# 拒绝没有选择器的 Service (除了 ExternalName 类型)
deny[msg] {
    input.kind == "Service"
    input.spec.type != "ExternalName"
    not input.spec.selector
    msg := sprintf("Service '%s' 缺少选择器", [input.metadata.name])
}

# 警告没有命名端口的 Service
warn[msg] {
    input.kind == "Service"
    port := input.spec.ports[_]
    not port.name
    msg := sprintf("Service '%s' 的端口 %d 没有名称", [input.metadata.name, port.port])
}

# 拒绝使用 NodePort 类型的 Service (生产环境应使用 LoadBalancer 或 ClusterIP + Ingress)
warn[msg] {
    input.kind == "Service"
    input.spec.type == "NodePort"
    msg := sprintf("Service '%s' 使用 NodePort 类型，建议使用 ClusterIP + Ingress", [input.metadata.name])
}

# 警告没有标签的 Service
warn[msg] {
    input.kind == "Service"
    not input.metadata.labels
    msg := sprintf("Service '%s' 缺少标签", [input.metadata.name])
}

# 验证端口号范围
deny[msg] {
    input.kind == "Service"
    port := input.spec.ports[_]
    port.port < 1
    msg := sprintf("Service '%s' 的端口号 %d 无效", [input.metadata.name, port.port])
}

deny[msg] {
    input.kind == "Service"
    port := input.spec.ports[_]
    port.port > 65535
    msg := sprintf("Service '%s' 的端口号 %d 超出范围", [input.metadata.name, port.port])
}

