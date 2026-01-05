# OPA/Conftest 策略 - Deployment 验证
# 使用方法: conftest test kubernetes/base/*.yaml

package main

import future.keywords.in

# 拒绝没有资源限制的容器
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.resources.limits
    msg := sprintf("容器 '%s' 在 Deployment '%s' 中缺少资源限制", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.resources.requests
    msg := sprintf("容器 '%s' 在 Deployment '%s' 中缺少资源请求", [container.name, input.metadata.name])
}

# 拒绝没有健康检查的容器
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.livenessProbe
    msg := sprintf("容器 '%s' 在 Deployment '%s' 中缺少存活探针", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.readinessProbe
    msg := sprintf("容器 '%s' 在 Deployment '%s' 中缺少就绪探针", [container.name, input.metadata.name])
}

# 拒绝以 root 用户运行的容器
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    container.securityContext.runAsUser == 0
    msg := sprintf("容器 '%s' 在 Deployment '%s' 中以 root 用户运行", [container.name, input.metadata.name])
}

# 警告没有设置 runAsNonRoot 的容器
warn[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.runAsNonRoot
    msg := sprintf("容器 '%s' 在 Deployment '%s' 中未设置 runAsNonRoot", [container.name, input.metadata.name])
}

# 拒绝使用 latest 标签的镜像
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    endswith(container.image, ":latest")
    msg := sprintf("容器 '%s' 在 Deployment '%s' 中使用 latest 标签", [container.name, input.metadata.name])
}

# 拒绝没有镜像标签的容器
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not contains(container.image, ":")
    msg := sprintf("容器 '%s' 在 Deployment '%s' 中没有指定镜像标签", [container.name, input.metadata.name])
}

# 警告没有反亲和性的 Deployment
warn[msg] {
    input.kind == "Deployment"
    input.spec.replicas > 1
    not input.spec.template.spec.affinity.podAntiAffinity
    msg := sprintf("Deployment '%s' 有多个副本但没有配置反亲和性", [input.metadata.name])
}

# 拒绝没有标签的 Deployment
deny[msg] {
    input.kind == "Deployment"
    not input.metadata.labels
    msg := sprintf("Deployment '%s' 缺少标签", [input.metadata.name])
}

# 警告没有标准化标签的资源
warn[msg] {
    input.kind == "Deployment"
    not input.metadata.labels["app.kubernetes.io/name"]
    msg := sprintf("Deployment '%s' 缺少 app.kubernetes.io/name 标签", [input.metadata.name])
}

# 拒绝 CPU 限制过高的容器 (> 8 cores)
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    cpu := container.resources.limits.cpu
    cpu_value := to_number(replace(cpu, "m", "")) / 1000
    cpu_value > 8
    msg := sprintf("容器 '%s' 的 CPU 限制过高: %s", [container.name, cpu])
}

# 拒绝内存限制过高的容器 (> 32Gi)
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    memory := container.resources.limits.memory
    endswith(memory, "Gi")
    memory_value := to_number(replace(memory, "Gi", ""))
    memory_value > 32
    msg := sprintf("容器 '%s' 的内存限制过高: %s", [container.name, memory])
}

