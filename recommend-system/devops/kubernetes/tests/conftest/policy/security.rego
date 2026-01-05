# OPA/Conftest 策略 - 安全验证

package main

# 拒绝特权容器
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    container.securityContext.privileged == true
    msg := sprintf("容器 '%s' 在 Deployment '%s' 中以特权模式运行", [container.name, input.metadata.name])
}

# 拒绝允许特权升级的容器
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    container.securityContext.allowPrivilegeEscalation == true
    msg := sprintf("容器 '%s' 在 Deployment '%s' 中允许特权升级", [container.name, input.metadata.name])
}

# 警告没有只读根文件系统的容器
warn[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.readOnlyRootFilesystem
    msg := sprintf("容器 '%s' 在 Deployment '%s' 中没有设置只读根文件系统", [container.name, input.metadata.name])
}

# 警告没有删除所有 capabilities 的容器
warn[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.capabilities.drop
    msg := sprintf("容器 '%s' 在 Deployment '%s' 中没有删除 capabilities", [container.name, input.metadata.name])
}

# 拒绝添加危险 capabilities 的容器
dangerous_capabilities := {"NET_ADMIN", "SYS_ADMIN", "SYS_PTRACE", "SYS_MODULE"}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    cap := container.securityContext.capabilities.add[_]
    dangerous_capabilities[cap]
    msg := sprintf("容器 '%s' 在 Deployment '%s' 中添加了危险 capability: %s", [container.name, input.metadata.name, cap])
}

# 拒绝挂载 hostPath 的容器 (除非明确允许)
deny[msg] {
    input.kind == "Deployment"
    volume := input.spec.template.spec.volumes[_]
    volume.hostPath
    msg := sprintf("Deployment '%s' 挂载了 hostPath: %s", [input.metadata.name, volume.name])
}

# 拒绝使用 hostNetwork 的 Pod
deny[msg] {
    input.kind == "Deployment"
    input.spec.template.spec.hostNetwork == true
    msg := sprintf("Deployment '%s' 使用 hostNetwork", [input.metadata.name])
}

# 拒绝使用 hostPID 的 Pod
deny[msg] {
    input.kind == "Deployment"
    input.spec.template.spec.hostPID == true
    msg := sprintf("Deployment '%s' 使用 hostPID", [input.metadata.name])
}

# 拒绝使用 hostIPC 的 Pod
deny[msg] {
    input.kind == "Deployment"
    input.spec.template.spec.hostIPC == true
    msg := sprintf("Deployment '%s' 使用 hostIPC", [input.metadata.name])
}

# 警告没有 ServiceAccount 的 Deployment
warn[msg] {
    input.kind == "Deployment"
    not input.spec.template.spec.serviceAccountName
    msg := sprintf("Deployment '%s' 没有指定 ServiceAccount", [input.metadata.name])
}

# 警告使用默认 ServiceAccount 的 Deployment
warn[msg] {
    input.kind == "Deployment"
    input.spec.template.spec.serviceAccountName == "default"
    msg := sprintf("Deployment '%s' 使用默认 ServiceAccount", [input.metadata.name])
}

