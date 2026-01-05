# OPA/Conftest 策略 - HPA 验证

package main

# 拒绝 minReplicas 小于 1 的 HPA
deny[msg] {
    input.kind == "HorizontalPodAutoscaler"
    input.spec.minReplicas < 1
    msg := sprintf("HPA '%s' 的 minReplicas 不能小于 1", [input.metadata.name])
}

# 警告 minReplicas 等于 1 的 HPA (可能影响高可用)
warn[msg] {
    input.kind == "HorizontalPodAutoscaler"
    input.spec.minReplicas == 1
    msg := sprintf("HPA '%s' 的 minReplicas 为 1，可能影响高可用性", [input.metadata.name])
}

# 拒绝 maxReplicas 小于 minReplicas 的 HPA
deny[msg] {
    input.kind == "HorizontalPodAutoscaler"
    input.spec.maxReplicas < input.spec.minReplicas
    msg := sprintf("HPA '%s' 的 maxReplicas 不能小于 minReplicas", [input.metadata.name])
}

# 警告 maxReplicas 过大的 HPA
warn[msg] {
    input.kind == "HorizontalPodAutoscaler"
    input.spec.maxReplicas > 100
    msg := sprintf("HPA '%s' 的 maxReplicas 过大: %d", [input.metadata.name, input.spec.maxReplicas])
}

# 警告没有配置行为策略的 HPA
warn[msg] {
    input.kind == "HorizontalPodAutoscaler"
    not input.spec.behavior
    msg := sprintf("HPA '%s' 没有配置 behavior 策略", [input.metadata.name])
}

# 警告缩容太激进的 HPA
warn[msg] {
    input.kind == "HorizontalPodAutoscaler"
    input.spec.behavior.scaleDown.stabilizationWindowSeconds < 60
    msg := sprintf("HPA '%s' 的缩容稳定窗口过短", [input.metadata.name])
}

# 验证至少有一个指标
deny[msg] {
    input.kind == "HorizontalPodAutoscaler"
    not input.spec.metrics
    msg := sprintf("HPA '%s' 没有配置任何指标", [input.metadata.name])
}

# 验证指标不为空
deny[msg] {
    input.kind == "HorizontalPodAutoscaler"
    count(input.spec.metrics) == 0
    msg := sprintf("HPA '%s' 的指标列表为空", [input.metadata.name])
}

