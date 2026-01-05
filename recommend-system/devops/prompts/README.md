# DevOps 开发任务分配

## 概述

本目录包含 DevOps/运维部署层开发的 6 个独立任务，每个任务由一位工程师独立完成。

---

## ⚠️ 重要：接口驱动开发

所有开发者必须先阅读接口定义文件：

```
devops/interfaces.yaml
```

定义了各模块之间的契约：
- CI/CD workflows、artifacts、secrets
- Kubernetes 服务端口、资源配额
- Prometheus 指标、告警规则
- 日志格式、标签
- 数据库迁移、备份策略
- 性能测试场景、SLA

---

## 任务分配

| 角色 | 负责模块 | 提示词文件 | 技术重点 |
|------|----------|------------|----------|
| **Person A** | CI/CD 流水线 | `person_a_cicd.md` | GitHub Actions, 金丝雀发布 |
| **Person B** | Kubernetes 配置 | `person_b_kubernetes.md` | Kustomize, Istio, HPA |
| **Person C** | 监控告警系统 | `person_c_monitoring.md` | Prometheus, Grafana, AlertManager |
| **Person D** | 日志系统 | `person_d_logging.md` | Loki, Promtail, Fluentd |
| **Person E** | 数据库管理 | `person_e_database.md` | Flyway, Milvus, 备份恢复 |
| **Person F** | 性能测试 | `person_f_testing.md` | K6, Locust, 报告生成 |

---

## 项目结构

```
devops/
├── interfaces.yaml              # 接口定义（核心）
├── ci-cd/
│   ├── .github/workflows/       # GitHub Actions (Person A)
│   ├── scripts/                 # 构建部署脚本
│   └── Makefile
├── kubernetes/
│   ├── base/                    # Kustomize base (Person B)
│   ├── overlays/
│   │   ├── dev/
│   │   └── prod/
│   ├── istio/                   # Service Mesh
│   └── ingress/
├── monitoring/
│   ├── prometheus/              # Prometheus (Person C)
│   │   ├── rules/
│   │   └── scrape-configs/
│   ├── grafana/
│   │   └── dashboards/
│   └── alertmanager/
├── logging/
│   ├── loki/                    # Loki (Person D)
│   ├── promtail/
│   └── fluentd/
├── database/
│   ├── migrations/              # Flyway SQL (Person E)
│   ├── backup/
│   └── milvus/
├── testing/
│   ├── load/                    # 负载测试 (Person F)
│   │   ├── k6/
│   │   └── locust/
│   ├── benchmark/
│   └── reports/
└── prompts/                     # 提示词文件（本目录）
```

---

## 技术栈

| 模块 | 工具 | 版本 |
|------|------|------|
| **CI/CD** | GitHub Actions | - |
| **容器编排** | Kubernetes | 1.28+ |
| **Service Mesh** | Istio | 1.20+ |
| **监控** | Prometheus | 2.x |
| **可视化** | Grafana | 10.x |
| **告警** | AlertManager | 0.26+ |
| **日志收集** | Loki + Promtail | 2.9+ |
| **日志处理** | Fluentd | 1.16+ |
| **数据库迁移** | Flyway | 9.x |
| **向量数据库** | Milvus | 2.3+ |
| **负载测试** | K6, Locust | - |

---

## 依赖关系

```
                     interfaces.yaml
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
    ▼                      ▼                      ▼
Person A              Person B              Person C
(CI/CD)               (K8s)                (监控)
    │                      │                      │
    └──────────┬───────────┘                      │
               │                                  │
               ▼                                  │
          Person D ◄──────────────────────────────┘
          (日志)
               │
    ┌──────────┴──────────┐
    ▼                     ▼
Person E              Person F
(数据库)              (测试)
```

**建议开发顺序：**
1. Person A + B 并行（CI/CD 和 K8s 基础）
2. Person C + D 并行（监控和日志）
3. Person E + F 并行（数据库和测试）
4. 集成测试

---

## 输出产物

| 角色 | 输出文件 |
|------|----------|
| **Person A** | `.github/workflows/*.yaml`, `Makefile` |
| **Person B** | `kubernetes/**/*.yaml` |
| **Person C** | `prometheus.yaml`, `alerting-rules.yaml`, `*.json` dashboards |
| **Person D** | `loki-config.yaml`, `promtail-config.yaml`, `fluent.conf` |
| **Person E** | `V*__.sql`, `backup.sh`, `restore.sh`, `collections.py` |
| **Person F** | `*.js` (K6), `locustfile.py`, `generate-report.py` |

---

## 注意事项

1. **接口优先** - 所有模块遵循 `interfaces.yaml` 定义的契约
2. **环境隔离** - dev/prod 配置分离
3. **安全性** - 密钥使用 K8s Secret 管理
4. **可观测性** - 日志、指标、追踪三件套
5. **自动化** - 一切皆代码，可重复部署

