# 数据管道层开发任务分配

## 概述

本目录包含数据管道层开发的 6 个独立任务，每个任务由一位工程师独立完成。

---

## ⚠️ 重要：接口驱动开发

所有开发者必须先阅读接口定义文件：

```
data-pipeline/interfaces.py
```

定义了各模块之间的契约：
- 事件采集接口 (`EventCollectorInterface`, `EventPublisherInterface`)
- ETL 接口 (`ExtractorInterface`, `TransformerInterface`, `LoaderInterface`)
- 特征工程接口 (`FeatureTransformerInterface`, `FeaturePipelineInterface`)
- 特征存储接口 (`OnlineFeatureStoreInterface`, `OfflineFeatureStoreInterface`)
- 数据质量接口 (`DataValidatorInterface`, `DataQualityMonitorInterface`)
- 数据治理接口 (`DataCatalogInterface`, `DataLineageInterface`)

---

## 任务分配

| 角色 | 负责模块 | 提示词文件 | 技术重点 |
|------|----------|------------|----------|
| **Person A** | 数据采集 | `person_a_collectors.md` | Kafka, REST API, Avro |
| **Person B** | ETL 流水线 | `person_b_etl.md` | Spark, Flink, Parquet |
| **Person C** | 特征工程 | `person_c_feature_engineering.md` | sklearn, Transformers |
| **Person D** | 特征存储 | `person_d_feature_store.md` | Redis, Parquet, Milvus |
| **Person E** | 数据质量 | `person_e_data_quality.md` | Validators, Profilers |
| **Person F** | 数据治理 | `person_f_data_governance.md` | Catalog, Lineage, RBAC |

---

## 项目结构

```
data-pipeline/
├── interfaces.py                 # 接口定义（核心）
├── collectors/
│   ├── kafka/                    # Kafka 消费者/生产者 (Person A)
│   ├── api/                      # HTTP API 收集器
│   └── validators/               # 事件验证器
├── etl/
│   ├── extractors/               # 数据抽取器 (Person B)
│   ├── transformers/             # 数据转换器
│   ├── loaders/                  # 数据加载器
│   ├── spark/                    # Spark 作业
│   └── flink/                    # Flink 作业
├── feature-engineering/
│   ├── transformers/             # 特征转换器 (Person C)
│   ├── pipelines/                # 特征管道
│   └── registry/                 # 特征注册
├── feature-store/
│   ├── online/                   # 在线特征存储 (Person D)
│   ├── offline/                  # 离线特征存储
│   ├── sync/                     # 特征同步
│   └── vector/                   # 向量存储
├── data-quality/
│   ├── validators/               # 数据验证器 (Person E)
│   ├── monitors/                 # 质量监控
│   └── profilers/                # 数据剖析
├── data-governance/
│   ├── catalog/                  # 数据目录 (Person F)
│   ├── lineage/                  # 数据血缘
│   └── access/                   # 访问控制
└── prompts/                      # 提示词文件（本目录）
```

---

## 技术栈

| 模块 | 工具 | 版本 |
|------|------|------|
| **消息队列** | Apache Kafka | 3.x |
| **批处理** | Apache Spark | 3.x |
| **流处理** | Apache Flink | 1.x |
| **在线存储** | Redis Cluster | 7.x |
| **离线存储** | Parquet / Delta Lake | - |
| **向量存储** | Milvus | 2.x |
| **特征工程** | scikit-learn | - |
| **数据验证** | Great Expectations | - |
| **图数据库** | Neo4j | 5.x |

---

## 数据流架构

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                     数据管道层                               │
                    └─────────────────────────────────────────────────────────────┘
                                              │
    ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
    │                                         │                                         │
    ▼                                         ▼                                         ▼
┌─────────────┐                       ┌─────────────┐                       ┌─────────────┐
│  Collectors │                       │     ETL     │                       │   Feature   │
│  (Person A) │──────────────────────►│  (Person B) │──────────────────────►│  Engineering│
│  Kafka/API  │                       │ Spark/Flink │                       │  (Person C) │
└─────────────┘                       └─────────────┘                       └─────────────┘
                                              │                                         │
                                              │                                         │
                                              ▼                                         ▼
                                      ┌─────────────┐                       ┌─────────────┐
                                      │   Feature   │                       │    Data     │
                                      │    Store    │◄──────────────────────│   Quality   │
                                      │  (Person D) │                       │  (Person E) │
                                      └─────────────┘                       └─────────────┘
                                              │
                                              ▼
                                      ┌─────────────┐
                                      │    Data     │
                                      │ Governance  │
                                      │  (Person F) │
                                      └─────────────┘
```

---

## 依赖关系

```
Person A (Collectors) ─────► Person B (ETL) ─────► Person C (Feature Eng)
                                   │                        │
                                   ▼                        ▼
                           Person D (Feature Store) ◄───────┘
                                   │
                                   ▼
                           Person E (Data Quality)
                                   │
                                   ▼
                           Person F (Governance)
```

**建议开发顺序：**
1. Person A (Collectors) - 数据入口
2. Person B (ETL) + Person C (Feature Eng) 并行
3. Person D (Feature Store) - 依赖 B, C
4. Person E (Data Quality) - 可与 D 并行
5. Person F (Governance) - 最后集成

---

## 输出产物

| 角色 | 输出文件 |
|------|----------|
| **Person A** | `kafka/consumer.py`, `kafka/producer.py`, `api/collector.py` |
| **Person B** | `extractors/*.py`, `transformers/*.py`, `loaders/*.py` |
| **Person C** | `transformers/*.py`, `pipelines/*.py`, `registry/*.py` |
| **Person D** | `online/*.py`, `offline/*.py`, `sync/*.py`, `vector/*.py` |
| **Person E** | `validators/*.py`, `monitors/*.py`, `profilers/*.py` |
| **Person F** | `catalog/*.py`, `lineage/*.py`, `access/*.py` |

---

## 注意事项

1. **接口优先** - 所有模块遵循 `interfaces.py` 定义的契约
2. **可测试性** - 每个模块独立可测试
3. **性能优先** - 在线模块延迟 < 10ms
4. **可观测性** - Prometheus 指标 + 结构化日志
5. **容错性** - 支持重试、幂等、断点续传

