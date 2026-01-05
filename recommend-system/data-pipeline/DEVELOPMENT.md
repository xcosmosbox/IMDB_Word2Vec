# 数据管道层开发指南

本文档旨在帮助新成员快速上手生成式推荐系统数据管道层的开发。本项目采用**接口驱动开发**模式，严格遵循分层架构设计。

## 核心架构

数据管道层由 6 个独立模块组成，通过 `interfaces.py` 定义的契约进行交互：

1.  **Collectors (Person A)**: 负责从 Kafka 和 API 采集原始事件。
2.  **ETL (Person B)**: 负责数据的抽取、转换和加载。
3.  **Feature Engineering (Person C)**: 负责特征转换和管道构建。
4.  **Feature Store (Person D)**: 负责特征的在线服务和离线存储同步。
5.  **Data Quality (Person E)**: 负责数据质量验证、监控和剖析。
6.  **Data Governance (Person F)**: 负责元数据管理、血缘追踪和访问控制。

## 目录结构

```
data-pipeline/
├── interfaces.py                 # 核心接口定义（勿动）
├── collectors/                   # 数据采集
│   ├── kafka/                    # Kafka 消费者/生产者
│   ├── api/                      # HTTP API
│   └── validators/               # 事件验证
├── etl/                          # ETL 流程
│   ├── extractors/               # 数据抽取
│   ├── transformers/             # 数据转换
│   ├── loaders/                  # 数据加载
│   └── pipeline/                 # 批处理管道
├── feature-engineering/          # 特征工程
│   ├── transformers/             # 特征转换算子
│   └── pipelines/                # 特征组合管道
├── feature-store/                # 特征存储
│   ├── online/                   # Redis 在线存储
│   ├── offline/                  # Parquet 离线存储
│   └── sync/                     # 同步作业
├── data-quality/                 # 数据质量
│   ├── validators/               # 规则验证
│   ├── monitors/                 # 质量监控
│   └── profilers/                # 数据剖析
├── data-governance/              # 数据治理
│   ├── catalog/                  # 数据目录
│   ├── lineage/                  # 血缘追踪
│   └── access/                   # 访问控制
└── tests/                        # 单元测试 (分布在各模块内)
```

## 开发流程

1.  **阅读接口**: 在开发任何功能前，首先阅读 `interfaces.py`，确认你的模块需要实现哪些接口。
2.  **实现功能**: 在对应模块的目录下创建或修改文件。尽量避免跨模块的直接依赖，而是通过接口交互。
3.  **编写测试**: 每个主要类都应有对应的单元测试。
    *   例如：`collectors/kafka/consumer.py` 对应 `collectors/tests/test_consumer.py` (或集成在 `tests/` 中)
4.  **运行测试**: 使用 `unittest` 或 `pytest` 运行测试。

## 快速上手示例

### 1. 添加新的事件验证规则 (Person A & E)

在 `collectors/validators/event_validator.py` 或 `data-quality/validators/rule_validator.py` 中：

```python
validator.add_rule(
    "price_check", 
    lambda e: e.properties.get("price", 0) > 0, 
    "Price must be positive"
)
```

### 2. 创建新的特征转换器 (Person C)

继承 `BaseFeatureTransformer` 并实现 `fit` 和 `transform`：

```python
class MyCustomScaler(BaseFeatureTransformer):
    def transform(self, data):
        # 你的逻辑
        return data
```

### 3. 注册新数据集 (Person F)

```python
catalog = DataCatalog()
catalog.register_dataset(
    name="new_user_logs",
    schema={"uid": "string", "ts": "datetime"},
    metadata={"source": "mobile_app"}
)
```

## 注意事项

*   **Mock 依赖**: 为了保证开发环境的轻量级，许多外部依赖（如 Kafka, Redis, Postgres）在代码中提供了 Mock 实现或通过 try-import 处理。在生产环境中请确保安装相应的库。
*   **配置管理**: 尽量使用 Config 类或字典传递配置，避免硬编码。
*   **异常处理**: 捕获并记录异常，而不是让服务崩溃。使用 `logging` 模块记录日志。

