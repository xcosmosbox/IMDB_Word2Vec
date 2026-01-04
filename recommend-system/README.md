# 生成式推荐系统 (Generative Recommendation System)

基于 Transformer 架构的生成式推荐系统后端服务，支持万亿级数据规模、低延迟、高并发。

## 🚀 技术特性

- **生成式推荐**: 基于 UGT (Unified Generative Transformer) 架构
- **语义 ID**: 层次化语义编码，支持高效召回
- **多模态支持**: 电影、视频、商品、文章等多种物品类型
- **向量检索**: Milvus 向量数据库，毫秒级相似度搜索
- **冷启动解决**: 结合 LLM 语义先验和热门推荐
- **高性能**: 多级缓存、自适应限流、分布式部署

## 📁 项目结构

```
recommend-system/
├── cmd/                          # 服务入口
│   └── recommend-service/        # 推荐服务主程序
├── internal/                     # 内部包
│   ├── cache/                    # 多级缓存
│   ├── inference/                # 模型推理客户端
│   ├── middleware/               # HTTP 中间件
│   ├── model/                    # 数据模型
│   ├── repository/               # 数据访问层
│   └── service/                  # 业务逻辑
│       └── recommend/            # 推荐服务
├── pkg/                          # 公共包
│   ├── config/                   # 配置管理
│   ├── database/                 # 数据库连接
│   ├── logger/                   # 日志管理
│   └── utils/                    # 工具函数
├── api/                          # API 定义
│   └── recommend/v1/             # 推荐 API v1
├── configs/                      # 配置文件
├── deployments/                  # 部署配置
│   ├── docker/                   # Docker 配置
│   └── kubernetes/               # Kubernetes 配置
├── scripts/                      # 脚本
└── tests/                        # 测试
```

## 🛠️ 技术栈

| 层级 | 技术选型 |
|------|---------|
| **语言** | Go 1.22 |
| **Web 框架** | Gin |
| **数据库** | PostgreSQL + pgvector |
| **缓存** | Redis Cluster |
| **向量数据库** | Milvus |
| **消息队列** | Kafka |
| **推理服务** | Triton Inference Server |
| **监控** | Prometheus + Grafana |
| **追踪** | Jaeger |
| **容器编排** | Kubernetes |

## 🚀 快速开始

### 环境要求

- Go 1.22+
- Docker & Docker Compose
- PostgreSQL 16+ (with pgvector)
- Redis 7+
- Milvus 2.4+

### 本地开发

```bash
# 克隆项目
git clone <repository-url>
cd recommend-system

# 安装依赖
go mod download

# 启动依赖服务
make compose-up

# 初始化数据库
make init-db

# 运行服务
make run
```

### Docker 部署

```bash
# 构建镜像
make docker-build

# 启动所有服务
make compose-up

# 查看日志
make compose-logs
```

### Kubernetes 部署

```bash
# 创建命名空间
kubectl create namespace recommend

# 部署服务
make k8s-deploy

# 查看状态
kubectl get pods -n recommend
```

## 📡 API 接口

### 获取推荐列表

```bash
POST /api/v1/recommend
Content-Type: application/json

{
    "user_id": "user_001",
    "size": 20,
    "context": {
        "device_type": "mobile",
        "platform": "ios"
    }
}
```

### 获取相似推荐

```bash
POST /api/v1/similar
Content-Type: application/json

{
    "item_id": "item_001",
    "size": 10
}
```

### 提交反馈

```bash
POST /api/v1/feedback
Content-Type: application/json

{
    "user_id": "user_001",
    "item_id": "item_001",
    "action": "click",
    "request_id": "req_xxx"
}
```

## ⚙️ 配置说明

主配置文件: `configs/config.yaml`

```yaml
server:
  http_port: 8080
  grpc_port: 9090

database:
  host: localhost
  port: 5432
  dbname: recommend

redis:
  addrs:
    - localhost:6379

milvus:
  address: localhost
  port: 19530
```

支持环境变量覆盖: `RECOMMEND_DATABASE_HOST=xxx`

## 📊 监控

- **Prometheus**: http://localhost:9092
- **Grafana**: http://localhost:3000 (admin/admin)
- **健康检查**: http://localhost:8080/health
- **就绪检查**: http://localhost:8080/ready
- **指标端点**: http://localhost:9091/metrics

## 🧪 测试

```bash
# 运行所有测试
make test

# 生成覆盖率报告
make test-coverage

# 代码检查
make lint
```

## 📝 开发指南

### 代码规范

- 使用 `gofmt` 格式化代码
- 遵循 [Effective Go](https://go.dev/doc/effective_go)
- 单元测试覆盖率 >= 80%

### 分支策略

- `main`: 生产分支
- `develop`: 开发分支
- `feature/*`: 功能分支
- `hotfix/*`: 热修复分支

### 提交规范

```
feat: 新功能
fix: Bug 修复
docs: 文档更新
refactor: 重构
test: 测试
chore: 其他
```

## 📄 License

MIT License

