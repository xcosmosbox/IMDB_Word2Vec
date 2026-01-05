# 生成式推荐系统 (UGT) - 全栈开发与部署指南

**版本**: 1.0.0
**最后更新**: 2026-01-05
**架构师**: AI & Full-Stack Architect

---

## 📖 1. 项目概览与地图

本项目是一个万亿级工业化生成式推荐系统，采用“流批一体、云原生、大模型驱动”的架构。

### 1.1 核心目录结构

```text
recommend-system/
├── algorithm/           # [核心] 算法层：UGT 模型、训练、推理 (Python/PyTorch)
├── backend/             # [核心] 后端层：业务逻辑微服务 (Go/Gin/gRPC)
│   ├── cmd/             # 服务入口
│   ├── internal/        # 业务逻辑
│   └── pkg/             # 公共库 (DB, Logger)
├── frontend/            # [核心] 前端层：用户端与管理端 (Vue 3/TypeScript)
│   ├── user-app/        # C端 App
│   └── admin/           # B端 管理后台
├── data-pipeline/       # [数据] 管道层：采集、ETL、特征工程 (Python/Spark/Flink)
│   ├── collectors/      # 数据采集
│   ├── etl/             # 数据清洗与转换
│   └── feature-store/   # 特征存储
├── security/            # [安全] 安全层：IAM, WAF, 审计, 隐私 (Go/Rego)
├── devops/              # [运维] 运维层：K8s, CI/CD, 监控, 数据库 (Shell/Yaml)
└── deployments/         # [部署] 本地开发与生产部署配置
```

---

## 🛠 2. 环境准备 (Prerequisites)

在开始之前，请确保你的开发环境满足以下要求：

### 2.1 基础软件
- **OS**: Linux (Ubuntu 22.04+) / macOS / Windows (WSL2)
- **Docker**: >= 24.0.0 (含 Docker Compose)
- **Git**: >= 2.30.0

### 2.2 语言环境
- **Go**: >= 1.21
- **Python**: >= 3.10 (建议使用 Conda)
- **Node.js**: >= 18.0.0 (推荐使用 pnpm)
- **Java**: >= 11 (仅用于 Spark/Flink 本地调试)

### 2.3 硬件建议
- **CPU**: 8 Cores+
- **RAM**: 16GB+ (32GB 推荐)
- **GPU**: NVIDIA RTX 3060+ (用于模型训练/推理，非必须但推荐)

---

## 🚀 3. 启动步骤 (Step-by-Step)

请**严格按照以下顺序**启动系统，因为层与层之间存在依赖关系。

### 第一步：启动基础设施与数据存储层 (Infrastructure)

这是所有服务的基础。

1. **进入部署目录**:
   ```bash
   cd recommend-system/deployments/docker
   ```

2. **启动基础服务** (Postgres, Redis, Milvus, MinIO, Prometheus, Grafana):
   ```bash
   docker-compose up -d
   ```

3. **初始化数据库**:
   *注意：首次启动需要初始化 Schema。*
   ```bash
   # 执行 Flyway 迁移或直接运行初始化脚本
   docker exec -i docker-postgres-1 psql -U postgres -d recommend < ../../devops/database/migrations/V001__initial_schema.sql
   # (依次执行 V002-V005)
   ```

4. **验证**:
   - Postgres: `localhost:5432`
   - Redis: `localhost:6379`
   - Milvus: `localhost:19530`

---

### 第二步：模型准备与服务化 (Model Layer)

在启动后端之前，必须先有可用的模型服务。

1. **环境安装**:
   ```bash
   cd recommend-system/algorithm
   pip install -r requirements.txt
   ```

2. **(可选) 模型训练**:
   如果有数据，可以运行训练脚本。如果没有，可以使用随机初始化的模型进行测试。
   ```bash
   python train.py --config configs/ugt_small.yaml
   ```

3. **导出模型**:
   将 PyTorch 模型导出为 ONNX 或 TorchScript 以供推理服务使用。
   ```bash
   python export.py --checkpoint checkpoints/best_model.pth --format onnx
   ```

4. **启动推理服务 (Inference Service)**:
   这是一个 gRPC 服务，供 Go 后端调用。
   ```bash
   # 开发模式
   python serving/server.py --port 50051
   ```

---

### 第三步：启动 Go 后端服务 (Backend Layer)

后端依赖于 数据库 和 推理服务。

1. **环境准备**:
   ```bash
   cd recommend-system
   go mod download
   ```

2. **配置文件**:
   复制并修改配置（指向本地 Docker 服务）。
   ```bash
   cp config.example.yaml config.yaml
   ```

3. **启动微服务**:
   建议在不同的终端窗口分别启动。

   *   **用户服务**:
       ```bash
       go run cmd/user-service/main.go
       ```
   *   **物品服务**:
       ```bash
       go run cmd/item-service/main.go
       ```
   *   **推荐核心服务** (连接 Python 推理层):
       ```bash
       go run cmd/recommend-service/main.go
       ```

4. **验证 API**:
   ```bash
   curl http://localhost:8080/health
   ```

---

### 第四步：启动数据管道 (Data Pipeline)

为了让推荐系统“动”起来，需要数据流。

1. **环境安装**:
   ```bash
   cd recommend-system/data-pipeline
   pip install -r requirements.txt
   ```

2. **启动采集器 (Collectors)**:
   模拟用户行为数据输入。
   ```bash
   python collectors/api/collector.py
   ```

3. **运行特征工程 (Feature Engineering)**:
   处理原始数据并写入 Redis/Milvus。
   ```bash
   # 示例：运行实时特征更新作业
   python etl/flink/jobs/realtime_features.py
   ```

---

### 第五步：启动前端应用 (Frontend Layer)

最后，启动面向用户的界面。

1. **用户端 App**:
   ```bash
   cd recommend-system/frontend/user-app
   pnpm install
   pnpm dev
   # 访问: http://localhost:5173
   ```

2. **管理后台 Admin**:
   ```bash
   cd recommend-system/frontend/admin
   pnpm install
   pnpm dev
   # 访问: http://localhost:5174
   ```

---

## 🔒 4. 安全层集成 (Security Layer)

在生产环境中，必须启用安全层。

1. **IAM 服务**:
   部署在网关之前，拦截所有请求。
   ```bash
   cd recommend-system/security/iam
   go run auth-service/main.go
   ```

2. **策略引擎**:
   启动 OPA 并加载 Rego 策略。
   ```bash
   opa run -s security/iam/policy-engine/policies/
   ```

---

## 🧪 5. 测试与验证

### 单元测试
```bash
# 后端
cd recommend-system && go test ./...

# 算法
cd recommend-system/algorithm && pytest

# 前端
cd recommend-system/frontend/user-app && pnpm test
```

### 性能测试 (Load Testing)
使用 K6 进行压测（确保基础设施已启动）。
```bash
cd recommend-system/devops/testing/load
./scripts/run-tests.sh baseline
```

---

## 📦 6. 生产部署 (Deployment)

使用 Kubernetes 进行集群部署。

1. **构建镜像**:
   ```bash
   make build-all
   ```

2. **推送到仓库**:
   ```bash
   docker push registry.example.com/recommend-service:v1.0.0
   ```

3. **部署到 K8s**:
   ```bash
   kubectl apply -k recommend-system/devops/kubernetes/overlays/prod
   ```

---

## ❓ 常见问题 (FAQ)

**Q: 启动 Go 服务时报错 "connection refused" 连接数据库失败？**
A: 检查 `config.yaml` 中的 DB Host。如果在宿主机运行 Go，Host 应该是 `localhost`；如果在 Docker 容器内运行，Host 应该是 `postgres` (服务名)。

**Q: Python 推理服务显存不足 (OOM)？**
A: 在 `configs/model_config.yaml` 中减小 `batch_size`，或者启用 CPU 模式 (`device: cpu`)。

**Q: 前端请求跨域 (CORS)？**
A: 确保 Go 后端中间件已配置 CORS 允许前端域名 (`localhost:5173`)。

---

**祝开发顺利！Happy Coding!**

