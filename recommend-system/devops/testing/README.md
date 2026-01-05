# 性能测试套件

> 生成式推荐系统性能测试套件 - Person F 开发

本套件提供完整的性能测试能力，包括负载测试、压力测试、基准测试和性能报告生成。

## 📋 目录结构

```
devops/testing/
├── load/                          # 负载测试
│   ├── k6/                        # K6 测试脚本
│   │   ├── config.js              # 全局配置
│   │   ├── lib/                   # 辅助库
│   │   │   ├── api.js             # API 调用封装
│   │   │   └── utils.js           # 工具函数
│   │   └── scenarios/             # 测试场景
│   │       ├── baseline.js        # 基线测试
│   │       ├── stress.js          # 压力测试
│   │       └── spike.js           # 峰值测试
│   ├── locust/                    # Locust 测试脚本
│   │   ├── config.py              # 配置模块
│   │   └── locustfile.py          # 测试用例
│   └── scripts/                   # 运行脚本
│       ├── run-tests.sh           # Linux/macOS
│       └── run-tests.ps1          # Windows PowerShell
├── benchmark/                     # 基准测试
│   ├── api-benchmark.go           # API 基准测试 (Go)
│   ├── inference-benchmark.py     # 推理基准测试 (Python)
│   └── database-benchmark.sql     # 数据库基准测试 (SQL)
├── reports/                       # 报告生成
│   ├── templates/
│   │   └── report.html            # HTML 报告模板
│   └── generate-report.py         # 报告生成器
├── tests/                         # 单元测试
│   ├── conftest.py                # Pytest 配置
│   ├── test_locust_config.py      # Locust 配置测试
│   ├── test_report_generator.py   # 报告生成器测试
│   └── test_inference_benchmark.py # 推理基准测试测试
├── requirements.txt               # Python 依赖
└── README.md                      # 本文档
```

## 🎯 SLA 目标

基于 `devops/interfaces.yaml` 定义的性能契约：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 可用性 | ≥ 99.9% | 服务可用率 |
| P50 延迟 | ≤ 50ms | 中位数响应时间 |
| P99 延迟 | ≤ 200ms | 99 分位响应时间 |
| 错误率 | ≤ 0.1% | 请求失败率 |

## 🚀 快速开始

### 1. 安装依赖

```bash
# Python 依赖
pip install -r requirements.txt

# K6 安装 (macOS)
brew install k6

# K6 安装 (Windows)
choco install k6

# K6 安装 (Linux)
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6
```

### 2. 运行测试

#### 使用运行脚本

```bash
# Linux/macOS
cd devops/testing/load/scripts
chmod +x run-tests.sh

# 基线测试
./run-tests.sh baseline

# 压力测试
./run-tests.sh stress

# 峰值测试
./run-tests.sh spike

# 运行所有测试
./run-tests.sh all

# 使用 Locust
./run-tests.sh locust baseline
```

```powershell
# Windows PowerShell
cd devops\testing\load\scripts

# 基线测试
.\run-tests.ps1 -TestType baseline

# 压力测试
.\run-tests.ps1 -TestType stress

# 指定目标 URL
.\run-tests.ps1 -TestType baseline -BaseUrl http://api.example.com
```

#### 直接运行 K6

```bash
cd devops/testing/load/k6

# 基线测试
k6 run --env BASE_URL=http://localhost:8080 scenarios/baseline.js

# 压力测试
k6 run --env BASE_URL=http://localhost:8080 scenarios/stress.js

# 输出 JSON 结果
k6 run --out json=results.json scenarios/baseline.js
```

#### 直接运行 Locust

```bash
cd devops/testing/load/locust

# Web UI 模式
locust -f locustfile.py --host http://localhost:8080

# 无头模式
locust -f locustfile.py --host http://localhost:8080 \
    --headless -u 100 -r 10 -t 5m

# 分布式模式 (Master)
locust -f locustfile.py --master --host http://localhost:8080

# 分布式模式 (Worker)
locust -f locustfile.py --worker --master-host <master-ip>
```

## 📊 测试场景

### 基线测试 (Baseline)

验证系统在正常负载下的性能表现。

- **RPS**: 100
- **持续时间**: 5 分钟
- **并发用户**: 50-200 VUs
- **场景分布**:
  - 50% 推荐请求
  - 20% 搜索请求
  - 20% 反馈请求
  - 10% 物品详情

### 压力测试 (Stress)

找到系统的性能极限和瓶颈。

- **RPS**: 逐步增加到 1000
- **持续时间**: 10 分钟
- **阶段**:
  1. 预热 (1min): 50 → 100 RPS
  2. 增加 (2min): 100 → 300 RPS
  3. 中等 (2min): 300 → 500 RPS
  4. 高负载 (2min): 500 → 800 RPS
  5. 峰值 (2min): 800 → 1000 RPS
  6. 恢复 (1min): 1000 → 0 RPS

### 峰值测试 (Spike)

测试系统对突发流量的应对能力。

- **峰值 RPS**: 5000
- **持续时间**: 2 分钟
- **阶段**:
  1. 预热 (30s): 100 RPS
  2. 突发 (10s): 100 → 5000 RPS
  3. 峰值 (60s): 5000 RPS
  4. 恢复 (20s): 5000 → 100 RPS

## 🔧 基准测试

### API 基准测试 (Go)

```bash
cd devops/testing/benchmark

# 编译
go build -o api-benchmark api-benchmark.go

# 运行
./api-benchmark -url http://localhost:8080 -duration 60s -concurrency 10
```

### 推理基准测试 (Python)

```bash
cd devops/testing/benchmark

# 运行
python inference-benchmark.py \
    --host localhost \
    --http-port 8080 \
    --duration 30 \
    --concurrency 10 \
    --batch-sizes 1,8,16 \
    --seq-lengths 64,128,256
```

### 数据库基准测试 (PostgreSQL)

```bash
# 使用 psql
psql -U postgres -d recommend_db -f database-benchmark.sql

# 使用 pgbench
pgbench -U postgres -d recommend_db -f database-benchmark.sql -c 10 -j 4 -T 60
```

## 📈 报告生成

### 生成 HTML 报告

```bash
cd devops/testing/reports

# 从 K6 JSON 输出生成
python generate-report.py results/baseline.json -o report.html

# 生成多种格式
python generate-report.py results/baseline.json -f html,json,junit
```

### 报告格式

- **HTML**: 可视化交互式报告，包含图表
- **JSON**: 结构化数据，便于自动化处理
- **JUnit**: CI/CD 集成用 XML 格式

## ✅ 运行单元测试

```bash
cd devops/testing

# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_locust_config.py -v

# 生成覆盖率报告
pytest tests/ --cov=. --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html
```

## 🔧 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `BASE_URL` | 目标服务 URL | `http://localhost:8080` |
| `TEST_ENV` | 测试环境 (local/dev/prod) | `local` |
| `API_KEY` | API 认证密钥 | `test-api-key` |
| `RESULTS_DIR` | 结果输出目录 | `./results` |

### K6 配置

编辑 `load/k6/config.js` 自定义：

```javascript
// 修改 SLA 阈值
export const slaThresholds = {
  p50Latency: 50,   // ms
  p99Latency: 200,  // ms
  errorRate: 0.001, // 0.1%
};

// 修改测试数据范围
export const testData = {
  userIdRange: { min: 1, max: 100000 },
  itemIdRange: { min: 1, max: 1000000 },
};
```

### Locust 配置

编辑 `load/locust/config.py` 自定义：

```python
# 修改 SLA 配置
SLA = SLAConfig(
    availability=0.999,
    p50_latency=50,
    p99_latency=200,
    error_rate=0.001,
)

# 修改负载场景
LOAD_SCENARIOS = {
    "baseline": LoadScenario(
        name="baseline",
        users=100,
        spawn_rate=10,
        duration="5m",
        rps_target=100,
    ),
}
```

## 🔍 故障排除

### K6 无法连接目标服务

1. 检查目标服务是否运行
2. 确认 `BASE_URL` 环境变量正确
3. 检查网络连接和防火墙

### Locust 内存使用过高

1. 减少并发用户数
2. 增加请求间隔 (`wait_time`)
3. 使用分布式模式

### 报告生成失败

1. 确保安装了 `jinja2` 包
2. 检查输入 JSON 文件格式
3. 查看错误日志

## 📚 参考文档

- [K6 官方文档](https://k6.io/docs/)
- [Locust 官方文档](https://docs.locust.io/)
- [接口定义](../interfaces.yaml)
- [架构设计](../../../docs/生成式推荐系统架构设计.md)

## 🛠 扩展开发

### 添加新测试场景

1. 在 `load/k6/scenarios/` 创建新的 `.js` 文件
2. 导入配置和工具库
3. 定义 `options` 和测试函数
4. 在运行脚本中添加对应命令

### 添加新基准测试

1. 在 `benchmark/` 创建新的测试文件
2. 实现数据生成、测试执行、结果统计
3. 添加 SLA 检查逻辑
4. 更新 README 文档

### 自定义报告模板

1. 编辑 `reports/templates/report.html`
2. 使用 Jinja2 模板语法
3. 添加新的图表或数据展示

## 📝 更新日志

- **v1.0.0** (2026-01-05)
  - 初始版本
  - K6 负载测试 (baseline/stress/spike)
  - Locust 负载测试
  - 基准测试 (API/推理/数据库)
  - HTML/JSON/JUnit 报告生成
  - 完整单元测试

## 👤 作者

Person F - DevOps 工程师

## 📄 许可证

内部使用

