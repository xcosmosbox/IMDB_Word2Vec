# 推理服务模块 (Serving)

## 📋 概述

本模块负责将训练好的 UGT 生成式推荐模型导出为可部署的高性能推理服务。

### 性能目标

| 指标 | 目标值 |
|------|--------|
| P99 延迟 | < 30ms |
| GPU 利用率 | > 80% |
| 支持动态 Batching | ✅ |

### 部署架构

```
PyTorch Model → ONNX → TensorRT → Triton Inference Server
```

## 📁 目录结构

```
algorithm/serving/
├── __init__.py              # 模块导出
├── config.py                # 配置类定义
├── export_onnx.py           # ONNX 模型导出
├── optimize_trt.py          # TensorRT 优化
├── triton_config.py         # Triton 配置生成
├── benchmark.py             # 性能基准测试
├── exporter.py              # 统一导出器（实现接口）
├── model_repository/        # Triton 模型仓库
├── scripts/                 # 部署脚本
│   ├── export.sh           # 导出脚本
│   └── benchmark.sh        # 基准测试脚本
├── tests/                   # 单元测试
│   ├── test_config.py
│   ├── test_export_onnx.py
│   ├── test_optimize_trt.py
│   ├── test_triton_config.py
│   ├── test_benchmark.py
│   └── test_exporter.py
└── README.md                # 本文档
```

## 🔧 核心组件

### 1. ServingExporter (统一接口)

实现 `interfaces.py` 中定义的 `ServingExporterInterface`：

```python
from algorithm.serving import ServingExporter, ExportConfig

# 创建导出器
exporter = ServingExporter()

# 导出 ONNX 模型
onnx_path = exporter.export_onnx(model, "models/ugt.onnx", config)

# TensorRT 优化
engine_path = exporter.optimize_tensorrt(onnx_path, "models/ugt.plan", config)

# 生成 Triton 配置
config_path = exporter.generate_triton_config("model_repository", config)

# 性能基准测试
metrics = exporter.benchmark("localhost:8001", "ugt_recommend", num_requests=10000)
```

### 2. 便捷函数

```python
from algorithm.serving import (
    export_to_onnx,
    build_trt_engine,
    generate_triton_config,
    run_benchmark,
    create_exporter,
)

# 快速创建配置好的导出器
exporter = create_exporter(
    model_name="my_model",
    precision="fp16",
    max_batch_size=64
)

# 一键完成完整部署
paths = exporter.deploy_full_pipeline(model, "./model_repository")
```

## ⚙️ 配置说明

### ExportConfig

```python
from algorithm.serving import ExportConfig

config = ExportConfig(
    model_name="ugt_recommend",     # 模型名称
    precision="fp16",               # 精度: fp32, fp16, int8
    max_batch_size=64,              # 最大批次大小
    max_seq_length=1024,            # 最大序列长度
    target_latency_ms=30.0,         # 目标延迟 (ms)
    opset_version=17,               # ONNX opset 版本
    workspace_size_gb=4,            # TensorRT 工作空间 (GB)
)
```

### TritonConfig

```python
from algorithm.serving import TritonConfig

config = TritonConfig(
    platform="tensorrt_plan",       # 推理平台
    instance_count=2,               # GPU 实例数
    preferred_batch_sizes=(8, 16, 32, 64),  # 首选批次大小
    max_queue_delay_us=100,         # 最大队列延迟 (微秒)
    gpus=(0, 1),                    # GPU 设备 ID
)
```

### BenchmarkConfig

```python
from algorithm.serving import BenchmarkConfig

config = BenchmarkConfig(
    triton_url="localhost:8001",    # Triton gRPC URL
    num_warmup_requests=100,        # 预热请求数
    num_requests=10000,             # 测试请求数
    concurrency=1,                  # 并发数
)
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch onnx onnxruntime tensorrt tritonclient[http]
```

### 2. 导出模型

```python
from algorithm.serving import ServingExporter, ExportConfig

# 加载训练好的模型
model = load_trained_model("checkpoints/ugt_best.pt")

# 配置
config = ExportConfig(
    model_name="ugt_recommend",
    precision="fp16"
)

# 导出
exporter = ServingExporter(config)
paths = exporter.deploy_full_pipeline(model, "./model_repository")

print(f"ONNX: {paths['onnx_path']}")
print(f"TensorRT: {paths['engine_path']}")
print(f"Config: {paths['config_path']}")
```

### 3. 启动 Triton Server

```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
```

### 4. 运行基准测试

```python
from algorithm.serving import run_benchmark

metrics = run_benchmark(
    triton_url="localhost:8001",
    model_name="ugt_recommend",
    num_requests=10000
)

print(f"吞吐量: {metrics['throughput']:.2f} req/s")
print(f"P99 延迟: {metrics['latency_p99']:.2f} ms")
```

## 🧪 运行测试

```bash
# 运行所有测试
pytest algorithm/serving/tests/ -v

# 运行特定测试
pytest algorithm/serving/tests/test_exporter.py -v

# 生成覆盖率报告
pytest algorithm/serving/tests/ --cov=algorithm/serving --cov-report=html
```

## 📝 脚本使用

### export.sh

```bash
# 基本用法
./scripts/export.sh

# 自定义配置
MODEL_NAME=my_model PRECISION=fp16 ./scripts/export.sh

# 跳过步骤
./scripts/export.sh --skip-export --skip-trt

# 启动服务
./scripts/export.sh --start-server
```

### benchmark.sh

```bash
# 基本用法
./scripts/benchmark.sh

# 自定义配置
./scripts/benchmark.sh --url localhost:8001 --model ugt_recommend --requests 5000

# 使用 perf_analyzer
./scripts/benchmark.sh --perf-analyzer
```

## ⚠️ 注意事项

1. **动态形状**: ONNX 导出时正确设置 `dynamic_axes`，支持可变批次和序列长度
2. **精度**: FP16 可提升 2-3x 性能，但需验证精度损失
3. **批处理**: Triton 动态批处理是延迟与吞吐的权衡
4. **内存**: TensorRT 构建需要足够的 GPU 内存（建议 >= 8GB）
5. **版本**: 确保 TensorRT 版本与 Triton Server 兼容

## 📊 性能参考

| 配置 | 吞吐量 (req/s) | P99 延迟 (ms) |
|------|---------------|--------------|
| BS=1, FP16 | ~500 | ~15 |
| BS=16, FP16 | ~2000 | ~20 |
| BS=32, FP16 | ~3000 | ~25 |
| BS=64, FP16 | ~4000 | ~30 |

*测试环境: NVIDIA A100 40GB, 序列长度 512*

## 🔗 相关文档

- [Triton Inference Server 文档](https://github.com/triton-inference-server/server)
- [TensorRT 文档](https://developer.nvidia.com/tensorrt)
- [ONNX 文档](https://onnx.ai/)

