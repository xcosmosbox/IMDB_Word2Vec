# UGT 推理服务模块 - 开发指南

> 本文档为 Person F（MLOps 工程师）开发的推理服务模块提供详细的学习和开发指南。

---

## 目录

1. [模块概述](#1-模块概述)
2. [架构设计](#2-架构设计)
3. [核心组件详解](#3-核心组件详解)
4. [API 参考](#4-api-参考)
5. [使用示例](#5-使用示例)
6. [开发指南](#6-开发指南)
7. [测试指南](#7-测试指南)
8. [部署指南](#8-部署指南)
9. [常见问题](#9-常见问题)
10. [扩展开发](#10-扩展开发)

---

## 1. 模块概述

### 1.1 功能定位

本模块是 UGT（Unified Generative Transformer）生成式推荐系统的**推理服务部署**模块，负责将训练好的 PyTorch 模型转换为高性能的在线推理服务。

### 1.2 核心职责

| 职责 | 说明 |
|------|------|
| **模型导出** | 将 PyTorch 模型导出为 ONNX 格式 |
| **模型优化** | 使用 TensorRT 进行推理优化 |
| **服务配置** | 生成 Triton Inference Server 配置 |
| **性能测试** | 对部署的服务进行基准测试 |

### 1.3 性能目标

```
┌─────────────────────────────────────────┐
│           性能目标                       │
├─────────────────────────────────────────┤
│  P99 延迟      │  < 30ms                │
│  GPU 利用率    │  > 80%                 │
│  动态批处理    │  支持 1-64 batch       │
│  动态序列长度  │  支持 1-1024 tokens    │
└─────────────────────────────────────────┘
```

### 1.4 技术栈

- **PyTorch**: 模型训练框架
- **ONNX**: 模型交换格式
- **TensorRT**: NVIDIA GPU 推理优化
- **Triton Inference Server**: 高性能推理服务器

---

## 2. 架构设计

### 2.1 部署流水线

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   PyTorch    │ ──▶ │    ONNX      │ ──▶ │  TensorRT    │ ──▶ │   Triton     │
│    Model     │     │   Export     │     │  Optimize    │     │   Deploy     │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
      │                    │                    │                    │
      │                    │                    │                    │
      ▼                    ▼                    ▼                    ▼
 训练好的模型         model.onnx          model.plan         在线服务
```

### 2.2 模块架构

```
serving/
├── 配置层 (config.py)
│   ├── ExportConfig      # 导出配置
│   ├── TritonConfig      # Triton 配置
│   └── BenchmarkConfig   # 测试配置
│
├── 核心层
│   ├── export_onnx.py    # ONNX 导出器
│   ├── optimize_trt.py   # TensorRT 优化器
│   ├── triton_config.py  # Triton 配置生成器
│   └── benchmark.py      # 性能测试器
│
├── 接口层 (exporter.py)
│   └── ServingExporter   # 统一导出接口
│
└── 工具层
    ├── scripts/          # 部署脚本
    └── tests/            # 单元测试
```

### 2.3 类关系图

```
                    ┌─────────────────────────┐
                    │  ServingExporterInterface│  (来自 interfaces.py)
                    │  (抽象接口)              │
                    └───────────┬─────────────┘
                                │ 实现
                                ▼
                    ┌─────────────────────────┐
                    │    ServingExporter      │
                    │    (统一导出器)          │
                    └───────────┬─────────────┘
                                │ 组合
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  ONNXExporter   │  │TensorRTOptimizer│  │TritonConfig     │
│                 │  │                 │  │Generator        │
└─────────────────┘  └─────────────────┘  └─────────────────┘
            │                   │                   │
            ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  ExportConfig   │  │  ExportConfig   │  │  TritonConfig   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 2.4 数据流

```
输入模型                    中间产物                      最终产物
─────────                  ─────────                    ─────────

PyTorch Model              
     │                     
     ▼ export_onnx()       
model.onnx ────────────────────┐
     │                         │
     ▼ optimize_tensorrt()     │
model.plan ────────────────────┼─────────────▶ model_repository/
     │                         │              ├── ugt_recommend/
     ▼ generate_triton_config()│              │   ├── config.pbtxt
config.pbtxt ──────────────────┘              │   └── 1/
                                              │       └── model.plan
```

---

## 3. 核心组件详解

### 3.1 配置类 (config.py)

#### ExportConfig - 导出配置

```python
@dataclass
class ExportConfig:
    """模型导出配置"""
    
    # 基本信息
    model_name: str = "ugt_recommend"      # 模型名称
    precision: str = "fp16"                # 精度: fp32, fp16, int8
    
    # 形状限制
    max_batch_size: int = 64               # 最大批次
    max_seq_length: int = 1024             # 最大序列长度
    
    # 动态形状配置
    min_batch_size: int = 1                # 最小批次
    opt_batch_size: int = 32               # 最优批次（用于 TensorRT 优化）
    min_seq_length: int = 1                # 最小序列长度
    opt_seq_length: int = 512              # 最优序列长度
    
    # ONNX 配置
    opset_version: int = 17                # ONNX opset 版本
    do_constant_folding: bool = True       # 常量折叠优化
    
    # TensorRT 配置
    workspace_size_gb: int = 4             # 工作空间大小
    
    # 语义 ID 配置
    codebook_sizes: Tuple[int, int, int] = (1024, 4096, 16384)
    num_recommendations: int = 50          # 推荐数量
```

**使用说明：**
- `precision`: 推荐使用 `fp16`，可获得约 2-3x 加速，精度损失通常可接受
- `opt_batch_size` 和 `opt_seq_length`: 应设置为实际场景中最常见的值，TensorRT 会针对这些值进行优化
- `workspace_size_gb`: 如果 GPU 内存充足，可以增加此值以获得更好的优化效果

#### TritonConfig - Triton 服务配置

```python
@dataclass
class TritonConfig:
    """Triton Inference Server 配置"""
    
    platform: str = "tensorrt_plan"        # 推理后端
    instance_count: int = 2                # 每个 GPU 的模型实例数
    preferred_batch_sizes: Tuple = (8, 16, 32, 64)  # 首选批次大小
    max_queue_delay_us: int = 100          # 最大队列延迟（微秒）
    gpus: Tuple[int, ...] = (0,)           # GPU 设备列表
    instance_kind: str = "KIND_GPU"        # 实例类型
    enable_dynamic_batching: bool = True   # 启用动态批处理
```

**配置技巧：**
- `instance_count`: 根据 GPU 内存和模型大小调整，通常 2-4 个实例
- `max_queue_delay_us`: 延迟与吞吐的权衡，值越大吞吐越高但延迟也越高
- `preferred_batch_sizes`: 应与实际流量模式匹配

#### BenchmarkConfig - 测试配置

```python
@dataclass
class BenchmarkConfig:
    """性能基准测试配置"""
    
    triton_url: str = "localhost:8001"     # Triton gRPC 端口
    num_warmup_requests: int = 100         # 预热请求数
    num_requests: int = 10000              # 测试请求数
    concurrency: int = 1                   # 并发客户端数
    test_batch_sizes: Tuple = (1, 8, 16, 32)      # 测试的批次大小
    test_seq_lengths: Tuple = (32, 64, 128, 256)  # 测试的序列长度
```

### 3.2 ONNX 导出器 (export_onnx.py)

#### ONNXExporter 类

核心职责：将 PyTorch 模型导出为 ONNX 格式

```python
class ONNXExporter:
    """ONNX 模型导出器"""
    
    def __init__(self, config: ExportConfig):
        """初始化导出器"""
        self.config = config
        self.input_spec = ModelInputSpec()
        config.validate()  # 验证配置
    
    def export(
        self,
        model: nn.Module,
        save_path: str,
        verify: bool = True
    ) -> str:
        """
        导出模型为 ONNX 格式
        
        工作流程：
        1. 设置模型为 eval 模式
        2. 创建示例输入
        3. 配置动态轴
        4. 调用 torch.onnx.export
        5. 验证导出的模型
        6. 尝试简化模型
        """
```

**关键实现细节：**

1. **动态形状支持**
```python
dynamic_axes = {
    "encoder_l1_ids": {0: "batch", 1: "seq_len"},
    "encoder_l2_ids": {0: "batch", 1: "seq_len"},
    # ... 其他输入
    "recommendations": {0: "batch"},
    "scores": {0: "batch"},
}
```

2. **模型包装器**
```python
class ModelWrapper(nn.Module):
    """
    包装 UGT 模型以适配 ONNX 导出格式
    
    原因：ONNX 导出需要固定的输入输出签名，
    而 UGT 模型使用字典/列表形式的输入输出
    """
    
    def forward(self, l1_ids, l2_ids, l3_ids, ...):
        # 组装输入并调用原模型
        # 将输出转换为张量格式
```

### 3.3 TensorRT 优化器 (optimize_trt.py)

#### TensorRTOptimizer 类

核心职责：将 ONNX 模型转换为 TensorRT 引擎

```python
class TensorRTOptimizer:
    """TensorRT 模型优化器"""
    
    def optimize(
        self,
        onnx_path: str,
        engine_path: str,
        calibration_data: Optional[Callable] = None
    ) -> str:
        """
        优化流程：
        1. 创建 TensorRT Builder 和 Network
        2. 解析 ONNX 模型
        3. 配置精度（FP16/INT8）
        4. 配置动态形状 Profile
        5. 构建序列化引擎
        6. 保存到文件
        """
```

**精度配置说明：**

| 精度 | 性能提升 | 精度损失 | 使用建议 |
|------|---------|---------|---------|
| FP32 | 基准 | 无 | 调试和精度验证 |
| FP16 | 2-3x | 极小 | **推荐用于生产** |
| INT8 | 4-8x | 可能明显 | 需要校准数据 |

**INT8 量化使用：**
```python
# 创建校准数据生成器
calibration_gen = create_calibration_data_generator(
    num_samples=100,
    batch_size=32,
    seq_length=512
)

# 使用校准数据进行 INT8 优化
optimizer.optimize(
    onnx_path="model.onnx",
    engine_path="model.plan",
    calibration_data=calibration_gen
)
```

### 3.4 Triton 配置生成器 (triton_config.py)

#### TritonConfigGenerator 类

核心职责：生成 Triton Inference Server 所需的配置文件

```python
class TritonConfigGenerator:
    """Triton 配置生成器"""
    
    def generate(
        self,
        model_repository: str,
        model_file: Optional[str] = None,
        version: int = 1
    ) -> str:
        """
        生成内容：
        1. 创建目录结构: model_name/version/
        2. 复制模型文件
        3. 生成 config.pbtxt
        """
```

**生成的 config.pbtxt 结构：**

```protobuf
name: "ugt_recommend"
platform: "tensorrt_plan"
max_batch_size: 64

input [
  { name: "encoder_l1_ids", data_type: TYPE_INT64, dims: [ -1 ] },
  { name: "encoder_l2_ids", data_type: TYPE_INT64, dims: [ -1 ] },
  # ... 其他输入
]

output [
  { name: "recommendations", data_type: TYPE_INT64, dims: [ 50, 3 ] },
  { name: "scores", data_type: TYPE_FP32, dims: [ 50 ] }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32, 64 ]
  max_queue_delay_microseconds: 100
}

instance_group [
  { count: 2, kind: KIND_GPU, gpus: [ 0, 1 ] }
]
```

### 3.5 性能测试器 (benchmark.py)

#### TritonBenchmark 类

核心职责：对部署的推理服务进行性能测试

```python
class TritonBenchmark:
    """Triton 性能基准测试器"""
    
    def run(
        self,
        triton_url: str,
        model_name: str,
        batch_size: int = 1,
        seq_length: int = 100
    ) -> BenchmarkResult:
        """
        测试流程：
        1. 创建测试输入
        2. 预热（消除冷启动影响）
        3. 执行测试请求
        4. 收集延迟数据
        5. 计算统计指标
        """
    
    def run_sweep(
        self,
        triton_url: str,
        model_name: str
    ) -> Dict[str, BenchmarkResult]:
        """
        参数扫描测试：
        遍历不同的 batch_size 和 seq_length 组合
        """
```

**BenchmarkResult 指标说明：**

```python
@dataclass
class BenchmarkResult:
    throughput: float      # 吞吐量 (请求/秒)
    latency_p50: float     # P50 延迟 (毫秒)
    latency_p90: float     # P90 延迟
    latency_p95: float     # P95 延迟
    latency_p99: float     # P99 延迟
    latency_avg: float     # 平均延迟
    latency_min: float     # 最小延迟
    latency_max: float     # 最大延迟
    total_requests: int    # 总请求数
    successful_requests: int  # 成功请求数
    failed_requests: int   # 失败请求数
```

### 3.6 统一导出器 (exporter.py)

#### ServingExporter 类

核心职责：实现 `ServingExporterInterface` 接口，提供统一的导出入口

```python
class ServingExporter(ServingExporterInterface):
    """统一推理服务导出器"""
    
    def __init__(
        self,
        export_config: Optional[ExportConfig] = None,
        triton_config: Optional[TritonConfig] = None,
        benchmark_config: Optional[BenchmarkConfig] = None
    ):
        # 延迟初始化子组件
        self._onnx_exporter = None
        self._trt_optimizer = None
        self._triton_generator = None
        self._benchmarker = None
    
    # 接口方法实现
    def export_onnx(self, model, save_path, config) -> str: ...
    def optimize_tensorrt(self, onnx_path, engine_path, config) -> str: ...
    def generate_triton_config(self, model_repository, config) -> str: ...
    def benchmark(self, triton_url, model_name, num_requests) -> Dict: ...
    
    # 扩展方法
    def deploy_full_pipeline(self, model, model_repository) -> Dict: ...
    def validate_deployment(self, model_repository) -> Dict: ...
```

---

## 4. API 参考

### 4.1 主要导出接口

```python
from algorithm.serving import (
    # 配置类
    ExportConfig,
    TritonConfig,
    BenchmarkConfig,
    
    # 核心类
    ServingExporter,
    ONNXExporter,
    TensorRTOptimizer,
    TritonConfigGenerator,
    TritonBenchmark,
    
    # 便捷函数
    export_to_onnx,
    build_trt_engine,
    generate_triton_config,
    run_benchmark,
    create_exporter,
)
```

### 4.2 ServingExporter API

#### export_onnx

```python
def export_onnx(
    self,
    model: nn.Module,
    save_path: str,
    config: Optional[ExportConfig] = None
) -> str:
    """
    导出 ONNX 模型
    
    参数:
        model: PyTorch 模型（UGT 模型）
        save_path: 保存路径
        config: 导出配置（可选）
    
    返回:
        导出的 ONNX 文件路径
    
    异常:
        RuntimeError: 导出失败
    """
```

#### optimize_tensorrt

```python
def optimize_tensorrt(
    self,
    onnx_path: str,
    engine_path: str,
    config: Optional[ExportConfig] = None
) -> str:
    """
    TensorRT 优化
    
    参数:
        onnx_path: ONNX 模型路径
        engine_path: TensorRT 引擎保存路径
        config: 导出配置（可选）
    
    返回:
        TensorRT 引擎路径
    
    异常:
        FileNotFoundError: ONNX 文件不存在
        RuntimeError: TensorRT 构建失败
    """
```

#### generate_triton_config

```python
def generate_triton_config(
    self,
    model_repository: str,
    config: Optional[ExportConfig] = None
) -> str:
    """
    生成 Triton 配置
    
    参数:
        model_repository: 模型仓库路径
        config: 导出配置（可选）
    
    返回:
        config.pbtxt 文件路径
    """
```

#### benchmark

```python
def benchmark(
    self,
    triton_url: str,
    model_name: str,
    num_requests: int = 10000
) -> Dict[str, float]:
    """
    性能基准测试
    
    参数:
        triton_url: Triton Server URL
        model_name: 模型名称
        num_requests: 测试请求数
    
    返回:
        {
            "throughput": float,   # req/s
            "latency_p50": float,  # ms
            "latency_p90": float,  # ms
            "latency_p99": float,  # ms
        }
    """
```

#### deploy_full_pipeline

```python
def deploy_full_pipeline(
    self,
    model: nn.Module,
    model_repository: str,
    onnx_path: Optional[str] = None,
    engine_path: Optional[str] = None
) -> Dict[str, str]:
    """
    完整部署流水线
    
    自动执行: ONNX 导出 → TensorRT 优化 → Triton 配置生成
    
    参数:
        model: PyTorch 模型
        model_repository: 模型仓库路径
        onnx_path: ONNX 文件路径（可选）
        engine_path: TensorRT 引擎路径（可选）
    
    返回:
        {
            "onnx_path": str,
            "engine_path": str,
            "config_path": str,
            "model_repository": str,
        }
    """
```

### 4.3 便捷函数 API

```python
# 快速创建导出器
exporter = create_exporter(
    model_name="my_model",
    precision="fp16",
    max_batch_size=64,
    max_seq_length=1024
)

# 快速导出 ONNX
onnx_path = export_to_onnx(model, "model.onnx", config)

# 快速构建 TensorRT 引擎
engine_path = build_trt_engine("model.onnx", "model.plan", config)

# 快速生成 Triton 配置
config_str = generate_triton_config("my_model", config)

# 快速运行基准测试
metrics = run_benchmark("localhost:8001", "my_model", num_requests=10000)
```

---

## 5. 使用示例

### 5.1 基础使用

```python
import torch
from algorithm.serving import ServingExporter, ExportConfig

# 1. 加载训练好的模型
model = load_your_trained_model("checkpoints/ugt_best.pt")

# 2. 创建配置
config = ExportConfig(
    model_name="ugt_recommend",
    precision="fp16",
    max_batch_size=64,
    max_seq_length=1024,
)

# 3. 创建导出器并执行部署
exporter = ServingExporter(config)
paths = exporter.deploy_full_pipeline(model, "./model_repository")

print(f"ONNX 模型: {paths['onnx_path']}")
print(f"TensorRT 引擎: {paths['engine_path']}")
print(f"Triton 配置: {paths['config_path']}")
```

### 5.2 分步导出

```python
from algorithm.serving import (
    ServingExporter,
    ExportConfig,
    TritonConfig,
)

# 配置
export_config = ExportConfig(
    model_name="ugt_recommend",
    precision="fp16",
)

triton_config = TritonConfig(
    instance_count=4,
    gpus=(0, 1, 2, 3),
)

# 创建导出器
exporter = ServingExporter(export_config, triton_config)

# Step 1: 导出 ONNX
onnx_path = exporter.export_onnx(model, "models/ugt.onnx")
print(f"ONNX 导出完成: {onnx_path}")

# Step 2: TensorRT 优化
engine_path = exporter.optimize_tensorrt(onnx_path, "models/ugt.plan")
print(f"TensorRT 优化完成: {engine_path}")

# Step 3: 生成 Triton 配置
config_path = exporter.generate_triton_config("./model_repository")
print(f"Triton 配置生成完成: {config_path}")
```

### 5.3 性能测试

```python
from algorithm.serving import TritonBenchmark, BenchmarkConfig

# 配置
config = BenchmarkConfig(
    triton_url="localhost:8001",
    num_warmup_requests=100,
    num_requests=10000,
    concurrency=4,
)

# 创建测试器
benchmark = TritonBenchmark(config)

# 单配置测试
result = benchmark.run("localhost:8001", "ugt_recommend", batch_size=16, seq_length=256)
print(result.summary())

# 参数扫描测试
results = benchmark.run_sweep("localhost:8001", "ugt_recommend")

# 生成报告
report = benchmark.generate_report(results, "benchmark_report.txt")
print(report)
```

### 5.4 INT8 量化

```python
from algorithm.serving import (
    TensorRTOptimizer,
    ExportConfig,
    create_calibration_data_generator,
)

# 配置 INT8 精度
config = ExportConfig(
    model_name="ugt_recommend",
    precision="int8",
)

# 创建校准数据生成器
calibration_gen = create_calibration_data_generator(
    num_samples=100,      # 校准样本数
    batch_size=32,        # 批次大小
    seq_length=512,       # 序列长度
    codebook_sizes=(1024, 4096, 16384),  # 码本大小
)

# 优化
optimizer = TensorRTOptimizer(config)
engine_path = optimizer.optimize(
    "models/ugt.onnx",
    "models/ugt_int8.plan",
    calibration_data=calibration_gen
)
```

### 5.5 验证部署

```python
from algorithm.serving import ServingExporter

exporter = ServingExporter()

# 验证单个模型
result = exporter.validate_deployment("./model_repository", "ugt_recommend")

if result["valid"]:
    print("✅ 部署验证通过")
    print(f"  版本: {result['info']['versions']}")
else:
    print("❌ 部署验证失败")
    for error in result["errors"]:
        print(f"  错误: {error}")
```

---

## 6. 开发指南

### 6.1 项目结构

```
serving/
├── __init__.py              # 模块导出定义
├── config.py                # 配置类
├── export_onnx.py           # ONNX 导出
├── optimize_trt.py          # TensorRT 优化
├── triton_config.py         # Triton 配置
├── benchmark.py             # 性能测试
├── exporter.py              # 统一接口
├── README.md                # 简要说明
├── DEVELOPMENT_GUIDE.md     # 本文档
├── model_repository/        # Triton 模型仓库
├── scripts/                 # 部署脚本
│   ├── export.sh
│   └── benchmark.sh
└── tests/                   # 单元测试
    ├── test_config.py
    ├── test_export_onnx.py
    ├── test_optimize_trt.py
    ├── test_triton_config.py
    ├── test_benchmark.py
    └── test_exporter.py
```

### 6.2 添加新功能

#### 示例：添加新的优化后端

```python
# 1. 在 config.py 中添加配置
@dataclass
class NewBackendConfig:
    """新后端配置"""
    option1: str = "default"
    option2: int = 100

# 2. 创建新的优化器类
# new_backend.py
class NewBackendOptimizer:
    """新后端优化器"""
    
    def __init__(self, config: ExportConfig):
        self.config = config
    
    def optimize(self, onnx_path: str, output_path: str) -> str:
        """优化逻辑"""
        # 实现优化逻辑
        return output_path

# 3. 在 exporter.py 中集成
class ServingExporter:
    
    def optimize_new_backend(self, onnx_path, output_path, config=None):
        """新后端优化"""
        optimizer = NewBackendOptimizer(config or self.export_config)
        return optimizer.optimize(onnx_path, output_path)

# 4. 在 __init__.py 中导出
from .new_backend import NewBackendOptimizer

# 5. 添加单元测试
# tests/test_new_backend.py
class TestNewBackendOptimizer:
    def test_optimize(self):
        ...
```

### 6.3 代码规范

#### 命名规范

```python
# 类名：PascalCase
class ServingExporter:
    pass

# 函数/方法名：snake_case
def export_to_onnx():
    pass

# 常量：UPPER_SNAKE_CASE
MAX_BATCH_SIZE = 64

# 私有成员：前缀下划线
self._internal_state = None
```

#### 文档字符串规范

```python
def export_onnx(
    self,
    model: nn.Module,
    save_path: str,
    config: Optional[ExportConfig] = None
) -> str:
    """
    导出 ONNX 模型
    
    将 PyTorch 模型导出为 ONNX 格式，支持动态批次和序列长度。
    
    Args:
        model: PyTorch 模型（UGT 模型）
        save_path: ONNX 文件保存路径
        config: 导出配置，如果为 None 则使用实例配置
    
    Returns:
        导出的 ONNX 文件路径
    
    Raises:
        RuntimeError: 导出失败时抛出
    
    Example:
        >>> exporter = ServingExporter()
        >>> path = exporter.export_onnx(model, "model.onnx")
    """
```

#### 类型注解

```python
from typing import Dict, List, Optional, Tuple, Any

def process_data(
    inputs: Dict[str, torch.Tensor],
    options: Optional[List[str]] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    ...
```

### 6.4 依赖管理

**核心依赖：**
```
torch>=2.0.0
onnx>=1.14.0
onnxruntime>=1.15.0
numpy>=1.24.0
```

**可选依赖：**
```
tensorrt>=8.6.0          # TensorRT 优化
tritonclient[http]>=2.34.0  # Triton 客户端
onnxsim>=0.4.0           # ONNX 简化
```

### 6.5 错误处理模式

```python
import logging

logger = logging.getLogger(__name__)

class CustomError(Exception):
    """自定义错误"""
    pass

def risky_operation():
    try:
        # 尝试操作
        result = do_something()
    except SpecificError as e:
        logger.error(f"特定错误: {e}")
        raise CustomError(f"操作失败: {e}") from e
    except Exception as e:
        logger.warning(f"未预期的错误: {e}")
        # 优雅降级或重新抛出
        return default_value
    
    return result
```

---

## 7. 测试指南

### 7.1 运行测试

```bash
# 运行所有测试
pytest algorithm/serving/tests/ -v

# 运行特定测试文件
pytest algorithm/serving/tests/test_exporter.py -v

# 运行特定测试类
pytest algorithm/serving/tests/test_config.py::TestExportConfig -v

# 运行特定测试方法
pytest algorithm/serving/tests/test_config.py::TestExportConfig::test_default_values -v

# 生成覆盖率报告
pytest algorithm/serving/tests/ --cov=algorithm/serving --cov-report=html

# 显示详细输出
pytest algorithm/serving/tests/ -v -s
```

### 7.2 测试结构

```python
# test_example.py

import pytest
from unittest.mock import Mock, patch

class TestMyClass:
    """MyClass 测试类"""
    
    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return ExportConfig(model_name="test")
    
    @pytest.fixture
    def instance(self, config):
        """创建测试实例"""
        return MyClass(config)
    
    def test_initialization(self, config):
        """测试初始化"""
        obj = MyClass(config)
        assert obj.config == config
    
    def test_method_with_mock(self, instance):
        """使用 mock 的测试"""
        with patch.object(instance, '_internal_method') as mock:
            mock.return_value = "mocked"
            result = instance.public_method()
            assert result == "expected"
    
    @pytest.mark.parametrize("input,expected", [
        ("a", 1),
        ("b", 2),
        ("c", 3),
    ])
    def test_parametrized(self, instance, input, expected):
        """参数化测试"""
        assert instance.process(input) == expected
```

### 7.3 Mock 使用指南

```python
from unittest.mock import Mock, patch, MagicMock

# Mock 外部依赖
@patch('serving.export_onnx.torch.onnx.export')
def test_export(mock_export):
    mock_export.return_value = None
    # 测试代码
    mock_export.assert_called_once()

# Mock 类方法
with patch.object(MyClass, 'method', return_value="mocked"):
    result = instance.method()

# Mock 导入
with patch.dict('sys.modules', {'tensorrt': Mock()}):
    # 在没有 tensorrt 的环境中测试
```

### 7.4 测试覆盖目标

| 组件 | 目标覆盖率 |
|------|-----------|
| config.py | >= 95% |
| export_onnx.py | >= 85% |
| optimize_trt.py | >= 80% |
| triton_config.py | >= 90% |
| benchmark.py | >= 85% |
| exporter.py | >= 90% |

---

## 8. 部署指南

### 8.1 环境准备

#### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA GPU (Compute >= 7.0) | A100 / H100 |
| GPU 内存 | 8GB | 40GB+ |
| 系统内存 | 16GB | 64GB+ |
| 存储 | 50GB SSD | 200GB NVMe |

#### 软件要求

```bash
# NVIDIA 驱动
nvidia-smi  # 确认驱动安装

# CUDA
nvcc --version  # CUDA >= 11.8

# Python 环境
python --version  # Python >= 3.8

# Docker（用于 Triton）
docker --version
nvidia-docker --version
```

### 8.2 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# 安装基础依赖
pip install torch>=2.0.0 onnx onnxruntime numpy

# 安装 TensorRT（需要 NVIDIA 账号）
pip install nvidia-tensorrt

# 安装 Triton 客户端
pip install tritonclient[http]

# 安装可选依赖
pip install onnxsim  # ONNX 简化
```

### 8.3 使用脚本部署

#### export.sh 使用

```bash
# 设置环境变量
export MODEL_NAME=ugt_recommend
export PRECISION=fp16
export MAX_BATCH_SIZE=64
export CHECKPOINT_PATH=./checkpoints/ugt_best.pt
export MODEL_REPOSITORY=./model_repository

# 运行导出
./scripts/export.sh

# 跳过某些步骤
./scripts/export.sh --skip-trt

# 启动 Triton 服务
./scripts/export.sh --start-server
```

#### benchmark.sh 使用

```bash
# 基本测试
./scripts/benchmark.sh

# 自定义配置
./scripts/benchmark.sh \
    --url localhost:8001 \
    --model ugt_recommend \
    --requests 10000 \
    --concurrency 4
```

### 8.4 手动启动 Triton

```bash
# 拉取 Triton 镜像
docker pull nvcr.io/nvidia/tritonserver:24.01-py3

# 启动 Triton
docker run --gpus all \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models

# 检查服务状态
curl http://localhost:8000/v2/health/ready

# 检查模型状态
curl http://localhost:8000/v2/models/ugt_recommend/ready
```

### 8.5 Kubernetes 部署

```yaml
# triton-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: triton
  template:
    metadata:
      labels:
        app: triton
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:24.01-py3
        args:
          - tritonserver
          - --model-repository=/models
        ports:
          - containerPort: 8000
            name: http
          - containerPort: 8001
            name: grpc
          - containerPort: 8002
            name: metrics
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
          - name: model-repository
            mountPath: /models
      volumes:
        - name: model-repository
          persistentVolumeClaim:
            claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: triton-service
spec:
  selector:
    app: triton
  ports:
    - port: 8001
      targetPort: 8001
      name: grpc
  type: LoadBalancer
```

---

## 9. 常见问题

### 9.1 ONNX 导出问题

**Q: 导出时出现 "Unsupported operation" 错误**

A: 检查使用的 PyTorch 操作是否支持 ONNX 导出。常见解决方案：
```python
# 使用 torch.jit.script 替代部分操作
@torch.jit.script
def custom_operation(x):
    ...

# 或使用 symbolic 函数注册自定义操作
```

**Q: 动态形状导出失败**

A: 确保正确设置 `dynamic_axes`：
```python
dynamic_axes = {
    "input": {0: "batch", 1: "sequence"},
    "output": {0: "batch"},
}
```

### 9.2 TensorRT 优化问题

**Q: TensorRT 构建内存不足**

A: 减少工作空间大小或使用更小的批次：
```python
config = ExportConfig(
    workspace_size_gb=2,  # 减少工作空间
    max_batch_size=32,    # 减少批次大小
)
```

**Q: FP16 精度损失过大**

A: 某些层可能需要保持 FP32：
```python
# 在 TensorRT 构建时标记特定层
build_config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
# 对特定层设置 FP32 精度
```

### 9.3 Triton 部署问题

**Q: 模型加载失败**

A: 检查以下项目：
1. 模型仓库目录结构是否正确
2. config.pbtxt 语法是否正确
3. 模型文件是否存在且完整
4. GPU 驱动和 CUDA 版本是否兼容

**Q: 推理延迟过高**

A: 优化建议：
1. 启用动态批处理
2. 增加模型实例数
3. 检查 GPU 利用率
4. 使用 CUDA Graph 优化

### 9.4 性能问题

**Q: 吞吐量低于预期**

A: 检查并优化：
1. 确认使用 FP16 精度
2. 增加 `instance_count`
3. 调整 `preferred_batch_sizes`
4. 检查是否有 CPU 瓶颈

**Q: 延迟波动大**

A: 可能原因：
1. GPU 温度过高导致降频
2. 其他进程占用 GPU
3. 动态批处理导致等待

---

## 10. 扩展开发

### 10.1 添加新的推理后端

```python
# backends/openvino.py
class OpenVINOOptimizer:
    """OpenVINO 优化器"""
    
    def __init__(self, config: ExportConfig):
        self.config = config
    
    def optimize(self, onnx_path: str, output_dir: str) -> str:
        """转换为 OpenVINO IR 格式"""
        from openvino.tools import mo
        
        mo.convert_model(
            onnx_path,
            output_dir=output_dir,
            compress_to_fp16=(self.config.precision == "fp16"),
        )
        return f"{output_dir}/model.xml"
```

### 10.2 添加新的基准测试指标

```python
# benchmark.py 扩展
@dataclass
class ExtendedBenchmarkResult(BenchmarkResult):
    """扩展的基准测试结果"""
    
    gpu_memory_peak_mb: float = 0.0
    gpu_utilization_avg: float = 0.0
    power_consumption_avg_w: float = 0.0

class GPUMonitoredBenchmark(TritonBenchmark):
    """带 GPU 监控的基准测试"""
    
    def run(self, ...):
        # 启动 GPU 监控
        with GPUMonitor() as monitor:
            result = super().run(...)
        
        # 添加 GPU 指标
        result.gpu_memory_peak_mb = monitor.peak_memory_mb
        result.gpu_utilization_avg = monitor.avg_utilization
        
        return result
```

### 10.3 添加模型版本管理

```python
# version_manager.py
class ModelVersionManager:
    """模型版本管理器"""
    
    def __init__(self, model_repository: str):
        self.repo = Path(model_repository)
    
    def deploy_version(
        self,
        model_name: str,
        model_file: str,
        version: Optional[int] = None
    ) -> int:
        """部署新版本"""
        if version is None:
            version = self._get_next_version(model_name)
        
        version_dir = self.repo / model_name / str(version)
        version_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy(model_file, version_dir / "model.plan")
        
        return version
    
    def rollback(self, model_name: str, version: int) -> None:
        """回滚到指定版本"""
        ...
    
    def cleanup_old_versions(self, model_name: str, keep: int = 3) -> None:
        """清理旧版本"""
        ...
```

### 10.4 添加 A/B 测试支持

```python
# ab_testing.py
class ABTestingRouter:
    """A/B 测试路由器"""
    
    def __init__(self, model_a: str, model_b: str, traffic_split: float = 0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
    
    def route(self, user_id: str) -> str:
        """根据用户 ID 路由到模型"""
        hash_value = hash(user_id) % 100
        if hash_value < self.traffic_split * 100:
            return self.model_a
        return self.model_b
```

---

## 附录

### A. 术语表

| 术语 | 说明 |
|------|------|
| ONNX | Open Neural Network Exchange，开放神经网络交换格式 |
| TensorRT | NVIDIA 高性能深度学习推理优化器 |
| Triton | NVIDIA Triton Inference Server，高性能推理服务器 |
| FP16 | 半精度浮点数 |
| INT8 | 8位整数量化 |
| Dynamic Batching | 动态批处理，将多个请求合并处理 |
| P99 Latency | 99% 请求的延迟上界 |
| Throughput | 吞吐量，单位时间处理的请求数 |

### B. 参考链接

- [ONNX 官方文档](https://onnx.ai/onnx/)
- [TensorRT 开发者指南](https://developer.nvidia.com/tensorrt)
- [Triton Inference Server 文档](https://github.com/triton-inference-server/server)
- [PyTorch ONNX 导出指南](https://pytorch.org/docs/stable/onnx.html)

### C. 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| 1.0.0 | 2026-01-04 | 初始版本 |

---

*文档维护者: Person F (MLOps Engineer)*
*最后更新: 2026-01-04*

