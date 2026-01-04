# Person F: 推理服务

## 你的角色
你是一名 MLOps 工程师，负责实现生成式推荐系统的 **推理服务部署** 模块。

## 背景知识

UGT 模型需要部署为高性能在线推理服务，要求：
- P99 延迟 < 30ms
- 支持动态 Batching
- GPU 利用率 > 80%

### 部署架构
```
PyTorch Model → ONNX → TensorRT → Triton Inference Server
```

## 你的任务

在 `algorithm/serving/` 目录下实现完整的部署模块。

### 目录结构
```
algorithm/serving/
├── __init__.py
├── config.py           # 配置类
├── export_onnx.py      # 导出 ONNX
├── optimize_trt.py     # TensorRT 优化
├── triton_config.py    # Triton 配置生成
├── benchmark.py        # 性能测试
├── model_repository/   # Triton 模型仓库
└── scripts/
    ├── export.sh
    └── benchmark.sh
```

### 接口要求

实现 `interfaces.py` 中的 `ServingExporterInterface`：

```python
class ServingExporter(ServingExporterInterface):
    def export_onnx(self, model, save_path, config) -> str:
        """导出 ONNX 模型"""
        pass
    
    def optimize_tensorrt(self, onnx_path, engine_path, config) -> str:
        """TensorRT 优化"""
        pass
    
    def generate_triton_config(self, model_repository, config) -> str:
        """生成 Triton 配置"""
        pass
    
    def benchmark(self, triton_url, model_name, num_requests) -> Dict[str, float]:
        """性能测试"""
        pass
```

### 核心实现

#### 1. config.py
```python
from dataclasses import dataclass
from typing import List

@dataclass
class ExportConfig:
    model_name: str = "ugt_recommend"
    precision: str = "fp16"  # fp32, fp16, int8
    max_batch_size: int = 64
    max_seq_length: int = 1024
    target_latency_ms: float = 30.0

@dataclass
class TritonConfig:
    instance_count: int = 2
    preferred_batch_sizes: List[int] = (8, 16, 32, 64)
    max_queue_delay_us: int = 100
```

#### 2. export_onnx.py
```python
import torch
import torch.onnx

def export_to_onnx(model, save_path, config):
    """
    导出支持动态 batch 和动态序列长度的 ONNX 模型
    """
    model.eval()
    
    # 示例输入
    batch_size = 1
    seq_len = 100
    example_inputs = {
        "encoder_l1_ids": torch.randint(0, 1024, (batch_size, seq_len)),
        "encoder_l2_ids": torch.randint(0, 4096, (batch_size, seq_len)),
        "encoder_l3_ids": torch.randint(0, 16384, (batch_size, seq_len)),
        "encoder_positions": torch.arange(seq_len).unsqueeze(0),
        "encoder_token_types": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "encoder_mask": torch.ones(batch_size, seq_len),
    }
    
    # 动态轴
    dynamic_axes = {
        "encoder_l1_ids": {0: "batch", 1: "seq_len"},
        "encoder_l2_ids": {0: "batch", 1: "seq_len"},
        "encoder_l3_ids": {0: "batch", 1: "seq_len"},
        "encoder_positions": {0: "batch", 1: "seq_len"},
        "encoder_token_types": {0: "batch", 1: "seq_len"},
        "encoder_mask": {0: "batch", 1: "seq_len"},
        "recommendations": {0: "batch"},
        "scores": {0: "batch"},
    }
    
    torch.onnx.export(
        model,
        tuple(example_inputs.values()),
        save_path,
        input_names=list(example_inputs.keys()),
        output_names=["recommendations", "scores"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    
    print(f"ONNX model exported to {save_path}")
    return save_path
```

#### 3. optimize_trt.py
```python
import tensorrt as trt

def build_trt_engine(onnx_path, engine_path, config):
    """
    使用 TensorRT 优化模型
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 解析 ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")
    
    # 配置
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    
    # 精度设置
    if config.precision == "fp16":
        build_config.set_flag(trt.BuilderFlag.FP16)
    elif config.precision == "int8":
        build_config.set_flag(trt.BuilderFlag.INT8)
    
    # 动态形状
    profile = builder.create_optimization_profile()
    profile.set_shape("encoder_l1_ids", (1, 1), (32, 512), (64, 1024))
    # ... 其他输入的形状设置
    build_config.add_optimization_profile(profile)
    
    # 构建引擎
    engine = builder.build_serialized_network(network, build_config)
    
    with open(engine_path, 'wb') as f:
        f.write(engine)
    
    print(f"TensorRT engine saved to {engine_path}")
    return engine_path
```

#### 4. triton_config.py
```python
def generate_triton_config(model_name, config):
    """生成 Triton config.pbtxt"""
    return f'''
name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: {config.max_batch_size}

input [
  {{ name: "encoder_l1_ids", data_type: TYPE_INT64, dims: [ -1 ] }},
  {{ name: "encoder_l2_ids", data_type: TYPE_INT64, dims: [ -1 ] }},
  {{ name: "encoder_l3_ids", data_type: TYPE_INT64, dims: [ -1 ] }},
  {{ name: "encoder_positions", data_type: TYPE_INT64, dims: [ -1 ] }},
  {{ name: "encoder_token_types", data_type: TYPE_INT64, dims: [ -1 ] }},
  {{ name: "encoder_mask", data_type: TYPE_FP32, dims: [ -1 ] }}
]

output [
  {{ name: "recommendations", data_type: TYPE_INT64, dims: [ 50, 3 ] }},
  {{ name: "scores", data_type: TYPE_FP32, dims: [ 50 ] }}
]

dynamic_batching {{
  preferred_batch_size: [ 8, 16, 32, 64 ]
  max_queue_delay_microseconds: 100
}}

instance_group [
  {{ count: 2, kind: KIND_GPU, gpus: [ 0, 1 ] }}
]
'''
```

#### 5. benchmark.py
```python
import time
import numpy as np
import tritonclient.http as httpclient

def run_benchmark(triton_url, model_name, num_requests=10000):
    """性能测试"""
    client = httpclient.InferenceServerClient(url=triton_url)
    
    # 准备输入
    seq_len = 100
    inputs = [
        httpclient.InferInput("encoder_l1_ids", [1, seq_len], "INT64"),
        httpclient.InferInput("encoder_l2_ids", [1, seq_len], "INT64"),
        httpclient.InferInput("encoder_l3_ids", [1, seq_len], "INT64"),
    ]
    
    inputs[0].set_data_from_numpy(np.random.randint(0, 1024, (1, seq_len)).astype(np.int64))
    inputs[1].set_data_from_numpy(np.random.randint(0, 4096, (1, seq_len)).astype(np.int64))
    inputs[2].set_data_from_numpy(np.random.randint(0, 16384, (1, seq_len)).astype(np.int64))
    
    # 预热
    for _ in range(100):
        client.infer(model_name, inputs)
    
    # 测试
    latencies = []
    start = time.time()
    
    for _ in range(num_requests):
        req_start = time.time()
        client.infer(model_name, inputs)
        latencies.append((time.time() - req_start) * 1000)
    
    total_time = time.time() - start
    
    return {
        "throughput": num_requests / total_time,
        "latency_p50": np.percentile(latencies, 50),
        "latency_p90": np.percentile(latencies, 90),
        "latency_p99": np.percentile(latencies, 99),
    }
```

### 部署脚本

```bash
#!/bin/bash
# scripts/export.sh

# 1. 导出 ONNX
python export_onnx.py --model_path checkpoints/ugt_best.pt --output models/ugt.onnx

# 2. 转换 TensorRT
python optimize_trt.py --onnx_path models/ugt.onnx --output models/ugt.plan --precision fp16

# 3. 生成 Triton 配置
python triton_config.py --model_name ugt_recommend --output model_repository/

# 4. 启动 Triton
docker run --gpus all -p 8001:8001 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
```

### 测试用例
```python
def test_serving():
    from algorithm.serving.config import ExportConfig
    from algorithm.serving.export_onnx import export_to_onnx
    
    config = ExportConfig()
    
    # 模拟模型
    class MockModel(nn.Module):
        def forward(self, *args):
            return torch.randint(0, 16384, (args[0].shape[0], 50, 3)), torch.randn(args[0].shape[0], 50)
    
    model = MockModel()
    export_to_onnx(model, "/tmp/test.onnx", config)
    
    print("Serving tests passed!")
```

## 注意事项

1. **动态形状**: ONNX 导出时正确设置 dynamic_axes
2. **精度**: FP16 可加速 2-3x，但要验证精度损失
3. **批处理**: Triton 的动态批处理是延迟与吞吐的权衡
4. **内存**: TensorRT 需要足够的 GPU 内存

## 输出要求

请输出完整的可运行代码，包含：
1. 所有 Python 文件
2. 详细的中文注释
3. 部署脚本
4. 使用示例

确保代码遵循 `algorithm/interfaces.py` 中定义的接口。

