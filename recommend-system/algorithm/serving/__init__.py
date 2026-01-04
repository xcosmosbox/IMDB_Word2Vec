"""
生成式推荐系统 - 推理服务模块

本模块负责将训练好的 UGT 模型导出为可部署的高性能推理服务。

功能：
1. ONNX 模型导出 - 支持动态 batch 和序列长度
2. TensorRT 优化 - FP16/INT8 精度，显著提升推理性能
3. Triton 配置生成 - 自动生成 Triton Inference Server 配置
4. 性能基准测试 - 延迟和吞吐量测试

性能目标：
- P99 延迟 < 30ms
- 支持动态 Batching
- GPU 利用率 > 80%

部署架构：
PyTorch Model → ONNX → TensorRT → Triton Inference Server

Author: Person F (MLOps Engineer)
"""

from .config import ExportConfig, TritonConfig, BenchmarkConfig
from .export_onnx import ONNXExporter, export_to_onnx
from .optimize_trt import TensorRTOptimizer, build_trt_engine
from .triton_config import TritonConfigGenerator, generate_triton_config
from .benchmark import TritonBenchmark, run_benchmark
from .exporter import ServingExporter

__all__ = [
    # 配置类
    "ExportConfig",
    "TritonConfig",
    "BenchmarkConfig",
    # 导出器
    "ONNXExporter",
    "export_to_onnx",
    # TensorRT 优化器
    "TensorRTOptimizer",
    "build_trt_engine",
    # Triton 配置
    "TritonConfigGenerator",
    "generate_triton_config",
    # 基准测试
    "TritonBenchmark",
    "run_benchmark",
    # 统一接口
    "ServingExporter",
]

__version__ = "1.0.0"

