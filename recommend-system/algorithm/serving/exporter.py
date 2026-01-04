"""
统一推理服务导出器

实现 ServingExporterInterface 接口，提供完整的模型导出和部署功能。

功能：
1. 导出 ONNX 模型
2. TensorRT 优化
3. 生成 Triton 配置
4. 性能基准测试

Author: Person F (MLOps Engineer)
"""

import os
import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

# 导入父目录的接口定义
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..interfaces import ServingExporterInterface, ExportConfig as BaseExportConfig

from .config import ExportConfig, TritonConfig, BenchmarkConfig, DeploymentConfig
from .export_onnx import ONNXExporter, wrap_model_for_export
from .optimize_trt import TensorRTOptimizer
from .triton_config import TritonConfigGenerator, setup_model_repository
from .benchmark import TritonBenchmark, BenchmarkResult

# 配置日志
logger = logging.getLogger(__name__)


class ServingExporter(ServingExporterInterface):
    """
    统一推理服务导出器
    
    实现 interfaces.py 中定义的 ServingExporterInterface 接口。
    提供从模型训练到部署的完整工作流。
    
    Attributes:
        export_config: 导出配置
        triton_config: Triton 配置
        benchmark_config: 基准测试配置
    
    Example:
        >>> exporter = ServingExporter()
        >>> # 导出 ONNX
        >>> onnx_path = exporter.export_onnx(model, "models/ugt.onnx", config)
        >>> # TensorRT 优化
        >>> engine_path = exporter.optimize_tensorrt(onnx_path, "models/ugt.plan", config)
        >>> # 生成 Triton 配置
        >>> config_path = exporter.generate_triton_config("model_repository", config)
        >>> # 性能测试
        >>> metrics = exporter.benchmark("localhost:8001", "ugt_recommend", 10000)
    """
    
    def __init__(
        self,
        export_config: Optional[ExportConfig] = None,
        triton_config: Optional[TritonConfig] = None,
        benchmark_config: Optional[BenchmarkConfig] = None
    ):
        """
        初始化导出器
        
        Args:
            export_config: 导出配置
            triton_config: Triton 配置
            benchmark_config: 基准测试配置
        """
        self.export_config = export_config or ExportConfig()
        self.triton_config = triton_config or TritonConfig()
        self.benchmark_config = benchmark_config or BenchmarkConfig()
        
        # 初始化子模块
        self._onnx_exporter = None
        self._trt_optimizer = None
        self._triton_generator = None
        self._benchmarker = None
    
    @property
    def onnx_exporter(self) -> ONNXExporter:
        """获取 ONNX 导出器（延迟初始化）"""
        if self._onnx_exporter is None:
            self._onnx_exporter = ONNXExporter(self.export_config)
        return self._onnx_exporter
    
    @property
    def trt_optimizer(self) -> TensorRTOptimizer:
        """获取 TensorRT 优化器（延迟初始化）"""
        if self._trt_optimizer is None:
            self._trt_optimizer = TensorRTOptimizer(self.export_config)
        return self._trt_optimizer
    
    @property
    def triton_generator(self) -> TritonConfigGenerator:
        """获取 Triton 配置生成器（延迟初始化）"""
        if self._triton_generator is None:
            self._triton_generator = TritonConfigGenerator(
                self.export_config,
                self.triton_config
            )
        return self._triton_generator
    
    @property
    def benchmarker(self) -> TritonBenchmark:
        """获取基准测试器（延迟初始化）"""
        if self._benchmarker is None:
            self._benchmarker = TritonBenchmark(
                self.benchmark_config,
                self.export_config
            )
        return self._benchmarker
    
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
        """
        if config is not None:
            self.export_config = config
            self._onnx_exporter = None  # 重置以使用新配置
        
        logger.info(f"开始导出 ONNX 模型: {save_path}")
        
        # 包装模型以适配 ONNX 导出格式
        wrapped_model = wrap_model_for_export(
            model,
            num_recommendations=self.export_config.num_recommendations
        )
        
        # 导出模型
        onnx_path = self.onnx_exporter.export(wrapped_model, save_path)
        
        logger.info(f"ONNX 模型导出成功: {onnx_path}")
        return onnx_path
    
    def optimize_tensorrt(
        self,
        onnx_path: str,
        engine_path: str,
        config: Optional[ExportConfig] = None
    ) -> str:
        """
        TensorRT 优化
        
        将 ONNX 模型转换为高性能的 TensorRT 引擎。
        
        Args:
            onnx_path: ONNX 模型路径
            engine_path: TensorRT 引擎保存路径
            config: 导出配置，如果为 None 则使用实例配置
        
        Returns:
            生成的 TensorRT 引擎路径
        
        Raises:
            FileNotFoundError: ONNX 文件不存在
            RuntimeError: TensorRT 构建失败
        """
        if config is not None:
            self.export_config = config
            self._trt_optimizer = None  # 重置以使用新配置
        
        logger.info(f"开始 TensorRT 优化: {onnx_path} -> {engine_path}")
        
        # 执行优化
        result_path = self.trt_optimizer.optimize(onnx_path, engine_path)
        
        logger.info(f"TensorRT 引擎生成成功: {result_path}")
        return result_path
    
    def generate_triton_config(
        self,
        model_repository: str,
        config: Optional[ExportConfig] = None
    ) -> str:
        """
        生成 Triton 配置
        
        创建 Triton Inference Server 所需的配置文件和目录结构。
        
        Args:
            model_repository: Triton 模型仓库路径
            config: 导出配置，如果为 None 则使用实例配置
        
        Returns:
            生成的 config.pbtxt 文件路径
        """
        if config is not None:
            self.export_config = config
            self._triton_generator = None  # 重置以使用新配置
        
        logger.info(f"生成 Triton 配置: {model_repository}")
        
        # 生成配置
        config_path = self.triton_generator.generate(model_repository)
        
        logger.info(f"Triton 配置生成成功: {config_path}")
        return config_path
    
    def benchmark(
        self,
        triton_url: str,
        model_name: str,
        num_requests: int = 10000
    ) -> Dict[str, float]:
        """
        性能测试
        
        对部署的 Triton 推理服务进行性能基准测试。
        
        Args:
            triton_url: Triton Server URL（如 "localhost:8001"）
            model_name: 模型名称
            num_requests: 测试请求数量
        
        Returns:
            性能指标字典：
                - "throughput": 吞吐量 (req/s)
                - "latency_p50": P50 延迟 (ms)
                - "latency_p90": P90 延迟 (ms)
                - "latency_p99": P99 延迟 (ms)
        """
        logger.info(f"开始性能测试: url={triton_url}, model={model_name}")
        
        # 更新配置
        self.benchmark_config.triton_url = triton_url
        self.benchmark_config.num_requests = num_requests
        self._benchmarker = None  # 重置以使用新配置
        
        # 运行测试
        result = self.benchmarker.run(triton_url, model_name)
        
        logger.info(f"性能测试完成: throughput={result.throughput:.2f} req/s, "
                   f"P99={result.latency_p99:.2f}ms")
        
        return {
            "throughput": result.throughput,
            "latency_p50": result.latency_p50,
            "latency_p90": result.latency_p90,
            "latency_p99": result.latency_p99,
        }
    
    def deploy_full_pipeline(
        self,
        model: nn.Module,
        model_repository: str,
        onnx_path: Optional[str] = None,
        engine_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        执行完整的部署流水线
        
        按顺序执行：ONNX 导出 → TensorRT 优化 → Triton 配置生成
        
        Args:
            model: PyTorch 模型
            model_repository: Triton 模型仓库路径
            onnx_path: ONNX 文件路径（可选，默认生成）
            engine_path: TensorRT 引擎路径（可选，默认生成）
        
        Returns:
            包含所有生成文件路径的字典
        """
        model_name = self.export_config.model_name
        
        # 设置默认路径
        if onnx_path is None:
            onnx_path = os.path.join(model_repository, f"{model_name}.onnx")
        
        if engine_path is None:
            engine_path = os.path.join(
                model_repository, model_name, "1", "model.plan"
            )
        
        logger.info("=" * 60)
        logger.info("开始完整部署流水线")
        logger.info("=" * 60)
        
        # Step 1: 导出 ONNX
        logger.info("\n[Step 1/3] 导出 ONNX 模型")
        onnx_result = self.export_onnx(model, onnx_path)
        
        # Step 2: TensorRT 优化
        logger.info("\n[Step 2/3] TensorRT 优化")
        engine_result = self.optimize_tensorrt(onnx_result, engine_path)
        
        # Step 3: 生成 Triton 配置
        logger.info("\n[Step 3/3] 生成 Triton 配置")
        config_result = self.generate_triton_config(model_repository)
        
        logger.info("\n" + "=" * 60)
        logger.info("部署流水线完成")
        logger.info("=" * 60)
        
        return {
            "onnx_path": onnx_result,
            "engine_path": engine_result,
            "config_path": config_result,
            "model_repository": model_repository,
        }
    
    def validate_deployment(
        self,
        model_repository: str,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        验证部署配置
        
        检查模型仓库的配置完整性和正确性。
        
        Args:
            model_repository: 模型仓库路径
            model_name: 模型名称（可选，默认使用配置中的名称）
        
        Returns:
            验证结果字典
        """
        from .triton_config import TritonModelValidator
        
        if model_name is None:
            model_name = self.export_config.model_name
        
        validator = TritonModelValidator(model_repository)
        result = validator.validate(model_name)
        
        if result["valid"]:
            logger.info(f"模型 {model_name} 验证通过")
        else:
            logger.error(f"模型 {model_name} 验证失败: {result['errors']}")
        
        return result
    
    def get_deployment_info(self) -> Dict[str, Any]:
        """
        获取当前部署配置信息
        
        Returns:
            部署配置信息字典
        """
        return {
            "export_config": self.export_config.to_dict(),
            "triton_config": {
                "platform": self.triton_config.platform,
                "instance_count": self.triton_config.instance_count,
                "preferred_batch_sizes": self.triton_config.preferred_batch_sizes,
                "max_queue_delay_us": self.triton_config.max_queue_delay_us,
                "gpus": self.triton_config.gpus,
            },
            "benchmark_config": {
                "num_requests": self.benchmark_config.num_requests,
                "concurrency": self.benchmark_config.concurrency,
                "num_warmup_requests": self.benchmark_config.num_warmup_requests,
            },
        }


def create_exporter(
    model_name: str = "ugt_recommend",
    precision: str = "fp16",
    max_batch_size: int = 64,
    max_seq_length: int = 1024
) -> ServingExporter:
    """
    便捷函数：创建配置好的导出器
    
    Args:
        model_name: 模型名称
        precision: 推理精度 (fp32, fp16, int8)
        max_batch_size: 最大批次大小
        max_seq_length: 最大序列长度
    
    Returns:
        配置好的 ServingExporter 实例
    
    Example:
        >>> exporter = create_exporter("my_model", precision="fp16")
        >>> exporter.deploy_full_pipeline(model, "./model_repo")
    """
    export_config = ExportConfig(
        model_name=model_name,
        precision=precision,
        max_batch_size=max_batch_size,
        max_seq_length=max_seq_length,
    )
    
    triton_config = TritonConfig(
        platform="tensorrt_plan" if precision != "fp32" else "onnxruntime_onnx",
    )
    
    return ServingExporter(export_config, triton_config)

