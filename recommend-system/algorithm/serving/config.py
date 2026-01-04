"""
推理服务配置模块

定义导出、部署和性能测试相关的配置类。

Author: Person F (MLOps Engineer)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class Precision(Enum):
    """模型精度枚举"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class InstanceKind(Enum):
    """Triton 实例类型枚举"""
    GPU = "KIND_GPU"
    CPU = "KIND_CPU"


@dataclass
class ExportConfig:
    """
    ONNX 和 TensorRT 导出配置
    
    Attributes:
        model_name: 模型名称，用于生成文件名和 Triton 配置
        precision: 推理精度 (fp32, fp16, int8)
        max_batch_size: 最大批次大小
        max_seq_length: 最大序列长度
        target_latency_ms: 目标延迟（毫秒）
        opset_version: ONNX opset 版本
        do_constant_folding: 是否进行常量折叠优化
        workspace_size_gb: TensorRT 工作空间大小（GB）
        enable_dynamic_shapes: 是否启用动态形状
        min_batch_size: 动态形状的最小批次大小
        opt_batch_size: 动态形状的最优批次大小
        min_seq_length: 动态形状的最小序列长度
        opt_seq_length: 动态形状的最优序列长度
    """
    # 基本配置
    model_name: str = "ugt_recommend"
    precision: str = "fp16"
    max_batch_size: int = 64
    max_seq_length: int = 1024
    target_latency_ms: float = 30.0
    
    # ONNX 配置
    opset_version: int = 17
    do_constant_folding: bool = True
    
    # TensorRT 配置
    workspace_size_gb: int = 4
    enable_dynamic_shapes: bool = True
    
    # 动态形状范围配置
    min_batch_size: int = 1
    opt_batch_size: int = 32
    min_seq_length: int = 1
    opt_seq_length: int = 512
    
    # 语义 ID 配置（与 ModelConfig 保持一致）
    codebook_sizes: Tuple[int, int, int] = (1024, 4096, 16384)
    num_recommendations: int = 50
    
    def validate(self) -> None:
        """验证配置有效性"""
        valid_precisions = ["fp32", "fp16", "int8"]
        if self.precision not in valid_precisions:
            raise ValueError(f"precision 必须是 {valid_precisions} 之一，当前值: {self.precision}")
        
        if self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size 必须大于 0，当前值: {self.max_batch_size}")
        
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length 必须大于 0，当前值: {self.max_seq_length}")
        
        if self.min_batch_size > self.opt_batch_size > self.max_batch_size:
            raise ValueError("批次大小必须满足: min <= opt <= max")
        
        if self.min_seq_length > self.opt_seq_length > self.max_seq_length:
            raise ValueError("序列长度必须满足: min <= opt <= max")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "model_name": self.model_name,
            "precision": self.precision,
            "max_batch_size": self.max_batch_size,
            "max_seq_length": self.max_seq_length,
            "target_latency_ms": self.target_latency_ms,
            "opset_version": self.opset_version,
            "do_constant_folding": self.do_constant_folding,
            "workspace_size_gb": self.workspace_size_gb,
            "enable_dynamic_shapes": self.enable_dynamic_shapes,
            "min_batch_size": self.min_batch_size,
            "opt_batch_size": self.opt_batch_size,
            "min_seq_length": self.min_seq_length,
            "opt_seq_length": self.opt_seq_length,
            "codebook_sizes": self.codebook_sizes,
            "num_recommendations": self.num_recommendations,
        }


@dataclass
class TritonConfig:
    """
    Triton Inference Server 配置
    
    Attributes:
        platform: 推理平台（tensorrt_plan, onnxruntime_onnx 等）
        instance_count: 每个 GPU 的模型实例数
        preferred_batch_sizes: 动态批处理的首选批次大小
        max_queue_delay_us: 最大队列延迟（微秒）
        gpus: GPU 设备 ID 列表
        instance_kind: 实例类型（GPU 或 CPU）
        enable_dynamic_batching: 是否启用动态批处理
        response_cache_byte_size: 响应缓存大小（字节）
        enable_request_priority: 是否启用请求优先级
    """
    platform: str = "tensorrt_plan"
    instance_count: int = 2
    preferred_batch_sizes: Tuple[int, ...] = (8, 16, 32, 64)
    max_queue_delay_us: int = 100
    gpus: Tuple[int, ...] = (0,)
    instance_kind: str = "KIND_GPU"
    enable_dynamic_batching: bool = True
    response_cache_byte_size: int = 0
    enable_request_priority: bool = False
    
    def validate(self) -> None:
        """验证配置有效性"""
        valid_platforms = ["tensorrt_plan", "onnxruntime_onnx", "pytorch_libtorch"]
        if self.platform not in valid_platforms:
            raise ValueError(f"platform 必须是 {valid_platforms} 之一")
        
        if self.instance_count <= 0:
            raise ValueError("instance_count 必须大于 0")
        
        if self.max_queue_delay_us < 0:
            raise ValueError("max_queue_delay_us 必须非负")


@dataclass
class BenchmarkConfig:
    """
    性能基准测试配置
    
    Attributes:
        triton_url: Triton Server URL
        num_warmup_requests: 预热请求数
        num_requests: 测试请求总数
        concurrency: 并发请求数
        test_seq_lengths: 测试序列长度列表
        test_batch_sizes: 测试批次大小列表
        timeout_seconds: 请求超时时间（秒）
        report_percentiles: 需要统计的延迟百分位数
    """
    triton_url: str = "localhost:8001"
    num_warmup_requests: int = 100
    num_requests: int = 10000
    concurrency: int = 1
    test_seq_lengths: Tuple[int, ...] = (32, 64, 128, 256, 512)
    test_batch_sizes: Tuple[int, ...] = (1, 8, 16, 32)
    timeout_seconds: float = 30.0
    report_percentiles: Tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)
    
    def validate(self) -> None:
        """验证配置有效性"""
        if self.num_warmup_requests < 0:
            raise ValueError("num_warmup_requests 必须非负")
        
        if self.num_requests <= 0:
            raise ValueError("num_requests 必须大于 0")
        
        if self.concurrency <= 0:
            raise ValueError("concurrency 必须大于 0")


@dataclass
class ModelInputSpec:
    """
    模型输入规格定义
    
    用于 ONNX 导出和 Triton 配置生成
    """
    # 输入张量名称和类型
    encoder_l1_ids: str = "encoder_l1_ids"
    encoder_l2_ids: str = "encoder_l2_ids"
    encoder_l3_ids: str = "encoder_l3_ids"
    encoder_positions: str = "encoder_positions"
    encoder_token_types: str = "encoder_token_types"
    encoder_mask: str = "encoder_mask"
    
    # 输出张量名称
    recommendations: str = "recommendations"
    scores: str = "scores"
    
    @classmethod
    def get_input_names(cls) -> List[str]:
        """获取所有输入名称列表"""
        return [
            cls.encoder_l1_ids,
            cls.encoder_l2_ids,
            cls.encoder_l3_ids,
            cls.encoder_positions,
            cls.encoder_token_types,
            cls.encoder_mask,
        ]
    
    @classmethod
    def get_output_names(cls) -> List[str]:
        """获取所有输出名称列表"""
        return [
            cls.recommendations,
            cls.scores,
        ]


@dataclass
class DeploymentConfig:
    """
    完整部署配置（聚合所有子配置）
    """
    export_config: ExportConfig = field(default_factory=ExportConfig)
    triton_config: TritonConfig = field(default_factory=TritonConfig)
    benchmark_config: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    input_spec: ModelInputSpec = field(default_factory=ModelInputSpec)
    
    # 路径配置
    model_repository: str = "./model_repository"
    checkpoint_path: str = "./checkpoints/ugt_best.pt"
    onnx_output_path: str = "./models/ugt.onnx"
    trt_output_path: str = "./models/ugt.plan"
    
    def validate_all(self) -> None:
        """验证所有子配置"""
        self.export_config.validate()
        self.triton_config.validate()
        self.benchmark_config.validate()

