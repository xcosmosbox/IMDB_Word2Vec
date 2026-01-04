"""
TensorRT 优化模块

将 ONNX 模型转换为 TensorRT 引擎，实现高性能 GPU 推理。

特性：
1. 多精度支持 - FP32/FP16/INT8
2. 动态形状 - 支持可变批次和序列长度
3. 优化配置 - 工作空间、层融合等
4. INT8 量化 - 使用校准数据集

Author: Person F (MLOps Engineer)
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, List, Callable

import torch
import numpy as np

from .config import ExportConfig, ModelInputSpec

# 配置日志
logger = logging.getLogger(__name__)


class TensorRTOptimizer:
    """
    TensorRT 模型优化器
    
    负责将 ONNX 模型转换为高性能的 TensorRT 引擎。
    
    Attributes:
        config: 导出配置
        input_spec: 模型输入规格
    
    Example:
        >>> config = ExportConfig(precision="fp16")
        >>> optimizer = TensorRTOptimizer(config)
        >>> engine_path = optimizer.optimize("model.onnx", "model.plan")
    """
    
    def __init__(self, config: ExportConfig):
        """
        初始化 TensorRT 优化器
        
        Args:
            config: 导出配置对象
        """
        self.config = config
        self.input_spec = ModelInputSpec()
        config.validate()
        
        # TensorRT 相关对象（延迟初始化）
        self._trt = None
        self._logger = None
    
    def _init_tensorrt(self):
        """延迟初始化 TensorRT"""
        if self._trt is None:
            try:
                import tensorrt as trt
                self._trt = trt
                self._logger = trt.Logger(trt.Logger.WARNING)
            except ImportError:
                raise ImportError(
                    "TensorRT 未安装。请安装 TensorRT: "
                    "pip install nvidia-tensorrt"
                )
    
    def _get_shape_profile(
        self,
        input_name: str
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """
        获取输入的动态形状配置
        
        Args:
            input_name: 输入名称
        
        Returns:
            Tuple[min_shape, opt_shape, max_shape]
        """
        min_batch = self.config.min_batch_size
        opt_batch = self.config.opt_batch_size
        max_batch = self.config.max_batch_size
        
        min_seq = self.config.min_seq_length
        opt_seq = self.config.opt_seq_length
        max_seq = self.config.max_seq_length
        
        # 所有输入都是 (batch, seq_len) 形状
        return (
            (min_batch, min_seq),  # min shape
            (opt_batch, opt_seq),  # optimal shape
            (max_batch, max_seq),  # max shape
        )
    
    def optimize(
        self,
        onnx_path: str,
        engine_path: str,
        calibration_data: Optional[Callable] = None
    ) -> str:
        """
        将 ONNX 模型优化为 TensorRT 引擎
        
        Args:
            onnx_path: ONNX 模型文件路径
            engine_path: TensorRT 引擎输出路径
            calibration_data: INT8 校准数据生成器（仅 INT8 模式需要）
        
        Returns:
            生成的 TensorRT 引擎文件路径
        
        Raises:
            FileNotFoundError: ONNX 文件不存在
            RuntimeError: TensorRT 构建失败
        """
        self._init_tensorrt()
        trt = self._trt
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 文件不存在: {onnx_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
        
        logger.info(f"开始 TensorRT 优化: {onnx_path} -> {engine_path}")
        logger.info(f"配置: precision={self.config.precision}, "
                   f"max_batch={self.config.max_batch_size}")
        
        # 创建构建器
        builder = trt.Builder(self._logger)
        
        # 创建网络（显式批次模式）
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        
        # 创建 ONNX 解析器
        parser = trt.OnnxParser(network, self._logger)
        
        # 解析 ONNX 模型
        logger.info("解析 ONNX 模型...")
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                error_msgs = []
                for i in range(parser.num_errors):
                    error_msgs.append(str(parser.get_error(i)))
                raise RuntimeError(
                    f"ONNX 解析失败:\n" + "\n".join(error_msgs)
                )
        
        logger.info(f"网络输入数: {network.num_inputs}, 输出数: {network.num_outputs}")
        
        # 创建构建配置
        build_config = builder.create_builder_config()
        
        # 设置工作空间
        workspace_bytes = self.config.workspace_size_gb << 30
        build_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_bytes
        )
        logger.info(f"工作空间大小: {self.config.workspace_size_gb} GB")
        
        # 设置精度
        self._configure_precision(build_config, calibration_data)
        
        # 配置动态形状
        if self.config.enable_dynamic_shapes:
            self._configure_dynamic_shapes(builder, build_config, network)
        
        # 构建引擎
        logger.info("构建 TensorRT 引擎（这可能需要几分钟）...")
        serialized_engine = builder.build_serialized_network(network, build_config)
        
        if serialized_engine is None:
            raise RuntimeError("TensorRT 引擎构建失败")
        
        # 保存引擎
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        logger.info(f"TensorRT 引擎保存成功: {engine_path} ({engine_size_mb:.2f} MB)")
        
        return engine_path
    
    def _configure_precision(
        self,
        build_config,
        calibration_data: Optional[Callable] = None
    ) -> None:
        """
        配置推理精度
        
        Args:
            build_config: TensorRT 构建配置
            calibration_data: INT8 校准数据生成器
        """
        trt = self._trt
        
        if self.config.precision == "fp16":
            if not trt.Builder(self._logger).platform_has_fast_fp16:
                logger.warning("当前平台不支持快速 FP16，将回退到 FP32")
            else:
                build_config.set_flag(trt.BuilderFlag.FP16)
                logger.info("启用 FP16 精度")
        
        elif self.config.precision == "int8":
            if not trt.Builder(self._logger).platform_has_fast_int8:
                logger.warning("当前平台不支持快速 INT8，将回退到 FP32")
            else:
                build_config.set_flag(trt.BuilderFlag.INT8)
                
                if calibration_data is not None:
                    calibrator = Int8Calibrator(calibration_data, self.config)
                    build_config.int8_calibrator = calibrator
                    logger.info("启用 INT8 精度，使用校准数据")
                else:
                    logger.warning("INT8 模式但未提供校准数据，精度可能受影响")
        
        else:
            logger.info("使用 FP32 精度")
    
    def _configure_dynamic_shapes(
        self,
        builder,
        build_config,
        network
    ) -> None:
        """
        配置动态形状优化配置
        
        Args:
            builder: TensorRT 构建器
            build_config: 构建配置
            network: TensorRT 网络
        """
        trt = self._trt
        
        profile = builder.create_optimization_profile()
        
        # 为每个输入配置动态形状
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_name = input_tensor.name
            
            min_shape, opt_shape, max_shape = self._get_shape_profile(input_name)
            
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            logger.info(f"输入 '{input_name}' 形状配置: "
                       f"min={min_shape}, opt={opt_shape}, max={max_shape}")
        
        build_config.add_optimization_profile(profile)
    
    def get_engine_info(self, engine_path: str) -> Dict[str, Any]:
        """
        获取 TensorRT 引擎信息
        
        Args:
            engine_path: 引擎文件路径
        
        Returns:
            引擎信息字典
        """
        self._init_tensorrt()
        trt = self._trt
        
        try:
            runtime = trt.Runtime(self._logger)
            
            with open(engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            
            if engine is None:
                return {"error": "无法加载引擎"}
            
            info = {
                "num_bindings": engine.num_bindings,
                "max_batch_size": engine.max_batch_size,
                "file_size_mb": os.path.getsize(engine_path) / (1024 * 1024),
                "bindings": [],
            }
            
            for i in range(engine.num_bindings):
                binding_info = {
                    "name": engine.get_binding_name(i),
                    "shape": list(engine.get_binding_shape(i)),
                    "dtype": str(engine.get_binding_dtype(i)),
                    "is_input": engine.binding_is_input(i),
                }
                info["bindings"].append(binding_info)
            
            return info
            
        except Exception as e:
            logger.error(f"获取引擎信息失败: {e}")
            return {"error": str(e)}


class Int8Calibrator:
    """
    INT8 量化校准器
    
    使用校准数据集来确定 INT8 量化的最优缩放因子。
    """
    
    def __init__(
        self,
        data_generator: Callable,
        config: ExportConfig,
        cache_file: str = "calibration.cache"
    ):
        """
        初始化校准器
        
        Args:
            data_generator: 生成校准数据的函数
            config: 导出配置
            cache_file: 校准缓存文件路径
        """
        self.data_generator = data_generator
        self.config = config
        self.cache_file = cache_file
        self.batch_idx = 0
        self.max_batches = 100  # 校准批次数
        
        # 生成校准数据
        self.calibration_data = list(data_generator())
    
    def get_batch_size(self) -> int:
        """返回校准批次大小"""
        return self.config.opt_batch_size
    
    def get_batch(self, names: List[str]) -> Optional[List]:
        """
        获取下一批校准数据
        
        Args:
            names: 输入名称列表
        
        Returns:
            包含各输入数据指针的列表，或 None 表示完成
        """
        if self.batch_idx >= min(len(self.calibration_data), self.max_batches):
            return None
        
        batch = self.calibration_data[self.batch_idx]
        self.batch_idx += 1
        
        # 返回 GPU 内存指针
        return [data.data_ptr() for data in batch.values()]
    
    def read_calibration_cache(self) -> Optional[bytes]:
        """读取校准缓存"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache: bytes) -> None:
        """写入校准缓存"""
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    config: Optional[ExportConfig] = None
) -> str:
    """
    便捷函数：构建 TensorRT 引擎
    
    Args:
        onnx_path: ONNX 模型路径
        engine_path: TensorRT 引擎输出路径
        config: 导出配置，如果为 None 则使用默认配置
    
    Returns:
        生成的 TensorRT 引擎文件路径
    
    Example:
        >>> engine_path = build_trt_engine("model.onnx", "model.plan")
    """
    if config is None:
        config = ExportConfig()
    
    optimizer = TensorRTOptimizer(config)
    return optimizer.optimize(onnx_path, engine_path)


class TensorRTInferenceEngine:
    """
    TensorRT 推理引擎
    
    加载和执行 TensorRT 引擎进行推理。
    
    Example:
        >>> engine = TensorRTInferenceEngine("model.plan")
        >>> outputs = engine.infer(inputs)
    """
    
    def __init__(self, engine_path: str):
        """
        初始化推理引擎
        
        Args:
            engine_path: TensorRT 引擎文件路径
        """
        self.engine_path = engine_path
        self._engine = None
        self._context = None
        self._stream = None
        self._bindings = None
        self._input_names = []
        self._output_names = []
        
        self._load_engine()
    
    def _load_engine(self) -> None:
        """加载 TensorRT 引擎"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as e:
            raise ImportError(f"缺少依赖: {e}")
        
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        with open(self.engine_path, 'rb') as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        
        if self._engine is None:
            raise RuntimeError(f"无法加载引擎: {self.engine_path}")
        
        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()
        
        # 收集输入输出信息
        self._bindings = []
        for i in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(i)
            if self._engine.binding_is_input(i):
                self._input_names.append(name)
            else:
                self._output_names.append(name)
    
    def infer(
        self,
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        执行推理
        
        Args:
            inputs: 输入数据字典，键为输入名称
        
        Returns:
            输出数据字典，键为输出名称
        """
        import pycuda.driver as cuda
        
        # 分配设备内存并拷贝输入
        device_inputs = {}
        for name, data in inputs.items():
            device_mem = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod_async(device_mem, data, self._stream)
            device_inputs[name] = device_mem
        
        # 分配输出内存
        outputs = {}
        device_outputs = {}
        for i in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(i)
            if not self._engine.binding_is_input(i):
                shape = self._engine.get_binding_shape(i)
                dtype = self._engine.get_binding_dtype(i)
                
                # 获取 numpy dtype
                if dtype == self._trt.float32:
                    np_dtype = np.float32
                elif dtype == self._trt.int64:
                    np_dtype = np.int64
                else:
                    np_dtype = np.float32
                
                output = np.empty(shape, dtype=np_dtype)
                device_mem = cuda.mem_alloc(output.nbytes)
                
                outputs[name] = output
                device_outputs[name] = device_mem
        
        # 构建绑定列表
        bindings = []
        for i in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(i)
            if name in device_inputs:
                bindings.append(int(device_inputs[name]))
            else:
                bindings.append(int(device_outputs[name]))
        
        # 执行推理
        self._context.execute_async_v2(
            bindings=bindings,
            stream_handle=self._stream.handle
        )
        
        # 拷贝输出到主机
        for name, device_mem in device_outputs.items():
            cuda.memcpy_dtoh_async(outputs[name], device_mem, self._stream)
        
        # 同步
        self._stream.synchronize()
        
        # 释放设备内存
        for mem in device_inputs.values():
            mem.free()
        for mem in device_outputs.values():
            mem.free()
        
        return outputs
    
    def __del__(self):
        """清理资源"""
        if self._context is not None:
            del self._context
        if self._engine is not None:
            del self._engine


def create_calibration_data_generator(
    num_samples: int = 100,
    batch_size: int = 32,
    seq_length: int = 512,
    codebook_sizes: Tuple[int, int, int] = (1024, 4096, 16384)
) -> Callable:
    """
    创建 INT8 校准数据生成器
    
    Args:
        num_samples: 样本数量
        batch_size: 批次大小
        seq_length: 序列长度
        codebook_sizes: 码本大小
    
    Returns:
        生成校准数据的函数
    """
    l1_size, l2_size, l3_size = codebook_sizes
    
    def generator():
        for _ in range(num_samples):
            batch = {
                "encoder_l1_ids": torch.randint(
                    0, l1_size, (batch_size, seq_length),
                    dtype=torch.long, device="cuda"
                ),
                "encoder_l2_ids": torch.randint(
                    0, l2_size, (batch_size, seq_length),
                    dtype=torch.long, device="cuda"
                ),
                "encoder_l3_ids": torch.randint(
                    0, l3_size, (batch_size, seq_length),
                    dtype=torch.long, device="cuda"
                ),
                "encoder_positions": torch.arange(
                    seq_length, dtype=torch.long, device="cuda"
                ).unsqueeze(0).expand(batch_size, -1),
                "encoder_token_types": torch.zeros(
                    batch_size, seq_length,
                    dtype=torch.long, device="cuda"
                ),
                "encoder_mask": torch.ones(
                    batch_size, seq_length,
                    dtype=torch.float32, device="cuda"
                ),
            }
            yield batch
    
    return generator

