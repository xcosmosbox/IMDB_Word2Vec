"""
TensorRT 优化模块单元测试

Author: Person F (MLOps Engineer)
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

import torch
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serving.config import ExportConfig
from serving.optimize_trt import (
    TensorRTOptimizer,
    build_trt_engine,
    Int8Calibrator,
    create_calibration_data_generator,
)


class TestTensorRTOptimizer:
    """TensorRTOptimizer 类测试"""
    
    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return ExportConfig(
            model_name="test_model",
            precision="fp16",
            max_batch_size=32,
            max_seq_length=512,
            workspace_size_gb=2,
        )
    
    @pytest.fixture
    def optimizer(self, config):
        """创建优化器实例"""
        return TensorRTOptimizer(config)
    
    def test_init(self, config):
        """测试初始化"""
        optimizer = TensorRTOptimizer(config)
        
        assert optimizer.config == config
        assert optimizer._trt is None  # 延迟初始化
        assert optimizer._logger is None
    
    def test_init_validates_config(self):
        """测试初始化时验证配置"""
        invalid_config = ExportConfig(precision="invalid")
        
        with pytest.raises(ValueError):
            TensorRTOptimizer(invalid_config)
    
    def test_get_shape_profile(self, optimizer):
        """测试获取形状配置"""
        min_shape, opt_shape, max_shape = optimizer._get_shape_profile("encoder_l1_ids")
        
        assert min_shape == (
            optimizer.config.min_batch_size,
            optimizer.config.min_seq_length
        )
        assert opt_shape == (
            optimizer.config.opt_batch_size,
            optimizer.config.opt_seq_length
        )
        assert max_shape == (
            optimizer.config.max_batch_size,
            optimizer.config.max_seq_length
        )
    
    def test_optimize_file_not_found(self, optimizer):
        """测试 ONNX 文件不存在"""
        with pytest.raises(FileNotFoundError, match="ONNX 文件不存在"):
            optimizer.optimize("nonexistent.onnx", "output.plan")
    
    @patch('serving.optimize_trt.TensorRTOptimizer._init_tensorrt')
    def test_optimize_creates_output_directory(self, mock_init, optimizer):
        """测试优化时创建输出目录"""
        # 创建临时 ONNX 文件
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            with open(onnx_path, 'wb') as f:
                f.write(b'dummy content')
            
            output_path = os.path.join(tmpdir, "subdir", "model.plan")
            
            # 模拟 TensorRT 初始化失败
            mock_init.side_effect = ImportError("TensorRT not available")
            
            with pytest.raises(ImportError):
                optimizer.optimize(onnx_path, output_path)
            
            # 输出目录应该被创建
            assert os.path.exists(os.path.dirname(output_path))


class TestInt8Calibrator:
    """Int8Calibrator 类测试"""
    
    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return ExportConfig(
            precision="int8",
            opt_batch_size=8,
        )
    
    def test_init(self, config):
        """测试初始化"""
        def data_gen():
            yield {"input": torch.randn(8, 512)}
        
        calibrator = Int8Calibrator(data_gen, config)
        
        assert calibrator.config == config
        assert calibrator.batch_idx == 0
    
    def test_get_batch_size(self, config):
        """测试获取批次大小"""
        def data_gen():
            yield {"input": torch.randn(8, 512)}
        
        calibrator = Int8Calibrator(data_gen, config)
        
        assert calibrator.get_batch_size() == config.opt_batch_size
    
    def test_get_batch_returns_none_after_max_batches(self, config):
        """测试超过最大批次后返回 None"""
        def data_gen():
            for _ in range(200):  # 生成足够多的数据
                yield {"input": torch.randn(8, 512)}
        
        calibrator = Int8Calibrator(data_gen, config)
        calibrator.max_batches = 5
        
        # 消费所有批次
        for _ in range(5):
            calibrator.get_batch(["input"])
        
        # 应该返回 None
        result = calibrator.get_batch(["input"])
        assert result is None
    
    def test_cache_file_operations(self, config):
        """测试缓存文件操作"""
        def data_gen():
            yield {"input": torch.randn(8, 512)}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "calibration.cache")
            calibrator = Int8Calibrator(data_gen, config, cache_file)
            
            # 初始时缓存不存在
            assert calibrator.read_calibration_cache() is None
            
            # 写入缓存
            test_data = b"test calibration data"
            calibrator.write_calibration_cache(test_data)
            
            # 读取缓存
            assert calibrator.read_calibration_cache() == test_data


class TestBuildTrtEngine:
    """build_trt_engine 便捷函数测试"""
    
    def test_with_default_config(self):
        """测试使用默认配置"""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            engine_path = os.path.join(tmpdir, "model.plan")
            
            # 创建空的 ONNX 文件
            with open(onnx_path, 'wb') as f:
                f.write(b'dummy')
            
            # 模拟 TensorRT 不可用
            with patch.object(TensorRTOptimizer, 'optimize') as mock_optimize:
                mock_optimize.return_value = engine_path
                
                result = build_trt_engine(onnx_path, engine_path)
                
                assert result == engine_path
                mock_optimize.assert_called_once()
    
    def test_with_custom_config(self):
        """测试使用自定义配置"""
        config = ExportConfig(precision="fp16")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            engine_path = os.path.join(tmpdir, "model.plan")
            
            with open(onnx_path, 'wb') as f:
                f.write(b'dummy')
            
            with patch.object(TensorRTOptimizer, 'optimize') as mock_optimize:
                mock_optimize.return_value = engine_path
                
                result = build_trt_engine(onnx_path, engine_path, config)
                
                assert result == engine_path


class TestCreateCalibrationDataGenerator:
    """create_calibration_data_generator 函数测试"""
    
    def test_returns_generator(self):
        """测试返回生成器函数"""
        generator = create_calibration_data_generator(
            num_samples=10,
            batch_size=4,
            seq_length=32,
        )
        
        assert callable(generator)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
    def test_generator_yields_correct_shapes(self):
        """测试生成器产生正确形状的数据"""
        num_samples = 5
        batch_size = 4
        seq_length = 32
        
        generator = create_calibration_data_generator(
            num_samples=num_samples,
            batch_size=batch_size,
            seq_length=seq_length,
        )
        
        batches = list(generator())
        
        assert len(batches) == num_samples
        
        for batch in batches:
            assert isinstance(batch, dict)
            assert batch["encoder_l1_ids"].shape == (batch_size, seq_length)
            assert batch["encoder_mask"].shape == (batch_size, seq_length)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
    def test_generator_data_on_cuda(self):
        """测试生成的数据在 CUDA 设备上"""
        generator = create_calibration_data_generator(num_samples=1)
        
        for batch in generator():
            for tensor in batch.values():
                assert tensor.device.type == "cuda"


class TestTensorRTOptimizerGetEngineInfo:
    """TensorRTOptimizer.get_engine_info 方法测试"""
    
    def test_nonexistent_file(self):
        """测试不存在的文件"""
        config = ExportConfig()
        optimizer = TensorRTOptimizer(config)
        
        # 模拟 TensorRT 初始化
        with patch.object(optimizer, '_init_tensorrt'):
            optimizer._trt = MagicMock()
            optimizer._logger = MagicMock()
            
            # 模拟文件读取失败
            with patch('builtins.open', side_effect=FileNotFoundError):
                info = optimizer.get_engine_info("nonexistent.plan")
                
                assert isinstance(info, dict)
                assert "error" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

