"""
统一导出器模块单元测试

Author: Person F (MLOps Engineer)
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serving.config import ExportConfig, TritonConfig, BenchmarkConfig
from serving.exporter import ServingExporter, create_exporter
from serving.benchmark import BenchmarkResult


class MockUGTModel(nn.Module):
    """用于测试的模拟 UGT 模型"""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 512)
    
    def forward(self, *args, **kwargs):
        batch_size = args[0].shape[0] if args else 1
        return (
            torch.randint(0, 16384, (batch_size, 50, 3)),
            torch.randn(batch_size, 50)
        )
    
    def generate(self, **kwargs):
        return [[(1, 2, 3)] * 50]


class TestServingExporter:
    """ServingExporter 类测试"""
    
    @pytest.fixture
    def export_config(self):
        """创建导出配置"""
        return ExportConfig(model_name="test_model")
    
    @pytest.fixture
    def triton_config(self):
        """创建 Triton 配置"""
        return TritonConfig()
    
    @pytest.fixture
    def benchmark_config(self):
        """创建基准测试配置"""
        return BenchmarkConfig()
    
    @pytest.fixture
    def exporter(self, export_config, triton_config, benchmark_config):
        """创建导出器"""
        return ServingExporter(export_config, triton_config, benchmark_config)
    
    @pytest.fixture
    def mock_model(self):
        """创建模拟模型"""
        return MockUGTModel()
    
    def test_init(self, export_config, triton_config, benchmark_config):
        """测试初始化"""
        exporter = ServingExporter(export_config, triton_config, benchmark_config)
        
        assert exporter.export_config == export_config
        assert exporter.triton_config == triton_config
        assert exporter.benchmark_config == benchmark_config
    
    def test_init_with_defaults(self):
        """测试使用默认配置初始化"""
        exporter = ServingExporter()
        
        assert isinstance(exporter.export_config, ExportConfig)
        assert isinstance(exporter.triton_config, TritonConfig)
        assert isinstance(exporter.benchmark_config, BenchmarkConfig)
    
    def test_lazy_initialization(self, exporter):
        """测试延迟初始化"""
        # 初始时子模块为 None
        assert exporter._onnx_exporter is None
        assert exporter._trt_optimizer is None
        assert exporter._triton_generator is None
        assert exporter._benchmarker is None
        
        # 访问属性时创建
        _ = exporter.onnx_exporter
        assert exporter._onnx_exporter is not None
    
    @patch('serving.exporter.ONNXExporter')
    @patch('serving.exporter.wrap_model_for_export')
    def test_export_onnx(self, mock_wrap, mock_exporter_class, exporter, mock_model):
        """测试 ONNX 导出"""
        mock_exporter = Mock()
        mock_exporter.export.return_value = "/path/to/model.onnx"
        mock_exporter_class.return_value = mock_exporter
        mock_wrap.return_value = mock_model
        
        result = exporter.export_onnx(mock_model, "/path/to/model.onnx")
        
        assert result == "/path/to/model.onnx"
        mock_exporter.export.assert_called_once()
    
    @patch('serving.exporter.TensorRTOptimizer')
    def test_optimize_tensorrt(self, mock_optimizer_class, exporter):
        """测试 TensorRT 优化"""
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = "/path/to/model.plan"
        mock_optimizer_class.return_value = mock_optimizer
        
        result = exporter.optimize_tensorrt(
            "/path/to/model.onnx",
            "/path/to/model.plan"
        )
        
        assert result == "/path/to/model.plan"
        mock_optimizer.optimize.assert_called_once()
    
    @patch('serving.exporter.TritonConfigGenerator')
    def test_generate_triton_config(self, mock_generator_class, exporter):
        """测试 Triton 配置生成"""
        mock_generator = Mock()
        mock_generator.generate.return_value = "/path/to/config.pbtxt"
        mock_generator_class.return_value = mock_generator
        
        result = exporter.generate_triton_config("/path/to/model_repo")
        
        assert result == "/path/to/config.pbtxt"
        mock_generator.generate.assert_called_once()
    
    @patch('serving.exporter.TritonBenchmark')
    def test_benchmark(self, mock_benchmark_class, exporter):
        """测试性能基准测试"""
        mock_benchmark = Mock()
        mock_benchmark.run.return_value = BenchmarkResult(
            throughput=1000.0,
            latency_p50=5.0,
            latency_p90=8.0,
            latency_p99=12.0,
        )
        mock_benchmark_class.return_value = mock_benchmark
        
        result = exporter.benchmark("localhost:8001", "test_model", 10000)
        
        assert isinstance(result, dict)
        assert result["throughput"] == 1000.0
        assert result["latency_p99"] == 12.0
    
    @patch.object(ServingExporter, 'export_onnx')
    @patch.object(ServingExporter, 'optimize_tensorrt')
    @patch.object(ServingExporter, 'generate_triton_config')
    def test_deploy_full_pipeline(
        self, mock_triton, mock_trt, mock_onnx, exporter, mock_model
    ):
        """测试完整部署流水线"""
        mock_onnx.return_value = "/path/to/model.onnx"
        mock_trt.return_value = "/path/to/model.plan"
        mock_triton.return_value = "/path/to/config.pbtxt"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = exporter.deploy_full_pipeline(mock_model, tmpdir)
        
        assert "onnx_path" in result
        assert "engine_path" in result
        assert "config_path" in result
        
        mock_onnx.assert_called_once()
        mock_trt.assert_called_once()
        mock_triton.assert_called_once()
    
    @patch('serving.exporter.TritonModelValidator')
    def test_validate_deployment(self, mock_validator_class, exporter):
        """测试部署验证"""
        mock_validator = Mock()
        mock_validator.validate.return_value = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        mock_validator_class.return_value = mock_validator
        
        result = exporter.validate_deployment("/path/to/model_repo")
        
        assert result["valid"] is True
    
    def test_get_deployment_info(self, exporter):
        """测试获取部署信息"""
        info = exporter.get_deployment_info()
        
        assert isinstance(info, dict)
        assert "export_config" in info
        assert "triton_config" in info
        assert "benchmark_config" in info
    
    def test_config_update_resets_components(self, exporter, mock_model):
        """测试更新配置时重置组件"""
        # 先访问组件以初始化
        _ = exporter.onnx_exporter
        original_exporter = exporter._onnx_exporter
        
        # 使用新配置导出
        new_config = ExportConfig(model_name="new_model")
        
        with patch('serving.exporter.ONNXExporter') as mock_class:
            mock_class.return_value = Mock()
            mock_class.return_value.export.return_value = "test.onnx"
            
            with patch('serving.exporter.wrap_model_for_export', return_value=mock_model):
                exporter.export_onnx(mock_model, "test.onnx", new_config)
        
        # 配置应该更新
        assert exporter.export_config == new_config


class TestCreateExporter:
    """create_exporter 便捷函数测试"""
    
    def test_returns_serving_exporter(self):
        """测试返回 ServingExporter 实例"""
        exporter = create_exporter()
        
        assert isinstance(exporter, ServingExporter)
    
    def test_uses_default_values(self):
        """测试使用默认值"""
        exporter = create_exporter()
        
        assert exporter.export_config.model_name == "ugt_recommend"
        assert exporter.export_config.precision == "fp16"
        assert exporter.export_config.max_batch_size == 64
    
    def test_uses_custom_values(self):
        """测试使用自定义值"""
        exporter = create_exporter(
            model_name="my_model",
            precision="int8",
            max_batch_size=128,
            max_seq_length=2048,
        )
        
        assert exporter.export_config.model_name == "my_model"
        assert exporter.export_config.precision == "int8"
        assert exporter.export_config.max_batch_size == 128
        assert exporter.export_config.max_seq_length == 2048
    
    def test_tensorrt_platform_for_non_fp32(self):
        """测试非 FP32 精度使用 TensorRT 平台"""
        exporter = create_exporter(precision="fp16")
        
        assert exporter.triton_config.platform == "tensorrt_plan"
    
    def test_onnx_platform_for_fp32(self):
        """测试 FP32 精度使用 ONNX 平台"""
        exporter = create_exporter(precision="fp32")
        
        assert exporter.triton_config.platform == "onnxruntime_onnx"


class TestServingExporterInterfaceCompliance:
    """测试 ServingExporter 是否符合接口定义"""
    
    def test_has_export_onnx_method(self):
        """测试有 export_onnx 方法"""
        exporter = ServingExporter()
        
        assert hasattr(exporter, 'export_onnx')
        assert callable(exporter.export_onnx)
    
    def test_has_optimize_tensorrt_method(self):
        """测试有 optimize_tensorrt 方法"""
        exporter = ServingExporter()
        
        assert hasattr(exporter, 'optimize_tensorrt')
        assert callable(exporter.optimize_tensorrt)
    
    def test_has_generate_triton_config_method(self):
        """测试有 generate_triton_config 方法"""
        exporter = ServingExporter()
        
        assert hasattr(exporter, 'generate_triton_config')
        assert callable(exporter.generate_triton_config)
    
    def test_has_benchmark_method(self):
        """测试有 benchmark 方法"""
        exporter = ServingExporter()
        
        assert hasattr(exporter, 'benchmark')
        assert callable(exporter.benchmark)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

