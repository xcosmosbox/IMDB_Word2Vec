"""
配置类单元测试

Author: Person F (MLOps Engineer)
"""

import pytest
from dataclasses import asdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serving.config import (
    ExportConfig,
    TritonConfig,
    BenchmarkConfig,
    ModelInputSpec,
    DeploymentConfig,
    Precision,
    InstanceKind,
)


class TestExportConfig:
    """ExportConfig 配置类测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = ExportConfig()
        
        assert config.model_name == "ugt_recommend"
        assert config.precision == "fp16"
        assert config.max_batch_size == 64
        assert config.max_seq_length == 1024
        assert config.target_latency_ms == 30.0
        assert config.opset_version == 17
        assert config.codebook_sizes == (1024, 4096, 16384)
    
    def test_custom_values(self):
        """测试自定义值"""
        config = ExportConfig(
            model_name="my_model",
            precision="int8",
            max_batch_size=128,
            max_seq_length=2048,
        )
        
        assert config.model_name == "my_model"
        assert config.precision == "int8"
        assert config.max_batch_size == 128
        assert config.max_seq_length == 2048
    
    def test_validate_valid_config(self):
        """测试有效配置验证"""
        config = ExportConfig(
            precision="fp16",
            max_batch_size=32,
            max_seq_length=512,
        )
        
        # 不应抛出异常
        config.validate()
    
    def test_validate_invalid_precision(self):
        """测试无效精度配置"""
        config = ExportConfig(precision="invalid")
        
        with pytest.raises(ValueError, match="precision 必须是"):
            config.validate()
    
    def test_validate_invalid_batch_size(self):
        """测试无效批次大小"""
        config = ExportConfig(max_batch_size=0)
        
        with pytest.raises(ValueError, match="max_batch_size 必须大于 0"):
            config.validate()
    
    def test_validate_invalid_seq_length(self):
        """测试无效序列长度"""
        config = ExportConfig(max_seq_length=-1)
        
        with pytest.raises(ValueError, match="max_seq_length 必须大于 0"):
            config.validate()
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = ExportConfig(model_name="test_model")
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert result["model_name"] == "test_model"
        assert "precision" in result
        assert "max_batch_size" in result


class TestTritonConfig:
    """TritonConfig 配置类测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = TritonConfig()
        
        assert config.platform == "tensorrt_plan"
        assert config.instance_count == 2
        assert config.preferred_batch_sizes == (8, 16, 32, 64)
        assert config.max_queue_delay_us == 100
        assert config.enable_dynamic_batching is True
    
    def test_custom_values(self):
        """测试自定义值"""
        config = TritonConfig(
            platform="onnxruntime_onnx",
            instance_count=4,
            gpus=(0, 1, 2, 3),
        )
        
        assert config.platform == "onnxruntime_onnx"
        assert config.instance_count == 4
        assert config.gpus == (0, 1, 2, 3)
    
    def test_validate_valid_config(self):
        """测试有效配置验证"""
        config = TritonConfig()
        config.validate()  # 不应抛出异常
    
    def test_validate_invalid_platform(self):
        """测试无效平台配置"""
        config = TritonConfig(platform="invalid_platform")
        
        with pytest.raises(ValueError, match="platform 必须是"):
            config.validate()
    
    def test_validate_invalid_instance_count(self):
        """测试无效实例数量"""
        config = TritonConfig(instance_count=0)
        
        with pytest.raises(ValueError, match="instance_count 必须大于 0"):
            config.validate()


class TestBenchmarkConfig:
    """BenchmarkConfig 配置类测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = BenchmarkConfig()
        
        assert config.triton_url == "localhost:8001"
        assert config.num_warmup_requests == 100
        assert config.num_requests == 10000
        assert config.concurrency == 1
    
    def test_custom_values(self):
        """测试自定义值"""
        config = BenchmarkConfig(
            triton_url="192.168.1.100:8001",
            num_requests=5000,
            concurrency=8,
        )
        
        assert config.triton_url == "192.168.1.100:8001"
        assert config.num_requests == 5000
        assert config.concurrency == 8
    
    def test_validate_valid_config(self):
        """测试有效配置验证"""
        config = BenchmarkConfig()
        config.validate()  # 不应抛出异常
    
    def test_validate_invalid_num_requests(self):
        """测试无效请求数量"""
        config = BenchmarkConfig(num_requests=0)
        
        with pytest.raises(ValueError, match="num_requests 必须大于 0"):
            config.validate()
    
    def test_validate_invalid_concurrency(self):
        """测试无效并发数"""
        config = BenchmarkConfig(concurrency=0)
        
        with pytest.raises(ValueError, match="concurrency 必须大于 0"):
            config.validate()


class TestModelInputSpec:
    """ModelInputSpec 配置类测试"""
    
    def test_default_values(self):
        """测试默认值"""
        spec = ModelInputSpec()
        
        assert spec.encoder_l1_ids == "encoder_l1_ids"
        assert spec.encoder_mask == "encoder_mask"
        assert spec.recommendations == "recommendations"
        assert spec.scores == "scores"
    
    def test_get_input_names(self):
        """测试获取输入名称列表"""
        names = ModelInputSpec.get_input_names()
        
        assert isinstance(names, list)
        assert len(names) == 6
        assert "encoder_l1_ids" in names
        assert "encoder_l2_ids" in names
        assert "encoder_l3_ids" in names
        assert "encoder_positions" in names
        assert "encoder_token_types" in names
        assert "encoder_mask" in names
    
    def test_get_output_names(self):
        """测试获取输出名称列表"""
        names = ModelInputSpec.get_output_names()
        
        assert isinstance(names, list)
        assert len(names) == 2
        assert "recommendations" in names
        assert "scores" in names


class TestDeploymentConfig:
    """DeploymentConfig 配置类测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = DeploymentConfig()
        
        assert isinstance(config.export_config, ExportConfig)
        assert isinstance(config.triton_config, TritonConfig)
        assert isinstance(config.benchmark_config, BenchmarkConfig)
        assert isinstance(config.input_spec, ModelInputSpec)
    
    def test_validate_all(self):
        """测试验证所有子配置"""
        config = DeploymentConfig()
        config.validate_all()  # 不应抛出异常
    
    def test_custom_sub_configs(self):
        """测试自定义子配置"""
        export_config = ExportConfig(model_name="custom_model")
        triton_config = TritonConfig(instance_count=4)
        
        config = DeploymentConfig(
            export_config=export_config,
            triton_config=triton_config,
        )
        
        assert config.export_config.model_name == "custom_model"
        assert config.triton_config.instance_count == 4


class TestPrecisionEnum:
    """Precision 枚举测试"""
    
    def test_enum_values(self):
        """测试枚举值"""
        assert Precision.FP32.value == "fp32"
        assert Precision.FP16.value == "fp16"
        assert Precision.INT8.value == "int8"


class TestInstanceKindEnum:
    """InstanceKind 枚举测试"""
    
    def test_enum_values(self):
        """测试枚举值"""
        assert InstanceKind.GPU.value == "KIND_GPU"
        assert InstanceKind.CPU.value == "KIND_CPU"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

