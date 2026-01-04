"""
ONNX 导出模块单元测试

Author: Person F (MLOps Engineer)
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Tuple

import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serving.config import ExportConfig, ModelInputSpec
from serving.export_onnx import (
    ONNXExporter,
    export_to_onnx,
    ModelWrapper,
    wrap_model_for_export,
)


class MockUGTModel(nn.Module):
    """用于测试的模拟 UGT 模型"""
    
    def __init__(self, d_model: int = 512, num_recommendations: int = 50):
        super().__init__()
        self.d_model = d_model
        self.num_recommendations = num_recommendations
        self.linear = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        encoder_l1_ids: torch.Tensor,
        encoder_l2_ids: torch.Tensor,
        encoder_l3_ids: torch.Tensor,
        encoder_positions: torch.Tensor,
        encoder_token_types: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoder_l1_ids.shape[0]
        
        # 模拟推荐输出
        recommendations = torch.randint(
            0, 16384, (batch_size, self.num_recommendations, 3),
            dtype=torch.long, device=encoder_l1_ids.device
        )
        scores = torch.randn(
            batch_size, self.num_recommendations,
            dtype=torch.float32, device=encoder_l1_ids.device
        )
        
        return recommendations, scores
    
    def generate(self, **kwargs):
        """模拟生成方法"""
        return [[(1, 2, 3)] * self.num_recommendations]


class TestONNXExporter:
    """ONNXExporter 类测试"""
    
    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return ExportConfig(
            model_name="test_model",
            precision="fp16",
            max_batch_size=32,
            max_seq_length=512,
        )
    
    @pytest.fixture
    def exporter(self, config):
        """创建导出器实例"""
        return ONNXExporter(config)
    
    @pytest.fixture
    def mock_model(self):
        """创建模拟模型"""
        return MockUGTModel()
    
    def test_init(self, config):
        """测试初始化"""
        exporter = ONNXExporter(config)
        
        assert exporter.config == config
        assert isinstance(exporter.input_spec, ModelInputSpec)
    
    def test_init_validates_config(self):
        """测试初始化时验证配置"""
        invalid_config = ExportConfig(precision="invalid")
        
        with pytest.raises(ValueError):
            ONNXExporter(invalid_config)
    
    def test_create_example_inputs(self, exporter):
        """测试创建示例输入"""
        batch_size = 2
        seq_len = 64
        
        inputs = exporter._create_example_inputs(batch_size, seq_len)
        
        assert isinstance(inputs, dict)
        assert len(inputs) == 6  # 6 个输入
        
        # 检查形状
        assert inputs["encoder_l1_ids"].shape == (batch_size, seq_len)
        assert inputs["encoder_l2_ids"].shape == (batch_size, seq_len)
        assert inputs["encoder_l3_ids"].shape == (batch_size, seq_len)
        assert inputs["encoder_positions"].shape == (batch_size, seq_len)
        assert inputs["encoder_token_types"].shape == (batch_size, seq_len)
        assert inputs["encoder_mask"].shape == (batch_size, seq_len)
        
        # 检查数据类型
        assert inputs["encoder_l1_ids"].dtype == torch.long
        assert inputs["encoder_mask"].dtype == torch.float32
    
    def test_create_example_inputs_value_ranges(self, exporter):
        """测试示例输入的值范围"""
        inputs = exporter._create_example_inputs(1, 10)
        
        l1_size, l2_size, l3_size = exporter.config.codebook_sizes
        
        assert inputs["encoder_l1_ids"].max() < l1_size
        assert inputs["encoder_l2_ids"].max() < l2_size
        assert inputs["encoder_l3_ids"].max() < l3_size
    
    def test_get_dynamic_axes(self, exporter):
        """测试动态轴配置"""
        dynamic_axes = exporter._get_dynamic_axes()
        
        assert isinstance(dynamic_axes, dict)
        
        # 检查输入动态轴
        for name in ModelInputSpec.get_input_names():
            assert name in dynamic_axes
            assert dynamic_axes[name] == {0: "batch", 1: "seq_len"}
        
        # 检查输出动态轴
        assert "recommendations" in dynamic_axes
        assert "scores" in dynamic_axes
    
    def test_export_creates_file(self, exporter, mock_model):
        """测试导出创建文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.onnx")
            
            # 导出（跳过验证以避免依赖 onnxruntime）
            result_path = exporter.export(mock_model, save_path, verify=False)
            
            assert os.path.exists(result_path)
            assert result_path.endswith(".onnx")
    
    def test_export_creates_directory(self, exporter, mock_model):
        """测试导出自动创建目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "subdir", "model.onnx")
            
            result_path = exporter.export(mock_model, save_path, verify=False)
            
            assert os.path.exists(os.path.dirname(result_path))
    
    def test_export_sets_eval_mode(self, exporter, mock_model):
        """测试导出时模型设置为评估模式"""
        mock_model.train()  # 先设置为训练模式
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.onnx")
            exporter.export(mock_model, save_path, verify=False)
        
        # 导出后模型应该处于评估模式
        assert not mock_model.training


class TestModelWrapper:
    """ModelWrapper 类测试"""
    
    @pytest.fixture
    def mock_model(self):
        """创建模拟模型"""
        return MockUGTModel()
    
    def test_init(self, mock_model):
        """测试初始化"""
        wrapper = ModelWrapper(mock_model, num_recommendations=20)
        
        assert wrapper.model == mock_model
        assert wrapper.num_recommendations == 20
    
    def test_forward_output_shapes(self, mock_model):
        """测试前向传播输出形状"""
        wrapper = ModelWrapper(mock_model, num_recommendations=50)
        
        batch_size = 4
        seq_len = 32
        
        # 创建输入
        l1_ids = torch.randint(0, 1024, (batch_size, seq_len))
        l2_ids = torch.randint(0, 4096, (batch_size, seq_len))
        l3_ids = torch.randint(0, 16384, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        token_types = torch.zeros(batch_size, seq_len, dtype=torch.long)
        mask = torch.ones(batch_size, seq_len)
        
        # 前向传播
        recommendations, scores = wrapper(
            l1_ids, l2_ids, l3_ids, positions, token_types, mask
        )
        
        assert recommendations.shape == (batch_size, 50, 3)
        assert scores.shape == (batch_size, 50)
    
    def test_forward_output_types(self, mock_model):
        """测试前向传播输出类型"""
        wrapper = ModelWrapper(mock_model)
        
        l1_ids = torch.randint(0, 1024, (1, 10))
        l2_ids = torch.randint(0, 4096, (1, 10))
        l3_ids = torch.randint(0, 16384, (1, 10))
        positions = torch.arange(10).unsqueeze(0)
        token_types = torch.zeros(1, 10, dtype=torch.long)
        mask = torch.ones(1, 10)
        
        recommendations, scores = wrapper(
            l1_ids, l2_ids, l3_ids, positions, token_types, mask
        )
        
        assert recommendations.dtype == torch.long
        assert scores.dtype == torch.float32


class TestExportToOnnx:
    """export_to_onnx 便捷函数测试"""
    
    def test_with_default_config(self):
        """测试使用默认配置"""
        model = MockUGTModel()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.onnx")
            
            # 使用 mock 避免实际导出
            with patch.object(ONNXExporter, 'export', return_value=save_path):
                result = export_to_onnx(model, save_path)
                assert result == save_path
    
    def test_with_custom_config(self):
        """测试使用自定义配置"""
        model = MockUGTModel()
        config = ExportConfig(model_name="custom_model")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.onnx")
            
            with patch.object(ONNXExporter, 'export', return_value=save_path):
                result = export_to_onnx(model, save_path, config)
                assert result == save_path


class TestWrapModelForExport:
    """wrap_model_for_export 函数测试"""
    
    def test_returns_model_wrapper(self):
        """测试返回 ModelWrapper 实例"""
        model = MockUGTModel()
        
        wrapped = wrap_model_for_export(model, num_recommendations=30)
        
        assert isinstance(wrapped, ModelWrapper)
        assert wrapped.num_recommendations == 30
    
    def test_default_num_recommendations(self):
        """测试默认推荐数量"""
        model = MockUGTModel()
        
        wrapped = wrap_model_for_export(model)
        
        assert wrapped.num_recommendations == 50


class TestONNXExporterGetModelInfo:
    """ONNXExporter.get_model_info 方法测试"""
    
    def test_returns_empty_dict_without_onnx(self):
        """测试没有 onnx 库时返回空字典"""
        config = ExportConfig()
        exporter = ONNXExporter(config)
        
        with patch.dict('sys.modules', {'onnx': None}):
            # 模拟导入失败
            with patch('builtins.__import__', side_effect=ImportError):
                info = exporter.get_model_info("nonexistent.onnx")
                # 应该返回空字典或包含警告信息
                assert isinstance(info, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

