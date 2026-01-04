"""
Triton 配置生成模块单元测试

Author: Person F (MLOps Engineer)
"""

import pytest
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serving.config import ExportConfig, TritonConfig
from serving.triton_config import (
    TritonConfigGenerator,
    generate_triton_config,
    setup_model_repository,
    TritonModelValidator,
)


class TestTritonConfigGenerator:
    """TritonConfigGenerator 类测试"""
    
    @pytest.fixture
    def export_config(self):
        """创建导出配置"""
        return ExportConfig(
            model_name="test_model",
            max_batch_size=64,
            num_recommendations=50,
        )
    
    @pytest.fixture
    def triton_config(self):
        """创建 Triton 配置"""
        return TritonConfig(
            platform="tensorrt_plan",
            instance_count=2,
            preferred_batch_sizes=(8, 16, 32),
            gpus=(0, 1),
        )
    
    @pytest.fixture
    def generator(self, export_config, triton_config):
        """创建配置生成器"""
        return TritonConfigGenerator(export_config, triton_config)
    
    def test_init(self, export_config, triton_config):
        """测试初始化"""
        generator = TritonConfigGenerator(export_config, triton_config)
        
        assert generator.export_config == export_config
        assert generator.triton_config == triton_config
    
    def test_init_with_default_triton_config(self, export_config):
        """测试使用默认 Triton 配置"""
        generator = TritonConfigGenerator(export_config)
        
        assert generator.triton_config is not None
        assert isinstance(generator.triton_config, TritonConfig)
    
    def test_generate_config_pbtxt_content(self, generator):
        """测试生成的配置内容"""
        config_content = generator._generate_config_pbtxt()
        
        # 检查基本字段
        assert 'name: "test_model"' in config_content
        assert 'platform: "tensorrt_plan"' in config_content
        assert 'max_batch_size: 64' in config_content
        
        # 检查输入配置
        assert 'input [' in config_content
        assert 'encoder_l1_ids' in config_content
        assert 'encoder_mask' in config_content
        
        # 检查输出配置
        assert 'output [' in config_content
        assert 'recommendations' in config_content
        assert 'scores' in config_content
        
        # 检查动态批处理
        assert 'dynamic_batching {' in config_content
        assert 'preferred_batch_size' in config_content
        
        # 检查实例组
        assert 'instance_group [' in config_content
        assert 'count: 2' in config_content
    
    def test_generate_creates_directory_structure(self, generator):
        """测试生成创建目录结构"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = generator.generate(tmpdir)
            
            # 检查目录结构
            model_dir = Path(tmpdir) / "test_model"
            assert model_dir.exists()
            
            version_dir = model_dir / "1"
            assert version_dir.exists()
            
            # 检查配置文件
            assert os.path.exists(config_path)
            assert config_path.endswith("config.pbtxt")
    
    def test_generate_copies_model_file(self, generator):
        """测试生成时复制模型文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建模拟模型文件
            model_file = os.path.join(tmpdir, "model.plan")
            with open(model_file, 'wb') as f:
                f.write(b'dummy model content')
            
            repo_dir = os.path.join(tmpdir, "repo")
            generator.generate(repo_dir, model_file)
            
            # 检查模型文件被复制
            copied_model = Path(repo_dir) / "test_model" / "1" / "model.plan"
            assert copied_model.exists()
    
    def test_generate_input_configs(self, generator):
        """测试生成输入配置"""
        inputs = generator._generate_input_configs()
        
        assert len(inputs) == 6  # 6 个输入
        
        # 检查所有输入
        input_str = '\n'.join(inputs)
        assert 'encoder_l1_ids' in input_str
        assert 'encoder_l2_ids' in input_str
        assert 'encoder_l3_ids' in input_str
        assert 'encoder_positions' in input_str
        assert 'encoder_token_types' in input_str
        assert 'encoder_mask' in input_str
        
        # 检查数据类型
        assert 'TYPE_INT64' in input_str
        assert 'TYPE_FP32' in input_str
    
    def test_generate_output_configs(self, generator):
        """测试生成输出配置"""
        outputs = generator._generate_output_configs()
        
        assert len(outputs) == 2  # 2 个输出
        
        output_str = '\n'.join(outputs)
        assert 'recommendations' in output_str
        assert 'scores' in output_str
        assert '50' in output_str  # num_recommendations
    
    def test_generate_ensemble_config(self, generator):
        """测试生成 Ensemble 配置"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = generator.generate_ensemble_config(
                tmpdir,
                encoder_name="ugt_encoder",
                decoder_name="ugt_decoder"
            )
            
            assert os.path.exists(config_path)
            
            # 读取并验证内容
            with open(config_path, 'r') as f:
                content = f.read()
            
            assert 'platform: "ensemble"' in content
            assert 'ugt_encoder' in content
            assert 'ugt_decoder' in content
            assert 'ensemble_scheduling' in content


class TestGenerateTritonConfig:
    """generate_triton_config 便捷函数测试"""
    
    def test_with_default_config(self):
        """测试使用默认配置"""
        config_str = generate_triton_config("my_model")
        
        assert 'name: "my_model"' in config_str
        assert 'input [' in config_str
        assert 'output [' in config_str
    
    def test_with_custom_config(self):
        """测试使用自定义配置"""
        export_config = ExportConfig(
            model_name="custom_model",
            max_batch_size=128,
        )
        
        config_str = generate_triton_config("custom_model", export_config)
        
        assert 'name: "custom_model"' in config_str
        assert 'max_batch_size: 128' in config_str


class TestSetupModelRepository:
    """setup_model_repository 便捷函数测试"""
    
    def test_creates_complete_structure(self):
        """测试创建完整结构"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建模拟模型文件
            model_file = os.path.join(tmpdir, "model.plan")
            with open(model_file, 'wb') as f:
                f.write(b'dummy')
            
            repo_dir = os.path.join(tmpdir, "repo")
            
            paths = setup_model_repository(repo_dir, model_file)
            
            assert "config_path" in paths
            assert "model_dir" in paths
            assert "model_repository" in paths
            
            assert os.path.exists(paths["config_path"])
    
    def test_with_custom_configs(self):
        """测试使用自定义配置"""
        export_config = ExportConfig(model_name="custom")
        triton_config = TritonConfig(instance_count=4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = os.path.join(tmpdir, "model.plan")
            with open(model_file, 'wb') as f:
                f.write(b'dummy')
            
            repo_dir = os.path.join(tmpdir, "repo")
            
            paths = setup_model_repository(
                repo_dir, model_file,
                export_config, triton_config
            )
            
            # 验证配置文件内容
            with open(paths["config_path"], 'r') as f:
                content = f.read()
            
            assert 'name: "custom"' in content
            assert 'count: 4' in content


class TestTritonModelValidator:
    """TritonModelValidator 类测试"""
    
    @pytest.fixture
    def valid_repository(self):
        """创建有效的模型仓库"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "test_model"
            version_dir = model_dir / "1"
            version_dir.mkdir(parents=True)
            
            # 创建配置文件
            config_path = model_dir / "config.pbtxt"
            config_path.write_text('name: "test_model"')
            
            # 创建模型文件
            model_file = version_dir / "model.plan"
            model_file.write_bytes(b'dummy model')
            
            yield tmpdir
    
    def test_validate_valid_model(self, valid_repository):
        """测试验证有效模型"""
        validator = TritonModelValidator(valid_repository)
        result = validator.validate("test_model")
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_nonexistent_model(self):
        """测试验证不存在的模型"""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = TritonModelValidator(tmpdir)
            result = validator.validate("nonexistent")
            
            assert result["valid"] is False
            assert any("不存在" in err for err in result["errors"])
    
    def test_validate_missing_config(self):
        """测试验证缺少配置文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "test_model"
            version_dir = model_dir / "1"
            version_dir.mkdir(parents=True)
            
            validator = TritonModelValidator(tmpdir)
            result = validator.validate("test_model")
            
            assert result["valid"] is False
            assert any("配置文件" in err for err in result["errors"])
    
    def test_validate_missing_version(self):
        """测试验证缺少版本目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "test_model"
            model_dir.mkdir()
            
            # 只创建配置文件，没有版本目录
            config_path = model_dir / "config.pbtxt"
            config_path.write_text('name: "test_model"')
            
            validator = TritonModelValidator(tmpdir)
            result = validator.validate("test_model")
            
            assert result["valid"] is False
            assert any("版本目录" in err for err in result["errors"])
    
    def test_validate_all(self, valid_repository):
        """测试验证所有模型"""
        validator = TritonModelValidator(valid_repository)
        results = validator.validate_all()
        
        assert "test_model" in results
        assert results["test_model"]["valid"] is True
    
    def test_validate_all_empty_repository(self):
        """测试验证空仓库"""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = TritonModelValidator(tmpdir)
            results = validator.validate_all()
            
            assert isinstance(results, dict)
            assert len(results) == 0
    
    def test_validate_all_nonexistent_repository(self):
        """测试验证不存在的仓库"""
        validator = TritonModelValidator("/nonexistent/path")
        results = validator.validate_all()
        
        assert "error" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

