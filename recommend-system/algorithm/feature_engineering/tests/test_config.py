"""
配置类单元测试
"""

import os
import tempfile
import pytest

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from algorithm.feature_engineering.config import FeatureConfig


class TestFeatureConfig:
    """FeatureConfig 测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = FeatureConfig()
        
        # 检查默认值
        assert config.max_seq_length == 1024
        assert config.vocab_size == 500000
        assert config.min_token_freq == 5
        assert config.pad_token == "[PAD]"
        assert config.cls_token == "[CLS]"
        assert config.sep_token == "[SEP]"
        assert config.mask_token == "[MASK]"
        assert config.unk_token == "[UNK]"
    
    def test_special_tokens(self):
        """测试特殊 Token"""
        config = FeatureConfig()
        
        special_tokens = config.get_special_tokens()
        assert len(special_tokens) == 5
        assert "[PAD]" in special_tokens
        assert "[CLS]" in special_tokens
        assert "[SEP]" in special_tokens
        assert "[MASK]" in special_tokens
        assert "[UNK]" in special_tokens
    
    def test_special_token_ids(self):
        """测试特殊 Token ID"""
        config = FeatureConfig()
        
        ids = config.get_special_token_ids()
        assert ids["[PAD]"] == 0
        assert ids["[CLS]"] == 1
        assert ids["[SEP]"] == 2
        assert ids["[MASK]"] == 3
        assert ids["[UNK]"] == 4
    
    def test_time_bucket(self):
        """测试时间分桶"""
        config = FeatureConfig()
        
        # 测试各时间段
        assert config.get_time_bucket(3) == "night"
        assert config.get_time_bucket(8) == "morning"
        assert config.get_time_bucket(14) == "afternoon"
        assert config.get_time_bucket(20) == "evening"
    
    def test_action_weight(self):
        """测试行为权重"""
        config = FeatureConfig()
        
        assert config.get_action_weight("click") == 1.0
        assert config.get_action_weight("buy") == 5.0
        assert config.get_action_weight("view") == 0.1
        assert config.get_action_weight("unknown_action") == 0.0
    
    def test_validate(self):
        """测试配置验证"""
        config = FeatureConfig()
        assert config.validate() == True
    
    def test_validate_invalid_seq_length(self):
        """测试无效序列长度验证"""
        config = FeatureConfig()
        config.max_seq_length = 0
        
        with pytest.raises(ValueError):
            config.validate()
    
    def test_validate_invalid_vocab_size(self):
        """测试无效词表大小验证"""
        config = FeatureConfig()
        config.vocab_size = 3  # 小于 special_token_count
        
        with pytest.raises(ValueError):
            config.validate()
    
    def test_save_and_load(self):
        """测试保存和加载"""
        config = FeatureConfig()
        config.max_seq_length = 512
        config.vocab_size = 100000
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # 保存
            config.save(temp_path)
            assert os.path.exists(temp_path)
            
            # 加载
            loaded_config = FeatureConfig.load(temp_path)
            
            # 验证
            assert loaded_config.max_seq_length == 512
            assert loaded_config.vocab_size == 100000
            assert loaded_config.pad_token == "[PAD]"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_repr(self):
        """测试字符串表示"""
        config = FeatureConfig()
        repr_str = repr(config)
        
        assert "FeatureConfig" in repr_str
        assert "max_seq_length" in repr_str
        assert "vocab_size" in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

