"""
词表管理器单元测试
"""

import os
import json
import tempfile
import pytest

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from algorithm.feature_engineering.config import FeatureConfig
from algorithm.feature_engineering.vocabulary import Vocabulary


class TestVocabulary:
    """Vocabulary 测试类"""
    
    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return FeatureConfig()
    
    @pytest.fixture
    def vocab(self, config):
        """创建测试词表"""
        return Vocabulary(config)
    
    def test_init(self, vocab, config):
        """测试初始化"""
        # 检查特殊 Token 已初始化
        assert len(vocab) == config.special_token_count
        assert "[PAD]" in vocab
        assert "[CLS]" in vocab
        assert "[SEP]" in vocab
        assert "[MASK]" in vocab
        assert "[UNK]" in vocab
    
    def test_add_token(self, vocab):
        """测试添加 Token"""
        initial_size = len(vocab)
        
        token_id = vocab.add_token("ACTION_click")
        assert token_id == initial_size
        assert len(vocab) == initial_size + 1
        assert "ACTION_click" in vocab
    
    def test_add_duplicate_token(self, vocab):
        """测试添加重复 Token"""
        token_id1 = vocab.add_token("ACTION_click")
        token_id2 = vocab.add_token("ACTION_click")
        
        assert token_id1 == token_id2
    
    def test_add_tokens_batch(self, vocab):
        """测试批量添加 Token"""
        initial_size = len(vocab)
        tokens = ["ACTION_click", "ACTION_view", "ACTION_buy"]
        
        ids = vocab.add_tokens(tokens)
        
        assert len(ids) == 3
        assert len(vocab) == initial_size + 3
    
    def test_encode(self, vocab):
        """测试编码"""
        vocab.add_token("ACTION_click")
        
        # 编码已存在的 Token
        token_id = vocab.encode("ACTION_click")
        assert token_id >= vocab.config.special_token_count
        
        # 编码未知 Token
        unk_id = vocab.encode("UNKNOWN_TOKEN")
        assert unk_id == vocab.config.unk_token_id
    
    def test_encode_batch(self, vocab):
        """测试批量编码"""
        vocab.add_token("ACTION_click")
        vocab.add_token("ACTION_view")
        
        ids = vocab.encode_batch(["ACTION_click", "ACTION_view", "UNKNOWN"])
        
        assert len(ids) == 3
        assert ids[2] == vocab.config.unk_token_id
    
    def test_decode(self, vocab):
        """测试解码"""
        vocab.add_token("ACTION_click")
        token_id = vocab.encode("ACTION_click")
        
        # 解码已存在的 ID
        token = vocab.decode(token_id)
        assert token == "ACTION_click"
        
        # 解码未知 ID
        unk_token = vocab.decode(999999)
        assert unk_token == vocab.config.unk_token
    
    def test_decode_batch(self, vocab):
        """测试批量解码"""
        vocab.add_token("ACTION_click")
        vocab.add_token("ACTION_view")
        
        ids = vocab.encode_batch(["ACTION_click", "ACTION_view"])
        tokens = vocab.decode_batch(ids)
        
        assert tokens == ["ACTION_click", "ACTION_view"]
    
    def test_contains(self, vocab):
        """测试包含检查"""
        vocab.add_token("ACTION_click")
        
        assert vocab.contains("ACTION_click") == True
        assert vocab.contains("ACTION_unknown") == False
        assert "ACTION_click" in vocab
        assert "ACTION_unknown" not in vocab
    
    def test_build_from_counts(self, vocab):
        """测试从频率统计构建词表"""
        # 模拟频率统计
        vocab.token_counts.update({
            "ACTION_click": 100,
            "ACTION_view": 50,
            "ACTION_rare": 2,  # 低于阈值
        })
        
        added = vocab.build_from_counts(min_freq=5)
        
        assert added == 2  # 只添加了 click 和 view
        assert "ACTION_click" in vocab
        assert "ACTION_view" in vocab
        assert "ACTION_rare" not in vocab
    
    def test_build_from_data(self, vocab):
        """测试从数据文件构建词表"""
        # 创建临时数据文件
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False, encoding='utf-8'
        ) as f:
            for i in range(10):
                event = {
                    "action": "click",
                    "semantic_id": [1, 2, 3],
                    "timestamp": 1704067200 + i * 60,
                    "device": "mobile"
                }
                f.write(json.dumps(event) + '\n')
            temp_path = f.name
        
        try:
            vocab.build_from_data(temp_path, min_freq=1)
            
            assert "ACTION_click" in vocab
            assert "DEVICE_mobile" in vocab
        finally:
            os.remove(temp_path)
    
    def test_save_and_load(self, vocab):
        """测试保存和加载词表"""
        vocab.add_token("ACTION_click")
        vocab.add_token("ITEM_1_2_3")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # 保存
            vocab.save(temp_path)
            assert os.path.exists(temp_path)
            
            # 创建新词表并加载
            new_vocab = Vocabulary(vocab.config)
            new_vocab.load(temp_path)
            
            # 验证
            assert len(new_vocab) == len(vocab)
            assert "ACTION_click" in new_vocab
            assert "ITEM_1_2_3" in new_vocab
        finally:
            os.remove(temp_path)
    
    def test_get_action_tokens(self, vocab):
        """测试获取行为 Token"""
        vocab.add_token("ACTION_click")
        vocab.add_token("ACTION_view")
        vocab.add_token("ITEM_1_2_3")
        
        action_tokens = vocab.get_action_tokens()
        
        assert "ACTION_click" in action_tokens
        assert "ACTION_view" in action_tokens
        assert "ITEM_1_2_3" not in action_tokens
    
    def test_get_item_tokens(self, vocab):
        """测试获取物品 Token"""
        vocab.add_token("ACTION_click")
        vocab.add_token("ITEM_1_2_3")
        vocab.add_token("ITEM_4_5_6")
        
        item_tokens = vocab.get_item_tokens()
        
        assert "ITEM_1_2_3" in item_tokens
        assert "ITEM_4_5_6" in item_tokens
        assert "ACTION_click" not in item_tokens
    
    def test_vocab_full(self, config):
        """测试词表已满的情况"""
        config.vocab_size = 10  # 设置很小的词表
        vocab = Vocabulary(config)
        
        # 添加超过词表容量的 Token
        for i in range(20):
            vocab.add_token(f"TOKEN_{i}")
        
        # 检查词表大小不超过限制
        assert len(vocab) <= config.vocab_size
    
    def test_repr(self, vocab):
        """测试字符串表示"""
        vocab.add_token("ACTION_click")
        vocab.add_token("ITEM_1_2_3")
        
        repr_str = repr(vocab)
        
        assert "Vocabulary" in repr_str
        assert "size" in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

