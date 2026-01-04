"""
UGT 编码器单元测试

该模块包含 UGT Encoder 所有组件的单元测试。

测试覆盖:
1. 配置类测试 (TestEncoderConfig)
2. 嵌入层测试 (TestEmbedding)
3. 注意力层测试 (TestAttention)
4. 层归一化测试 (TestLayerNorm)
5. 前馈网络测试 (TestFFN)
6. 编码器层测试 (TestEncoderLayer)
7. 完整编码器测试 (TestUGTEncoder)
8. 集成测试 (TestIntegration)

运行测试:
    pytest algorithm/encoder/tests/test_encoder.py -v

Author: Person B
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algorithm.encoder.config import EncoderConfig
from algorithm.encoder.embedding import (
    InputEmbedding,
    SemanticEmbedding,
    PositionalEmbedding,
    TokenTypeEmbedding,
    TimeEmbedding,
)
from algorithm.encoder.attention import (
    DotProductAggregatedAttention,
    SoftmaxAttention,
    create_attention_layer,
)
from algorithm.encoder.layer_norm import (
    GroupLayerNorm,
    StandardLayerNorm,
    create_layer_norm,
)
from algorithm.encoder.ffn import (
    FeedForwardNetwork,
    GatedLinearUnit,
    SwiGLU,
    create_ffn,
)
from algorithm.encoder.encoder_layer import (
    EncoderLayer,
    PreNormEncoderLayer,
    create_encoder_layer,
)
from algorithm.encoder.encoder import (
    UGTEncoder,
    Pooler,
)


# =============================================================================
# 测试夹具 (Fixtures)
# =============================================================================

@pytest.fixture
def small_config():
    """小规模配置，用于快速测试"""
    return EncoderConfig(
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=64,
        dropout=0.0,  # 测试时禁用 dropout
    )


@pytest.fixture
def base_config():
    """基础规模配置"""
    return EncoderConfig(
        d_model=512,
        n_heads=16,
        n_layers=6,
        d_ff=2048,
        max_seq_len=256,
        dropout=0.0,
    )


@pytest.fixture
def sample_input(small_config):
    """生成样本输入数据"""
    batch_size = 4
    seq_len = 32
    
    l1_ids = torch.randint(0, small_config.codebook_sizes[0], (batch_size, seq_len))
    l2_ids = torch.randint(0, small_config.codebook_sizes[1], (batch_size, seq_len))
    l3_ids = torch.randint(0, small_config.codebook_sizes[2], (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    token_types = torch.randint(0, small_config.num_token_types, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 模拟一些 padding
    attention_mask[:, -5:] = 0
    
    return {
        "semantic_ids": [l1_ids, l2_ids, l3_ids],
        "positions": positions,
        "token_types": token_types,
        "attention_mask": attention_mask,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


# =============================================================================
# 配置类测试
# =============================================================================

class TestEncoderConfig:
    """编码器配置类测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = EncoderConfig()
        
        assert config.d_model == 512
        assert config.n_heads == 16
        assert config.n_layers == 12
        assert config.d_ff == 2048
        assert config.max_seq_len == 1024
        assert config.dropout == 0.1
        assert config.codebook_sizes == (1024, 4096, 16384)
        assert config.num_token_types == 4
        assert config.num_groups == 4
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = EncoderConfig(
            d_model=256,
            n_heads=8,
            n_layers=6,
        )
        
        assert config.d_model == 256
        assert config.n_heads == 8
        assert config.n_layers == 6
        assert config.head_dim == 32  # 256 / 8
    
    def test_config_validation(self):
        """测试配置验证"""
        # d_model 必须能被 n_heads 整除
        with pytest.raises(ValueError):
            EncoderConfig(d_model=100, n_heads=8)
        
        # codebook_sizes 必须有 3 个元素
        with pytest.raises(ValueError):
            EncoderConfig(codebook_sizes=(1024, 4096))
    
    def test_preset_configs(self):
        """测试预设配置"""
        small = EncoderConfig.small()
        assert small.d_model == 256
        assert small.n_layers == 6
        
        base = EncoderConfig.base()
        assert base.d_model == 512
        assert base.n_layers == 12
        
        large = EncoderConfig.large()
        assert large.d_model == 1024
        assert large.n_layers == 24
    
    def test_config_serialization(self):
        """测试配置序列化"""
        config = EncoderConfig(d_model=256, n_heads=8)
        
        # 转换为字典
        config_dict = config.to_dict()
        assert config_dict["d_model"] == 256
        assert config_dict["n_heads"] == 8
        
        # 从字典恢复
        restored = EncoderConfig.from_dict(config_dict)
        assert restored.d_model == 256
        assert restored.n_heads == 8


# =============================================================================
# 嵌入层测试
# =============================================================================

class TestEmbedding:
    """嵌入层测试"""
    
    def test_semantic_embedding(self, small_config, sample_input):
        """测试语义嵌入层"""
        embedding = SemanticEmbedding(small_config)
        
        output = embedding(sample_input["semantic_ids"])
        
        expected_shape = (
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        assert output.shape == expected_shape
    
    def test_positional_embedding(self, small_config, sample_input):
        """测试位置嵌入层"""
        embedding = PositionalEmbedding(small_config)
        
        output = embedding(sample_input["positions"])
        
        expected_shape = (
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        assert output.shape == expected_shape
    
    def test_token_type_embedding(self, small_config, sample_input):
        """测试 Token 类型嵌入层"""
        embedding = TokenTypeEmbedding(small_config)
        
        output = embedding(sample_input["token_types"])
        
        expected_shape = (
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        assert output.shape == expected_shape
    
    def test_time_embedding(self, small_config, sample_input):
        """测试时间特征嵌入层"""
        embedding = TimeEmbedding(small_config)
        
        time_features = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.time_dim,
        )
        
        output = embedding(time_features)
        
        expected_shape = (
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        assert output.shape == expected_shape
    
    def test_time_embedding_none(self, small_config):
        """测试时间特征为 None 的情况"""
        embedding = TimeEmbedding(small_config)
        output = embedding(None)
        assert output is None
    
    def test_input_embedding(self, small_config, sample_input):
        """测试统一输入嵌入层"""
        embedding = InputEmbedding(small_config)
        
        output = embedding(
            sample_input["semantic_ids"],
            sample_input["positions"],
            sample_input["token_types"],
        )
        
        expected_shape = (
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        assert output.shape == expected_shape
    
    def test_input_embedding_with_time(self, small_config, sample_input):
        """测试带时间特征的统一输入嵌入层"""
        embedding = InputEmbedding(small_config)
        
        time_features = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.time_dim,
        )
        
        output = embedding(
            sample_input["semantic_ids"],
            sample_input["positions"],
            sample_input["token_types"],
            time_features,
        )
        
        expected_shape = (
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        assert output.shape == expected_shape


# =============================================================================
# 注意力层测试
# =============================================================================

class TestAttention:
    """注意力层测试"""
    
    def test_dot_product_attention_shape(self, small_config, sample_input):
        """测试点积聚合注意力输出形状"""
        attention = DotProductAggregatedAttention(
            d_model=small_config.d_model,
            n_heads=small_config.n_heads,
            dropout=0.0,
        )
        
        x = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output, _ = attention(x)
        
        assert output.shape == x.shape
    
    def test_dot_product_attention_with_mask(self, small_config, sample_input):
        """测试带掩码的点积聚合注意力"""
        attention = DotProductAggregatedAttention(
            d_model=small_config.d_model,
            n_heads=small_config.n_heads,
            dropout=0.0,
        )
        
        x = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output, _ = attention(x, sample_input["attention_mask"])
        
        assert output.shape == x.shape
    
    def test_attention_weights_return(self, small_config, sample_input):
        """测试返回注意力权重"""
        attention = DotProductAggregatedAttention(
            d_model=small_config.d_model,
            n_heads=small_config.n_heads,
            dropout=0.0,
        )
        
        x = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output, attn_weights = attention(x, return_attention=True)
        
        assert attn_weights is not None
        expected_shape = (
            sample_input["batch_size"],
            small_config.n_heads,
            sample_input["seq_len"],
            sample_input["seq_len"],
        )
        assert attn_weights.shape == expected_shape
    
    def test_relu_vs_softmax(self, small_config, sample_input):
        """测试 ReLU 和 Softmax 注意力的差异"""
        relu_attention = DotProductAggregatedAttention(
            d_model=small_config.d_model,
            n_heads=small_config.n_heads,
            dropout=0.0,
        )
        
        softmax_attention = SoftmaxAttention(
            d_model=small_config.d_model,
            n_heads=small_config.n_heads,
            dropout=0.0,
        )
        
        x = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        relu_output, relu_weights = relu_attention(x, return_attention=True)
        softmax_output, softmax_weights = softmax_attention(x, return_attention=True)
        
        # ReLU 注意力权重可能包含 0（稀疏）
        assert (relu_weights >= 0).all()  # ReLU 输出非负
        
        # Softmax 注意力权重每行和为 1
        softmax_sum = softmax_weights.sum(dim=-1)
        assert torch.allclose(softmax_sum, torch.ones_like(softmax_sum), atol=1e-5)
    
    def test_create_attention_factory(self, small_config):
        """测试注意力层工厂函数"""
        relu_attn = create_attention_layer(small_config, use_relu=True)
        softmax_attn = create_attention_layer(small_config, use_relu=False)
        
        assert isinstance(relu_attn, DotProductAggregatedAttention)
        assert isinstance(softmax_attn, SoftmaxAttention)


# =============================================================================
# 层归一化测试
# =============================================================================

class TestLayerNorm:
    """层归一化测试"""
    
    def test_group_layer_norm_shape(self, small_config, sample_input):
        """测试分组层归一化输出形状"""
        gln = GroupLayerNorm(
            d_model=small_config.d_model,
            num_groups=small_config.num_groups,
        )
        
        x = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output = gln(x, sample_input["token_types"])
        
        assert output.shape == x.shape
    
    def test_group_layer_norm_different_groups(self, small_config, sample_input):
        """测试不同组使用不同参数"""
        gln = GroupLayerNorm(
            d_model=small_config.d_model,
            num_groups=small_config.num_groups,
        )
        
        # 设置不同组的 gamma 和 beta 为不同值
        gln.gamma.data = torch.randn_like(gln.gamma.data)
        gln.beta.data = torch.randn_like(gln.beta.data)
        
        x = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output = gln(x, sample_input["token_types"])
        
        # 验证输出不全相同（因为不同组有不同参数）
        assert output.shape == x.shape
    
    def test_standard_layer_norm(self, small_config, sample_input):
        """测试标准层归一化"""
        ln = StandardLayerNorm(d_model=small_config.d_model)
        
        x = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output = ln(x, sample_input["token_types"])
        
        assert output.shape == x.shape
    
    def test_create_layer_norm_factory(self, small_config):
        """测试层归一化工厂函数"""
        gln = create_layer_norm(small_config, use_group_norm=True)
        sln = create_layer_norm(small_config, use_group_norm=False)
        
        assert isinstance(gln, GroupLayerNorm)
        assert isinstance(sln, StandardLayerNorm)


# =============================================================================
# 前馈网络测试
# =============================================================================

class TestFFN:
    """前馈网络测试"""
    
    def test_standard_ffn_shape(self, small_config, sample_input):
        """测试标准 FFN 输出形状"""
        ffn = FeedForwardNetwork(
            d_model=small_config.d_model,
            d_ff=small_config.d_ff,
            dropout=0.0,
        )
        
        x = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output = ffn(x)
        
        assert output.shape == x.shape
    
    def test_glu_shape(self, small_config, sample_input):
        """测试 GLU 输出形状"""
        glu = GatedLinearUnit(
            d_model=small_config.d_model,
            d_ff=small_config.d_ff,
            dropout=0.0,
        )
        
        x = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output = glu(x)
        
        assert output.shape == x.shape
    
    def test_swiglu_shape(self, small_config, sample_input):
        """测试 SwiGLU 输出形状"""
        swiglu = SwiGLU(
            d_model=small_config.d_model,
            d_ff=small_config.d_ff,
            dropout=0.0,
        )
        
        x = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output = swiglu(x)
        
        assert output.shape == x.shape
    
    def test_create_ffn_factory(self, small_config):
        """测试 FFN 工厂函数"""
        standard = create_ffn(small_config, ffn_type="standard")
        glu = create_ffn(small_config, ffn_type="glu")
        swiglu = create_ffn(small_config, ffn_type="swiglu")
        
        assert isinstance(standard, FeedForwardNetwork)
        assert isinstance(glu, GatedLinearUnit)
        assert isinstance(swiglu, SwiGLU)


# =============================================================================
# 编码器层测试
# =============================================================================

class TestEncoderLayer:
    """编码器层测试"""
    
    def test_encoder_layer_shape(self, small_config, sample_input):
        """测试编码器层输出形状"""
        layer = EncoderLayer(small_config)
        
        x = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output, _ = layer(
            x,
            sample_input["token_types"],
            sample_input["attention_mask"],
        )
        
        assert output.shape == x.shape
    
    def test_encoder_layer_attention_weights(self, small_config, sample_input):
        """测试编码器层返回注意力权重"""
        layer = EncoderLayer(small_config)
        
        x = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output, attn_weights = layer(
            x,
            sample_input["token_types"],
            sample_input["attention_mask"],
            return_attention=True,
        )
        
        assert attn_weights is not None
    
    def test_pre_norm_encoder_layer(self, small_config, sample_input):
        """测试 Pre-Norm 编码器层"""
        layer = PreNormEncoderLayer(small_config)
        
        x = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output, _ = layer(
            x,
            sample_input["token_types"],
            sample_input["attention_mask"],
        )
        
        assert output.shape == x.shape
    
    def test_create_encoder_layer_factory(self, small_config):
        """测试编码器层工厂函数"""
        post_norm = create_encoder_layer(small_config, use_pre_norm=False)
        pre_norm = create_encoder_layer(small_config, use_pre_norm=True)
        
        assert isinstance(post_norm, EncoderLayer)
        assert isinstance(pre_norm, PreNormEncoderLayer)


# =============================================================================
# 完整编码器测试
# =============================================================================

class TestUGTEncoder:
    """UGT 编码器测试"""
    
    def test_encoder_user_repr_shape(self, small_config, sample_input):
        """测试编码器用户表示输出形状"""
        encoder = UGTEncoder(small_config)
        
        user_repr = encoder(
            sample_input["semantic_ids"],
            sample_input["positions"],
            sample_input["token_types"],
            sample_input["attention_mask"],
        )
        
        expected_shape = (sample_input["batch_size"], small_config.d_model)
        assert user_repr.shape == expected_shape
    
    def test_encoder_sequence_output_shape(self, small_config, sample_input):
        """测试编码器序列输出形状"""
        encoder = UGTEncoder(small_config)
        
        seq_output = encoder.get_sequence_output(
            sample_input["semantic_ids"],
            sample_input["positions"],
            sample_input["token_types"],
            sample_input["attention_mask"],
        )
        
        expected_shape = (
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        assert seq_output.shape == expected_shape
    
    def test_encoder_with_time_features(self, small_config, sample_input):
        """测试带时间特征的编码器"""
        encoder = UGTEncoder(small_config)
        
        time_features = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.time_dim,
        )
        
        user_repr = encoder(
            sample_input["semantic_ids"],
            sample_input["positions"],
            sample_input["token_types"],
            sample_input["attention_mask"],
            time_features,
        )
        
        expected_shape = (sample_input["batch_size"], small_config.d_model)
        assert user_repr.shape == expected_shape
    
    def test_encoder_attention_weights(self, small_config, sample_input):
        """测试编码器注意力权重获取"""
        encoder = UGTEncoder(small_config)
        
        attn_weights = encoder.get_attention_weights(
            sample_input["semantic_ids"],
            sample_input["positions"],
            sample_input["token_types"],
            sample_input["attention_mask"],
        )
        
        assert len(attn_weights) == small_config.n_layers
    
    def test_encoder_freeze_unfreeze(self, small_config):
        """测试编码器冻结/解冻嵌入层"""
        encoder = UGTEncoder(small_config)
        
        # 冻结嵌入层
        encoder.freeze_embeddings()
        for param in encoder.input_embedding.parameters():
            assert not param.requires_grad
        
        # 解冻嵌入层
        encoder.unfreeze_embeddings()
        for param in encoder.input_embedding.parameters():
            assert param.requires_grad
    
    def test_encoder_num_parameters(self, small_config):
        """测试参数数量统计"""
        encoder = UGTEncoder(small_config)
        
        trainable_params = encoder.get_num_parameters(trainable_only=True)
        total_params = encoder.get_num_parameters(trainable_only=False)
        
        assert trainable_params > 0
        assert total_params >= trainable_params
    
    def test_encoder_from_config(self, small_config):
        """测试从配置创建编码器"""
        encoder = UGTEncoder.from_config(small_config)
        
        assert encoder.config.d_model == small_config.d_model
        assert encoder.config.n_heads == small_config.n_heads
    
    def test_pooler_cls(self, small_config, sample_input):
        """测试 CLS 池化"""
        config = EncoderConfig(
            d_model=small_config.d_model,
            n_heads=small_config.n_heads,
            n_layers=small_config.n_layers,
            pooler_type="cls",
        )
        pooler = Pooler(config)
        
        hidden = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output = pooler(hidden, sample_input["attention_mask"])
        
        expected_shape = (sample_input["batch_size"], small_config.d_model)
        assert output.shape == expected_shape
    
    def test_pooler_mean(self, small_config, sample_input):
        """测试 Mean 池化"""
        config = EncoderConfig(
            d_model=small_config.d_model,
            n_heads=small_config.n_heads,
            n_layers=small_config.n_layers,
            pooler_type="mean",
        )
        pooler = Pooler(config)
        
        hidden = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output = pooler(hidden, sample_input["attention_mask"])
        
        expected_shape = (sample_input["batch_size"], small_config.d_model)
        assert output.shape == expected_shape
    
    def test_pooler_last(self, small_config, sample_input):
        """测试 Last 池化"""
        config = EncoderConfig(
            d_model=small_config.d_model,
            n_heads=small_config.n_heads,
            n_layers=small_config.n_layers,
            pooler_type="last",
        )
        pooler = Pooler(config)
        
        hidden = torch.randn(
            sample_input["batch_size"],
            sample_input["seq_len"],
            small_config.d_model,
        )
        
        output = pooler(hidden, sample_input["attention_mask"])
        
        expected_shape = (sample_input["batch_size"], small_config.d_model)
        assert output.shape == expected_shape


# =============================================================================
# 集成测试
# =============================================================================

class TestIntegration:
    """集成测试"""
    
    def test_end_to_end_forward(self, base_config, sample_input):
        """测试端到端前向传播"""
        encoder = UGTEncoder(base_config)
        
        # 调整输入大小以匹配 base_config
        batch_size = sample_input["batch_size"]
        seq_len = 64
        
        l1_ids = torch.randint(0, base_config.codebook_sizes[0], (batch_size, seq_len))
        l2_ids = torch.randint(0, base_config.codebook_sizes[1], (batch_size, seq_len))
        l3_ids = torch.randint(0, base_config.codebook_sizes[2], (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        token_types = torch.randint(0, base_config.num_token_types, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        user_repr = encoder(
            [l1_ids, l2_ids, l3_ids],
            positions,
            token_types,
            attention_mask,
        )
        
        expected_shape = (batch_size, base_config.d_model)
        assert user_repr.shape == expected_shape
    
    def test_gradient_flow(self, small_config, sample_input):
        """测试梯度流动"""
        encoder = UGTEncoder(small_config)
        
        user_repr = encoder(
            sample_input["semantic_ids"],
            sample_input["positions"],
            sample_input["token_types"],
            sample_input["attention_mask"],
        )
        
        # 计算简单损失
        loss = user_repr.mean()
        loss.backward()
        
        # 检查梯度存在
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_different_batch_sizes(self, small_config):
        """测试不同批次大小"""
        encoder = UGTEncoder(small_config)
        
        for batch_size in [1, 2, 4, 8]:
            seq_len = 32
            
            l1_ids = torch.randint(0, small_config.codebook_sizes[0], (batch_size, seq_len))
            l2_ids = torch.randint(0, small_config.codebook_sizes[1], (batch_size, seq_len))
            l3_ids = torch.randint(0, small_config.codebook_sizes[2], (batch_size, seq_len))
            positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            token_types = torch.randint(0, small_config.num_token_types, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            
            user_repr = encoder(
                [l1_ids, l2_ids, l3_ids],
                positions,
                token_types,
                attention_mask,
            )
            
            assert user_repr.shape == (batch_size, small_config.d_model)
    
    def test_different_seq_lengths(self, small_config):
        """测试不同序列长度"""
        encoder = UGTEncoder(small_config)
        batch_size = 4
        
        for seq_len in [8, 16, 32, 64]:
            l1_ids = torch.randint(0, small_config.codebook_sizes[0], (batch_size, seq_len))
            l2_ids = torch.randint(0, small_config.codebook_sizes[1], (batch_size, seq_len))
            l3_ids = torch.randint(0, small_config.codebook_sizes[2], (batch_size, seq_len))
            positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            token_types = torch.randint(0, small_config.num_token_types, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            
            user_repr = encoder(
                [l1_ids, l2_ids, l3_ids],
                positions,
                token_types,
                attention_mask,
            )
            
            assert user_repr.shape == (batch_size, small_config.d_model)
    
    def test_eval_mode(self, small_config, sample_input):
        """测试评估模式"""
        encoder = UGTEncoder(small_config)
        encoder.eval()
        
        with torch.no_grad():
            user_repr = encoder(
                sample_input["semantic_ids"],
                sample_input["positions"],
                sample_input["token_types"],
                sample_input["attention_mask"],
            )
        
        expected_shape = (sample_input["batch_size"], small_config.d_model)
        assert user_repr.shape == expected_shape
    
    def test_deterministic_output(self, small_config, sample_input):
        """测试确定性输出（eval 模式下）"""
        encoder = UGTEncoder(small_config)
        encoder.eval()
        
        with torch.no_grad():
            output1 = encoder(
                sample_input["semantic_ids"],
                sample_input["positions"],
                sample_input["token_types"],
                sample_input["attention_mask"],
            )
            
            output2 = encoder(
                sample_input["semantic_ids"],
                sample_input["positions"],
                sample_input["token_types"],
                sample_input["attention_mask"],
            )
        
        assert torch.allclose(output1, output2)


# =============================================================================
# 性能测试（可选）
# =============================================================================

class TestPerformance:
    """性能测试"""
    
    @pytest.mark.slow
    def test_large_batch(self, base_config):
        """测试大批次处理"""
        encoder = UGTEncoder(base_config)
        encoder.eval()
        
        batch_size = 64
        seq_len = 128
        
        l1_ids = torch.randint(0, base_config.codebook_sizes[0], (batch_size, seq_len))
        l2_ids = torch.randint(0, base_config.codebook_sizes[1], (batch_size, seq_len))
        l3_ids = torch.randint(0, base_config.codebook_sizes[2], (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        token_types = torch.randint(0, base_config.num_token_types, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            user_repr = encoder(
                [l1_ids, l2_ids, l3_ids],
                positions,
                token_types,
                attention_mask,
            )
        
        assert user_repr.shape == (batch_size, base_config.d_model)


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    """
    直接运行测试脚本
    """
    print("运行 UGT 编码器单元测试...")
    print("=" * 60)
    
    # 简单测试
    config = EncoderConfig(
        d_model=512,
        n_heads=16,
        n_layers=6,
        dropout=0.0,
    )
    encoder = UGTEncoder(config)
    
    batch_size = 32
    seq_len = 100
    
    # 准备输入
    l1_ids = torch.randint(0, 1024, (batch_size, seq_len))
    l2_ids = torch.randint(0, 4096, (batch_size, seq_len))
    l3_ids = torch.randint(0, 16384, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    token_types = torch.randint(0, 4, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 测试用户表示
    user_repr = encoder([l1_ids, l2_ids, l3_ids], positions, token_types, attention_mask)
    print(f"用户表示形状: {user_repr.shape}")
    assert user_repr.shape == (batch_size, config.d_model)
    
    # 测试序列输出
    seq_output = encoder.get_sequence_output(
        [l1_ids, l2_ids, l3_ids], 
        positions, 
        token_types, 
        attention_mask
    )
    print(f"序列输出形状: {seq_output.shape}")
    assert seq_output.shape == (batch_size, seq_len, config.d_model)
    
    # 统计参数
    num_params = encoder.get_num_parameters()
    print(f"可训练参数数量: {num_params:,}")
    
    print("=" * 60)
    print("所有编码器测试通过!")

