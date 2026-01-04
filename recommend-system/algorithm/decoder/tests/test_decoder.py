"""
UGT Decoder 单元测试

测试内容：
1. 配置类测试
2. MoE 模块测试
3. 注意力机制测试
4. 解码器层测试
5. 完整解码器测试
6. 生成器测试
7. 接口兼容性测试

运行方式:
    pytest recommend-system/algorithm/decoder/tests/test_decoder.py -v
"""

import pytest
import torch
import torch.nn as nn

# 导入被测试的模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from decoder.config import DecoderConfig
from decoder.moe import Expert, Router, MoEFeedForward, SharedExpertMoE
from decoder.cross_attention import (
    CausalSelfAttention,
    CrossAttention,
    GroupLayerNorm,
    FeedForward,
)
from decoder.decoder_layer import (
    DecoderLayer,
    DecoderLayerWithoutCrossAttention,
    DecoderInputEmbedding,
)
from decoder.decoder import UGTDecoder, UGTDecoderForInference
from decoder.generator import (
    GenerationConfig,
    BeamSearchGenerator,
    DiverseBeamSearchGenerator,
    NucleusSamplingGenerator,
)


# ==============================================================================
# 测试夹具
# ==============================================================================

@pytest.fixture
def small_config():
    """小规模配置，用于快速测试"""
    return DecoderConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=128,
        num_experts=4,
        top_k_experts=2,
        codebook_sizes=(32, 64, 128),
        num_token_types=4,
        num_groups=4,
        dropout=0.1,
    )


@pytest.fixture
def device():
    """测试设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def src_len():
    return 32


@pytest.fixture
def tgt_len():
    return 16


# ==============================================================================
# 配置类测试
# ==============================================================================

class TestDecoderConfig:
    """测试解码器配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = DecoderConfig()
        assert config.d_model == 512
        assert config.n_heads == 16
        assert config.n_layers == 12
        assert len(config.codebook_sizes) == 3
    
    def test_small_config(self):
        """测试小规模配置"""
        config = DecoderConfig.small()
        assert config.d_model == 256
        assert config.n_heads == 8
        assert config.n_layers == 6
    
    def test_medium_config(self):
        """测试中规模配置"""
        config = DecoderConfig.medium()
        assert config.d_model == 512
        assert config.n_heads == 16
    
    def test_large_config(self):
        """测试大规模配置"""
        config = DecoderConfig.large()
        assert config.d_model == 1024
        assert config.n_heads == 32
        assert config.use_flash_attention == True
    
    def test_config_validation(self):
        """测试配置验证"""
        # d_model 必须能被 n_heads 整除
        with pytest.raises(AssertionError):
            DecoderConfig(d_model=512, n_heads=15)
        
        # top_k 不能超过 num_experts
        with pytest.raises(AssertionError):
            DecoderConfig(num_experts=4, top_k_experts=8)
    
    def test_head_dim_property(self):
        """测试 head_dim 属性"""
        config = DecoderConfig(d_model=512, n_heads=16)
        assert config.head_dim == 32
    
    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化"""
        config = DecoderConfig.medium()
        d = config.to_dict()
        restored = DecoderConfig.from_dict(d)
        assert restored.d_model == config.d_model
        assert restored.n_heads == config.n_heads
        assert restored.codebook_sizes == config.codebook_sizes


# ==============================================================================
# MoE 模块测试
# ==============================================================================

class TestMoE:
    """测试 Mixture of Experts 模块"""
    
    def test_expert(self, small_config, device):
        """测试单个专家网络"""
        expert = Expert(
            d_model=small_config.d_model,
            d_ff=small_config.d_ff,
            dropout=small_config.dropout
        ).to(device)
        
        x = torch.randn(4, 16, small_config.d_model, device=device)
        output = expert(x)
        
        assert output.shape == x.shape
    
    def test_router(self, small_config, device):
        """测试路由网络"""
        router = Router(
            d_model=small_config.d_model,
            num_experts=small_config.num_experts,
            top_k=small_config.top_k_experts,
        ).to(device)
        
        x = torch.randn(64, small_config.d_model, device=device)  # 扁平化输入
        top_k_gates, top_k_indices, router_probs = router(x)
        
        assert top_k_gates.shape == (64, small_config.top_k_experts)
        assert top_k_indices.shape == (64, small_config.top_k_experts)
        assert router_probs.shape == (64, small_config.num_experts)
        
        # 验证概率和为 1
        assert torch.allclose(router_probs.sum(dim=-1), torch.ones(64, device=device), atol=1e-5)
    
    def test_moe_feedforward(self, small_config, device, batch_size):
        """测试 MoE 前馈网络"""
        moe = MoEFeedForward(small_config).to(device)
        
        seq_len = 16
        x = torch.randn(batch_size, seq_len, small_config.d_model, device=device)
        output = moe(x)
        
        # 检查输出形状
        assert output.shape == x.shape
        
        # 检查辅助损失
        aux_loss = moe.get_aux_loss()
        assert aux_loss.ndim == 0  # 标量
        assert aux_loss >= 0
    
    def test_moe_expert_utilization(self, small_config, device, batch_size):
        """测试专家利用率"""
        moe = MoEFeedForward(small_config).to(device)
        
        x = torch.randn(batch_size, 16, small_config.d_model, device=device)
        _ = moe(x)
        
        stats = moe.get_expert_utilization()
        assert stats["num_experts"] == small_config.num_experts
        assert stats["top_k"] == small_config.top_k_experts
    
    def test_shared_expert_moe(self, small_config, device, batch_size):
        """测试带共享专家的 MoE"""
        shared_moe = SharedExpertMoE(small_config).to(device)
        
        x = torch.randn(batch_size, 16, small_config.d_model, device=device)
        output = shared_moe(x)
        
        assert output.shape == x.shape
        
        # 检查辅助损失
        aux_loss = shared_moe.get_aux_loss()
        assert aux_loss.ndim == 0


# ==============================================================================
# 注意力机制测试
# ==============================================================================

class TestAttention:
    """测试注意力机制"""
    
    def test_causal_self_attention(self, small_config, device, batch_size):
        """测试因果自注意力"""
        attn = CausalSelfAttention(
            d_model=small_config.d_model,
            n_heads=small_config.n_heads,
            dropout=small_config.dropout,
            max_seq_len=small_config.max_seq_len,
        ).to(device)
        
        seq_len = 16
        x = torch.randn(batch_size, seq_len, small_config.d_model, device=device)
        
        output, cache = attn(x)
        
        assert output.shape == x.shape
        assert cache is None  # 没有提供初始缓存时返回 None
    
    def test_causal_self_attention_with_cache(self, small_config, device, batch_size):
        """测试带 KV 缓存的因果自注意力"""
        attn = CausalSelfAttention(
            d_model=small_config.d_model,
            n_heads=small_config.n_heads,
            dropout=small_config.dropout,
            max_seq_len=small_config.max_seq_len,
        ).to(device)
        
        # 第一步
        x1 = torch.randn(batch_size, 10, small_config.d_model, device=device)
        output1, cache1 = attn(x1, cache=None)
        
        # 第二步（增量）
        x2 = torch.randn(batch_size, 1, small_config.d_model, device=device)
        output2, cache2 = attn(x2, cache=cache1)
        
        assert output2.shape == (batch_size, 1, small_config.d_model)
        assert cache2 is not None
    
    def test_cross_attention(self, small_config, device, batch_size, src_len, tgt_len):
        """测试交叉注意力"""
        attn = CrossAttention(
            d_model=small_config.d_model,
            n_heads=small_config.n_heads,
            dropout=small_config.dropout,
        ).to(device)
        
        query = torch.randn(batch_size, tgt_len, small_config.d_model, device=device)
        encoder_output = torch.randn(batch_size, src_len, small_config.d_model, device=device)
        
        output, cache = attn(query, encoder_output)
        
        assert output.shape == query.shape
        assert cache is not None
    
    def test_group_layer_norm(self, small_config, device, batch_size):
        """测试分组层归一化"""
        gln = GroupLayerNorm(
            d_model=small_config.d_model,
            num_groups=small_config.num_groups,
        ).to(device)
        
        seq_len = 16
        x = torch.randn(batch_size, seq_len, small_config.d_model, device=device)
        token_types = torch.randint(0, small_config.num_groups, (batch_size, seq_len), device=device)
        
        output = gln(x, token_types)
        
        assert output.shape == x.shape
    
    def test_feedforward(self, small_config, device, batch_size):
        """测试前馈网络"""
        ffn = FeedForward(
            d_model=small_config.d_model,
            d_ff=small_config.d_ff,
            dropout=small_config.dropout,
        ).to(device)
        
        x = torch.randn(batch_size, 16, small_config.d_model, device=device)
        output = ffn(x)
        
        assert output.shape == x.shape


# ==============================================================================
# 解码器层测试
# ==============================================================================

class TestDecoderLayer:
    """测试解码器层"""
    
    def test_decoder_layer(self, small_config, device, batch_size, src_len, tgt_len):
        """测试完整解码器层"""
        layer = DecoderLayer(small_config).to(device)
        
        x = torch.randn(batch_size, tgt_len, small_config.d_model, device=device)
        encoder_output = torch.randn(batch_size, src_len, small_config.d_model, device=device)
        token_types = torch.randint(0, small_config.num_groups, (batch_size, tgt_len), device=device)
        
        output, aux_loss, self_cache, cross_cache = layer(
            x=x,
            encoder_output=encoder_output,
            token_types=token_types,
        )
        
        assert output.shape == x.shape
        assert aux_loss.ndim == 0
    
    def test_decoder_layer_without_cross_attention(self, small_config, device, batch_size):
        """测试不带交叉注意力的解码器层"""
        layer = DecoderLayerWithoutCrossAttention(small_config).to(device)
        
        seq_len = 16
        x = torch.randn(batch_size, seq_len, small_config.d_model, device=device)
        token_types = torch.randint(0, small_config.num_groups, (batch_size, seq_len), device=device)
        
        output, aux_loss, cache = layer(x, token_types)
        
        assert output.shape == x.shape
        assert aux_loss.ndim == 0
    
    def test_decoder_input_embedding(self, small_config, device, batch_size):
        """测试解码器输入嵌入"""
        embedding = DecoderInputEmbedding(small_config).to(device)
        
        seq_len = 16
        l1_ids = torch.randint(0, small_config.codebook_sizes[0], (batch_size, seq_len), device=device)
        l2_ids = torch.randint(0, small_config.codebook_sizes[1], (batch_size, seq_len), device=device)
        l3_ids = torch.randint(0, small_config.codebook_sizes[2], (batch_size, seq_len), device=device)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_types = torch.randint(0, small_config.num_token_types, (batch_size, seq_len), device=device)
        
        output = embedding(
            semantic_ids=(l1_ids, l2_ids, l3_ids),
            positions=positions,
            token_types=token_types,
        )
        
        assert output.shape == (batch_size, seq_len, small_config.d_model)


# ==============================================================================
# 完整解码器测试
# ==============================================================================

class TestUGTDecoder:
    """测试完整的 UGT 解码器"""
    
    def test_decoder_forward(self, small_config, device, batch_size, src_len, tgt_len):
        """测试解码器前向传播"""
        decoder = UGTDecoder(small_config).to(device)
        
        encoder_output = torch.randn(batch_size, src_len, small_config.d_model, device=device)
        
        l1_ids = torch.randint(0, small_config.codebook_sizes[0], (batch_size, tgt_len), device=device)
        l2_ids = torch.randint(0, small_config.codebook_sizes[1], (batch_size, tgt_len), device=device)
        l3_ids = torch.randint(0, small_config.codebook_sizes[2], (batch_size, tgt_len), device=device)
        target_positions = torch.arange(tgt_len, device=device).unsqueeze(0).expand(batch_size, -1)
        target_token_types = torch.ones(batch_size, tgt_len, dtype=torch.long, device=device)
        
        l1_logits, l2_logits, l3_logits, aux_loss = decoder(
            encoder_output=encoder_output,
            target_semantic_ids=[l1_ids, l2_ids, l3_ids],
            target_positions=target_positions,
            target_token_types=target_token_types,
        )
        
        # 检查输出形状
        assert l1_logits.shape == (batch_size, tgt_len, small_config.codebook_sizes[0])
        assert l2_logits.shape == (batch_size, tgt_len, small_config.codebook_sizes[1])
        assert l3_logits.shape == (batch_size, tgt_len, small_config.codebook_sizes[2])
        assert aux_loss.ndim == 0  # 标量
    
    def test_decoder_generate(self, small_config, device, batch_size, src_len):
        """测试解码器生成"""
        decoder = UGTDecoder(small_config).to(device)
        decoder.eval()
        
        encoder_output = torch.randn(batch_size, src_len, small_config.d_model, device=device)
        
        num_recommendations = 5
        recommendations = decoder.generate(
            encoder_output=encoder_output,
            num_recommendations=num_recommendations,
            beam_size=2,
            temperature=1.0,
        )
        
        # 检查输出格式
        assert len(recommendations) == batch_size
        for batch_recs in recommendations:
            assert len(batch_recs) <= num_recommendations
            for rec in batch_recs:
                assert len(rec) == 3  # (L1, L2, L3)
    
    def test_decoder_compute_loss(self, small_config, device, batch_size, src_len, tgt_len):
        """测试损失计算"""
        decoder = UGTDecoder(small_config).to(device)
        
        encoder_output = torch.randn(batch_size, src_len, small_config.d_model, device=device)
        
        l1_ids = torch.randint(0, small_config.codebook_sizes[0], (batch_size, tgt_len), device=device)
        l2_ids = torch.randint(0, small_config.codebook_sizes[1], (batch_size, tgt_len), device=device)
        l3_ids = torch.randint(0, small_config.codebook_sizes[2], (batch_size, tgt_len), device=device)
        
        l1_logits, l2_logits, l3_logits, aux_loss = decoder(
            encoder_output=encoder_output,
            target_semantic_ids=[l1_ids, l2_ids, l3_ids],
        )
        
        losses = decoder.compute_loss(
            l1_logits, l2_logits, l3_logits,
            l1_ids, l2_ids, l3_ids,
            aux_loss,
        )
        
        assert "total_loss" in losses
        assert "ntp_loss" in losses
        assert "l1_loss" in losses
        assert "l2_loss" in losses
        assert "l3_loss" in losses
        assert "aux_loss" in losses
        
        # 检查所有损失都是有限值
        for key, value in losses.items():
            assert torch.isfinite(value), f"{key} 损失不是有限值"
    
    def test_decoder_num_params(self, small_config):
        """测试参数数量统计"""
        decoder = UGTDecoder(small_config)
        
        num_params = decoder.get_num_params()
        num_trainable = decoder.get_num_trainable_params()
        
        assert num_params > 0
        assert num_trainable == num_params  # 默认全部可训练


# ==============================================================================
# 生成器测试
# ==============================================================================

class TestGenerators:
    """测试推荐生成器"""
    
    def test_beam_search_generator(self, small_config, device, batch_size, src_len):
        """测试 Beam Search 生成器"""
        decoder = UGTDecoder(small_config).to(device)
        decoder.eval()
        
        generator = BeamSearchGenerator(decoder, small_config)
        
        encoder_output = torch.randn(batch_size, src_len, small_config.d_model, device=device)
        
        num_recommendations = 5
        recommendations = generator.generate(
            encoder_output=encoder_output,
            num_recommendations=num_recommendations,
            beam_size=3,
            temperature=1.0,
        )
        
        assert len(recommendations) == batch_size
        for batch_recs in recommendations:
            assert len(batch_recs) <= num_recommendations
    
    def test_diverse_beam_search_generator(self, small_config, device, batch_size, src_len):
        """测试多样性 Beam Search 生成器"""
        decoder = UGTDecoder(small_config).to(device)
        decoder.eval()
        
        generator = DiverseBeamSearchGenerator(decoder, small_config)
        
        encoder_output = torch.randn(batch_size, src_len, small_config.d_model, device=device)
        
        num_recommendations = 8
        recommendations = generator.generate(
            encoder_output=encoder_output,
            num_recommendations=num_recommendations,
            num_beam_groups=2,
            beam_size_per_group=2,
            diversity_penalty=0.5,
        )
        
        assert len(recommendations) == batch_size
        for batch_recs in recommendations:
            assert len(batch_recs) <= num_recommendations
    
    def test_nucleus_sampling_generator(self, small_config, device, batch_size, src_len):
        """测试核采样生成器"""
        decoder = UGTDecoder(small_config).to(device)
        decoder.eval()
        
        generator = NucleusSamplingGenerator(decoder, small_config)
        
        encoder_output = torch.randn(batch_size, src_len, small_config.d_model, device=device)
        
        num_recommendations = 5
        recommendations = generator.generate(
            encoder_output=encoder_output,
            num_recommendations=num_recommendations,
            top_p=0.9,
            temperature=1.0,
        )
        
        assert len(recommendations) == batch_size
        for batch_recs in recommendations:
            assert len(batch_recs) <= num_recommendations


# ==============================================================================
# 接口兼容性测试
# ==============================================================================

class TestInterfaceCompatibility:
    """测试与 interfaces.py 中定义的接口的兼容性"""
    
    def test_decoder_interface_forward(self, small_config, device, batch_size, src_len, tgt_len):
        """
        测试 forward 方法是否符合 RecommendDecoderInterface
        
        接口要求:
        def forward(
            self,
            encoder_output: torch.Tensor,
            target_semantic_ids: Optional[List[torch.Tensor]] = None,
            target_positions: Optional[torch.Tensor] = None,
            target_token_types: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        decoder = UGTDecoder(small_config).to(device)
        
        encoder_output = torch.randn(batch_size, src_len, small_config.d_model, device=device)
        
        l1_ids = torch.randint(0, small_config.codebook_sizes[0], (batch_size, tgt_len), device=device)
        l2_ids = torch.randint(0, small_config.codebook_sizes[1], (batch_size, tgt_len), device=device)
        l3_ids = torch.randint(0, small_config.codebook_sizes[2], (batch_size, tgt_len), device=device)
        target_positions = torch.arange(tgt_len, device=device).unsqueeze(0).expand(batch_size, -1)
        target_token_types = torch.ones(batch_size, tgt_len, dtype=torch.long, device=device)
        
        # 调用 forward
        result = decoder.forward(
            encoder_output=encoder_output,
            target_semantic_ids=[l1_ids, l2_ids, l3_ids],
            target_positions=target_positions,
            target_token_types=target_token_types,
        )
        
        # 验证返回值格式
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        l1_logits, l2_logits, l3_logits, aux_loss = result
        
        # 验证各返回值形状
        assert l1_logits.shape == (batch_size, tgt_len, small_config.codebook_sizes[0])
        assert l2_logits.shape == (batch_size, tgt_len, small_config.codebook_sizes[1])
        assert l3_logits.shape == (batch_size, tgt_len, small_config.codebook_sizes[2])
        assert aux_loss.ndim == 0  # 标量
    
    def test_decoder_interface_generate(self, small_config, device, batch_size, src_len):
        """
        测试 generate 方法是否符合 RecommendDecoderInterface
        
        接口要求:
        def generate(
            self,
            encoder_output: torch.Tensor,
            num_recommendations: int = 20,
            beam_size: int = 4,
            temperature: float = 1.0,
        ) -> List[List[Tuple[int, int, int]]]:
        """
        decoder = UGTDecoder(small_config).to(device)
        decoder.eval()
        
        encoder_output = torch.randn(batch_size, src_len, small_config.d_model, device=device)
        
        # 调用 generate
        result = decoder.generate(
            encoder_output=encoder_output,
            num_recommendations=10,
            beam_size=4,
            temperature=1.0,
        )
        
        # 验证返回值格式
        assert isinstance(result, list)
        assert len(result) == batch_size
        
        for batch_result in result:
            assert isinstance(batch_result, list)
            for rec in batch_result:
                assert isinstance(rec, tuple)
                assert len(rec) == 3
                l1, l2, l3 = rec
                assert isinstance(l1, int)
                assert isinstance(l2, int)
                assert isinstance(l3, int)


# ==============================================================================
# 梯度测试
# ==============================================================================

class TestGradients:
    """测试梯度流动"""
    
    def test_gradient_flow(self, small_config, device, batch_size, src_len, tgt_len):
        """测试梯度是否正常流动"""
        decoder = UGTDecoder(small_config).to(device)
        decoder.train()
        
        encoder_output = torch.randn(
            batch_size, src_len, small_config.d_model,
            device=device, requires_grad=True
        )
        
        l1_ids = torch.randint(0, small_config.codebook_sizes[0], (batch_size, tgt_len), device=device)
        l2_ids = torch.randint(0, small_config.codebook_sizes[1], (batch_size, tgt_len), device=device)
        l3_ids = torch.randint(0, small_config.codebook_sizes[2], (batch_size, tgt_len), device=device)
        
        l1_logits, l2_logits, l3_logits, aux_loss = decoder(
            encoder_output=encoder_output,
            target_semantic_ids=[l1_ids, l2_ids, l3_ids],
        )
        
        # 计算损失
        losses = decoder.compute_loss(
            l1_logits, l2_logits, l3_logits,
            l1_ids, l2_ids, l3_ids,
            aux_loss,
        )
        
        total_loss = losses['total_loss']
        
        # 反向传播
        total_loss.backward()
        
        # 检查梯度
        assert encoder_output.grad is not None
        assert not torch.isnan(encoder_output.grad).any()
        
        # 检查模型参数梯度
        for name, param in decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 {name} 没有梯度"
                assert not torch.isnan(param.grad).any(), f"参数 {name} 的梯度包含 NaN"


# ==============================================================================
# 运行测试
# ==============================================================================

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])

