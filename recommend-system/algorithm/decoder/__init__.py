"""
UGT Decoder 模块

生成式推荐系统的解码器组件，负责从用户表示向量自回归地生成推荐物品的 Semantic ID 序列。

模块组件：
- config: 解码器配置类
- moe: Mixture of Experts 前馈网络
- cross_attention: 注意力机制（因果自注意力、交叉注意力、分组层归一化）
- decoder_layer: 解码器层
- decoder: 完整的 UGT 解码器
- generator: 推荐生成器（Beam Search、核采样等）

使用示例:

```python
from algorithm.decoder import UGTDecoder, DecoderConfig, BeamSearchGenerator

# 创建配置
config = DecoderConfig.medium()

# 创建解码器
decoder = UGTDecoder(config)

# 训练模式
encoder_output = ...  # 来自编码器
target_semantic_ids = [l1_ids, l2_ids, l3_ids]
l1_logits, l2_logits, l3_logits, aux_loss = decoder(
    encoder_output=encoder_output,
    target_semantic_ids=target_semantic_ids,
    target_positions=positions,
    target_token_types=token_types,
)

# 推理模式
generator = BeamSearchGenerator(decoder, config)
recommendations = generator.generate(
    encoder_output=encoder_output,
    num_recommendations=20,
    beam_size=4,
)
```

接口实现：
本模块实现了 algorithm.interfaces.RecommendDecoderInterface 接口。
"""

from .config import DecoderConfig
from .moe import Expert, Router, MoEFeedForward, SharedExpertMoE
from .cross_attention import (
    CausalSelfAttention,
    CrossAttention,
    GroupLayerNorm,
    FeedForward,
)
from .decoder_layer import (
    DecoderLayer,
    DecoderLayerWithoutCrossAttention,
    DecoderInputEmbedding,
)
from .decoder import UGTDecoder, UGTDecoderForInference
from .generator import (
    GenerationConfig,
    BeamSearchGenerator,
    DiverseBeamSearchGenerator,
    NucleusSamplingGenerator,
)


__all__ = [
    # 配置
    "DecoderConfig",
    "GenerationConfig",
    
    # MoE 组件
    "Expert",
    "Router",
    "MoEFeedForward",
    "SharedExpertMoE",
    
    # 注意力组件
    "CausalSelfAttention",
    "CrossAttention",
    "GroupLayerNorm",
    "FeedForward",
    
    # 解码器层
    "DecoderLayer",
    "DecoderLayerWithoutCrossAttention",
    "DecoderInputEmbedding",
    
    # 解码器
    "UGTDecoder",
    "UGTDecoderForInference",
    
    # 生成器
    "BeamSearchGenerator",
    "DiverseBeamSearchGenerator",
    "NucleusSamplingGenerator",
]


def get_decoder(
    config: DecoderConfig = None,
    pretrained_path: str = None,
) -> UGTDecoder:
    """
    获取解码器实例的工厂函数
    
    Args:
        config: 解码器配置，如果为 None 则使用默认配置
        pretrained_path: 预训练模型路径，如果提供则加载权重
    
    Returns:
        UGTDecoder 实例
    """
    if config is None:
        config = DecoderConfig()
    
    decoder = UGTDecoder(config)
    
    if pretrained_path is not None:
        import torch
        state_dict = torch.load(pretrained_path, map_location="cpu")
        decoder.load_state_dict(state_dict)
    
    return decoder


def get_generator(
    decoder: UGTDecoder,
    generator_type: str = "beam_search",
) -> object:
    """
    获取生成器实例的工厂函数
    
    Args:
        decoder: UGT 解码器实例
        generator_type: 生成器类型
            - "beam_search": 标准 Beam Search
            - "diverse_beam_search": 多样性 Beam Search
            - "nucleus": 核采样
    
    Returns:
        生成器实例
    """
    config = decoder.config
    
    if generator_type == "beam_search":
        return BeamSearchGenerator(decoder, config)
    elif generator_type == "diverse_beam_search":
        return DiverseBeamSearchGenerator(decoder, config)
    elif generator_type == "nucleus":
        return NucleusSamplingGenerator(decoder, config)
    else:
        raise ValueError(f"未知的生成器类型: {generator_type}")

