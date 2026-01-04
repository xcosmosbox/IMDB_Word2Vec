"""
UGT 编码器模块

该模块实现了 Unified Generative Transformer (UGT) 的编码器部分，
负责将用户历史行为序列编码为用户表示向量。

主要组件:
- EncoderConfig: 编码器配置类
- InputEmbedding: 统一输入嵌入层
- DotProductAggregatedAttention: 点积聚合注意力 (ReLU 替代 Softmax)
- GroupLayerNorm: 分组层归一化
- EncoderLayer: 单层编码器
- UGTEncoder: 完整编码器

核心创新点:
1. Dot-Product Aggregated Attention (来自 Meta HSTU)
   - 使用 ReLU 替代 Softmax，避免归一化导致的信息损失
   - 更适合处理推荐场景中的非平稳词汇表

2. Group Layer Normalization (来自美团 MTGR)
   - 对不同类型的 Token 使用不同的归一化参数
   - 增强不同语义空间的编码能力

使用示例:
    >>> from algorithm.encoder import EncoderConfig, UGTEncoder
    >>> 
    >>> # 创建配置
    >>> config = EncoderConfig(d_model=512, n_heads=16, n_layers=12)
    >>> 
    >>> # 创建编码器
    >>> encoder = UGTEncoder(config)
    >>> 
    >>> # 准备输入
    >>> batch_size, seq_len = 32, 100
    >>> l1_ids = torch.randint(0, 1024, (batch_size, seq_len))
    >>> l2_ids = torch.randint(0, 4096, (batch_size, seq_len))
    >>> l3_ids = torch.randint(0, 16384, (batch_size, seq_len))
    >>> positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    >>> token_types = torch.randint(0, 4, (batch_size, seq_len))
    >>> attention_mask = torch.ones(batch_size, seq_len)
    >>> 
    >>> # 获取用户表示
    >>> user_repr = encoder([l1_ids, l2_ids, l3_ids], positions, token_types, attention_mask)
    >>> print(user_repr.shape)  # torch.Size([32, 512])
    >>> 
    >>> # 获取序列输出（用于解码器）
    >>> seq_output = encoder.get_sequence_output([l1_ids, l2_ids, l3_ids], positions, token_types, attention_mask)
    >>> print(seq_output.shape)  # torch.Size([32, 100, 512])

对应架构文档:
- 第三章 核心模型架构：Unified Generative Transformer (UGT)
- 3.2.2 节 点积聚合注意力
- 3.2.3 节 Group Layer Normalization

Author: Person B
"""

# 配置类
from .config import EncoderConfig

# 嵌入层
from .embedding import (
    InputEmbedding,
    SemanticEmbedding,
    PositionalEmbedding,
    TokenTypeEmbedding,
    TimeEmbedding,
)

# 注意力层
from .attention import (
    DotProductAggregatedAttention,
    MultiHeadSelfAttention,
    SoftmaxAttention,
    create_attention_layer,
)

# 层归一化
from .layer_norm import (
    GroupLayerNorm,
    ConditionalLayerNorm,
    StandardLayerNorm,
    create_layer_norm,
)

# 前馈网络
from .ffn import (
    FeedForwardNetwork,
    GatedLinearUnit,
    SwiGLU,
    EncoderFFN,
    create_ffn,
    get_activation,
)

# 编码器层
from .encoder_layer import (
    EncoderLayer,
    PreNormEncoderLayer,
    EncoderLayerWithCrossAttention,
    create_encoder_layer,
)

# 完整编码器
from .encoder import (
    UGTEncoder,
    Pooler,
    MultiScaleEncoder,
)

# 版本信息
__version__ = "0.1.0"
__author__ = "Person B"

# 公开的类和函数
__all__ = [
    # 配置
    "EncoderConfig",
    
    # 嵌入
    "InputEmbedding",
    "SemanticEmbedding",
    "PositionalEmbedding",
    "TokenTypeEmbedding",
    "TimeEmbedding",
    
    # 注意力
    "DotProductAggregatedAttention",
    "MultiHeadSelfAttention",
    "SoftmaxAttention",
    "create_attention_layer",
    
    # 层归一化
    "GroupLayerNorm",
    "ConditionalLayerNorm",
    "StandardLayerNorm",
    "create_layer_norm",
    
    # 前馈网络
    "FeedForwardNetwork",
    "GatedLinearUnit",
    "SwiGLU",
    "EncoderFFN",
    "create_ffn",
    "get_activation",
    
    # 编码器层
    "EncoderLayer",
    "PreNormEncoderLayer",
    "EncoderLayerWithCrossAttention",
    "create_encoder_layer",
    
    # 完整编码器
    "UGTEncoder",
    "Pooler",
    "MultiScaleEncoder",
]


def get_encoder(config: EncoderConfig = None) -> UGTEncoder:
    """
    获取编码器实例的便捷函数
    
    Args:
        config: 编码器配置，如果为 None 则使用默认配置
    
    Returns:
        UGTEncoder 实例
    """
    if config is None:
        config = EncoderConfig.base()
    return UGTEncoder(config)


def get_small_encoder() -> UGTEncoder:
    """获取小规模编码器（用于调试）"""
    return UGTEncoder(EncoderConfig.small())


def get_base_encoder() -> UGTEncoder:
    """获取基础规模编码器"""
    return UGTEncoder(EncoderConfig.base())


def get_large_encoder() -> UGTEncoder:
    """获取大规模编码器（生产环境）"""
    return UGTEncoder(EncoderConfig.large())

