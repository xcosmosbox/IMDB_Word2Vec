"""
Semantic ID 编码器模块

该模块实现了基于 RQ-VAE（残差向量量化）的语义 ID 编码器，
用于将连续的物品特征向量编码为层次化的离散语义 ID。

主要组件：
- SemanticIDConfig: 配置类
- VectorQuantizer: 单层向量量化器
- ResidualVectorQuantizer: 残差向量量化器主体
- SemanticIDEncoder: 完整的语义 ID 编码器实现
- SemanticIDTrainer: 码本训练器

使用示例：
    >>> from algorithm.semantic_id import SemanticIDEncoder, SemanticIDConfig
    >>> 
    >>> # 创建配置
    >>> config = SemanticIDConfig()
    >>> 
    >>> # 创建编码器
    >>> encoder = SemanticIDEncoder(config)
    >>> 
    >>> # 编码物品特征
    >>> features = torch.randn(32, 256)  # batch_size=32, embedding_dim=256
    >>> l1_ids, l2_ids, l3_ids = encoder.encode(features)
    >>> 
    >>> # 解码重建特征
    >>> reconstructed = encoder.decode(l1_ids, l2_ids, l3_ids)

作者: Person A
版本: 1.0.0
"""

from .config import SemanticIDConfig
from .codebook import VectorQuantizer
from .rq_vae import ResidualVectorQuantizer
from .encoder import SemanticIDEncoder
from .trainer import SemanticIDTrainer

__all__ = [
    "SemanticIDConfig",
    "VectorQuantizer",
    "ResidualVectorQuantizer",
    "SemanticIDEncoder",
    "SemanticIDTrainer",
]

__version__ = "1.0.0"
__author__ = "Person A"

