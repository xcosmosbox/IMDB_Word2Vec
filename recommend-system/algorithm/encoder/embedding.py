"""
统一输入嵌入层模块

该模块实现了 UGT 编码器的输入嵌入层，将各种类型的输入统一编码为隐藏向量。

嵌入公式:
    E_input = E_semantic + E_position + E_type + E_time

其中:
    - E_semantic: 三层语义 ID 嵌入的拼接后投影
    - E_position: 位置嵌入
    - E_type: Token 类型嵌入
    - E_time: 时间特征嵌入（可选）

对应架构文档: 
- 3.1 节 Unified Input Embedding
- 4.1 节 事件序列化

Author: Person B
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .config import EncoderConfig


class SemanticEmbedding(nn.Module):
    """
    语义 ID 嵌入层
    
    将三层语义 ID 分别嵌入后拼接，然后投影到 d_model 维度。
    
    三层语义 ID 的含义:
    - Level 1: 粗粒度类目 (如电影类型、商品大类)
    - Level 2: 细粒度属性 (如子类目、品牌)
    - Level 3: 实例区分 (用于区分具体物品)
    
    Attributes:
        embeddings: 三层语义 ID 的嵌入表
        projection: 拼接后的投影层
    """
    
    def __init__(self, config: EncoderConfig):
        """
        初始化语义嵌入层
        
        Args:
            config: 编码器配置对象
        """
        super().__init__()
        self.config = config
        
        # 为三层语义 ID 分别创建嵌入表
        # 每层嵌入维度约为 d_model // 3
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=codebook_size,
                embedding_dim=embed_dim,
            )
            for codebook_size, embed_dim in zip(
                config.codebook_sizes, 
                config.semantic_dims
            )
        ])
        
        # 拼接后的总维度
        total_dim = sum(config.semantic_dims)
        
        # 投影层：将拼接后的嵌入投影到 d_model 维度
        # 如果 total_dim 正好等于 d_model，这个投影可以保持原样或做非线性变换
        self.projection = nn.Linear(total_dim, config.d_model)
        
        # 初始化嵌入权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化嵌入权重"""
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def forward(
        self, 
        semantic_ids: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            semantic_ids: 三层语义 ID 列表 [L1_ids, L2_ids, L3_ids]
                         每个张量形状为 (batch_size, seq_len)
        
        Returns:
            语义嵌入向量，形状为 (batch_size, seq_len, d_model)
        """
        # 分别获取三层嵌入
        embeddings = []
        for emb_layer, ids in zip(self.embeddings, semantic_ids):
            embeddings.append(emb_layer(ids))
        
        # 在最后一个维度拼接
        # (batch, seq_len, dim1) + (batch, seq_len, dim2) + (batch, seq_len, dim3)
        # -> (batch, seq_len, dim1 + dim2 + dim3)
        concat_emb = torch.cat(embeddings, dim=-1)
        
        # 投影到 d_model 维度
        output = self.projection(concat_emb)
        
        return output


class PositionalEmbedding(nn.Module):
    """
    位置嵌入层
    
    使用可学习的位置嵌入表示序列中每个位置的信息。
    
    Attributes:
        embedding: 位置嵌入表
    """
    
    def __init__(self, config: EncoderConfig):
        """
        初始化位置嵌入层
        
        Args:
            config: 编码器配置对象
        """
        super().__init__()
        self.config = config
        
        # 可学习的位置嵌入
        self.embedding = nn.Embedding(
            num_embeddings=config.max_seq_len,
            embedding_dim=config.d_model,
        )
        
        # 初始化权重
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            positions: 位置索引，形状为 (batch_size, seq_len)
                      值域为 [0, max_seq_len)
        
        Returns:
            位置嵌入向量，形状为 (batch_size, seq_len, d_model)
        """
        return self.embedding(positions)


class TokenTypeEmbedding(nn.Module):
    """
    Token 类型嵌入层
    
    为不同类型的 Token 提供类型信息嵌入。
    
    Token 类型:
    - 0: USER Token (用户属性)
    - 1: ITEM Token (物品语义 ID)
    - 2: ACTION Token (行为类型)
    - 3: CONTEXT Token (上下文信息)
    
    Attributes:
        embedding: Token 类型嵌入表
    """
    
    def __init__(self, config: EncoderConfig):
        """
        初始化 Token 类型嵌入层
        
        Args:
            config: 编码器配置对象
        """
        super().__init__()
        self.config = config
        
        # Token 类型嵌入
        self.embedding = nn.Embedding(
            num_embeddings=config.num_token_types,
            embedding_dim=config.d_model,
        )
        
        # 初始化权重
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, token_types: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            token_types: Token 类型索引，形状为 (batch_size, seq_len)
                        值域为 [0, num_token_types)
        
        Returns:
            Token 类型嵌入向量，形状为 (batch_size, seq_len, d_model)
        """
        return self.embedding(token_types)


class TimeEmbedding(nn.Module):
    """
    时间特征嵌入层
    
    将时间特征向量投影到 d_model 维度。
    时间特征可以包含:
    - 绝对时间（年、月、日、时）
    - 相对时间（距今多久）
    - 周期特征（星期几、节假日等）
    
    Attributes:
        projection: 时间特征投影层
    """
    
    def __init__(self, config: EncoderConfig):
        """
        初始化时间特征嵌入层
        
        Args:
            config: 编码器配置对象
        """
        super().__init__()
        self.config = config
        
        if config.time_dim > 0:
            # 时间特征投影层
            self.projection = nn.Linear(config.time_dim, config.d_model)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        else:
            self.projection = None
    
    def forward(
        self, 
        time_features: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        前向传播
        
        Args:
            time_features: 时间特征向量，形状为 (batch_size, seq_len, time_dim)
                          如果为 None 或 time_dim=0，则返回 None
        
        Returns:
            时间嵌入向量，形状为 (batch_size, seq_len, d_model)
            如果不使用时间特征，返回 None
        """
        if time_features is None or self.projection is None:
            return None
        
        return self.projection(time_features)


class InputEmbedding(nn.Module):
    """
    统一输入嵌入层
    
    将所有输入信号（语义 ID、位置、类型、时间）统一编码为隐藏向量。
    
    嵌入公式:
        E_input = E_semantic + E_position + E_type + E_time
    
    处理流程:
    1. 将三层语义 ID 分别嵌入后拼接并投影
    2. 添加位置嵌入
    3. 添加 Token 类型嵌入
    4. 添加时间特征嵌入（可选）
    5. Layer Normalization
    6. Dropout
    
    Attributes:
        semantic_embedding: 语义 ID 嵌入层
        position_embedding: 位置嵌入层
        type_embedding: Token 类型嵌入层
        time_embedding: 时间特征嵌入层
        layer_norm: 层归一化
        dropout: Dropout 层
    """
    
    def __init__(self, config: EncoderConfig):
        """
        初始化统一输入嵌入层
        
        Args:
            config: 编码器配置对象
        """
        super().__init__()
        self.config = config
        
        # 语义 ID 嵌入（三层拼接后投影）
        self.semantic_embedding = SemanticEmbedding(config)
        
        # 位置嵌入
        self.position_embedding = PositionalEmbedding(config)
        
        # Token 类型嵌入
        self.type_embedding = TokenTypeEmbedding(config)
        
        # 时间特征嵌入（可选）
        self.time_embedding = TimeEmbedding(config)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(
            config.d_model, 
            eps=config.layer_norm_eps
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        semantic_ids: List[torch.Tensor],
        positions: torch.Tensor,
        token_types: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            semantic_ids: 三层语义 ID 列表 [L1_ids, L2_ids, L3_ids]
                         每个张量形状为 (batch_size, seq_len)
            positions: 位置索引，形状为 (batch_size, seq_len)
            token_types: Token 类型索引，形状为 (batch_size, seq_len)
            time_features: 时间特征向量，形状为 (batch_size, seq_len, time_dim)
                          可选，如果不使用时间特征可传 None
        
        Returns:
            输入嵌入向量，形状为 (batch_size, seq_len, d_model)
        
        Example:
            >>> config = EncoderConfig(d_model=512)
            >>> embedding = InputEmbedding(config)
            >>> batch_size, seq_len = 32, 100
            >>> l1_ids = torch.randint(0, 1024, (batch_size, seq_len))
            >>> l2_ids = torch.randint(0, 4096, (batch_size, seq_len))
            >>> l3_ids = torch.randint(0, 16384, (batch_size, seq_len))
            >>> positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            >>> token_types = torch.randint(0, 4, (batch_size, seq_len))
            >>> output = embedding([l1_ids, l2_ids, l3_ids], positions, token_types)
            >>> output.shape
            torch.Size([32, 100, 512])
        """
        # 语义嵌入
        semantic_emb = self.semantic_embedding(semantic_ids)
        
        # 位置嵌入
        position_emb = self.position_embedding(positions)
        
        # 类型嵌入
        type_emb = self.type_embedding(token_types)
        
        # 组合各嵌入（加法）
        embeddings = semantic_emb + position_emb + type_emb
        
        # 时间特征嵌入（可选）
        time_emb = self.time_embedding(time_features)
        if time_emb is not None:
            embeddings = embeddings + time_emb
        
        # 层归一化
        embeddings = self.layer_norm(embeddings)
        
        # Dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.config.d_model

