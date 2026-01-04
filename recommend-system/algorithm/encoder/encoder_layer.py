"""
编码器层模块

该模块实现了单层 UGT 编码器，包含:
1. Dot-Product Aggregated Attention (自注意力)
2. 残差连接
3. Group Layer Normalization
4. 前馈网络 (FFN)

编码器层结构:
    x → Attention → Add & GLN → FFN → Add & GLN → output

对应架构文档:
- 3.1 节 UGT 总体结构
- Encoder (Multi-Scale) 部分

Author: Person B
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import EncoderConfig
from .attention import DotProductAggregatedAttention
from .layer_norm import GroupLayerNorm
from .ffn import FeedForwardNetwork, EncoderFFN


class EncoderLayer(nn.Module):
    """
    单层编码器
    
    结构: x → Attention → Add & GLN → FFN → Add & GLN → output
    
    这是 UGT 编码器的基本构建块，包含:
    1. 自注意力层 (Dot-Product Aggregated Attention)
    2. 第一个残差连接 + Group Layer Normalization
    3. 前馈网络 (FFN)
    4. 第二个残差连接 + Group Layer Normalization
    
    使用 Post-Norm 架构（先计算后归一化）。
    
    Attributes:
        attention: 自注意力层
        ffn: 前馈网络
        norm1: 第一个 Group Layer Normalization
        norm2: 第二个 Group Layer Normalization
        dropout: 残差连接的 Dropout
    
    Example:
        >>> config = EncoderConfig(d_model=512, n_heads=16)
        >>> layer = EncoderLayer(config)
        >>> x = torch.randn(32, 100, 512)
        >>> token_types = torch.randint(0, 4, (32, 100))
        >>> mask = torch.ones(32, 100)
        >>> output = layer(x, token_types, mask)
        >>> output.shape
        torch.Size([32, 100, 512])
    """
    
    def __init__(self, config: EncoderConfig):
        """
        初始化编码器层
        
        Args:
            config: 编码器配置对象
        """
        super().__init__()
        self.config = config
        
        # 自注意力层
        self.attention = DotProductAggregatedAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        
        # Group Layer Normalization
        self.norm1 = GroupLayerNorm(
            d_model=config.d_model, 
            num_groups=config.num_groups,
            eps=config.layer_norm_eps,
        )
        self.norm2 = GroupLayerNorm(
            d_model=config.d_model, 
            num_groups=config.num_groups,
            eps=config.layer_norm_eps,
        )
        
        # 残差连接的 Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # 初始化 FFN 权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        token_types: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            token_types: Token 类型索引，形状为 (batch_size, seq_len)
                        用于 Group Layer Normalization
            mask: 注意力掩码，形状为 (batch_size, seq_len)
                  1 表示有效位置，0 表示 padding
            return_attention: 是否返回注意力权重
        
        Returns:
            output: 输出张量，形状为 (batch_size, seq_len, d_model)
            attention_weights: 注意力权重（如果 return_attention=True）
        """
        # Self-Attention + Residual + GLN
        attn_out, attn_weights = self.attention(x, mask, return_attention)
        x = self.norm1(x + self.dropout(attn_out), token_types)
        
        # FFN + Residual + GLN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out, token_types)
        
        if return_attention:
            return x, attn_weights
        return x, None


class PreNormEncoderLayer(nn.Module):
    """
    Pre-Norm 编码器层
    
    结构: x → GLN → Attention → Add → GLN → FFN → Add → output
    
    Pre-Norm 架构在归一化后再进行计算，通常具有更好的梯度流动性。
    适用于更深的模型。
    
    Attributes:
        attention: 自注意力层
        ffn: 前馈网络
        norm1: 第一个 Group Layer Normalization
        norm2: 第二个 Group Layer Normalization
        dropout: 残差连接的 Dropout
    """
    
    def __init__(self, config: EncoderConfig):
        """
        初始化 Pre-Norm 编码器层
        
        Args:
            config: 编码器配置对象
        """
        super().__init__()
        self.config = config
        
        # 自注意力层
        self.attention = DotProductAggregatedAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        
        # Group Layer Normalization (Pre-Norm)
        self.norm1 = GroupLayerNorm(
            d_model=config.d_model, 
            num_groups=config.num_groups,
            eps=config.layer_norm_eps,
        )
        self.norm2 = GroupLayerNorm(
            d_model=config.d_model, 
            num_groups=config.num_groups,
            eps=config.layer_norm_eps,
        )
        
        # 残差连接的 Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # 初始化 FFN 权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        token_types: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            token_types: Token 类型索引，形状为 (batch_size, seq_len)
            mask: 注意力掩码
            return_attention: 是否返回注意力权重
        
        Returns:
            output: 输出张量
            attention_weights: 注意力权重（可选）
        """
        # Pre-Norm: GLN → Attention → Add
        normed_x = self.norm1(x, token_types)
        attn_out, attn_weights = self.attention(normed_x, mask, return_attention)
        x = x + self.dropout(attn_out)
        
        # Pre-Norm: GLN → FFN → Add
        normed_x = self.norm2(x, token_types)
        ffn_out = self.ffn(normed_x)
        x = x + ffn_out
        
        if return_attention:
            return x, attn_weights
        return x, None


class EncoderLayerWithCrossAttention(nn.Module):
    """
    带交叉注意力的编码器层
    
    除了自注意力，还支持与外部上下文的交叉注意力。
    可用于多尺度编码或条件编码场景。
    
    结构:
        x → Self-Attn → Add & GLN → Cross-Attn → Add & GLN → FFN → Add & GLN → output
    """
    
    def __init__(self, config: EncoderConfig):
        """
        初始化带交叉注意力的编码器层
        
        Args:
            config: 编码器配置对象
        """
        super().__init__()
        self.config = config
        
        # 自注意力层
        self.self_attention = DotProductAggregatedAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
        )
        
        # 交叉注意力层
        self.cross_attention = DotProductAggregatedAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        
        # Group Layer Normalization
        self.norm1 = GroupLayerNorm(config.d_model, config.num_groups)
        self.norm2 = GroupLayerNorm(config.d_model, config.num_groups)
        self.norm3 = GroupLayerNorm(config.d_model, config.num_groups)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        token_types: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            token_types: Token 类型索引
            context: 交叉注意力的上下文（如果为 None，跳过交叉注意力）
            self_mask: 自注意力掩码
            cross_mask: 交叉注意力掩码
        
        Returns:
            输出张量
        """
        # Self-Attention
        attn_out, _ = self.self_attention(x, self_mask)
        x = self.norm1(x + self.dropout(attn_out), token_types)
        
        # Cross-Attention (if context provided)
        if context is not None:
            # 使用 context 作为 key 和 value
            cross_out, _ = self.cross_attention(x, cross_mask)
            x = self.norm2(x + self.dropout(cross_out), token_types)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out, token_types)
        
        return x


def create_encoder_layer(
    config: EncoderConfig,
    use_pre_norm: bool = False,
) -> nn.Module:
    """
    创建编码器层的工厂函数
    
    Args:
        config: 编码器配置
        use_pre_norm: 是否使用 Pre-Norm 架构（默认 False，使用 Post-Norm）
    
    Returns:
        编码器层实例
    """
    if use_pre_norm:
        return PreNormEncoderLayer(config)
    else:
        return EncoderLayer(config)

