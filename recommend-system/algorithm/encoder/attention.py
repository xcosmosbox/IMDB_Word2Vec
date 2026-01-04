"""
点积聚合注意力模块 (Dot-Product Aggregated Attention)

该模块实现了来自 Meta HSTU 论文的核心注意力机制，
使用 ReLU 替代 Softmax，更适合推荐场景中的非平稳词汇表。

核心创新点:
1. 使用 ReLU 替代 Softmax，避免归一化导致的信息损失
2. 允许注意力权重为 0（完全忽略不相关项）
3. 更适合处理推荐场景中的非平稳词汇表

对应架构文档:
- 2.4 节 点积聚合注意力
- 3.2.2 节 Dot-Product Aggregated Attention

Author: Person B
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import EncoderConfig


class DotProductAggregatedAttention(nn.Module):
    """
    点积聚合注意力 (来自 HSTU)
    
    核心特点: 使用 ReLU 替代 Softmax
    
    公式:
        Attention(Q, K, V) = ReLU(QK^T / √d) · V
    
    优势:
    1. 避免 Softmax 的归一化导致信息损失
       - Softmax 必须分配权重，即使所有项都不相关
       - ReLU 允许所有权重都为 0
    
    2. 更适合推荐场景中的非平稳词汇表
       - 物品分布随时间变化
       - 不需要重新归一化注意力分布
    
    3. 允许注意力权重为 0（忽略不相关项）
       - 提高模型的稀疏性
       - 减少噪声信号的影响
    
    Attributes:
        d_model: 模型隐藏维度
        n_heads: 注意力头数
        d_k: 每个头的维度 (d_model // n_heads)
        W_q: Query 投影矩阵
        W_k: Key 投影矩阵
        W_v: Value 投影矩阵
        W_o: 输出投影矩阵
        scale: 缩放因子 (√d_k)
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        """
        初始化点积聚合注意力层
        
        Args:
            d_model: 模型隐藏维度
            n_heads: 注意力头数
            dropout: 输出 Dropout 概率
            attention_dropout: 注意力权重 Dropout 概率
        """
        super().__init__()
        
        # 验证 d_model 能被 n_heads 整除
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除"
            )
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Query, Key, Value 投影层
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        
        # 输出投影层
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        
        # Dropout 层
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = math.sqrt(self.d_k)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 注意力掩码，形状为 (batch_size, seq_len) 或 
                  (batch_size, seq_len, seq_len) 或 (batch_size, 1, seq_len)
                  1 表示有效位置，0 表示需要掩码的位置
            return_attention: 是否返回注意力权重
        
        Returns:
            output: 注意力输出，形状为 (batch_size, seq_len, d_model)
            attention_weights: 如果 return_attention=True，返回注意力权重
                              形状为 (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # 线性变换并重塑为多头形式
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # 转置为 (batch, n_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 计算注意力分数
        # (batch, n_heads, seq_len, d_k) @ (batch, n_heads, d_k, seq_len)
        # -> (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码
        # 将无效位置设为负无穷，ReLU 后变为 0
        if mask is not None:
            # 处理不同维度的掩码
            if mask.dim() == 2:
                # (batch, seq_len) -> (batch, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                # 或 (batch, 1, seq_len) -> (batch, 1, 1, seq_len)
                if mask.size(1) != 1:
                    mask = mask.unsqueeze(1)
                else:
                    mask = mask.unsqueeze(2)
            
            # 将掩码为 0 的位置设为负无穷
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # ReLU 替代 Softmax（核心创新）
        # 负值变为 0，正值保留
        attn_weights = F.relu(scores)
        
        # Dropout
        attn_weights = self.attention_dropout(attn_weights)
        
        # 聚合 Value
        # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, d_k)
        # -> (batch, n_heads, seq_len, d_k)
        context = torch.matmul(attn_weights, V)
        
        # 转置并重塑
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k)
        context = context.transpose(1, 2).contiguous()
        # (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
        context = context.view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.W_o(context)
        output = self.output_dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class MultiHeadSelfAttention(DotProductAggregatedAttention):
    """
    多头自注意力层
    
    继承自 DotProductAggregatedAttention，提供更清晰的接口。
    
    这个类是 DotProductAggregatedAttention 的别名，
    方便在代码中使用更通用的名称。
    """
    
    @classmethod
    def from_config(cls, config: EncoderConfig) -> "MultiHeadSelfAttention":
        """
        从配置创建注意力层
        
        Args:
            config: 编码器配置对象
            
        Returns:
            MultiHeadSelfAttention 实例
        """
        return cls(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
        )


class SoftmaxAttention(nn.Module):
    """
    标准 Softmax 注意力（用于对比实验）
    
    这是传统的 Transformer 注意力机制，使用 Softmax 归一化。
    提供用于对比 ReLU 注意力效果的基线实现。
    
    公式:
        Attention(Q, K, V) = Softmax(QK^T / √d) · V
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        """
        初始化 Softmax 注意力层
        
        Args:
            d_model: 模型隐藏维度
            n_heads: 注意力头数
            dropout: 输出 Dropout 概率
            attention_dropout: 注意力权重 Dropout 概率
        """
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除"
            )
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.d_k)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 注意力掩码
            return_attention: 是否返回注意力权重
        
        Returns:
            output: 注意力输出
            attention_weights: 注意力权重（可选）
        """
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3 and mask.size(1) != 1:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 使用 Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.W_o(context)
        output = self.output_dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


def create_attention_layer(
    config: EncoderConfig,
    use_relu: bool = True,
) -> nn.Module:
    """
    创建注意力层的工厂函数
    
    Args:
        config: 编码器配置
        use_relu: 是否使用 ReLU 注意力（默认 True）
                 如果为 False，使用标准 Softmax 注意力
    
    Returns:
        注意力层实例
    """
    if use_relu:
        return DotProductAggregatedAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
        )
    else:
        return SoftmaxAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
        )

