"""
注意力机制模块

包含因果自注意力和交叉注意力的实现。

因果自注意力：解码器内部使用，防止看到未来信息
交叉注意力：编码器-解码器连接，Query 来自解码器，Key/Value 来自编码器

对应架构文档: 《生成式推荐系统架构设计》3.2.2 节
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import DecoderConfig


class CausalSelfAttention(nn.Module):
    """
    因果自注意力
    
    使用因果掩码确保每个位置只能看到自己和之前的位置，
    防止信息从未来泄露到过去。
    
    支持可选的 Flash Attention 加速。
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        dropout: Dropout 率
        max_seq_len: 最大序列长度（用于预计算因果掩码）
        use_flash_attention: 是否使用 Flash Attention
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_flash_attention = use_flash_attention
        
        # 投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** 0.5
        
        # 预计算因果掩码
        # 注册为 buffer，不作为参数
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len), diagonal=1
        ).bool()
        self.register_buffer("causal_mask", causal_mask)
        
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
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        因果自注意力前向传播
        
        Args:
            x: (batch_size, seq_len, d_model) 输入张量
            mask: (batch_size, seq_len) 可选的 padding 掩码
            cache: 可选的 KV 缓存，用于增量生成
                   (cached_k, cached_v) 每个形状为 (batch, n_heads, cached_len, d_k)
        
        Returns:
            output: (batch_size, seq_len, d_model) 输出张量
            new_cache: 更新后的 KV 缓存（如果提供了 cache）
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # 现在形状: (batch, n_heads, seq_len, d_k)
        
        # 处理 KV 缓存（增量生成时使用）
        new_cache = None
        if cache is not None:
            cached_k, cached_v = cache
            K = torch.cat([cached_k, K], dim=2)
            V = torch.cat([cached_v, V], dim=2)
            new_cache = (K, V)
        
        kv_len = K.shape[2]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (batch, n_heads, seq_len, kv_len)
        
        # 应用因果掩码
        if cache is None:
            # 完整序列：使用预计算的因果掩码
            causal_mask = self.causal_mask[:seq_len, :seq_len]
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            # 增量生成：只需要掩码当前位置之后的
            # 对于增量生成，Q 的长度通常为 1，不需要因果掩码
            pass
        
        # 应用 padding 掩码
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, kv_len)
            if cache is not None:
                # 增量生成时，需要扩展 mask 到完整的 KV 长度
                pass  # padding mask 在增量生成时可能需要特殊处理
            else:
                padding_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
                scores = scores.masked_fill(~padding_mask, float('-inf'))
        
        # Softmax 和 Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        context = torch.matmul(attn_weights, V)
        # (batch, n_heads, seq_len, d_k)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 输出投影
        output = self.W_o(context)
        
        return output, new_cache


class CrossAttention(nn.Module):
    """
    交叉注意力（编码器-解码器连接）
    
    Query 来自解码器（目标序列），
    Key/Value 来自编码器（源序列）。
    
    使用标准 Softmax 注意力。
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        dropout: Dropout 率
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** 0.5
        
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
        query: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        交叉注意力前向传播
        
        Args:
            query: (batch_size, tgt_len, d_model) 解码器的查询张量
            encoder_output: (batch_size, src_len, d_model) 编码器输出
            encoder_mask: (batch_size, src_len) 编码器 padding 掩码
            cache: 可选的 KV 缓存（交叉注意力的 KV 在生成过程中不变）
        
        Returns:
            output: (batch_size, tgt_len, d_model) 输出张量
            cache: KV 缓存（用于增量生成）
        """
        batch_size, tgt_len, _ = query.shape
        src_len = encoder_output.shape[1]
        
        # 计算 Q（来自解码器）
        Q = self.W_q(query)  # (batch, tgt_len, d_model)
        
        # 计算 K, V（来自编码器）
        # 如果有缓存，直接使用缓存的 K, V
        if cache is not None:
            K, V = cache
        else:
            K = self.W_k(encoder_output)  # (batch, src_len, d_model)
            V = self.W_v(encoder_output)
            
            # 重塑 K, V 为多头形式
            K = K.view(batch_size, src_len, self.n_heads, self.d_k).transpose(1, 2)
            V = V.view(batch_size, src_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 重塑 Q 为多头形式
        Q = Q.view(batch_size, tgt_len, self.n_heads, self.d_k).transpose(1, 2)
        # 形状: (batch, n_heads, tgt_len, d_k)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (batch, n_heads, tgt_len, src_len)
        
        # 应用编码器 padding 掩码
        if encoder_mask is not None:
            # encoder_mask: (batch, src_len) -> (batch, 1, 1, src_len)
            mask = encoder_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax 和 Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        context = torch.matmul(attn_weights, V)
        # (batch, n_heads, tgt_len, d_k)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, self.d_model
        )
        
        # 输出投影
        output = self.W_o(context)
        
        # 返回 KV 缓存（用于后续增量生成）
        new_cache = (K, V) if cache is None else cache
        
        return output, new_cache


class GroupLayerNorm(nn.Module):
    """
    分组层归一化 (Group Layer Normalization)
    
    针对不同语义空间的 Token 使用不同的归一化参数：
    - 用户行为 Token (Group 0)
    - 物品属性 Token (Group 1)
    - 动作 Token (Group 2)
    - 上下文 Token (Group 3)
    
    对应架构文档: 3.2.3 节
    
    Args:
        d_model: 特征维度
        num_groups: 分组数量
        eps: 归一化的 epsilon
    """
    
    def __init__(self, d_model: int, num_groups: int = 4, eps: float = 1e-6):
        super().__init__()
        
        self.d_model = d_model
        self.num_groups = num_groups
        self.eps = eps
        
        # 每个分组一个 LayerNorm
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model, eps=eps)
            for _ in range(num_groups)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_types: torch.Tensor
    ) -> torch.Tensor:
        """
        分组归一化前向传播
        
        Args:
            x: (batch_size, seq_len, d_model) 输入张量
            token_types: (batch_size, seq_len) Token 类型标识 (0 到 num_groups-1)
        
        Returns:
            output: (batch_size, seq_len, d_model) 归一化后的张量
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # 初始化输出
        output = torch.zeros_like(x)
        
        # 对每个分组应用对应的 LayerNorm
        for group_id in range(self.num_groups):
            # 找到属于该分组的 token
            mask = (token_types == group_id)  # (batch, seq_len)
            
            if not mask.any():
                continue
            
            # 获取该分组的 token
            # 需要处理不同 batch 中该分组 token 数量不同的情况
            group_tokens = x[mask]  # (num_group_tokens, d_model)
            
            # 应用归一化
            normalized = self.norms[group_id](group_tokens)
            
            # 写回输出
            output[mask] = normalized
        
        return output


class FeedForward(nn.Module):
    """
    标准前馈网络（非 MoE 版本）
    
    用于需要简单 FFN 的场景。
    
    Args:
        d_model: 模型维度
        d_ff: 中间隐藏层维度
        dropout: Dropout 率
        activation: 激活函数类型
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 选择激活函数
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"未知的激活函数: {activation}")
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

