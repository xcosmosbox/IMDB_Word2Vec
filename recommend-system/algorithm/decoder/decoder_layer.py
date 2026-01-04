"""
解码器层模块

实现单层解码器的结构，包括：
1. 因果自注意力
2. 交叉注意力（编码器-解码器连接）
3. MoE 增强的前馈网络
4. 分组层归一化

对应架构文档: 《生成式推荐系统架构设计》第三章
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .config import DecoderConfig
from .cross_attention import CausalSelfAttention, CrossAttention, GroupLayerNorm
from .moe import MoEFeedForward


class DecoderLayer(nn.Module):
    """
    单层解码器
    
    结构:
    x → Causal Self-Attention → Add & GLN
      → Cross Attention → Add & GLN
      → MoE FFN → Add & GLN → output
    
    每层都返回 MoE 的辅助损失，用于训练时的负载均衡。
    
    Args:
        config: 解码器配置
        layer_idx: 层索引（用于梯度检查点等）
    """
    
    def __init__(self, config: DecoderConfig, layer_idx: int = 0):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        
        # 因果自注意力
        self.self_attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            use_flash_attention=config.use_flash_attention,
        )
        
        # 交叉注意力（编码器-解码器连接）
        self.cross_attn = CrossAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        
        # MoE FFN
        self.moe_ffn = MoEFeedForward(config)
        
        # 分组层归一化
        self.norm1 = GroupLayerNorm(config.d_model, config.num_groups, config.layer_norm_eps)
        self.norm2 = GroupLayerNorm(config.d_model, config.num_groups, config.layer_norm_eps)
        self.norm3 = GroupLayerNorm(config.d_model, config.num_groups, config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        token_types: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        self_attn_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_attn_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple], Optional[Tuple]]:
        """
        单层解码器前向传播
        
        Args:
            x: (batch_size, tgt_len, d_model) 解码器输入
            encoder_output: (batch_size, src_len, d_model) 编码器输出
            token_types: (batch_size, tgt_len) Token 类型标识
            self_attn_mask: (batch_size, tgt_len) 自注意力 padding 掩码
            cross_attn_mask: (batch_size, src_len) 交叉注意力 padding 掩码
            self_attn_cache: 自注意力 KV 缓存
            cross_attn_cache: 交叉注意力 KV 缓存
        
        Returns:
            output: (batch_size, tgt_len, d_model) 输出张量
            aux_loss: MoE 辅助损失（标量）
            new_self_attn_cache: 更新后的自注意力 KV 缓存
            new_cross_attn_cache: 交叉注意力 KV 缓存
        """
        residual = x
        
        # 1. 因果自注意力
        attn_out, new_self_attn_cache = self.self_attn(
            x, mask=self_attn_mask, cache=self_attn_cache
        )
        x = residual + self.dropout(attn_out)
        x = self.norm1(x, token_types)
        
        # 2. 交叉注意力
        residual = x
        cross_out, new_cross_attn_cache = self.cross_attn(
            query=x,
            encoder_output=encoder_output,
            encoder_mask=cross_attn_mask,
            cache=cross_attn_cache,
        )
        x = residual + self.dropout(cross_out)
        x = self.norm2(x, token_types)
        
        # 3. MoE FFN
        residual = x
        ffn_out = self.moe_ffn(x)
        x = residual + ffn_out  # MoE 内部已包含 dropout
        x = self.norm3(x, token_types)
        
        # 获取 MoE 辅助损失
        aux_loss = self.moe_ffn.get_aux_loss()
        
        return x, aux_loss, new_self_attn_cache, new_cross_attn_cache


class DecoderLayerWithoutCrossAttention(nn.Module):
    """
    不带交叉注意力的解码器层
    
    用于纯解码器架构（如 GPT 风格），或编码器部分。
    
    结构:
    x → Causal Self-Attention → Add & GLN
      → MoE FFN → Add & GLN → output
    
    Args:
        config: 解码器配置
        layer_idx: 层索引
    """
    
    def __init__(self, config: DecoderConfig, layer_idx: int = 0):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        
        # 因果自注意力
        self.self_attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            use_flash_attention=config.use_flash_attention,
        )
        
        # MoE FFN
        self.moe_ffn = MoEFeedForward(config)
        
        # 分组层归一化
        self.norm1 = GroupLayerNorm(config.d_model, config.num_groups, config.layer_norm_eps)
        self.norm2 = GroupLayerNorm(config.d_model, config.num_groups, config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        前向传播
        
        Args:
            x: (batch_size, seq_len, d_model) 输入
            token_types: (batch_size, seq_len) Token 类型
            attention_mask: (batch_size, seq_len) padding 掩码
            cache: KV 缓存
        
        Returns:
            output: (batch_size, seq_len, d_model) 输出
            aux_loss: MoE 辅助损失
            new_cache: 更新后的 KV 缓存
        """
        residual = x
        
        # 1. 因果自注意力
        attn_out, new_cache = self.self_attn(x, mask=attention_mask, cache=cache)
        x = residual + self.dropout(attn_out)
        x = self.norm1(x, token_types)
        
        # 2. MoE FFN
        residual = x
        ffn_out = self.moe_ffn(x)
        x = residual + ffn_out
        x = self.norm2(x, token_types)
        
        # 获取 MoE 辅助损失
        aux_loss = self.moe_ffn.get_aux_loss()
        
        return x, aux_loss, new_cache


class DecoderInputEmbedding(nn.Module):
    """
    解码器输入嵌入层
    
    将语义 ID 和位置信息转换为稠密向量表示。
    
    组成部分：
    - L1 语义 ID 嵌入
    - L2 语义 ID 嵌入
    - L3 语义 ID 嵌入
    - 位置编码
    - Token 类型嵌入
    
    最终嵌入 = L1_emb + L2_emb + L3_emb + pos_emb + type_emb
    
    Args:
        config: 解码器配置
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        
        self.config = config
        
        # 语义 ID 嵌入（三层码本）
        self.l1_embedding = nn.Embedding(
            config.codebook_sizes[0], 
            config.d_model, 
            padding_idx=config.pad_token_id
        )
        self.l2_embedding = nn.Embedding(
            config.codebook_sizes[1], 
            config.d_model, 
            padding_idx=config.pad_token_id
        )
        self.l3_embedding = nn.Embedding(
            config.codebook_sizes[2], 
            config.d_model, 
            padding_idx=config.pad_token_id
        )
        
        # 位置编码
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Token 类型嵌入
        self.token_type_embedding = nn.Embedding(config.num_token_types, config.d_model)
        
        # 层归一化和 Dropout
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化嵌入权重"""
        # 使用较小的标准差初始化
        std = 0.02
        nn.init.normal_(self.l1_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.l2_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.l3_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.token_type_embedding.weight, mean=0.0, std=std)
        
        # padding 位置置零
        if self.config.pad_token_id is not None:
            nn.init.zeros_(self.l1_embedding.weight[self.config.pad_token_id])
            nn.init.zeros_(self.l2_embedding.weight[self.config.pad_token_id])
            nn.init.zeros_(self.l3_embedding.weight[self.config.pad_token_id])
    
    def forward(
        self,
        semantic_ids: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        positions: torch.Tensor,
        token_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算输入嵌入
        
        Args:
            semantic_ids: (L1_ids, L2_ids, L3_ids) 每个形状为 (batch_size, seq_len)
            positions: (batch_size, seq_len) 位置索引
            token_types: (batch_size, seq_len) Token 类型
        
        Returns:
            embeddings: (batch_size, seq_len, d_model) 嵌入向量
        """
        l1_ids, l2_ids, l3_ids = semantic_ids
        
        # 计算各部分嵌入
        l1_emb = self.l1_embedding(l1_ids)
        l2_emb = self.l2_embedding(l2_ids)
        l3_emb = self.l3_embedding(l3_ids)
        pos_emb = self.position_embedding(positions)
        type_emb = self.token_type_embedding(token_types)
        
        # 组合嵌入
        embeddings = l1_emb + l2_emb + l3_emb + pos_emb + type_emb
        
        # 层归一化和 Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def forward_with_cache(
        self,
        l1_id: torch.Tensor,
        l2_id: torch.Tensor,
        l3_id: torch.Tensor,
        position: torch.Tensor,
        token_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        增量生成时的嵌入计算（单个 Token）
        
        Args:
            l1_id: (batch_size, 1) L1 语义 ID
            l2_id: (batch_size, 1) L2 语义 ID
            l3_id: (batch_size, 1) L3 语义 ID
            position: (batch_size, 1) 位置索引
            token_type: (batch_size, 1) Token 类型
        
        Returns:
            embedding: (batch_size, 1, d_model) 嵌入向量
        """
        l1_emb = self.l1_embedding(l1_id)
        l2_emb = self.l2_embedding(l2_id)
        l3_emb = self.l3_embedding(l3_id)
        pos_emb = self.position_embedding(position)
        type_emb = self.token_type_embedding(token_type)
        
        embedding = l1_emb + l2_emb + l3_emb + pos_emb + type_emb
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        
        return embedding

