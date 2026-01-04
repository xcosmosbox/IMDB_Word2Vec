"""
UGT 用户行为编码器模块

该模块实现了完整的 UGT Encoder，负责将用户历史行为序列编码为用户表示向量。

功能:
1. 将用户历史行为序列（语义 ID + 位置 + 类型 + 时间）编码为隐藏向量
2. 通过多层 Transformer 编码器层进行深度编码
3. 通过池化层提取最终的用户表示向量

核心特点:
1. Dot-Product Aggregated Attention (ReLU 替代 Softmax)
2. Group Layer Normalization (分组归一化)
3. 多层语义 ID 嵌入

对应架构文档:
- 3.1 节 UGT 总体结构
- 3.2 节 Encoder (Multi-Scale)

Author: Person B
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict

import sys
from pathlib import Path
# 添加父目录到路径以导入 interfaces
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import EncoderConfig
from .embedding import InputEmbedding
from .encoder_layer import EncoderLayer, PreNormEncoderLayer

# 尝试导入接口，如果失败则定义一个占位符
try:
    from algorithm.interfaces import UserEncoderInterface
except ImportError:
    # 定义一个空的基类作为占位符
    class UserEncoderInterface:
        pass


class Pooler(nn.Module):
    """
    池化层
    
    从编码器输出中提取用户表示向量。
    
    支持三种池化方式:
    1. cls: 使用第一个 Token (类似 BERT 的 [CLS])
    2. mean: 使用所有有效 Token 的平均值
    3. last: 使用最后一个有效 Token
    
    Attributes:
        pooler_type: 池化方式
        dense: 全连接层
        activation: 激活函数
    """
    
    def __init__(self, config: EncoderConfig):
        """
        初始化池化层
        
        Args:
            config: 编码器配置对象
        """
        super().__init__()
        self.config = config
        self.pooler_type = config.pooler_type
        
        # 全连接层
        self.dense = nn.Linear(config.d_model, config.d_model)
        
        # 激活函数
        self.activation = nn.Tanh()
        
        # 初始化权重
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 编码器输出，形状为 (batch_size, seq_len, d_model)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len)
                           1 表示有效位置，0 表示 padding
        
        Returns:
            用户表示向量，形状为 (batch_size, d_model)
        """
        if self.pooler_type == "cls":
            # 使用第一个 Token
            pooled = hidden_states[:, 0, :]
        
        elif self.pooler_type == "mean":
            # 使用有效 Token 的平均值
            if attention_mask is not None:
                # 扩展 mask 维度
                mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
                # 加权求和
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)
        
        elif self.pooler_type == "last":
            # 使用最后一个有效 Token
            if attention_mask is not None:
                # 找到每个序列最后一个有效位置
                seq_lengths = attention_mask.sum(dim=1).long() - 1
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                pooled = hidden_states[batch_indices, seq_lengths, :]
            else:
                pooled = hidden_states[:, -1, :]
        
        else:
            raise ValueError(f"不支持的池化方式: {self.pooler_type}")
        
        # 全连接层 + 激活
        pooled = self.dense(pooled)
        pooled = self.activation(pooled)
        
        return pooled


class UGTEncoder(nn.Module, UserEncoderInterface):
    """
    UGT 用户行为编码器
    
    将用户历史行为序列编码为用户表示向量。
    
    架构:
    1. 输入嵌入层: 将语义 ID + 位置 + 类型 + 时间编码为统一表示
    2. 编码器层堆叠: 多层 Transformer 编码器进行深度编码
    3. 池化层: 提取最终的用户表示向量
    
    实现了 UserEncoderInterface 接口，提供:
    - forward(): 返回用户表示向量
    - get_sequence_output(): 返回完整序列输出
    
    Attributes:
        config: 编码器配置
        input_embedding: 输入嵌入层
        layers: 编码器层列表
        final_norm: 最终的 Layer Normalization
        pooler: 池化层
    
    Example:
        >>> config = EncoderConfig(d_model=512, n_heads=16, n_layers=12)
        >>> encoder = UGTEncoder(config)
        >>> batch_size, seq_len = 32, 100
        >>> l1_ids = torch.randint(0, 1024, (batch_size, seq_len))
        >>> l2_ids = torch.randint(0, 4096, (batch_size, seq_len))
        >>> l3_ids = torch.randint(0, 16384, (batch_size, seq_len))
        >>> positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        >>> token_types = torch.randint(0, 4, (batch_size, seq_len))
        >>> attention_mask = torch.ones(batch_size, seq_len)
        >>> user_repr = encoder([l1_ids, l2_ids, l3_ids], positions, token_types, attention_mask)
        >>> user_repr.shape
        torch.Size([32, 512])
    """
    
    def __init__(self, config: EncoderConfig):
        """
        初始化 UGT 编码器
        
        Args:
            config: 编码器配置对象
        """
        super().__init__()
        self.config = config
        
        # 输入嵌入层
        self.input_embedding = InputEmbedding(config)
        
        # 编码器层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.n_layers)
        ])
        
        # 最终的 Layer Normalization (用于 Pre-Norm 架构)
        self.final_norm = nn.LayerNorm(
            config.d_model, 
            eps=config.layer_norm_eps
        )
        
        # 池化层（提取用户表示）
        self.pooler = Pooler(config)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 对 final_norm 进行初始化
        nn.init.ones_(self.final_norm.weight)
        nn.init.zeros_(self.final_norm.bias)
    
    def forward(
        self,
        semantic_ids: List[torch.Tensor],
        positions: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播，返回用户表示向量
        
        Args:
            semantic_ids: [L1_ids, L2_ids, L3_ids] 三层语义 ID
                         每个张量形状为 (batch_size, seq_len)
            positions: 位置索引，形状为 (batch_size, seq_len)
            token_types: Token 类型索引，形状为 (batch_size, seq_len)
                        (0=USER, 1=ITEM, 2=ACTION, 3=CONTEXT)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len)
                           1 表示有效位置，0 表示 padding
            time_features: 时间特征，形状为 (batch_size, seq_len, time_dim)
                          可选，如果不使用时间特征可传 None
        
        Returns:
            user_repr: 用户表示向量，形状为 (batch_size, d_model)
        """
        # 输入嵌入
        hidden = self.input_embedding(
            semantic_ids, 
            positions, 
            token_types, 
            time_features
        )
        
        # 编码器层
        for layer in self.layers:
            hidden, _ = layer(hidden, token_types, attention_mask)
        
        # 最终归一化
        hidden = self.final_norm(hidden)
        
        # 池化，提取用户表示
        user_repr = self.pooler(hidden, attention_mask)
        
        return user_repr
    
    def get_sequence_output(
        self,
        semantic_ids: List[torch.Tensor],
        positions: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        获取完整序列的编码输出（用于解码器的交叉注意力）
        
        Args:
            semantic_ids: [L1_ids, L2_ids, L3_ids] 三层语义 ID
            positions: 位置索引
            token_types: Token 类型索引
            attention_mask: 注意力掩码
            time_features: 时间特征（可选）
        
        Returns:
            sequence_output: 完整序列输出，形状为 (batch_size, seq_len, d_model)
        """
        # 输入嵌入
        hidden = self.input_embedding(
            semantic_ids, 
            positions, 
            token_types, 
            time_features
        )
        
        # 编码器层
        for layer in self.layers:
            hidden, _ = layer(hidden, token_types, attention_mask)
        
        # 最终归一化
        hidden = self.final_norm(hidden)
        
        return hidden
    
    def get_attention_weights(
        self,
        semantic_ids: List[torch.Tensor],
        positions: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        获取所有层的注意力权重（用于可视化和分析）
        
        Args:
            semantic_ids: 三层语义 ID
            positions: 位置索引
            token_types: Token 类型索引
            attention_mask: 注意力掩码
        
        Returns:
            attention_weights: 每层的注意力权重列表
                              每个张量形状为 (batch_size, n_heads, seq_len, seq_len)
        """
        # 输入嵌入
        hidden = self.input_embedding(semantic_ids, positions, token_types)
        
        attention_weights = []
        
        # 编码器层
        for layer in self.layers:
            hidden, attn_weights = layer(
                hidden, 
                token_types, 
                attention_mask,
                return_attention=True
            )
            if attn_weights is not None:
                attention_weights.append(attn_weights)
        
        return attention_weights
    
    def freeze_embeddings(self):
        """冻结嵌入层参数（用于微调）"""
        for param in self.input_embedding.parameters():
            param.requires_grad = False
    
    def unfreeze_embeddings(self):
        """解冻嵌入层参数"""
        for param in self.input_embedding.parameters():
            param.requires_grad = True
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        获取参数数量
        
        Args:
            trainable_only: 是否只统计可训练参数
        
        Returns:
            参数数量
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    @classmethod
    def from_config(cls, config: EncoderConfig) -> "UGTEncoder":
        """
        从配置创建编码器
        
        Args:
            config: 编码器配置对象
        
        Returns:
            UGTEncoder 实例
        """
        return cls(config)
    
    @classmethod
    def from_pretrained(cls, path: str) -> "UGTEncoder":
        """
        从预训练权重加载编码器
        
        Args:
            path: 预训练权重路径
        
        Returns:
            UGTEncoder 实例
        """
        checkpoint = torch.load(path, map_location="cpu")
        config = EncoderConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    
    def save_pretrained(self, path: str):
        """
        保存预训练权重
        
        Args:
            path: 保存路径
        """
        checkpoint = {
            "config": self.config.to_dict(),
            "model_state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)


class MultiScaleEncoder(nn.Module):
    """
    多尺度编码器
    
    分别编码长期、短期、实时行为，然后融合。
    
    结构:
    - Long-Term Encoder: 编码压缩的历史行为
    - Short-Term Encoder: 编码近期会话行为
    - Real-Time Encoder: 编码当前上下文
    
    对应架构文档: 3.1 节 Multi-Scale Encoder
    """
    
    def __init__(self, config: EncoderConfig):
        """
        初始化多尺度编码器
        
        Args:
            config: 编码器配置对象
        """
        super().__init__()
        self.config = config
        
        # 长期历史编码器（使用较少的层）
        long_term_config = EncoderConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers // 3,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len // 4,
            dropout=config.dropout,
        )
        self.long_term_encoder = UGTEncoder(long_term_config)
        
        # 短期会话编码器
        short_term_config = EncoderConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers // 2,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len // 2,
            dropout=config.dropout,
        )
        self.short_term_encoder = UGTEncoder(short_term_config)
        
        # 实时上下文编码器（使用较少的层）
        real_time_config = EncoderConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers // 4,
            d_ff=config.d_ff,
            max_seq_len=64,  # 实时上下文较短
            dropout=config.dropout,
        )
        self.real_time_encoder = UGTEncoder(real_time_config)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
        )
    
    def forward(
        self,
        long_term_inputs: Dict[str, torch.Tensor],
        short_term_inputs: Dict[str, torch.Tensor],
        real_time_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            long_term_inputs: 长期历史输入字典
            short_term_inputs: 短期会话输入字典
            real_time_inputs: 实时上下文输入字典
            
            每个字典包含:
            - semantic_ids: 三层语义 ID
            - positions: 位置索引
            - token_types: Token 类型索引
            - attention_mask: 注意力掩码
        
        Returns:
            用户表示向量，形状为 (batch_size, d_model)
        """
        # 分别编码
        long_term_repr = self.long_term_encoder(
            long_term_inputs["semantic_ids"],
            long_term_inputs["positions"],
            long_term_inputs["token_types"],
            long_term_inputs["attention_mask"],
        )
        
        short_term_repr = self.short_term_encoder(
            short_term_inputs["semantic_ids"],
            short_term_inputs["positions"],
            short_term_inputs["token_types"],
            short_term_inputs["attention_mask"],
        )
        
        real_time_repr = self.real_time_encoder(
            real_time_inputs["semantic_ids"],
            real_time_inputs["positions"],
            real_time_inputs["token_types"],
            real_time_inputs["attention_mask"],
        )
        
        # 融合
        concat = torch.cat([long_term_repr, short_term_repr, real_time_repr], dim=-1)
        fused = self.fusion(concat)
        
        return fused

