"""
前馈网络模块 (Feed-Forward Network)

该模块实现了 Transformer 中的前馈网络层，
包括标准 FFN 和 GLU (Gated Linear Unit) 变体。

标准 FFN 结构:
    FFN(x) = Dropout(Activation(xW_1 + b_1))W_2 + b_2

GLU 变体结构:
    FFN(x) = (xW_1 * σ(xW_g)) W_2

对应架构文档:
- 2.3 节 Encoder 层结构

Author: Person B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import EncoderConfig


def get_activation(activation: str) -> nn.Module:
    """
    获取激活函数
    
    Args:
        activation: 激活函数名称
                   支持: "gelu", "relu", "swish", "silu", "tanh", "sigmoid"
    
    Returns:
        激活函数模块
    """
    activation = activation.lower()
    
    if activation == "gelu":
        return nn.GELU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation in ("swish", "silu"):
        return nn.SiLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"不支持的激活函数: {activation}")


class FeedForwardNetwork(nn.Module):
    """
    标准前馈网络
    
    结构: Linear -> Activation -> Dropout -> Linear -> Dropout
    
    公式:
        FFN(x) = Dropout(Linear_2(Dropout(Activation(Linear_1(x)))))
    
    Attributes:
        linear1: 第一个线性变换，升维到 d_ff
        linear2: 第二个线性变换，降维回 d_model
        activation: 激活函数
        dropout1: 激活后的 Dropout
        dropout2: 输出的 Dropout
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        初始化前馈网络
        
        Args:
            d_model: 输入/输出维度
            d_ff: 中间隐藏维度（通常为 d_model 的 4 倍）
            dropout: Dropout 概率
            activation: 激活函数名称
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 线性变换层
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # 激活函数
        self.activation = get_activation(activation)
        
        # Dropout 层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # 第一层：升维 + 激活 + Dropout
        hidden = self.linear1(x)
        hidden = self.activation(hidden)
        hidden = self.dropout1(hidden)
        
        # 第二层：降维 + Dropout
        output = self.linear2(hidden)
        output = self.dropout2(output)
        
        return output


class GatedLinearUnit(nn.Module):
    """
    门控线性单元 (GLU) 前馈网络
    
    GLU 通过门控机制控制信息流，通常比标准 FFN 效果更好。
    
    公式:
        GLU(x) = (xW_1) * σ(xW_g)
        FFN(x) = GLU(x) W_2
    
    其中 σ 是 Sigmoid 函数（或其他门控激活函数）。
    
    Attributes:
        linear1: 内容投影
        gate: 门控投影
        linear2: 输出投影
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        """
        初始化 GLU 前馈网络
        
        Args:
            d_model: 输入/输出维度
            d_ff: 中间隐藏维度
            dropout: Dropout 概率
            activation: 门控激活函数名称
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 内容投影
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # 门控投影
        self.gate = nn.Linear(d_model, d_ff)
        
        # 输出投影
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # 激活函数（用于门控）
        self.activation = get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.linear1, self.gate, self.linear2]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # 内容和门控
        content = self.linear1(x)
        gate = self.activation(self.gate(x))
        
        # 门控线性单元
        hidden = content * gate
        
        # 输出投影
        output = self.linear2(hidden)
        output = self.dropout(output)
        
        return output


class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络
    
    SwiGLU 是 GLU 的一个变体，使用 SiLU (Swish) 作为激活函数。
    在 LLaMA、PaLM 等模型中广泛使用。
    
    公式:
        SwiGLU(x) = (xW_1) * SiLU(xW_g)
        FFN(x) = SwiGLU(x) W_2
    
    注意: SwiGLU 的实际 d_ff 是标准 FFN 的 2/3，
    以保持相同的参数量。
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        """
        初始化 SwiGLU 前馈网络
        
        Args:
            d_model: 输入/输出维度
            d_ff: 中间隐藏维度
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 为了保持参数量，调整隐藏维度
        # 标准 FFN: d_model -> d_ff -> d_model (2 * d_model * d_ff 参数)
        # SwiGLU: d_model -> d_ff * 2 个投影 -> d_model (3 * d_model * d_ff 参数)
        # 所以 SwiGLU 的 d_ff 可以设为 2/3 来保持参数量
        hidden_dim = int(d_ff * 2 / 3)
        # 确保 hidden_dim 是 64 的倍数（优化内存对齐）
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.w1, self.w2, self.w3]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # SwiGLU: (x @ W1) * SiLU(x @ W3) @ W2
        output = self.w2(F.silu(self.w1(x)) * self.w3(x))
        output = self.dropout(output)
        return output


class EncoderFFN(nn.Module):
    """
    编码器前馈网络
    
    这是用于编码器层的前馈网络，封装了标准 FFN。
    提供与 encoder_layer.py 中的接口一致。
    
    结构: Linear -> GELU -> Dropout -> Linear -> Dropout
    """
    
    def __init__(self, config: EncoderConfig):
        """
        初始化编码器前馈网络
        
        Args:
            config: 编码器配置对象
        """
        super().__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        return self.ffn(x)


def create_ffn(
    config: EncoderConfig,
    ffn_type: str = "standard",
) -> nn.Module:
    """
    创建前馈网络的工厂函数
    
    Args:
        config: 编码器配置
        ffn_type: FFN 类型
                 - "standard": 标准 FFN
                 - "glu": 门控线性单元
                 - "swiglu": SwiGLU
    
    Returns:
        前馈网络实例
    """
    if ffn_type == "standard":
        return FeedForwardNetwork(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.hidden_act,
        )
    elif ffn_type == "glu":
        return GatedLinearUnit(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.hidden_act,
        )
    elif ffn_type == "swiglu":
        return SwiGLU(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"不支持的 FFN 类型: {ffn_type}")

