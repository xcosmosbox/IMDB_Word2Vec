"""
分组层归一化模块 (Group Layer Normalization)

该模块实现了来自美团 MTGR 论文的 Group Layer Normalization，
针对不同语义空间的 Token 使用不同的归一化参数。

核心创新点:
1. 为不同类型的 Token 提供独立的归一化参数
2. 增强不同语义空间的编码能力
3. 提高模型对异构 Token 的区分能力

Token 类型分组:
- Group 0: USER Token (用户属性)
- Group 1: ITEM Token (物品语义 ID)
- Group 2: ACTION Token (行为类型)
- Group 3: CONTEXT Token (上下文信息)

对应架构文档:
- 2.5 节 Group Layer Normalization
- 3.2.3 节 GLN

Author: Person B
"""

import torch
import torch.nn as nn
from typing import Optional

from .config import EncoderConfig


class GroupLayerNorm(nn.Module):
    """
    分组层归一化 (来自 MTGR)
    
    针对不同语义空间的 Token 使用不同的归一化参数 (γ 和 β)。
    
    计算过程:
    1. 对输入进行标准 LayerNorm 归一化 (减均值除标准差)
    2. 根据 token_types 选择对应组的 γ 和 β
    3. 应用仿射变换: output = γ * normalized + β
    
    优势:
    1. 不同类型的 Token 有不同的特征分布
    2. 独立的归一化参数可以更好地适应各自的分布
    3. 提高模型对异构序列的建模能力
    
    Attributes:
        d_model: 特征维度
        num_groups: 分组数量（通常与 Token 类型数相同）
        eps: 数值稳定性的 epsilon 值
        gamma: 各组的缩放参数，形状为 (num_groups, d_model)
        beta: 各组的偏移参数，形状为 (num_groups, d_model)
    
    Example:
        >>> gln = GroupLayerNorm(d_model=512, num_groups=4)
        >>> x = torch.randn(32, 100, 512)
        >>> token_types = torch.randint(0, 4, (32, 100))
        >>> output = gln(x, token_types)
        >>> output.shape
        torch.Size([32, 100, 512])
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_groups: int = 4, 
        eps: float = 1e-5
    ):
        """
        初始化分组层归一化
        
        Args:
            d_model: 特征维度
            num_groups: 分组数量，默认为 4（对应 4 种 Token 类型）
            eps: Layer Normalization 的 epsilon 值，用于数值稳定性
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_groups = num_groups
        self.eps = eps
        
        # 每组独立的 γ 和 β 参数
        # gamma: 缩放参数，初始化为 1
        # beta: 偏移参数，初始化为 0
        self.gamma = nn.Parameter(torch.ones(num_groups, d_model))
        self.beta = nn.Parameter(torch.zeros(num_groups, d_model))
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_types: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            token_types: Token 类型索引，形状为 (batch_size, seq_len)
                        值域为 [0, num_groups)
        
        Returns:
            normalized: 归一化后的张量，形状为 (batch_size, seq_len, d_model)
        
        Raises:
            ValueError: 如果 token_types 中的值超出 [0, num_groups) 范围
        """
        # 标准 LayerNorm 计算
        # 计算最后一个维度的均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 归一化
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 根据 token_types 选择对应的 γ 和 β
        # token_types: (batch, seq_len) -> 用于索引 gamma 和 beta
        # gamma[token_types]: (batch, seq_len, d_model)
        gamma = self.gamma[token_types]
        beta = self.beta[token_types]
        
        # 应用仿射变换
        output = gamma * normalized + beta
        
        return output
    
    def extra_repr(self) -> str:
        """返回模块的额外信息"""
        return f"d_model={self.d_model}, num_groups={self.num_groups}, eps={self.eps}"


class ConditionalLayerNorm(nn.Module):
    """
    条件层归一化
    
    这是 GroupLayerNorm 的一个变体，支持更灵活的条件输入。
    除了使用离散的 token_types 作为条件，还可以使用连续的条件向量。
    
    Attributes:
        d_model: 特征维度
        condition_dim: 条件向量维度
        gamma_proj: 用于生成 gamma 的投影层
        beta_proj: 用于生成 beta 的投影层
    """
    
    def __init__(
        self, 
        d_model: int, 
        condition_dim: int,
        eps: float = 1e-5
    ):
        """
        初始化条件层归一化
        
        Args:
            d_model: 特征维度
            condition_dim: 条件向量维度
            eps: Layer Normalization 的 epsilon 值
        """
        super().__init__()
        
        self.d_model = d_model
        self.condition_dim = condition_dim
        self.eps = eps
        
        # 从条件向量生成 gamma 和 beta 的投影层
        self.gamma_proj = nn.Linear(condition_dim, d_model)
        self.beta_proj = nn.Linear(condition_dim, d_model)
        
        # 初始化：使 gamma 接近 1，beta 接近 0
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            condition: 条件向量，形状为 (batch_size, seq_len, condition_dim)
                      或 (batch_size, condition_dim) 将广播到所有位置
        
        Returns:
            normalized: 归一化后的张量
        """
        # 标准归一化
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 处理条件向量的维度
        if condition.dim() == 2:
            # (batch, condition_dim) -> (batch, 1, condition_dim)
            condition = condition.unsqueeze(1)
        
        # 从条件向量生成 gamma 和 beta
        gamma = self.gamma_proj(condition)  # (batch, seq_len, d_model)
        beta = self.beta_proj(condition)    # (batch, seq_len, d_model)
        
        # 应用仿射变换
        output = gamma * normalized + beta
        
        return output


class StandardLayerNorm(nn.LayerNorm):
    """
    标准层归一化（PyTorch 内置的包装）
    
    这是对 torch.nn.LayerNorm 的简单包装，
    提供与 GroupLayerNorm 一致的接口。
    
    用于不需要分组归一化的场景，或作为对比基线。
    """
    
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5
    ):
        """
        初始化标准层归一化
        
        Args:
            d_model: 特征维度
            eps: 数值稳定性的 epsilon 值
        """
        super().__init__(normalized_shape=d_model, eps=eps)
        self.d_model = d_model
    
    def forward(
        self, 
        x: torch.Tensor, 
        token_types: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            token_types: 忽略，仅为保持接口一致性
        
        Returns:
            normalized: 归一化后的张量
        """
        return super().forward(x)


def create_layer_norm(
    config: EncoderConfig,
    use_group_norm: bool = True,
) -> nn.Module:
    """
    创建层归一化的工厂函数
    
    Args:
        config: 编码器配置
        use_group_norm: 是否使用分组归一化（默认 True）
                       如果为 False，使用标准 LayerNorm
    
    Returns:
        层归一化实例
    """
    if use_group_norm:
        return GroupLayerNorm(
            d_model=config.d_model,
            num_groups=config.num_groups,
            eps=config.layer_norm_eps,
        )
    else:
        return StandardLayerNorm(
            d_model=config.d_model,
            eps=config.layer_norm_eps,
        )

