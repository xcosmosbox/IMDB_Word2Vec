"""
UGT 编码器配置模块

该模块定义了 UGT Encoder 的所有配置参数。

对应架构文档: 
- 2.2 节 模型超参数配置
- 3.2.1 节 语义 ID 系统

Author: Person B
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class EncoderConfig:
    """
    UGT 编码器配置类
    
    该配置类包含了编码器的所有超参数设置，包括：
    - 模型架构参数（维度、层数、头数等）
    - 语义 ID 配置（码本大小）
    - Token 类型配置
    - 时间特征配置
    - 训练相关配置（Dropout 等）
    
    Attributes:
        d_model: 隐藏层维度，决定了模型的表示能力
        n_heads: 注意力头数，用于多头注意力机制
        n_layers: 编码器层数，决定了模型的深度
        d_ff: 前馈网络中间维度，通常为 d_model 的 4 倍
        max_seq_len: 最大序列长度，限制输入序列的最大长度
        dropout: Dropout 率，用于正则化防止过拟合
        codebook_sizes: 三层语义 ID 的码本大小，对应粗/中/细三个粒度
        num_token_types: Token 类型数量 (USER=0, ITEM=1, ACTION=2, CONTEXT=3)
        num_groups: Group Layer Normalization 的分组数，与 token_types 对应
        time_dim: 时间嵌入维度，0 表示不使用时间特征
        layer_norm_eps: Layer Normalization 的 epsilon 值
        hidden_act: 隐藏层激活函数类型
    
    Example:
        >>> config = EncoderConfig(d_model=512, n_heads=16, n_layers=12)
        >>> print(config.d_model)
        512
        >>> print(config.codebook_sizes)
        (1024, 4096, 16384)
    """
    
    # ========================
    # 模型架构参数
    # ========================
    d_model: int = 512
    """隐藏层维度，决定了 Token 嵌入和隐藏状态的维度大小"""
    
    n_heads: int = 16
    """注意力头数，d_model 必须能被 n_heads 整除"""
    
    n_layers: int = 12
    """编码器 Transformer 层数"""
    
    d_ff: int = 2048
    """前馈网络 (FFN) 的中间隐藏维度"""
    
    max_seq_len: int = 1024
    """支持的最大输入序列长度"""
    
    dropout: float = 0.1
    """Dropout 概率，用于正则化"""
    
    # ========================
    # 语义 ID 配置
    # ========================
    codebook_sizes: Tuple[int, int, int] = (1024, 4096, 16384)
    """
    三层语义 ID 的码本大小
    - Level 1 (1024): 粗粒度类目，如电影类型、商品大类
    - Level 2 (4096): 细粒度属性，如子类目、品牌
    - Level 3 (16384): 实例区分，用于区分具体物品
    """
    
    semantic_embedding_dim: int = 0  # 0 表示自动计算为 d_model // 3
    """每层语义 ID 嵌入的维度，默认为 d_model // 3"""
    
    # ========================
    # Token 类型配置
    # ========================
    num_token_types: int = 4
    """
    Token 类型数量：
    - 0: USER Token (用户属性)
    - 1: ITEM Token (物品语义 ID)
    - 2: ACTION Token (行为类型：点击、购买等)
    - 3: CONTEXT Token (上下文信息：时间、设备等)
    """
    
    num_groups: int = 4
    """
    Group Layer Normalization 的分组数
    通常与 num_token_types 相同，为不同类型的 Token 提供独立的归一化参数
    """
    
    # ========================
    # 时间特征配置
    # ========================
    time_dim: int = 32
    """
    时间特征的原始维度
    时间特征将被投影到 d_model 维度后与其他嵌入相加
    设置为 0 表示不使用时间特征
    """
    
    # ========================
    # 层归一化配置
    # ========================
    layer_norm_eps: float = 1e-5
    """Layer Normalization 的 epsilon 值，用于数值稳定性"""
    
    # ========================
    # 激活函数配置
    # ========================
    hidden_act: str = "gelu"
    """
    隐藏层激活函数类型
    支持: "gelu", "relu", "swish", "silu"
    """
    
    # ========================
    # 注意力配置
    # ========================
    use_causal_mask: bool = False
    """
    是否使用因果掩码（单向注意力）
    编码器通常使用双向注意力，设为 False
    """
    
    attention_dropout: float = 0.1
    """注意力权重的 Dropout 概率"""
    
    # ========================
    # 池化配置
    # ========================
    pooler_type: str = "cls"
    """
    用户表示的池化方式
    - "cls": 使用第一个 Token (类似 BERT 的 [CLS])
    - "mean": 使用所有有效 Token 的平均值
    - "last": 使用最后一个有效 Token
    """
    
    def __post_init__(self):
        """初始化后的验证和自动计算"""
        # 验证 d_model 能被 n_heads 整除
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) 必须能被 n_heads ({self.n_heads}) 整除"
            )
        
        # 验证码本大小
        if len(self.codebook_sizes) != 3:
            raise ValueError(
                f"codebook_sizes 必须包含 3 个元素，当前为 {len(self.codebook_sizes)}"
            )
        
        # 自动计算语义嵌入维度
        if self.semantic_embedding_dim == 0:
            # 确保三层嵌入拼接后正好等于 d_model
            # 如果 d_model 不能被 3 整除，最后一层会多一些维度
            base_dim = self.d_model // 3
            remainder = self.d_model % 3
            self._semantic_dims = (base_dim, base_dim, base_dim + remainder)
        else:
            self._semantic_dims = (
                self.semantic_embedding_dim,
                self.semantic_embedding_dim,
                self.semantic_embedding_dim
            )
        
        # 验证 num_groups 与 num_token_types
        if self.num_groups != self.num_token_types:
            import warnings
            warnings.warn(
                f"num_groups ({self.num_groups}) 与 num_token_types ({self.num_token_types}) 不相等，"
                "这可能导致 Group Layer Normalization 无法正确工作"
            )
    
    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.d_model // self.n_heads
    
    @property
    def semantic_dims(self) -> Tuple[int, int, int]:
        """三层语义 ID 嵌入的维度"""
        return self._semantic_dims
    
    def to_dict(self) -> dict:
        """将配置转换为字典"""
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "codebook_sizes": self.codebook_sizes,
            "num_token_types": self.num_token_types,
            "num_groups": self.num_groups,
            "time_dim": self.time_dim,
            "layer_norm_eps": self.layer_norm_eps,
            "hidden_act": self.hidden_act,
            "use_causal_mask": self.use_causal_mask,
            "attention_dropout": self.attention_dropout,
            "pooler_type": self.pooler_type,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "EncoderConfig":
        """从字典创建配置"""
        return cls(**config_dict)
    
    @classmethod
    def small(cls) -> "EncoderConfig":
        """小规模模型配置，适用于调试和快速实验"""
        return cls(
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=1024,
            max_seq_len=512,
        )
    
    @classmethod
    def base(cls) -> "EncoderConfig":
        """基础规模模型配置，适用于中等规模数据"""
        return cls(
            d_model=512,
            n_heads=16,
            n_layers=12,
            d_ff=2048,
            max_seq_len=1024,
        )
    
    @classmethod
    def large(cls) -> "EncoderConfig":
        """大规模模型配置，适用于生产环境"""
        return cls(
            d_model=1024,
            n_heads=32,
            n_layers=24,
            d_ff=4096,
            max_seq_len=2048,
        )

