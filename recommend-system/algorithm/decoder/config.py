"""
UGT Decoder 配置模块

定义解码器的所有超参数配置。

对应架构文档: 《生成式推荐系统架构设计》第三章
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class DecoderConfig:
    """
    解码器配置类
    
    包含模型结构、MoE、语义 ID 等所有配置参数。
    
    Attributes:
        d_model: 隐藏层维度
        n_heads: 注意力头数
        n_layers: 解码器层数
        d_ff: FFN 中间维度
        max_seq_len: 最大序列长度
        dropout: Dropout 率
        
        num_experts: MoE 专家数量
        top_k_experts: 每次激活的专家数
        expert_capacity_factor: 专家容量因子
        moe_loss_weight: MoE 负载均衡损失权重
        
        codebook_sizes: 各层语义 ID 码本大小
        num_token_types: Token 类型数量
        num_groups: GLN 分组数量
        
        use_flash_attention: 是否使用 Flash Attention
        use_gradient_checkpointing: 是否使用梯度检查点
    """
    
    # 模型基础配置
    d_model: int = 512              # 隐藏层维度
    n_heads: int = 16               # 注意力头数
    n_layers: int = 12              # 解码器层数
    d_ff: int = 2048                # FFN 中间维度
    max_seq_len: int = 1024         # 最大序列长度
    dropout: float = 0.1            # Dropout 率
    
    # MoE 配置
    num_experts: int = 16           # 专家数量
    top_k_experts: int = 4          # 每次激活的专家数
    expert_capacity_factor: float = 1.25  # 专家容量因子
    moe_loss_weight: float = 0.01   # MoE 负载均衡损失权重
    moe_jitter_noise: float = 0.0   # 路由噪声（训练时使用）
    
    # 语义 ID 配置
    codebook_sizes: Tuple[int, int, int] = (1024, 4096, 16384)
    
    # Token 类型配置
    num_token_types: int = 4        # USER=0, ITEM=1, ACTION=2, CONTEXT=3
    num_groups: int = 4             # GLN 分组数量
    
    # 特殊 Token ID
    pad_token_id: int = 0
    bos_token_id: int = 1           # 开始 Token
    eos_token_id: int = 2           # 结束 Token
    
    # 生成配置
    max_generate_length: int = 100  # 最大生成长度
    
    # 优化配置
    use_flash_attention: bool = False       # 是否使用 Flash Attention
    use_gradient_checkpointing: bool = False  # 是否使用梯度检查点
    layer_norm_eps: float = 1e-6            # LayerNorm epsilon
    
    def __post_init__(self):
        """验证配置有效性"""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) 必须能被 n_heads ({self.n_heads}) 整除"
        assert len(self.codebook_sizes) == 3, \
            f"codebook_sizes 必须包含 3 个元素，当前为 {len(self.codebook_sizes)}"
        assert self.top_k_experts <= self.num_experts, \
            f"top_k_experts ({self.top_k_experts}) 不能超过 num_experts ({self.num_experts})"
        assert self.dropout >= 0 and self.dropout < 1, \
            f"dropout 必须在 [0, 1) 范围内，当前为 {self.dropout}"
    
    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.d_model // self.n_heads
    
    @property
    def total_codebook_size(self) -> int:
        """所有码本的总大小"""
        return sum(self.codebook_sizes)
    
    def get_codebook_size(self, level: int) -> int:
        """
        获取指定层级的码本大小
        
        Args:
            level: 层级 (1, 2, 或 3)
            
        Returns:
            对应层级的码本大小
        """
        if level < 1 or level > 3:
            raise ValueError(f"level 必须在 [1, 3] 范围内，当前为 {level}")
        return self.codebook_sizes[level - 1]
    
    @classmethod
    def small(cls) -> "DecoderConfig":
        """小规模配置（用于调试和测试）"""
        return cls(
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=1024,
            num_experts=8,
            top_k_experts=2,
            max_seq_len=512,
        )
    
    @classmethod
    def medium(cls) -> "DecoderConfig":
        """中规模配置（用于实验）"""
        return cls(
            d_model=512,
            n_heads=16,
            n_layers=12,
            d_ff=2048,
            num_experts=16,
            top_k_experts=4,
            max_seq_len=1024,
        )
    
    @classmethod
    def large(cls) -> "DecoderConfig":
        """大规模配置（生产环境）"""
        return cls(
            d_model=1024,
            n_heads=32,
            n_layers=24,
            d_ff=4096,
            num_experts=32,
            top_k_experts=4,
            max_seq_len=2048,
            use_flash_attention=True,
            use_gradient_checkpointing=True,
        )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "num_experts": self.num_experts,
            "top_k_experts": self.top_k_experts,
            "expert_capacity_factor": self.expert_capacity_factor,
            "moe_loss_weight": self.moe_loss_weight,
            "codebook_sizes": self.codebook_sizes,
            "num_token_types": self.num_token_types,
            "num_groups": self.num_groups,
            "use_flash_attention": self.use_flash_attention,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "DecoderConfig":
        """从字典创建配置"""
        # 处理 codebook_sizes 可能是 list 的情况
        if "codebook_sizes" in d and isinstance(d["codebook_sizes"], list):
            d["codebook_sizes"] = tuple(d["codebook_sizes"])
        return cls(**d)

