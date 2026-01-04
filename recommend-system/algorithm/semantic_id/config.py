"""
Semantic ID 编码器配置类

该模块定义了语义 ID 编码器的所有配置参数。

配置说明：
- embedding_dim: 物品特征向量维度，默认 256
- num_codebooks: 码本层数，默认 3（对应 L1, L2, L3 三层语义 ID）
- codebook_sizes: 各层码本大小，默认 (1024, 4096, 16384)
  - L1 (粗粒度): 1024 个码本，表示大类
  - L2 (中粒度): 4096 个码本，表示子类
  - L3 (细粒度): 16384 个码本，表示具体物品
- commitment_cost: 承诺损失权重 β，用于平衡编码器和码本的优化
- ema_decay: EMA 衰减率，用于码本的平滑更新
- epsilon: 数值稳定性常数

作者: Person A
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import yaml


@dataclass
class SemanticIDConfig:
    """
    语义 ID 编码器配置类
    
    该配置类定义了 RQ-VAE 语义 ID 编码器的所有超参数。
    
    Attributes:
        embedding_dim: 输入特征向量维度，默认 256
        num_codebooks: 码本数量（层数），默认 3
        codebook_sizes: 各层码本大小的元组，默认 (1024, 4096, 16384)
        commitment_cost: 承诺损失权重 β，默认 0.25
        ema_decay: EMA 衰减率，用于码本更新，默认 0.99
        epsilon: 数值稳定性常数，默认 1e-5
        use_ema: 是否使用 EMA 更新码本，默认 True
        codebook_init_scale: 码本初始化缩放因子，默认 1.0
        dead_code_threshold: 死码本阈值，低于此使用频率的码本会被重置
    
    Example:
        >>> config = SemanticIDConfig()
        >>> config.embedding_dim
        256
        >>> config.codebook_sizes
        (1024, 4096, 16384)
        
        >>> # 自定义配置
        >>> custom_config = SemanticIDConfig(
        ...     embedding_dim=512,
        ...     codebook_sizes=(2048, 8192, 32768),
        ... )
    """
    
    # 基本配置
    embedding_dim: int = 256
    num_codebooks: int = 3
    codebook_sizes: Tuple[int, ...] = (1024, 4096, 16384)
    
    # 损失函数配置
    commitment_cost: float = 0.25
    
    # EMA 更新配置
    ema_decay: float = 0.99
    use_ema: bool = True
    
    # 数值稳定性
    epsilon: float = 1e-5
    
    # 码本初始化配置
    codebook_init_scale: float = 1.0
    
    # 码本健康度配置
    dead_code_threshold: float = 0.01  # 使用频率低于 1% 视为死码本
    dead_code_reset_prob: float = 0.01  # 重置死码本的概率
    
    # 训练配置
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    
    def __post_init__(self):
        """配置验证"""
        # 验证码本数量与码本大小一致
        if len(self.codebook_sizes) != self.num_codebooks:
            raise ValueError(
                f"码本大小元组长度 ({len(self.codebook_sizes)}) "
                f"必须等于码本数量 ({self.num_codebooks})"
            )
        
        # 验证嵌入维度为正整数
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim 必须为正整数，当前值: {self.embedding_dim}")
        
        # 验证所有码本大小为正整数
        for i, size in enumerate(self.codebook_sizes):
            if size <= 0:
                raise ValueError(f"第 {i+1} 层码本大小必须为正整数，当前值: {size}")
        
        # 验证 commitment_cost 为非负数
        if self.commitment_cost < 0:
            raise ValueError(f"commitment_cost 必须为非负数，当前值: {self.commitment_cost}")
        
        # 验证 ema_decay 在有效范围内
        if not (0 < self.ema_decay < 1):
            raise ValueError(f"ema_decay 必须在 (0, 1) 范围内，当前值: {self.ema_decay}")
    
    @property
    def total_vocab_size(self) -> int:
        """
        计算总词表大小
        
        Returns:
            所有码本大小的总和
        """
        return sum(self.codebook_sizes)
    
    @property
    def l1_codebook_size(self) -> int:
        """获取 L1 层（粗粒度）码本大小"""
        return self.codebook_sizes[0]
    
    @property
    def l2_codebook_size(self) -> int:
        """获取 L2 层（中粒度）码本大小"""
        return self.codebook_sizes[1]
    
    @property
    def l3_codebook_size(self) -> int:
        """获取 L3 层（细粒度）码本大小"""
        return self.codebook_sizes[2]
    
    def get_codebook_size(self, level: int) -> int:
        """
        获取指定层级的码本大小
        
        Args:
            level: 层级 (1, 2, 或 3)
        
        Returns:
            对应层级的码本大小
        
        Raises:
            ValueError: 如果层级无效
        """
        if level < 1 or level > self.num_codebooks:
            raise ValueError(
                f"层级必须在 1 到 {self.num_codebooks} 之间，当前值: {level}"
            )
        return self.codebook_sizes[level - 1]
    
    def to_dict(self) -> dict:
        """
        将配置转换为字典
        
        Returns:
            配置字典
        """
        return {
            "embedding_dim": self.embedding_dim,
            "num_codebooks": self.num_codebooks,
            "codebook_sizes": list(self.codebook_sizes),
            "commitment_cost": self.commitment_cost,
            "ema_decay": self.ema_decay,
            "use_ema": self.use_ema,
            "epsilon": self.epsilon,
            "codebook_init_scale": self.codebook_init_scale,
            "dead_code_threshold": self.dead_code_threshold,
            "dead_code_reset_prob": self.dead_code_reset_prob,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "SemanticIDConfig":
        """
        从字典创建配置对象
        
        Args:
            config_dict: 配置字典
        
        Returns:
            SemanticIDConfig 实例
        """
        # 处理 codebook_sizes 类型转换
        if "codebook_sizes" in config_dict:
            config_dict["codebook_sizes"] = tuple(config_dict["codebook_sizes"])
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """
        保存配置到 YAML 文件
        
        Args:
            path: 文件路径
        """
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> "SemanticIDConfig":
        """
        从 YAML 文件加载配置
        
        Args:
            path: 文件路径
        
        Returns:
            SemanticIDConfig 实例
        """
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def __repr__(self) -> str:
        return (
            f"SemanticIDConfig(\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  num_codebooks={self.num_codebooks},\n"
            f"  codebook_sizes={self.codebook_sizes},\n"
            f"  commitment_cost={self.commitment_cost},\n"
            f"  ema_decay={self.ema_decay},\n"
            f"  use_ema={self.use_ema}\n"
            f")"
        )


# 预定义配置
class PresetConfigs:
    """预定义的配置集合"""
    
    @staticmethod
    def small() -> SemanticIDConfig:
        """
        小规模配置，适合调试和小数据集
        
        - 嵌入维度: 128
        - 码本大小: (256, 1024, 4096)
        """
        return SemanticIDConfig(
            embedding_dim=128,
            codebook_sizes=(256, 1024, 4096),
        )
    
    @staticmethod
    def medium() -> SemanticIDConfig:
        """
        中等规模配置，默认配置
        
        - 嵌入维度: 256
        - 码本大小: (1024, 4096, 16384)
        """
        return SemanticIDConfig()
    
    @staticmethod
    def large() -> SemanticIDConfig:
        """
        大规模配置，适合大数据集
        
        - 嵌入维度: 512
        - 码本大小: (2048, 8192, 32768)
        """
        return SemanticIDConfig(
            embedding_dim=512,
            codebook_sizes=(2048, 8192, 32768),
        )
    
    @staticmethod
    def production() -> SemanticIDConfig:
        """
        生产环境配置
        
        - 嵌入维度: 256
        - 码本大小: (1024, 4096, 16384)
        - 使用保守的超参数
        """
        return SemanticIDConfig(
            embedding_dim=256,
            codebook_sizes=(1024, 4096, 16384),
            commitment_cost=0.25,
            ema_decay=0.99,
            dead_code_threshold=0.005,  # 更保守的死码本阈值
        )

