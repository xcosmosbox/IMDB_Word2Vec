"""
特征工程配置模块

定义特征工程相关的所有配置参数，包括：
- 序列配置：最大长度、物品数量限制等
- 词表配置：词表大小、最小频率等
- 特殊 Token：PAD、CLS、SEP、MASK、UNK
- 时间分桶：时间段划分
- 行为类型：支持的用户行为
- 设备类型：支持的设备
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
import os


@dataclass
class FeatureConfig:
    """
    特征工程配置类
    
    包含特征工程模块的所有配置参数，支持从文件加载和保存。
    
    Attributes:
        max_seq_length: 最大序列长度
        max_items_per_user: 每用户最大物品数
        vocab_size: 词表大小
        min_token_freq: 最小 Token 频率
        
    Example:
        >>> config = FeatureConfig()
        >>> config.max_seq_length
        1024
        >>> config.save("config.json")
        >>> loaded_config = FeatureConfig.load("config.json")
    """
    
    # ==================== 序列配置 ====================
    max_seq_length: int = 1024
    """最大序列长度（Token 数量）"""
    
    max_items_per_user: int = 500
    """每个用户保留的最大物品交互数"""
    
    min_sequence_length: int = 5
    """最小序列长度，过短的序列将被过滤"""
    
    # ==================== 词表配置 ====================
    vocab_size: int = 500000
    """词表最大容量"""
    
    min_token_freq: int = 5
    """Token 最小出现频率，低于此频率的 Token 将被替换为 UNK"""
    
    # ==================== 特殊 Token ====================
    pad_token: str = "[PAD]"
    """填充 Token"""
    
    cls_token: str = "[CLS]"
    """序列开始 Token"""
    
    sep_token: str = "[SEP]"
    """序列分隔 Token"""
    
    mask_token: str = "[MASK]"
    """掩码 Token，用于 MLM 训练"""
    
    unk_token: str = "[UNK]"
    """未知 Token"""
    
    # ==================== 特殊 Token ID ====================
    pad_token_id: int = 0
    """填充 Token ID"""
    
    cls_token_id: int = 1
    """序列开始 Token ID"""
    
    sep_token_id: int = 2
    """序列分隔 Token ID"""
    
    mask_token_id: int = 3
    """掩码 Token ID"""
    
    unk_token_id: int = 4
    """未知 Token ID"""
    
    special_token_count: int = 5
    """特殊 Token 数量"""
    
    # ==================== 语义 ID 配置 ====================
    semantic_id_levels: int = 3
    """语义 ID 层级数"""
    
    codebook_sizes: Tuple[int, int, int] = (1024, 4096, 16384)
    """各层码本大小：(L1, L2, L3)"""
    
    embedding_dim: int = 256
    """物品特征嵌入维度"""
    
    # ==================== 时间分桶 ====================
    time_buckets: List[str] = field(default_factory=lambda: [
        "night",      # 0:00 - 6:00
        "morning",    # 6:00 - 12:00
        "afternoon",  # 12:00 - 18:00
        "evening",    # 18:00 - 24:00
    ])
    """时间段分桶名称"""
    
    time_bucket_hours: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 6),       # night: 0:00 - 6:00
        (6, 12),      # morning: 6:00 - 12:00
        (12, 18),     # afternoon: 12:00 - 18:00
        (18, 24),     # evening: 18:00 - 24:00
    ])
    """时间段对应的小时范围"""
    
    # ==================== 行为类型 ====================
    action_types: List[str] = field(default_factory=lambda: [
        "view",       # 浏览
        "click",      # 点击
        "like",       # 点赞
        "dislike",    # 不喜欢
        "favorite",   # 收藏
        "share",      # 分享
        "comment",    # 评论
        "buy",        # 购买
        "rate",       # 评分
        "cart",       # 加购物车
    ])
    """支持的用户行为类型"""
    
    action_weights: Dict[str, float] = field(default_factory=lambda: {
        "view": 0.1,
        "click": 1.0,
        "like": 1.5,
        "dislike": -1.0,
        "favorite": 2.0,
        "share": 2.0,
        "comment": 1.5,
        "buy": 5.0,
        "rate": 1.0,
        "cart": 2.0,
    })
    """各行为类型的权重，用于计算用户偏好"""
    
    # ==================== 设备类型 ====================
    device_types: List[str] = field(default_factory=lambda: [
        "mobile",     # 手机
        "desktop",    # 桌面电脑
        "tablet",     # 平板
        "tv",         # 电视
        "other",      # 其他
    ])
    """支持的设备类型"""
    
    # ==================== 上下文类型 ====================
    context_types: List[str] = field(default_factory=lambda: [
        "home",       # 首页
        "search",     # 搜索
        "detail",     # 详情页
        "cart",       # 购物车
        "recommend",  # 推荐列表
        "other",      # 其他
    ])
    """支持的上下文场景类型"""
    
    # ==================== Token 类型 ID ====================
    token_type_ids: Dict[str, int] = field(default_factory=lambda: {
        "USER": 0,      # 用户 Token
        "ITEM": 1,      # 物品 Token
        "ACTION": 2,    # 行为 Token
        "CONTEXT": 3,   # 上下文 Token
    })
    """Token 类型到 ID 的映射"""
    
    # ==================== 负采样配置 ====================
    num_negative_samples: int = 4
    """负采样数量"""
    
    negative_sampling_strategy: str = "uniform"
    """负采样策略：uniform（均匀采样）或 popularity（按热度采样）"""
    
    # ==================== 数据处理配置 ====================
    shuffle_buffer_size: int = 10000
    """数据打乱缓冲区大小"""
    
    prefetch_buffer_size: int = 2
    """预取缓冲区大小"""
    
    num_parallel_calls: int = 8
    """并行处理线程数"""
    
    def get_special_tokens(self) -> List[str]:
        """
        获取所有特殊 Token 列表
        
        Returns:
            特殊 Token 列表，按 ID 顺序排列
        """
        return [
            self.pad_token,
            self.cls_token,
            self.sep_token,
            self.mask_token,
            self.unk_token,
        ]
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """
        获取特殊 Token 到 ID 的映射
        
        Returns:
            特殊 Token 到 ID 的字典
        """
        return {
            self.pad_token: self.pad_token_id,
            self.cls_token: self.cls_token_id,
            self.sep_token: self.sep_token_id,
            self.mask_token: self.mask_token_id,
            self.unk_token: self.unk_token_id,
        }
    
    def get_time_bucket(self, hour: int) -> str:
        """
        根据小时获取时间分桶名称
        
        Args:
            hour: 小时（0-23）
            
        Returns:
            时间分桶名称
            
        Example:
            >>> config = FeatureConfig()
            >>> config.get_time_bucket(10)
            'morning'
        """
        for bucket_name, (start, end) in zip(self.time_buckets, self.time_bucket_hours):
            if start <= hour < end:
                return bucket_name
        return self.time_buckets[-1]  # 默认返回最后一个（evening）
    
    def get_action_weight(self, action: str) -> float:
        """
        获取行为类型的权重
        
        Args:
            action: 行为类型
            
        Returns:
            行为权重，未知行为返回 0.0
        """
        return self.action_weights.get(action, 0.0)
    
    def validate(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            配置是否有效
            
        Raises:
            ValueError: 配置无效时抛出
        """
        # 验证序列长度
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length 必须为正数，当前值：{self.max_seq_length}")
        
        # 验证词表大小
        if self.vocab_size <= self.special_token_count:
            raise ValueError(f"vocab_size 必须大于 special_token_count")
        
        # 验证语义 ID 配置
        if len(self.codebook_sizes) != self.semantic_id_levels:
            raise ValueError(f"codebook_sizes 长度必须等于 semantic_id_levels")
        
        # 验证时间分桶配置
        if len(self.time_buckets) != len(self.time_bucket_hours):
            raise ValueError(f"time_buckets 和 time_bucket_hours 长度必须相同")
        
        return True
    
    def save(self, path: str) -> None:
        """
        保存配置到 JSON 文件
        
        Args:
            path: 保存路径
        """
        config_dict = {
            "max_seq_length": self.max_seq_length,
            "max_items_per_user": self.max_items_per_user,
            "min_sequence_length": self.min_sequence_length,
            "vocab_size": self.vocab_size,
            "min_token_freq": self.min_token_freq,
            "pad_token": self.pad_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token,
            "unk_token": self.unk_token,
            "pad_token_id": self.pad_token_id,
            "cls_token_id": self.cls_token_id,
            "sep_token_id": self.sep_token_id,
            "mask_token_id": self.mask_token_id,
            "unk_token_id": self.unk_token_id,
            "special_token_count": self.special_token_count,
            "semantic_id_levels": self.semantic_id_levels,
            "codebook_sizes": list(self.codebook_sizes),
            "embedding_dim": self.embedding_dim,
            "time_buckets": self.time_buckets,
            "time_bucket_hours": self.time_bucket_hours,
            "action_types": self.action_types,
            "action_weights": self.action_weights,
            "device_types": self.device_types,
            "context_types": self.context_types,
            "token_type_ids": self.token_type_ids,
            "num_negative_samples": self.num_negative_samples,
            "negative_sampling_strategy": self.negative_sampling_strategy,
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "FeatureConfig":
        """
        从 JSON 文件加载配置
        
        Args:
            path: 配置文件路径
            
        Returns:
            FeatureConfig 实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 将 codebook_sizes 列表转换为元组
        if "codebook_sizes" in config_dict:
            config_dict["codebook_sizes"] = tuple(config_dict["codebook_sizes"])
        
        # 将 time_bucket_hours 列表转换为元组列表
        if "time_bucket_hours" in config_dict:
            config_dict["time_bucket_hours"] = [tuple(h) for h in config_dict["time_bucket_hours"]]
        
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        return (
            f"FeatureConfig(\n"
            f"  max_seq_length={self.max_seq_length},\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  special_tokens={self.get_special_tokens()},\n"
            f"  action_types={self.action_types},\n"
            f"  device_types={self.device_types}\n"
            f")"
        )

