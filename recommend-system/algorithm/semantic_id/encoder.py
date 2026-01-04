"""
Semantic ID 编码器 - 完整实现

该模块实现了 SemanticIDEncoderInterface 接口，提供完整的语义 ID 编码功能。

核心功能：
1. 将物品特征向量编码为三层语义 ID (L1, L2, L3)
2. 从语义 ID 重建物品特征向量
3. 提供码本嵌入矩阵访问接口

设计考虑：
1. 继承 SemanticIDEncoderInterface 接口
2. 内部使用 ResidualVectorQuantizer 实现核心量化功能
3. 支持训练和推理模式
4. 提供模型保存和加载功能

作者: Person A
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional, Union
import os
import json

# 导入接口定义
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import SemanticIDEncoderInterface

from .config import SemanticIDConfig
from .rq_vae import ResidualVectorQuantizer


class SemanticIDEncoder(SemanticIDEncoderInterface, nn.Module):
    """
    语义 ID 编码器完整实现
    
    实现 SemanticIDEncoderInterface 接口，将物品特征向量编码为层次化的语义 ID。
    
    语义 ID 是一种层次化的物品编码方式：
    - L1 (粗粒度): 1024 个码本，表示大类（如"电影"、"商品"）
    - L2 (中粒度): 4096 个码本，表示子类（如"科幻电影"）
    - L3 (细粒度): 16384 个码本，表示具体物品
    
    Attributes:
        config: 语义 ID 配置
        rq_vae: 残差向量量化器
        _is_trained: 标记模型是否已训练
    
    Example:
        >>> from algorithm.semantic_id import SemanticIDEncoder, SemanticIDConfig
        >>> 
        >>> config = SemanticIDConfig()
        >>> encoder = SemanticIDEncoder(config)
        >>> 
        >>> # 编码
        >>> features = torch.randn(32, 256)
        >>> l1, l2, l3 = encoder.encode(features)
        >>> 
        >>> # 解码
        >>> reconstructed = encoder.decode(l1, l2, l3)
        >>> 
        >>> # 获取码本
        >>> codebook_l1 = encoder.get_codebook_embeddings(1)
    """
    
    def __init__(self, config: Optional[SemanticIDConfig] = None):
        """
        初始化语义 ID 编码器
        
        Args:
            config: 语义 ID 配置，如果为 None 则使用默认配置
        """
        # 调用两个父类的初始化
        nn.Module.__init__(self)
        
        # 使用默认配置
        if config is None:
            config = SemanticIDConfig()
        
        self.config = config
        
        # 创建残差向量量化器
        self.rq_vae = ResidualVectorQuantizer(config)
        
        # 标记是否已训练
        self._is_trained = False
    
    def encode(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码物品特征为语义 ID
        
        实现 SemanticIDEncoderInterface.encode 接口。
        
        Args:
            features: 物品特征向量 (batch_size, embedding_dim)
                      embedding_dim 默认为 256
        
        Returns:
            Tuple[L1_ids, L2_ids, L3_ids]:
                - L1_ids: (batch_size,) 第一层语义ID，范围 [0, 1024)
                - L2_ids: (batch_size,) 第二层语义ID，范围 [0, 4096)
                - L3_ids: (batch_size,) 第三层语义ID，范围 [0, 16384)
        
        Example:
            >>> encoder = SemanticIDEncoder(SemanticIDConfig())
            >>> features = torch.randn(32, 256)
            >>> l1, l2, l3 = encoder.encode(features)
            >>> l1.shape
            torch.Size([32])
            >>> l1.max() < 1024
            True
        """
        # 验证输入维度
        self._validate_input(features)
        
        # 使用 RQ-VAE 编码
        indices = self.rq_vae.encode(features)
        
        # 返回三层 ID
        return indices[0], indices[1], indices[2]
    
    def decode(
        self,
        l1_ids: torch.Tensor,
        l2_ids: torch.Tensor,
        l3_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        从语义 ID 重建物品特征向量
        
        实现 SemanticIDEncoderInterface.decode 接口。
        
        Args:
            l1_ids: (batch_size,) 第一层语义ID
            l2_ids: (batch_size,) 第二层语义ID
            l3_ids: (batch_size,) 第三层语义ID
        
        Returns:
            reconstructed: (batch_size, embedding_dim) 重建的特征向量
        
        Example:
            >>> encoder = SemanticIDEncoder(SemanticIDConfig())
            >>> features = torch.randn(32, 256)
            >>> l1, l2, l3 = encoder.encode(features)
            >>> reconstructed = encoder.decode(l1, l2, l3)
            >>> reconstructed.shape
            torch.Size([32, 256])
        """
        # 验证输入
        self._validate_indices(l1_ids, l2_ids, l3_ids)
        
        # 使用 RQ-VAE 解码
        return self.rq_vae.decode(l1_ids, l2_ids, l3_ids)
    
    def get_codebook_embeddings(self, level: int) -> torch.Tensor:
        """
        获取指定层级的码本嵌入
        
        实现 SemanticIDEncoderInterface.get_codebook_embeddings 接口。
        
        Args:
            level: 层级 (1, 2, 或 3)
        
        Returns:
            embeddings: (codebook_size, embedding_dim) 码本嵌入矩阵
        
        Example:
            >>> encoder = SemanticIDEncoder(SemanticIDConfig())
            >>> codebook = encoder.get_codebook_embeddings(1)
            >>> codebook.shape
            torch.Size([1024, 256])
        """
        return self.rq_vae.get_codebook_embeddings(level)
    
    def forward(
        self,
        features: torch.Tensor,
        return_loss: bool = True,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        前向传播（训练用）
        
        在训练模式下使用此方法，可以获取损失值。
        
        Args:
            features: 输入特征 (batch_size, embedding_dim)
            return_loss: 是否返回损失
        
        Returns:
            包含以下键的字典:
            - l1_ids, l2_ids, l3_ids: 各层语义 ID
            - quantized: 量化后的特征
            - reconstruction_loss: 重建损失
            - commitment_loss: 承诺损失
            - total_loss: 总损失
        
        Example:
            >>> encoder = SemanticIDEncoder(SemanticIDConfig())
            >>> features = torch.randn(32, 256)
            >>> output = encoder(features)
            >>> output['total_loss']  # 用于反向传播
        """
        # 验证输入
        self._validate_input(features)
        
        # RQ-VAE 前向传播
        rq_output = self.rq_vae(features, return_all_losses=True)
        
        # 构建输出
        result = {
            "l1_ids": rq_output["indices"][0],
            "l2_ids": rq_output["indices"][1],
            "l3_ids": rq_output["indices"][2],
            "quantized": rq_output["quantized"],
        }
        
        if return_loss:
            result["reconstruction_loss"] = rq_output["reconstruction_loss"]
            result["commitment_loss"] = rq_output["commitment_loss"]
            result["total_loss"] = rq_output["total_loss"]
            if "per_level_losses" in rq_output:
                result["per_level_losses"] = rq_output["per_level_losses"]
        
        return result
    
    def _validate_input(self, features: torch.Tensor) -> None:
        """验证输入特征"""
        if features.dim() != 2:
            raise ValueError(
                f"输入特征必须是 2D 张量 (batch_size, embedding_dim)，"
                f"当前维度: {features.dim()}"
            )
        
        if features.size(-1) != self.config.embedding_dim:
            raise ValueError(
                f"输入特征维度 ({features.size(-1)}) "
                f"与配置的嵌入维度 ({self.config.embedding_dim}) 不匹配"
            )
    
    def _validate_indices(
        self,
        l1_ids: torch.Tensor,
        l2_ids: torch.Tensor,
        l3_ids: torch.Tensor,
    ) -> None:
        """验证索引"""
        # 验证形状一致
        if not (l1_ids.shape == l2_ids.shape == l3_ids.shape):
            raise ValueError(
                f"所有层级的索引形状必须一致: "
                f"L1={l1_ids.shape}, L2={l2_ids.shape}, L3={l3_ids.shape}"
            )
        
        # 验证维度
        if l1_ids.dim() != 1:
            raise ValueError(
                f"索引必须是 1D 张量，当前维度: {l1_ids.dim()}"
            )
        
        # 验证范围
        if l1_ids.max() >= self.config.codebook_sizes[0]:
            raise ValueError(
                f"L1 索引超出范围: max={l1_ids.max()}, "
                f"codebook_size={self.config.codebook_sizes[0]}"
            )
        
        if l2_ids.max() >= self.config.codebook_sizes[1]:
            raise ValueError(
                f"L2 索引超出范围: max={l2_ids.max()}, "
                f"codebook_size={self.config.codebook_sizes[1]}"
            )
        
        if l3_ids.max() >= self.config.codebook_sizes[2]:
            raise ValueError(
                f"L3 索引超出范围: max={l3_ids.max()}, "
                f"codebook_size={self.config.codebook_sizes[2]}"
            )
    
    def compute_reconstruction_error(
        self,
        features: torch.Tensor,
    ) -> Dict[str, float]:
        """
        计算重建误差
        
        Args:
            features: 输入特征 (batch_size, embedding_dim)
        
        Returns:
            包含各种误差指标的字典:
            - mse: 均方误差
            - rmse: 均方根误差
            - mae: 平均绝对误差
            - cosine_sim: 平均余弦相似度
        """
        with torch.no_grad():
            # 编码
            l1, l2, l3 = self.encode(features)
            
            # 解码
            reconstructed = self.decode(l1, l2, l3)
            
            # 计算各种误差
            mse = F.mse_loss(reconstructed, features).item()
            rmse = mse ** 0.5
            mae = F.l1_loss(reconstructed, features).item()
            
            # 余弦相似度
            cos_sim = F.cosine_similarity(reconstructed, features, dim=-1).mean().item()
            
            return {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "cosine_sim": cos_sim,
            }
    
    def get_codebook_usage(self) -> Dict[str, Dict[str, float]]:
        """
        获取码本使用统计信息
        
        Returns:
            各层的使用统计
        """
        return self.rq_vae.get_codebook_usage_stats()
    
    def reset_codebook_usage(self) -> None:
        """重置码本使用统计"""
        self.rq_vae.reset_usage_stats()
    
    def save_pretrained(self, save_directory: str) -> None:
        """
        保存模型到目录
        
        Args:
            save_directory: 保存目录路径
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(save_directory, "config.yaml")
        self.config.save(config_path)
        
        # 保存模型权重
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # 保存元信息
        meta_path = os.path.join(save_directory, "meta.json")
        meta = {
            "model_type": "SemanticIDEncoder",
            "version": "1.0.0",
            "is_trained": self._is_trained,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_directory: str) -> "SemanticIDEncoder":
        """
        从目录加载模型
        
        Args:
            load_directory: 模型目录路径
        
        Returns:
            加载的 SemanticIDEncoder 实例
        """
        # 加载配置
        config_path = os.path.join(load_directory, "config.yaml")
        config = SemanticIDConfig.load(config_path)
        
        # 创建模型
        encoder = cls(config)
        
        # 加载权重
        model_path = os.path.join(load_directory, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu")
        encoder.load_state_dict(state_dict)
        
        # 加载元信息
        meta_path = os.path.join(load_directory, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                encoder._is_trained = meta.get("is_trained", False)
        
        return encoder
    
    def get_num_parameters(self) -> Dict[str, int]:
        """
        获取模型参数数量
        
        Returns:
            参数数量统计
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 每层码本的参数量
        codebook_params = {}
        for i, quantizer in enumerate(self.rq_vae.quantizers):
            level = i + 1
            codebook_params[f"level_{level}"] = quantizer.embedding.weight.numel()
        
        return {
            "total": total,
            "trainable": trainable,
            "codebook": codebook_params,
        }
    
    @property
    def is_trained(self) -> bool:
        """返回模型是否已训练"""
        return self._is_trained
    
    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        """设置模型训练状态"""
        self._is_trained = value
    
    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.config.embedding_dim}, "
            f"codebook_sizes={self.config.codebook_sizes}, "
            f"is_trained={self._is_trained}"
        )


# 导入 F 模块用于损失计算
import torch.nn.functional as F


def create_encoder(
    preset: str = "medium",
    **kwargs,
) -> SemanticIDEncoder:
    """
    创建语义 ID 编码器的工厂函数
    
    Args:
        preset: 预设配置，可选 "small", "medium", "large", "production"
        **kwargs: 额外的配置参数，会覆盖预设值
    
    Returns:
        SemanticIDEncoder 实例
    
    Example:
        >>> encoder = create_encoder("small")
        >>> encoder = create_encoder("medium", commitment_cost=0.5)
    """
    from .config import PresetConfigs
    
    # 获取预设配置
    preset_map = {
        "small": PresetConfigs.small,
        "medium": PresetConfigs.medium,
        "large": PresetConfigs.large,
        "production": PresetConfigs.production,
    }
    
    if preset not in preset_map:
        raise ValueError(
            f"未知的预设配置: {preset}，可选值: {list(preset_map.keys())}"
        )
    
    config = preset_map[preset]()
    
    # 应用额外参数
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        config = SemanticIDConfig.from_dict(config_dict)
    
    return SemanticIDEncoder(config)

