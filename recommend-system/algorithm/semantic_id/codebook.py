"""
单层向量量化器 (Vector Quantizer)

该模块实现了单层向量量化器，是 RQ-VAE 的基础组件。

核心功能：
1. 将连续的特征向量映射到最近的码本向量
2. 使用 EMA (Exponential Moving Average) 更新码本，避免码本坍塌问题
3. 使用 Straight-Through Estimator 实现梯度传播

技术细节：
- 码本初始化使用均匀分布，范围 [-1/K, 1/K]，K 为码本大小
- EMA 更新比梯度下降更稳定，且不需要额外的码本损失
- Straight-Through Estimator: 前向传播使用量化值，反向传播使用原始梯度

作者: Person A
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class VectorQuantizer(nn.Module):
    """
    单层向量量化器
    
    使用 EMA (Exponential Moving Average) 更新码本，避免码本坍塌问题。
    
    工作原理:
    1. 计算输入向量与所有码本向量的距离
    2. 选择最近的码本向量作为量化结果
    3. 使用 Straight-Through Estimator 传播梯度
    4. 使用 EMA 更新码本嵌入
    
    Attributes:
        num_embeddings: 码本大小（码本向量数量）
        embedding_dim: 嵌入维度
        commitment_cost: 承诺损失权重 β
        ema_decay: EMA 衰减率
        epsilon: 数值稳定性常数
        use_ema: 是否使用 EMA 更新
    
    Example:
        >>> quantizer = VectorQuantizer(num_embeddings=1024, embedding_dim=256)
        >>> x = torch.randn(32, 256)  # batch_size=32
        >>> quantized, indices, loss = quantizer(x)
        >>> quantized.shape
        torch.Size([32, 256])
        >>> indices.shape
        torch.Size([32])
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        use_ema: bool = True,
        init_scale: float = 1.0,
    ):
        """
        初始化向量量化器
        
        Args:
            num_embeddings: 码本大小
            embedding_dim: 嵌入维度
            commitment_cost: 承诺损失权重 β，用于控制编码器输出与量化结果的接近程度
            ema_decay: EMA 衰减率，较大的值意味着更慢的更新
            epsilon: 数值稳定性常数
            use_ema: 是否使用 EMA 更新码本
            init_scale: 码本初始化缩放因子
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.use_ema = use_ema
        
        # 码本嵌入矩阵
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # 初始化码本：使用均匀分布
        init_bound = init_scale / num_embeddings
        self.embedding.weight.data.uniform_(-init_bound, init_bound)
        
        # EMA 统计量（仅在使用 EMA 时需要）
        if use_ema:
            # 每个码本向量被选中的次数（平滑后）
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            # 码本向量的加权和
            self.register_buffer("ema_embedding_sum", self.embedding.weight.clone())
            # 标记是否已经初始化
            self.register_buffer("_initialized", torch.tensor(False))
        
        # 统计信息
        self.register_buffer("_usage_count", torch.zeros(num_embeddings))
        self.register_buffer("_total_count", torch.tensor(0.0))
    
    def forward(
        self,
        x: torch.Tensor,
        return_distances: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：量化输入向量
        
        Args:
            x: 输入张量 (batch_size, embedding_dim)
            return_distances: 是否返回距离矩阵（用于调试）
        
        Returns:
            quantized: 量化后的向量 (batch_size, embedding_dim)
            indices: 码本索引 (batch_size,)
            commitment_loss: 承诺损失（标量）
        """
        # 确保输入维度正确
        assert x.dim() == 2, f"输入必须是 2D 张量，当前维度: {x.dim()}"
        assert x.size(-1) == self.embedding_dim, (
            f"输入特征维度 ({x.size(-1)}) 与嵌入维度 ({self.embedding_dim}) 不匹配"
        )
        
        # 计算输入与所有码本向量的欧氏距离
        # distances[i, j] = ||x[i] - embedding[j]||^2
        distances = self._compute_distances(x)
        
        # 找到最近的码本向量
        indices = distances.argmin(dim=-1)  # (batch_size,)
        
        # 获取量化向量
        quantized = self.embedding(indices)  # (batch_size, embedding_dim)
        
        # 计算承诺损失
        # 鼓励编码器输出接近其对应的量化向量
        commitment_loss = F.mse_loss(x, quantized.detach())
        
        # Straight-Through Estimator
        # 前向传播使用量化值，反向传播使用原始梯度
        quantized = x + (quantized - x).detach()
        
        # EMA 更新码本（仅在训练模式下）
        if self.training and self.use_ema:
            self._ema_update(x, indices)
        
        # 更新使用统计
        if self.training:
            self._update_usage_stats(indices)
        
        return quantized, indices, commitment_loss
    
    def _compute_distances(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算输入向量与码本向量的平方欧氏距离
        
        使用优化的计算方式:
        ||x - e||^2 = ||x||^2 + ||e||^2 - 2 * x @ e^T
        
        Args:
            x: 输入张量 (batch_size, embedding_dim)
        
        Returns:
            distances: 距离矩阵 (batch_size, num_embeddings)
        """
        # 使用 torch.cdist 计算欧氏距离
        # 这是 PyTorch 的优化实现
        return torch.cdist(x, self.embedding.weight, p=2.0).pow(2)
    
    def _ema_update(self, x: torch.Tensor, indices: torch.Tensor) -> None:
        """
        使用 EMA 更新码本嵌入
        
        更新规则:
        1. 更新每个码本的使用计数（平滑）
        2. 更新码本向量的加权和
        3. 归一化得到新的码本向量
        
        Args:
            x: 输入向量 (batch_size, embedding_dim)
            indices: 分配的码本索引 (batch_size,)
        """
        # 获取设备
        device = x.device
        
        # 转换为 one-hot 编码
        encodings = F.one_hot(indices, self.num_embeddings).float()  # (batch_size, num_embeddings)
        
        # 计算每个码本在这个 batch 中被选中的次数
        batch_cluster_size = encodings.sum(dim=0)  # (num_embeddings,)
        
        # 计算每个码本在这个 batch 中对应输入的和
        batch_embedding_sum = encodings.T @ x  # (num_embeddings, embedding_dim)
        
        # EMA 更新聚类大小
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            batch_cluster_size, alpha=1 - self.ema_decay
        )
        
        # EMA 更新嵌入和
        self.ema_embedding_sum.mul_(self.ema_decay).add_(
            batch_embedding_sum, alpha=1 - self.ema_decay
        )
        
        # Laplace 平滑：防止除零和稳定训练
        n = self.ema_cluster_size.sum()
        smoothed_cluster_size = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon)
            * n
        )
        
        # 更新码本权重
        self.embedding.weight.data.copy_(
            self.ema_embedding_sum / smoothed_cluster_size.unsqueeze(1)
        )
    
    def _update_usage_stats(self, indices: torch.Tensor) -> None:
        """
        更新码本使用统计信息
        
        Args:
            indices: 分配的码本索引 (batch_size,)
        """
        # 统计每个码本被使用的次数
        for idx in indices:
            self._usage_count[idx] += 1
        self._total_count += indices.size(0)
    
    def get_codebook_usage(self) -> Dict[str, float]:
        """
        获取码本使用统计信息
        
        Returns:
            包含以下信息的字典:
            - utilization: 使用率（使用过的码本比例）
            - perplexity: 困惑度（有效码本数量的度量）
            - min_usage: 最小使用率
            - max_usage: 最大使用率
            - dead_codes: 死码本数量（从未使用的码本）
        """
        if self._total_count == 0:
            return {
                "utilization": 0.0,
                "perplexity": 0.0,
                "min_usage": 0.0,
                "max_usage": 0.0,
                "dead_codes": self.num_embeddings,
            }
        
        # 计算使用频率
        usage_freq = self._usage_count / self._total_count
        
        # 使用率：至少使用过一次的码本比例
        utilization = (self._usage_count > 0).float().mean().item()
        
        # 困惑度：exp(entropy)，表示有效码本数量
        # 避免 log(0)
        safe_freq = usage_freq.clamp(min=1e-10)
        entropy = -(safe_freq * safe_freq.log()).sum()
        perplexity = torch.exp(entropy).item()
        
        # 死码本数量
        dead_codes = (self._usage_count == 0).sum().item()
        
        return {
            "utilization": utilization,
            "perplexity": perplexity,
            "min_usage": usage_freq.min().item(),
            "max_usage": usage_freq.max().item(),
            "dead_codes": dead_codes,
        }
    
    def reset_usage_stats(self) -> None:
        """重置使用统计信息"""
        self._usage_count.zero_()
        self._total_count.zero_()
    
    def reset_dead_codes(self, x: torch.Tensor, threshold: float = 0.01) -> int:
        """
        重置死码本（使用频率低于阈值的码本）
        
        将死码本重新初始化为输入数据中的随机向量
        
        Args:
            x: 用于重新初始化的输入数据 (batch_size, embedding_dim)
            threshold: 使用频率阈值
        
        Returns:
            重置的码本数量
        """
        if self._total_count == 0:
            return 0
        
        # 找到死码本
        usage_freq = self._usage_count / self._total_count
        dead_mask = usage_freq < threshold
        num_dead = dead_mask.sum().item()
        
        if num_dead == 0:
            return 0
        
        # 获取死码本的索引
        dead_indices = torch.where(dead_mask)[0]
        
        # 从输入中随机采样来重新初始化
        num_samples = min(len(dead_indices), x.size(0))
        if num_samples > 0:
            perm = torch.randperm(x.size(0))[:num_samples]
            sampled = x[perm]
            
            # 添加小量噪声
            noise = torch.randn_like(sampled) * 0.01
            new_embeddings = sampled + noise
            
            # 更新码本
            self.embedding.weight.data[dead_indices[:num_samples]] = new_embeddings
            
            # 重置 EMA 统计
            if self.use_ema:
                self.ema_cluster_size[dead_indices[:num_samples]] = 0
                self.ema_embedding_sum[dead_indices[:num_samples]] = new_embeddings
        
        return num_dead
    
    def quantize_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        根据索引获取量化向量
        
        Args:
            indices: 码本索引 (batch_size,) 或 (batch_size, seq_len)
        
        Returns:
            quantized: 量化向量
        """
        return self.embedding(indices)
    
    def get_codebook_embeddings(self) -> torch.Tensor:
        """
        获取码本嵌入矩阵
        
        Returns:
            embeddings: 码本嵌入矩阵 (num_embeddings, embedding_dim)
        """
        return self.embedding.weight.data.clone()
    
    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"commitment_cost={self.commitment_cost}, "
            f"ema_decay={self.ema_decay}, "
            f"use_ema={self.use_ema}"
        )

