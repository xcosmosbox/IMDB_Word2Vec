"""
残差向量量化器 (Residual Vector Quantizer, RQ-VAE)

该模块实现了残差向量量化器，通过多层残差量化生成层次化的语义 ID。

核心原理：
1. 第一层量化器对原始特征进行量化，得到粗粒度表示
2. 后续层量化器对残差（原始特征 - 已量化部分）进行量化
3. 最终的量化结果是所有层量化结果的累加

层次化表示的优势：
- L1 (粗粒度): 1024 个码本，表示大类（如"电影"、"商品"）
- L2 (中粒度): 4096 个码本，表示子类（如"科幻电影"）
- L3 (细粒度): 16384 个码本，表示具体物品

语义相近的物品会具有相似的 ID 前缀，便于 Transformer 处理。

作者: Person A
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union

from .config import SemanticIDConfig
from .codebook import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """
    残差向量量化器 (RQ-VAE)
    
    通过多层残差量化生成层次化的语义 ID。
    
    工作原理：
    1. 输入特征 x
    2. 第一层: z1 = quantize(x), 残差 r1 = x - z1
    3. 第二层: z2 = quantize(r1), 残差 r2 = r1 - z2
    4. 第三层: z3 = quantize(r2)
    5. 最终量化结果: z = z1 + z2 + z3
    
    Attributes:
        config: 语义 ID 配置
        quantizers: 各层向量量化器列表
    
    Example:
        >>> config = SemanticIDConfig()
        >>> rq_vae = ResidualVectorQuantizer(config)
        >>> x = torch.randn(32, 256)
        >>> output = rq_vae(x)
        >>> output['indices']  # [L1_ids, L2_ids, L3_ids]
        >>> output['quantized']  # 量化后的特征
    """
    
    def __init__(self, config: SemanticIDConfig):
        """
        初始化残差向量量化器
        
        Args:
            config: 语义 ID 配置
        """
        super().__init__()
        
        self.config = config
        self.num_codebooks = config.num_codebooks
        
        # 创建多层量化器
        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                num_embeddings=size,
                embedding_dim=config.embedding_dim,
                commitment_cost=config.commitment_cost,
                ema_decay=config.ema_decay,
                epsilon=config.epsilon,
                use_ema=config.use_ema,
                init_scale=config.codebook_init_scale,
            )
            for size in config.codebook_sizes
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_losses: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        完整前向传播
        
        Args:
            x: 输入特征 (batch_size, embedding_dim)
            return_all_losses: 是否返回每层的损失
        
        Returns:
            包含以下键的字典:
            - quantized: 最终量化向量 (batch_size, embedding_dim)
            - indices: 各层码本索引列表 [L1_ids, L2_ids, L3_ids]
            - reconstruction_loss: 重建损失（标量）
            - commitment_loss: 承诺损失（标量）
            - total_loss: 总损失（标量）
            - per_level_losses: (可选) 每层的损失列表
        """
        batch_size = x.size(0)
        device = x.device
        
        # 初始化
        residual = x
        quantized_sum = torch.zeros_like(x)
        all_indices = []
        total_commitment_loss = torch.tensor(0.0, device=device)
        per_level_losses = []
        
        # 逐层量化
        for level, quantizer in enumerate(self.quantizers):
            # 量化当前残差
            quantized, indices, commitment_loss = quantizer(residual)
            
            # 记录索引
            all_indices.append(indices)
            
            # 累加量化结果
            quantized_sum = quantized_sum + quantized
            
            # 计算残差（停止梯度，防止梯度通过残差传播到前面的层）
            residual = residual - quantized.detach()
            
            # 累加承诺损失
            total_commitment_loss = total_commitment_loss + commitment_loss
            
            if return_all_losses:
                per_level_losses.append(commitment_loss.item())
        
        # 计算重建损失
        reconstruction_loss = F.mse_loss(quantized_sum, x)
        
        # 计算总损失
        total_loss = reconstruction_loss + self.config.commitment_cost * total_commitment_loss
        
        result = {
            "quantized": quantized_sum,
            "indices": all_indices,
            "reconstruction_loss": reconstruction_loss,
            "commitment_loss": total_commitment_loss,
            "total_loss": total_loss,
        }
        
        if return_all_losses:
            result["per_level_losses"] = per_level_losses
        
        return result
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码为多层语义 ID（推理用，不计算损失）
        
        Args:
            x: 输入特征 (batch_size, embedding_dim)
        
        Returns:
            Tuple[L1_ids, L2_ids, L3_ids]: 各层的码本索引
        """
        with torch.no_grad():
            residual = x
            all_indices = []
            
            for quantizer in self.quantizers:
                # 计算距离，找最近邻
                distances = torch.cdist(residual, quantizer.embedding.weight)
                indices = distances.argmin(dim=-1)
                all_indices.append(indices)
                
                # 计算残差
                quantized = quantizer.embedding(indices)
                residual = residual - quantized
            
            return tuple(all_indices)
    
    def decode(
        self,
        l1_ids: torch.Tensor,
        l2_ids: torch.Tensor,
        l3_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        从语义 ID 解码重建向量
        
        Args:
            l1_ids: 第一层码本索引 (batch_size,)
            l2_ids: 第二层码本索引 (batch_size,)
            l3_ids: 第三层码本索引 (batch_size,)
        
        Returns:
            reconstructed: 重建向量 (batch_size, embedding_dim)
        """
        indices_list = [l1_ids, l2_ids, l3_ids]
        
        # 确保层数匹配
        assert len(indices_list) == self.num_codebooks, (
            f"提供的索引数量 ({len(indices_list)}) 与码本数量 ({self.num_codebooks}) 不匹配"
        )
        
        # 初始化重建向量
        device = l1_ids.device
        batch_size = l1_ids.size(0)
        reconstructed = torch.zeros(batch_size, self.config.embedding_dim, device=device)
        
        # 累加各层的量化向量
        for quantizer, indices in zip(self.quantizers, indices_list):
            reconstructed = reconstructed + quantizer.embedding(indices)
        
        return reconstructed
    
    def decode_from_list(self, indices_list: List[torch.Tensor]) -> torch.Tensor:
        """
        从索引列表解码重建向量
        
        Args:
            indices_list: 各层码本索引列表
        
        Returns:
            reconstructed: 重建向量 (batch_size, embedding_dim)
        """
        assert len(indices_list) == self.num_codebooks
        return self.decode(*indices_list)
    
    def get_codebook_embeddings(self, level: int) -> torch.Tensor:
        """
        获取指定层级的码本嵌入矩阵
        
        Args:
            level: 层级 (1, 2, 或 3)
        
        Returns:
            embeddings: 码本嵌入矩阵 (codebook_size, embedding_dim)
        """
        if level < 1 or level > self.num_codebooks:
            raise ValueError(
                f"层级必须在 1 到 {self.num_codebooks} 之间，当前值: {level}"
            )
        return self.quantizers[level - 1].get_codebook_embeddings()
    
    def get_all_codebook_embeddings(self) -> List[torch.Tensor]:
        """
        获取所有层级的码本嵌入矩阵
        
        Returns:
            embeddings_list: 各层码本嵌入矩阵列表
        """
        return [q.get_codebook_embeddings() for q in self.quantizers]
    
    def get_codebook_usage_stats(self) -> Dict[str, Dict[str, float]]:
        """
        获取所有层的码本使用统计信息
        
        Returns:
            包含各层统计信息的字典
        """
        stats = {}
        for i, quantizer in enumerate(self.quantizers):
            level = i + 1
            stats[f"level_{level}"] = quantizer.get_codebook_usage()
        return stats
    
    def reset_usage_stats(self) -> None:
        """重置所有层的使用统计信息"""
        for quantizer in self.quantizers:
            quantizer.reset_usage_stats()
    
    def reset_dead_codes(self, x: torch.Tensor, threshold: float = 0.01) -> Dict[str, int]:
        """
        重置所有层的死码本
        
        Args:
            x: 用于重新初始化的输入数据
            threshold: 使用频率阈值
        
        Returns:
            各层重置的码本数量
        """
        result = {}
        for i, quantizer in enumerate(self.quantizers):
            level = i + 1
            num_reset = quantizer.reset_dead_codes(x, threshold)
            result[f"level_{level}"] = num_reset
        return result
    
    def compute_perplexity(self, indices_list: List[torch.Tensor]) -> Dict[str, float]:
        """
        计算各层的困惑度
        
        困惑度衡量码本的使用均匀程度，越高表示码本利用率越好。
        
        Args:
            indices_list: 各层的码本索引
        
        Returns:
            各层的困惑度
        """
        result = {}
        for i, (quantizer, indices) in enumerate(zip(self.quantizers, indices_list)):
            level = i + 1
            
            # 计算每个码本的使用频率
            counts = torch.bincount(
                indices.view(-1),
                minlength=quantizer.num_embeddings,
            ).float()
            probs = counts / counts.sum()
            
            # 计算困惑度 = exp(entropy)
            safe_probs = probs.clamp(min=1e-10)
            entropy = -(safe_probs * safe_probs.log()).sum()
            perplexity = torch.exp(entropy).item()
            
            result[f"level_{level}"] = perplexity
        
        return result
    
    def extra_repr(self) -> str:
        return (
            f"num_codebooks={self.num_codebooks}, "
            f"codebook_sizes={self.config.codebook_sizes}, "
            f"embedding_dim={self.config.embedding_dim}"
        )


class RQVAELoss(nn.Module):
    """
    RQ-VAE 损失函数
    
    总损失 = 重建损失 + β * 承诺损失
    
    其中：
    - 重建损失: ||x - x_hat||^2
    - 承诺损失: ||sg[z] - e||^2 (sg = stop_gradient)
    - β: 承诺损失权重 (默认 0.25)
    """
    
    def __init__(self, commitment_cost: float = 0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
    
    def forward(
        self,
        rq_vae_output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            rq_vae_output: RQ-VAE 的输出字典
        
        Returns:
            包含各项损失的字典
        """
        reconstruction_loss = rq_vae_output["reconstruction_loss"]
        commitment_loss = rq_vae_output["commitment_loss"]
        
        total_loss = reconstruction_loss + self.commitment_cost * commitment_loss
        
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "commitment_loss": commitment_loss,
        }

