"""
Mixture of Experts (MoE) 模块

实现 MoE-Enhanced FFN，用于解码器中的稀疏激活计算。

核心特点：
1. 稀疏激活：每次只激活 Top-K 个专家
2. 负载均衡：通过辅助损失避免专家坍塌
3. 专家专业化：不同专家处理不同类型的推荐场景

参考论文：
- Switch Transformer (Google)
- OneRec (快手)

对应架构文档: 《生成式推荐系统架构设计》3.2.4 节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .config import DecoderConfig


class Expert(nn.Module):
    """
    单个专家网络（FFN）
    
    结构: Linear -> GELU -> Dropout -> Linear -> Dropout
    
    Args:
        d_model: 输入/输出维度
        d_ff: 中间隐藏层维度
        dropout: Dropout 率
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, seq_len, d_model) 或 (num_tokens, d_model)
            
        Returns:
            输出张量，形状与输入相同
        """
        return self.net(x)


class Router(nn.Module):
    """
    路由网络
    
    决定每个 token 应该被发送到哪些专家。
    使用 Top-K 选择策略。
    
    Args:
        d_model: 输入维度
        num_experts: 专家数量
        top_k: 每个 token 激活的专家数
        jitter_noise: 训练时添加的噪声（提高探索性）
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        jitter_noise: float = 0.0,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        
        # 路由层
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # 初始化
        nn.init.xavier_uniform_(self.router.weight)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算路由权重
        
        Args:
            x: (batch_size * seq_len, d_model) 扁平化的输入
            
        Returns:
            top_k_gates: (batch_size * seq_len, top_k) 归一化的门控权重
            top_k_indices: (batch_size * seq_len, top_k) 选中的专家索引
            router_probs: (batch_size * seq_len, num_experts) 完整的路由概率
        """
        # 计算路由 logits
        router_logits = self.router(x)  # (N, num_experts)
        
        # 训练时添加噪声
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise
        
        # 计算路由概率
        router_probs = F.softmax(router_logits, dim=-1)  # (N, num_experts)
        
        # 选择 Top-K 专家
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # (N, top_k)
        
        # 归一化选中专家的权重
        top_k_gates = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
        
        return top_k_gates, top_k_indices, router_probs


class MoEFeedForward(nn.Module):
    """
    Mixture of Experts 前馈网络
    
    核心特点：
    1. 稀疏激活：每次只激活 Top-K 个专家
    2. 负载均衡：通过辅助损失避免专家坍塌
    3. 专家专业化：不同专家处理不同类型的推荐
    
    实现要点：
    - 使用批量计算提高效率
    - 记录辅助损失用于训练
    
    Args:
        config: 解码器配置
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.d_model = config.d_model
        self.moe_loss_weight = config.moe_loss_weight
        
        # 路由网络
        self.router = Router(
            d_model=config.d_model,
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
            jitter_noise=config.moe_jitter_noise,
        )
        
        # 专家网络
        self.experts = nn.ModuleList([
            Expert(config.d_model, config.d_ff, config.dropout)
            for _ in range(config.num_experts)
        ])
        
        # 记录辅助损失
        self._aux_loss: torch.Tensor = torch.tensor(0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoE 前向传播
        
        Args:
            x: (batch_size, seq_len, d_model) 输入张量
        
        Returns:
            output: (batch_size, seq_len, d_model) 输出张量
        """
        batch_size, seq_len, d_model = x.shape
        
        # 扁平化输入: (batch_size * seq_len, d_model)
        x_flat = x.view(-1, d_model)
        num_tokens = x_flat.shape[0]
        
        # 计算路由权重
        top_k_gates, top_k_indices, router_probs = self.router(x_flat)
        
        # 计算负载均衡损失
        self._aux_loss = self._compute_balance_loss(router_probs, top_k_indices)
        
        # 初始化输出
        output = torch.zeros_like(x_flat)
        
        # 为每个专家计算输出
        for expert_idx, expert in enumerate(self.experts):
            # 找到选择了当前专家的 token
            # (N, top_k) 中哪些位置选择了 expert_idx
            expert_mask = (top_k_indices == expert_idx)  # (N, top_k)
            
            # 获取选择该专家的 token 索引
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            
            if token_indices.numel() == 0:
                continue
            
            # 获取对应的输入
            expert_input = x_flat[token_indices]  # (num_selected, d_model)
            
            # 计算专家输出
            expert_output = expert(expert_input)  # (num_selected, d_model)
            
            # 获取对应的门控权重
            # 对于每个选中的 token，找到该专家对应的权重
            expert_gates = torch.where(
                expert_mask[token_indices],
                top_k_gates[token_indices],
                torch.zeros_like(top_k_gates[token_indices])
            ).sum(dim=-1, keepdim=True)  # (num_selected, 1)
            
            # 加权累加到输出
            output[token_indices] += expert_output * expert_gates
        
        # 恢复形状
        output = output.view(batch_size, seq_len, d_model)
        
        return output
    
    def _compute_balance_loss(
        self,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算负载均衡损失
        
        目标：让所有专家被均匀使用
        
        L_balance = α * num_experts * Σ(f_i * P_i)
        
        其中:
        - f_i: 第 i 个专家被选中的 token 比例
        - P_i: 第 i 个专家的平均路由概率
        
        Args:
            router_probs: (N, num_experts) 路由概率
            top_k_indices: (N, top_k) 选中的专家索引
            
        Returns:
            balance_loss: 标量损失
        """
        num_tokens = router_probs.shape[0]
        device = router_probs.device
        
        if num_tokens == 0:
            return torch.tensor(0.0, device=device)
        
        # 计算每个专家被选中的频率 f_i
        # 将 top_k_indices 展平并统计每个专家被选中的次数
        expert_counts = torch.zeros(self.num_experts, device=device)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            expert_counts.scatter_add_(
                0, 
                expert_indices, 
                torch.ones_like(expert_indices, dtype=torch.float)
            )
        
        # f_i: 被选中的比例
        f = expert_counts / (num_tokens * self.top_k)
        
        # P_i: 平均路由概率
        P = router_probs.mean(dim=0)
        
        # 负载均衡损失
        balance_loss = self.num_experts * (f * P).sum()
        
        return balance_loss * self.moe_loss_weight
    
    def get_aux_loss(self) -> torch.Tensor:
        """
        获取辅助损失（用于训练）
        
        Returns:
            aux_loss: 标量损失张量
        """
        return self._aux_loss
    
    def get_expert_utilization(self) -> dict:
        """
        获取专家利用率统计（用于监控）
        
        Returns:
            统计信息字典
        """
        return {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "aux_loss": self._aux_loss.item() if self._aux_loss.numel() == 1 else 0.0,
        }


class SharedExpertMoE(nn.Module):
    """
    带共享专家的 MoE
    
    在 Top-K 路由专家之外，添加一个始终激活的共享专家，
    用于处理所有类型的通用特征。
    
    Args:
        config: 解码器配置
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        
        self.config = config
        
        # 共享专家（始终激活）
        self.shared_expert = Expert(config.d_model, config.d_ff, config.dropout)
        
        # 路由 MoE
        self.moe = MoEFeedForward(config)
        
        # 共享专家的权重
        self.shared_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # 共享专家输出
        shared_out = self.shared_expert(x)
        
        # MoE 输出
        moe_out = self.moe(x)
        
        # 加权组合
        weight = torch.sigmoid(self.shared_weight)
        output = weight * shared_out + (1 - weight) * moe_out
        
        return output
    
    def get_aux_loss(self) -> torch.Tensor:
        """获取辅助损失"""
        return self.moe.get_aux_loss()

