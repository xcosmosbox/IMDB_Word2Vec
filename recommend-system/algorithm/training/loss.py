"""
损失函数模块

实现生成式推荐系统的多种损失函数：
- NextTokenPredictionLoss: 下一个 Token 预测损失
- ContrastiveLoss: 用户-物品对比学习损失
- DPOLoss: Direct Preference Optimization 损失
- UnifiedLoss: 统一训练损失

对应架构文档: 第八章 训练损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class NextTokenPredictionLoss(nn.Module):
    """
    下一个 Token 预测损失 (NTP Loss)
    
    分别计算三层语义 ID 的交叉熵损失，并进行层次化加权：
    - L1 (粗粒度类目): 权重最高，错误代价最大
    - L2 (细粒度属性): 中等权重
    - L3 (实例区分): 权重最低，允许一定的变化
    
    L_ntp = w1 * CE(L1_logits, L1_labels) + 
            w2 * CE(L2_logits, L2_labels) + 
            w3 * CE(L3_logits, L3_labels)
    """
    
    def __init__(
        self,
        l1_weight: float = 0.5,
        l2_weight: float = 0.3,
        l3_weight: float = 0.2,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        """
        初始化 NTP 损失
        
        Args:
            l1_weight: L1 层损失权重
            l2_weight: L2 层损失权重
            l3_weight: L3 层损失权重
            ignore_index: 忽略的标签索引（用于 padding）
            label_smoothing: 标签平滑系数
            reduction: 损失归约方式 ("mean", "sum", "none")
        """
        super().__init__()
        
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.l3_weight = l3_weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        # 验证权重和为 1
        total_weight = l1_weight + l2_weight + l3_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"层次化损失权重之和应为 1.0，当前为 {total_weight}")
    
    def forward(
        self,
        l1_logits: torch.Tensor,
        l2_logits: torch.Tensor,
        l3_logits: torch.Tensor,
        labels_l1: torch.Tensor,
        labels_l2: torch.Tensor,
        labels_l3: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 NTP 损失
        
        Args:
            l1_logits: L1 层预测 logits (batch_size, seq_len, 1024)
            l2_logits: L2 层预测 logits (batch_size, seq_len, 4096)
            l3_logits: L3 层预测 logits (batch_size, seq_len, 16384)
            labels_l1: L1 层标签 (batch_size, seq_len)
            labels_l2: L2 层标签 (batch_size, seq_len)
            labels_l3: L3 层标签 (batch_size, seq_len)
        
        Returns:
            total_loss: 加权总损失
            loss_dict: 各层损失详情
        """
        # 计算各层交叉熵损失
        loss_l1 = F.cross_entropy(
            l1_logits.view(-1, l1_logits.size(-1)),
            labels_l1.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )
        
        loss_l2 = F.cross_entropy(
            l2_logits.view(-1, l2_logits.size(-1)),
            labels_l2.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )
        
        loss_l3 = F.cross_entropy(
            l3_logits.view(-1, l3_logits.size(-1)),
            labels_l3.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )
        
        # 层次化加权
        total_loss = (
            self.l1_weight * loss_l1 +
            self.l2_weight * loss_l2 +
            self.l3_weight * loss_l3
        )
        
        # 计算准确率（用于监控）
        with torch.no_grad():
            mask = labels_l1 != self.ignore_index
            acc_l1 = self._compute_accuracy(l1_logits, labels_l1, mask)
            acc_l2 = self._compute_accuracy(l2_logits, labels_l2, mask)
            acc_l3 = self._compute_accuracy(l3_logits, labels_l3, mask)
        
        loss_dict = {
            "ntp_loss": total_loss,
            "ntp_l1_loss": loss_l1,
            "ntp_l2_loss": loss_l2,
            "ntp_l3_loss": loss_l3,
            "ntp_l1_acc": acc_l1,
            "ntp_l2_acc": acc_l2,
            "ntp_l3_acc": acc_l3,
        }
        
        return total_loss, loss_dict
    
    def _compute_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算准确率"""
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        return accuracy


class ContrastiveLoss(nn.Module):
    """
    用户-物品对比学习损失 (InfoNCE)
    
    拉近正样本对（用户与其交互物品），推远负样本对：
    
    L_contrast = -log(exp(sim(u, i+)/τ) / Σ exp(sim(u, i)/τ))
    
    其中：
    - u: 用户表示向量
    - i+: 正样本物品表示
    - τ: 温度参数
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
    ):
        """
        初始化对比学习损失
        
        Args:
            temperature: 温度参数，控制分布的锐度
            normalize: 是否对表示向量进行 L2 归一化
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算对比学习损失
        
        Args:
            user_repr: 用户表示向量 (batch_size, d_model)
            item_repr: 物品表示向量 (batch_size, d_model)
            labels: 可选的配对标签，默认对角线为正样本
        
        Returns:
            loss: 对比学习损失
            metrics: 相关指标
        """
        batch_size = user_repr.shape[0]
        
        # L2 归一化
        if self.normalize:
            user_repr = F.normalize(user_repr, dim=-1)
            item_repr = F.normalize(item_repr, dim=-1)
        
        # 计算相似度矩阵 (batch_size, batch_size)
        sim_matrix = torch.matmul(user_repr, item_repr.T) / self.temperature
        
        # 默认对角线为正样本
        if labels is None:
            labels = torch.arange(batch_size, device=user_repr.device)
        
        # 双向 InfoNCE 损失
        # User -> Item
        loss_u2i = F.cross_entropy(sim_matrix, labels)
        
        # Item -> User
        loss_i2u = F.cross_entropy(sim_matrix.T, labels)
        
        # 平均双向损失
        loss = (loss_u2i + loss_i2u) / 2
        
        # 计算相关指标
        with torch.no_grad():
            # 正样本相似度
            pos_sim = torch.diag(sim_matrix).mean() * self.temperature
            
            # 准确率（Top-1）
            u2i_pred = sim_matrix.argmax(dim=1)
            i2u_pred = sim_matrix.T.argmax(dim=1)
            u2i_acc = (u2i_pred == labels).float().mean()
            i2u_acc = (i2u_pred == labels).float().mean()
        
        metrics = {
            "contrastive_loss": loss,
            "contrastive_u2i_loss": loss_u2i,
            "contrastive_i2u_loss": loss_i2u,
            "contrastive_pos_sim": pos_sim,
            "contrastive_u2i_acc": u2i_acc,
            "contrastive_i2u_acc": i2u_acc,
        }
        
        return loss, metrics


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization 损失
    
    让模型偏好用户选择的物品，远离未选择的物品：
    
    L_DPO = -log(σ(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
    
    其中：
    - y_w: chosen (用户选择的物品)
    - y_l: rejected (用户未选择的物品)
    - β: 温度参数
    - π: 当前策略模型
    - π_ref: 参考模型
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        reference_free: bool = False,
    ):
        """
        初始化 DPO 损失
        
        Args:
            beta: 温度参数，控制对偏好的敏感程度
            label_smoothing: 标签平滑系数
            reference_free: 是否使用无参考模型的 DPO
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.reference_free = reference_free
    
    def forward(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor] = None,
        reference_rejected_logps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 DPO 损失
        
        Args:
            chosen_logps: 当前模型对 chosen 的 log probability (batch_size,)
            rejected_logps: 当前模型对 rejected 的 log probability (batch_size,)
            reference_chosen_logps: 参考模型对 chosen 的 log probability
            reference_rejected_logps: 参考模型对 rejected 的 log probability
        
        Returns:
            loss: DPO 损失
            metrics: 相关指标
        """
        if self.reference_free:
            # 无参考模型版本
            log_ratio = chosen_logps - rejected_logps
        else:
            # 标准 DPO
            if reference_chosen_logps is None or reference_rejected_logps is None:
                raise ValueError("标准 DPO 需要参考模型的 log probabilities")
            
            # 计算相对于参考模型的 log ratio
            chosen_relative = chosen_logps - reference_chosen_logps
            rejected_relative = rejected_logps - reference_rejected_logps
            log_ratio = chosen_relative - rejected_relative
        
        # 计算 DPO 损失
        if self.label_smoothing > 0:
            # 带标签平滑的 DPO
            loss = (
                -F.logsigmoid(self.beta * log_ratio) * (1 - self.label_smoothing) +
                -F.logsigmoid(-self.beta * log_ratio) * self.label_smoothing
            ).mean()
        else:
            loss = -F.logsigmoid(self.beta * log_ratio).mean()
        
        # 计算相关指标
        with torch.no_grad():
            # 奖励差异
            rewards_chosen = self.beta * chosen_logps
            rewards_rejected = self.beta * rejected_logps
            reward_margins = (rewards_chosen - rewards_rejected).mean()
            
            # 准确率（模型是否正确偏好 chosen）
            accuracy = (chosen_logps > rejected_logps).float().mean()
        
        metrics = {
            "dpo_loss": loss,
            "dpo_reward_margin": reward_margins,
            "dpo_accuracy": accuracy,
            "dpo_chosen_logps": chosen_logps.mean(),
            "dpo_rejected_logps": rejected_logps.mean(),
        }
        
        return loss, metrics


class MoEBalanceLoss(nn.Module):
    """
    MoE 负载均衡损失
    
    确保专家之间的负载均衡，避免部分专家过载或空闲：
    
    L_balance = α * Σ_i (f_i * P_i)
    
    其中：
    - f_i: 专家 i 被分配的 token 比例
    - P_i: 专家 i 的平均路由概率
    """
    
    def __init__(self, num_experts: int = 16, alpha: float = 0.01):
        """
        初始化 MoE 平衡损失
        
        Args:
            num_experts: 专家数量
            alpha: 损失权重系数
        """
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
    
    def forward(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 MoE 平衡损失
        
        Args:
            router_probs: 路由概率 (batch_size, seq_len, num_experts)
            expert_indices: 选中的专家索引 (batch_size, seq_len, top_k)
        
        Returns:
            loss: 平衡损失
            metrics: 相关指标
        """
        batch_size, seq_len, _ = router_probs.shape
        
        # 计算每个专家的平均路由概率
        mean_probs = router_probs.mean(dim=[0, 1])  # (num_experts,)
        
        # 计算每个专家被选中的 token 比例
        one_hot = F.one_hot(expert_indices, num_classes=self.num_experts)  # (B, L, K, E)
        tokens_per_expert = one_hot.float().sum(dim=[0, 1, 2])  # (num_experts,)
        total_tokens = batch_size * seq_len * expert_indices.shape[-1]
        expert_fractions = tokens_per_expert / total_tokens
        
        # 负载均衡损失
        loss = self.alpha * self.num_experts * (mean_probs * expert_fractions).sum()
        
        # 计算相关指标
        with torch.no_grad():
            # 专家利用率（被使用的专家比例）
            expert_utilization = (tokens_per_expert > 0).float().mean()
            
            # 负载均衡度（越接近 1 越均衡）
            balance = 1.0 / (self.num_experts * expert_fractions.var() + 1e-8)
        
        metrics = {
            "moe_balance_loss": loss,
            "moe_expert_utilization": expert_utilization,
            "moe_balance_score": balance,
        }
        
        return loss, metrics


class UnifiedLoss(nn.Module):
    """
    统一训练损失
    
    组合所有损失函数：
    
    L_total = L_ntp + λ₁ * L_contrastive + λ₂ * L_preference + λ₃ * L_moe_balance
    
    对应架构文档: 第八章 训练损失函数
    """
    
    def __init__(
        self,
        # NTP 损失参数
        l1_weight: float = 0.5,
        l2_weight: float = 0.3,
        l3_weight: float = 0.2,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        # 对比学习参数
        contrastive_temperature: float = 0.07,
        # DPO 参数
        dpo_beta: float = 0.1,
        dpo_reference_free: bool = False,
        # 损失权重
        lambda_contrastive: float = 0.1,
        lambda_preference: float = 0.1,
        lambda_moe_balance: float = 0.01,
        # MoE 参数
        num_experts: int = 16,
    ):
        """
        初始化统一损失函数
        
        Args:
            l1_weight: L1 层 NTP 损失权重
            l2_weight: L2 层 NTP 损失权重
            l3_weight: L3 层 NTP 损失权重
            ignore_index: 忽略的标签索引
            label_smoothing: 标签平滑系数
            contrastive_temperature: 对比学习温度
            dpo_beta: DPO 温度参数
            dpo_reference_free: 是否使用无参考模型 DPO
            lambda_contrastive: 对比学习损失权重
            lambda_preference: 偏好损失权重
            lambda_moe_balance: MoE 平衡损失权重
            num_experts: MoE 专家数量
        """
        super().__init__()
        
        # 各损失函数
        self.ntp_loss = NextTokenPredictionLoss(
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            l3_weight=l3_weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        
        self.contrastive_loss = ContrastiveLoss(
            temperature=contrastive_temperature,
        )
        
        self.dpo_loss = DPOLoss(
            beta=dpo_beta,
            reference_free=dpo_reference_free,
        )
        
        self.moe_balance_loss = MoEBalanceLoss(
            num_experts=num_experts,
        )
        
        # 损失权重
        self.lambda_contrastive = lambda_contrastive
        self.lambda_preference = lambda_preference
        self.lambda_moe_balance = lambda_moe_balance
    
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        aux_loss: Optional[torch.Tensor] = None,
        contrastive_data: Optional[Dict[str, torch.Tensor]] = None,
        preference_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算统一损失
        
        Args:
            model_outputs: 模型输出字典
                - l1_logits: L1 层 logits
                - l2_logits: L2 层 logits
                - l3_logits: L3 层 logits
                - user_repr (可选): 用户表示
                - item_repr (可选): 物品表示
            labels: 标签字典
                - l1: L1 层标签
                - l2: L2 层标签
                - l3: L3 层标签
            aux_loss: 来自模型的辅助损失（如 MoE 平衡损失）
            contrastive_data: 对比学习数据（可选）
            preference_data: 偏好数据（可选）
        
        Returns:
            losses: 损失字典，包含所有损失及指标
        """
        losses = {}
        total_loss = 0.0
        
        # 1. NTP 损失
        ntp_loss, ntp_metrics = self.ntp_loss(
            model_outputs["l1_logits"],
            model_outputs["l2_logits"],
            model_outputs["l3_logits"],
            labels["l1"],
            labels["l2"],
            labels["l3"],
        )
        total_loss = total_loss + ntp_loss
        losses.update(ntp_metrics)
        
        # 2. 对比学习损失（如果有用户和物品表示）
        if (
            self.lambda_contrastive > 0 and
            "user_repr" in model_outputs and 
            "item_repr" in model_outputs
        ):
            contrastive_loss, contrastive_metrics = self.contrastive_loss(
                model_outputs["user_repr"],
                model_outputs["item_repr"],
            )
            total_loss = total_loss + self.lambda_contrastive * contrastive_loss
            losses.update(contrastive_metrics)
        else:
            losses["contrastive_loss"] = torch.tensor(0.0)
        
        # 3. 偏好损失（如果有偏好数据）
        if self.lambda_preference > 0 and preference_data is not None:
            dpo_loss, dpo_metrics = self.dpo_loss(
                preference_data["chosen_logps"],
                preference_data["rejected_logps"],
                preference_data.get("reference_chosen_logps"),
                preference_data.get("reference_rejected_logps"),
            )
            total_loss = total_loss + self.lambda_preference * dpo_loss
            losses.update(dpo_metrics)
        else:
            losses["dpo_loss"] = torch.tensor(0.0)
        
        # 4. MoE 负载均衡损失
        if aux_loss is not None:
            total_loss = total_loss + self.lambda_moe_balance * aux_loss
            losses["moe_balance_loss"] = aux_loss
        else:
            losses["moe_balance_loss"] = torch.tensor(0.0)
        
        # 总损失
        losses["total_loss"] = total_loss
        
        return losses


def compute_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    计算序列的 log probability
    
    Args:
        logits: 模型输出的 logits (batch_size, seq_len, vocab_size)
        labels: 标签 (batch_size, seq_len)
        ignore_index: 忽略的索引
    
    Returns:
        log_probs: 每个样本的 log probability (batch_size,)
    """
    # 计算每个位置的 log probability
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 收集对应标签的 log probability
    batch_size, seq_len = labels.shape
    log_probs = log_probs.view(-1, log_probs.size(-1))
    labels_flat = labels.view(-1)
    
    # 创建索引
    indices = labels_flat.unsqueeze(-1)
    
    # 忽略 padding
    mask = labels_flat != ignore_index
    indices = indices.clamp(min=0)  # 避免负数索引
    
    # 收集 log probabilities
    gathered = torch.gather(log_probs, dim=-1, index=indices).squeeze(-1)
    gathered = gathered * mask.float()
    
    # 重塑并求和
    gathered = gathered.view(batch_size, seq_len)
    mask = mask.view(batch_size, seq_len)
    
    # 返回每个样本的平均 log probability
    seq_log_probs = gathered.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    
    return seq_log_probs

