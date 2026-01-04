"""
优化器模块

实现训练优化器的创建和配置：
- AdamW: 带权重衰减的 Adam
- LAMB: 大批量训练优化器
- Adafactor: 内存高效优化器

对应架构文档: 第八章 训练流程
"""

import math
from typing import Dict, List, Optional, Tuple, Iterator

import torch
import torch.nn as nn
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    AdamW 优化器
    
    带有解耦权重衰减的 Adam 优化器实现
    
    参考: Loshchilov & Hutter, "Decoupled Weight Decay Regularization"
    """
    
    def __init__(
        self,
        params: Iterator[nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ):
        """
        初始化 AdamW 优化器
        
        Args:
            params: 模型参数
            lr: 学习率
            betas: Adam 的 beta1 和 beta2
            eps: 防止除零的小常数
            weight_decay: 权重衰减系数
            amsgrad: 是否使用 AMSGrad 变体
        """
        if lr < 0.0:
            raise ValueError(f"无效的学习率: {lr}")
        if eps < 0.0:
            raise ValueError(f"无效的 epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的 beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的 beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"无效的权重衰减: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """执行单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW 不支持稀疏梯度")
                
                amsgrad = group['amsgrad']
                
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 解耦权重衰减
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                
                # 一阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 二阶矩估计
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # 偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # 参数更新
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


class LAMB(Optimizer):
    """
    LAMB 优化器
    
    Layer-wise Adaptive Moments optimizer for Batch training
    适用于大批量训练，可以使用更大的学习率
    
    参考: You et al., "Large Batch Optimization for Deep Learning"
    """
    
    def __init__(
        self,
        params: Iterator[nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        adam: bool = False,
        trust_clip: bool = False,
    ):
        """
        初始化 LAMB 优化器
        
        Args:
            params: 模型参数
            lr: 学习率
            betas: Adam 的 beta1 和 beta2
            eps: 防止除零的小常数
            weight_decay: 权重衰减系数
            adam: 如果为 True，则退化为 Adam
            trust_clip: 是否裁剪信任比率
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            adam=adam,
            trust_clip=trust_clip,
        )
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """执行单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB 不支持稀疏梯度")
                
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 一阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 二阶矩估计
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                
                # Adam 更新方向
                adam_step = exp_avg_hat / (exp_avg_sq_hat.sqrt().add(group['eps']))
                
                # 加入权重衰减
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])
                
                # 计算信任比率
                if group['adam']:
                    trust_ratio = 1.0
                else:
                    weight_norm = p.data.norm()
                    adam_norm = adam_step.norm()
                    
                    if weight_norm == 0 or adam_norm == 0:
                        trust_ratio = 1.0
                    else:
                        trust_ratio = weight_norm / adam_norm
                        if group['trust_clip']:
                            trust_ratio = min(trust_ratio, 10.0)
                
                # 参数更新
                p.data.add_(adam_step, alpha=-group['lr'] * trust_ratio)
        
        return loss


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    no_decay_params: Optional[List[str]] = None,
) -> Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型
        optimizer_type: 优化器类型 ("adamw", "adam", "lamb", "sgd")
        learning_rate: 学习率
        weight_decay: 权重衰减系数
        adam_beta1: Adam beta1
        adam_beta2: Adam beta2
        adam_epsilon: Adam epsilon
        no_decay_params: 不应用权重衰减的参数名称模式列表
    
    Returns:
        优化器实例
    """
    if no_decay_params is None:
        no_decay_params = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    
    # 分离需要和不需要权重衰减的参数
    decay_params = []
    no_decay_params_list = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 检查是否需要权重衰减
        should_decay = True
        for pattern in no_decay_params:
            if pattern in name:
                should_decay = False
                break
        
        if should_decay:
            decay_params.append(param)
        else:
            no_decay_params_list.append(param)
    
    # 参数组
    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params_list,
            "weight_decay": 0.0,
        },
    ]
    
    # 创建优化器
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
        )
    elif optimizer_type == "lamb":
        optimizer = LAMB(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adafactor":
        try:
            from transformers import Adafactor
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=learning_rate,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
        except ImportError:
            raise ImportError("使用 Adafactor 需要安装 transformers")
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    return optimizer


def get_parameter_groups(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    layer_lr_decay: float = 1.0,
    no_decay_params: Optional[List[str]] = None,
) -> List[Dict]:
    """
    获取分层学习率的参数组
    
    对于深层模型，可以对不同层使用不同的学习率（layer-wise decay）
    
    Args:
        model: 模型
        learning_rate: 基础学习率
        weight_decay: 权重衰减
        layer_lr_decay: 层学习率衰减因子（越深层学习率越低）
        no_decay_params: 不应用权重衰减的参数模式
    
    Returns:
        参数组列表
    """
    if no_decay_params is None:
        no_decay_params = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    
    # 收集所有参数及其层信息
    param_groups = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 确定层级
        layer_id = 0
        if "encoder.layers" in name:
            layer_id = int(name.split("encoder.layers.")[1].split(".")[0])
        elif "decoder.layers" in name:
            layer_id = int(name.split("decoder.layers.")[1].split(".")[0])
        
        # 确定是否需要权重衰减
        should_decay = True
        for pattern in no_decay_params:
            if pattern in name:
                should_decay = False
                break
        
        # 创建参数组键
        group_key = (layer_id, should_decay)
        
        if group_key not in param_groups:
            param_groups[group_key] = {
                "params": [],
                "lr": learning_rate * (layer_lr_decay ** layer_id),
                "weight_decay": weight_decay if should_decay else 0.0,
            }
        
        param_groups[group_key]["params"].append(param)
    
    return list(param_groups.values())

