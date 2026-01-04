"""
学习率调度模块

实现多种学习率调度策略：
- Linear: 线性衰减
- Cosine: 余弦退火
- Polynomial: 多项式衰减
- CosineWithRestarts: 带重启的余弦退火
- InverseSqrt: 逆平方根衰减

对应架构文档: 第八章 训练流程
"""

import math
from typing import Optional, List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


class WarmupLR(_LRScheduler):
    """
    带预热的学习率调度器基类
    
    在预热阶段，学习率从 0 线性增加到初始学习率
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 10000,
        last_epoch: int = -1,
    ):
        """
        初始化
        
        Args:
            optimizer: 优化器
            warmup_steps: 预热步数
            last_epoch: 上一个 epoch（用于恢复训练）
        """
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_warmup_factor(self, step: int) -> float:
        """获取预热因子"""
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return 1.0


class LinearLR(WarmupLR):
    """
    线性学习率调度
    
    预热后线性衰减到 min_lr
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 10000,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        初始化
        
        Args:
            optimizer: 优化器
            total_steps: 总训练步数
            warmup_steps: 预热步数
            min_lr_ratio: 最小学习率比例
            last_epoch: 上一个 epoch
        """
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, warmup_steps, last_epoch)
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        step = self.last_epoch
        warmup_factor = self.get_warmup_factor(step)
        
        if step < self.warmup_steps:
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # 线性衰减
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        factor = max(self.min_lr_ratio, 1.0 - progress)
        
        return [base_lr * factor for base_lr in self.base_lrs]


class CosineLR(WarmupLR):
    """
    余弦退火学习率调度
    
    预热后使用余弦函数衰减到 min_lr
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 10000,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        """
        初始化
        
        Args:
            optimizer: 优化器
            total_steps: 总训练步数
            warmup_steps: 预热步数
            min_lr_ratio: 最小学习率比例
            last_epoch: 上一个 epoch
        """
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, warmup_steps, last_epoch)
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        step = self.last_epoch
        warmup_factor = self.get_warmup_factor(step)
        
        if step < self.warmup_steps:
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # 余弦衰减
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        return [base_lr * factor for base_lr in self.base_lrs]


class PolynomialLR(WarmupLR):
    """
    多项式学习率调度
    
    预热后使用多项式函数衰减
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 10000,
        min_lr_ratio: float = 0.0,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
        """
        初始化
        
        Args:
            optimizer: 优化器
            total_steps: 总训练步数
            warmup_steps: 预热步数
            min_lr_ratio: 最小学习率比例
            power: 多项式幂次
            last_epoch: 上一个 epoch
        """
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.power = power
        super().__init__(optimizer, warmup_steps, last_epoch)
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        step = self.last_epoch
        warmup_factor = self.get_warmup_factor(step)
        
        if step < self.warmup_steps:
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # 多项式衰减
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        factor = (1 - progress) ** self.power
        factor = max(self.min_lr_ratio, factor)
        
        return [base_lr * factor for base_lr in self.base_lrs]


class CosineWithRestartsLR(WarmupLR):
    """
    带重启的余弦退火学习率调度
    
    每隔一定步数重启学习率
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 10000,
        num_cycles: int = 1,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        初始化
        
        Args:
            optimizer: 优化器
            total_steps: 总训练步数
            warmup_steps: 预热步数
            num_cycles: 重启周期数
            min_lr_ratio: 最小学习率比例
            last_epoch: 上一个 epoch
        """
        self.total_steps = total_steps
        self.num_cycles = num_cycles
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, warmup_steps, last_epoch)
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        step = self.last_epoch
        warmup_factor = self.get_warmup_factor(step)
        
        if step < self.warmup_steps:
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # 带重启的余弦衰减
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * ((self.num_cycles * progress) % 1.0))
        )
        
        return [base_lr * factor for base_lr in self.base_lrs]


class InverseSqrtLR(WarmupLR):
    """
    逆平方根学习率调度
    
    Transformer 原始论文使用的调度策略
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 10000,
        last_epoch: int = -1,
    ):
        """
        初始化
        
        Args:
            optimizer: 优化器
            warmup_steps: 预热步数
            last_epoch: 上一个 epoch
        """
        super().__init__(optimizer, warmup_steps, last_epoch)
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        step = max(1, self.last_epoch)
        
        if step < self.warmup_steps:
            factor = float(step) / float(max(1, self.warmup_steps))
        else:
            factor = math.sqrt(self.warmup_steps / step)
        
        return [base_lr * factor for base_lr in self.base_lrs]


class ConstantLR(WarmupLR):
    """
    常数学习率调度
    
    预热后保持恒定学习率
    """
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        step = self.last_epoch
        warmup_factor = self.get_warmup_factor(step)
        
        if step < self.warmup_steps:
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        return list(self.base_lrs)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    total_steps: int = 100000,
    warmup_steps: int = 10000,
    warmup_ratio: float = 0.0,
    min_lr_ratio: float = 0.1,
    num_cycles: int = 1,
    power: float = 1.0,
) -> _LRScheduler:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        total_steps: 总训练步数
        warmup_steps: 预热步数
        warmup_ratio: 预热比例（与 warmup_steps 二选一）
        min_lr_ratio: 最小学习率比例
        num_cycles: 重启周期数（仅用于 cosine_with_restarts）
        power: 多项式幂次（仅用于 polynomial）
    
    Returns:
        学习率调度器
    """
    # 使用比例计算预热步数
    if warmup_ratio > 0:
        warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "linear":
        return LinearLR(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == "cosine":
        return CosineLR(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == "polynomial":
        return PolynomialLR(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
            power=power,
        )
    elif scheduler_type == "cosine_with_restarts":
        return CosineWithRestartsLR(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            num_cycles=num_cycles,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_type == "inverse_sqrt":
        return InverseSqrtLR(
            optimizer,
            warmup_steps=warmup_steps,
        )
    elif scheduler_type == "constant":
        return ConstantLR(
            optimizer,
            warmup_steps=warmup_steps,
        )
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")


def get_scheduler_with_warmup(
    optimizer: Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
    **kwargs,
) -> LambdaLR:
    """
    兼容 HuggingFace Transformers 风格的调度器创建函数
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        **kwargs: 其他参数
    
    Returns:
        LambdaLR 调度器
    """
    return create_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        total_steps=num_training_steps,
        warmup_steps=num_warmup_steps,
        **kwargs,
    )

