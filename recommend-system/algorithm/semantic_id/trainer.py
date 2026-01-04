"""
Semantic ID 编码器训练器

该模块实现了码本训练器，用于训练 RQ-VAE 模型。

训练流程：
1. 加载物品特征数据
2. 优化码本嵌入，最小化重建损失和承诺损失
3. 监控码本利用率，防止码本坍塌
4. 保存训练好的模型

训练技巧：
1. 使用 EMA 更新码本，比梯度下降更稳定
2. 监控死码本，及时重置
3. 使用学习率预热和衰减
4. 支持分布式训练

作者: Person A
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LambdaLR
from typing import Dict, List, Optional, Tuple, Any, Callable
import os
import time
import logging
from dataclasses import dataclass, field
from tqdm import tqdm
import json

from .config import SemanticIDConfig
from .encoder import SemanticIDEncoder

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    训练配置
    
    Attributes:
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        warmup_epochs: 学习率预热轮数
        grad_clip: 梯度裁剪阈值
        log_interval: 日志记录间隔（步数）
        eval_interval: 评估间隔（轮数）
        save_interval: 模型保存间隔（轮数）
        dead_code_threshold: 死码本阈值
        reset_dead_codes: 是否自动重置死码本
        use_amp: 是否使用混合精度训练
        seed: 随机种子
    """
    
    batch_size: int = 256
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    log_interval: int = 100
    eval_interval: int = 1
    save_interval: int = 1
    dead_code_threshold: float = 0.01
    reset_dead_codes: bool = True
    use_amp: bool = False
    seed: int = 42


class SemanticIDTrainer:
    """
    语义 ID 编码器训练器
    
    用于训练 RQ-VAE 模型，优化码本嵌入。
    
    Attributes:
        encoder: 语义 ID 编码器
        config: 训练配置
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 计算设备
    
    Example:
        >>> from algorithm.semantic_id import SemanticIDEncoder, SemanticIDConfig
        >>> from algorithm.semantic_id.trainer import SemanticIDTrainer, TrainingConfig
        >>> 
        >>> # 创建编码器和训练器
        >>> encoder = SemanticIDEncoder(SemanticIDConfig())
        >>> trainer = SemanticIDTrainer(encoder, TrainingConfig())
        >>> 
        >>> # 准备数据
        >>> train_features = torch.randn(10000, 256)
        >>> val_features = torch.randn(1000, 256)
        >>> 
        >>> # 训练
        >>> trainer.train(train_features, val_features, save_dir="./checkpoints")
    """
    
    def __init__(
        self,
        encoder: SemanticIDEncoder,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        """
        初始化训练器
        
        Args:
            encoder: 语义 ID 编码器
            config: 训练配置
            device: 计算设备，如果为 None 则自动选择
        """
        self.encoder = encoder
        self.config = config
        
        # 设置设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.encoder.to(device)
        
        # 创建优化器
        self.optimizer = AdamW(
            encoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # 学习率调度器
        self.scheduler = None  # 在 train 方法中初始化
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        
        # 训练历史
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "reconstruction_loss": [],
            "commitment_loss": [],
            "codebook_usage": [],
            "learning_rate": [],
        }
        
        # 最佳验证损失
        self.best_val_loss = float("inf")
        
        # 设置随机种子
        self._set_seed(config.seed)
    
    def _set_seed(self, seed: int) -> None:
        """设置随机种子"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _create_scheduler(self, num_training_steps: int) -> None:
        """创建学习率调度器"""
        warmup_steps = int(num_training_steps * self.config.warmup_epochs / self.config.num_epochs)
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) 
                / float(max(1, num_training_steps - warmup_steps))
            )
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
    
    def train(
        self,
        train_features: torch.Tensor,
        val_features: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_features: 训练数据特征 (num_samples, embedding_dim)
            val_features: 验证数据特征，可选
            save_dir: 模型保存目录，可选
            callbacks: 回调函数列表，可选
        
        Returns:
            训练历史字典
        """
        # 创建数据加载器
        train_loader = self._create_dataloader(train_features, shuffle=True)
        val_loader = self._create_dataloader(val_features, shuffle=False) if val_features is not None else None
        
        # 计算总训练步数
        num_training_steps = len(train_loader) * self.config.num_epochs
        self._create_scheduler(num_training_steps)
        
        # 创建保存目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"开始训练，设备: {self.device}")
        logger.info(f"训练样本数: {len(train_features)}")
        if val_features is not None:
            logger.info(f"验证样本数: {len(val_features)}")
        
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            # 训练一个 epoch
            train_metrics = self._train_epoch(train_loader, epoch, global_step)
            global_step += len(train_loader)
            
            # 记录训练指标
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["reconstruction_loss"].append(train_metrics["reconstruction_loss"])
            self.history["commitment_loss"].append(train_metrics["commitment_loss"])
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            
            # 验证
            if val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self._evaluate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f}"
                )
                
                # 保存最佳模型
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    if save_dir:
                        best_path = os.path.join(save_dir, "best_model")
                        self.encoder.save_pretrained(best_path)
                        logger.info(f"保存最佳模型到 {best_path}")
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f}"
                )
            
            # 码本使用统计
            usage_stats = self.encoder.get_codebook_usage()
            self.history["codebook_usage"].append(usage_stats)
            
            # 日志记录码本使用情况
            for level, stats in usage_stats.items():
                logger.info(
                    f"  {level}: 利用率={stats['utilization']:.2%}, "
                    f"困惑度={stats['perplexity']:.1f}, "
                    f"死码本={stats['dead_codes']}"
                )
            
            # 重置死码本
            if self.config.reset_dead_codes and (epoch + 1) % 2 == 0:
                # 使用一批训练数据重置死码本
                sample_batch = next(iter(train_loader)).to(self.device)
                reset_counts = self.encoder.rq_vae.reset_dead_codes(
                    sample_batch,
                    self.config.dead_code_threshold,
                )
                total_reset = sum(reset_counts.values())
                if total_reset > 0:
                    logger.info(f"重置了 {total_reset} 个死码本")
            
            # 重置使用统计
            self.encoder.reset_codebook_usage()
            
            # 保存检查点
            if save_dir and (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}")
                self.encoder.save_pretrained(checkpoint_path)
            
            # 调用回调
            if callbacks:
                for callback in callbacks:
                    callback(epoch, train_metrics, val_metrics if val_loader else None)
        
        # 标记模型已训练
        self.encoder.is_trained = True
        
        # 保存最终模型
        if save_dir:
            final_path = os.path.join(save_dir, "final_model")
            self.encoder.save_pretrained(final_path)
            
            # 保存训练历史
            history_path = os.path.join(save_dir, "training_history.json")
            with open(history_path, "w", encoding="utf-8") as f:
                # 转换不可序列化的类型
                serializable_history = {}
                for key, value in self.history.items():
                    if key == "codebook_usage":
                        serializable_history[key] = value
                    else:
                        serializable_history[key] = [float(v) for v in value]
                json.dump(serializable_history, f, indent=2)
        
        logger.info("训练完成！")
        return self.history
    
    def _create_dataloader(
        self,
        features: Optional[torch.Tensor],
        shuffle: bool = True,
    ) -> Optional[DataLoader]:
        """创建数据加载器"""
        if features is None:
            return None
        
        dataset = TensorDataset(features)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,  # 避免多进程问题
            pin_memory=True if self.device.type == "cuda" else False,
            drop_last=True,
        )
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        global_step: int,
    ) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Returns:
            训练指标字典
        """
        self.encoder.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_commit_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            leave=False,
        )
        
        for batch_idx, (batch,) in enumerate(progress_bar):
            batch = batch.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.encoder(batch)
                    loss = output["total_loss"]
            else:
                output = self.encoder(batch)
                loss = output["total_loss"]
            
            # 反向传播
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(),
                    self.config.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(),
                    self.config.grad_clip,
                )
                self.optimizer.step()
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 累计损失
            total_loss += loss.item()
            total_recon_loss += output["reconstruction_loss"].item()
            total_commit_loss += output["commitment_loss"].item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{output['reconstruction_loss'].item():.4f}",
                "commit": f"{output['commitment_loss'].item():.4f}",
            })
            
            # 日志记录
            current_step = global_step + batch_idx
            if (current_step + 1) % self.config.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.debug(
                    f"Step {current_step + 1} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {lr:.6f}"
                )
        
        return {
            "loss": total_loss / num_batches,
            "reconstruction_loss": total_recon_loss / num_batches,
            "commitment_loss": total_commit_loss / num_batches,
        }
    
    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        在验证集上评估
        
        Returns:
            验证指标字典
        """
        self.encoder.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_commit_loss = 0.0
        num_batches = 0
        
        for batch, in dataloader:
            batch = batch.to(self.device)
            
            output = self.encoder(batch)
            
            total_loss += output["total_loss"].item()
            total_recon_loss += output["reconstruction_loss"].item()
            total_commit_loss += output["commitment_loss"].item()
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "reconstruction_loss": total_recon_loss / num_batches,
            "commitment_loss": total_commit_loss / num_batches,
        }
    
    @torch.no_grad()
    def compute_codebook_utilization(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, Dict[str, float]]:
        """
        计算码本利用率
        
        Args:
            dataloader: 数据加载器
        
        Returns:
            各层的利用率统计
        """
        self.encoder.eval()
        self.encoder.reset_codebook_usage()
        
        # 统计码本使用情况
        all_l1_ids = []
        all_l2_ids = []
        all_l3_ids = []
        
        for batch, in dataloader:
            batch = batch.to(self.device)
            l1, l2, l3 = self.encoder.encode(batch)
            
            all_l1_ids.append(l1)
            all_l2_ids.append(l2)
            all_l3_ids.append(l3)
        
        # 合并所有索引
        all_l1_ids = torch.cat(all_l1_ids)
        all_l2_ids = torch.cat(all_l2_ids)
        all_l3_ids = torch.cat(all_l3_ids)
        
        # 计算困惑度
        perplexity = self.encoder.rq_vae.compute_perplexity([all_l1_ids, all_l2_ids, all_l3_ids])
        
        # 计算使用率
        def compute_usage(ids: torch.Tensor, num_embeddings: int) -> float:
            unique_ids = ids.unique()
            return len(unique_ids) / num_embeddings
        
        usage = {
            "level_1": {
                "utilization": compute_usage(all_l1_ids, self.encoder.config.codebook_sizes[0]),
                "perplexity": perplexity["level_1"],
            },
            "level_2": {
                "utilization": compute_usage(all_l2_ids, self.encoder.config.codebook_sizes[1]),
                "perplexity": perplexity["level_2"],
            },
            "level_3": {
                "utilization": compute_usage(all_l3_ids, self.encoder.config.codebook_sizes[2]),
                "perplexity": perplexity["level_3"],
            },
        }
        
        return usage


def train_codebook(
    encoder: SemanticIDEncoder,
    item_features: torch.Tensor,
    num_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    save_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    便捷的训练函数
    
    Args:
        encoder: 语义 ID 编码器
        item_features: 物品特征数据 (num_items, embedding_dim)
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        save_dir: 模型保存目录
        device: 计算设备
    
    Returns:
        训练历史
    
    Example:
        >>> encoder = SemanticIDEncoder(SemanticIDConfig())
        >>> features = torch.randn(10000, 256)
        >>> history = train_codebook(encoder, features, num_epochs=10)
    """
    config = TrainingConfig(
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )
    
    trainer = SemanticIDTrainer(encoder, config, device)
    
    # 划分训练集和验证集
    num_samples = len(item_features)
    num_val = int(num_samples * 0.1)
    perm = torch.randperm(num_samples)
    
    train_features = item_features[perm[num_val:]]
    val_features = item_features[perm[:num_val]]
    
    history = trainer.train(train_features, val_features, save_dir)
    
    return history

