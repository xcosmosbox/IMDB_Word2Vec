"""
训练器模块

实现完整的模型训练流程，遵循 interfaces.py 中定义的 TrainerInterface：
- 混合精度训练 (FP16/BF16)
- 梯度累积
- 分布式训练支持 (DDP / DeepSpeed)
- 检查点保存和恢复
- 评估和指标计算

对应架构文档: 第八章 训练与部署流水线
"""

import os
import time
import logging
from typing import Dict, Optional, List, Tuple, Any, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

from .config import TrainingConfig, Stage1Config, Stage2Config, Stage3Config
from .loss import UnifiedLoss, compute_log_probs
from .optimizer import create_optimizer
from .scheduler import create_scheduler
from .checkpoint import CheckpointManager
from .metrics import MetricsCalculator


# 设置日志
logger = logging.getLogger(__name__)


class Trainer:
    """
    模型训练器
    
    遵循 interfaces.py 中的 TrainerInterface 接口
    
    支持：
    - 混合精度训练 (FP16/BF16)
    - 梯度累积
    - 分布式训练 (DDP / DeepSpeed)
    - 检查点保存和恢复
    - 三阶段训练策略
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        reference_model: Optional[nn.Module] = None,
    ):
        """
        初始化训练器
        
        Args:
            model: 待训练的模型
            config: 训练配置
            train_dataset: 训练数据集
            eval_dataset: 验证数据集（可选）
            optimizer: 优化器（可选，不提供则自动创建）
            scheduler: 学习率调度器（可选，不提供则自动创建）
            reference_model: 参考模型（用于 DPO，可选）
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.reference_model = reference_model
        
        # 设置设备
        self.device = self._setup_device()
        self.model = self.model.to(self.device)
        if self.reference_model is not None:
            self.reference_model = self.reference_model.to(self.device)
            self.reference_model.eval()
        
        # 创建数据加载器
        self.train_dataloader = self._create_dataloader(train_dataset, shuffle=True)
        if eval_dataset is not None:
            self.eval_dataloader = self._create_dataloader(eval_dataset, shuffle=False)
        else:
            self.eval_dataloader = None
        
        # 计算总训练步数
        self.total_steps = self._compute_total_steps()
        
        # 创建优化器
        self.optimizer = optimizer or self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = scheduler or self._create_scheduler()
        
        # 创建损失函数
        self.loss_fn = self._create_loss_function()
        
        # 混合精度
        self.scaler = self._setup_mixed_precision()
        
        # 检查点管理
        self.checkpoint_manager = CheckpointManager(
            save_dir=config.output_dir,
            max_checkpoints=config.save_total_limit,
        )
        
        # 指标计算器
        self.metrics_calculator = MetricsCalculator()
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_metric = float('inf')
        
        # TensorBoard
        self.writer = self._setup_tensorboard()
    
    def _setup_device(self) -> torch.device:
        """设置训练设备"""
        if self.config.local_rank >= 0:
            # 分布式训练
            device = torch.device(f"cuda:{self.config.local_rank}")
            torch.cuda.set_device(device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logger.warning("未检测到 GPU，将使用 CPU 训练")
        
        return device
    
    def _create_dataloader(
        self, 
        dataset: Dataset, 
        shuffle: bool = True
    ) -> DataLoader:
        """创建数据加载器"""
        from .dataset import DataCollator
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            drop_last=self.config.dataloader_drop_last,
            collate_fn=DataCollator(),
        )
    
    def _compute_total_steps(self) -> int:
        """计算总训练步数"""
        steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.max_epochs
        
        if self.config.max_steps > 0:
            total_steps = min(total_steps, self.config.max_steps)
        
        return total_steps
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        return create_optimizer(
            model=self.model,
            optimizer_type=self.config.optimizer_type.value,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            adam_epsilon=self.config.adam_epsilon,
        )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        return create_scheduler(
            optimizer=self.optimizer,
            scheduler_type=self.config.lr_scheduler_type.value,
            total_steps=self.total_steps,
            warmup_steps=self.config.warmup_steps,
            warmup_ratio=self.config.warmup_ratio,
            min_lr_ratio=self.config.min_lr_ratio,
        )
    
    def _create_loss_function(self) -> UnifiedLoss:
        """创建损失函数"""
        return UnifiedLoss(
            l1_weight=self.config.l1_loss_weight,
            l2_weight=self.config.l2_loss_weight,
            l3_weight=self.config.l3_loss_weight,
            label_smoothing=self.config.label_smoothing,
            lambda_contrastive=self.config.lambda_contrastive,
            lambda_preference=self.config.lambda_preference,
            lambda_moe_balance=self.config.lambda_moe_balance,
        )
    
    def _setup_mixed_precision(self) -> Optional[GradScaler]:
        """设置混合精度训练"""
        if self.config.fp16:
            return GradScaler()
        return None
    
    def _setup_tensorboard(self):
        """设置 TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(self.config.output_dir, self.config.logging_dir)
            os.makedirs(log_dir, exist_ok=True)
            return SummaryWriter(log_dir=log_dir)
        except ImportError:
            logger.warning("TensorBoard 未安装，跳过日志记录")
            return None
    
    def train(self) -> Dict[str, float]:
        """
        完整训练流程
        
        Returns:
            最终训练指标
        """
        logger.info("=" * 60)
        logger.info("开始训练")
        logger.info(f"  实验名称: {self.config.experiment_name}")
        logger.info(f"  总轮数: {self.config.max_epochs}")
        logger.info(f"  批次大小: {self.config.batch_size}")
        logger.info(f"  梯度累积步数: {self.config.gradient_accumulation_steps}")
        logger.info(f"  有效批次大小: {self.config.effective_batch_size}")
        logger.info(f"  总步数: {self.total_steps}")
        logger.info(f"  预热步数: {self.config.warmup_steps}")
        logger.info(f"  学习率: {self.config.learning_rate}")
        logger.info("=" * 60)
        
        # 从检查点恢复
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        
        # 训练前评估
        if self.config.eval_on_start and self.eval_dataloader is not None:
            eval_metrics = self.evaluate()
            logger.info(f"初始评估: {eval_metrics}")
        
        # 训练循环
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            
            # 训练一个 epoch
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch} 训练指标: {train_metrics}")
            
            # 验证
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                logger.info(f"Epoch {epoch} 验证指标: {eval_metrics}")
                
                # 保存最佳模型
                if eval_metrics["loss"] < self.best_eval_metric:
                    self.best_eval_metric = eval_metrics["loss"]
                    self.save_checkpoint(
                        os.path.join(self.config.output_dir, "best_model"),
                        is_best=True,
                    )
            
            # 保存检查点
            self.save_checkpoint(
                os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch}")
            )
            
            # 检查是否达到最大步数
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                logger.info(f"达到最大步数 {self.config.max_steps}，停止训练")
                break
        
        # 保存最终模型
        self.save_checkpoint(
            os.path.join(self.config.output_dir, "final_model")
        )
        
        logger.info("训练完成！")
        
        return {"best_eval_loss": self.best_eval_metric}
    
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Returns:
            训练指标字典
        """
        self.model.train()
        
        # 累积指标
        total_loss = 0.0
        total_ntp_loss = 0.0
        total_contrastive_loss = 0.0
        total_moe_loss = 0.0
        num_steps = 0
        
        # 进度条
        try:
            from tqdm import tqdm
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {self.current_epoch}",
                disable=self.config.local_rank > 0,
            )
        except ImportError:
            progress_bar = self.train_dataloader
        
        for step, batch in enumerate(progress_bar):
            # 移动到设备
            batch = self._move_to_device(batch)
            
            # 前向传播
            with autocast(enabled=self.config.fp16 or self.config.bf16):
                outputs = self.model(
                    encoder_semantic_ids=batch["encoder_semantic_ids"],
                    encoder_positions=batch["encoder_positions"],
                    encoder_token_types=batch["encoder_token_types"],
                    encoder_attention_mask=batch["encoder_mask"],
                    decoder_semantic_ids=batch["decoder_semantic_ids"],
                    decoder_positions=batch["decoder_positions"],
                    decoder_token_types=batch["decoder_token_types"],
                )
                
                # 计算损失
                losses = self.loss_fn(
                    model_outputs=outputs,
                    labels={
                        "l1": batch["labels"][0],
                        "l2": batch["labels"][1],
                        "l3": batch["labels"][2],
                    },
                    aux_loss=outputs.get("aux_loss"),
                )
                
                loss = losses["total_loss"] / self.config.gradient_accumulation_steps
            
            # 反向传播
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.config.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                
                # 优化器步骤
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # 日志记录
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics(losses)
                
                # 保存检查点
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(
                        os.path.join(
                            self.config.output_dir, 
                            f"checkpoint-step-{self.global_step}"
                        )
                    )
                
                # 评估
                if (
                    self.config.eval_steps > 0 and 
                    self.global_step % self.config.eval_steps == 0 and
                    self.eval_dataloader is not None
                ):
                    eval_metrics = self.evaluate()
                    logger.info(f"Step {self.global_step} 验证指标: {eval_metrics}")
                    self.model.train()
            
            # 累积指标
            total_loss += losses["total_loss"].item()
            total_ntp_loss += losses["ntp_loss"].item()
            if "contrastive_loss" in losses and isinstance(losses["contrastive_loss"], torch.Tensor):
                total_contrastive_loss += losses["contrastive_loss"].item()
            if "moe_balance_loss" in losses and isinstance(losses["moe_balance_loss"], torch.Tensor):
                total_moe_loss += losses["moe_balance_loss"].item()
            num_steps += 1
            
            # 更新进度条
            if hasattr(progress_bar, 'set_postfix'):
                progress_bar.set_postfix({
                    "loss": f"{total_loss / num_steps:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                })
            
            # 检查最大步数
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break
        
        return {
            "loss": total_loss / num_steps,
            "ntp_loss": total_ntp_loss / num_steps,
            "contrastive_loss": total_contrastive_loss / num_steps,
            "moe_balance_loss": total_moe_loss / num_steps,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        在验证集上评估
        
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        total_loss = 0.0
        total_ntp_loss = 0.0
        all_predictions = []
        all_labels = []
        num_steps = 0
        
        with torch.no_grad():
            try:
                from tqdm import tqdm
                progress_bar = tqdm(
                    self.eval_dataloader, 
                    desc="Evaluating",
                    disable=self.config.local_rank > 0,
                )
            except ImportError:
                progress_bar = self.eval_dataloader
            
            for batch in progress_bar:
                batch = self._move_to_device(batch)
                
                outputs = self.model(
                    encoder_semantic_ids=batch["encoder_semantic_ids"],
                    encoder_positions=batch["encoder_positions"],
                    encoder_token_types=batch["encoder_token_types"],
                    encoder_attention_mask=batch["encoder_mask"],
                    decoder_semantic_ids=batch["decoder_semantic_ids"],
                    decoder_positions=batch["decoder_positions"],
                    decoder_token_types=batch["decoder_token_types"],
                )
                
                losses = self.loss_fn(
                    model_outputs=outputs,
                    labels={
                        "l1": batch["labels"][0],
                        "l2": batch["labels"][1],
                        "l3": batch["labels"][2],
                    },
                )
                
                total_loss += losses["total_loss"].item()
                total_ntp_loss += losses["ntp_loss"].item()
                num_steps += 1
                
                # 收集预测结果用于计算 Recall 等指标
                predictions = self._get_predictions(outputs)
                labels = self._get_labels(batch)
                all_predictions.extend(predictions)
                all_labels.extend(labels)
        
        # 计算指标
        metrics = {
            "loss": total_loss / num_steps,
            "ntp_loss": total_ntp_loss / num_steps,
        }
        
        # 计算推荐指标
        if all_predictions and all_labels:
            recall_10 = self.metrics_calculator.recall_at_k(
                all_predictions, all_labels, k=10
            )
            recall_50 = self.metrics_calculator.recall_at_k(
                all_predictions, all_labels, k=50
            )
            ndcg_10 = self.metrics_calculator.ndcg_at_k(
                all_predictions, all_labels, k=10
            )
            
            metrics.update({
                "recall@10": recall_10,
                "recall@50": recall_50,
                "ndcg@10": ndcg_10,
            })
        
        return metrics
    
    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        """
        保存检查点
        
        Args:
            path: 保存路径
            is_best: 是否是最佳模型
        """
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_eval_metric": self.best_eval_metric,
            "config": self.config.to_dict(),
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # 使用检查点管理器保存
        self.checkpoint_manager.save(
            checkpoint=checkpoint,
            path=path,
            step=self.global_step,
            is_best=is_best,
        )
        
        logger.info(f"检查点已保存至 {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        加载检查点
        
        Args:
            path: 检查点路径
        """
        checkpoint = self.checkpoint_manager.load(path)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.best_eval_metric = checkpoint.get("best_eval_metric", float('inf'))
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"检查点已从 {path} 加载")
        logger.info(f"  从 Epoch {self.current_epoch}, Step {self.global_step} 继续训练")
    
    def _move_to_device(self, batch: Dict) -> Dict:
        """将批次数据移动到设备"""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, list):
                result[key] = [
                    v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for v in value
                ]
            else:
                result[key] = value
        return result
    
    def _get_predictions(self, outputs: Dict) -> List[List[Tuple[int, int, int]]]:
        """从模型输出中获取预测结果"""
        predictions = []
        
        l1_logits = outputs["l1_logits"]  # (batch, seq_len, 1024)
        l2_logits = outputs["l2_logits"]  # (batch, seq_len, 4096)
        l3_logits = outputs["l3_logits"]  # (batch, seq_len, 16384)
        
        batch_size = l1_logits.size(0)
        
        for i in range(batch_size):
            # 取最后一个位置的预测
            l1_pred = l1_logits[i, -1].topk(50).indices.tolist()
            l2_pred = l2_logits[i, -1].topk(50).indices.tolist()
            l3_pred = l3_logits[i, -1].topk(50).indices.tolist()
            
            # 组合成语义 ID
            sample_preds = [
                (l1_pred[j], l2_pred[j], l3_pred[j])
                for j in range(min(50, len(l1_pred)))
            ]
            predictions.append(sample_preds)
        
        return predictions
    
    def _get_labels(self, batch: Dict) -> List[Tuple[int, int, int]]:
        """从批次中获取标签"""
        labels = []
        
        labels_l1 = batch["labels"][0]  # (batch, seq_len)
        labels_l2 = batch["labels"][1]
        labels_l3 = batch["labels"][2]
        
        batch_size = labels_l1.size(0)
        
        for i in range(batch_size):
            # 找到第一个有效标签（非 -100）
            mask = labels_l1[i] != -100
            if mask.any():
                idx = mask.nonzero()[0].item()
                labels.append((
                    labels_l1[i, idx].item(),
                    labels_l2[i, idx].item(),
                    labels_l3[i, idx].item(),
                ))
            else:
                labels.append((0, 0, 0))
        
        return labels
    
    def _log_metrics(self, losses: Dict[str, torch.Tensor]) -> None:
        """记录指标到 TensorBoard"""
        if self.writer is None:
            return
        
        for name, value in losses.items():
            if isinstance(value, torch.Tensor):
                self.writer.add_scalar(f"train/{name}", value.item(), self.global_step)
            else:
                self.writer.add_scalar(f"train/{name}", value, self.global_step)
        
        self.writer.add_scalar(
            "train/learning_rate",
            self.scheduler.get_last_lr()[0],
            self.global_step,
        )
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()

