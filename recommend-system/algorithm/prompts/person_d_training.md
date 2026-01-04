# Person D: 训练 Pipeline

## 你的角色
你是一名深度学习算法工程师，负责实现生成式推荐系统的 **训练 Pipeline** 模块。

## 背景知识

UGT 模型采用三阶段训练策略：

1. **阶段1：基础预训练** - 大规模 Next Token Prediction
2. **阶段2：多任务微调** - 加入对比学习和多任务目标
3. **阶段3：偏好对齐** - DPO (Direct Preference Optimization)

### 统一损失函数

```
L_total = L_ntp + λ₁ * L_contrastive + λ₂ * L_preference + λ₃ * L_moe_balance

其中：
- L_ntp: Next Token Prediction 损失（交叉熵）
- L_contrastive: 用户-物品对比学习损失
- L_preference: DPO 偏好损失
- L_moe_balance: MoE 负载均衡损失（来自 Decoder）
```

## 你的任务

在 `algorithm/training/` 目录下实现完整的训练流程。

### 目录结构

```
algorithm/training/
├── __init__.py
├── config.py           # 训练配置
├── dataset.py          # 数据集类
├── loss.py             # 损失函数
├── optimizer.py        # 优化器配置
├── scheduler.py        # 学习率调度
├── trainer.py          # 训练器主类
├── checkpoint.py       # 检查点管理
├── metrics.py          # 评估指标
├── distributed.py      # 分布式训练
└── scripts/
    ├── train_stage1.py # 阶段1：预训练
    ├── train_stage2.py # 阶段2：多任务微调
    └── train_stage3.py # 阶段3：偏好对齐
```

### 接口要求

你必须实现 `interfaces.py` 中定义的 `TrainerInterface`：

```python
from algorithm.interfaces import TrainerInterface, TrainingConfig

class Trainer(TrainerInterface):
    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch，返回训练指标"""
        pass
    
    def evaluate(self) -> Dict[str, float]:
        """在验证集上评估，返回评估指标"""
        pass
    
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        pass
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        pass
```

### 核心实现

#### 1. config.py - 训练配置

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    output_dir: str = "checkpoints"
    seed: int = 42
    
    # 批次配置
    batch_size: int = 256
    gradient_accumulation_steps: int = 4
    max_epochs: int = 10
    max_steps: int = -1  # -1 表示不限制
    
    # 优化器配置
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 学习率调度
    warmup_steps: int = 10000
    lr_scheduler_type: str = "cosine"  # linear, cosine, polynomial
    
    # 混合精度
    fp16: bool = True
    bf16: bool = False
    
    # 损失权重
    lambda_contrastive: float = 0.1
    lambda_preference: float = 0.1
    lambda_moe_balance: float = 0.01
    
    # 日志和保存
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # 分布式训练
    local_rank: int = -1
    deepspeed: bool = True
    zero_stage: int = 2


@dataclass
class Stage1Config(TrainingConfig):
    """阶段1：预训练配置"""
    max_epochs: int = 5
    learning_rate: float = 1e-4
    lambda_contrastive: float = 0.0  # 阶段1不使用对比学习
    lambda_preference: float = 0.0   # 阶段1不使用偏好学习


@dataclass
class Stage2Config(TrainingConfig):
    """阶段2：多任务微调配置"""
    max_epochs: int = 3
    learning_rate: float = 5e-5  # 较小学习率
    lambda_contrastive: float = 0.1


@dataclass
class Stage3Config(TrainingConfig):
    """阶段3：偏好对齐配置"""
    max_epochs: int = 2
    learning_rate: float = 1e-5  # 更小学习率
    lambda_preference: float = 0.1
    dpo_beta: float = 0.1  # DPO 温度参数
```

#### 2. dataset.py - 数据集类

```python
from torch.utils.data import Dataset, DataLoader
import torch

class RecommendDataset(Dataset):
    """
    推荐训练数据集
    
    数据格式：
    {
        "user_id": str,
        "encoder_l1_ids": List[int],
        "encoder_l2_ids": List[int],
        "encoder_l3_ids": List[int],
        "encoder_positions": List[int],
        "encoder_token_types": List[int],
        "encoder_mask": List[int],
        "decoder_l1_ids": List[int],
        "decoder_l2_ids": List[int],
        "decoder_l3_ids": List[int],
        "decoder_positions": List[int],
        "decoder_token_types": List[int],
        "decoder_mask": List[int],
        "labels_l1": List[int],
        "labels_l2": List[int],
        "labels_l3": List[int],
    }
    """
    
    def __init__(
        self,
        data_path: str,
        max_seq_length: int = 1024,
        tokenizer = None,
    ):
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        
        # 加载数据
        self.samples = self._load_data()
    
    def _load_data(self) -> List[dict]:
        """加载数据文件"""
        # 支持 JSON, Parquet, TFRecord 等格式
        import json
        with open(self.data_path, 'r') as f:
            return [json.loads(line) for line in f]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        return {
            "encoder_semantic_ids": [
                torch.tensor(sample["encoder_l1_ids"], dtype=torch.long),
                torch.tensor(sample["encoder_l2_ids"], dtype=torch.long),
                torch.tensor(sample["encoder_l3_ids"], dtype=torch.long),
            ],
            "encoder_positions": torch.tensor(sample["encoder_positions"], dtype=torch.long),
            "encoder_token_types": torch.tensor(sample["encoder_token_types"], dtype=torch.long),
            "encoder_mask": torch.tensor(sample["encoder_mask"], dtype=torch.long),
            "decoder_semantic_ids": [
                torch.tensor(sample["decoder_l1_ids"], dtype=torch.long),
                torch.tensor(sample["decoder_l2_ids"], dtype=torch.long),
                torch.tensor(sample["decoder_l3_ids"], dtype=torch.long),
            ],
            "decoder_positions": torch.tensor(sample["decoder_positions"], dtype=torch.long),
            "decoder_token_types": torch.tensor(sample["decoder_token_types"], dtype=torch.long),
            "labels": [
                torch.tensor(sample["labels_l1"], dtype=torch.long),
                torch.tensor(sample["labels_l2"], dtype=torch.long),
                torch.tensor(sample["labels_l3"], dtype=torch.long),
            ],
        }


class PreferenceDataset(Dataset):
    """
    偏好对齐数据集（用于 DPO）
    
    数据格式：
    {
        "user_sequence": ...,      # 用户历史
        "chosen_item": ...,        # 用户选择的物品
        "rejected_item": ...,      # 用户未选择的物品
    }
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.samples = self._load_data()
    
    def _load_data(self):
        import json
        with open(self.data_path, 'r') as f:
            return [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # ... 转换为张量
        return sample
```

#### 3. loss.py - 损失函数

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NextTokenPredictionLoss(nn.Module):
    """
    下一个 Token 预测损失
    
    分别计算 L1, L2, L3 的交叉熵损失
    """
    
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(
        self,
        l1_logits: torch.Tensor,  # (batch, seq_len, 1024)
        l2_logits: torch.Tensor,  # (batch, seq_len, 4096)
        l3_logits: torch.Tensor,  # (batch, seq_len, 16384)
        labels_l1: torch.Tensor,  # (batch, seq_len)
        labels_l2: torch.Tensor,
        labels_l3: torch.Tensor,
    ) -> torch.Tensor:
        # 计算各层损失
        loss_l1 = F.cross_entropy(
            l1_logits.view(-1, l1_logits.size(-1)),
            labels_l1.view(-1),
            ignore_index=self.ignore_index,
        )
        
        loss_l2 = F.cross_entropy(
            l2_logits.view(-1, l2_logits.size(-1)),
            labels_l2.view(-1),
            ignore_index=self.ignore_index,
        )
        
        loss_l3 = F.cross_entropy(
            l3_logits.view(-1, l3_logits.size(-1)),
            labels_l3.view(-1),
            ignore_index=self.ignore_index,
        )
        
        # 层次加权（L1 最重要）
        total_loss = 0.5 * loss_l1 + 0.3 * loss_l2 + 0.2 * loss_l3
        
        return total_loss


class ContrastiveLoss(nn.Module):
    """
    用户-物品对比学习损失
    
    InfoNCE 损失：拉近正样本，推远负样本
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        user_repr: torch.Tensor,   # (batch, d_model)
        item_repr: torch.Tensor,   # (batch, d_model)
    ) -> torch.Tensor:
        # 归一化
        user_repr = F.normalize(user_repr, dim=-1)
        item_repr = F.normalize(item_repr, dim=-1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(user_repr, item_repr.T) / self.temperature
        
        # 对角线是正样本
        batch_size = user_repr.shape[0]
        labels = torch.arange(batch_size, device=user_repr.device)
        
        # InfoNCE 损失（双向）
        loss_u2i = F.cross_entropy(sim_matrix, labels)
        loss_i2u = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_u2i + loss_i2u) / 2


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization 损失
    
    让模型偏好用户选择的物品，远离未选择的物品
    
    L_DPO = -log(σ(β * (log π(y_w|x) - log π(y_l|x))))
    
    其中：
    - y_w: chosen (用户选择的)
    - y_l: rejected (用户未选择的)
    - β: 温度参数
    """
    
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta
    
    def forward(
        self,
        chosen_logps: torch.Tensor,    # (batch,) chosen 的 log probability
        rejected_logps: torch.Tensor,  # (batch,) rejected 的 log probability
    ) -> torch.Tensor:
        # 计算 log ratio
        log_ratio = chosen_logps - rejected_logps
        
        # DPO 损失
        loss = -F.logsigmoid(self.beta * log_ratio).mean()
        
        return loss


class UnifiedLoss(nn.Module):
    """
    统一训练损失
    
    L_total = L_ntp + λ₁ * L_contrastive + λ₂ * L_preference + λ₃ * L_moe_balance
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        self.ntp_loss = NextTokenPredictionLoss()
        self.contrastive_loss = ContrastiveLoss()
        self.dpo_loss = DPOLoss()
    
    def forward(
        self,
        model_outputs: dict,
        labels: dict,
        aux_loss: torch.Tensor = None,
    ) -> dict:
        losses = {}
        
        # NTP 损失
        ntp = self.ntp_loss(
            model_outputs["l1_logits"],
            model_outputs["l2_logits"],
            model_outputs["l3_logits"],
            labels["l1"],
            labels["l2"],
            labels["l3"],
        )
        losses["ntp_loss"] = ntp
        
        # 对比学习损失（如果有用户和物品表示）
        if "user_repr" in model_outputs and "item_repr" in model_outputs:
            contrastive = self.contrastive_loss(
                model_outputs["user_repr"],
                model_outputs["item_repr"],
            )
            losses["contrastive_loss"] = contrastive
        else:
            contrastive = 0
        
        # MoE 负载均衡损失
        if aux_loss is not None:
            losses["moe_balance_loss"] = aux_loss
            moe_balance = aux_loss
        else:
            moe_balance = 0
        
        # 总损失
        total = (
            ntp +
            self.config.lambda_contrastive * contrastive +
            self.config.lambda_moe_balance * moe_balance
        )
        losses["total_loss"] = total
        
        return losses
```

#### 4. trainer.py - 训练器主类

```python
import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class Trainer:
    """
    模型训练器
    
    支持：
    - 混合精度训练 (FP16/BF16)
    - 梯度累积
    - 分布式训练 (DDP / DeepSpeed)
    - 检查点保存和恢复
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # 数据加载器
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        if eval_dataset:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
            )
        
        # 优化器
        self.optimizer = optimizer or self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = scheduler or self._create_scheduler()
        
        # 损失函数
        self.loss_fn = UnifiedLoss(config)
        
        # 混合精度
        self.scaler = GradScaler() if config.fp16 else None
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        # 分离权重衰减和非权重衰减参数
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        from transformers import get_scheduler
        
        num_training_steps = len(self.train_dataloader) * self.config.max_epochs
        
        return get_scheduler(
            self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    def train(self) -> Dict[str, float]:
        """完整训练流程"""
        logger.info("Starting training...")
        logger.info(f"  Num epochs = {self.config.max_epochs}")
        logger.info(f"  Batch size = {self.config.batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        
        best_eval_loss = float("inf")
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # 训练一个 epoch
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch}: {train_metrics}")
            
            # 评估
            if self.eval_dataset:
                eval_metrics = self.evaluate()
                logger.info(f"Eval: {eval_metrics}")
                
                # 保存最佳模型
                if eval_metrics["loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["loss"]
                    self.save_checkpoint(
                        os.path.join(self.config.output_dir, "best_model")
                    )
            
            # 定期保存
            self.save_checkpoint(
                os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch}")
            )
        
        return {"best_eval_loss": best_eval_loss}
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0
        total_ntp_loss = 0
        total_contrastive_loss = 0
        total_moe_loss = 0
        num_steps = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for step, batch in enumerate(progress_bar):
            # 移动到设备
            batch = self._move_to_device(batch)
            
            # 前向传播
            with autocast(enabled=self.config.fp16):
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
                    aux_loss=outputs.get("aux_loss"),
                )
                
                loss = losses["total_loss"] / self.config.gradient_accumulation_steps
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.config.max_grad_norm > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                
                # 优化器步骤
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # 统计
            total_loss += losses["total_loss"].item()
            total_ntp_loss += losses["ntp_loss"].item()
            if "contrastive_loss" in losses:
                total_contrastive_loss += losses["contrastive_loss"].item()
            if "moe_balance_loss" in losses:
                total_moe_loss += losses["moe_balance_loss"].item()
            num_steps += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{total_loss / num_steps:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
            
            # 日志
            if self.global_step % self.config.logging_steps == 0:
                logger.info(
                    f"Step {self.global_step}: "
                    f"loss={total_loss/num_steps:.4f}, "
                    f"ntp={total_ntp_loss/num_steps:.4f}, "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                )
        
        return {
            "loss": total_loss / num_steps,
            "ntp_loss": total_ntp_loss / num_steps,
            "contrastive_loss": total_contrastive_loss / num_steps,
            "moe_balance_loss": total_moe_loss / num_steps,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }
    
    def evaluate(self) -> Dict[str, float]:
        """在验证集上评估"""
        self.model.eval()
        
        total_loss = 0
        num_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
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
                num_steps += 1
        
        return {"loss": total_loss / num_steps}
    
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "config": self.config,
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, os.path.join(path, "checkpoint.pt"))
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(os.path.join(path, "checkpoint.pt"))
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Checkpoint loaded from {path}")
    
    def _move_to_device(self, batch: dict) -> dict:
        """将 batch 移动到设备"""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, list):
                result[key] = [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in value]
            else:
                result[key] = value
        return result
```

#### 5. metrics.py - 评估指标

```python
import torch
import numpy as np
from typing import List, Tuple

def recall_at_k(
    predictions: List[List[Tuple[int, int, int]]],
    ground_truth: List[Tuple[int, int, int]],
    k: int = 10,
) -> float:
    """
    计算 Recall@K
    
    预测列表中包含正确答案的比例
    """
    hits = 0
    for preds, gt in zip(predictions, ground_truth):
        top_k = preds[:k]
        if gt in top_k:
            hits += 1
    return hits / len(predictions)


def ndcg_at_k(
    predictions: List[List[Tuple[int, int, int]]],
    ground_truth: List[Tuple[int, int, int]],
    k: int = 10,
) -> float:
    """
    计算 NDCG@K
    
    考虑正确答案在预测列表中的位置
    """
    dcg_sum = 0
    for preds, gt in zip(predictions, ground_truth):
        top_k = preds[:k]
        for i, pred in enumerate(top_k):
            if pred == gt:
                dcg_sum += 1 / np.log2(i + 2)
                break
    
    # IDCG = 1 (因为只有一个正确答案)
    idcg = 1.0
    return dcg_sum / (len(predictions) * idcg)


def mrr(
    predictions: List[List[Tuple[int, int, int]]],
    ground_truth: List[Tuple[int, int, int]],
) -> float:
    """
    计算 Mean Reciprocal Rank
    """
    rr_sum = 0
    for preds, gt in zip(predictions, ground_truth):
        for i, pred in enumerate(preds):
            if pred == gt:
                rr_sum += 1 / (i + 1)
                break
    return rr_sum / len(predictions)
```

### 测试用例

```python
def test_trainer():
    from algorithm.training.config import TrainingConfig
    from algorithm.training.dataset import RecommendDataset
    from algorithm.training.trainer import Trainer
    
    # 模拟模型
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 512)
        
        def forward(self, **kwargs):
            batch_size = kwargs["encoder_semantic_ids"][0].shape[0]
            seq_len = kwargs["decoder_semantic_ids"][0].shape[1]
            return {
                "l1_logits": torch.randn(batch_size, seq_len, 1024),
                "l2_logits": torch.randn(batch_size, seq_len, 4096),
                "l3_logits": torch.randn(batch_size, seq_len, 16384),
                "aux_loss": torch.tensor(0.1),
            }
    
    config = TrainingConfig(
        batch_size=4,
        max_epochs=1,
        logging_steps=1,
    )
    
    # 创建模拟数据集
    # ...
    
    trainer = Trainer(
        model=MockModel(),
        config=config,
        train_dataset=train_dataset,
    )
    
    # 测试训练
    metrics = trainer.train_epoch()
    assert "loss" in metrics
    
    # 测试保存/加载
    trainer.save_checkpoint("/tmp/test_checkpoint")
    trainer.load_checkpoint("/tmp/test_checkpoint")
    
    print("All trainer tests passed!")
```

## 注意事项

1. **梯度累积**: 当 GPU 内存不足时，使用梯度累积模拟大 batch
2. **混合精度**: FP16 可以加速训练并减少内存，但要注意数值稳定性
3. **学习率调度**: Warmup 对大模型训练很重要
4. **检查点**: 定期保存检查点，支持断点续训

## 输出要求

请输出完整的可运行代码，包含：
1. 所有 Python 文件
2. 详细的中文注释
3. 三阶段训练脚本
4. 使用示例

确保代码遵循 `algorithm/interfaces.py` 中定义的接口。

