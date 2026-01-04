# 训练 Pipeline 模块 (Person D)

> 生成式推荐系统 - 训练模块完整文档  
> 版本: 1.0.0  
> 作者: Person D  
> 更新日期: 2026-01-04

---

## 📑 目录

1. [概述](#1-概述)
2. [快速开始](#2-快速开始)
3. [模块架构](#3-模块架构)
4. [配置详解](#4-配置详解)
5. [数据集](#5-数据集)
6. [损失函数](#6-损失函数)
7. [训练器](#7-训练器)
8. [三阶段训练](#8-三阶段训练)
9. [分布式训练](#9-分布式训练)
10. [评估指标](#10-评估指标)
11. [API 参考](#11-api-参考)
12. [常见问题](#12-常见问题)

---

## 1. 概述

### 1.1 模块职责

本模块实现生成式推荐系统（UGT）的完整训练流程，包括：

- **三阶段训练策略**：预训练 → 多任务微调 → 偏好对齐
- **统一损失函数**：NTP + 对比学习 + DPO + MoE平衡
- **分布式训练**：支持 DDP 和 DeepSpeed
- **训练工程**：混合精度、梯度累积、检查点管理

### 1.2 技术架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         训练 Pipeline 架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        三阶段训练策略                                  │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │   │
│  │  │   Stage 1     │→ │   Stage 2     │→ │   Stage 3     │            │   │
│  │  │  基础预训练    │  │  多任务微调    │  │  偏好对齐     │            │   │
│  │  │  (NTP Only)   │  │  (NTP + CL)   │  │ (NTP+CL+DPO) │            │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────┴───────────────────────────────────┐   │
│  │                        统一损失函数                                   │   │
│  │  L_total = L_ntp + λ₁·L_contrastive + λ₂·L_preference + λ₃·L_moe   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │ Dataset  │ │ Optimizer│ │ Scheduler│ │Checkpoint│ │ Metrics  │        │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 接口遵循

本模块遵循 `algorithm/interfaces.py` 中定义的 `TrainerInterface`：

```python
class TrainerInterface(ABC):
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch，返回训练指标"""
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """在验证集上评估，返回评估指标"""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        pass
```

---

## 2. 快速开始

### 2.1 安装依赖

```bash
pip install torch>=2.0.0
pip install numpy tqdm tensorboard
pip install deepspeed  # 可选，用于大规模分布式训练
```

### 2.2 最简示例

```python
from algorithm.training import (
    TrainingConfig,
    Trainer,
    RecommendDataset,
)

# 1. 准备配置
config = TrainingConfig(
    batch_size=256,
    max_epochs=5,
    learning_rate=1e-4,
    output_dir="checkpoints",
)

# 2. 加载数据
train_dataset = RecommendDataset("data/train.jsonl")
eval_dataset = RecommendDataset("data/eval.jsonl")

# 3. 创建模型（需从 encoder/decoder 模块导入）
model = create_ugt_model()

# 4. 训练
trainer = Trainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

result = trainer.train()
print(f"训练完成！最佳验证损失: {result['best_eval_loss']:.4f}")
```

### 2.3 使用训练脚本

```bash
# 阶段 1: 预训练
python -m algorithm.training.scripts.train_stage1 \
    --train_data data/pretrain/train.jsonl \
    --eval_data data/pretrain/eval.jsonl \
    --output_dir checkpoints/stage1 \
    --batch_size 256 \
    --max_epochs 5

# 阶段 2: 多任务微调
python -m algorithm.training.scripts.train_stage2 \
    --pretrained checkpoints/stage1/best_model \
    --train_data data/multitask/train.jsonl \
    --output_dir checkpoints/stage2 \
    --lambda_contrastive 0.1

# 阶段 3: 偏好对齐
python -m algorithm.training.scripts.train_stage3 \
    --pretrained checkpoints/stage2/best_model \
    --train_data data/preference/train.jsonl \
    --output_dir checkpoints/stage3 \
    --dpo_beta 0.1
```

---

## 3. 模块架构

### 3.1 文件结构

```
training/
├── __init__.py              # 模块导出，公开 API
├── config.py                # 配置类定义
│   ├── TrainingConfig       # 基础配置
│   ├── Stage1Config         # 阶段1配置
│   ├── Stage2Config         # 阶段2配置
│   └── Stage3Config         # 阶段3配置
├── dataset.py               # 数据集实现
│   ├── RecommendDataset     # 推荐训练数据集
│   ├── PreferenceDataset    # 偏好对齐数据集
│   ├── StreamingDataset     # 流式数据集（大规模）
│   └── DataCollator         # 批次整理器
├── loss.py                  # 损失函数
│   ├── NextTokenPredictionLoss  # NTP 损失
│   ├── ContrastiveLoss      # 对比学习损失
│   ├── DPOLoss              # DPO 偏好损失
│   └── UnifiedLoss          # 统一损失
├── optimizer.py             # 优化器
│   ├── AdamW                # 带权重衰减的 Adam
│   ├── LAMB                 # 大批量优化器
│   └── create_optimizer()   # 优化器工厂函数
├── scheduler.py             # 学习率调度
│   ├── LinearLR             # 线性衰减
│   ├── CosineLR             # 余弦退火
│   ├── PolynomialLR         # 多项式衰减
│   └── create_scheduler()   # 调度器工厂函数
├── trainer.py               # 训练器主类
│   └── Trainer              # 核心训练逻辑
├── checkpoint.py            # 检查点管理
│   └── CheckpointManager    # 检查点保存/加载/清理
├── metrics.py               # 评估指标
│   ├── recall_at_k()        # Recall@K
│   ├── ndcg_at_k()          # NDCG@K
│   ├── mrr()                # MRR
│   └── MetricsCalculator    # 指标计算器
├── distributed.py           # 分布式训练
│   ├── setup_distributed()  # 初始化分布式
│   ├── DistributedTrainer   # DDP 训练器
│   └── DeepSpeedTrainer     # DeepSpeed 训练器
├── scripts/                 # 训练脚本
│   ├── train_stage1.py      # 阶段1脚本
│   ├── train_stage2.py      # 阶段2脚本
│   └── train_stage3.py      # 阶段3脚本
└── tests/                   # 单元测试
    └── test_training.py     # 完整测试用例
```

### 3.2 模块依赖关系

```
config.py ─────────────────────────────────────────────┐
    │                                                   │
    ▼                                                   │
dataset.py                                              │
    │                                                   │
    ▼                                                   │
loss.py ◄──────────────────────────────────────────────┤
    │                                                   │
    ▼                                                   │
optimizer.py ──────► scheduler.py                      │
    │                    │                              │
    └──────────┬─────────┘                              │
               ▼                                        │
          trainer.py ◄──────────────────────────────────┤
               │                                        │
    ┌──────────┼──────────┐                             │
    ▼          ▼          ▼                             │
checkpoint.py metrics.py distributed.py ◄──────────────┘
```

---

## 4. 配置详解

### 4.1 基础配置 (TrainingConfig)

```python
from algorithm.training import TrainingConfig

config = TrainingConfig(
    # ===== 基础配置 =====
    output_dir="checkpoints",       # 输出目录
    experiment_name="ugt_training", # 实验名称
    seed=42,                        # 随机种子
    
    # ===== 批次配置 =====
    batch_size=256,                 # 每 GPU 批次大小
    gradient_accumulation_steps=4,  # 梯度累积步数
    max_epochs=10,                  # 最大训练轮数
    max_steps=-1,                   # 最大步数 (-1 表示不限制)
    
    # ===== 序列配置 =====
    max_seq_length=1024,            # 最大序列长度
    encoder_max_length=512,         # 编码器最大长度
    decoder_max_length=128,         # 解码器最大长度
    
    # ===== 优化器配置 =====
    learning_rate=1e-4,             # 学习率
    weight_decay=0.01,              # 权重衰减
    adam_beta1=0.9,                 # Adam β1
    adam_beta2=0.999,               # Adam β2
    adam_epsilon=1e-8,              # Adam ε
    max_grad_norm=1.0,              # 梯度裁剪阈值
    
    # ===== 学习率调度 =====
    lr_scheduler_type="cosine",     # 调度类型
    warmup_steps=10000,             # 预热步数
    min_lr_ratio=0.1,               # 最小学习率比例
    
    # ===== 混合精度 =====
    fp16=True,                      # FP16 混合精度
    bf16=False,                     # BF16 混合精度
    
    # ===== 损失权重 =====
    lambda_contrastive=0.1,         # 对比学习权重 λ₁
    lambda_preference=0.1,          # 偏好损失权重 λ₂
    lambda_moe_balance=0.01,        # MoE 平衡权重 λ₃
    
    # ===== 层次化损失权重 =====
    l1_loss_weight=0.5,             # L1 层权重
    l2_loss_weight=0.3,             # L2 层权重
    l3_loss_weight=0.2,             # L3 层权重
    
    # ===== 日志和保存 =====
    logging_steps=100,              # 日志间隔
    save_steps=1000,                # 保存间隔
    eval_steps=500,                 # 评估间隔
    save_total_limit=3,             # 最多保留检查点数
    
    # ===== 分布式训练 =====
    ddp=False,                      # 是否使用 DDP
    deepspeed=False,                # 是否使用 DeepSpeed
    zero_stage=2,                   # ZeRO 阶段
)
```

### 4.2 阶段特定配置

#### Stage 1: 预训练配置

```python
from algorithm.training import Stage1Config

config = Stage1Config(
    max_epochs=5,
    learning_rate=1e-4,
    batch_size=512,
    
    # 阶段1特有：不使用对比学习和偏好学习
    lambda_contrastive=0.0,
    lambda_preference=0.0,
)
```

#### Stage 2: 多任务微调配置

```python
from algorithm.training import Stage2Config

config = Stage2Config(
    max_epochs=3,
    learning_rate=5e-5,  # 较小学习率
    
    # 阶段2特有：加入对比学习
    lambda_contrastive=0.1,
    contrastive_temperature=0.07,
    num_negatives=127,
    
    # 预训练模型路径
    pretrained_model_path="checkpoints/stage1/best_model",
)
```

#### Stage 3: 偏好对齐配置

```python
from algorithm.training import Stage3Config

config = Stage3Config(
    max_epochs=2,
    learning_rate=1e-5,  # 更小学习率
    
    # 阶段3特有：DPO 参数
    lambda_preference=0.1,
    dpo_beta=0.1,
    dpo_reference_free=False,
    
    # 模型路径
    pretrained_model_path="checkpoints/stage2/best_model",
    reference_model_path="checkpoints/stage2/best_model",
)
```

### 4.3 配置文件 (YAML)

支持从 YAML 文件加载配置：

```yaml
# configs/stage1.yaml
experiment_name: ugt_stage1_pretrain
batch_size: 256
max_epochs: 5
learning_rate: 1.0e-4
warmup_steps: 10000

fp16: true
lambda_contrastive: 0.0
lambda_preference: 0.0

output_dir: checkpoints/stage1
logging_steps: 100
save_steps: 1000
```

---

## 5. 数据集

### 5.1 数据格式

#### 推荐训练数据 (JSON Lines)

```json
{
    "user_id": "user_123",
    "encoder_l1_ids": [1, 2, 3, 4, 5],
    "encoder_l2_ids": [10, 20, 30, 40, 50],
    "encoder_l3_ids": [100, 200, 300, 400, 500],
    "encoder_positions": [0, 1, 2, 3, 4],
    "encoder_token_types": [0, 1, 1, 1, 1],
    "encoder_mask": [1, 1, 1, 1, 1],
    "decoder_l1_ids": [6, 7],
    "decoder_l2_ids": [60, 70],
    "decoder_l3_ids": [600, 700],
    "decoder_positions": [0, 1],
    "decoder_token_types": [1, 1],
    "decoder_mask": [1, 1],
    "labels_l1": [7, 8],
    "labels_l2": [70, 80],
    "labels_l3": [700, 800]
}
```

#### 偏好对齐数据 (DPO)

```json
{
    "user_id": "user_123",
    "user_sequence": {
        "encoder_l1_ids": [1, 2, 3],
        "encoder_l2_ids": [10, 20, 30],
        "encoder_l3_ids": [100, 200, 300],
        "encoder_positions": [0, 1, 2],
        "encoder_token_types": [0, 1, 1],
        "encoder_mask": [1, 1, 1]
    },
    "chosen_item": {
        "l1_id": 5,
        "l2_id": 50,
        "l3_id": 500
    },
    "rejected_item": {
        "l1_id": 6,
        "l2_id": 60,
        "l3_id": 600
    },
    "preference_score": 0.8
}
```

### 5.2 数据集使用

```python
from algorithm.training import RecommendDataset, PreferenceDataset, DataCollator

# 推荐数据集
train_dataset = RecommendDataset(
    data_path="data/train.jsonl",
    max_encoder_length=512,
    max_decoder_length=128,
    pad_token_id=0,
    lazy_loading=False,  # 小数据集立即加载
)

# 偏好数据集（用于 Stage 3）
preference_dataset = PreferenceDataset(
    data_path="data/preference.jsonl",
    max_encoder_length=512,
)

# 数据整理器
collator = DataCollator(
    pad_token_id=0,
    dynamic_padding=True,
)

# 创建 DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collator,
)
```

### 5.3 流式数据集（大规模训练）

```python
from algorithm.training.dataset import StreamingDataset

# 用于超大规模数据
dataset = StreamingDataset(
    data_paths=["data/shard_0.jsonl", "data/shard_1.jsonl", ...],
    shuffle=True,
    world_size=8,  # 分布式训练
    rank=0,
)
```

---

## 6. 损失函数

### 6.1 统一损失公式

```
L_total = L_ntp + λ₁ * L_contrastive + λ₂ * L_preference + λ₃ * L_moe_balance

其中：
- L_ntp = 0.5 * CE(L1) + 0.3 * CE(L2) + 0.2 * CE(L3)
- L_contrastive = InfoNCE(user_repr, item_repr)
- L_preference = DPO(chosen, rejected)
- L_moe_balance = 专家负载均衡损失
```

### 6.2 各损失函数详解

#### Next Token Prediction (NTP)

```python
from algorithm.training import NextTokenPredictionLoss

ntp_loss = NextTokenPredictionLoss(
    l1_weight=0.5,       # L1 层权重（粗粒度，最重要）
    l2_weight=0.3,       # L2 层权重
    l3_weight=0.2,       # L3 层权重（细粒度）
    ignore_index=-100,   # 忽略 padding
    label_smoothing=0.0, # 标签平滑
)

loss, metrics = ntp_loss(
    l1_logits, l2_logits, l3_logits,
    labels_l1, labels_l2, labels_l3,
)
# metrics: {"ntp_loss", "ntp_l1_loss", "ntp_l2_loss", "ntp_l3_loss", 
#           "ntp_l1_acc", "ntp_l2_acc", "ntp_l3_acc"}
```

#### 对比学习损失 (InfoNCE)

```python
from algorithm.training import ContrastiveLoss

contrastive_loss = ContrastiveLoss(
    temperature=0.07,  # 温度参数
    normalize=True,    # L2 归一化
)

loss, metrics = contrastive_loss(user_repr, item_repr)
# metrics: {"contrastive_loss", "contrastive_u2i_acc", "contrastive_i2u_acc"}
```

#### DPO 偏好损失

```python
from algorithm.training import DPOLoss

dpo_loss = DPOLoss(
    beta=0.1,              # 温度参数
    reference_free=False,  # 是否使用参考模型
)

loss, metrics = dpo_loss(
    chosen_logps,
    rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
)
# metrics: {"dpo_loss", "dpo_accuracy", "dpo_reward_margin"}
```

#### 统一损失

```python
from algorithm.training import UnifiedLoss

loss_fn = UnifiedLoss(
    l1_weight=0.5,
    l2_weight=0.3,
    l3_weight=0.2,
    lambda_contrastive=0.1,
    lambda_preference=0.1,
    lambda_moe_balance=0.01,
)

losses = loss_fn(
    model_outputs={"l1_logits": ..., "l2_logits": ..., "l3_logits": ...},
    labels={"l1": ..., "l2": ..., "l3": ...},
    aux_loss=moe_balance_loss,
)
# losses: {"total_loss", "ntp_loss", "contrastive_loss", "dpo_loss", ...}
```

---

## 7. 训练器

### 7.1 基础训练器

```python
from algorithm.training import Trainer, TrainingConfig

trainer = Trainer(
    model=model,                    # PyTorch 模型
    config=config,                  # 训练配置
    train_dataset=train_dataset,    # 训练数据集
    eval_dataset=eval_dataset,      # 验证数据集（可选）
    optimizer=None,                 # 自定义优化器（可选）
    scheduler=None,                 # 自定义调度器（可选）
    reference_model=None,           # 参考模型，用于 DPO（可选）
)

# 完整训练
result = trainer.train()

# 单个 epoch 训练
metrics = trainer.train_epoch()

# 评估
eval_metrics = trainer.evaluate()

# 保存/加载检查点
trainer.save_checkpoint("checkpoints/step_1000")
trainer.load_checkpoint("checkpoints/step_1000")
```

### 7.2 训练流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           训练流程                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  for epoch in range(max_epochs):                                            │
│      │                                                                      │
│      ├──► for batch in dataloader:                                          │
│      │        │                                                             │
│      │        ├──► 1. 前向传播 (with autocast if fp16)                       │
│      │        │        outputs = model(**batch)                             │
│      │        │                                                             │
│      │        ├──► 2. 计算损失                                               │
│      │        │        losses = loss_fn(outputs, labels)                    │
│      │        │                                                             │
│      │        ├──► 3. 反向传播                                               │
│      │        │        loss.backward()                                      │
│      │        │                                                             │
│      │        ├──► 4. 梯度累积 (if step % accumulation == 0)                 │
│      │        │        ├── 梯度裁剪                                          │
│      │        │        ├── optimizer.step()                                 │
│      │        │        ├── scheduler.step()                                 │
│      │        │        └── optimizer.zero_grad()                            │
│      │        │                                                             │
│      │        ├──► 5. 日志记录 (if step % logging_steps == 0)                │
│      │        │                                                             │
│      │        ├──► 6. 保存检查点 (if step % save_steps == 0)                 │
│      │        │                                                             │
│      │        └──► 7. 评估 (if step % eval_steps == 0)                       │
│      │                                                                      │
│      └──► 保存 epoch 检查点                                                  │
│                                                                             │
│  保存最终模型                                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 训练输出示例

```
================================================================================
开始训练
  实验名称: ugt_stage1_pretrain
  总轮数: 5
  批次大小: 256
  梯度累积步数: 4
  有效批次大小: 1024
  总步数: 50000
  预热步数: 10000
  学习率: 0.0001
================================================================================

Epoch 0: 100%|██████████| 10000/10000 [2:30:00<00:00, loss=3.2145, lr=1.00e-04]
Epoch 0 训练指标: {'loss': 3.2145, 'ntp_loss': 3.1823, 'learning_rate': 0.0001}
Epoch 0 验证指标: {'loss': 2.8934, 'recall@10': 0.1523, 'ndcg@10': 0.0892}

检查点已保存至 checkpoints/stage1/checkpoint-epoch-0

...

训练完成！
```

---

## 8. 三阶段训练

### 8.1 训练策略概述

| 阶段 | 目标 | 损失函数 | 学习率 | 典型轮数 |
|------|------|----------|--------|----------|
| **Stage 1** | 基础序列建模 | L_ntp | 1e-4 | 5 |
| **Stage 2** | 用户-物品表示对齐 | L_ntp + λ₁·L_cl | 5e-5 | 3 |
| **Stage 3** | 偏好对齐 | L_ntp + λ₁·L_cl + λ₂·L_dpo | 1e-5 | 2 |

### 8.2 阶段 1: 基础预训练

**目标**：让模型学会基础的序列建模能力

```python
from algorithm.training import Stage1Config, Trainer

config = Stage1Config(
    train_data_path="data/pretrain/train.jsonl",
    eval_data_path="data/pretrain/eval.jsonl",
    output_dir="checkpoints/stage1",
    
    batch_size=512,
    max_epochs=5,
    learning_rate=1e-4,
    
    # 只使用 NTP 损失
    lambda_contrastive=0.0,
    lambda_preference=0.0,
)

trainer = Trainer(model=model, config=config, ...)
trainer.train()
```

### 8.3 阶段 2: 多任务微调

**目标**：学习用户和物品的表示对齐

```python
from algorithm.training import Stage2Config, Trainer

config = Stage2Config(
    pretrained_model_path="checkpoints/stage1/best_model",
    train_data_path="data/multitask/train.jsonl",
    output_dir="checkpoints/stage2",
    
    batch_size=256,
    max_epochs=3,
    learning_rate=5e-5,  # 较小学习率
    
    # 加入对比学习
    lambda_contrastive=0.1,
    contrastive_temperature=0.07,
)

# 加载预训练模型
model.load_state_dict(torch.load("checkpoints/stage1/best_model/model.pt"))

trainer = Trainer(model=model, config=config, ...)
trainer.train()
```

### 8.4 阶段 3: 偏好对齐

**目标**：让模型偏好用户选择的物品

```python
from algorithm.training import Stage3Config
from algorithm.training.scripts.train_stage3 import DPOTrainer

config = Stage3Config(
    pretrained_model_path="checkpoints/stage2/best_model",
    reference_model_path="checkpoints/stage2/best_model",
    train_data_path="data/preference/train.jsonl",
    output_dir="checkpoints/stage3",
    
    batch_size=128,
    max_epochs=2,
    learning_rate=1e-5,
    
    # DPO 参数
    lambda_preference=0.1,
    dpo_beta=0.1,
)

# 创建参考模型（冻结）
reference_model = create_model()
reference_model.load_state_dict(torch.load("checkpoints/stage2/best_model/model.pt"))
reference_model.eval()
for param in reference_model.parameters():
    param.requires_grad = False

trainer = DPOTrainer(
    model=model,
    config=config,
    reference_model=reference_model,
    ...
)
trainer.train()
```

---

## 9. 分布式训练

### 9.1 DDP 训练

```bash
# 单机多卡
torchrun --nproc_per_node=8 \
    -m algorithm.training.scripts.train_stage1 \
    --config configs/stage1.yaml \
    --ddp

# 多机多卡
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_id=job1 --rdzv_backend=c10d --rdzv_endpoint=master:29400 \
    -m algorithm.training.scripts.train_stage1 \
    --config configs/stage1.yaml \
    --ddp
```

### 9.2 使用 DistributedTrainer

```python
from algorithm.training import DistributedTrainer, Stage1Config

config = Stage1Config(ddp=True, local_rank=local_rank)

trainer = DistributedTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
)

trainer.train()
```

### 9.3 DeepSpeed 训练

```bash
deepspeed \
    algorithm/training/scripts/train_stage1.py \
    --config configs/stage1.yaml \
    --deepspeed \
    --zero_stage 2
```

```python
from algorithm.training.distributed import DeepSpeedTrainer

trainer = DeepSpeedTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
)
trainer.train()
```

### 9.4 DeepSpeed 配置示例

```json
{
    "train_batch_size": 1024,
    "train_micro_batch_size_per_gpu": 256,
    "gradient_accumulation_steps": 4,
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "weight_decay": 0.01
        }
    },
    
    "fp16": {
        "enabled": true,
        "loss_scale": 0
    },
    
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "overlap_comm": true
    }
}
```

---

## 10. 评估指标

### 10.1 离线评估指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **Recall@K** | 召回率 | 前 K 个预测中包含正确答案的比例 |
| **NDCG@K** | 归一化折损累计增益 | 考虑正确答案位置的加权得分 |
| **MRR** | 平均倒数排名 | 正确答案排名的倒数平均值 |
| **Hit Rate@K** | 命中率 | 等同于单标签的 Recall@K |
| **Coverage** | 覆盖率 | 被推荐物品占总物品的比例 |
| **Diversity** | 多样性 | 推荐列表内类别的多样性 |

### 10.2 使用指标计算器

```python
from algorithm.training import MetricsCalculator, recall_at_k, ndcg_at_k, mrr

# 直接使用函数
predictions = [
    [(1, 1, 1), (2, 2, 2), (3, 3, 3)],  # 样本1的预测
    [(4, 4, 4), (5, 5, 5), (6, 6, 6)],  # 样本2的预测
]
ground_truth = [(1, 1, 1), (5, 5, 5)]

recall_10 = recall_at_k(predictions, ground_truth, k=10)
ndcg_10 = ndcg_at_k(predictions, ground_truth, k=10)
mrr_value = mrr(predictions, ground_truth)

# 使用指标计算器（批量处理）
calculator = MetricsCalculator()
calculator.add_batch(predictions, ground_truth)
metrics = calculator.compute(k_values=[5, 10, 20, 50])
# 返回: {"recall@5", "recall@10", ..., "ndcg@5", ..., "mrr", "gini"}
```

---

## 11. API 参考

### 11.1 配置类

```python
# 基础配置
TrainingConfig(
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    max_epochs: int = 10,
    ...
)

# 阶段配置
Stage1Config(...)  # 预训练
Stage2Config(...)  # 多任务微调
Stage3Config(...)  # 偏好对齐
```

### 11.2 数据集类

```python
# 推荐数据集
RecommendDataset(
    data_path: str,
    max_encoder_length: int = 512,
    max_decoder_length: int = 128,
    pad_token_id: int = 0,
    lazy_loading: bool = False,
)

# 偏好数据集
PreferenceDataset(
    data_path: str,
    ...
)

# 数据整理器
DataCollator(
    pad_token_id: int = 0,
    dynamic_padding: bool = True,
)
```

### 11.3 损失函数

```python
# NTP 损失
NextTokenPredictionLoss(l1_weight, l2_weight, l3_weight, ...)
    .forward(l1_logits, l2_logits, l3_logits, labels_l1, labels_l2, labels_l3)
    -> (loss, metrics_dict)

# 对比学习损失
ContrastiveLoss(temperature=0.07)
    .forward(user_repr, item_repr)
    -> (loss, metrics_dict)

# DPO 损失
DPOLoss(beta=0.1, reference_free=False)
    .forward(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps)
    -> (loss, metrics_dict)

# 统一损失
UnifiedLoss(lambda_contrastive, lambda_preference, lambda_moe_balance, ...)
    .forward(model_outputs, labels, aux_loss)
    -> losses_dict
```

### 11.4 优化器和调度器

```python
# 创建优化器
create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    ...
) -> Optimizer

# 创建调度器
create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    total_steps: int = 100000,
    warmup_steps: int = 10000,
    ...
) -> LRScheduler
```

### 11.5 训练器

```python
Trainer(
    model: nn.Module,
    config: TrainingConfig,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    ...
)

# 方法
.train() -> Dict[str, float]           # 完整训练
.train_epoch() -> Dict[str, float]     # 单 epoch 训练
.evaluate() -> Dict[str, float]        # 验证
.save_checkpoint(path: str) -> None    # 保存检查点
.load_checkpoint(path: str) -> None    # 加载检查点
```

### 11.6 检查点管理

```python
CheckpointManager(
    save_dir: str,
    max_checkpoints: int = 3,
    keep_best: bool = True,
)

.save(checkpoint, path, step, is_best) -> str
.load(path) -> Dict
.load_latest() -> Optional[Dict]
.load_best() -> Optional[Dict]
```

### 11.7 评估指标

```python
# 函数
recall_at_k(predictions, ground_truth, k=10) -> float
ndcg_at_k(predictions, ground_truth, k=10) -> float
mrr(predictions, ground_truth) -> float
hit_rate(predictions, ground_truth, k=10) -> float
coverage(predictions, all_items, k=10) -> float

# 计算器类
MetricsCalculator()
    .add_batch(predictions, ground_truth)
    .compute(k_values=[5, 10, 20, 50]) -> Dict[str, float]
    .reset()
```

---

## 12. 常见问题

### Q1: 如何调整超参数？

**推荐的超参数搜索顺序**：
1. **学习率**：先固定其他参数，搜索 1e-5 ~ 1e-3
2. **批次大小**：在 GPU 内存允许的范围内尽量大
3. **损失权重**：λ₁, λ₂ 通常在 0.05 ~ 0.2 之间
4. **预热步数**：通常为总步数的 5-10%

### Q2: 训练过程中 loss 不下降？

**排查步骤**：
1. 检查学习率是否过大或过小
2. 检查数据是否正确加载
3. 检查梯度是否正常（使用 `torch.nn.utils.clip_grad_norm_` 的返回值）
4. 检查是否有 NaN 值（特别是使用 FP16 时）

### Q3: GPU 内存不足？

**解决方案**：
1. 减小 `batch_size`
2. 增大 `gradient_accumulation_steps` 保持有效批次大小
3. 使用 FP16 混合精度训练
4. 使用 DeepSpeed ZeRO-2/3

### Q4: 如何从检查点恢复训练？

```python
config = TrainingConfig(
    resume_from_checkpoint="checkpoints/checkpoint-step-5000",
    ...
)

trainer = Trainer(model=model, config=config, ...)
trainer.train()  # 自动从检查点继续
```

### Q5: 如何进行多机多卡训练？

```bash
# 机器 1
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=master_ip:29400 \
    train_stage1.py --ddp

# 机器 2
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=master_ip:29400 \
    train_stage1.py --ddp
```

---

## 附录

### A. 依赖版本

```
torch>=2.0.0
numpy>=1.21.0
tqdm>=4.64.0
tensorboard>=2.10.0
pyyaml>=6.0
deepspeed>=0.9.0  # 可选
```

### B. 运行测试

```bash
# 运行所有测试
pytest algorithm/training/tests/ -v

# 运行特定测试
pytest algorithm/training/tests/test_training.py::TestLoss -v

# 运行并显示覆盖率
pytest algorithm/training/tests/ --cov=algorithm/training --cov-report=html
```

### C. 参考资料

- [架构文档: 生成式推荐系统架构设计](../../docs/生成式推荐系统架构设计.md)
- [接口定义: algorithm/interfaces.py](../interfaces.py)
- [任务描述: prompts/person_d_training.md](../prompts/person_d_training.md)

---

> 📝 **维护说明**  
> 本文档随代码更新同步维护。如有问题，请联系 Person D 或提交 Issue。

