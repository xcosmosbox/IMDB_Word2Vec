"""
生成式推荐系统 - 训练 Pipeline 模块

本模块实现完整的三阶段训练流程：
- Stage 1: 基础预训练 (Next Token Prediction)
- Stage 2: 多任务微调 (加入对比学习)
- Stage 3: 偏好对齐 (DPO)

Author: Person D
Version: 1.0.0
"""

from .config import (
    TrainingConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
)
from .dataset import (
    RecommendDataset,
    PreferenceDataset,
    DataCollator,
)
from .loss import (
    NextTokenPredictionLoss,
    ContrastiveLoss,
    DPOLoss,
    UnifiedLoss,
)
from .optimizer import create_optimizer
from .scheduler import create_scheduler
from .trainer import Trainer
from .checkpoint import CheckpointManager
from .metrics import (
    recall_at_k,
    ndcg_at_k,
    mrr,
    hit_rate,
    MetricsCalculator,
)
from .distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    DistributedTrainer,
)

__all__ = [
    # 配置
    "TrainingConfig",
    "Stage1Config",
    "Stage2Config",
    "Stage3Config",
    # 数据集
    "RecommendDataset",
    "PreferenceDataset",
    "DataCollator",
    # 损失函数
    "NextTokenPredictionLoss",
    "ContrastiveLoss",
    "DPOLoss",
    "UnifiedLoss",
    # 优化器和调度器
    "create_optimizer",
    "create_scheduler",
    # 训练器
    "Trainer",
    "DistributedTrainer",
    # 检查点
    "CheckpointManager",
    # 评估指标
    "recall_at_k",
    "ndcg_at_k",
    "mrr",
    "hit_rate",
    "MetricsCalculator",
    # 分布式
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
]

__version__ = "1.0.0"

