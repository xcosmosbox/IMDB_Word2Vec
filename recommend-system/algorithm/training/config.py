"""
训练配置模块

定义训练过程中的所有配置参数，包括：
- 基础训练配置
- 三阶段特定配置
- 分布式训练配置

对应架构文档: 第八章 训练与部署流水线
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class LRSchedulerType(str, Enum):
    """学习率调度器类型"""
    LINEAR = "linear"
    COSINE = "cosine"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    INVERSE_SQRT = "inverse_sqrt"


class OptimizerType(str, Enum):
    """优化器类型"""
    ADAM = "adam"
    ADAMW = "adamw"
    LAMB = "lamb"
    ADAFACTOR = "adafactor"


@dataclass
class TrainingConfig:
    """
    训练配置基类
    
    包含所有训练阶段共用的配置参数
    """
    
    # ==================== 基础配置 ====================
    output_dir: str = "checkpoints"
    """检查点和日志输出目录"""
    
    experiment_name: str = "ugt_training"
    """实验名称，用于日志和检查点命名"""
    
    seed: int = 42
    """随机种子，确保可复现性"""
    
    # ==================== 批次配置 ====================
    batch_size: int = 256
    """每个 GPU 的批次大小"""
    
    gradient_accumulation_steps: int = 4
    """梯度累积步数，有效批次 = batch_size * gradient_accumulation_steps * world_size"""
    
    max_epochs: int = 10
    """最大训练轮数"""
    
    max_steps: int = -1
    """最大训练步数，-1 表示不限制（由 max_epochs 决定）"""
    
    # ==================== 序列配置 ====================
    max_seq_length: int = 1024
    """最大序列长度"""
    
    encoder_max_length: int = 512
    """编码器最大输入长度"""
    
    decoder_max_length: int = 128
    """解码器最大输出长度"""
    
    # ==================== 优化器配置 ====================
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    """优化器类型"""
    
    learning_rate: float = 1e-4
    """初始学习率"""
    
    weight_decay: float = 0.01
    """权重衰减系数"""
    
    adam_beta1: float = 0.9
    """Adam 优化器 beta1 参数"""
    
    adam_beta2: float = 0.999
    """Adam 优化器 beta2 参数"""
    
    adam_epsilon: float = 1e-8
    """Adam 优化器 epsilon 参数，防止除零"""
    
    max_grad_norm: float = 1.0
    """梯度裁剪阈值，0 表示不裁剪"""
    
    # ==================== 学习率调度 ====================
    lr_scheduler_type: LRSchedulerType = LRSchedulerType.COSINE
    """学习率调度器类型"""
    
    warmup_steps: int = 10000
    """学习率预热步数"""
    
    warmup_ratio: float = 0.0
    """学习率预热比例，与 warmup_steps 二选一，优先使用 warmup_steps"""
    
    min_lr_ratio: float = 0.1
    """最小学习率比例（相对于初始学习率）"""
    
    # ==================== 混合精度 ====================
    fp16: bool = True
    """是否使用 FP16 混合精度训练"""
    
    bf16: bool = False
    """是否使用 BF16 混合精度训练（需要 Ampere 或更新架构的 GPU）"""
    
    fp16_opt_level: str = "O1"
    """FP16 优化级别 (O1: 动态损失缩放, O2: 更激进的混合精度)"""
    
    # ==================== 损失权重 ====================
    lambda_contrastive: float = 0.1
    """对比学习损失权重 λ₁"""
    
    lambda_preference: float = 0.1
    """偏好对齐损失权重 λ₂"""
    
    lambda_moe_balance: float = 0.01
    """MoE 负载均衡损失权重 λ₃"""
    
    # ==================== 层次化损失权重 ====================
    l1_loss_weight: float = 0.5
    """第一层语义 ID 预测损失权重"""
    
    l2_loss_weight: float = 0.3
    """第二层语义 ID 预测损失权重"""
    
    l3_loss_weight: float = 0.2
    """第三层语义 ID 预测损失权重"""
    
    # ==================== 日志和保存 ====================
    logging_steps: int = 100
    """日志记录间隔步数"""
    
    logging_dir: str = "logs"
    """TensorBoard 日志目录"""
    
    save_steps: int = 1000
    """检查点保存间隔步数"""
    
    save_total_limit: int = 3
    """最多保留的检查点数量"""
    
    eval_steps: int = 500
    """验证间隔步数"""
    
    eval_on_start: bool = False
    """是否在训练开始前进行一次验证"""
    
    # ==================== 分布式训练 ====================
    local_rank: int = -1
    """本地进程 rank，-1 表示非分布式"""
    
    ddp: bool = False
    """是否使用 DistributedDataParallel"""
    
    ddp_backend: str = "nccl"
    """DDP 后端 (nccl, gloo, mpi)"""
    
    ddp_find_unused_parameters: bool = False
    """是否查找未使用的参数"""
    
    # ==================== DeepSpeed ====================
    deepspeed: bool = False
    """是否使用 DeepSpeed"""
    
    zero_stage: int = 2
    """DeepSpeed ZeRO 阶段 (0, 1, 2, 3)"""
    
    deepspeed_config: Optional[str] = None
    """DeepSpeed 配置文件路径"""
    
    # ==================== 数据加载 ====================
    dataloader_num_workers: int = 4
    """数据加载器工作进程数"""
    
    dataloader_pin_memory: bool = True
    """是否使用 pin_memory 加速数据传输"""
    
    dataloader_drop_last: bool = False
    """是否丢弃最后一个不完整的批次"""
    
    dataloader_prefetch_factor: int = 2
    """预取因子"""
    
    # ==================== 其他 ====================
    resume_from_checkpoint: Optional[str] = None
    """从检查点恢复训练的路径"""
    
    ignore_data_skip: bool = False
    """恢复训练时是否跳过数据"""
    
    label_smoothing: float = 0.0
    """标签平滑系数"""
    
    def __post_init__(self):
        """配置验证"""
        # 确保 FP16 和 BF16 不同时启用
        if self.fp16 and self.bf16:
            raise ValueError("fp16 和 bf16 不能同时启用")
        
        # 验证层次化损失权重
        total_weight = self.l1_loss_weight + self.l2_loss_weight + self.l3_loss_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"层次化损失权重之和应为 1.0，当前为 {total_weight}")
        
        # 验证 ZeRO 阶段
        if self.deepspeed and self.zero_stage not in [0, 1, 2, 3]:
            raise ValueError(f"无效的 ZeRO 阶段: {self.zero_stage}")
    
    @property
    def effective_batch_size(self) -> int:
        """计算有效批次大小"""
        return self.batch_size * self.gradient_accumulation_steps
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
        }


@dataclass
class Stage1Config(TrainingConfig):
    """
    阶段 1: 基础预训练配置
    
    目标：学习基础的序列建模能力
    损失：仅使用 Next Token Prediction 损失
    """
    
    experiment_name: str = "ugt_stage1_pretrain"
    
    # 阶段 1 特定参数
    max_epochs: int = 5
    learning_rate: float = 1e-4
    warmup_steps: int = 10000
    
    # 阶段 1 不使用对比学习和偏好学习
    lambda_contrastive: float = 0.0
    lambda_preference: float = 0.0
    
    # 较大的批次用于预训练
    batch_size: int = 512
    gradient_accumulation_steps: int = 2
    
    # 预训练数据配置
    train_data_path: str = "data/pretrain/train.jsonl"
    eval_data_path: str = "data/pretrain/eval.jsonl"


@dataclass
class Stage2Config(TrainingConfig):
    """
    阶段 2: 多任务微调配置
    
    目标：学习用户-物品表示对齐
    损失：NTP + 对比学习
    """
    
    experiment_name: str = "ugt_stage2_multitask"
    
    # 阶段 2 特定参数
    max_epochs: int = 3
    learning_rate: float = 5e-5  # 较小的学习率
    warmup_steps: int = 5000
    
    # 加入对比学习
    lambda_contrastive: float = 0.1
    lambda_preference: float = 0.0  # 阶段 2 不使用偏好学习
    
    # 对比学习温度参数
    contrastive_temperature: float = 0.07
    
    # 负样本配置
    num_negatives: int = 127  # 每个正样本对应的负样本数
    hard_negative_ratio: float = 0.3  # 困难负样本比例
    
    # 从阶段 1 加载
    pretrained_model_path: str = "checkpoints/stage1_best"
    
    # 多任务数据配置
    train_data_path: str = "data/multitask/train.jsonl"
    eval_data_path: str = "data/multitask/eval.jsonl"


@dataclass
class Stage3Config(TrainingConfig):
    """
    阶段 3: 偏好对齐配置 (DPO)
    
    目标：让模型偏好用户选择的物品
    损失：NTP + 对比学习 + DPO
    """
    
    experiment_name: str = "ugt_stage3_dpo"
    
    # 阶段 3 特定参数
    max_epochs: int = 2
    learning_rate: float = 1e-5  # 更小的学习率
    warmup_steps: int = 2000
    
    # 全部损失
    lambda_contrastive: float = 0.1
    lambda_preference: float = 0.1
    
    # DPO 特定参数
    dpo_beta: float = 0.1
    """DPO 温度参数 β，控制对偏好的敏感程度"""
    
    dpo_reference_free: bool = False
    """是否使用无参考模型的 DPO"""
    
    dpo_label_smoothing: float = 0.0
    """DPO 标签平滑"""
    
    # 从阶段 2 加载
    pretrained_model_path: str = "checkpoints/stage2_best"
    reference_model_path: Optional[str] = None
    """参考模型路径，用于计算 DPO 损失，None 表示使用初始模型"""
    
    # 偏好数据配置
    train_data_path: str = "data/preference/train.jsonl"
    eval_data_path: str = "data/preference/eval.jsonl"


@dataclass
class ModelConfig:
    """
    模型配置（与 interfaces.py 保持一致）
    """
    
    # 模型维度
    d_model: int = 512
    n_heads: int = 16
    d_ff: int = 2048
    dropout: float = 0.1
    
    # 编码器/解码器层数
    n_enc_layers: int = 12
    n_dec_layers: int = 12
    
    # 语义 ID 配置
    semantic_id_levels: int = 3
    codebook_sizes: Tuple[int, int, int] = (1024, 4096, 16384)
    embedding_dim: int = 256
    
    # Token 类型
    num_token_types: int = 4  # USER=0, ITEM=1, ACTION=2, CONTEXT=3
    
    # 特殊 Token ID
    pad_token_id: int = 0
    cls_token_id: int = 1
    sep_token_id: int = 2
    mask_token_id: int = 3
    unk_token_id: int = 4
    
    # MoE 配置
    num_experts: int = 16
    top_k_experts: int = 4
    expert_capacity_factor: float = 1.25


def get_deepspeed_config(config: TrainingConfig) -> dict:
    """
    生成 DeepSpeed 配置
    
    Args:
        config: 训练配置
    
    Returns:
        DeepSpeed 配置字典
    """
    ds_config = {
        "train_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clipping": config.max_grad_norm,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.learning_rate,
                "betas": [config.adam_beta1, config.adam_beta2],
                "eps": config.adam_epsilon,
                "weight_decay": config.weight_decay,
            }
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": config.learning_rate,
                "warmup_num_steps": config.warmup_steps,
                "total_num_steps": config.max_steps if config.max_steps > 0 else 100000,
            }
        },
        
        "fp16": {
            "enabled": config.fp16,
            "loss_scale": 0,  # 动态损失缩放
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        
        "bf16": {
            "enabled": config.bf16,
        },
    }
    
    # ZeRO 配置
    if config.zero_stage == 0:
        ds_config["zero_optimization"] = {"stage": 0}
    elif config.zero_stage == 1:
        ds_config["zero_optimization"] = {
            "stage": 1,
            "reduce_bucket_size": 5e8,
        }
    elif config.zero_stage == 2:
        ds_config["zero_optimization"] = {
            "stage": 2,
            "offload_optimizer": {"device": "none"},
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        }
    elif config.zero_stage == 3:
        ds_config["zero_optimization"] = {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "sub_group_size": 1e12,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
        }
    
    return ds_config

