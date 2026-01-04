"""
分布式训练模块

实现分布式训练支持：
- DDP (DistributedDataParallel)
- DeepSpeed 集成
- 多 GPU / 多节点训练

对应架构文档: 第八章 训练与部署流水线
"""

import os
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .config import TrainingConfig, get_deepspeed_config
from .trainer import Trainer


logger = logging.getLogger(__name__)


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> int:
    """
    初始化分布式训练环境
    
    Args:
        backend: 分布式后端 ("nccl", "gloo", "mpi")
        init_method: 初始化方法
    
    Returns:
        local_rank: 本地进程 rank
    """
    # 从环境变量获取分布式配置
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    if local_rank == -1:
        logger.info("未检测到分布式环境，使用单 GPU 训练")
        return -1
    
    # 设置 CUDA 设备
    torch.cuda.set_device(local_rank)
    
    # 初始化进程组
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    
    logger.info(f"分布式训练初始化完成: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    
    return local_rank


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("分布式训练环境已清理")


def is_main_process() -> bool:
    """检查是否是主进程"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """获取总进程数"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """获取当前进程 rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def barrier():
    """同步所有进程"""
    if dist.is_initialized():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM):
    """
    所有进程归约张量
    
    Args:
        tensor: 待归约的张量
        op: 归约操作
    """
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    收集所有进程的张量
    
    Args:
        tensor: 本地张量
    
    Returns:
        收集后的张量
    """
    if not dist.is_initialized():
        return tensor
    
    world_size = get_world_size()
    
    # 创建用于收集的张量列表
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    
    return torch.cat(gathered, dim=0)


def broadcast(tensor: torch.Tensor, src: int = 0):
    """
    从源进程广播张量
    
    Args:
        tensor: 待广播的张量
        src: 源进程 rank
    """
    if dist.is_initialized():
        dist.broadcast(tensor, src=src)


def reduce_dict(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    归约指标字典
    
    Args:
        metrics: 指标字典
    
    Returns:
        归约后的指标字典
    """
    if not dist.is_initialized():
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
    
    world_size = get_world_size()
    
    result = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            tensor = value.clone()
            all_reduce(tensor)
            result[key] = tensor.item() / world_size
        else:
            result[key] = value
    
    return result


def wrap_model_for_ddp(
    model: nn.Module,
    local_rank: int,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """
    包装模型为 DDP
    
    Args:
        model: 模型
        local_rank: 本地 rank
        find_unused_parameters: 是否查找未使用的参数
    
    Returns:
        DDP 包装后的模型
    """
    model = model.cuda(local_rank)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
    )
    
    logger.info(f"模型已包装为 DDP (local_rank={local_rank})")
    
    return model


def create_distributed_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    collate_fn=None,
) -> DataLoader:
    """
    创建分布式数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        pin_memory: 是否使用 pin_memory
        drop_last: 是否丢弃最后一个批次
        collate_fn: 数据整理函数
    
    Returns:
        分布式数据加载器
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    
    return dataloader


class DistributedTrainer(Trainer):
    """
    分布式训练器
    
    扩展基础训练器以支持分布式训练
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset,
        eval_dataset=None,
        optimizer=None,
        scheduler=None,
        reference_model=None,
    ):
        """
        初始化分布式训练器
        """
        # 初始化分布式环境
        self.local_rank = setup_distributed(backend=config.ddp_backend)
        config.local_rank = self.local_rank
        
        # 设置随机种子（确保不同进程有不同的数据顺序）
        seed = config.seed + get_rank()
        torch.manual_seed(seed)
        
        # 调用父类初始化
        super().__init__(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            reference_model=reference_model,
        )
    
    def _setup_device(self):
        """设置设备（分布式版本）"""
        if self.local_rank >= 0:
            device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        return device
    
    def _create_dataloader(self, dataset, shuffle=True):
        """创建分布式数据加载器"""
        from .dataset import DataCollator
        
        if self.local_rank >= 0:
            return create_distributed_dataloader(
                dataset=dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.config.dataloader_pin_memory,
                drop_last=self.config.dataloader_drop_last,
                collate_fn=DataCollator(),
            )
        else:
            return super()._create_dataloader(dataset, shuffle)
    
    def train(self):
        """分布式训练"""
        # 包装模型
        if self.local_rank >= 0:
            self.model = wrap_model_for_ddp(
                self.model,
                self.local_rank,
                self.config.ddp_find_unused_parameters,
            )
        
        # 调用父类训练
        result = super().train()
        
        # 清理分布式环境
        cleanup_distributed()
        
        return result
    
    def train_epoch(self):
        """训练一个 epoch（分布式版本）"""
        # 设置 sampler 的 epoch（确保每个 epoch 数据顺序不同）
        if hasattr(self.train_dataloader, 'sampler') and \
           hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.current_epoch)
        
        return super().train_epoch()
    
    def evaluate(self):
        """评估（分布式版本）"""
        metrics = super().evaluate()
        
        # 归约所有进程的指标
        if self.local_rank >= 0:
            metrics_tensor = {
                k: torch.tensor(v, device=self.device) 
                for k, v in metrics.items()
            }
            metrics = reduce_dict(metrics_tensor)
        
        return metrics
    
    def save_checkpoint(self, path, is_best=False):
        """保存检查点（仅主进程）"""
        if is_main_process():
            # 获取原始模型（去除 DDP 包装）
            model_to_save = self.model
            if isinstance(self.model, DDP):
                model_to_save = self.model.module
            
            # 临时替换模型
            original_model = self.model
            self.model = model_to_save
            
            super().save_checkpoint(path, is_best)
            
            # 恢复模型
            self.model = original_model
        
        # 同步所有进程
        barrier()


class DeepSpeedTrainer:
    """
    DeepSpeed 训练器
    
    使用 DeepSpeed 进行高效的分布式训练
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset,
        eval_dataset=None,
    ):
        """
        初始化 DeepSpeed 训练器
        """
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 检查 DeepSpeed 是否可用
        try:
            import deepspeed
        except ImportError:
            raise ImportError("使用 DeepSpeedTrainer 需要安装 deepspeed")
        
        # 生成 DeepSpeed 配置
        ds_config = get_deepspeed_config(config)
        
        # 初始化 DeepSpeed
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=model,
            config=ds_config,
            model_parameters=model.parameters(),
        )
        
        # 创建数据加载器
        from .dataset import DataCollator
        self.train_dataloader = create_distributed_dataloader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.dataloader_num_workers,
            collate_fn=DataCollator(),
        )
        
        if eval_dataset is not None:
            self.eval_dataloader = create_distributed_dataloader(
                dataset=eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.dataloader_num_workers,
                collate_fn=DataCollator(),
            )
        else:
            self.eval_dataloader = None
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
    
    def train(self):
        """使用 DeepSpeed 训练"""
        from .loss import UnifiedLoss
        
        loss_fn = UnifiedLoss(
            l1_weight=self.config.l1_loss_weight,
            l2_weight=self.config.l2_loss_weight,
            l3_weight=self.config.l3_loss_weight,
            lambda_contrastive=self.config.lambda_contrastive,
            lambda_preference=self.config.lambda_preference,
            lambda_moe_balance=self.config.lambda_moe_balance,
        )
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # 设置 sampler 的 epoch
            if hasattr(self.train_dataloader, 'sampler') and \
               hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            self.model.train()
            
            for step, batch in enumerate(self.train_dataloader):
                # 移动到设备
                batch = {
                    k: v.cuda() if isinstance(v, torch.Tensor) else [t.cuda() for t in v]
                    for k, v in batch.items()
                }
                
                # 前向传播
                outputs = self.model(
                    encoder_semantic_ids=batch["encoder_semantic_ids"],
                    encoder_positions=batch["encoder_positions"],
                    encoder_token_types=batch["encoder_token_types"],
                    encoder_attention_mask=batch["encoder_mask"],
                    decoder_semantic_ids=batch["decoder_semantic_ids"],
                    decoder_positions=batch["decoder_positions"],
                    decoder_token_types=batch["decoder_token_types"],
                )
                
                losses = loss_fn(
                    model_outputs=outputs,
                    labels={
                        "l1": batch["labels"][0],
                        "l2": batch["labels"][1],
                        "l3": batch["labels"][2],
                    },
                    aux_loss=outputs.get("aux_loss"),
                )
                
                loss = losses["total_loss"]
                
                # 反向传播（DeepSpeed 自动处理）
                self.model.backward(loss)
                self.model.step()
                
                self.global_step += 1
                
                # 日志
                if is_main_process() and self.global_step % self.config.logging_steps == 0:
                    logger.info(
                        f"Step {self.global_step}: loss={loss.item():.4f}"
                    )
            
            # 保存检查点
            if is_main_process():
                self.save_checkpoint(
                    os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch}")
                )
        
        return {"final_step": self.global_step}
    
    def save_checkpoint(self, path: str):
        """保存 DeepSpeed 检查点"""
        self.model.save_checkpoint(path)
        logger.info(f"DeepSpeed 检查点已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """加载 DeepSpeed 检查点"""
        self.model.load_checkpoint(path)
        logger.info(f"DeepSpeed 检查点已加载: {path}")

