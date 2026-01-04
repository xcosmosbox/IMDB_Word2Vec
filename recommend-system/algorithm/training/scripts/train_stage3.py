#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 3: 偏好对齐脚本 (DPO)

目标：让模型偏好用户选择的物品
损失：NTP + 对比学习 + DPO

使用方法:
    # 单 GPU 训练
    python train_stage3.py --config configs/stage3.yaml --pretrained checkpoints/stage2/best_model
    
    # 多 GPU 训练 (DDP)
    torchrun --nproc_per_node=8 train_stage3.py --config configs/stage3.yaml --pretrained checkpoints/stage2/best_model
"""

import os
import sys
import argparse
import logging
import yaml
import copy

import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.config import Stage3Config
from training.dataset import PreferenceDataset, RecommendDataset
from training.trainer import Trainer
from training.distributed import DistributedTrainer, is_main_process
from training.loss import compute_log_probs


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="UGT 模型阶段 3 偏好对齐 (DPO)")
    
    # 配置文件
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    
    # 数据路径
    parser.add_argument("--train_data", type=str, default="data/preference/train.jsonl")
    parser.add_argument("--eval_data", type=str, default="data/preference/eval.jsonl")
    
    # 预训练模型
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="阶段 2 模型路径",
    )
    parser.add_argument(
        "--reference_model",
        type=str,
        default=None,
        help="参考模型路径（用于 DPO），默认使用 pretrained",
    )
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO 温度参数")
    parser.add_argument("--lambda_preference", type=float, default=0.1)
    
    # 输出目录
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage3")
    
    # 分布式训练
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    
    return parser.parse_args()


def load_config(args) -> Stage3Config:
    """加载配置"""
    config = Stage3Config()
    
    # 从配置文件加载
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # 从命令行参数覆盖
    config.batch_size = args.batch_size
    config.max_epochs = args.max_epochs
    config.learning_rate = args.learning_rate
    config.dpo_beta = args.dpo_beta
    config.lambda_preference = args.lambda_preference
    config.output_dir = args.output_dir
    config.seed = args.seed
    config.fp16 = args.fp16
    config.train_data_path = args.train_data
    config.eval_data_path = args.eval_data
    config.pretrained_model_path = args.pretrained
    config.reference_model_path = args.reference_model or args.pretrained
    config.ddp = args.ddp
    config.local_rank = args.local_rank
    
    return config


def create_model_from_pretrained(checkpoint_path: str):
    """从检查点加载模型"""
    from .train_stage1 import create_model
    
    # 创建模型结构
    model = create_model(None)
    
    # 加载权重
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "checkpoint.pt")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"已加载模型: {checkpoint_path}")
    else:
        logger.warning(f"模型不存在: {checkpoint_path}")
    
    return model


class DPOTrainer(Trainer):
    """
    DPO 训练器
    
    扩展基础训练器以支持 DPO 偏好对齐训练
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Stage3Config,
        train_dataset,
        eval_dataset=None,
        reference_model: nn.Module = None,
    ):
        """
        初始化 DPO 训练器
        
        Args:
            model: 待训练的模型
            config: 训练配置
            train_dataset: 偏好训练数据集
            eval_dataset: 验证数据集
            reference_model: 参考模型（用于计算 DPO 损失）
        """
        super().__init__(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            reference_model=reference_model,
        )
        
        self.dpo_beta = config.dpo_beta
    
    def _compute_dpo_loss(
        self,
        batch,
        model_outputs,
    ):
        """
        计算 DPO 损失
        
        Args:
            batch: 包含 chosen 和 rejected 的批次
            model_outputs: 模型输出
        
        Returns:
            DPO 损失和相关指标
        """
        # 获取 chosen 和 rejected 的 log probabilities
        chosen_ids = batch["chosen_ids"]  # (batch, 3)
        rejected_ids = batch["rejected_ids"]  # (batch, 3)
        
        # 计算当前模型的 log probs
        l1_logits = model_outputs["l1_logits"][:, -1, :]  # 取最后一个位置
        l2_logits = model_outputs["l2_logits"][:, -1, :]
        l3_logits = model_outputs["l3_logits"][:, -1, :]
        
        # Chosen log probs
        chosen_l1_logps = torch.log_softmax(l1_logits, dim=-1).gather(1, chosen_ids[:, 0:1]).squeeze()
        chosen_l2_logps = torch.log_softmax(l2_logits, dim=-1).gather(1, chosen_ids[:, 1:2]).squeeze()
        chosen_l3_logps = torch.log_softmax(l3_logits, dim=-1).gather(1, chosen_ids[:, 2:3]).squeeze()
        chosen_logps = (chosen_l1_logps + chosen_l2_logps + chosen_l3_logps) / 3
        
        # Rejected log probs
        rejected_l1_logps = torch.log_softmax(l1_logits, dim=-1).gather(1, rejected_ids[:, 0:1]).squeeze()
        rejected_l2_logps = torch.log_softmax(l2_logits, dim=-1).gather(1, rejected_ids[:, 1:2]).squeeze()
        rejected_l3_logps = torch.log_softmax(l3_logits, dim=-1).gather(1, rejected_ids[:, 2:3]).squeeze()
        rejected_logps = (rejected_l1_logps + rejected_l2_logps + rejected_l3_logps) / 3
        
        # 计算参考模型的 log probs（如果有）
        reference_chosen_logps = None
        reference_rejected_logps = None
        
        if self.reference_model is not None:
            with torch.no_grad():
                ref_outputs = self.reference_model(
                    encoder_semantic_ids=batch["encoder_semantic_ids"],
                    encoder_positions=batch["encoder_positions"],
                    encoder_token_types=batch["encoder_token_types"],
                    encoder_attention_mask=batch["encoder_mask"],
                )
                
                ref_l1_logits = ref_outputs["l1_logits"][:, -1, :]
                ref_l2_logits = ref_outputs["l2_logits"][:, -1, :]
                ref_l3_logits = ref_outputs["l3_logits"][:, -1, :]
                
                ref_chosen_l1 = torch.log_softmax(ref_l1_logits, dim=-1).gather(1, chosen_ids[:, 0:1]).squeeze()
                ref_chosen_l2 = torch.log_softmax(ref_l2_logits, dim=-1).gather(1, chosen_ids[:, 1:2]).squeeze()
                ref_chosen_l3 = torch.log_softmax(ref_l3_logits, dim=-1).gather(1, chosen_ids[:, 2:3]).squeeze()
                reference_chosen_logps = (ref_chosen_l1 + ref_chosen_l2 + ref_chosen_l3) / 3
                
                ref_rejected_l1 = torch.log_softmax(ref_l1_logits, dim=-1).gather(1, rejected_ids[:, 0:1]).squeeze()
                ref_rejected_l2 = torch.log_softmax(ref_l2_logits, dim=-1).gather(1, rejected_ids[:, 1:2]).squeeze()
                ref_rejected_l3 = torch.log_softmax(ref_l3_logits, dim=-1).gather(1, rejected_ids[:, 2:3]).squeeze()
                reference_rejected_logps = (ref_rejected_l1 + ref_rejected_l2 + ref_rejected_l3) / 3
        
        # 使用 DPO 损失函数
        from training.loss import DPOLoss
        dpo_loss_fn = DPOLoss(
            beta=self.dpo_beta,
            reference_free=(self.reference_model is None),
        )
        
        loss, metrics = dpo_loss_fn(
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
        )
        
        return loss, metrics


def main():
    """主函数"""
    args = parse_args()
    config = load_config(args)
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 记录配置
    if is_main_process():
        logger.info("=" * 60)
        logger.info("阶段 3: 偏好对齐 (DPO)")
        logger.info("=" * 60)
        logger.info(f"预训练模型: {config.pretrained_model_path}")
        logger.info(f"参考模型: {config.reference_model_path}")
        logger.info(f"DPO Beta: {config.dpo_beta}")
        logger.info(f"偏好损失权重: {config.lambda_preference}")
    
    # 加载数据集
    if is_main_process():
        logger.info("加载偏好数据集...")
    
    train_dataset = None
    eval_dataset = None
    
    if os.path.exists(config.train_data_path):
        train_dataset = PreferenceDataset(
            data_path=config.train_data_path,
            max_encoder_length=config.encoder_max_length,
            max_decoder_length=config.decoder_max_length,
        )
    else:
        logger.warning(f"训练数据不存在: {config.train_data_path}，使用模拟数据")
        from training.tests.test_training import MockPreferenceDataset
        train_dataset = MockPreferenceDataset(size=1000)
    
    # 创建模型
    if is_main_process():
        logger.info("加载模型...")
    
    model = create_model_from_pretrained(config.pretrained_model_path)
    
    # 创建参考模型（冻结）
    reference_model = None
    if config.reference_model_path:
        reference_model = create_model_from_pretrained(config.reference_model_path)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
    
    # 创建训练器
    trainer = DPOTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reference_model=reference_model,
    )
    
    # 开始训练
    if is_main_process():
        logger.info("开始偏好对齐训练...")
    
    result = trainer.train()
    
    if is_main_process():
        logger.info(f"训练完成！结果: {result}")
        logger.info(f"模型已保存至: {config.output_dir}")


if __name__ == "__main__":
    main()

