#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 2: 多任务微调脚本

目标：学习用户-物品表示对齐
损失：NTP + 对比学习

使用方法:
    # 单 GPU 训练
    python train_stage2.py --config configs/stage2.yaml --pretrained checkpoints/stage1/best_model
    
    # 多 GPU 训练 (DDP)
    torchrun --nproc_per_node=8 train_stage2.py --config configs/stage2.yaml --pretrained checkpoints/stage1/best_model
"""

import os
import sys
import argparse
import logging
import yaml

import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.config import Stage2Config
from training.dataset import RecommendDataset
from training.trainer import Trainer
from training.distributed import DistributedTrainer, is_main_process
from training.checkpoint import CheckpointManager


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="UGT 模型阶段 2 多任务微调")
    
    # 配置文件
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    
    # 数据路径
    parser.add_argument("--train_data", type=str, default="data/multitask/train.jsonl")
    parser.add_argument("--eval_data", type=str, default="data/multitask/eval.jsonl")
    
    # 预训练模型
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="阶段 1 预训练模型路径",
    )
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lambda_contrastive", type=float, default=0.1)
    
    # 输出目录
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage2")
    
    # 分布式训练
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    
    return parser.parse_args()


def load_config(args) -> Stage2Config:
    """加载配置"""
    config = Stage2Config()
    
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
    config.lambda_contrastive = args.lambda_contrastive
    config.output_dir = args.output_dir
    config.seed = args.seed
    config.fp16 = args.fp16
    config.train_data_path = args.train_data
    config.eval_data_path = args.eval_data
    config.pretrained_model_path = args.pretrained
    config.ddp = args.ddp
    config.deepspeed = args.deepspeed
    config.local_rank = args.local_rank
    
    return config


def create_model_from_pretrained(config: Stage2Config):
    """从预训练检查点加载模型"""
    from .train_stage1 import create_model
    
    model = create_model(config)
    
    # 加载预训练权重
    checkpoint_path = config.pretrained_model_path
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "checkpoint.pt")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"已加载预训练模型: {checkpoint_path}")
    else:
        logger.warning(f"预训练模型不存在: {checkpoint_path}")
    
    return model


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
        logger.info("阶段 2: 多任务微调")
        logger.info("=" * 60)
        logger.info(f"预训练模型: {config.pretrained_model_path}")
        logger.info(f"对比学习权重: {config.lambda_contrastive}")
    
    # 加载数据集
    if is_main_process():
        logger.info("加载数据集...")
    
    train_dataset = None
    eval_dataset = None
    
    if os.path.exists(config.train_data_path):
        train_dataset = RecommendDataset(
            data_path=config.train_data_path,
            max_encoder_length=config.encoder_max_length,
            max_decoder_length=config.decoder_max_length,
        )
    else:
        logger.warning(f"训练数据不存在: {config.train_data_path}，使用模拟数据")
        from training.tests.test_training import MockDataset
        train_dataset = MockDataset(size=1000)
    
    if os.path.exists(config.eval_data_path):
        eval_dataset = RecommendDataset(
            data_path=config.eval_data_path,
            max_encoder_length=config.encoder_max_length,
            max_decoder_length=config.decoder_max_length,
        )
    
    # 创建模型
    if is_main_process():
        logger.info("加载预训练模型...")
    
    model = create_model_from_pretrained(config)
    
    # 创建训练器
    if config.ddp or config.local_rank >= 0:
        trainer = DistributedTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    else:
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    
    # 开始训练
    if is_main_process():
        logger.info("开始多任务微调...")
    
    result = trainer.train()
    
    if is_main_process():
        logger.info(f"训练完成！结果: {result}")
        logger.info(f"模型已保存至: {config.output_dir}")


if __name__ == "__main__":
    main()

