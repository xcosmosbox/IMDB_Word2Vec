#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 1: 基础预训练脚本

目标：学习基础的序列建模能力
损失：仅使用 Next Token Prediction 损失

使用方法:
    # 单 GPU 训练
    python train_stage1.py --config configs/stage1.yaml
    
    # 多 GPU 训练 (DDP)
    torchrun --nproc_per_node=8 train_stage1.py --config configs/stage1.yaml
    
    # DeepSpeed 训练
    deepspeed train_stage1.py --config configs/stage1.yaml --deepspeed
"""

import os
import sys
import argparse
import logging
import yaml
from typing import Optional

import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.config import Stage1Config
from training.dataset import RecommendDataset, DataCollator, create_dataloader
from training.trainer import Trainer
from training.distributed import DistributedTrainer, setup_distributed, is_main_process


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="UGT 模型阶段 1 预训练")
    
    # 配置文件
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (YAML 格式)",
    )
    
    # 数据路径
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/pretrain/train.jsonl",
        help="训练数据路径",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="data/pretrain/eval.jsonl",
        help="验证数据路径",
    )
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--max_epochs", type=int, default=5, help="最大训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="预热步数")
    
    # 输出目录
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/stage1",
        help="输出目录",
    )
    
    # 分布式训练
    parser.add_argument("--ddp", action="store_true", help="使用 DDP 分布式训练")
    parser.add_argument("--deepspeed", action="store_true", help="使用 DeepSpeed")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地 rank (自动设置)")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fp16", action="store_true", help="使用 FP16 混合精度")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复")
    
    return parser.parse_args()


def load_config(args) -> Stage1Config:
    """加载配置"""
    # 默认配置
    config = Stage1Config()
    
    # 从配置文件加载
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # 从命令行参数覆盖
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_epochs:
        config.max_epochs = args.max_epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.warmup_steps:
        config.warmup_steps = args.warmup_steps
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed:
        config.seed = args.seed
    if args.fp16:
        config.fp16 = True
    if args.resume:
        config.resume_from_checkpoint = args.resume
    
    config.train_data_path = args.train_data
    config.eval_data_path = args.eval_data
    config.ddp = args.ddp
    config.deepspeed = args.deepspeed
    config.local_rank = args.local_rank
    
    return config


def create_model(config: Stage1Config):
    """
    创建模型
    
    注意：这里需要导入实际的模型实现
    目前使用 Mock 模型用于测试
    """
    try:
        # 尝试导入实际模型
        from encoder import UserEncoder
        from decoder import RecommendDecoder
        
        # 创建完整 UGT 模型
        class UGTModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = UserEncoder()
                self.decoder = RecommendDecoder()
            
            def forward(self, **kwargs):
                encoder_output = self.encoder.get_sequence_output(
                    semantic_ids=kwargs["encoder_semantic_ids"],
                    positions=kwargs["encoder_positions"],
                    token_types=kwargs["encoder_token_types"],
                    attention_mask=kwargs["encoder_mask"],
                )
                
                l1_logits, l2_logits, l3_logits, aux_loss = self.decoder(
                    encoder_output=encoder_output,
                    target_semantic_ids=kwargs.get("decoder_semantic_ids"),
                    target_positions=kwargs.get("decoder_positions"),
                    target_token_types=kwargs.get("decoder_token_types"),
                )
                
                return {
                    "l1_logits": l1_logits,
                    "l2_logits": l2_logits,
                    "l3_logits": l3_logits,
                    "aux_loss": aux_loss,
                    "encoder_output": encoder_output,
                }
        
        return UGTModel()
    
    except ImportError:
        # 使用 Mock 模型
        logger.warning("未找到模型实现，使用 Mock 模型")
        
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(100000, 512)
                self.linear = torch.nn.Linear(512, 512)
            
            def forward(self, **kwargs):
                # 获取批次大小和序列长度
                batch_size = kwargs["encoder_semantic_ids"][0].shape[0]
                seq_len = kwargs.get("decoder_semantic_ids", kwargs["encoder_semantic_ids"])[0].shape[1]
                device = kwargs["encoder_semantic_ids"][0].device
                
                return {
                    "l1_logits": torch.randn(batch_size, seq_len, 1024, device=device),
                    "l2_logits": torch.randn(batch_size, seq_len, 4096, device=device),
                    "l3_logits": torch.randn(batch_size, seq_len, 16384, device=device),
                    "aux_loss": torch.tensor(0.1, device=device),
                }
        
        return MockModel()


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
        logger.info("阶段 1: 基础预训练")
        logger.info("=" * 60)
        logger.info(f"配置: {config}")
    
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
        # 使用模拟数据集
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
        logger.info("创建模型...")
    
    model = create_model(config)
    
    # 创建训练器
    if config.deepspeed:
        from training.distributed import DeepSpeedTrainer
        trainer = DeepSpeedTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    elif config.ddp or config.local_rank >= 0:
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
        logger.info("开始训练...")
    
    result = trainer.train()
    
    if is_main_process():
        logger.info(f"训练完成！结果: {result}")
        logger.info(f"模型已保存至: {config.output_dir}")


if __name__ == "__main__":
    main()

