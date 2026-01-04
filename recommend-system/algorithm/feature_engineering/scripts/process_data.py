#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理脚本

处理原始用户行为日志，生成训练所需的序列数据。

使用方法:
    python process_data.py --input data/raw.jsonl --output data/processed/
    python process_data.py --input data/raw.jsonl --output data/processed/ --vocab vocab.json
    
参数说明:
    --input: 原始数据路径
    --output: 输出目录
    --vocab: 词表路径（可选）
    --max-seq-length: 最大序列长度
    --window-size: 滑动窗口大小
    --stride: 滑动步长
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from algorithm.feature_engineering.config import FeatureConfig
from algorithm.feature_engineering.vocabulary import Vocabulary
from algorithm.feature_engineering.tokenizer import RecommendTokenizer
from algorithm.feature_engineering.sequence_builder import SequenceBuilder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='处理原始数据生成训练序列',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本用法
    python process_data.py --input data/raw.jsonl --output data/processed/
    
    # 指定词表
    python process_data.py --input data/raw.jsonl --output data/processed/ --vocab vocab.json
    
    # 自定义窗口参数
    python process_data.py --input data/raw.jsonl --output data/processed/ \\
        --window-size 100 --stride 50
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='原始数据路径（JSON Lines 格式）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录'
    )
    
    parser.add_argument(
        '--vocab',
        type=str,
        default=None,
        help='词表路径（JSON 格式，可选）'
    )
    
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=1024,
        help='最大序列长度（默认: 1024）'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=100,
        help='滑动窗口大小（默认: 100）'
    )
    
    parser.add_argument(
        '--stride',
        type=int,
        default=50,
        help='滑动步长（默认: 50）'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='训练集比例（默认: 0.8）'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='验证集比例（默认: 0.1）'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（JSON 格式，可选）'
    )
    
    return parser.parse_args()


def load_events_by_user(input_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    按用户加载事件
    
    Args:
        input_path: 输入文件路径
        
    Returns:
        按用户分组的事件字典
    """
    events_by_user = defaultdict(list)
    
    logger.info(f"加载数据: {input_path}")
    
    line_count = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                user_id = event.get('user_id', 'unknown')
                events_by_user[user_id].append(event)
                line_count += 1
                
                if line_count % 100000 == 0:
                    logger.info(f"已加载 {line_count} 条事件...")
                    
            except json.JSONDecodeError:
                continue
    
    logger.info(f"加载完成: {line_count} 条事件, {len(events_by_user)} 个用户")
    return events_by_user


def save_samples(samples: List[Dict], output_path: str) -> None:
    """保存样本到文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    logger.info(f"保存 {len(samples)} 个样本到 {output_path}")


def main():
    """主函数"""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("数据处理脚本")
    logger.info("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载或创建配置
    if args.config and os.path.exists(args.config):
        config = FeatureConfig.load(args.config)
    else:
        config = FeatureConfig()
    
    config.max_seq_length = args.max_seq_length
    
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出目录: {args.output}")
    logger.info(f"最大序列长度: {args.max_seq_length}")
    logger.info(f"窗口大小: {args.window_size}")
    logger.info(f"滑动步长: {args.stride}")
    
    # 创建 Tokenizer
    tokenizer = RecommendTokenizer(config)
    
    # 加载词表（如果提供）
    if args.vocab and os.path.exists(args.vocab):
        logger.info(f"加载词表: {args.vocab}")
        tokenizer.load_vocab(args.vocab)
    
    # 创建序列构建器
    builder = SequenceBuilder(config)
    
    # 加载数据
    events_by_user = load_events_by_user(args.input)
    
    # 生成训练样本
    logger.info("生成训练样本...")
    
    all_samples = []
    for user_id, events in events_by_user.items():
        # 按时间排序
        sorted_events = sorted(events, key=lambda x: x.get('timestamp', 0))
        
        # 使用滑动窗口生成样本
        for sample in builder.build_sequences(
            sorted_events,
            window_size=args.window_size,
            stride=args.stride,
        ):
            # Token 化
            tokenized = tokenizer.build_training_sample(
                sample['events'],
                sample['target'],
                max_length=args.max_seq_length,
            )
            
            # 转换为可序列化格式
            all_samples.append({
                'user_id': user_id,
                'input_ids': tokenized.input_ids.tolist(),
                'attention_mask': tokenized.attention_mask.tolist(),
                'token_types': tokenized.token_types.tolist(),
                'semantic_ids': [
                    tokenized.semantic_ids[0].tolist(),
                    tokenized.semantic_ids[1].tolist(),
                    tokenized.semantic_ids[2].tolist(),
                ],
                'labels': tokenized.labels.tolist() if tokenized.labels is not None else None,
            })
    
    logger.info(f"生成 {len(all_samples)} 个训练样本")
    
    # 划分数据集
    import random
    random.seed(42)
    random.shuffle(all_samples)
    
    train_size = int(len(all_samples) * args.train_ratio)
    val_size = int(len(all_samples) * args.val_ratio)
    
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:train_size + val_size]
    test_samples = all_samples[train_size + val_size:]
    
    # 保存数据集
    save_samples(train_samples, os.path.join(args.output, 'train.jsonl'))
    save_samples(val_samples, os.path.join(args.output, 'val.jsonl'))
    save_samples(test_samples, os.path.join(args.output, 'test.jsonl'))
    
    # 保存词表（如果尚未保存）
    vocab_path = os.path.join(args.output, 'vocab.json')
    tokenizer.save_vocab(vocab_path)
    
    # 保存配置
    config_path = os.path.join(args.output, 'config.json')
    config.save(config_path)
    
    # 输出统计信息
    logger.info("-" * 40)
    logger.info("处理统计:")
    logger.info(f"  - 总用户数: {len(events_by_user)}")
    logger.info(f"  - 总样本数: {len(all_samples)}")
    logger.info(f"  - 训练样本: {len(train_samples)}")
    logger.info(f"  - 验证样本: {len(val_samples)}")
    logger.info(f"  - 测试样本: {len(test_samples)}")
    logger.info(f"  - 词表大小: {tokenizer.get_vocab_size()}")
    logger.info("-" * 40)
    
    logger.info("=" * 60)
    logger.info("数据处理完成!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

