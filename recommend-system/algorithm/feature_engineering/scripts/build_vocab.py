#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建词表脚本

从原始数据中统计 Token 频率并构建词表。

使用方法:
    python build_vocab.py --input data/train.jsonl --output vocab.json
    python build_vocab.py --input data/train.jsonl --output vocab.json --min-freq 10
    
参数说明:
    --input: 输入数据路径（JSON Lines 格式）
    --output: 词表输出路径
    --min-freq: 最小 Token 频率（默认: 5）
    --max-size: 词表最大容量（默认: 500000）
    --sample: 采样行数（用于大文件，默认: 全部）
"""

import argparse
import logging
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from algorithm.feature_engineering.config import FeatureConfig
from algorithm.feature_engineering.vocabulary import Vocabulary

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='构建推荐系统词表',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本用法
    python build_vocab.py --input data/train.jsonl --output vocab.json
    
    # 指定最小频率
    python build_vocab.py --input data/train.jsonl --output vocab.json --min-freq 10
    
    # 限制词表大小
    python build_vocab.py --input data/train.jsonl --output vocab.json --max-size 100000
    
    # 采样处理（用于大文件）
    python build_vocab.py --input data/train.jsonl --output vocab.json --sample 1000000
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入数据路径（JSON Lines 格式）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='词表输出路径（JSON 格式）'
    )
    
    parser.add_argument(
        '--min-freq',
        type=int,
        default=5,
        help='最小 Token 频率（默认: 5）'
    )
    
    parser.add_argument(
        '--max-size',
        type=int,
        default=500000,
        help='词表最大容量（默认: 500000）'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='采样行数（用于大文件，默认: 全部）'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（JSON 格式，可选）'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("词表构建脚本")
    logger.info("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 加载或创建配置
    if args.config and os.path.exists(args.config):
        logger.info(f"加载配置文件: {args.config}")
        config = FeatureConfig.load(args.config)
    else:
        config = FeatureConfig()
    
    # 更新配置
    config.vocab_size = args.max_size
    config.min_token_freq = args.min_freq
    
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出路径: {args.output}")
    logger.info(f"最小频率: {args.min_freq}")
    logger.info(f"最大词表: {args.max_size}")
    
    # 创建词表
    vocab = Vocabulary(config)
    
    # 从数据构建词表
    logger.info("开始构建词表...")
    vocab.build_from_data(
        data_path=args.input,
        min_freq=args.min_freq,
        max_lines=args.sample,
    )
    
    # 保存词表
    logger.info(f"保存词表到: {args.output}")
    vocab.save(args.output)
    
    # 输出统计信息
    logger.info("-" * 40)
    logger.info("词表统计:")
    logger.info(f"  - 总 Token 数: {len(vocab)}")
    logger.info(f"  - 行为 Token 数: {len(vocab.get_action_tokens())}")
    logger.info(f"  - 物品 Token 数: {len(vocab.get_item_tokens())}")
    logger.info("-" * 40)
    
    # 输出高频 Token
    logger.info("Top 10 高频 Token:")
    for token, count in vocab.get_most_common_tokens(10):
        logger.info(f"  {token}: {count}")
    
    logger.info("=" * 60)
    logger.info("词表构建完成!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

