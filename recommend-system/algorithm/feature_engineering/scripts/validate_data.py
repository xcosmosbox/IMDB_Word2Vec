#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据验证脚本

验证处理后的数据质量，检查潜在问题。

使用方法:
    python validate_data.py --input data/processed/train.jsonl
    python validate_data.py --input data/processed/ --output report.json
    
参数说明:
    --input: 输入数据路径（文件或目录）
    --output: 验证报告输出路径（可选）
    --vocab: 词表路径（用于检查 Token 覆盖率）
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional

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
        description='验证处理后的数据质量',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 验证单个文件
    python validate_data.py --input data/processed/train.jsonl
    
    # 验证目录下所有文件
    python validate_data.py --input data/processed/
    
    # 保存验证报告
    python validate_data.py --input data/processed/ --output report.json
    
    # 检查 Token 覆盖率
    python validate_data.py --input data/processed/ --vocab vocab.json
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入数据路径（文件或目录）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='验证报告输出路径（JSON 格式，可选）'
    )
    
    parser.add_argument(
        '--vocab',
        type=str,
        default=None,
        help='词表路径（用于检查 Token 覆盖率，可选）'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大检查样本数（用于大文件，默认: 全部）'
    )
    
    return parser.parse_args()


class DataValidator:
    """数据验证器"""
    
    def __init__(self, vocab: Optional[Vocabulary] = None):
        """
        初始化验证器
        
        Args:
            vocab: 词表（用于检查 Token 覆盖率）
        """
        self.vocab = vocab
        self.issues: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
    
    def validate_sample(self, sample: Dict[str, Any], sample_idx: int) -> bool:
        """
        验证单个样本
        
        Args:
            sample: 样本数据
            sample_idx: 样本索引
            
        Returns:
            是否有效
        """
        is_valid = True
        
        # 检查必要字段
        required_fields = ['input_ids', 'attention_mask', 'token_types']
        for field in required_fields:
            if field not in sample:
                self.issues.append({
                    'type': 'missing_field',
                    'sample_idx': sample_idx,
                    'field': field,
                    'message': f'样本缺少必要字段: {field}'
                })
                is_valid = False
        
        if not is_valid:
            return False
        
        # 检查序列长度一致性
        input_len = len(sample['input_ids'])
        mask_len = len(sample['attention_mask'])
        types_len = len(sample['token_types'])
        
        if not (input_len == mask_len == types_len):
            self.issues.append({
                'type': 'length_mismatch',
                'sample_idx': sample_idx,
                'message': f'序列长度不一致: input_ids={input_len}, attention_mask={mask_len}, token_types={types_len}'
            })
            is_valid = False
        
        # 检查语义 ID
        if 'semantic_ids' in sample:
            semantic_ids = sample['semantic_ids']
            if len(semantic_ids) != 3:
                self.issues.append({
                    'type': 'invalid_semantic_ids',
                    'sample_idx': sample_idx,
                    'message': f'语义 ID 应有 3 层，实际: {len(semantic_ids)}'
                })
                is_valid = False
            else:
                for i, sid in enumerate(semantic_ids):
                    if len(sid) != input_len:
                        self.issues.append({
                            'type': 'semantic_id_length_mismatch',
                            'sample_idx': sample_idx,
                            'message': f'语义 ID L{i+1} 长度 ({len(sid)}) 与 input_ids 长度 ({input_len}) 不匹配'
                        })
                        is_valid = False
        
        # 检查 attention_mask 有效性
        valid_positions = sum(1 for m in sample['attention_mask'] if m == 1)
        if valid_positions == 0:
            self.issues.append({
                'type': 'empty_sequence',
                'sample_idx': sample_idx,
                'message': '序列为空（attention_mask 全为 0）'
            })
            is_valid = False
        
        # 检查 Token ID 范围
        if self.vocab:
            vocab_size = len(self.vocab)
            for idx, token_id in enumerate(sample['input_ids']):
                if token_id < 0 or token_id >= vocab_size:
                    self.issues.append({
                        'type': 'invalid_token_id',
                        'sample_idx': sample_idx,
                        'position': idx,
                        'token_id': token_id,
                        'message': f'Token ID 超出词表范围: {token_id} (词表大小: {vocab_size})'
                    })
                    is_valid = False
                    break  # 只报告第一个错误
        
        return is_valid
    
    def validate_file(
        self, 
        file_path: str, 
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        验证单个文件
        
        Args:
            file_path: 文件路径
            max_samples: 最大检查样本数
            
        Returns:
            验证结果
        """
        logger.info(f"验证文件: {file_path}")
        
        self.issues.clear()
        self.warnings.clear()
        
        total_samples = 0
        valid_samples = 0
        sequence_lengths = []
        token_type_dist = Counter()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if max_samples and line_idx >= max_samples:
                    break
                
                try:
                    sample = json.loads(line.strip())
                    total_samples += 1
                    
                    if self.validate_sample(sample, line_idx):
                        valid_samples += 1
                        
                        # 收集统计信息
                        valid_len = sum(1 for m in sample['attention_mask'] if m == 1)
                        sequence_lengths.append(valid_len)
                        
                        for tt in sample['token_types']:
                            token_type_dist[tt] += 1
                            
                except json.JSONDecodeError as e:
                    self.issues.append({
                        'type': 'json_error',
                        'line_idx': line_idx,
                        'message': f'JSON 解析错误: {str(e)}'
                    })
        
        # 计算统计信息
        if sequence_lengths:
            avg_length = sum(sequence_lengths) / len(sequence_lengths)
            max_length = max(sequence_lengths)
            min_length = min(sequence_lengths)
        else:
            avg_length = max_length = min_length = 0
        
        result = {
            'file_path': file_path,
            'total_samples': total_samples,
            'valid_samples': valid_samples,
            'invalid_samples': total_samples - valid_samples,
            'validity_rate': valid_samples / total_samples if total_samples > 0 else 0,
            'sequence_length': {
                'min': min_length,
                'max': max_length,
                'avg': avg_length,
            },
            'token_type_distribution': dict(token_type_dist),
            'issues': self.issues[:100],  # 只保留前 100 个问题
            'total_issues': len(self.issues),
            'warnings': self.warnings,
        }
        
        return result
    
    def validate_directory(
        self, 
        dir_path: str, 
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        验证目录下所有文件
        
        Args:
            dir_path: 目录路径
            max_samples: 每个文件的最大检查样本数
            
        Returns:
            验证结果
        """
        logger.info(f"验证目录: {dir_path}")
        
        results = {}
        total_samples = 0
        total_valid = 0
        total_issues = 0
        
        for filename in os.listdir(dir_path):
            if filename.endswith('.jsonl') or filename.endswith('.json'):
                file_path = os.path.join(dir_path, filename)
                result = self.validate_file(file_path, max_samples)
                results[filename] = result
                
                total_samples += result['total_samples']
                total_valid += result['valid_samples']
                total_issues += result['total_issues']
        
        return {
            'directory': dir_path,
            'files': results,
            'summary': {
                'total_files': len(results),
                'total_samples': total_samples,
                'total_valid': total_valid,
                'total_issues': total_issues,
                'overall_validity_rate': total_valid / total_samples if total_samples > 0 else 0,
            }
        }


def print_report(report: Dict[str, Any]) -> None:
    """打印验证报告"""
    logger.info("=" * 60)
    logger.info("数据验证报告")
    logger.info("=" * 60)
    
    if 'summary' in report:
        # 目录验证结果
        summary = report['summary']
        logger.info(f"验证目录: {report['directory']}")
        logger.info(f"文件数量: {summary['total_files']}")
        logger.info(f"总样本数: {summary['total_samples']}")
        logger.info(f"有效样本: {summary['total_valid']}")
        logger.info(f"问题数量: {summary['total_issues']}")
        logger.info(f"有效率: {summary['overall_validity_rate']:.2%}")
        
        logger.info("-" * 40)
        logger.info("各文件详情:")
        for filename, result in report['files'].items():
            logger.info(f"  {filename}:")
            logger.info(f"    - 样本数: {result['total_samples']}")
            logger.info(f"    - 有效率: {result['validity_rate']:.2%}")
            logger.info(f"    - 平均长度: {result['sequence_length']['avg']:.1f}")
    else:
        # 单文件验证结果
        logger.info(f"验证文件: {report['file_path']}")
        logger.info(f"总样本数: {report['total_samples']}")
        logger.info(f"有效样本: {report['valid_samples']}")
        logger.info(f"无效样本: {report['invalid_samples']}")
        logger.info(f"有效率: {report['validity_rate']:.2%}")
        
        logger.info("-" * 40)
        logger.info("序列长度统计:")
        logger.info(f"  - 最小: {report['sequence_length']['min']}")
        logger.info(f"  - 最大: {report['sequence_length']['max']}")
        logger.info(f"  - 平均: {report['sequence_length']['avg']:.1f}")
        
        logger.info("-" * 40)
        logger.info("Token 类型分布:")
        type_names = {0: 'USER', 1: 'ITEM', 2: 'ACTION', 3: 'CONTEXT'}
        for type_id, count in sorted(report['token_type_distribution'].items()):
            type_name = type_names.get(int(type_id), f'TYPE_{type_id}')
            logger.info(f"  - {type_name}: {count}")
    
    # 问题摘要
    if 'issues' in report and report['issues']:
        logger.info("-" * 40)
        logger.info(f"发现问题 (显示前 10 个，共 {report.get('total_issues', len(report['issues']))} 个):")
        for issue in report['issues'][:10]:
            logger.warning(f"  [{issue['type']}] {issue['message']}")
    
    logger.info("=" * 60)


def main():
    """主函数"""
    args = parse_args()
    
    # 加载词表（如果提供）
    vocab = None
    if args.vocab and os.path.exists(args.vocab):
        config = FeatureConfig()
        vocab = Vocabulary(config)
        vocab.load(args.vocab)
        logger.info(f"加载词表: {args.vocab} ({len(vocab)} 个 Token)")
    
    # 创建验证器
    validator = DataValidator(vocab)
    
    # 检查输入路径
    if not os.path.exists(args.input):
        logger.error(f"输入路径不存在: {args.input}")
        sys.exit(1)
    
    # 执行验证
    if os.path.isdir(args.input):
        report = validator.validate_directory(args.input, args.max_samples)
    else:
        report = validator.validate_file(args.input, args.max_samples)
    
    # 打印报告
    print_report(report)
    
    # 保存报告（如果指定）
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"验证报告已保存到: {args.output}")
    
    # 根据验证结果设置退出码
    if 'summary' in report:
        validity_rate = report['summary']['overall_validity_rate']
    else:
        validity_rate = report['validity_rate']
    
    if validity_rate < 0.95:
        logger.warning("数据质量警告：有效率低于 95%")
        sys.exit(1)
    else:
        logger.info("数据验证通过！")
        sys.exit(0)


if __name__ == '__main__':
    main()

