"""
特征工程模块

本模块实现生成式推荐系统的特征工程流水线，负责将原始用户行为日志
转换为模型可处理的 Token 序列。

主要功能：
1. 事件解析：将原始事件转换为 Token 列表
2. Token 化：构建词表并进行编码
3. 序列构建：构建训练样本序列
4. 大规模处理：使用 Spark 处理 TB 级数据

模块结构：
- config.py: 配置类定义
- vocabulary.py: 词表管理
- event_parser.py: 事件解析器
- tokenizer.py: Token 化器（实现 TokenizerInterface）
- sequence_builder.py: 序列构建器
- spark_pipeline.py: Spark 大规模处理

使用示例：
    >>> from algorithm.feature_engineering import RecommendTokenizer, FeatureConfig
    >>> config = FeatureConfig()
    >>> tokenizer = RecommendTokenizer(config)
    >>> events = [{"item_id": "movie_001", "action": "click", "timestamp": 1704067200}]
    >>> result = tokenizer.tokenize_events(events)
"""

from algorithm.feature_engineering.config import FeatureConfig
from algorithm.feature_engineering.vocabulary import Vocabulary
from algorithm.feature_engineering.event_parser import EventParser
from algorithm.feature_engineering.tokenizer import RecommendTokenizer
from algorithm.feature_engineering.sequence_builder import SequenceBuilder

__all__ = [
    "FeatureConfig",
    "Vocabulary",
    "EventParser",
    "RecommendTokenizer",
    "SequenceBuilder",
]

__version__ = "1.0.0"
__author__ = "Person E - 特征工程专家"

