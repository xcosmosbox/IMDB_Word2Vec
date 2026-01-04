# Person E: 特征工程

## 你的角色
你是一名数据工程师，负责实现生成式推荐系统的 **特征工程** 模块。

## 背景知识

生成式推荐系统将所有用户交互事件统一编码为 Token 序列，类似于 NLP 中的文本 Token 化。

### Token 类型设计

```
TOKEN 类型              示例                         说明
────────────────────────────────────────────────────────────────
SPECIAL           [PAD], [CLS], [SEP], [MASK]     特殊控制 Token
ACTION_*          ACTION_click, ACTION_buy        用户行为类型
ITEM_L1_L2_L3     ITEM_256_1234_8901              物品语义 ID
TIME_*            TIME_morning, TIME_weekend      时间特征
DEVICE_*          DEVICE_mobile, DEVICE_pc        设备类型
CONTEXT_*         CONTEXT_home, CONTEXT_search    场景上下文
USER_*            USER_age_25, USER_gender_m      用户属性
```

### 数据流

```
原始日志 (JSON/CSV)
       ↓
   事件解析
       ↓
   Token 化
       ↓
  序列构建
       ↓
训练数据 (Parquet/TFRecord)
```

## 你的任务

在 `algorithm/feature_engineering/` 目录下实现完整的特征工程模块。

### 目录结构

```
algorithm/feature_engineering/
├── __init__.py
├── config.py           # 配置类
├── vocabulary.py       # 词表管理
├── event_parser.py     # 事件解析器
├── tokenizer.py        # Token 化器
├── sequence_builder.py # 序列构建器
├── spark_pipeline.py   # Spark 大规模处理
└── scripts/
    ├── build_vocab.py      # 构建词表
    ├── process_data.py     # 处理数据
    └── validate_data.py    # 验证数据
```

### 接口要求

你必须实现 `interfaces.py` 中定义的 `TokenizerInterface`：

```python
from algorithm.interfaces import TokenizerInterface, TokenizedSequence

class RecommendTokenizer(TokenizerInterface):
    def tokenize_events(
        self,
        events: List[Dict[str, Any]],
        max_length: int = 1024,
    ) -> TokenizedSequence:
        """将事件列表转换为 Token 序列"""
        pass
    
    def build_training_sample(
        self,
        events: List[Dict[str, Any]],
        target_item: Dict[str, Any],
    ) -> TokenizedSequence:
        """构建训练样本"""
        pass
    
    def get_vocab_size(self) -> int:
        """返回词表大小"""
        pass
    
    def save_vocab(self, path: str) -> None:
        """保存词表"""
        pass
    
    def load_vocab(self, path: str) -> None:
        """加载词表"""
        pass
```

### 核心实现

#### 1. config.py - 配置类

```python
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class FeatureConfig:
    """特征工程配置"""
    
    # 序列配置
    max_seq_length: int = 1024
    max_items_per_user: int = 500
    
    # 词表配置
    vocab_size: int = 500000
    min_token_freq: int = 5
    
    # 特殊 Token
    pad_token: str = "[PAD]"
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    mask_token: str = "[MASK]"
    unk_token: str = "[UNK]"
    
    # Token ID 预留
    pad_token_id: int = 0
    cls_token_id: int = 1
    sep_token_id: int = 2
    mask_token_id: int = 3
    unk_token_id: int = 4
    special_token_count: int = 5
    
    # 时间分桶
    time_buckets: List[str] = field(default_factory=lambda: [
        "night",      # 0-6
        "morning",    # 6-12
        "afternoon",  # 12-18
        "evening",    # 18-24
    ])
    
    # 行为类型
    action_types: List[str] = field(default_factory=lambda: [
        "view", "click", "like", "dislike", 
        "favorite", "share", "comment", "buy", "rate",
    ])
    
    # 设备类型
    device_types: List[str] = field(default_factory=lambda: [
        "mobile", "desktop", "tablet", "tv",
    ])
    
    # Token 类型 ID
    token_type_ids: Dict[str, int] = field(default_factory=lambda: {
        "USER": 0,
        "ITEM": 1,
        "ACTION": 2,
        "CONTEXT": 3,
    })
```

#### 2. vocabulary.py - 词表管理

```python
import json
from collections import Counter
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class Vocabulary:
    """
    词表管理器
    
    负责 Token 到 ID 的双向映射
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
        # Token <-> ID 映射
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        
        # 初始化特殊 Token
        self._init_special_tokens()
        
        # Token 频率统计（用于构建词表）
        self.token_counts: Counter = Counter()
    
    def _init_special_tokens(self):
        """初始化特殊 Token"""
        special_tokens = [
            self.config.pad_token,
            self.config.cls_token,
            self.config.sep_token,
            self.config.mask_token,
            self.config.unk_token,
        ]
        
        for idx, token in enumerate(special_tokens):
            self.token2id[token] = idx
            self.id2token[idx] = token
    
    def add_token(self, token: str) -> int:
        """添加 Token 到词表"""
        if token in self.token2id:
            return self.token2id[token]
        
        idx = len(self.token2id)
        if idx >= self.config.vocab_size:
            logger.warning(f"Vocabulary full, using UNK for: {token}")
            return self.config.unk_token_id
        
        self.token2id[token] = idx
        self.id2token[idx] = token
        return idx
    
    def encode(self, token: str) -> int:
        """Token -> ID"""
        return self.token2id.get(token, self.config.unk_token_id)
    
    def encode_batch(self, tokens: List[str]) -> List[int]:
        """批量 Token -> ID"""
        return [self.encode(token) for token in tokens]
    
    def decode(self, idx: int) -> str:
        """ID -> Token"""
        return self.id2token.get(idx, self.config.unk_token)
    
    def decode_batch(self, ids: List[int]) -> List[str]:
        """批量 ID -> Token"""
        return [self.decode(idx) for idx in ids]
    
    def build_from_data(self, data_path: str, min_freq: int = None):
        """
        从数据构建词表
        
        Args:
            data_path: 数据文件路径
            min_freq: 最小频率阈值
        """
        min_freq = min_freq or self.config.min_token_freq
        
        logger.info(f"Building vocabulary from {data_path}")
        
        # 统计 Token 频率
        with open(data_path, 'r') as f:
            for line in f:
                event = json.loads(line)
                tokens = self._extract_tokens(event)
                self.token_counts.update(tokens)
        
        # 按频率筛选
        for token, count in self.token_counts.most_common():
            if count < min_freq:
                break
            if len(self.token2id) >= self.config.vocab_size:
                break
            self.add_token(token)
        
        logger.info(f"Vocabulary built: {len(self.token2id)} tokens")
    
    def _extract_tokens(self, event: dict) -> List[str]:
        """从事件中提取所有 Token"""
        tokens = []
        
        # 行为 Token
        if "action" in event:
            tokens.append(f"ACTION_{event['action']}")
        
        # 物品 Token（假设已有语义 ID）
        if "semantic_id" in event:
            l1, l2, l3 = event["semantic_id"]
            tokens.append(f"ITEM_{l1}_{l2}_{l3}")
        
        # 时间 Token
        if "timestamp" in event:
            hour = self._get_hour(event["timestamp"])
            time_bucket = self._get_time_bucket(hour)
            tokens.append(f"TIME_{time_bucket}")
        
        # 设备 Token
        if "device" in event:
            tokens.append(f"DEVICE_{event['device']}")
        
        return tokens
    
    def _get_hour(self, timestamp: int) -> int:
        """从时间戳获取小时"""
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).hour
    
    def _get_time_bucket(self, hour: int) -> str:
        """获取时间分桶"""
        if 0 <= hour < 6:
            return "night"
        elif 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        else:
            return "evening"
    
    def save(self, path: str):
        """保存词表"""
        vocab_data = {
            "token2id": self.token2id,
            "config": {
                "vocab_size": self.config.vocab_size,
                "special_tokens": {
                    "pad": self.config.pad_token,
                    "cls": self.config.cls_token,
                    "sep": self.config.sep_token,
                    "mask": self.config.mask_token,
                    "unk": self.config.unk_token,
                }
            }
        }
        
        with open(path, 'w') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Vocabulary saved to {path}")
    
    def load(self, path: str):
        """加载词表"""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.token2id = vocab_data["token2id"]
        self.id2token = {int(v): k for k, v in self.token2id.items()}
        
        logger.info(f"Vocabulary loaded from {path}: {len(self.token2id)} tokens")
    
    def __len__(self) -> int:
        return len(self.token2id)
```

#### 3. event_parser.py - 事件解析器

```python
from typing import Dict, List, Any, Tuple
from datetime import datetime

class EventParser:
    """
    事件解析器
    
    将原始事件转换为 Token 列表
    """
    
    def __init__(self, config: FeatureConfig, semantic_id_encoder = None):
        self.config = config
        self.semantic_id_encoder = semantic_id_encoder
    
    def parse_event(self, event: Dict[str, Any]) -> List[str]:
        """
        解析单个事件为 Token 列表
        
        Args:
            event: 原始事件
                {
                    "item_id": "movie_001",
                    "action": "click",
                    "timestamp": 1704067200,
                    "device": "mobile",
                    "context": {"source": "search"},
                    "item_features": [...],  # 可选，用于生成语义 ID
                }
        
        Returns:
            Token 列表: ["ACTION_click", "ITEM_256_1234_8901", "TIME_evening", ...]
        """
        tokens = []
        
        # 1. 行为 Token
        action = event.get("action", "view")
        tokens.append(f"ACTION_{action}")
        
        # 2. 物品 Token（语义 ID）
        item_token = self._get_item_token(event)
        tokens.append(item_token)
        
        # 3. 时间 Token
        if "timestamp" in event:
            time_tokens = self._get_time_tokens(event["timestamp"])
            tokens.extend(time_tokens)
        
        # 4. 设备 Token
        if "device" in event:
            tokens.append(f"DEVICE_{event['device']}")
        
        # 5. 上下文 Token
        if "context" in event:
            context_tokens = self._get_context_tokens(event["context"])
            tokens.extend(context_tokens)
        
        return tokens
    
    def _get_item_token(self, event: Dict[str, Any]) -> str:
        """获取物品的语义 ID Token"""
        # 如果事件已包含语义 ID
        if "semantic_id" in event:
            l1, l2, l3 = event["semantic_id"]
            return f"ITEM_{l1}_{l2}_{l3}"
        
        # 如果有物品特征，使用编码器生成语义 ID
        if "item_features" in event and self.semantic_id_encoder:
            import torch
            features = torch.tensor(event["item_features"]).unsqueeze(0)
            l1, l2, l3 = self.semantic_id_encoder.encode(features)
            return f"ITEM_{l1.item()}_{l2.item()}_{l3.item()}"
        
        # 回退：使用物品 ID 的哈希
        item_id = event.get("item_id", "unknown")
        hash_val = hash(item_id) % 1000000
        l1 = hash_val % 1024
        l2 = (hash_val // 1024) % 4096
        l3 = hash_val % 16384
        return f"ITEM_{l1}_{l2}_{l3}"
    
    def _get_time_tokens(self, timestamp: int) -> List[str]:
        """获取时间相关 Token"""
        tokens = []
        
        dt = datetime.fromtimestamp(timestamp)
        
        # 时间段
        hour = dt.hour
        if 0 <= hour < 6:
            tokens.append("TIME_night")
        elif 6 <= hour < 12:
            tokens.append("TIME_morning")
        elif 12 <= hour < 18:
            tokens.append("TIME_afternoon")
        else:
            tokens.append("TIME_evening")
        
        # 周末/工作日
        if dt.weekday() >= 5:
            tokens.append("TIME_weekend")
        else:
            tokens.append("TIME_weekday")
        
        return tokens
    
    def _get_context_tokens(self, context: Dict[str, Any]) -> List[str]:
        """获取上下文 Token"""
        tokens = []
        
        # 来源
        if "source" in context:
            tokens.append(f"CONTEXT_{context['source']}")
        
        # 位置
        if "position" in context:
            pos = context["position"]
            if pos <= 3:
                tokens.append("CONTEXT_top")
            elif pos <= 10:
                tokens.append("CONTEXT_mid")
            else:
                tokens.append("CONTEXT_bottom")
        
        return tokens
    
    def parse_semantic_id(self, token: str) -> Tuple[int, int, int]:
        """
        解析物品 Token 为语义 ID
        
        Args:
            token: "ITEM_256_1234_8901"
        
        Returns:
            (256, 1234, 8901)
        """
        if not token.startswith("ITEM_"):
            raise ValueError(f"Invalid item token: {token}")
        
        parts = token[5:].split("_")
        return int(parts[0]), int(parts[1]), int(parts[2])
```

#### 4. tokenizer.py - Token 化器

```python
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TokenizedSequence:
    """Token 化后的序列"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_types: torch.Tensor
    positions: torch.Tensor
    semantic_ids: List[torch.Tensor]  # [L1_ids, L2_ids, L3_ids]
    labels: Optional[torch.Tensor] = None


class RecommendTokenizer:
    """
    推荐系统 Token 化器
    
    将用户行为事件序列转换为模型输入格式
    """
    
    def __init__(self, config: FeatureConfig, vocab: Vocabulary = None):
        self.config = config
        self.vocab = vocab or Vocabulary(config)
        self.event_parser = EventParser(config)
    
    def tokenize_events(
        self,
        events: List[Dict[str, Any]],
        max_length: int = None,
        add_special_tokens: bool = True,
    ) -> TokenizedSequence:
        """
        将事件列表转换为 Token 序列
        
        Args:
            events: 用户行为事件列表（按时间排序）
            max_length: 最大序列长度
            add_special_tokens: 是否添加 [CLS] 和 [SEP]
        
        Returns:
            TokenizedSequence 对象
        """
        max_length = max_length or self.config.max_seq_length
        
        # 解析事件为 Token
        all_tokens = []
        all_token_types = []
        all_semantic_ids = {"l1": [], "l2": [], "l3": []}
        
        if add_special_tokens:
            all_tokens.append(self.config.cls_token)
            all_token_types.append(self.config.token_type_ids["CONTEXT"])
            all_semantic_ids["l1"].append(0)
            all_semantic_ids["l2"].append(0)
            all_semantic_ids["l3"].append(0)
        
        for event in events:
            tokens = self.event_parser.parse_event(event)
            
            for token in tokens:
                all_tokens.append(token)
                
                # 确定 Token 类型
                if token.startswith("ACTION_"):
                    all_token_types.append(self.config.token_type_ids["ACTION"])
                    all_semantic_ids["l1"].append(0)
                    all_semantic_ids["l2"].append(0)
                    all_semantic_ids["l3"].append(0)
                elif token.startswith("ITEM_"):
                    all_token_types.append(self.config.token_type_ids["ITEM"])
                    l1, l2, l3 = self.event_parser.parse_semantic_id(token)
                    all_semantic_ids["l1"].append(l1)
                    all_semantic_ids["l2"].append(l2)
                    all_semantic_ids["l3"].append(l3)
                else:
                    all_token_types.append(self.config.token_type_ids["CONTEXT"])
                    all_semantic_ids["l1"].append(0)
                    all_semantic_ids["l2"].append(0)
                    all_semantic_ids["l3"].append(0)
        
        if add_special_tokens:
            all_tokens.append(self.config.sep_token)
            all_token_types.append(self.config.token_type_ids["CONTEXT"])
            all_semantic_ids["l1"].append(0)
            all_semantic_ids["l2"].append(0)
            all_semantic_ids["l3"].append(0)
        
        # 截断或填充
        seq_len = len(all_tokens)
        if seq_len > max_length:
            # 截断（保留最近的事件）
            all_tokens = all_tokens[-max_length:]
            all_token_types = all_token_types[-max_length:]
            all_semantic_ids["l1"] = all_semantic_ids["l1"][-max_length:]
            all_semantic_ids["l2"] = all_semantic_ids["l2"][-max_length:]
            all_semantic_ids["l3"] = all_semantic_ids["l3"][-max_length:]
            seq_len = max_length
        
        # 创建注意力掩码
        attention_mask = [1] * seq_len
        
        # 填充
        padding_length = max_length - seq_len
        if padding_length > 0:
            all_tokens.extend([self.config.pad_token] * padding_length)
            all_token_types.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)
            all_semantic_ids["l1"].extend([0] * padding_length)
            all_semantic_ids["l2"].extend([0] * padding_length)
            all_semantic_ids["l3"].extend([0] * padding_length)
        
        # 转换为 ID
        input_ids = self.vocab.encode_batch(all_tokens)
        positions = list(range(max_length))
        
        return TokenizedSequence(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            token_types=torch.tensor(all_token_types, dtype=torch.long),
            positions=torch.tensor(positions, dtype=torch.long),
            semantic_ids=[
                torch.tensor(all_semantic_ids["l1"], dtype=torch.long),
                torch.tensor(all_semantic_ids["l2"], dtype=torch.long),
                torch.tensor(all_semantic_ids["l3"], dtype=torch.long),
            ],
        )
    
    def build_training_sample(
        self,
        events: List[Dict[str, Any]],
        target_item: Dict[str, Any],
    ) -> TokenizedSequence:
        """
        构建训练样本
        
        Args:
            events: 用户历史事件（输入）
            target_item: 目标物品（标签）
        
        Returns:
            包含 labels 的 TokenizedSequence
        """
        # Token 化输入序列
        result = self.tokenize_events(events)
        
        # 生成标签（目标物品的语义 ID）
        target_tokens = self.event_parser.parse_event(target_item)
        item_token = [t for t in target_tokens if t.startswith("ITEM_")][0]
        l1, l2, l3 = self.event_parser.parse_semantic_id(item_token)
        
        # 标签：shifted right (预测下一个 token)
        labels = torch.full((len(result.input_ids),), -100, dtype=torch.long)
        labels[-1] = self.vocab.encode(item_token)
        
        result.labels = labels
        
        return result
    
    def get_vocab_size(self) -> int:
        """返回词表大小"""
        return len(self.vocab)
    
    def save_vocab(self, path: str) -> None:
        """保存词表"""
        self.vocab.save(path)
    
    def load_vocab(self, path: str) -> None:
        """加载词表"""
        self.vocab.load(path)
```

#### 5. sequence_builder.py - 序列构建器

```python
from typing import List, Dict, Any, Iterator
import random

class SequenceBuilder:
    """
    序列构建器
    
    从用户行为日志构建训练序列
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def build_sequences(
        self,
        user_events: List[Dict[str, Any]],
        window_size: int = 100,
        stride: int = 50,
    ) -> Iterator[Dict[str, Any]]:
        """
        滑动窗口构建序列
        
        Args:
            user_events: 用户的所有事件（按时间排序）
            window_size: 窗口大小
            stride: 滑动步长
        
        Yields:
            训练样本: {"events": [...], "target": {...}}
        """
        if len(user_events) < 2:
            return
        
        for i in range(0, len(user_events) - 1, stride):
            end = min(i + window_size, len(user_events) - 1)
            
            # 输入序列
            input_events = user_events[i:end]
            
            # 目标：下一个事件
            target_event = user_events[end]
            
            yield {
                "events": input_events,
                "target": target_event,
            }
    
    def generate_negative_samples(
        self,
        positive_item: Dict[str, Any],
        item_pool: List[Dict[str, Any]],
        num_negatives: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        生成负样本
        
        Args:
            positive_item: 正样本物品
            item_pool: 候选物品池
            num_negatives: 负样本数量
        
        Returns:
            负样本列表
        """
        positive_id = positive_item.get("item_id")
        
        # 过滤掉正样本
        candidates = [item for item in item_pool if item.get("item_id") != positive_id]
        
        # 随机采样
        if len(candidates) <= num_negatives:
            return candidates
        
        return random.sample(candidates, num_negatives)
```

#### 6. spark_pipeline.py - Spark 大规模处理

```python
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
import logging

logger = logging.getLogger(__name__)

class SparkFeaturePipeline:
    """
    Spark 大规模特征处理 Pipeline
    
    处理 TB 级用户行为日志
    """
    
    def __init__(self, app_name: str = "RecommendFeatureEngineering"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.shuffle.partitions", 200) \
            .getOrCreate()
    
    def process_raw_logs(
        self,
        input_path: str,
        output_path: str,
        semantic_id_mapping_path: str = None,
    ):
        """
        处理原始日志
        
        Args:
            input_path: 原始日志路径 (JSON/Parquet)
            output_path: 输出路径
            semantic_id_mapping_path: 物品语义 ID 映射表路径
        """
        logger.info(f"Processing raw logs from {input_path}")
        
        # 读取原始日志
        df = self.spark.read.json(input_path)
        
        # 加载语义 ID 映射
        if semantic_id_mapping_path:
            semantic_df = self.spark.read.parquet(semantic_id_mapping_path)
            df = df.join(semantic_df, on="item_id", how="left")
        
        # 添加时间特征
        df = self._add_time_features(df)
        
        # 按用户分组，构建序列
        sequences_df = self._build_user_sequences(df)
        
        # 保存
        sequences_df.write.parquet(output_path, mode="overwrite")
        
        logger.info(f"Processed data saved to {output_path}")
    
    def _add_time_features(self, df: DataFrame) -> DataFrame:
        """添加时间特征"""
        return df \
            .withColumn("hour", F.hour(F.from_unixtime("timestamp"))) \
            .withColumn("day_of_week", F.dayofweek(F.from_unixtime("timestamp"))) \
            .withColumn(
                "time_bucket",
                F.when(F.col("hour") < 6, "night")
                 .when(F.col("hour") < 12, "morning")
                 .when(F.col("hour") < 18, "afternoon")
                 .otherwise("evening")
            ) \
            .withColumn(
                "is_weekend",
                F.when(F.col("day_of_week").isin([1, 7]), True).otherwise(False)
            )
    
    def _build_user_sequences(self, df: DataFrame) -> DataFrame:
        """构建用户序列"""
        from pyspark.sql.window import Window
        
        # 按用户和时间排序
        window = Window.partitionBy("user_id").orderBy("timestamp")
        
        # 添加序列位置
        df = df.withColumn("seq_position", F.row_number().over(window))
        
        # 收集为数组
        sequences_df = df.groupBy("user_id").agg(
            F.collect_list(
                F.struct(
                    "item_id", "action", "timestamp",
                    "semantic_l1", "semantic_l2", "semantic_l3",
                    "time_bucket", "device",
                )
            ).alias("events"),
            F.count("*").alias("seq_length"),
        )
        
        # 过滤太短的序列
        sequences_df = sequences_df.filter(F.col("seq_length") >= 5)
        
        return sequences_df
    
    def build_vocabulary(
        self,
        input_path: str,
        output_path: str,
        min_freq: int = 5,
    ):
        """构建词表"""
        df = self.spark.read.parquet(input_path)
        
        # 展开事件
        events_df = df.select(F.explode("events").alias("event"))
        
        # 提取 Token
        tokens_df = events_df.select(
            F.concat(F.lit("ACTION_"), F.col("event.action")).alias("token")
        ).union(
            events_df.select(
                F.concat(
                    F.lit("ITEM_"),
                    F.col("event.semantic_l1"), F.lit("_"),
                    F.col("event.semantic_l2"), F.lit("_"),
                    F.col("event.semantic_l3"),
                ).alias("token")
            )
        ).union(
            events_df.select(
                F.concat(F.lit("TIME_"), F.col("event.time_bucket")).alias("token")
            )
        )
        
        # 统计频率
        vocab_df = tokens_df \
            .groupBy("token") \
            .count() \
            .filter(F.col("count") >= min_freq) \
            .orderBy(F.desc("count"))
        
        # 保存
        vocab_df.write.csv(output_path, header=True, mode="overwrite")
        
        logger.info(f"Vocabulary saved to {output_path}")
    
    def stop(self):
        """停止 Spark"""
        self.spark.stop()
```

### 测试用例

```python
def test_tokenizer():
    from algorithm.feature_engineering.config import FeatureConfig
    from algorithm.feature_engineering.tokenizer import RecommendTokenizer
    
    config = FeatureConfig()
    tokenizer = RecommendTokenizer(config)
    
    # 测试事件
    events = [
        {"item_id": "movie_001", "action": "click", "timestamp": 1704067200, "semantic_id": (256, 1234, 8901)},
        {"item_id": "movie_002", "action": "view", "timestamp": 1704067260, "semantic_id": (256, 1234, 7652)},
        {"item_id": "movie_003", "action": "like", "timestamp": 1704067320, "semantic_id": (256, 1235, 4521)},
    ]
    
    # Token 化
    result = tokenizer.tokenize_events(events, max_length=50)
    
    assert result.input_ids.shape[0] == 50
    assert result.attention_mask.shape[0] == 50
    assert result.token_types.shape[0] == 50
    assert len(result.semantic_ids) == 3
    
    print("Input IDs:", result.input_ids[:20])
    print("Attention Mask:", result.attention_mask[:20])
    print("Token Types:", result.token_types[:20])
    
    print("All tokenizer tests passed!")
```

## 注意事项

1. **序列长度**: 注意处理超长序列的截断策略（保留最近的事件）
2. **语义 ID**: 确保物品的语义 ID 一致性（使用 Person A 的编码器）
3. **时间对齐**: 确保事件按时间戳正确排序
4. **大规模处理**: 使用 Spark 处理大规模数据，避免内存溢出

## 输出要求

请输出完整的可运行代码，包含：
1. 所有 Python 文件
2. 详细的中文注释
3. 数据处理脚本
4. 使用示例

确保代码遵循 `algorithm/interfaces.py` 中定义的接口。

