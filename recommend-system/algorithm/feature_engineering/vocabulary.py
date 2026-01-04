"""
词表管理模块

负责管理 Token 到 ID 的双向映射，包括：
- 特殊 Token 初始化
- Token 添加与编码
- 批量编解码
- 从数据构建词表
- 词表的保存与加载
"""

import json
import logging
from collections import Counter
from typing import List, Dict, Optional, Union, Iterator, Any
from datetime import datetime
import os

from algorithm.feature_engineering.config import FeatureConfig

logger = logging.getLogger(__name__)


class Vocabulary:
    """
    词表管理器
    
    负责维护 Token 与 ID 之间的双向映射，支持从数据构建词表、
    增量添加 Token、批量编解码等功能。
    
    Attributes:
        config: 特征工程配置
        token2id: Token 到 ID 的映射字典
        id2token: ID 到 Token 的映射字典
        token_counts: Token 频率统计
        
    Example:
        >>> config = FeatureConfig()
        >>> vocab = Vocabulary(config)
        >>> token_id = vocab.add_token("ACTION_click")
        >>> vocab.encode("ACTION_click")
        5
        >>> vocab.decode(5)
        'ACTION_click'
    """
    
    def __init__(self, config: FeatureConfig):
        """
        初始化词表管理器
        
        Args:
            config: 特征工程配置
        """
        self.config = config
        
        # Token <-> ID 双向映射
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        
        # Token 频率统计（用于构建词表时的频率筛选）
        self.token_counts: Counter = Counter()
        
        # 词表元数据
        self.metadata: Dict[str, Any] = {
            "version": "1.0.0",
            "created_at": None,
            "updated_at": None,
            "source": None,
        }
        
        # 初始化特殊 Token
        self._init_special_tokens()
    
    def _init_special_tokens(self) -> None:
        """
        初始化特殊 Token
        
        将 PAD、CLS、SEP、MASK、UNK 等特殊 Token 添加到词表开头，
        确保它们拥有固定的 ID。
        """
        special_tokens = self.config.get_special_tokens()
        
        for idx, token in enumerate(special_tokens):
            self.token2id[token] = idx
            self.id2token[idx] = token
        
        logger.debug(f"初始化 {len(special_tokens)} 个特殊 Token")
    
    def add_token(self, token: str) -> int:
        """
        添加单个 Token 到词表
        
        如果 Token 已存在，返回其 ID；如果词表已满，返回 UNK ID。
        
        Args:
            token: 要添加的 Token
            
        Returns:
            Token 对应的 ID
            
        Example:
            >>> vocab = Vocabulary(FeatureConfig())
            >>> vocab.add_token("ACTION_click")
            5
        """
        # 如果 Token 已存在，直接返回其 ID
        if token in self.token2id:
            return self.token2id[token]
        
        # 检查词表是否已满
        idx = len(self.token2id)
        if idx >= self.config.vocab_size:
            logger.warning(f"词表已满，使用 UNK 替代: {token}")
            return self.config.unk_token_id
        
        # 添加新 Token
        self.token2id[token] = idx
        self.id2token[idx] = token
        
        return idx
    
    def add_tokens(self, tokens: List[str]) -> List[int]:
        """
        批量添加 Token 到词表
        
        Args:
            tokens: Token 列表
            
        Returns:
            Token ID 列表
        """
        return [self.add_token(token) for token in tokens]
    
    def encode(self, token: str) -> int:
        """
        将单个 Token 编码为 ID
        
        如果 Token 不在词表中，返回 UNK ID。
        
        Args:
            token: 要编码的 Token
            
        Returns:
            Token ID
            
        Example:
            >>> vocab.encode("ACTION_click")
            5
            >>> vocab.encode("UNKNOWN_TOKEN")
            4  # UNK ID
        """
        return self.token2id.get(token, self.config.unk_token_id)
    
    def encode_batch(self, tokens: List[str]) -> List[int]:
        """
        批量将 Token 编码为 ID
        
        Args:
            tokens: Token 列表
            
        Returns:
            Token ID 列表
            
        Example:
            >>> vocab.encode_batch(["ACTION_click", "ITEM_1_2_3"])
            [5, 6]
        """
        return [self.encode(token) for token in tokens]
    
    def decode(self, idx: int) -> str:
        """
        将 ID 解码为 Token
        
        如果 ID 不在词表中，返回 UNK Token。
        
        Args:
            idx: Token ID
            
        Returns:
            Token 字符串
            
        Example:
            >>> vocab.decode(5)
            'ACTION_click'
        """
        return self.id2token.get(idx, self.config.unk_token)
    
    def decode_batch(self, ids: List[int]) -> List[str]:
        """
        批量将 ID 解码为 Token
        
        Args:
            ids: Token ID 列表
            
        Returns:
            Token 字符串列表
        """
        return [self.decode(idx) for idx in ids]
    
    def contains(self, token: str) -> bool:
        """
        检查 Token 是否在词表中
        
        Args:
            token: 要检查的 Token
            
        Returns:
            是否存在
        """
        return token in self.token2id
    
    def update_token_counts(self, tokens: List[str]) -> None:
        """
        更新 Token 频率统计
        
        用于从数据构建词表时统计 Token 出现次数。
        
        Args:
            tokens: Token 列表
        """
        self.token_counts.update(tokens)
    
    def build_from_counts(self, min_freq: Optional[int] = None) -> int:
        """
        从频率统计构建词表
        
        根据已收集的 Token 频率，按频率降序添加 Token 到词表，
        忽略低于 min_freq 的低频 Token。
        
        Args:
            min_freq: 最小频率阈值，默认使用配置中的值
            
        Returns:
            实际添加的 Token 数量（不含特殊 Token）
        """
        min_freq = min_freq or self.config.min_token_freq
        added_count = 0
        
        # 按频率降序排列
        for token, count in self.token_counts.most_common():
            # 跳过低频 Token
            if count < min_freq:
                break
            
            # 检查词表容量
            if len(self.token2id) >= self.config.vocab_size:
                logger.warning(f"词表已满，停止添加。剩余 Token 数: {len(self.token_counts) - added_count}")
                break
            
            # 跳过已存在的 Token（如特殊 Token）
            if token not in self.token2id:
                self.add_token(token)
                added_count += 1
        
        logger.info(f"从频率统计构建词表完成，添加 {added_count} 个 Token")
        return added_count
    
    def build_from_data(
        self, 
        data_path: str, 
        min_freq: Optional[int] = None,
        max_lines: Optional[int] = None,
    ) -> None:
        """
        从数据文件构建词表
        
        支持 JSON Lines 格式的数据文件，每行是一个事件 JSON 对象。
        
        Args:
            data_path: 数据文件路径
            min_freq: 最小频率阈值
            max_lines: 最大读取行数（用于采样）
            
        Example:
            >>> vocab.build_from_data("train_data.jsonl", min_freq=5)
        """
        min_freq = min_freq or self.config.min_token_freq
        
        logger.info(f"开始从 {data_path} 构建词表")
        
        # 重置频率统计
        self.token_counts.clear()
        
        # 读取数据并统计 Token 频率
        line_count = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_lines and line_count >= max_lines:
                    break
                
                try:
                    event = json.loads(line.strip())
                    tokens = self._extract_tokens_from_event(event)
                    self.token_counts.update(tokens)
                    line_count += 1
                except json.JSONDecodeError:
                    logger.warning(f"跳过无效 JSON 行: {line[:50]}...")
                    continue
        
        logger.info(f"读取 {line_count} 行，统计到 {len(self.token_counts)} 个不同 Token")
        
        # 从频率统计构建词表
        added_count = self.build_from_counts(min_freq)
        
        # 更新元数据
        self.metadata["created_at"] = datetime.now().isoformat()
        self.metadata["source"] = data_path
        
        logger.info(f"词表构建完成，总大小: {len(self.token2id)}")
    
    def build_from_iterator(
        self,
        events_iterator: Iterator[Dict[str, Any]],
        min_freq: Optional[int] = None,
    ) -> None:
        """
        从事件迭代器构建词表
        
        适用于大规模数据处理或流式数据场景。
        
        Args:
            events_iterator: 事件字典的迭代器
            min_freq: 最小频率阈值
        """
        min_freq = min_freq or self.config.min_token_freq
        
        # 重置频率统计
        self.token_counts.clear()
        
        # 统计 Token 频率
        event_count = 0
        for event in events_iterator:
            tokens = self._extract_tokens_from_event(event)
            self.token_counts.update(tokens)
            event_count += 1
            
            if event_count % 100000 == 0:
                logger.info(f"已处理 {event_count} 个事件")
        
        logger.info(f"处理完成，共 {event_count} 个事件，{len(self.token_counts)} 个不同 Token")
        
        # 从频率统计构建词表
        self.build_from_counts(min_freq)
        
        # 更新元数据
        self.metadata["created_at"] = datetime.now().isoformat()
        self.metadata["source"] = "iterator"
    
    def _extract_tokens_from_event(self, event: Dict[str, Any]) -> List[str]:
        """
        从事件中提取所有 Token
        
        这是一个简化版本的 Token 提取，完整版本在 EventParser 中实现。
        
        Args:
            event: 事件字典
            
        Returns:
            Token 列表
        """
        tokens = []
        
        # 行为 Token
        if "action" in event:
            tokens.append(f"ACTION_{event['action']}")
        
        # 物品语义 ID Token
        if "semantic_id" in event:
            sid = event["semantic_id"]
            if isinstance(sid, (list, tuple)) and len(sid) == 3:
                l1, l2, l3 = sid
                tokens.append(f"ITEM_{l1}_{l2}_{l3}")
        
        # 时间 Token
        if "timestamp" in event:
            hour = self._get_hour_from_timestamp(event["timestamp"])
            time_bucket = self.config.get_time_bucket(hour)
            tokens.append(f"TIME_{time_bucket}")
        
        # 设备 Token
        if "device" in event:
            tokens.append(f"DEVICE_{event['device']}")
        
        # 上下文 Token
        if "context" in event and isinstance(event["context"], dict):
            context = event["context"]
            if "source" in context:
                tokens.append(f"CONTEXT_{context['source']}")
        
        return tokens
    
    def _get_hour_from_timestamp(self, timestamp: Union[int, float]) -> int:
        """
        从时间戳获取小时
        
        Args:
            timestamp: Unix 时间戳（秒）
            
        Returns:
            小时（0-23）
        """
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.hour
        except (OSError, OverflowError, ValueError):
            return 12  # 默认返回中午
    
    def save(self, path: str) -> None:
        """
        保存词表到文件
        
        保存内容包括：
        - Token 到 ID 的映射
        - 配置信息
        - 元数据
        
        Args:
            path: 保存路径（JSON 格式）
            
        Example:
            >>> vocab.save("vocab.json")
        """
        vocab_data = {
            "token2id": self.token2id,
            "metadata": self.metadata,
            "config": {
                "vocab_size": self.config.vocab_size,
                "min_token_freq": self.config.min_token_freq,
                "special_tokens": {
                    "pad": self.config.pad_token,
                    "cls": self.config.cls_token,
                    "sep": self.config.sep_token,
                    "mask": self.config.mask_token,
                    "unk": self.config.unk_token,
                },
                "special_token_ids": {
                    "pad": self.config.pad_token_id,
                    "cls": self.config.cls_token_id,
                    "sep": self.config.sep_token_id,
                    "mask": self.config.mask_token_id,
                    "unk": self.config.unk_token_id,
                },
            },
            "statistics": {
                "total_tokens": len(self.token2id),
                "unique_token_count": len(self.token_counts),
            },
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"词表已保存到 {path}，共 {len(self.token2id)} 个 Token")
    
    def load(self, path: str) -> None:
        """
        从文件加载词表
        
        Args:
            path: 词表文件路径
            
        Example:
            >>> vocab.load("vocab.json")
        """
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # 加载 Token 映射
        self.token2id = vocab_data["token2id"]
        # 注意：JSON 的键必须是字符串，需要转换 ID 为整数
        self.id2token = {int(v): k for k, v in self.token2id.items()}
        
        # 加载元数据
        if "metadata" in vocab_data:
            self.metadata = vocab_data["metadata"]
        
        logger.info(f"词表已从 {path} 加载，共 {len(self.token2id)} 个 Token")
    
    def get_token_frequency(self, token: str) -> int:
        """
        获取 Token 的出现频率
        
        Args:
            token: Token 字符串
            
        Returns:
            出现次数，如果未统计过返回 0
        """
        return self.token_counts.get(token, 0)
    
    def get_most_common_tokens(self, n: int = 100) -> List[tuple]:
        """
        获取出现频率最高的 Token
        
        Args:
            n: 返回的 Token 数量
            
        Returns:
            (Token, 频率) 元组列表
        """
        return self.token_counts.most_common(n)
    
    def get_action_tokens(self) -> List[str]:
        """
        获取所有行为类型 Token
        
        Returns:
            行为 Token 列表
        """
        return [token for token in self.token2id.keys() if token.startswith("ACTION_")]
    
    def get_item_tokens(self) -> List[str]:
        """
        获取所有物品 Token
        
        Returns:
            物品 Token 列表
        """
        return [token for token in self.token2id.keys() if token.startswith("ITEM_")]
    
    def __len__(self) -> int:
        """返回词表大小"""
        return len(self.token2id)
    
    def __contains__(self, token: str) -> bool:
        """检查 Token 是否在词表中"""
        return token in self.token2id
    
    def __repr__(self) -> str:
        return (
            f"Vocabulary(\n"
            f"  size={len(self.token2id)},\n"
            f"  special_tokens={len(self.config.get_special_tokens())},\n"
            f"  action_tokens={len(self.get_action_tokens())},\n"
            f"  item_tokens={len(self.get_item_tokens())}\n"
            f")"
        )

