"""
序列构建器模块

负责从用户行为日志构建训练序列，包括：
- 滑动窗口序列构建
- 负采样生成
- 数据增强
- 批次构建
"""

import logging
import random
from typing import List, Dict, Any, Iterator, Tuple, Optional
from collections import defaultdict
from datetime import datetime

from algorithm.feature_engineering.config import FeatureConfig

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """
    序列构建器
    
    从用户行为日志构建训练序列，支持滑动窗口、负采样、数据增强等功能。
    
    Attributes:
        config: 特征工程配置
        
    Example:
        >>> config = FeatureConfig()
        >>> builder = SequenceBuilder(config)
        >>> user_events = [
        ...     {"item_id": "item1", "action": "click", "timestamp": 1000},
        ...     {"item_id": "item2", "action": "view", "timestamp": 2000},
        ...     {"item_id": "item3", "action": "buy", "timestamp": 3000},
        ... ]
        >>> for sample in builder.build_sequences(user_events, window_size=2):
        ...     print(sample)
    """
    
    def __init__(self, config: FeatureConfig):
        """
        初始化序列构建器
        
        Args:
            config: 特征工程配置
        """
        self.config = config
        
        # 物品池（用于负采样）
        self._item_pool: List[Dict[str, Any]] = []
        self._item_popularity: Dict[str, int] = defaultdict(int)
    
    def build_sequences(
        self,
        user_events: List[Dict[str, Any]],
        window_size: int = 100,
        stride: int = 50,
        min_events: int = 2,
    ) -> Iterator[Dict[str, Any]]:
        """
        使用滑动窗口构建训练序列
        
        将用户的事件序列按滑动窗口切分，每个窗口的最后一个事件作为预测目标，
        之前的事件作为输入序列。
        
        Args:
            user_events: 用户的所有事件（应按时间戳排序）
            window_size: 窗口大小（输入序列的最大长度）
            stride: 滑动步长
            min_events: 最小事件数量，少于此数量的序列将被过滤
            
        Yields:
            训练样本字典: {"events": [...], "target": {...}, "user_id": ...}
            
        Example:
            >>> for sample in builder.build_sequences(events, window_size=10, stride=5):
            ...     input_events = sample["events"]
            ...     target_item = sample["target"]
        """
        if len(user_events) < min_events:
            return
        
        # 确保事件按时间排序
        sorted_events = sorted(user_events, key=lambda x: x.get("timestamp", 0))
        
        # 滑动窗口遍历
        for i in range(0, len(sorted_events) - 1, stride):
            end_idx = min(i + window_size, len(sorted_events) - 1)
            
            # 输入序列（不包含最后一个事件）
            input_events = sorted_events[i:end_idx]
            
            # 目标事件（窗口的下一个事件）
            target_event = sorted_events[end_idx]
            
            # 过滤过短的序列
            if len(input_events) < min_events - 1:
                continue
            
            yield {
                "events": input_events,
                "target": target_event,
                "user_id": user_events[0].get("user_id"),
                "window_start": i,
                "window_end": end_idx,
            }
    
    def build_next_item_prediction_samples(
        self,
        user_events: List[Dict[str, Any]],
        max_history: int = 100,
    ) -> Iterator[Dict[str, Any]]:
        """
        构建下一物品预测样本
        
        对于用户历史中的每个物品，使用它之前的所有事件作为输入，
        该物品作为预测目标。这种方式可以充分利用用户的历史数据。
        
        Args:
            user_events: 用户事件列表（应按时间排序）
            max_history: 最大历史长度
            
        Yields:
            训练样本字典
        """
        if len(user_events) < 2:
            return
        
        sorted_events = sorted(user_events, key=lambda x: x.get("timestamp", 0))
        
        # 从第二个事件开始，每个事件都可以作为一个预测目标
        for i in range(1, len(sorted_events)):
            # 取前面的事件作为输入
            start_idx = max(0, i - max_history)
            input_events = sorted_events[start_idx:i]
            target_event = sorted_events[i]
            
            yield {
                "events": input_events,
                "target": target_event,
                "user_id": user_events[0].get("user_id"),
                "position": i,
            }
    
    def generate_negative_samples(
        self,
        positive_item: Dict[str, Any],
        item_pool: Optional[List[Dict[str, Any]]] = None,
        num_negatives: int = 4,
        strategy: str = "uniform",
        exclude_items: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        生成负样本
        
        从物品池中采样与正样本不同的物品作为负样本。
        
        Args:
            positive_item: 正样本物品
            item_pool: 候选物品池，如果为 None 则使用内部物品池
            num_negatives: 负样本数量
            strategy: 采样策略
                - "uniform": 均匀随机采样
                - "popularity": 按热度采样（更难的负样本）
            exclude_items: 需要排除的物品 ID 列表
            
        Returns:
            负样本物品列表
        """
        pool = item_pool or self._item_pool
        if not pool:
            logger.warning("物品池为空，无法生成负样本")
            return []
        
        # 获取正样本和排除物品的 ID
        positive_id = positive_item.get("item_id")
        exclude_set = set(exclude_items or [])
        if positive_id:
            exclude_set.add(positive_id)
        
        # 过滤候选物品
        candidates = [item for item in pool if item.get("item_id") not in exclude_set]
        
        if not candidates:
            return []
        
        if len(candidates) <= num_negatives:
            return candidates
        
        # 根据策略采样
        if strategy == "popularity":
            return self._popularity_sample(candidates, num_negatives)
        else:
            return random.sample(candidates, num_negatives)
    
    def _popularity_sample(
        self, 
        candidates: List[Dict[str, Any]], 
        num_samples: int
    ) -> List[Dict[str, Any]]:
        """
        按热度采样负样本
        
        热门物品更可能被采样为负样本，这提供了更难的负样本，
        有助于模型学习更精细的区分能力。
        
        Args:
            candidates: 候选物品列表
            num_samples: 采样数量
            
        Returns:
            采样的物品列表
        """
        # 计算采样权重（基于热度）
        weights = []
        for item in candidates:
            item_id = item.get("item_id", "")
            popularity = self._item_popularity.get(item_id, 1)
            weights.append(popularity)
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight == 0:
            return random.sample(candidates, num_samples)
        
        weights = [w / total_weight for w in weights]
        
        # 按权重采样
        indices = random.choices(range(len(candidates)), weights=weights, k=num_samples)
        return [candidates[i] for i in set(indices)][:num_samples]
    
    def add_to_item_pool(self, items: List[Dict[str, Any]]) -> None:
        """
        将物品添加到物品池
        
        物品池用于负采样。
        
        Args:
            items: 物品列表
        """
        self._item_pool.extend(items)
        
        # 更新热度统计
        for item in items:
            item_id = item.get("item_id")
            if item_id:
                self._item_popularity[item_id] += 1
        
        logger.debug(f"物品池更新，当前大小: {len(self._item_pool)}")
    
    def clear_item_pool(self) -> None:
        """清空物品池"""
        self._item_pool.clear()
        self._item_popularity.clear()
    
    def build_contrastive_pairs(
        self,
        user_events: List[Dict[str, Any]],
        num_negatives: int = 4,
    ) -> Iterator[Dict[str, Any]]:
        """
        构建对比学习样本对
        
        为每个正样本生成对应的负样本，用于对比学习训练。
        
        Args:
            user_events: 用户事件列表
            num_negatives: 每个正样本的负样本数量
            
        Yields:
            对比学习样本: {"anchor": {...}, "positive": {...}, "negatives": [...]}
        """
        if len(user_events) < 2:
            return
        
        sorted_events = sorted(user_events, key=lambda x: x.get("timestamp", 0))
        
        # 收集用户交互过的物品 ID
        interacted_items = set()
        for event in sorted_events:
            item_id = event.get("item_id")
            if item_id:
                interacted_items.add(item_id)
        
        # 构建样本对
        for i in range(1, len(sorted_events)):
            anchor_events = sorted_events[:i]
            positive_event = sorted_events[i]
            
            # 生成负样本
            negatives = self.generate_negative_samples(
                positive_event,
                num_negatives=num_negatives,
                exclude_items=list(interacted_items),
            )
            
            yield {
                "anchor": anchor_events,
                "positive": positive_event,
                "negatives": negatives,
                "user_id": user_events[0].get("user_id"),
            }
    
    def augment_sequence(
        self,
        events: List[Dict[str, Any]],
        augmentation_type: str = "dropout",
        dropout_rate: float = 0.1,
        shuffle_window: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        序列数据增强
        
        支持多种数据增强策略：
        - dropout: 随机丢弃部分事件
        - shuffle: 在小窗口内随机打乱顺序
        - crop: 随机裁剪序列
        
        Args:
            events: 原始事件序列
            augmentation_type: 增强类型
            dropout_rate: dropout 概率
            shuffle_window: 打乱窗口大小
            
        Returns:
            增强后的事件序列
        """
        if len(events) < 2:
            return events.copy()
        
        if augmentation_type == "dropout":
            return self._augment_dropout(events, dropout_rate)
        elif augmentation_type == "shuffle":
            return self._augment_shuffle(events, shuffle_window)
        elif augmentation_type == "crop":
            return self._augment_crop(events)
        else:
            return events.copy()
    
    def _augment_dropout(
        self, 
        events: List[Dict[str, Any]], 
        dropout_rate: float
    ) -> List[Dict[str, Any]]:
        """随机丢弃部分事件"""
        result = []
        for event in events:
            if random.random() > dropout_rate:
                result.append(event.copy())
        
        # 确保至少保留一个事件
        if not result:
            result.append(random.choice(events).copy())
        
        return result
    
    def _augment_shuffle(
        self, 
        events: List[Dict[str, Any]], 
        window_size: int
    ) -> List[Dict[str, Any]]:
        """在小窗口内打乱顺序"""
        result = events.copy()
        
        for i in range(0, len(result) - window_size + 1, window_size):
            window = result[i:i + window_size]
            random.shuffle(window)
            result[i:i + window_size] = window
        
        return result
    
    def _augment_crop(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """随机裁剪序列"""
        if len(events) <= 2:
            return events.copy()
        
        # 随机选择裁剪比例（保留 50%-100%）
        keep_ratio = random.uniform(0.5, 1.0)
        keep_length = max(2, int(len(events) * keep_ratio))
        
        # 随机选择起始位置
        start_idx = random.randint(0, len(events) - keep_length)
        
        return events[start_idx:start_idx + keep_length]
    
    def filter_events_by_action(
        self,
        events: List[Dict[str, Any]],
        include_actions: Optional[List[str]] = None,
        exclude_actions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        按行为类型过滤事件
        
        Args:
            events: 事件列表
            include_actions: 包含的行为类型列表
            exclude_actions: 排除的行为类型列表
            
        Returns:
            过滤后的事件列表
        """
        result = events
        
        if include_actions:
            result = [e for e in result if e.get("action") in include_actions]
        
        if exclude_actions:
            result = [e for e in result if e.get("action") not in exclude_actions]
        
        return result
    
    def filter_events_by_time(
        self,
        events: List[Dict[str, Any]],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        按时间范围过滤事件
        
        Args:
            events: 事件列表
            start_time: 起始时间戳
            end_time: 结束时间戳
            
        Returns:
            过滤后的事件列表
        """
        result = events
        
        if start_time is not None:
            result = [e for e in result if e.get("timestamp", 0) >= start_time]
        
        if end_time is not None:
            result = [e for e in result if e.get("timestamp", float("inf")) <= end_time]
        
        return result
    
    def deduplicate_events(
        self,
        events: List[Dict[str, Any]],
        by: str = "item_id",
        keep: str = "last",
    ) -> List[Dict[str, Any]]:
        """
        去重事件
        
        Args:
            events: 事件列表
            by: 去重依据字段
            keep: 保留策略 ("first" 或 "last")
            
        Returns:
            去重后的事件列表
        """
        seen = {}
        
        for i, event in enumerate(events):
            key = event.get(by)
            if key is None:
                continue
            
            if keep == "first" and key not in seen:
                seen[key] = i
            elif keep == "last":
                seen[key] = i
        
        # 按原始顺序返回
        indices = sorted(seen.values())
        return [events[i] for i in indices]
    
    def compute_session_boundaries(
        self,
        events: List[Dict[str, Any]],
        session_gap_seconds: int = 1800,
    ) -> List[List[Dict[str, Any]]]:
        """
        计算会话边界，将事件分割为多个会话
        
        如果两个相邻事件的时间间隔超过阈值，则认为是不同的会话。
        
        Args:
            events: 事件列表（应按时间排序）
            session_gap_seconds: 会话间隔阈值（秒）
            
        Returns:
            会话列表，每个会话是一个事件列表
        """
        if not events:
            return []
        
        sorted_events = sorted(events, key=lambda x: x.get("timestamp", 0))
        
        sessions = []
        current_session = [sorted_events[0]]
        
        for i in range(1, len(sorted_events)):
            prev_time = sorted_events[i - 1].get("timestamp", 0)
            curr_time = sorted_events[i].get("timestamp", 0)
            
            if curr_time - prev_time > session_gap_seconds:
                # 新会话开始
                sessions.append(current_session)
                current_session = []
            
            current_session.append(sorted_events[i])
        
        # 添加最后一个会话
        if current_session:
            sessions.append(current_session)
        
        return sessions
    
    def build_session_aware_samples(
        self,
        user_events: List[Dict[str, Any]],
        session_gap_seconds: int = 1800,
        max_history_sessions: int = 5,
    ) -> Iterator[Dict[str, Any]]:
        """
        构建会话感知的训练样本
        
        考虑会话边界，使用历史会话作为输入，当前会话的物品作为目标。
        
        Args:
            user_events: 用户事件列表
            session_gap_seconds: 会话间隔阈值
            max_history_sessions: 最大历史会话数
            
        Yields:
            训练样本字典
        """
        sessions = self.compute_session_boundaries(user_events, session_gap_seconds)
        
        if len(sessions) < 2:
            return
        
        for i in range(1, len(sessions)):
            # 历史会话
            start_session = max(0, i - max_history_sessions)
            history_events = []
            for session in sessions[start_session:i]:
                history_events.extend(session)
            
            # 当前会话的第一个事件作为目标
            target_event = sessions[i][0]
            
            yield {
                "events": history_events,
                "target": target_event,
                "user_id": user_events[0].get("user_id"),
                "session_id": i,
                "num_history_sessions": i - start_session,
            }
    
    def get_statistics(
        self,
        events_by_user: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        计算数据集统计信息
        
        Args:
            events_by_user: 按用户分组的事件字典
            
        Returns:
            统计信息字典
        """
        user_counts = []
        item_counts = defaultdict(int)
        action_counts = defaultdict(int)
        
        for user_id, events in events_by_user.items():
            user_counts.append(len(events))
            
            for event in events:
                item_id = event.get("item_id")
                if item_id:
                    item_counts[item_id] += 1
                
                action = event.get("action")
                if action:
                    action_counts[action] += 1
        
        return {
            "num_users": len(events_by_user),
            "num_events": sum(user_counts),
            "num_unique_items": len(item_counts),
            "avg_events_per_user": sum(user_counts) / len(user_counts) if user_counts else 0,
            "max_events_per_user": max(user_counts) if user_counts else 0,
            "min_events_per_user": min(user_counts) if user_counts else 0,
            "action_distribution": dict(action_counts),
            "top_items": sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        }
    
    def __repr__(self) -> str:
        return (
            f"SequenceBuilder(\n"
            f"  item_pool_size={len(self._item_pool)},\n"
            f"  num_unique_items={len(self._item_popularity)}\n"
            f")"
        )

