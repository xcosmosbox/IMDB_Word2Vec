"""
事件解析器模块

负责将原始用户行为事件解析为 Token 列表，包括：
- 行为类型 Token (ACTION_*)
- 物品语义 ID Token (ITEM_*)
- 时间特征 Token (TIME_*)
- 设备类型 Token (DEVICE_*)
- 上下文 Token (CONTEXT_*)
- 用户属性 Token (USER_*)
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union

from algorithm.feature_engineering.config import FeatureConfig

logger = logging.getLogger(__name__)


class EventParser:
    """
    事件解析器
    
    将原始用户行为事件转换为 Token 列表，支持多种事件格式和特征提取。
    
    Attributes:
        config: 特征工程配置
        semantic_id_encoder: 语义 ID 编码器（可选）
        
    Example:
        >>> config = FeatureConfig()
        >>> parser = EventParser(config)
        >>> event = {
        ...     "item_id": "movie_001",
        ...     "action": "click",
        ...     "timestamp": 1704067200,
        ...     "device": "mobile"
        ... }
        >>> tokens = parser.parse_event(event)
        >>> print(tokens)
        ['ACTION_click', 'ITEM_256_1234_8901', 'TIME_morning', 'TIME_weekday', 'DEVICE_mobile']
    """
    
    def __init__(
        self, 
        config: FeatureConfig, 
        semantic_id_encoder: Optional[Any] = None
    ):
        """
        初始化事件解析器
        
        Args:
            config: 特征工程配置
            semantic_id_encoder: 语义 ID 编码器实例（可选）
                如果提供，将使用编码器将物品特征转换为语义 ID
        """
        self.config = config
        self.semantic_id_encoder = semantic_id_encoder
        
        # 预计算的时间特征
        self._weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday"]
        self._weekend_names = ["saturday", "sunday"]
    
    def parse_event(self, event: Dict[str, Any]) -> List[str]:
        """
        解析单个事件为 Token 列表
        
        按照固定顺序提取事件的各类特征并转换为 Token：
        1. 行为类型 (ACTION_*)
        2. 物品语义 ID (ITEM_*)
        3. 时间特征 (TIME_*)
        4. 设备类型 (DEVICE_*)
        5. 上下文特征 (CONTEXT_*)
        6. 用户属性 (USER_*)
        
        Args:
            event: 原始事件字典，典型格式：
                {
                    "item_id": str,           # 物品 ID
                    "action": str,            # 行为类型
                    "timestamp": int,         # Unix 时间戳
                    "device": str,            # 设备类型（可选）
                    "context": dict,          # 上下文信息（可选）
                    "semantic_id": tuple,     # 语义 ID（可选）
                    "item_features": list,    # 物品特征向量（可选）
                    "user_profile": dict,     # 用户属性（可选）
                }
        
        Returns:
            Token 列表
            
        Example:
            >>> parser.parse_event({
            ...     "item_id": "movie_001",
            ...     "action": "click",
            ...     "timestamp": 1704067200,
            ...     "semantic_id": (256, 1234, 8901)
            ... })
            ['ACTION_click', 'ITEM_256_1234_8901', 'TIME_evening', 'TIME_weekday']
        """
        tokens = []
        
        # 1. 行为 Token
        action_token = self._get_action_token(event)
        if action_token:
            tokens.append(action_token)
        
        # 2. 物品 Token（语义 ID）
        item_token = self._get_item_token(event)
        if item_token:
            tokens.append(item_token)
        
        # 3. 时间 Token
        time_tokens = self._get_time_tokens(event)
        tokens.extend(time_tokens)
        
        # 4. 设备 Token
        device_token = self._get_device_token(event)
        if device_token:
            tokens.append(device_token)
        
        # 5. 上下文 Token
        context_tokens = self._get_context_tokens(event)
        tokens.extend(context_tokens)
        
        # 6. 用户属性 Token
        user_tokens = self._get_user_tokens(event)
        tokens.extend(user_tokens)
        
        return tokens
    
    def parse_events(self, events: List[Dict[str, Any]]) -> List[List[str]]:
        """
        批量解析事件列表
        
        Args:
            events: 事件字典列表
            
        Returns:
            Token 列表的列表
        """
        return [self.parse_event(event) for event in events]
    
    def _get_action_token(self, event: Dict[str, Any]) -> Optional[str]:
        """
        获取行为类型 Token
        
        Args:
            event: 事件字典
            
        Returns:
            行为 Token，如 "ACTION_click"
        """
        action = event.get("action")
        if action is None:
            action = "view"  # 默认行为
        
        # 标准化行为类型
        action = str(action).lower().strip()
        
        # 验证行为类型是否支持
        if action not in self.config.action_types:
            logger.debug(f"未知行为类型: {action}，使用原始值")
        
        return f"ACTION_{action}"
    
    def _get_item_token(self, event: Dict[str, Any]) -> Optional[str]:
        """
        获取物品的语义 ID Token
        
        优先级：
        1. 使用事件中已有的 semantic_id
        2. 使用 semantic_id_encoder 编码物品特征
        3. 使用 item_id 的哈希值生成伪语义 ID
        
        Args:
            event: 事件字典
            
        Returns:
            物品 Token，如 "ITEM_256_1234_8901"
        """
        # 方式 1：直接使用事件中的语义 ID
        if "semantic_id" in event:
            sid = event["semantic_id"]
            if isinstance(sid, (list, tuple)) and len(sid) >= 3:
                l1, l2, l3 = int(sid[0]), int(sid[1]), int(sid[2])
                return f"ITEM_{l1}_{l2}_{l3}"
        
        # 方式 2：使用编码器生成语义 ID
        if "item_features" in event and self.semantic_id_encoder is not None:
            try:
                import torch
                features = event["item_features"]
                if not isinstance(features, torch.Tensor):
                    features = torch.tensor(features, dtype=torch.float32)
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                
                l1, l2, l3 = self.semantic_id_encoder.encode(features)
                return f"ITEM_{l1.item()}_{l2.item()}_{l3.item()}"
            except Exception as e:
                logger.warning(f"语义 ID 编码失败: {e}")
        
        # 方式 3：使用 item_id 的哈希值生成伪语义 ID
        item_id = event.get("item_id", "unknown")
        return self._generate_hash_based_item_token(item_id)
    
    def _generate_hash_based_item_token(self, item_id: str) -> str:
        """
        基于哈希值生成物品 Token
        
        当没有语义 ID 时，使用物品 ID 的哈希值生成伪语义 ID。
        这确保相同的 item_id 始终映射到相同的 Token。
        
        Args:
            item_id: 物品 ID
            
        Returns:
            物品 Token
        """
        # 使用稳定的哈希函数
        hash_val = self._stable_hash(str(item_id))
        
        # 映射到各层的码本范围
        l1_size, l2_size, l3_size = self.config.codebook_sizes
        l1 = hash_val % l1_size
        l2 = (hash_val // l1_size) % l2_size
        l3 = (hash_val // (l1_size * l2_size)) % l3_size
        
        return f"ITEM_{l1}_{l2}_{l3}"
    
    def _stable_hash(self, s: str) -> int:
        """
        计算字符串的稳定哈希值
        
        Python 的内置 hash() 在不同运行时可能返回不同结果，
        这里使用一个简单的确定性哈希算法。
        
        Args:
            s: 输入字符串
            
        Returns:
            非负整数哈希值
        """
        h = 0
        for c in s:
            h = (h * 31 + ord(c)) & 0xFFFFFFFF
        return h
    
    def _get_time_tokens(self, event: Dict[str, Any]) -> List[str]:
        """
        获取时间相关的 Token
        
        提取以下时间特征：
        - 时间段（TIME_morning, TIME_afternoon, etc.）
        - 周末/工作日（TIME_weekend, TIME_weekday）
        
        Args:
            event: 事件字典
            
        Returns:
            时间 Token 列表
        """
        tokens = []
        
        if "timestamp" not in event:
            return tokens
        
        try:
            timestamp = event["timestamp"]
            dt = datetime.fromtimestamp(timestamp)
            
            # 时间段 Token
            hour = dt.hour
            time_bucket = self.config.get_time_bucket(hour)
            tokens.append(f"TIME_{time_bucket}")
            
            # 周末/工作日 Token
            if dt.weekday() >= 5:  # 周六(5)、周日(6)
                tokens.append("TIME_weekend")
            else:
                tokens.append("TIME_weekday")
            
        except (OSError, OverflowError, ValueError) as e:
            logger.debug(f"时间戳解析失败: {event.get('timestamp')}, 错误: {e}")
        
        return tokens
    
    def _get_device_token(self, event: Dict[str, Any]) -> Optional[str]:
        """
        获取设备类型 Token
        
        Args:
            event: 事件字典
            
        Returns:
            设备 Token，如 "DEVICE_mobile"
        """
        device = event.get("device")
        if device is None:
            return None
        
        # 标准化设备类型
        device = str(device).lower().strip()
        
        # 验证设备类型是否支持
        if device not in self.config.device_types:
            device = "other"
        
        return f"DEVICE_{device}"
    
    def _get_context_tokens(self, event: Dict[str, Any]) -> List[str]:
        """
        获取上下文 Token
        
        提取以下上下文特征：
        - 来源场景（CONTEXT_home, CONTEXT_search, etc.）
        - 位置信息（CONTEXT_top, CONTEXT_mid, CONTEXT_bottom）
        
        Args:
            event: 事件字典
            
        Returns:
            上下文 Token 列表
        """
        tokens = []
        context = event.get("context")
        
        if context is None or not isinstance(context, dict):
            return tokens
        
        # 来源场景 Token
        if "source" in context:
            source = str(context["source"]).lower().strip()
            tokens.append(f"CONTEXT_{source}")
        
        # 位置 Token
        if "position" in context:
            try:
                position = int(context["position"])
                if position <= 3:
                    tokens.append("CONTEXT_top")
                elif position <= 10:
                    tokens.append("CONTEXT_mid")
                else:
                    tokens.append("CONTEXT_bottom")
            except (TypeError, ValueError):
                pass
        
        # 搜索关键词 Token（如果有）
        if "keyword" in context and context["keyword"]:
            tokens.append("CONTEXT_has_keyword")
        
        return tokens
    
    def _get_user_tokens(self, event: Dict[str, Any]) -> List[str]:
        """
        获取用户属性 Token
        
        提取以下用户特征：
        - 年龄段（USER_age_*）
        - 性别（USER_gender_*）
        - 会员等级（USER_level_*）
        
        Args:
            event: 事件字典
            
        Returns:
            用户属性 Token 列表
        """
        tokens = []
        user_profile = event.get("user_profile")
        
        if user_profile is None or not isinstance(user_profile, dict):
            return tokens
        
        # 年龄段 Token
        if "age" in user_profile:
            try:
                age = int(user_profile["age"])
                age_bucket = self._get_age_bucket(age)
                tokens.append(f"USER_age_{age_bucket}")
            except (TypeError, ValueError):
                pass
        
        # 性别 Token
        if "gender" in user_profile:
            gender = str(user_profile["gender"]).lower().strip()
            if gender in ["m", "male", "男"]:
                tokens.append("USER_gender_m")
            elif gender in ["f", "female", "女"]:
                tokens.append("USER_gender_f")
            else:
                tokens.append("USER_gender_u")  # unknown
        
        # 会员等级 Token
        if "level" in user_profile:
            try:
                level = int(user_profile["level"])
                tokens.append(f"USER_level_{level}")
            except (TypeError, ValueError):
                pass
        
        return tokens
    
    def _get_age_bucket(self, age: int) -> str:
        """
        获取年龄分桶
        
        Args:
            age: 年龄
            
        Returns:
            年龄分桶名称
        """
        if age < 18:
            return "teen"
        elif age < 25:
            return "young"
        elif age < 35:
            return "adult"
        elif age < 50:
            return "middle"
        else:
            return "senior"
    
    def parse_semantic_id(self, token: str) -> Tuple[int, int, int]:
        """
        解析物品 Token 为语义 ID 三元组
        
        Args:
            token: 物品 Token，如 "ITEM_256_1234_8901"
            
        Returns:
            语义 ID 三元组 (L1, L2, L3)
            
        Raises:
            ValueError: Token 格式无效时抛出
            
        Example:
            >>> parser.parse_semantic_id("ITEM_256_1234_8901")
            (256, 1234, 8901)
        """
        if not token.startswith("ITEM_"):
            raise ValueError(f"无效的物品 Token: {token}")
        
        parts = token[5:].split("_")
        if len(parts) != 3:
            raise ValueError(f"物品 Token 格式错误，需要 3 个部分: {token}")
        
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            raise ValueError(f"物品 Token 包含非数字部分: {token}")
    
    def is_item_token(self, token: str) -> bool:
        """
        检查是否是物品 Token
        
        Args:
            token: Token 字符串
            
        Returns:
            是否是物品 Token
        """
        return token.startswith("ITEM_")
    
    def is_action_token(self, token: str) -> bool:
        """
        检查是否是行为 Token
        
        Args:
            token: Token 字符串
            
        Returns:
            是否是行为 Token
        """
        return token.startswith("ACTION_")
    
    def is_time_token(self, token: str) -> bool:
        """
        检查是否是时间 Token
        
        Args:
            token: Token 字符串
            
        Returns:
            是否是时间 Token
        """
        return token.startswith("TIME_")
    
    def is_context_token(self, token: str) -> bool:
        """
        检查是否是上下文 Token
        
        Args:
            token: Token 字符串
            
        Returns:
            是否是上下文 Token
        """
        return token.startswith("CONTEXT_") or token.startswith("DEVICE_")
    
    def get_token_type(self, token: str) -> str:
        """
        获取 Token 的类型
        
        Args:
            token: Token 字符串
            
        Returns:
            Token 类型: "ITEM", "ACTION", "USER", "CONTEXT"
        """
        if token.startswith("ITEM_"):
            return "ITEM"
        elif token.startswith("ACTION_"):
            return "ACTION"
        elif token.startswith("USER_"):
            return "USER"
        else:
            return "CONTEXT"
    
    def get_token_type_id(self, token: str) -> int:
        """
        获取 Token 的类型 ID
        
        Args:
            token: Token 字符串
            
        Returns:
            Token 类型 ID
        """
        token_type = self.get_token_type(token)
        return self.config.token_type_ids.get(token_type, 3)  # 默认为 CONTEXT
    
    def extract_item_tokens(self, tokens: List[str]) -> List[str]:
        """
        从 Token 列表中提取所有物品 Token
        
        Args:
            tokens: Token 列表
            
        Returns:
            物品 Token 列表
        """
        return [t for t in tokens if self.is_item_token(t)]
    
    def extract_action_tokens(self, tokens: List[str]) -> List[str]:
        """
        从 Token 列表中提取所有行为 Token
        
        Args:
            tokens: Token 列表
            
        Returns:
            行为 Token 列表
        """
        return [t for t in tokens if self.is_action_token(t)]
    
    def __repr__(self) -> str:
        return (
            f"EventParser(\n"
            f"  action_types={self.config.action_types},\n"
            f"  device_types={self.config.device_types},\n"
            f"  has_semantic_encoder={self.semantic_id_encoder is not None}\n"
            f")"
        )

