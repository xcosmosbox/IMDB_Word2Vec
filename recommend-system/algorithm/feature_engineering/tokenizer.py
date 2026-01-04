"""
Token 化器模块

实现 TokenizerInterface 接口，负责将用户行为事件序列转换为模型输入格式。

主要功能：
- 事件序列 Token 化
- 训练样本构建
- 词表管理
- 序列截断与填充
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

import torch

from algorithm.interfaces import TokenizerInterface, TokenizedSequence
from algorithm.feature_engineering.config import FeatureConfig
from algorithm.feature_engineering.vocabulary import Vocabulary
from algorithm.feature_engineering.event_parser import EventParser

logger = logging.getLogger(__name__)


class RecommendTokenizer(TokenizerInterface):
    """
    推荐系统 Token 化器
    
    实现 TokenizerInterface 接口，将用户行为事件序列转换为模型可处理的
    Token 序列格式。支持特殊 Token 添加、序列截断、填充等功能。
    
    Attributes:
        config: 特征工程配置
        vocab: 词表管理器
        event_parser: 事件解析器
        
    Example:
        >>> config = FeatureConfig()
        >>> tokenizer = RecommendTokenizer(config)
        >>> events = [
        ...     {"item_id": "movie_001", "action": "click", "timestamp": 1704067200},
        ...     {"item_id": "movie_002", "action": "view", "timestamp": 1704067260},
        ... ]
        >>> result = tokenizer.tokenize_events(events, max_length=50)
        >>> print(result.input_ids.shape)
        torch.Size([50])
    """
    
    def __init__(
        self, 
        config: Optional[FeatureConfig] = None,
        vocab: Optional[Vocabulary] = None,
        semantic_id_encoder: Optional[Any] = None,
    ):
        """
        初始化 Token 化器
        
        Args:
            config: 特征工程配置，如果为 None 则使用默认配置
            vocab: 词表管理器，如果为 None 则创建新的词表
            semantic_id_encoder: 语义 ID 编码器（可选）
        """
        self.config = config or FeatureConfig()
        self.vocab = vocab or Vocabulary(self.config)
        self.event_parser = EventParser(self.config, semantic_id_encoder)
        
        # 预先添加常用 Token 到词表
        self._init_common_tokens()
    
    def _init_common_tokens(self) -> None:
        """
        初始化常用 Token
        
        将行为类型、设备类型、时间分桶等常用 Token 预先添加到词表，
        确保它们拥有稳定的 ID。
        """
        # 添加行为类型 Token
        for action in self.config.action_types:
            self.vocab.add_token(f"ACTION_{action}")
        
        # 添加设备类型 Token
        for device in self.config.device_types:
            self.vocab.add_token(f"DEVICE_{device}")
        
        # 添加时间分桶 Token
        for bucket in self.config.time_buckets:
            self.vocab.add_token(f"TIME_{bucket}")
        self.vocab.add_token("TIME_weekend")
        self.vocab.add_token("TIME_weekday")
        
        # 添加上下文类型 Token
        for context in self.config.context_types:
            self.vocab.add_token(f"CONTEXT_{context}")
        self.vocab.add_token("CONTEXT_top")
        self.vocab.add_token("CONTEXT_mid")
        self.vocab.add_token("CONTEXT_bottom")
        self.vocab.add_token("CONTEXT_has_keyword")
        
        logger.debug(f"初始化常用 Token 完成，当前词表大小: {len(self.vocab)}")
    
    def tokenize_events(
        self,
        events: List[Dict[str, Any]],
        max_length: int = 1024,
        add_special_tokens: bool = True,
        padding: bool = True,
        return_tensors: bool = True,
    ) -> TokenizedSequence:
        """
        将事件列表转换为 Token 序列
        
        处理流程：
        1. 解析每个事件为 Token 列表
        2. 拼接所有 Token
        3. 添加特殊 Token ([CLS] 和 [SEP])
        4. 截断或填充到指定长度
        5. 生成各类辅助张量
        
        Args:
            events: 用户行为事件列表，按时间顺序排列
            max_length: 最大序列长度
            add_special_tokens: 是否添加 [CLS] 和 [SEP] Token
            padding: 是否填充到 max_length
            return_tensors: 是否返回 PyTorch 张量
        
        Returns:
            TokenizedSequence 对象，包含：
            - input_ids: Token ID 序列
            - attention_mask: 注意力掩码
            - token_types: Token 类型 ID
            - positions: 位置索引
            - semantic_ids: [L1_ids, L2_ids, L3_ids]
        """
        max_length = max_length or self.config.max_seq_length
        
        # 解析事件为 Token
        all_tokens: List[str] = []
        all_token_types: List[int] = []
        all_semantic_ids: Dict[str, List[int]] = {"l1": [], "l2": [], "l3": []}
        
        # 添加 [CLS] Token
        if add_special_tokens:
            all_tokens.append(self.config.cls_token)
            all_token_types.append(self.config.token_type_ids["CONTEXT"])
            all_semantic_ids["l1"].append(0)
            all_semantic_ids["l2"].append(0)
            all_semantic_ids["l3"].append(0)
        
        # 解析每个事件
        for event in events:
            tokens = self.event_parser.parse_event(event)
            
            for token in tokens:
                all_tokens.append(token)
                
                # 确定 Token 类型
                token_type = self.event_parser.get_token_type(token)
                all_token_types.append(self.config.token_type_ids.get(token_type, 3))
                
                # 提取语义 ID
                if self.event_parser.is_item_token(token):
                    try:
                        l1, l2, l3 = self.event_parser.parse_semantic_id(token)
                        all_semantic_ids["l1"].append(l1)
                        all_semantic_ids["l2"].append(l2)
                        all_semantic_ids["l3"].append(l3)
                    except ValueError:
                        all_semantic_ids["l1"].append(0)
                        all_semantic_ids["l2"].append(0)
                        all_semantic_ids["l3"].append(0)
                else:
                    all_semantic_ids["l1"].append(0)
                    all_semantic_ids["l2"].append(0)
                    all_semantic_ids["l3"].append(0)
        
        # 添加 [SEP] Token
        if add_special_tokens:
            all_tokens.append(self.config.sep_token)
            all_token_types.append(self.config.token_type_ids["CONTEXT"])
            all_semantic_ids["l1"].append(0)
            all_semantic_ids["l2"].append(0)
            all_semantic_ids["l3"].append(0)
        
        # 处理截断与填充
        seq_len = len(all_tokens)
        
        if seq_len > max_length:
            # 截断（保留最近的事件，但确保 [CLS] 在开头）
            if add_special_tokens:
                # 保留 [CLS] + 最后的 (max_length - 1) 个 Token
                all_tokens = [all_tokens[0]] + all_tokens[-(max_length - 1):]
                all_token_types = [all_token_types[0]] + all_token_types[-(max_length - 1):]
                all_semantic_ids["l1"] = [all_semantic_ids["l1"][0]] + all_semantic_ids["l1"][-(max_length - 1):]
                all_semantic_ids["l2"] = [all_semantic_ids["l2"][0]] + all_semantic_ids["l2"][-(max_length - 1):]
                all_semantic_ids["l3"] = [all_semantic_ids["l3"][0]] + all_semantic_ids["l3"][-(max_length - 1):]
            else:
                all_tokens = all_tokens[-max_length:]
                all_token_types = all_token_types[-max_length:]
                all_semantic_ids["l1"] = all_semantic_ids["l1"][-max_length:]
                all_semantic_ids["l2"] = all_semantic_ids["l2"][-max_length:]
                all_semantic_ids["l3"] = all_semantic_ids["l3"][-max_length:]
            seq_len = max_length
        
        # 创建注意力掩码
        attention_mask = [1] * seq_len
        
        # 填充
        padding_length = max_length - seq_len if padding else 0
        if padding_length > 0:
            all_tokens.extend([self.config.pad_token] * padding_length)
            all_token_types.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)
            all_semantic_ids["l1"].extend([0] * padding_length)
            all_semantic_ids["l2"].extend([0] * padding_length)
            all_semantic_ids["l3"].extend([0] * padding_length)
        
        # 将 Token 转换为 ID
        input_ids = self.vocab.encode_batch(all_tokens)
        
        # 生成位置索引
        final_length = max_length if padding else seq_len
        positions = list(range(final_length))
        
        # 返回结果
        if return_tensors:
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
        else:
            return TokenizedSequence(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_types=all_token_types,
                positions=positions,
                semantic_ids=[
                    all_semantic_ids["l1"],
                    all_semantic_ids["l2"],
                    all_semantic_ids["l3"],
                ],
            )
    
    def build_training_sample(
        self,
        events: List[Dict[str, Any]],
        target_item: Dict[str, Any],
        max_length: int = 1024,
    ) -> TokenizedSequence:
        """
        构建训练样本
        
        训练样本格式：
        - 输入：用户历史事件序列
        - 标签：目标物品的语义 ID
        
        Args:
            events: 用户历史事件列表
            target_item: 目标物品信息（下一个交互的物品）
            max_length: 最大序列长度
        
        Returns:
            包含 labels 的 TokenizedSequence
        """
        # Token 化输入序列
        result = self.tokenize_events(events, max_length=max_length)
        
        # 解析目标物品的语义 ID
        target_tokens = self.event_parser.parse_event(target_item)
        item_tokens = [t for t in target_tokens if self.event_parser.is_item_token(t)]
        
        if not item_tokens:
            # 如果没有提取到物品 Token，创建一个
            item_token = self.event_parser._get_item_token(target_item)
            item_tokens = [item_token] if item_token else []
        
        if item_tokens:
            item_token = item_tokens[0]
            l1, l2, l3 = self.event_parser.parse_semantic_id(item_token)
            target_token_id = self.vocab.encode(item_token)
        else:
            target_token_id = self.config.unk_token_id
        
        # 创建标签张量
        # 使用 -100 表示不计算损失的位置（PyTorch CrossEntropyLoss 的 ignore_index）
        labels = torch.full((len(result.input_ids),), -100, dtype=torch.long)
        
        # 在序列末尾（[SEP] 之前）设置目标标签
        # 找到最后一个非 padding 的位置
        if isinstance(result.attention_mask, torch.Tensor):
            valid_length = result.attention_mask.sum().item()
        else:
            valid_length = sum(result.attention_mask)
        
        if valid_length > 1:
            # 标签放在 [SEP] 的位置
            labels[valid_length - 1] = target_token_id
        
        result.labels = labels
        
        return result
    
    def build_training_batch(
        self,
        samples: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
        max_length: int = 1024,
    ) -> Dict[str, torch.Tensor]:
        """
        构建训练批次
        
        Args:
            samples: (events, target_item) 元组列表
            max_length: 最大序列长度
            
        Returns:
            批次数据字典
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_types = []
        batch_positions = []
        batch_semantic_ids_l1 = []
        batch_semantic_ids_l2 = []
        batch_semantic_ids_l3 = []
        batch_labels = []
        
        for events, target_item in samples:
            result = self.build_training_sample(events, target_item, max_length)
            
            batch_input_ids.append(result.input_ids)
            batch_attention_mask.append(result.attention_mask)
            batch_token_types.append(result.token_types)
            batch_positions.append(result.positions)
            batch_semantic_ids_l1.append(result.semantic_ids[0])
            batch_semantic_ids_l2.append(result.semantic_ids[1])
            batch_semantic_ids_l3.append(result.semantic_ids[2])
            batch_labels.append(result.labels)
        
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "token_types": torch.stack(batch_token_types),
            "positions": torch.stack(batch_positions),
            "semantic_ids": [
                torch.stack(batch_semantic_ids_l1),
                torch.stack(batch_semantic_ids_l2),
                torch.stack(batch_semantic_ids_l3),
            ],
            "labels": torch.stack(batch_labels),
        }
    
    def get_vocab_size(self) -> int:
        """
        返回词表大小
        
        Returns:
            词表中的 Token 数量
        """
        return len(self.vocab)
    
    def save_vocab(self, path: str) -> None:
        """
        保存词表到文件
        
        Args:
            path: 保存路径
        """
        self.vocab.save(path)
        logger.info(f"词表已保存到 {path}")
    
    def load_vocab(self, path: str) -> None:
        """
        从文件加载词表
        
        Args:
            path: 词表文件路径
        """
        self.vocab.load(path)
        logger.info(f"词表已从 {path} 加载，大小: {len(self.vocab)}")
    
    def encode_item(self, item: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        编码物品为语义 ID
        
        Args:
            item: 物品信息字典
            
        Returns:
            语义 ID 三元组 (L1, L2, L3)
        """
        item_token = self.event_parser._get_item_token(item)
        if item_token:
            return self.event_parser.parse_semantic_id(item_token)
        return (0, 0, 0)
    
    def decode_semantic_id(self, l1: int, l2: int, l3: int) -> str:
        """
        解码语义 ID 为物品 Token
        
        Args:
            l1: 第一层语义 ID
            l2: 第二层语义 ID
            l3: 第三层语义 ID
            
        Returns:
            物品 Token 字符串
        """
        return f"ITEM_{l1}_{l2}_{l3}"
    
    def add_item_to_vocab(self, item: Dict[str, Any]) -> int:
        """
        将物品添加到词表
        
        Args:
            item: 物品信息字典
            
        Returns:
            物品 Token 的 ID
        """
        item_token = self.event_parser._get_item_token(item)
        if item_token:
            return self.vocab.add_token(item_token)
        return self.config.unk_token_id
    
    def get_special_tokens_mask(
        self, 
        token_ids: List[int]
    ) -> List[int]:
        """
        获取特殊 Token 掩码
        
        Args:
            token_ids: Token ID 列表
            
        Returns:
            掩码列表，1 表示特殊 Token，0 表示普通 Token
        """
        special_ids = set([
            self.config.pad_token_id,
            self.config.cls_token_id,
            self.config.sep_token_id,
            self.config.mask_token_id,
            self.config.unk_token_id,
        ])
        return [1 if tid in special_ids else 0 for tid in token_ids]
    
    def create_mask_for_mlm(
        self,
        input_ids: torch.Tensor,
        mask_probability: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为 MLM（掩码语言模型）训练创建掩码
        
        Args:
            input_ids: 输入 Token ID 张量
            mask_probability: 掩码概率
            
        Returns:
            (masked_input_ids, labels) 元组
        """
        labels = input_ids.clone()
        
        # 创建概率矩阵
        probability_matrix = torch.full(input_ids.shape, mask_probability)
        
        # 特殊 Token 不进行掩码
        special_tokens_mask = torch.tensor(
            self.get_special_tokens_mask(input_ids.tolist()),
            dtype=torch.bool
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # 随机选择要掩码的位置
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 只计算被掩码位置的损失
        
        # 80% 的情况替换为 [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.config.mask_token_id
        
        # 10% 的情况替换为随机 Token
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(
            self.config.special_token_count, 
            len(self.vocab), 
            input_ids.shape, 
            dtype=torch.long
        )
        input_ids[indices_random] = random_tokens[indices_random]
        
        # 剩余 10% 保持不变
        
        return input_ids, labels
    
    def __repr__(self) -> str:
        return (
            f"RecommendTokenizer(\n"
            f"  vocab_size={len(self.vocab)},\n"
            f"  max_seq_length={self.config.max_seq_length},\n"
            f"  special_tokens={self.config.get_special_tokens()}\n"
            f")"
        )

