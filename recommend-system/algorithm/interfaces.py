"""
生成式推荐系统 - 公共接口定义

所有模块开发者必须遵循这些接口定义，以确保模块间的无缝集成。

模块分工：
- Person A: SemanticIDEncoder (semantic_id/)
- Person B: UserEncoder (encoder/)
- Person C: RecommendDecoder (decoder/)
- Person D: Trainer (training/)
- Person E: Tokenizer (feature_engineering/)
- Person F: ServingExporter (serving/)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import torch
import torch.nn as nn


# =============================================================================
# 公共配置
# =============================================================================

@dataclass
class ModelConfig:
    """模型统一配置（所有模块共享）"""
    
    # 模型维度
    d_model: int = 512
    n_heads: int = 16
    d_ff: int = 2048
    dropout: float = 0.1
    
    # 序列配置
    max_seq_length: int = 1024
    
    # 语义 ID 配置
    semantic_id_levels: int = 3
    codebook_sizes: Tuple[int, int, int] = (1024, 4096, 16384)
    embedding_dim: int = 256  # 物品特征维度
    
    # Token 类型
    num_token_types: int = 4  # USER=0, ITEM=1, ACTION=2, CONTEXT=3
    
    # 特殊 Token ID
    pad_token_id: int = 0
    cls_token_id: int = 1
    sep_token_id: int = 2
    mask_token_id: int = 3
    unk_token_id: int = 4


# =============================================================================
# Person A: Semantic ID 编码器接口
# =============================================================================

class SemanticIDEncoderInterface(ABC):
    """
    语义 ID 编码器接口 (Person A 实现)
    
    功能：将物品特征向量编码为层次化的语义 ID
    
    使用场景：
    - 新物品入库时，生成其语义 ID
    - 模型输入时，将物品特征转换为离散 Token
    """
    
    @abstractmethod
    def encode(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码物品特征为语义 ID
        
        Args:
            features: 物品特征向量 (batch_size, embedding_dim)
                      embedding_dim 默认为 256
        
        Returns:
            Tuple[L1_ids, L2_ids, L3_ids]:
                - L1_ids: (batch_size,) 第一层语义ID，范围 [0, 1024)
                - L2_ids: (batch_size,) 第二层语义ID，范围 [0, 4096)
                - L3_ids: (batch_size,) 第三层语义ID，范围 [0, 16384)
        """
        pass
    
    @abstractmethod
    def decode(self, l1_ids: torch.Tensor, l2_ids: torch.Tensor, l3_ids: torch.Tensor) -> torch.Tensor:
        """
        从语义 ID 重建物品特征向量
        
        Args:
            l1_ids: (batch_size,) 第一层语义ID
            l2_ids: (batch_size,) 第二层语义ID
            l3_ids: (batch_size,) 第三层语义ID
        
        Returns:
            reconstructed: (batch_size, embedding_dim) 重建的特征向量
        """
        pass
    
    @abstractmethod
    def get_codebook_embeddings(self, level: int) -> torch.Tensor:
        """
        获取指定层级的码本嵌入
        
        Args:
            level: 层级 (1, 2, 或 3)
        
        Returns:
            embeddings: (codebook_size, embedding_dim) 码本嵌入矩阵
        """
        pass


# =============================================================================
# Person B: 用户编码器接口
# =============================================================================

class UserEncoderInterface(ABC):
    """
    用户行为编码器接口 (Person B 实现)
    
    功能：将用户历史行为序列编码为用户表示向量
    
    使用场景：
    - 推荐请求时，编码用户历史以生成推荐
    """
    
    @abstractmethod
    def forward(
        self,
        semantic_ids: List[torch.Tensor],
        positions: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        编码用户行为序列
        
        Args:
            semantic_ids: [L1_ids, L2_ids, L3_ids] 每个形状为 (batch_size, seq_len)
            positions: (batch_size, seq_len) 位置索引
            token_types: (batch_size, seq_len) Token类型 (0=USER, 1=ITEM, 2=ACTION, 3=CONTEXT)
            attention_mask: (batch_size, seq_len) 注意力掩码 (1=有效, 0=padding)
            time_features: (batch_size, seq_len, time_dim) 可选的时间特征
        
        Returns:
            user_repr: (batch_size, d_model) 用户表示向量
        """
        pass
    
    @abstractmethod
    def get_sequence_output(
        self,
        semantic_ids: List[torch.Tensor],
        positions: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        获取完整序列的编码输出（用于解码器的交叉注意力）
        
        Returns:
            sequence_output: (batch_size, seq_len, d_model)
        """
        pass


# =============================================================================
# Person C: 推荐解码器接口
# =============================================================================

class RecommendDecoderInterface(ABC):
    """
    推荐生成解码器接口 (Person C 实现)
    
    功能：从用户表示向量生成推荐物品的语义 ID 序列
    
    使用场景：
    - 推荐请求时，生成 Top-K 推荐列表
    """
    
    @abstractmethod
    def forward(
        self,
        encoder_output: torch.Tensor,
        target_semantic_ids: Optional[List[torch.Tensor]] = None,
        target_positions: Optional[torch.Tensor] = None,
        target_token_types: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练模式：计算目标序列的预测 logits
        
        Args:
            encoder_output: (batch_size, src_len, d_model) 编码器输出
            target_semantic_ids: [L1_ids, L2_ids, L3_ids] 目标序列
            target_positions: (batch_size, tgt_len) 目标位置
            target_token_types: (batch_size, tgt_len) 目标Token类型
        
        Returns:
            Tuple[L1_logits, L2_logits, L3_logits, aux_loss]:
                - L1_logits: (batch_size, tgt_len, 1024)
                - L2_logits: (batch_size, tgt_len, 4096)
                - L3_logits: (batch_size, tgt_len, 16384)
                - aux_loss: MoE 负载均衡损失 (scalar)
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        encoder_output: torch.Tensor,
        num_recommendations: int = 20,
        beam_size: int = 4,
        temperature: float = 1.0,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        推理模式：自回归生成推荐列表
        
        Args:
            encoder_output: (batch_size, src_len, d_model) 编码器输出
            num_recommendations: 生成的推荐数量
            beam_size: Beam Search 宽度
            temperature: 采样温度
        
        Returns:
            recommendations: List[List[Tuple[L1, L2, L3]]]
                外层 List 长度为 batch_size
                内层 List 长度为 num_recommendations
                每个 Tuple 是 (L1_id, L2_id, L3_id)
        """
        pass


# =============================================================================
# Person D: 训练器接口
# =============================================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    max_epochs: int = 10
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    
    # 损失权重
    lambda_contrastive: float = 0.1
    lambda_preference: float = 0.1
    lambda_moe_balance: float = 0.01


class TrainerInterface(ABC):
    """
    模型训练器接口 (Person D 实现)
    
    功能：管理模型的完整训练流程
    """
    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Returns:
            metrics: 训练指标字典
                - "loss": 总损失
                - "ntp_loss": Next Token Prediction 损失
                - "contrastive_loss": 对比学习损失
                - "moe_balance_loss": MoE 负载均衡损失
                - "learning_rate": 当前学习率
        """
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        在验证集上评估
        
        Returns:
            metrics: 评估指标字典
                - "loss": 验证损失
                - "recall@10": Recall@10
                - "recall@50": Recall@50
                - "ndcg@10": NDCG@10
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        pass


# =============================================================================
# Person E: Token 化器接口
# =============================================================================

@dataclass
class TokenizedSequence:
    """Token 化后的序列"""
    input_ids: torch.Tensor          # (seq_len,) 或 (batch, seq_len)
    attention_mask: torch.Tensor     # (seq_len,) 或 (batch, seq_len)
    token_types: torch.Tensor        # (seq_len,) 或 (batch, seq_len)
    positions: torch.Tensor          # (seq_len,) 或 (batch, seq_len)
    semantic_ids: List[torch.Tensor] # [L1_ids, L2_ids, L3_ids]
    labels: Optional[torch.Tensor] = None  # 用于训练的标签


class TokenizerInterface(ABC):
    """
    Token 化器接口 (Person E 实现)
    
    功能：将原始事件日志转换为模型输入格式
    """
    
    @abstractmethod
    def tokenize_events(
        self,
        events: List[Dict[str, Any]],
        max_length: int = 1024,
    ) -> TokenizedSequence:
        """
        将事件列表转换为 Token 序列
        
        Args:
            events: 事件列表，每个事件格式：
                {
                    "item_id": str,
                    "action": str,  # "click", "view", "buy", etc.
                    "timestamp": int,
                    "device": str,  # optional
                    "context": dict,  # optional
                }
            max_length: 最大序列长度
        
        Returns:
            TokenizedSequence 对象
        """
        pass
    
    @abstractmethod
    def build_training_sample(
        self,
        events: List[Dict[str, Any]],
        target_item: Dict[str, Any],
    ) -> TokenizedSequence:
        """
        构建训练样本（包含输入和标签）
        
        Args:
            events: 用户历史事件
            target_item: 目标物品（下一个交互的物品）
        
        Returns:
            TokenizedSequence 对象，包含 labels
        """
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """返回词表大小"""
        pass
    
    @abstractmethod
    def save_vocab(self, path: str) -> None:
        """保存词表"""
        pass
    
    @abstractmethod
    def load_vocab(self, path: str) -> None:
        """加载词表"""
        pass


# =============================================================================
# Person F: 推理服务导出接口
# =============================================================================

@dataclass
class ExportConfig:
    """导出配置"""
    model_name: str = "ugt_recommend"
    precision: str = "fp16"  # fp32, fp16, int8
    max_batch_size: int = 64
    max_seq_length: int = 1024
    target_latency_ms: float = 30.0


class ServingExporterInterface(ABC):
    """
    推理服务导出器接口 (Person F 实现)
    
    功能：将训练好的模型导出为可部署的推理服务
    """
    
    @abstractmethod
    def export_onnx(
        self,
        model: nn.Module,
        save_path: str,
        config: ExportConfig,
    ) -> str:
        """
        导出 ONNX 模型
        
        Args:
            model: PyTorch 模型
            save_path: 保存路径
            config: 导出配置
        
        Returns:
            onnx_path: 导出的 ONNX 文件路径
        """
        pass
    
    @abstractmethod
    def optimize_tensorrt(
        self,
        onnx_path: str,
        engine_path: str,
        config: ExportConfig,
    ) -> str:
        """
        TensorRT 优化
        
        Args:
            onnx_path: ONNX 模型路径
            engine_path: TensorRT 引擎保存路径
            config: 导出配置
        
        Returns:
            engine_path: TensorRT 引擎路径
        """
        pass
    
    @abstractmethod
    def generate_triton_config(
        self,
        model_repository: str,
        config: ExportConfig,
    ) -> str:
        """
        生成 Triton 配置
        
        Args:
            model_repository: Triton 模型仓库路径
            config: 导出配置
        
        Returns:
            config_path: 配置文件路径
        """
        pass
    
    @abstractmethod
    def benchmark(
        self,
        triton_url: str,
        model_name: str,
        num_requests: int = 10000,
    ) -> Dict[str, float]:
        """
        性能测试
        
        Returns:
            metrics: 性能指标
                - "throughput": 吞吐量 (req/s)
                - "latency_p50": P50 延迟 (ms)
                - "latency_p90": P90 延迟 (ms)
                - "latency_p99": P99 延迟 (ms)
        """
        pass


# =============================================================================
# 完整 UGT 模型接口（组合所有模块）
# =============================================================================

class UGTModelInterface(ABC):
    """
    完整 UGT 模型接口
    
    由 Person B (Encoder) + Person C (Decoder) 组合实现
    Person A 的 SemanticIDEncoder 作为预处理模块
    """
    
    @abstractmethod
    def forward(
        self,
        encoder_semantic_ids: List[torch.Tensor],
        encoder_positions: torch.Tensor,
        encoder_token_types: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_semantic_ids: Optional[List[torch.Tensor]] = None,
        decoder_positions: Optional[torch.Tensor] = None,
        decoder_token_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        完整前向传播
        
        Returns:
            dict:
                - "l1_logits": (batch, tgt_len, 1024)
                - "l2_logits": (batch, tgt_len, 4096)
                - "l3_logits": (batch, tgt_len, 16384)
                - "aux_loss": scalar
                - "encoder_output": (batch, src_len, d_model)
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        encoder_semantic_ids: List[torch.Tensor],
        encoder_positions: torch.Tensor,
        encoder_token_types: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        num_recommendations: int = 20,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        生成推荐
        
        Returns:
            recommendations: List[List[Tuple[L1, L2, L3]]]
        """
        pass

