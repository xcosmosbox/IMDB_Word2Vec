# Person B: UGT Encoder（用户行为编码器）

## 你的角色
你是一名深度学习算法工程师，负责实现生成式推荐系统的 **UGT Encoder** 模块。

## 背景知识

UGT (Unified Generative Transformer) 是一个生成式推荐模型。Encoder 负责将用户的历史行为序列编码为统一的用户表示向量。

### 核心创新点

1. **Dot-Product Aggregated Attention (来自 Meta HSTU)**
   - 使用 ReLU 替代 Softmax，避免归一化导致的信息损失
   - 更适合处理推荐场景中的非平稳词汇表
   
2. **Group Layer Normalization (来自美团 MTGR)**
   - 对不同类型的 Token（USER/ITEM/ACTION/CONTEXT）使用不同的归一化参数
   - 增强不同语义空间的编码能力

## 你的任务

在 `algorithm/encoder/` 目录下实现完整的 Encoder 模块。

### 目录结构

```
algorithm/encoder/
├── __init__.py
├── config.py           # 配置类
├── embedding.py        # 统一输入嵌入层
├── attention.py        # Dot-Product Aggregated Attention
├── layer_norm.py       # Group Layer Normalization
├── ffn.py              # 前馈网络
├── encoder_layer.py    # 编码器层
├── encoder.py          # 完整编码器
└── tests/
    └── test_encoder.py
```

### 接口要求

你必须实现 `interfaces.py` 中定义的 `UserEncoderInterface`：

```python
from algorithm.interfaces import UserEncoderInterface

class UGTEncoder(nn.Module, UserEncoderInterface):
    def forward(
        self,
        semantic_ids: List[torch.Tensor],   # [L1_ids, L2_ids, L3_ids]
        positions: torch.Tensor,             # (batch, seq_len)
        token_types: torch.Tensor,           # (batch, seq_len)
        attention_mask: torch.Tensor,        # (batch, seq_len)
        time_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """返回 (batch_size, d_model) 的用户表示向量"""
        pass
    
    def get_sequence_output(self, ...) -> torch.Tensor:
        """返回 (batch, seq_len, d_model) 完整序列输出"""
        pass
```

### 核心实现

#### 1. config.py - 配置类

```python
from dataclasses import dataclass
from typing import Tuple

@dataclass
class EncoderConfig:
    """编码器配置"""
    d_model: int = 512              # 隐藏层维度
    n_heads: int = 16               # 注意力头数
    n_layers: int = 12              # 编码器层数
    d_ff: int = 2048                # FFN 中间维度
    max_seq_len: int = 1024         # 最大序列长度
    dropout: float = 0.1            # Dropout 率
    
    # 语义 ID 配置
    codebook_sizes: Tuple[int, int, int] = (1024, 4096, 16384)
    
    # Token 类型
    num_token_types: int = 4        # USER=0, ITEM=1, ACTION=2, CONTEXT=3
    num_groups: int = 4             # GLN 分组数（与 token_types 对应）
    
    # 时间特征
    time_dim: int = 32              # 时间嵌入维度
```

#### 2. embedding.py - 统一输入嵌入

```python
class InputEmbedding(nn.Module):
    """
    统一输入嵌入层
    
    E_input = E_semantic + E_position + E_type + E_time
    
    其中 E_semantic 是三层语义 ID 嵌入的拼接后投影
    """
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        
        # 语义 ID 嵌入（三层）
        self.semantic_embeddings = nn.ModuleList([
            nn.Embedding(size, config.d_model // 3)
            for size in config.codebook_sizes
        ])
        self.semantic_proj = nn.Linear(config.d_model, config.d_model)
        
        # 位置嵌入
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Token 类型嵌入
        self.type_embedding = nn.Embedding(config.num_token_types, config.d_model)
        
        # 时间特征投影（可选）
        self.time_proj = nn.Linear(config.time_dim, config.d_model) if config.time_dim > 0 else None
        
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        semantic_ids: List[torch.Tensor],  # [L1, L2, L3], each (batch, seq_len)
        positions: torch.Tensor,            # (batch, seq_len)
        token_types: torch.Tensor,          # (batch, seq_len)
        time_features: Optional[torch.Tensor] = None,  # (batch, seq_len, time_dim)
    ) -> torch.Tensor:
        """
        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        # 语义嵌入
        semantic_embs = [emb(ids) for emb, ids in zip(self.semantic_embeddings, semantic_ids)]
        semantic_emb = torch.cat(semantic_embs, dim=-1)  # (batch, seq_len, d_model)
        semantic_emb = self.semantic_proj(semantic_emb)
        
        # 位置嵌入
        position_emb = self.position_embedding(positions)
        
        # 类型嵌入
        type_emb = self.type_embedding(token_types)
        
        # 组合
        embeddings = semantic_emb + position_emb + type_emb
        
        # 时间特征（可选）
        if time_features is not None and self.time_proj is not None:
            time_emb = self.time_proj(time_features)
            embeddings = embeddings + time_emb
        
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
```

#### 3. attention.py - Dot-Product Aggregated Attention

```python
class DotProductAggregatedAttention(nn.Module):
    """
    点积聚合注意力 (来自 HSTU)
    
    核心特点：使用 ReLU 替代 Softmax
    
    Attention(Q, K, V) = ReLU(QK^T / √d) · V
    
    优势：
    1. 避免 Softmax 的归一化导致信息损失
    2. 更适合推荐场景中的非平稳词汇表
    3. 允许注意力权重为 0（忽略不相关项）
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** 0.5
    
    def forward(
        self,
        x: torch.Tensor,                    # (batch, seq_len, d_model)
        mask: Optional[torch.Tensor] = None, # (batch, seq_len) or (batch, seq_len, seq_len)
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 线性变换
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (batch, n_heads, seq_len, d_k)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, n_heads, seq_len, seq_len)
        
        # 应用掩码（将无效位置设为负无穷，ReLU 后变为 0）
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # ReLU 替代 Softmax（核心创新）
        attn_weights = F.relu(scores)
        attn_weights = self.dropout(attn_weights)
        
        # 聚合
        context = torch.matmul(attn_weights, V)  # (batch, n_heads, seq_len, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.W_o(context)
        return output
```

#### 4. layer_norm.py - Group Layer Normalization

```python
class GroupLayerNorm(nn.Module):
    """
    分组层归一化 (来自 MTGR)
    
    针对不同语义空间的 Token 使用不同的归一化参数
    
    Token 类型分组：
    - Group 0: USER Token (用户属性)
    - Group 1: ITEM Token (物品语义ID)
    - Group 2: ACTION Token (行为类型)
    - Group 3: CONTEXT Token (上下文信息)
    """
    
    def __init__(self, d_model: int, num_groups: int = 4, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.num_groups = num_groups
        self.eps = eps
        
        # 每组独立的 γ 和 β
        self.gamma = nn.Parameter(torch.ones(num_groups, d_model))
        self.beta = nn.Parameter(torch.zeros(num_groups, d_model))
    
    def forward(self, x: torch.Tensor, token_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            token_types: (batch, seq_len) 值域 [0, num_groups)
        
        Returns:
            normalized: (batch, seq_len, d_model)
        """
        # 标准 LayerNorm 计算
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 根据 token_types 选择对应的 γ 和 β
        gamma = self.gamma[token_types]  # (batch, seq_len, d_model)
        beta = self.beta[token_types]    # (batch, seq_len, d_model)
        
        return gamma * normalized + beta
```

#### 5. encoder_layer.py - 编码器层

```python
class EncoderLayer(nn.Module):
    """
    单层编码器
    
    结构: x → Attention → Add & GLN → FFN → Add & GLN → output
    """
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        
        self.attention = DotProductAggregatedAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        
        self.norm1 = GroupLayerNorm(config.d_model, config.num_groups)
        self.norm2 = GroupLayerNorm(config.d_model, config.num_groups)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        token_types: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-Attention + Residual + GLN
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out), token_types)
        
        # FFN + Residual + GLN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out, token_types)
        
        return x
```

#### 6. encoder.py - 完整编码器

```python
class UGTEncoder(nn.Module):
    """
    UGT 用户行为编码器
    
    将用户历史行为序列编码为用户表示向量
    """
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        
        # 输入嵌入
        self.input_embedding = InputEmbedding(config)
        
        # 编码器层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.n_layers)
        ])
        
        # 池化层（提取用户表示）
        self.pooler = nn.Linear(config.d_model, config.d_model)
    
    def forward(
        self,
        semantic_ids: List[torch.Tensor],
        positions: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            semantic_ids: [L1_ids, L2_ids, L3_ids], each (batch, seq_len)
            positions: (batch, seq_len)
            token_types: (batch, seq_len)
            attention_mask: (batch, seq_len), 1=有效, 0=padding
            time_features: (batch, seq_len, time_dim), 可选
        
        Returns:
            user_repr: (batch, d_model) 用户表示向量
        """
        # 输入嵌入
        hidden = self.input_embedding(semantic_ids, positions, token_types, time_features)
        
        # 编码器层
        for layer in self.layers:
            hidden = layer(hidden, token_types, attention_mask)
        
        # 取第一个 Token（类似 BERT 的 [CLS]）作为用户表示
        user_repr = self.pooler(hidden[:, 0, :])
        user_repr = torch.tanh(user_repr)
        
        return user_repr
    
    def get_sequence_output(
        self,
        semantic_ids: List[torch.Tensor],
        positions: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        获取完整序列输出（用于解码器的交叉注意力）
        
        Returns:
            sequence_output: (batch, seq_len, d_model)
        """
        hidden = self.input_embedding(semantic_ids, positions, token_types, None)
        
        for layer in self.layers:
            hidden = layer(hidden, token_types, attention_mask)
        
        return hidden
```

### 测试用例

```python
def test_encoder():
    config = EncoderConfig(d_model=512, n_heads=16, n_layers=6)
    encoder = UGTEncoder(config)
    
    batch_size = 32
    seq_len = 100
    
    # 准备输入
    l1_ids = torch.randint(0, 1024, (batch_size, seq_len))
    l2_ids = torch.randint(0, 4096, (batch_size, seq_len))
    l3_ids = torch.randint(0, 16384, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    token_types = torch.randint(0, 4, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 测试用户表示
    user_repr = encoder([l1_ids, l2_ids, l3_ids], positions, token_types, attention_mask)
    assert user_repr.shape == (batch_size, config.d_model)
    
    # 测试序列输出
    seq_output = encoder.get_sequence_output([l1_ids, l2_ids, l3_ids], positions, token_types, attention_mask)
    assert seq_output.shape == (batch_size, seq_len, config.d_model)
    
    print("All encoder tests passed!")
```

## 注意事项

1. **ReLU vs Softmax**: ReLU 注意力允许完全忽略某些 Token（权重为 0），而 Softmax 必须分配权重
2. **Group LayerNorm**: 确保 `token_types` 的值域与 `num_groups` 匹配
3. **序列长度**: 注意处理不同长度序列的 padding
4. **梯度稳定性**: 使用 Pre-Norm（先归一化再计算）可能更稳定

## 输出要求

请输出完整的可运行代码，包含：
1. 所有 Python 文件
2. 详细的中文注释
3. 单元测试
4. 使用示例

确保代码遵循 `algorithm/interfaces.py` 中定义的接口。

