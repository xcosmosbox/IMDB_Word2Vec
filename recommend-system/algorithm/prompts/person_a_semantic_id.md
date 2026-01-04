# Person A: Semantic ID 编码器 (RQ-VAE)

## 你的角色
你是一名深度学习算法工程师，负责实现生成式推荐系统的 **Semantic ID 编码器** 模块。

## 背景知识

Semantic ID 是一种层次化的物品编码方式，使用 RQ-VAE（残差向量量化）将连续的物品特征向量编码为离散的三层 ID：

- **L1 (粗粒度)**: 1024 个码本，表示大类（如"电影"、"商品"）
- **L2 (中粒度)**: 4096 个码本，表示子类（如"科幻电影"）
- **L3 (细粒度)**: 16384 个码本，表示具体物品

这种编码方式的优势：
1. 层次化结构支持从粗到细的生成
2. 离散化便于 Transformer 处理
3. 语义相近的物品具有相似的 ID 前缀

## 你的任务

在 `algorithm/semantic_id/` 目录下实现完整的 RQ-VAE 模块。

### 目录结构

```
algorithm/semantic_id/
├── __init__.py
├── config.py          # 配置类
├── codebook.py        # 单层向量量化器
├── rq_vae.py          # 残差向量量化器主体
├── encoder.py         # 特征编码网络
├── trainer.py         # 码本训练器
└── tests/
    └── test_rq_vae.py
```

### 接口要求

你必须实现 `interfaces.py` 中定义的 `SemanticIDEncoderInterface`：

```python
from algorithm.interfaces import SemanticIDEncoderInterface

class SemanticIDEncoder(SemanticIDEncoderInterface):
    def encode(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码物品特征为语义 ID
        
        Args:
            features: (batch_size, 256) 物品特征向量
        
        Returns:
            (L1_ids, L2_ids, L3_ids): 每个形状为 (batch_size,)
        """
        pass
    
    def decode(self, l1_ids, l2_ids, l3_ids) -> torch.Tensor:
        """从语义 ID 重建特征向量"""
        pass
    
    def get_codebook_embeddings(self, level: int) -> torch.Tensor:
        """获取码本嵌入矩阵"""
        pass
```

### 核心实现

#### 1. config.py - 配置类

```python
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SemanticIDConfig:
    """语义 ID 编码器配置"""
    embedding_dim: int = 256                          # 输入特征维度
    num_codebooks: int = 3                            # 码本数量（层数）
    codebook_sizes: Tuple[int, ...] = (1024, 4096, 16384)  # 各层码本大小
    commitment_cost: float = 0.25                     # 承诺损失权重 (β)
    ema_decay: float = 0.99                           # EMA 衰减率
    epsilon: float = 1e-5                             # 数值稳定性
```

#### 2. codebook.py - 单层向量量化器

```python
class VectorQuantizer(nn.Module):
    """
    单层向量量化器
    
    使用 EMA (Exponential Moving Average) 更新码本，避免码本坍塌问题
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, ema_decay: float = 0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.ema_decay = ema_decay
        
        # 码本嵌入
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # EMA 统计量
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, embedding_dim) 输入向量
        
        Returns:
            quantized: (batch_size, embedding_dim) 量化后的向量
            indices: (batch_size,) 最近邻索引
            commitment_loss: 承诺损失
        """
        # 计算距离，找最近邻
        distances = torch.cdist(x, self.embedding.weight)  # (batch, num_embeddings)
        indices = distances.argmin(dim=-1)  # (batch,)
        
        # 获取量化向量
        quantized = self.embedding(indices)  # (batch, embedding_dim)
        
        # 计算承诺损失
        commitment_loss = F.mse_loss(x, quantized.detach())
        
        # Straight-Through Estimator: 前向用量化值，反向用原始梯度
        quantized = x + (quantized - x).detach()
        
        # EMA 更新码本（仅训练时）
        if self.training:
            self._ema_update(x, indices)
        
        return quantized, indices, commitment_loss
    
    def _ema_update(self, x: torch.Tensor, indices: torch.Tensor):
        """EMA 更新码本"""
        # 统计每个码本被选中的次数
        encodings = F.one_hot(indices, self.num_embeddings).float()
        
        # 更新簇大小
        self.ema_cluster_size = self.ema_decay * self.ema_cluster_size + \
                                (1 - self.ema_decay) * encodings.sum(0)
        
        # 更新码本权重
        dw = encodings.T @ x  # (num_embeddings, embedding_dim)
        self.ema_w = self.ema_decay * self.ema_w + (1 - self.ema_decay) * dw
        
        # 归一化
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
        self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)
```

#### 3. rq_vae.py - 残差向量量化器

```python
class ResidualVectorQuantizer(nn.Module):
    """
    残差向量量化器 (RQ-VAE)
    
    通过多层残差量化生成层次化的 Semantic ID
    """
    
    def __init__(self, config: SemanticIDConfig):
        super().__init__()
        self.config = config
        
        # 创建多层量化器
        self.quantizers = nn.ModuleList([
            VectorQuantizer(size, config.embedding_dim, config.ema_decay)
            for size in config.codebook_sizes
        ])
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        编码为多层语义 ID
        
        Args:
            x: (batch_size, embedding_dim)
        
        Returns:
            Tuple of indices for each level
        """
        all_indices = []
        residual = x
        
        for quantizer in self.quantizers:
            _, indices, _ = quantizer(residual)
            all_indices.append(indices)
            
            # 计算残差
            quantized = quantizer.embedding(indices)
            residual = residual - quantized
        
        return tuple(all_indices)
    
    def decode(self, *indices: torch.Tensor) -> torch.Tensor:
        """
        从语义 ID 重建向量
        """
        reconstructed = torch.zeros_like(self.quantizers[0].embedding(indices[0]))
        
        for quantizer, idx in zip(self.quantizers, indices):
            reconstructed = reconstructed + quantizer.embedding(idx)
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        完整前向传播
        
        Returns:
            dict with: quantized, indices (list), reconstruction_loss, commitment_loss
        """
        all_indices = []
        total_commitment_loss = 0
        residual = x
        quantized_sum = torch.zeros_like(x)
        
        for quantizer in self.quantizers:
            quantized, indices, commitment_loss = quantizer(residual)
            all_indices.append(indices)
            total_commitment_loss += commitment_loss
            
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized.detach()  # 停止梯度
        
        # 重建损失
        reconstruction_loss = F.mse_loss(quantized_sum, x)
        
        return {
            'quantized': quantized_sum,
            'indices': all_indices,
            'reconstruction_loss': reconstruction_loss,
            'commitment_loss': total_commitment_loss,
            'total_loss': reconstruction_loss + self.config.commitment_cost * total_commitment_loss,
        }
```

### 损失函数

总损失：
```
L_total = L_reconstruction + β * L_commitment

其中：
- L_reconstruction = ||x - x_hat||^2   (重建损失)
- L_commitment = ||sg[z] - e||^2        (承诺损失，sg = stop_gradient)
- β = 0.25 (默认值)
```

### 测试用例

```python
def test_semantic_id_encoder():
    config = SemanticIDConfig()
    encoder = SemanticIDEncoder(config)
    
    # 测试编码
    features = torch.randn(32, 256)
    l1, l2, l3 = encoder.encode(features)
    
    assert l1.shape == (32,), f"L1 shape mismatch: {l1.shape}"
    assert l2.shape == (32,), f"L2 shape mismatch: {l2.shape}"
    assert l3.shape == (32,), f"L3 shape mismatch: {l3.shape}"
    
    assert l1.max() < 1024, "L1 out of range"
    assert l2.max() < 4096, "L2 out of range"
    assert l3.max() < 16384, "L3 out of range"
    
    # 测试解码
    reconstructed = encoder.decode(l1, l2, l3)
    assert reconstructed.shape == (32, 256)
    
    # 测试重建误差
    error = F.mse_loss(reconstructed, features)
    print(f"Reconstruction error: {error.item():.4f}")
    
    print("All tests passed!")
```

### 训练码本

码本需要在物品特征数据上预训练：

```python
def train_codebook(encoder, item_features, num_epochs=10):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        for batch in DataLoader(item_features, batch_size=256):
            optimizer.zero_grad()
            output = encoder.rq_vae(batch)
            loss = output['total_loss']
            loss.backward()
            optimizer.step()
        
        # 监控码本利用率
        utilization = compute_codebook_utilization(encoder)
        print(f"Epoch {epoch}: Loss={loss:.4f}, Utilization={utilization:.2%}")
```

## 注意事项

1. **码本初始化**：使用均匀分布初始化，避免码本坍塌
2. **EMA 更新**：使用指数移动平均更新码本，比梯度下降更稳定
3. **Straight-Through Estimator**：前向传播使用量化值，反向传播使用原始梯度
4. **码本利用率**：监控每个码本条目的使用频率，避免死码本

## 输出要求

请输出完整的可运行代码，包含：
1. 所有 Python 文件
2. 详细的中文注释
3. 单元测试
4. 使用示例

确保代码遵循 `algorithm/interfaces.py` 中定义的接口。

