# UGT Decoder 模块开发文档

> **版本**: v1.0  
> **更新日期**: 2026-01-04  
> **作者**: Person C  
> **对应架构文档**: 《生成式推荐系统架构设计》第三章

---

## 目录

1. [模块概述](#1-模块概述)
2. [快速开始](#2-快速开始)
3. [目录结构](#3-目录结构)
4. [核心组件详解](#4-核心组件详解)
5. [API 参考](#5-api-参考)
6. [使用示例](#6-使用示例)
7. [扩展开发指南](#7-扩展开发指南)
8. [测试指南](#8-测试指南)
9. [常见问题](#9-常见问题)

---

## 1. 模块概述

### 1.1 功能定位

UGT Decoder 是生成式推荐系统的核心解码器模块，负责：

- **从用户表示生成推荐**：接收编码器输出的用户表示向量，自回归地生成推荐物品的语义 ID 序列
- **层次化生成**：依次生成 L1（粗粒度类目）→ L2（细粒度属性）→ L3（实例区分）
- **MoE 增强**：使用 Mixture of Experts 处理不同推荐场景

### 1.2 技术特点

| 特性 | 说明 |
|------|------|
| **MoE-Enhanced FFN** | 稀疏激活的专家网络，不同专家专注于不同推荐场景 |
| **分组层归一化 (GLN)** | 针对不同语义空间的 Token 使用不同归一化参数 |
| **因果注意力** | 确保自回归生成的正确性，防止信息泄露 |
| **交叉注意力** | 连接编码器和解码器，融合用户行为信息 |
| **多种生成策略** | 支持 Beam Search、多样性搜索、核采样等 |

### 1.3 接口规范

本模块实现 `algorithm/interfaces.py` 中定义的 `RecommendDecoderInterface`：

```python
class RecommendDecoderInterface(ABC):
    @abstractmethod
    def forward(
        self,
        encoder_output: torch.Tensor,
        target_semantic_ids: Optional[List[torch.Tensor]] = None,
        target_positions: Optional[torch.Tensor] = None,
        target_token_types: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练模式：返回 (L1_logits, L2_logits, L3_logits, aux_loss)"""
        pass
    
    @abstractmethod
    def generate(
        self,
        encoder_output: torch.Tensor,
        num_recommendations: int = 20,
        beam_size: int = 4,
        temperature: float = 1.0,
    ) -> List[List[Tuple[int, int, int]]]:
        """推理模式：返回推荐列表 [(L1, L2, L3), ...]"""
        pass
```

---

## 2. 快速开始

### 2.1 安装依赖

```bash
pip install torch>=2.0.0
pip install pytest  # 用于运行测试
```

### 2.2 基础使用

```python
import torch
from algorithm.decoder import UGTDecoder, DecoderConfig, BeamSearchGenerator

# 1. 创建配置
config = DecoderConfig.medium()  # 或 .small(), .large()

# 2. 创建解码器
decoder = UGTDecoder(config)

# 3. 准备输入数据
batch_size = 32
src_len = 100
encoder_output = torch.randn(batch_size, src_len, config.d_model)

# 4. 训练模式
tgt_len = 20
target_l1 = torch.randint(0, 1024, (batch_size, tgt_len))
target_l2 = torch.randint(0, 4096, (batch_size, tgt_len))
target_l3 = torch.randint(0, 16384, (batch_size, tgt_len))

l1_logits, l2_logits, l3_logits, aux_loss = decoder(
    encoder_output=encoder_output,
    target_semantic_ids=[target_l1, target_l2, target_l3],
)

# 5. 推理模式
decoder.eval()
recommendations = decoder.generate(
    encoder_output=encoder_output,
    num_recommendations=20,
    beam_size=4,
)
```

---

## 3. 目录结构

```
decoder/
├── __init__.py           # 模块导出和工厂函数
├── config.py             # 配置类 DecoderConfig
├── moe.py                # Mixture of Experts 实现
│   ├── Expert            # 单个专家网络
│   ├── Router            # 路由网络
│   ├── MoEFeedForward    # MoE 前馈网络
│   └── SharedExpertMoE   # 带共享专家的 MoE
├── cross_attention.py    # 注意力机制
│   ├── CausalSelfAttention   # 因果自注意力
│   ├── CrossAttention        # 交叉注意力
│   ├── GroupLayerNorm        # 分组层归一化
│   └── FeedForward           # 标准前馈网络
├── decoder_layer.py      # 解码器层
│   ├── DecoderLayer              # 完整解码器层
│   ├── DecoderLayerWithoutCrossAttention  # 无交叉注意力版本
│   └── DecoderInputEmbedding     # 输入嵌入层
├── decoder.py            # 完整解码器
│   ├── UGTDecoder            # 主解码器类
│   └── UGTDecoderForInference  # 推理优化版本
├── generator.py          # 生成策略
│   ├── BeamSearchGenerator       # Beam Search
│   ├── DiverseBeamSearchGenerator  # 多样性搜索
│   └── NucleusSamplingGenerator    # 核采样
├── tests/
│   ├── __init__.py
│   └── test_decoder.py   # 单元测试
└── README.md             # 本文档
```

---

## 4. 核心组件详解

### 4.1 配置类 (config.py)

`DecoderConfig` 是解码器的配置中心，包含所有超参数：

```python
@dataclass
class DecoderConfig:
    # 模型基础配置
    d_model: int = 512              # 隐藏层维度
    n_heads: int = 16               # 注意力头数
    n_layers: int = 12              # 解码器层数
    d_ff: int = 2048                # FFN 中间维度
    max_seq_len: int = 1024         # 最大序列长度
    dropout: float = 0.1            # Dropout 率
    
    # MoE 配置
    num_experts: int = 16           # 专家数量
    top_k_experts: int = 4          # 每次激活的专家数
    moe_loss_weight: float = 0.01   # 负载均衡损失权重
    
    # 语义 ID 配置
    codebook_sizes: Tuple[int, int, int] = (1024, 4096, 16384)
    
    # Token 类型
    num_token_types: int = 4        # USER=0, ITEM=1, ACTION=2, CONTEXT=3
    num_groups: int = 4             # GLN 分组数量
```

**预设配置**：

| 配置 | d_model | n_heads | n_layers | 用途 |
|------|---------|---------|----------|------|
| `DecoderConfig.small()` | 256 | 8 | 6 | 调试/测试 |
| `DecoderConfig.medium()` | 512 | 16 | 12 | 实验 |
| `DecoderConfig.large()` | 1024 | 32 | 24 | 生产环境 |

### 4.2 Mixture of Experts (moe.py)

MoE 是本模块的核心创新之一，来自快手 OneRec 论文：

```
┌─────────────────────────────────────────────────────────┐
│                   MoE FFN 结构                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  输入 x ────→ Router ────→ Top-K 专家选择               │
│                  │                                      │
│                  ▼                                      │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐    (16 个专家)        │
│  │ E_1 │ │ E_2 │ │ ... │ │E_16 │                       │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                       │
│     └───────┴───────┴───────┘                          │
│               │                                         │
│               ▼ (加权求和)                              │
│            输出 y                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**关键特性**：

1. **稀疏激活**：每个 token 只激活 Top-K 个专家，大大减少计算量
2. **负载均衡损失**：防止所有 token 都路由到少数专家（专家坍塌）

```python
# 负载均衡损失公式
L_balance = α * num_experts * Σ(f_i * P_i)
# f_i: 第 i 个专家被选中的 token 比例
# P_i: 第 i 个专家的平均路由概率
```

### 4.3 注意力机制 (cross_attention.py)

#### 4.3.1 因果自注意力

确保每个位置只能看到自己和之前的位置：

```python
# 因果掩码示例 (4x4)
[[True,  False, False, False],
 [True,  True,  False, False],
 [True,  True,  True,  False],
 [True,  True,  True,  True ]]
```

#### 4.3.2 交叉注意力

连接编码器和解码器：
- **Query**: 来自解码器（要生成什么）
- **Key/Value**: 来自编码器（用户历史信息）

#### 4.3.3 分组层归一化 (GLN)

针对不同类型的 Token 使用不同的归一化参数：

```python
# Token 类型分组
Group 0: 用户行为 Token
Group 1: 物品属性 Token
Group 2: 动作 Token
Group 3: 上下文 Token
```

### 4.4 解码器层 (decoder_layer.py)

单层解码器结构：

```
x → Causal Self-Attention → Add & GLN
  → Cross Attention → Add & GLN
  → MoE FFN → Add & GLN → output
```

### 4.5 输入嵌入 (decoder_layer.py)

组合多种嵌入：

```python
E_input = E_L1 + E_L2 + E_L3 + E_position + E_type
```

- **E_L1, E_L2, E_L3**: 三层语义 ID 嵌入
- **E_position**: 位置编码
- **E_type**: Token 类型嵌入

### 4.6 生成器 (generator.py)

提供三种生成策略：

| 策略 | 特点 | 适用场景 |
|------|------|----------|
| **BeamSearchGenerator** | 高质量，稳定 | 追求最优推荐 |
| **DiverseBeamSearchGenerator** | 多样性高 | 需要多样化推荐 |
| **NucleusSamplingGenerator** | 平衡质量和多样性 | 探索性推荐 |

---

## 5. API 参考

### 5.1 UGTDecoder

#### 构造函数

```python
UGTDecoder(config: DecoderConfig)
```

#### forward 方法

```python
def forward(
    self,
    encoder_output: torch.Tensor,           # (batch, src_len, d_model)
    target_semantic_ids: List[torch.Tensor], # [L1, L2, L3] 每个 (batch, tgt_len)
    target_positions: torch.Tensor = None,   # (batch, tgt_len)
    target_token_types: torch.Tensor = None, # (batch, tgt_len)
    encoder_mask: torch.Tensor = None,       # (batch, src_len)
    target_mask: torch.Tensor = None,        # (batch, tgt_len)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回:
        l1_logits: (batch, tgt_len, 1024)   - L1 预测
        l2_logits: (batch, tgt_len, 4096)   - L2 预测
        l3_logits: (batch, tgt_len, 16384)  - L3 预测
        aux_loss: scalar                     - MoE 辅助损失
    """
```

#### generate 方法

```python
def generate(
    self,
    encoder_output: torch.Tensor,    # (batch, src_len, d_model)
    num_recommendations: int = 20,   # 生成数量
    beam_size: int = 4,              # Beam 宽度
    temperature: float = 1.0,        # 采样温度
    encoder_mask: torch.Tensor = None,
) -> List[List[Tuple[int, int, int]]]:
    """
    返回:
        recommendations[b][i] = (L1_id, L2_id, L3_id)
        - 外层 List 长度为 batch_size
        - 内层 List 长度为 num_recommendations
    """
```

#### compute_loss 方法

```python
def compute_loss(
    self,
    l1_logits, l2_logits, l3_logits,  # 模型输出
    target_l1, target_l2, target_l3,   # 目标
    aux_loss,                          # MoE 辅助损失
    mask=None,
) -> Dict[str, torch.Tensor]:
    """
    返回损失字典:
        - 'total_loss': 总损失
        - 'ntp_loss': Next Token Prediction 损失
        - 'l1_loss', 'l2_loss', 'l3_loss': 各层损失
        - 'aux_loss': MoE 辅助损失
    """
```

### 5.2 MoEFeedForward

```python
class MoEFeedForward(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
    
    def get_aux_loss(self) -> torch.Tensor:
        """获取负载均衡辅助损失（标量）"""
    
    def get_expert_utilization(self) -> dict:
        """获取专家利用率统计"""
```

### 5.3 生成器

```python
# Beam Search
generator = BeamSearchGenerator(decoder, config)
recommendations = generator.generate(
    encoder_output,
    num_recommendations=20,
    beam_size=4,
    temperature=1.0,
    diversity_penalty=0.0,
)

# 多样性 Beam Search
generator = DiverseBeamSearchGenerator(decoder, config)
recommendations = generator.generate(
    encoder_output,
    num_recommendations=20,
    num_beam_groups=4,
    beam_size_per_group=4,
    diversity_penalty=0.5,
)

# 核采样
generator = NucleusSamplingGenerator(decoder, config)
recommendations = generator.generate(
    encoder_output,
    num_recommendations=20,
    top_p=0.9,
    temperature=1.0,
)
```

---

## 6. 使用示例

### 6.1 完整训练流程

```python
import torch
import torch.optim as optim
from algorithm.decoder import UGTDecoder, DecoderConfig

# 配置
config = DecoderConfig.medium()
decoder = UGTDecoder(config).cuda()
optimizer = optim.AdamW(decoder.parameters(), lr=1e-4)

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        encoder_output = batch['encoder_output'].cuda()
        target_l1 = batch['target_l1'].cuda()
        target_l2 = batch['target_l2'].cuda()
        target_l3 = batch['target_l3'].cuda()
        
        # 前向传播
        l1_logits, l2_logits, l3_logits, aux_loss = decoder(
            encoder_output=encoder_output,
            target_semantic_ids=[target_l1, target_l2, target_l3],
        )
        
        # 计算损失
        losses = decoder.compute_loss(
            l1_logits, l2_logits, l3_logits,
            target_l1, target_l2, target_l3,
            aux_loss,
        )
        
        # 反向传播
        optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        
        print(f"Loss: {losses['total_loss']:.4f}, "
              f"NTP: {losses['ntp_loss']:.4f}, "
              f"MoE: {losses['aux_loss']:.4f}")
```

### 6.2 推理服务

```python
import torch
from algorithm.decoder import UGTDecoder, DecoderConfig, BeamSearchGenerator

# 加载模型
config = DecoderConfig.medium()
decoder = UGTDecoder(config)
decoder.load_state_dict(torch.load('decoder.pt'))
decoder.eval().cuda()

# 创建生成器
generator = BeamSearchGenerator(decoder, config)

@torch.no_grad()
def recommend(user_encoder_output: torch.Tensor, top_k: int = 20):
    """
    为用户生成推荐
    
    Args:
        user_encoder_output: 用户编码器输出 (1, seq_len, d_model)
        top_k: 返回的推荐数量
    
    Returns:
        List[Tuple[int, int, int]]: 推荐列表 [(L1, L2, L3), ...]
    """
    recommendations = generator.generate(
        encoder_output=user_encoder_output.cuda(),
        num_recommendations=top_k,
        beam_size=4,
        temperature=0.8,  # 略低温度提高质量
    )
    return recommendations[0]  # 返回第一个 batch 的结果
```

### 6.3 与编码器集成

```python
from algorithm.encoder import UGTEncoder  # Person B 实现
from algorithm.decoder import UGTDecoder, DecoderConfig

class UGTModel(nn.Module):
    """完整的编码器-解码器模型"""
    
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder = UGTEncoder(encoder_config)
        self.decoder = UGTDecoder(decoder_config)
    
    def forward(self, encoder_inputs, decoder_targets):
        # 编码用户历史
        encoder_output = self.encoder.get_sequence_output(**encoder_inputs)
        
        # 解码生成推荐
        l1_logits, l2_logits, l3_logits, aux_loss = self.decoder(
            encoder_output=encoder_output,
            target_semantic_ids=decoder_targets['semantic_ids'],
        )
        
        return l1_logits, l2_logits, l3_logits, aux_loss
    
    @torch.no_grad()
    def recommend(self, encoder_inputs, num_recommendations=20):
        self.eval()
        encoder_output = self.encoder.get_sequence_output(**encoder_inputs)
        return self.decoder.generate(
            encoder_output=encoder_output,
            num_recommendations=num_recommendations,
        )
```

---

## 7. 扩展开发指南

### 7.1 添加新的专家类型

在 `moe.py` 中扩展 `Expert` 类：

```python
class GatedExpert(nn.Module):
    """带门控的专家网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(x))
        up = F.gelu(self.up(x))
        return self.dropout(self.down(gate * up))
```

### 7.2 添加新的生成策略

在 `generator.py` 中添加新的生成器：

```python
class ConstrainedGenerator:
    """带约束的生成器（如类目限制）"""
    
    def __init__(self, decoder, config, allowed_l1_ids=None):
        self.decoder = decoder
        self.config = config
        self.allowed_l1_ids = allowed_l1_ids
    
    def generate(self, encoder_output, **kwargs):
        # 实现带约束的生成逻辑
        pass
```

### 7.3 自定义注意力机制

在 `cross_attention.py` 中扩展：

```python
class LinearAttention(nn.Module):
    """
    线性注意力（O(n) 复杂度）
    
    适用于超长序列场景
    """
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        # 实现线性注意力
        pass
```

### 7.4 配置扩展

在 `config.py` 中添加新的配置项：

```python
@dataclass
class ExtendedDecoderConfig(DecoderConfig):
    """扩展配置"""
    
    # 新增配置项
    use_rotary_embedding: bool = False
    use_linear_attention: bool = False
    expert_type: str = "standard"  # "standard", "gated", "moe"
```

---

## 8. 测试指南

### 8.1 运行测试

```bash
# 运行所有测试
pytest recommend-system/algorithm/decoder/tests/ -v

# 运行特定测试类
pytest recommend-system/algorithm/decoder/tests/test_decoder.py::TestMoE -v

# 运行带覆盖率
pytest recommend-system/algorithm/decoder/tests/ --cov=decoder --cov-report=html
```

### 8.2 测试类别

| 测试类 | 测试内容 |
|--------|----------|
| `TestDecoderConfig` | 配置类验证、序列化 |
| `TestMoE` | 专家网络、路由、负载均衡 |
| `TestAttention` | 自注意力、交叉注意力、GLN |
| `TestDecoderLayer` | 解码器层、输入嵌入 |
| `TestUGTDecoder` | 前向传播、生成、损失计算 |
| `TestGenerators` | 各种生成策略 |
| `TestInterfaceCompatibility` | 接口兼容性验证 |
| `TestGradients` | 梯度流动测试 |

### 8.3 编写新测试

```python
class TestNewFeature:
    """新功能测试"""
    
    def test_feature_basic(self, small_config, device):
        """基础功能测试"""
        # 准备数据
        # 调用功能
        # 验证结果
        pass
    
    def test_feature_edge_cases(self, small_config, device):
        """边界情况测试"""
        pass
```

---

## 9. 常见问题

### Q1: MoE 辅助损失过高怎么办？

**原因**：专家分布不均匀，部分专家过载

**解决方案**：
1. 降低 `moe_loss_weight`（如从 0.01 降到 0.001）
2. 增加 `top_k_experts`
3. 增加训练数据多样性

### Q2: 生成结果重复度高怎么办？

**解决方案**：
1. 提高 `temperature`（如从 1.0 提到 1.2）
2. 使用 `DiverseBeamSearchGenerator` 并增加 `diversity_penalty`
3. 使用核采样生成器

### Q3: 推理速度慢怎么办？

**解决方案**：
1. 使用 `UGTDecoderForInference` 启用 KV 缓存
2. 减小 `beam_size`
3. 使用 `DecoderConfig.small()` 配置
4. 启用 Flash Attention（需要硬件支持）

### Q4: 显存不足怎么办？

**解决方案**：
1. 减小 `batch_size`
2. 使用梯度检查点（设置 `use_gradient_checkpointing=True`）
3. 使用混合精度训练
4. 减少模型层数或隐藏维度

### Q5: 如何监控专家利用率？

```python
moe = decoder.layers[0].moe_ffn
stats = moe.get_expert_utilization()
print(f"专家数: {stats['num_experts']}, Top-K: {stats['top_k']}")
print(f"辅助损失: {stats['aux_loss']:.4f}")
```

---

## 附录 A: 模型参数量估算

| 配置 | 参数量估算 | 推荐 GPU |
|------|-----------|----------|
| small | ~10M | 4GB |
| medium | ~50M | 8GB |
| large | ~200M | 24GB+ |

## 附录 B: 相关论文

1. **HSTU**: Generative Retrieval for Long Sequences (Meta, 2024)
2. **OneRec**: A Unified Sequence-to-Sequence Framework (Kuaishou, 2024)
3. **MTGR**: Multi-Task Generative Retrieval (Alibaba, 2024)
4. **Switch Transformer**: Scaling to Trillion Parameter Models (Google, 2022)

---

*如有问题，请联系 Person C 或查阅架构设计文档。*

