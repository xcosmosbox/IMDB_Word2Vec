# Person C: UGT Decoder + MoE（推荐生成器）

## 你的角色
你是一名深度学习算法工程师，负责实现生成式推荐系统的 **UGT Decoder** 模块。

## 背景知识

UGT Decoder 负责从用户表示向量自回归地生成推荐物品的 Semantic ID 序列。

### 核心创新点

1. **MoE-Enhanced FFN (来自快手 OneRec)**
   - 使用稀疏激活的 Mixture of Experts 网络
   - 不同专家专注于不同推荐场景（电影/商品/视频等）
   - 通过负载均衡损失避免专家坍塌

2. **层次化生成**
   - 依次生成 L1 → L2 → L3
   - 粗粒度到细粒度的推荐

## 你的任务

在 `algorithm/decoder/` 目录下实现完整的 Decoder 模块。

### 目录结构

```
algorithm/decoder/
├── __init__.py
├── config.py           # 配置类
├── cross_attention.py  # 交叉注意力
├── moe.py              # Mixture of Experts
├── decoder_layer.py    # 解码器层
├── decoder.py          # 完整解码器
├── generator.py        # 自回归生成器
└── tests/
    └── test_decoder.py
```

### 接口要求

你必须实现 `interfaces.py` 中定义的 `RecommendDecoderInterface`：

```python
from algorithm.interfaces import RecommendDecoderInterface

class UGTDecoder(nn.Module, RecommendDecoderInterface):
    def forward(
        self,
        encoder_output: torch.Tensor,
        target_semantic_ids: Optional[List[torch.Tensor]] = None,
        target_positions: Optional[torch.Tensor] = None,
        target_token_types: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (L1_logits, L2_logits, L3_logits, aux_loss)"""
        pass
    
    def generate(
        self,
        encoder_output: torch.Tensor,
        num_recommendations: int = 20,
        beam_size: int = 4,
        temperature: float = 1.0,
    ) -> List[List[Tuple[int, int, int]]]:
        """自回归生成推荐列表"""
        pass
```

### 核心实现

#### 1. config.py - 配置类

```python
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DecoderConfig:
    """解码器配置"""
    d_model: int = 512              # 隐藏层维度
    n_heads: int = 16               # 注意力头数
    n_layers: int = 12              # 解码器层数
    d_ff: int = 2048                # FFN 中间维度
    max_seq_len: int = 1024         # 最大序列长度
    dropout: float = 0.1            # Dropout 率
    
    # MoE 配置
    num_experts: int = 16           # 专家数量
    top_k_experts: int = 4          # 每次激活的专家数
    expert_capacity_factor: float = 1.25  # 专家容量因子
    moe_loss_weight: float = 0.01   # MoE 负载均衡损失权重
    
    # 语义 ID 配置
    codebook_sizes: Tuple[int, int, int] = (1024, 4096, 16384)
    
    # Token 类型
    num_token_types: int = 4
    num_groups: int = 4
```

#### 2. moe.py - Mixture of Experts

```python
class Expert(nn.Module):
    """单个专家网络（FFN）"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEFeedForward(nn.Module):
    """
    Mixture of Experts 前馈网络
    
    核心特点：
    1. 稀疏激活：每次只激活 Top-K 个专家
    2. 负载均衡：通过辅助损失避免专家坍塌
    3. 专家专业化：不同专家处理不同类型的推荐
    
    参考论文：
    - Switch Transformer (Google)
    - OneRec (快手)
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.d_model = config.d_model
        
        # 路由网络
        self.router = nn.Linear(config.d_model, config.num_experts)
        
        # 专家网络
        self.experts = nn.ModuleList([
            Expert(config.d_model, config.d_ff, config.dropout)
            for _ in range(config.num_experts)
        ])
        
        # 记录辅助损失
        self.aux_loss = 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)
        
        # 计算路由分数
        router_logits = self.router(x_flat)  # (batch * seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 选择 Top-K 专家
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # 归一化
        
        # 计算负载均衡损失
        self.aux_loss = self._compute_balance_loss(router_probs, top_k_indices)
        
        # 专家计算
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # 找到选择了第 i 个专家的 token
            mask = (top_k_indices == i).any(dim=-1)  # (batch * seq_len,)
            if mask.sum() == 0:
                continue
            
            # 获取对应的权重
            expert_weight = torch.where(
                top_k_indices == i,
                top_k_probs,
                torch.zeros_like(top_k_probs)
            ).sum(dim=-1)  # (batch * seq_len,)
            
            # 计算专家输出
            expert_out = expert(x_flat)  # (batch * seq_len, d_model)
            output += expert_out * expert_weight.unsqueeze(-1)
        
        return output.view(batch_size, seq_len, d_model)
    
    def _compute_balance_loss(
        self,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算负载均衡损失
        
        目标：让所有专家被均匀使用
        
        L_balance = α * Σ(f_i * P_i)
        - f_i: 第 i 个专家被选中的 token 比例
        - P_i: 第 i 个专家的平均路由概率
        """
        num_tokens = router_probs.shape[0]
        
        # 计算每个专家被选中的频率
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()  # (N, top_k, E)
        f = expert_mask.sum(dim=(0, 1)) / (num_tokens * self.top_k)  # (E,)
        
        # 计算平均路由概率
        P = router_probs.mean(dim=0)  # (E,)
        
        # 负载均衡损失
        balance_loss = self.num_experts * (f * P).sum()
        
        return balance_loss
    
    def get_aux_loss(self) -> torch.Tensor:
        """获取辅助损失（用于训练）"""
        return self.aux_loss
```

#### 3. cross_attention.py - 交叉注意力

```python
class CrossAttention(nn.Module):
    """
    交叉注意力（编码器-解码器连接）
    
    Query 来自解码器，Key/Value 来自编码器
    使用标准 Softmax 注意力
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
        query: torch.Tensor,              # (batch, tgt_len, d_model)
        encoder_output: torch.Tensor,     # (batch, src_len, d_model)
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, tgt_len, _ = query.shape
        src_len = encoder_output.shape[1]
        
        Q = self.W_q(query).view(batch_size, tgt_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(encoder_output).view(batch_size, src_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(encoder_output).view(batch_size, src_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 标准 Softmax 注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if encoder_mask is not None:
            scores = scores.masked_fill(encoder_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        
        return self.W_o(context)
```

#### 4. decoder_layer.py - 解码器层

```python
class DecoderLayer(nn.Module):
    """
    单层解码器
    
    结构:
    x → Causal Self-Attention → Add & GLN
      → Cross Attention → Add & GLN
      → MoE FFN → Add & GLN → output
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        
        # 因果自注意力
        self.self_attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        
        # 交叉注意力
        self.cross_attn = CrossAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        
        # MoE FFN
        self.moe_ffn = MoEFeedForward(config)
        
        # 归一化层
        self.norm1 = GroupLayerNorm(config.d_model, config.num_groups)
        self.norm2 = GroupLayerNorm(config.d_model, config.num_groups)
        self.norm3 = GroupLayerNorm(config.d_model, config.num_groups)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        token_types: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 因果自注意力
        attn_out = self.self_attn(x, mask=self_mask)
        x = self.norm1(x + self.dropout(attn_out), token_types)
        
        # 交叉注意力
        cross_out = self.cross_attn(x, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout(cross_out), token_types)
        
        # MoE FFN
        ffn_out = self.moe_ffn(x)
        x = self.norm3(x + ffn_out, token_types)
        
        # 返回 MoE 辅助损失
        aux_loss = self.moe_ffn.get_aux_loss()
        
        return x, aux_loss
```

#### 5. decoder.py - 完整解码器

```python
class UGTDecoder(nn.Module):
    """
    UGT 推荐生成解码器
    
    从用户表示生成推荐物品的语义 ID 序列
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        
        # 输入嵌入（与编码器共享或独立）
        self.input_embedding = DecoderInputEmbedding(config)
        
        # 解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_layers)
        ])
        
        # 三个输出头，分别预测 L1, L2, L3
        self.output_heads = nn.ModuleList([
            nn.Linear(config.d_model, size)
            for size in config.codebook_sizes
        ])
    
    def forward(
        self,
        encoder_output: torch.Tensor,
        target_semantic_ids: Optional[List[torch.Tensor]] = None,
        target_positions: Optional[torch.Tensor] = None,
        target_token_types: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练模式：计算目标序列的预测 logits
        """
        # 输入嵌入
        hidden = self.input_embedding(target_semantic_ids, target_positions, target_token_types)
        
        # 因果掩码
        tgt_len = hidden.shape[1]
        causal_mask = self._generate_causal_mask(tgt_len, hidden.device)
        
        # 解码器层
        total_aux_loss = 0
        for layer in self.layers:
            hidden, aux_loss = layer(
                hidden, encoder_output, target_token_types,
                self_mask=causal_mask,
            )
            total_aux_loss += aux_loss
        
        # 输出 logits
        l1_logits = self.output_heads[0](hidden)
        l2_logits = self.output_heads[1](hidden)
        l3_logits = self.output_heads[2](hidden)
        
        avg_aux_loss = total_aux_loss / len(self.layers)
        
        return l1_logits, l2_logits, l3_logits, avg_aux_loss
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """生成因果掩码"""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask == 0  # True 表示可见
```

#### 6. generator.py - 自回归生成器

```python
class BeamSearchGenerator:
    """
    Beam Search 推荐生成器
    
    层次化生成：L1 → L2 → L3
    """
    
    def __init__(self, decoder: UGTDecoder, config: DecoderConfig):
        self.decoder = decoder
        self.config = config
    
    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        num_recommendations: int = 20,
        beam_size: int = 4,
        temperature: float = 1.0,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        生成推荐列表
        
        Returns:
            List[List[Tuple[L1, L2, L3]]]
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device
        
        all_recommendations = []
        
        for b in range(batch_size):
            enc_out = encoder_output[b:b+1]  # (1, src_len, d_model)
            recommendations = self._beam_search_single(
                enc_out, num_recommendations, beam_size, temperature
            )
            all_recommendations.append(recommendations)
        
        return all_recommendations
    
    def _beam_search_single(
        self,
        encoder_output: torch.Tensor,
        num_recommendations: int,
        beam_size: int,
        temperature: float,
    ) -> List[Tuple[int, int, int]]:
        """单个样本的 Beam Search"""
        device = encoder_output.device
        recommendations = []
        
        # 生成 num_recommendations 个推荐
        for _ in range(num_recommendations):
            # Step 1: 生成 L1
            l1_id = self._generate_level(encoder_output, [], 0, beam_size, temperature)
            
            # Step 2: 生成 L2（条件于 L1）
            l2_id = self._generate_level(encoder_output, [l1_id], 1, beam_size, temperature)
            
            # Step 3: 生成 L3（条件于 L1, L2）
            l3_id = self._generate_level(encoder_output, [l1_id, l2_id], 2, beam_size, temperature)
            
            recommendations.append((l1_id, l2_id, l3_id))
        
        return recommendations
    
    def _generate_level(
        self,
        encoder_output: torch.Tensor,
        prefix: List[int],
        level: int,
        beam_size: int,
        temperature: float,
    ) -> int:
        """生成指定层级的 ID"""
        # 构建输入（包含已生成的前缀）
        # ... 实现细节
        
        # 获取 logits
        with torch.no_grad():
            logits = self.decoder.output_heads[level](hidden)
        
        # 温度采样
        probs = F.softmax(logits / temperature, dim=-1)
        
        # 采样或取 argmax
        if temperature > 0:
            idx = torch.multinomial(probs.squeeze(), 1).item()
        else:
            idx = probs.argmax().item()
        
        return idx
```

### 测试用例

```python
def test_decoder():
    config = DecoderConfig(d_model=512, n_heads=16, n_layers=6, num_experts=8, top_k_experts=2)
    decoder = UGTDecoder(config)
    
    batch_size = 32
    src_len = 100
    tgt_len = 20
    
    # 模拟编码器输出
    encoder_output = torch.randn(batch_size, src_len, config.d_model)
    
    # 训练模式测试
    target_l1 = torch.randint(0, 1024, (batch_size, tgt_len))
    target_l2 = torch.randint(0, 4096, (batch_size, tgt_len))
    target_l3 = torch.randint(0, 16384, (batch_size, tgt_len))
    target_positions = torch.arange(tgt_len).unsqueeze(0).expand(batch_size, -1)
    target_types = torch.ones(batch_size, tgt_len, dtype=torch.long)
    
    l1_logits, l2_logits, l3_logits, aux_loss = decoder(
        encoder_output,
        [target_l1, target_l2, target_l3],
        target_positions,
        target_types,
    )
    
    assert l1_logits.shape == (batch_size, tgt_len, 1024)
    assert l2_logits.shape == (batch_size, tgt_len, 4096)
    assert l3_logits.shape == (batch_size, tgt_len, 16384)
    assert aux_loss.ndim == 0  # scalar
    
    # 生成模式测试
    generator = BeamSearchGenerator(decoder, config)
    recommendations = generator.generate(encoder_output, num_recommendations=10, beam_size=4)
    
    assert len(recommendations) == batch_size
    assert len(recommendations[0]) == 10
    assert len(recommendations[0][0]) == 3
    
    print("All decoder tests passed!")
```

## 注意事项

1. **MoE 负载均衡**: 辅助损失很重要，否则会出现专家坍塌（所有 token 都路由到少数专家）
2. **因果掩码**: 解码器自注意力必须使用因果掩码，防止看到未来信息
3. **层次生成**: L2 的生成应该条件于 L1，L3 应该条件于 L1 和 L2
4. **温度采样**: 温度参数控制生成多样性，温度越高越多样

## 输出要求

请输出完整的可运行代码，包含：
1. 所有 Python 文件
2. 详细的中文注释
3. 单元测试
4. 使用示例

确保代码遵循 `algorithm/interfaces.py` 中定义的接口。

