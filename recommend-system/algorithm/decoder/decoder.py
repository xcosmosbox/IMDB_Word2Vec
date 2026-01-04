"""
UGT 推荐生成解码器

完整的解码器实现，从用户表示生成推荐物品的语义 ID 序列。

核心功能：
1. 训练模式：计算目标序列的预测 logits
2. 推理模式：自回归生成推荐列表

实现接口：algorithm.interfaces.RecommendDecoderInterface

对应架构文档: 《生成式推荐系统架构设计》第三章
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any

from .config import DecoderConfig
from .decoder_layer import DecoderLayer, DecoderInputEmbedding


class UGTDecoder(nn.Module):
    """
    UGT 推荐生成解码器
    
    从用户表示（编码器输出）生成推荐物品的语义 ID 序列。
    
    核心特点：
    1. 层次化生成：依次生成 L1 → L2 → L3
    2. MoE 增强：使用 Mixture of Experts 处理不同推荐场景
    3. 因果注意力：确保自回归生成的正确性
    
    Args:
        config: 解码器配置
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        
        self.config = config
        
        # 输入嵌入层
        self.input_embedding = DecoderInputEmbedding(config)
        
        # 解码器层堆叠
        self.layers = nn.ModuleList([
            DecoderLayer(config, layer_idx=i)
            for i in range(config.n_layers)
        ])
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # 三个输出头，分别预测 L1, L2, L3 语义 ID
        self.output_heads = nn.ModuleList([
            nn.Linear(config.d_model, size, bias=False)
            for size in config.codebook_sizes
        ])
        
        # 初始化输出头权重
        self._init_output_heads()
    
    def _init_output_heads(self):
        """初始化输出头权重"""
        for head in self.output_heads:
            nn.init.xavier_uniform_(head.weight)
    
    def forward(
        self,
        encoder_output: torch.Tensor,
        target_semantic_ids: Optional[List[torch.Tensor]] = None,
        target_positions: Optional[torch.Tensor] = None,
        target_token_types: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练模式前向传播
        
        计算目标序列的预测 logits。
        
        Args:
            encoder_output: (batch_size, src_len, d_model) 编码器输出
            target_semantic_ids: [L1_ids, L2_ids, L3_ids] 目标语义 ID 序列
                每个形状为 (batch_size, tgt_len)
            target_positions: (batch_size, tgt_len) 目标位置索引
            target_token_types: (batch_size, tgt_len) 目标 Token 类型
            encoder_mask: (batch_size, src_len) 编码器 padding 掩码
            target_mask: (batch_size, tgt_len) 目标序列 padding 掩码
        
        Returns:
            Tuple[L1_logits, L2_logits, L3_logits, aux_loss]:
                - L1_logits: (batch_size, tgt_len, codebook_size[0])
                - L2_logits: (batch_size, tgt_len, codebook_size[1])
                - L3_logits: (batch_size, tgt_len, codebook_size[2])
                - aux_loss: MoE 负载均衡损失 (scalar)
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device
        
        # 处理目标序列
        if target_semantic_ids is None:
            raise ValueError("训练模式需要提供 target_semantic_ids")
        
        l1_ids, l2_ids, l3_ids = target_semantic_ids
        tgt_len = l1_ids.shape[1]
        
        # 如果没有提供位置和类型，使用默认值
        if target_positions is None:
            target_positions = torch.arange(tgt_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        if target_token_types is None:
            # 默认都是 ITEM 类型 (type=1)
            target_token_types = torch.ones(batch_size, tgt_len, dtype=torch.long, device=device)
        
        # 计算输入嵌入
        hidden = self.input_embedding(
            semantic_ids=(l1_ids, l2_ids, l3_ids),
            positions=target_positions,
            token_types=target_token_types,
        )
        # hidden: (batch_size, tgt_len, d_model)
        
        # 通过解码器层
        total_aux_loss = torch.tensor(0.0, device=device)
        
        for layer in self.layers:
            hidden, aux_loss, _, _ = layer(
                x=hidden,
                encoder_output=encoder_output,
                token_types=target_token_types,
                self_attn_mask=target_mask,
                cross_attn_mask=encoder_mask,
            )
            total_aux_loss = total_aux_loss + aux_loss
        
        # 最终层归一化
        hidden = self.final_norm(hidden)
        
        # 计算输出 logits
        l1_logits = self.output_heads[0](hidden)  # (batch, tgt_len, codebook_sizes[0])
        l2_logits = self.output_heads[1](hidden)  # (batch, tgt_len, codebook_sizes[1])
        l3_logits = self.output_heads[2](hidden)  # (batch, tgt_len, codebook_sizes[2])
        
        # 平均辅助损失
        avg_aux_loss = total_aux_loss / len(self.layers)
        
        return l1_logits, l2_logits, l3_logits, avg_aux_loss
    
    def generate(
        self,
        encoder_output: torch.Tensor,
        num_recommendations: int = 20,
        beam_size: int = 4,
        temperature: float = 1.0,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        推理模式：自回归生成推荐列表
        
        使用层次化生成策略：L1 → L2 → L3
        
        Args:
            encoder_output: (batch_size, src_len, d_model) 编码器输出
            num_recommendations: 生成的推荐数量
            beam_size: Beam Search 宽度
            temperature: 采样温度（>1 更多样，<1 更确定）
            encoder_mask: (batch_size, src_len) 编码器 padding 掩码
        
        Returns:
            recommendations: List[List[Tuple[L1, L2, L3]]]
                外层 List 长度为 batch_size
                内层 List 长度为 num_recommendations
                每个 Tuple 是 (L1_id, L2_id, L3_id)
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device
        
        all_recommendations = []
        
        for b in range(batch_size):
            enc_out = encoder_output[b:b+1]  # (1, src_len, d_model)
            enc_mask = encoder_mask[b:b+1] if encoder_mask is not None else None
            
            recommendations = self._generate_single_batch(
                encoder_output=enc_out,
                num_recommendations=num_recommendations,
                beam_size=beam_size,
                temperature=temperature,
                encoder_mask=enc_mask,
            )
            all_recommendations.append(recommendations)
        
        return all_recommendations
    
    def _generate_single_batch(
        self,
        encoder_output: torch.Tensor,
        num_recommendations: int,
        beam_size: int,
        temperature: float,
        encoder_mask: Optional[torch.Tensor],
    ) -> List[Tuple[int, int, int]]:
        """
        为单个样本生成推荐
        
        采用层次化生成策略：
        1. 先生成 L1（粗粒度类目）
        2. 条件于 L1 生成 L2（细粒度属性）
        3. 条件于 L1、L2 生成 L3（实例区分）
        
        Args:
            encoder_output: (1, src_len, d_model)
            num_recommendations: 推荐数量
            beam_size: Beam 宽度
            temperature: 采样温度
            encoder_mask: (1, src_len)
        
        Returns:
            recommendations: List[Tuple[L1, L2, L3]]
        """
        device = encoder_output.device
        recommendations = []
        generated_set = set()  # 用于去重
        
        # 生成多于需要的候选，以便去重后仍有足够数量
        max_attempts = num_recommendations * 3
        
        for _ in range(max_attempts):
            if len(recommendations) >= num_recommendations:
                break
            
            # 层次化生成
            l1_id = self._generate_level(
                encoder_output=encoder_output,
                prefix_l1=None,
                prefix_l2=None,
                level=0,
                beam_size=beam_size,
                temperature=temperature,
                encoder_mask=encoder_mask,
            )
            
            l2_id = self._generate_level(
                encoder_output=encoder_output,
                prefix_l1=l1_id,
                prefix_l2=None,
                level=1,
                beam_size=beam_size,
                temperature=temperature,
                encoder_mask=encoder_mask,
            )
            
            l3_id = self._generate_level(
                encoder_output=encoder_output,
                prefix_l1=l1_id,
                prefix_l2=l2_id,
                level=2,
                beam_size=beam_size,
                temperature=temperature,
                encoder_mask=encoder_mask,
            )
            
            # 去重
            rec_tuple = (l1_id, l2_id, l3_id)
            if rec_tuple not in generated_set:
                generated_set.add(rec_tuple)
                recommendations.append(rec_tuple)
        
        return recommendations[:num_recommendations]
    
    def _generate_level(
        self,
        encoder_output: torch.Tensor,
        prefix_l1: Optional[int],
        prefix_l2: Optional[int],
        level: int,
        beam_size: int,
        temperature: float,
        encoder_mask: Optional[torch.Tensor],
    ) -> int:
        """
        生成指定层级的语义 ID
        
        Args:
            encoder_output: (1, src_len, d_model)
            prefix_l1: 已生成的 L1 ID（如果有）
            prefix_l2: 已生成的 L2 ID（如果有）
            level: 当前要生成的层级 (0=L1, 1=L2, 2=L3)
            beam_size: Beam 宽度
            temperature: 采样温度
            encoder_mask: (1, src_len)
        
        Returns:
            生成的 ID
        """
        device = encoder_output.device
        
        # 构建输入序列
        if level == 0:
            # 生成 L1：使用 BOS token
            l1_ids = torch.tensor([[self.config.bos_token_id]], device=device)
            l2_ids = torch.tensor([[self.config.bos_token_id]], device=device)
            l3_ids = torch.tensor([[self.config.bos_token_id]], device=device)
        elif level == 1:
            # 生成 L2：已有 L1
            l1_ids = torch.tensor([[prefix_l1]], device=device)
            l2_ids = torch.tensor([[self.config.bos_token_id]], device=device)
            l3_ids = torch.tensor([[self.config.bos_token_id]], device=device)
        else:
            # 生成 L3：已有 L1, L2
            l1_ids = torch.tensor([[prefix_l1]], device=device)
            l2_ids = torch.tensor([[prefix_l2]], device=device)
            l3_ids = torch.tensor([[self.config.bos_token_id]], device=device)
        
        # 位置和类型
        positions = torch.tensor([[0]], device=device)
        token_types = torch.ones(1, 1, dtype=torch.long, device=device)  # ITEM 类型
        
        # 计算嵌入
        hidden = self.input_embedding(
            semantic_ids=(l1_ids, l2_ids, l3_ids),
            positions=positions,
            token_types=token_types,
        )
        
        # 通过解码器层
        for layer in self.layers:
            hidden, _, _, _ = layer(
                x=hidden,
                encoder_output=encoder_output,
                token_types=token_types,
                cross_attn_mask=encoder_mask,
            )
        
        # 最终归一化
        hidden = self.final_norm(hidden)
        
        # 获取对应层级的 logits
        logits = self.output_heads[level](hidden)  # (1, 1, codebook_size)
        logits = logits[0, 0]  # (codebook_size,)
        
        # 温度缩放
        if temperature != 1.0:
            logits = logits / temperature
        
        # 采样
        if temperature > 0:
            probs = F.softmax(logits, dim=-1)
            # Top-K 采样
            if beam_size > 1:
                top_k_probs, top_k_indices = torch.topk(probs, beam_size)
                # 从 top-k 中采样
                sampled_idx = torch.multinomial(top_k_probs, 1)
                generated_id = top_k_indices[sampled_idx].item()
            else:
                generated_id = torch.multinomial(probs, 1).item()
        else:
            # Greedy 解码
            generated_id = logits.argmax().item()
        
        return generated_id
    
    def compute_loss(
        self,
        l1_logits: torch.Tensor,
        l2_logits: torch.Tensor,
        l3_logits: torch.Tensor,
        target_l1: torch.Tensor,
        target_l2: torch.Tensor,
        target_l3: torch.Tensor,
        aux_loss: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算训练损失
        
        包括三层语义 ID 的交叉熵损失和 MoE 辅助损失。
        
        Args:
            l1_logits: (batch, seq_len, codebook_size[0])
            l2_logits: (batch, seq_len, codebook_size[1])
            l3_logits: (batch, seq_len, codebook_size[2])
            target_l1: (batch, seq_len) L1 目标
            target_l2: (batch, seq_len) L2 目标
            target_l3: (batch, seq_len) L3 目标
            aux_loss: MoE 辅助损失
            mask: (batch, seq_len) 有效位置掩码
        
        Returns:
            损失字典
        """
        batch_size, seq_len = target_l1.shape
        
        # 展平 logits 和目标
        l1_logits_flat = l1_logits.view(-1, l1_logits.size(-1))
        l2_logits_flat = l2_logits.view(-1, l2_logits.size(-1))
        l3_logits_flat = l3_logits.view(-1, l3_logits.size(-1))
        
        target_l1_flat = target_l1.view(-1)
        target_l2_flat = target_l2.view(-1)
        target_l3_flat = target_l3.view(-1)
        
        # 计算交叉熵损失
        l1_loss = F.cross_entropy(
            l1_logits_flat, target_l1_flat,
            ignore_index=self.config.pad_token_id,
            reduction='mean'
        )
        l2_loss = F.cross_entropy(
            l2_logits_flat, target_l2_flat,
            ignore_index=self.config.pad_token_id,
            reduction='mean'
        )
        l3_loss = F.cross_entropy(
            l3_logits_flat, target_l3_flat,
            ignore_index=self.config.pad_token_id,
            reduction='mean'
        )
        
        # 组合损失（三层权重相等）
        ntp_loss = (l1_loss + l2_loss + l3_loss) / 3.0
        
        # 总损失
        total_loss = ntp_loss + aux_loss
        
        return {
            'total_loss': total_loss,
            'ntp_loss': ntp_loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'l3_loss': l3_loss,
            'aux_loss': aux_loss,
        }
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UGTDecoderForInference(nn.Module):
    """
    用于推理的 UGT 解码器封装
    
    添加了 KV 缓存支持，提高增量生成效率。
    
    Args:
        decoder: UGTDecoder 实例
    """
    
    def __init__(self, decoder: UGTDecoder):
        super().__init__()
        self.decoder = decoder
        self.config = decoder.config
        
        # KV 缓存
        self._cache: Optional[List[Tuple]] = None
    
    def reset_cache(self):
        """重置 KV 缓存"""
        self._cache = None
    
    def forward_step(
        self,
        l1_id: torch.Tensor,
        l2_id: torch.Tensor,
        l3_id: torch.Tensor,
        position: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单步前向传播（用于增量生成）
        
        Args:
            l1_id: (batch, 1) L1 ID
            l2_id: (batch, 1) L2 ID
            l3_id: (batch, 1) L3 ID
            position: (batch, 1) 位置
            encoder_output: (batch, src_len, d_model)
            encoder_mask: (batch, src_len)
        
        Returns:
            (L1_logits, L2_logits, L3_logits) 每个形状为 (batch, 1, vocab_size)
        """
        batch_size = l1_id.shape[0]
        device = l1_id.device
        
        # 计算嵌入
        token_type = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        hidden = self.decoder.input_embedding.forward_with_cache(
            l1_id, l2_id, l3_id, position, token_type
        )
        
        # 初始化或更新缓存
        if self._cache is None:
            self._cache = [None] * len(self.decoder.layers)
        
        # 通过解码器层
        new_caches = []
        for i, layer in enumerate(self.decoder.layers):
            layer_cache = self._cache[i]
            if layer_cache is not None:
                self_cache, cross_cache = layer_cache
            else:
                self_cache, cross_cache = None, None
            
            hidden, _, new_self_cache, new_cross_cache = layer(
                x=hidden,
                encoder_output=encoder_output,
                token_types=token_type,
                self_attn_cache=self_cache,
                cross_attn_cache=cross_cache,
                cross_attn_mask=encoder_mask,
            )
            new_caches.append((new_self_cache, new_cross_cache))
        
        self._cache = new_caches
        
        # 最终归一化
        hidden = self.decoder.final_norm(hidden)
        
        # 输出 logits
        l1_logits = self.decoder.output_heads[0](hidden)
        l2_logits = self.decoder.output_heads[1](hidden)
        l3_logits = self.decoder.output_heads[2](hidden)
        
        return l1_logits, l2_logits, l3_logits

