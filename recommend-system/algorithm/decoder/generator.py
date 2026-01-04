"""
推荐生成器模块

实现多种生成策略：
1. Beam Search：高质量生成
2. Top-K 采样：多样性生成
3. 核采样（Nucleus Sampling）：平衡质量和多样性

对应架构文档: 《生成式推荐系统架构设计》第三章
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .config import DecoderConfig
from .decoder import UGTDecoder


@dataclass
class GenerationConfig:
    """
    生成配置
    
    Attributes:
        num_recommendations: 生成的推荐数量
        beam_size: Beam Search 宽度
        temperature: 采样温度
        top_k: Top-K 采样的 K 值
        top_p: 核采样的累积概率阈值
        do_sample: 是否使用采样（否则使用 greedy/beam）
        repetition_penalty: 重复惩罚系数
        diversity_penalty: 多样性惩罚系数
        length_penalty: 长度惩罚系数
    """
    num_recommendations: int = 20
    beam_size: int = 4
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.0
    diversity_penalty: float = 0.0
    length_penalty: float = 1.0


class BeamSearchGenerator:
    """
    Beam Search 推荐生成器
    
    使用层次化 Beam Search 生成推荐：
    1. 先为 L1 生成 top-k 候选
    2. 对每个 L1 候选，生成 L2 候选
    3. 对每个 (L1, L2) 组合，生成 L3 候选
    4. 根据联合概率选择最终推荐
    
    Args:
        decoder: UGT 解码器
        config: 解码器配置
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
        encoder_mask: Optional[torch.Tensor] = None,
        diversity_penalty: float = 0.0,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        使用 Beam Search 生成推荐列表
        
        Args:
            encoder_output: (batch_size, src_len, d_model) 编码器输出
            num_recommendations: 生成的推荐数量
            beam_size: Beam 宽度
            temperature: 采样温度
            encoder_mask: (batch_size, src_len) 编码器掩码
            diversity_penalty: 多样性惩罚（鼓励生成不同的推荐）
        
        Returns:
            recommendations: List[List[Tuple[L1, L2, L3]]]
                外层 List 长度为 batch_size
                内层 List 长度为 num_recommendations
        """
        self.decoder.eval()
        
        batch_size = encoder_output.shape[0]
        device = encoder_output.device
        
        all_recommendations = []
        
        for b in range(batch_size):
            enc_out = encoder_output[b:b+1]
            enc_mask = encoder_mask[b:b+1] if encoder_mask is not None else None
            
            recommendations = self._beam_search_single(
                encoder_output=enc_out,
                num_recommendations=num_recommendations,
                beam_size=beam_size,
                temperature=temperature,
                encoder_mask=enc_mask,
                diversity_penalty=diversity_penalty,
            )
            all_recommendations.append(recommendations)
        
        return all_recommendations
    
    def _beam_search_single(
        self,
        encoder_output: torch.Tensor,
        num_recommendations: int,
        beam_size: int,
        temperature: float,
        encoder_mask: Optional[torch.Tensor],
        diversity_penalty: float,
    ) -> List[Tuple[int, int, int]]:
        """
        为单个样本执行层次化 Beam Search
        
        Args:
            encoder_output: (1, src_len, d_model)
            num_recommendations: 推荐数量
            beam_size: Beam 宽度
            temperature: 采样温度
            encoder_mask: (1, src_len)
            diversity_penalty: 多样性惩罚
        
        Returns:
            recommendations: List[Tuple[L1, L2, L3]]
        """
        device = encoder_output.device
        
        # 获取 L1 候选
        l1_candidates = self._get_level_candidates(
            encoder_output=encoder_output,
            prefix_l1=None,
            prefix_l2=None,
            level=0,
            num_candidates=beam_size * 2,  # 多获取一些候选
            temperature=temperature,
            encoder_mask=encoder_mask,
        )
        
        # 对每个 L1，获取 L2 候选
        all_candidates: List[Tuple[int, int, int, float]] = []  # (L1, L2, L3, score)
        
        for l1_id, l1_score in l1_candidates[:beam_size]:
            l2_candidates = self._get_level_candidates(
                encoder_output=encoder_output,
                prefix_l1=l1_id,
                prefix_l2=None,
                level=1,
                num_candidates=beam_size,
                temperature=temperature,
                encoder_mask=encoder_mask,
            )
            
            for l2_id, l2_score in l2_candidates:
                l3_candidates = self._get_level_candidates(
                    encoder_output=encoder_output,
                    prefix_l1=l1_id,
                    prefix_l2=l2_id,
                    level=2,
                    num_candidates=beam_size,
                    temperature=temperature,
                    encoder_mask=encoder_mask,
                )
                
                for l3_id, l3_score in l3_candidates:
                    # 联合得分（log 概率之和）
                    joint_score = l1_score + l2_score + l3_score
                    all_candidates.append((l1_id, l2_id, l3_id, joint_score))
        
        # 按得分排序
        all_candidates.sort(key=lambda x: x[3], reverse=True)
        
        # 去重并选择 top-k
        seen = set()
        recommendations = []
        for l1_id, l2_id, l3_id, score in all_candidates:
            if (l1_id, l2_id, l3_id) not in seen:
                seen.add((l1_id, l2_id, l3_id))
                recommendations.append((l1_id, l2_id, l3_id))
                
                if len(recommendations) >= num_recommendations:
                    break
        
        return recommendations
    
    def _get_level_candidates(
        self,
        encoder_output: torch.Tensor,
        prefix_l1: Optional[int],
        prefix_l2: Optional[int],
        level: int,
        num_candidates: int,
        temperature: float,
        encoder_mask: Optional[torch.Tensor],
    ) -> List[Tuple[int, float]]:
        """
        获取指定层级的候选及其得分
        
        Args:
            encoder_output: (1, src_len, d_model)
            prefix_l1: 已生成的 L1
            prefix_l2: 已生成的 L2
            level: 当前层级 (0, 1, 2)
            num_candidates: 候选数量
            temperature: 采样温度
            encoder_mask: (1, src_len)
        
        Returns:
            candidates: List[(id, log_prob)]
        """
        device = encoder_output.device
        
        # 构建输入
        if level == 0:
            l1_ids = torch.tensor([[self.config.bos_token_id]], device=device)
            l2_ids = torch.tensor([[self.config.bos_token_id]], device=device)
            l3_ids = torch.tensor([[self.config.bos_token_id]], device=device)
        elif level == 1:
            l1_ids = torch.tensor([[prefix_l1]], device=device)
            l2_ids = torch.tensor([[self.config.bos_token_id]], device=device)
            l3_ids = torch.tensor([[self.config.bos_token_id]], device=device)
        else:
            l1_ids = torch.tensor([[prefix_l1]], device=device)
            l2_ids = torch.tensor([[prefix_l2]], device=device)
            l3_ids = torch.tensor([[self.config.bos_token_id]], device=device)
        
        positions = torch.tensor([[0]], device=device)
        token_types = torch.ones(1, 1, dtype=torch.long, device=device)
        
        # 计算嵌入
        hidden = self.decoder.input_embedding(
            semantic_ids=(l1_ids, l2_ids, l3_ids),
            positions=positions,
            token_types=token_types,
        )
        
        # 通过解码器层
        for layer in self.decoder.layers:
            hidden, _, _, _ = layer(
                x=hidden,
                encoder_output=encoder_output,
                token_types=token_types,
                cross_attn_mask=encoder_mask,
            )
        
        hidden = self.decoder.final_norm(hidden)
        
        # 获取 logits
        logits = self.decoder.output_heads[level](hidden)[0, 0]  # (vocab_size,)
        
        # 温度缩放
        if temperature != 1.0:
            logits = logits / temperature
        
        # 计算 log 概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取 top-k
        top_k_log_probs, top_k_indices = torch.topk(log_probs, num_candidates)
        
        candidates = [
            (idx.item(), log_prob.item())
            for idx, log_prob in zip(top_k_indices, top_k_log_probs)
        ]
        
        return candidates


class DiverseBeamSearchGenerator:
    """
    多样性 Beam Search 生成器
    
    使用 Diverse Beam Search 算法生成更多样化的推荐。
    将 beam 分成多个组，每组独立搜索，组间添加多样性惩罚。
    
    Args:
        decoder: UGT 解码器
        config: 解码器配置
    """
    
    def __init__(self, decoder: UGTDecoder, config: DecoderConfig):
        self.decoder = decoder
        self.config = config
    
    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        num_recommendations: int = 20,
        num_beam_groups: int = 4,
        beam_size_per_group: int = 4,
        diversity_penalty: float = 0.5,
        temperature: float = 1.0,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        使用多样性 Beam Search 生成推荐
        
        Args:
            encoder_output: (batch_size, src_len, d_model)
            num_recommendations: 推荐数量
            num_beam_groups: beam 组数
            beam_size_per_group: 每组的 beam 大小
            diversity_penalty: 多样性惩罚系数
            temperature: 采样温度
            encoder_mask: 编码器掩码
        
        Returns:
            recommendations: List[List[Tuple[L1, L2, L3]]]
        """
        self.decoder.eval()
        
        batch_size = encoder_output.shape[0]
        all_recommendations = []
        
        for b in range(batch_size):
            enc_out = encoder_output[b:b+1]
            enc_mask = encoder_mask[b:b+1] if encoder_mask is not None else None
            
            # 每个组独立生成
            group_recommendations: List[List[Tuple[int, int, int]]] = []
            selected_items: Dict[Tuple[int, int, int], float] = {}  # 用于多样性惩罚
            
            recs_per_group = max(1, num_recommendations // num_beam_groups)
            
            for group_idx in range(num_beam_groups):
                group_recs = self._generate_group(
                    encoder_output=enc_out,
                    num_recs=recs_per_group,
                    beam_size=beam_size_per_group,
                    temperature=temperature,
                    encoder_mask=enc_mask,
                    selected_items=selected_items,
                    diversity_penalty=diversity_penalty,
                )
                
                # 更新已选择的物品
                for rec in group_recs:
                    selected_items[rec] = selected_items.get(rec, 0) + 1
                
                group_recommendations.extend(group_recs)
            
            # 合并并截取
            all_recommendations.append(group_recommendations[:num_recommendations])
        
        return all_recommendations
    
    def _generate_group(
        self,
        encoder_output: torch.Tensor,
        num_recs: int,
        beam_size: int,
        temperature: float,
        encoder_mask: Optional[torch.Tensor],
        selected_items: Dict[Tuple[int, int, int], float],
        diversity_penalty: float,
    ) -> List[Tuple[int, int, int]]:
        """
        为单个组生成推荐
        
        对已经被其他组选择的物品施加惩罚，鼓励多样性。
        """
        device = encoder_output.device
        recommendations = []
        
        for _ in range(num_recs):
            # 生成候选
            l1_id = self._sample_level_with_penalty(
                encoder_output, None, None, 0, temperature, encoder_mask,
                selected_items, diversity_penalty
            )
            
            l2_id = self._sample_level_with_penalty(
                encoder_output, l1_id, None, 1, temperature, encoder_mask,
                selected_items, diversity_penalty
            )
            
            l3_id = self._sample_level_with_penalty(
                encoder_output, l1_id, l2_id, 2, temperature, encoder_mask,
                selected_items, diversity_penalty
            )
            
            rec = (l1_id, l2_id, l3_id)
            if rec not in [r for r in recommendations]:
                recommendations.append(rec)
        
        return recommendations
    
    def _sample_level_with_penalty(
        self,
        encoder_output: torch.Tensor,
        prefix_l1: Optional[int],
        prefix_l2: Optional[int],
        level: int,
        temperature: float,
        encoder_mask: Optional[torch.Tensor],
        selected_items: Dict[Tuple[int, int, int], float],
        diversity_penalty: float,
    ) -> int:
        """采样并应用多样性惩罚"""
        device = encoder_output.device
        
        # 构建输入（与 BeamSearchGenerator 类似）
        if level == 0:
            l1_ids = torch.tensor([[self.config.bos_token_id]], device=device)
            l2_ids = torch.tensor([[self.config.bos_token_id]], device=device)
            l3_ids = torch.tensor([[self.config.bos_token_id]], device=device)
        elif level == 1:
            l1_ids = torch.tensor([[prefix_l1]], device=device)
            l2_ids = torch.tensor([[self.config.bos_token_id]], device=device)
            l3_ids = torch.tensor([[self.config.bos_token_id]], device=device)
        else:
            l1_ids = torch.tensor([[prefix_l1]], device=device)
            l2_ids = torch.tensor([[prefix_l2]], device=device)
            l3_ids = torch.tensor([[self.config.bos_token_id]], device=device)
        
        positions = torch.tensor([[0]], device=device)
        token_types = torch.ones(1, 1, dtype=torch.long, device=device)
        
        hidden = self.decoder.input_embedding(
            semantic_ids=(l1_ids, l2_ids, l3_ids),
            positions=positions,
            token_types=token_types,
        )
        
        for layer in self.decoder.layers:
            hidden, _, _, _ = layer(
                x=hidden,
                encoder_output=encoder_output,
                token_types=token_types,
                cross_attn_mask=encoder_mask,
            )
        
        hidden = self.decoder.final_norm(hidden)
        logits = self.decoder.output_heads[level](hidden)[0, 0]
        
        # 温度缩放
        if temperature != 1.0:
            logits = logits / temperature
        
        # 应用多样性惩罚（对已选择的物品的相关层级 ID 惩罚）
        if diversity_penalty > 0 and level == 0:
            for (l1, l2, l3), count in selected_items.items():
                if l1 < logits.size(0):
                    logits[l1] -= diversity_penalty * count
        
        # 采样
        probs = F.softmax(logits, dim=-1)
        sampled_id = torch.multinomial(probs, 1).item()
        
        return sampled_id


class NucleusSamplingGenerator:
    """
    核采样（Nucleus Sampling）生成器
    
    也称为 Top-p 采样，只从累积概率超过 p 的最小词汇集合中采样。
    这样可以在保持多样性的同时避免采样到低概率的异常结果。
    
    Args:
        decoder: UGT 解码器
        config: 解码器配置
    """
    
    def __init__(self, decoder: UGTDecoder, config: DecoderConfig):
        self.decoder = decoder
        self.config = config
    
    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        num_recommendations: int = 20,
        top_p: float = 0.9,
        temperature: float = 1.0,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        使用核采样生成推荐
        
        Args:
            encoder_output: (batch_size, src_len, d_model)
            num_recommendations: 推荐数量
            top_p: 核采样的累积概率阈值
            temperature: 采样温度
            encoder_mask: 编码器掩码
        
        Returns:
            recommendations: List[List[Tuple[L1, L2, L3]]]
        """
        self.decoder.eval()
        
        batch_size = encoder_output.shape[0]
        all_recommendations = []
        
        for b in range(batch_size):
            enc_out = encoder_output[b:b+1]
            enc_mask = encoder_mask[b:b+1] if encoder_mask is not None else None
            
            recommendations = []
            seen = set()
            max_attempts = num_recommendations * 3
            
            for _ in range(max_attempts):
                if len(recommendations) >= num_recommendations:
                    break
                
                # 层次化采样
                l1_id = self._nucleus_sample_level(
                    enc_out, None, None, 0, top_p, temperature, enc_mask
                )
                l2_id = self._nucleus_sample_level(
                    enc_out, l1_id, None, 1, top_p, temperature, enc_mask
                )
                l3_id = self._nucleus_sample_level(
                    enc_out, l1_id, l2_id, 2, top_p, temperature, enc_mask
                )
                
                rec = (l1_id, l2_id, l3_id)
                if rec not in seen:
                    seen.add(rec)
                    recommendations.append(rec)
            
            all_recommendations.append(recommendations)
        
        return all_recommendations
    
    def _nucleus_sample_level(
        self,
        encoder_output: torch.Tensor,
        prefix_l1: Optional[int],
        prefix_l2: Optional[int],
        level: int,
        top_p: float,
        temperature: float,
        encoder_mask: Optional[torch.Tensor],
    ) -> int:
        """
        核采样指定层级
        """
        device = encoder_output.device
        
        # 构建输入
        if level == 0:
            l1_ids = torch.tensor([[self.config.bos_token_id]], device=device)
            l2_ids = torch.tensor([[self.config.bos_token_id]], device=device)
            l3_ids = torch.tensor([[self.config.bos_token_id]], device=device)
        elif level == 1:
            l1_ids = torch.tensor([[prefix_l1]], device=device)
            l2_ids = torch.tensor([[self.config.bos_token_id]], device=device)
            l3_ids = torch.tensor([[self.config.bos_token_id]], device=device)
        else:
            l1_ids = torch.tensor([[prefix_l1]], device=device)
            l2_ids = torch.tensor([[prefix_l2]], device=device)
            l3_ids = torch.tensor([[self.config.bos_token_id]], device=device)
        
        positions = torch.tensor([[0]], device=device)
        token_types = torch.ones(1, 1, dtype=torch.long, device=device)
        
        hidden = self.decoder.input_embedding(
            semantic_ids=(l1_ids, l2_ids, l3_ids),
            positions=positions,
            token_types=token_types,
        )
        
        for layer in self.decoder.layers:
            hidden, _, _, _ = layer(
                x=hidden,
                encoder_output=encoder_output,
                token_types=token_types,
                cross_attn_mask=encoder_mask,
            )
        
        hidden = self.decoder.final_norm(hidden)
        logits = self.decoder.output_heads[level](hidden)[0, 0]
        
        # 温度缩放
        if temperature != 1.0:
            logits = logits / temperature
        
        # 核采样
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过 top_p 的位置
        nucleus_mask = cumsum_probs <= top_p
        # 确保至少包含一个 token
        nucleus_mask[0] = True
        
        # 只保留 nucleus 中的 token
        nucleus_probs = sorted_probs * nucleus_mask.float()
        nucleus_probs = nucleus_probs / nucleus_probs.sum()
        
        # 采样
        sampled_idx = torch.multinomial(nucleus_probs, 1)
        sampled_id = sorted_indices[sampled_idx].item()
        
        return sampled_id

