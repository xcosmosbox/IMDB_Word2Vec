"""
评估指标模块

实现推荐系统的评估指标：
- Recall@K: 召回率
- NDCG@K: 归一化折损累计增益
- MRR: 平均倒数排名
- Hit Rate: 命中率
- Coverage: 覆盖率
- Diversity: 多样性

对应架构文档: 第九章 效果评估与监控
"""

import math
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict

import numpy as np


def recall_at_k(
    predictions: List[List[Tuple[int, int, int]]],
    ground_truth: List[Tuple[int, int, int]],
    k: int = 10,
) -> float:
    """
    计算 Recall@K
    
    预测列表的前 K 个中包含正确答案的比例
    
    Args:
        predictions: 预测列表，每个样本的预测序列
        ground_truth: 真实标签列表
        k: 取前 K 个预测
    
    Returns:
        Recall@K 值
    """
    if not predictions or not ground_truth:
        return 0.0
    
    hits = 0
    for preds, gt in zip(predictions, ground_truth):
        top_k = preds[:k]
        if gt in top_k:
            hits += 1
    
    return hits / len(predictions)


def ndcg_at_k(
    predictions: List[List[Tuple[int, int, int]]],
    ground_truth: List[Tuple[int, int, int]],
    k: int = 10,
) -> float:
    """
    计算 NDCG@K (Normalized Discounted Cumulative Gain)
    
    考虑正确答案在预测列表中的位置，位置越靠前得分越高
    
    Args:
        predictions: 预测列表
        ground_truth: 真实标签列表
        k: 取前 K 个预测
    
    Returns:
        NDCG@K 值
    """
    if not predictions or not ground_truth:
        return 0.0
    
    dcg_sum = 0.0
    
    for preds, gt in zip(predictions, ground_truth):
        top_k = preds[:k]
        for i, pred in enumerate(top_k):
            if pred == gt:
                # DCG 增益 = 1 / log2(rank + 2)
                dcg_sum += 1.0 / math.log2(i + 2)
                break
    
    # IDCG = 1（因为只有一个正确答案，理想情况是在第一位）
    idcg = 1.0
    
    return dcg_sum / (len(predictions) * idcg)


def mrr(
    predictions: List[List[Tuple[int, int, int]]],
    ground_truth: List[Tuple[int, int, int]],
) -> float:
    """
    计算 MRR (Mean Reciprocal Rank)
    
    正确答案的倒数排名的平均值
    
    Args:
        predictions: 预测列表
        ground_truth: 真实标签列表
    
    Returns:
        MRR 值
    """
    if not predictions or not ground_truth:
        return 0.0
    
    rr_sum = 0.0
    
    for preds, gt in zip(predictions, ground_truth):
        for i, pred in enumerate(preds):
            if pred == gt:
                rr_sum += 1.0 / (i + 1)
                break
    
    return rr_sum / len(predictions)


def hit_rate(
    predictions: List[List[Tuple[int, int, int]]],
    ground_truth: List[Tuple[int, int, int]],
    k: int = 10,
) -> float:
    """
    计算 Hit Rate@K
    
    与 Recall@K 相同，当每个样本只有一个正确答案时
    
    Args:
        predictions: 预测列表
        ground_truth: 真实标签列表
        k: 取前 K 个预测
    
    Returns:
        Hit Rate@K 值
    """
    return recall_at_k(predictions, ground_truth, k)


def precision_at_k(
    predictions: List[List[Tuple[int, int, int]]],
    ground_truth_sets: List[Set[Tuple[int, int, int]]],
    k: int = 10,
) -> float:
    """
    计算 Precision@K
    
    预测列表的前 K 个中正确答案的比例
    
    Args:
        predictions: 预测列表
        ground_truth_sets: 真实标签集合列表（每个样本可能有多个正确答案）
        k: 取前 K 个预测
    
    Returns:
        Precision@K 值
    """
    if not predictions or not ground_truth_sets:
        return 0.0
    
    precision_sum = 0.0
    
    for preds, gt_set in zip(predictions, ground_truth_sets):
        top_k = preds[:k]
        hits = sum(1 for pred in top_k if pred in gt_set)
        precision_sum += hits / k
    
    return precision_sum / len(predictions)


def map_at_k(
    predictions: List[List[Tuple[int, int, int]]],
    ground_truth_sets: List[Set[Tuple[int, int, int]]],
    k: int = 10,
) -> float:
    """
    计算 MAP@K (Mean Average Precision)
    
    Args:
        predictions: 预测列表
        ground_truth_sets: 真实标签集合列表
        k: 取前 K 个预测
    
    Returns:
        MAP@K 值
    """
    if not predictions or not ground_truth_sets:
        return 0.0
    
    ap_sum = 0.0
    
    for preds, gt_set in zip(predictions, ground_truth_sets):
        top_k = preds[:k]
        hits = 0
        precision_sum = 0.0
        
        for i, pred in enumerate(top_k):
            if pred in gt_set:
                hits += 1
                precision_sum += hits / (i + 1)
        
        if len(gt_set) > 0:
            ap_sum += precision_sum / min(len(gt_set), k)
    
    return ap_sum / len(predictions)


def coverage(
    predictions: List[List[Tuple[int, int, int]]],
    all_items: Set[Tuple[int, int, int]],
    k: int = 10,
) -> float:
    """
    计算 Coverage
    
    被推荐的物品占总物品的比例
    
    Args:
        predictions: 预测列表
        all_items: 所有物品集合
        k: 取前 K 个预测
    
    Returns:
        Coverage 值
    """
    if not predictions or not all_items:
        return 0.0
    
    recommended_items = set()
    
    for preds in predictions:
        top_k = preds[:k]
        recommended_items.update(top_k)
    
    return len(recommended_items) / len(all_items)


def intra_list_diversity(
    predictions: List[List[Tuple[int, int, int]]],
    item_categories: Dict[Tuple[int, int, int], str],
    k: int = 10,
) -> float:
    """
    计算列表内多样性 (Intra-List Diversity)
    
    推荐列表内不同类别物品的比例
    
    Args:
        predictions: 预测列表
        item_categories: 物品到类别的映射
        k: 取前 K 个预测
    
    Returns:
        ILD 值
    """
    if not predictions:
        return 0.0
    
    diversity_sum = 0.0
    
    for preds in predictions:
        top_k = preds[:k]
        categories = set()
        
        for pred in top_k:
            if pred in item_categories:
                categories.add(item_categories[pred])
        
        if len(top_k) > 0:
            diversity_sum += len(categories) / len(top_k)
    
    return diversity_sum / len(predictions)


def gini_index(item_counts: Dict[Tuple[int, int, int], int]) -> float:
    """
    计算 Gini 指数
    
    衡量推荐的公平性，越低表示推荐越均匀
    
    Args:
        item_counts: 物品被推荐的次数
    
    Returns:
        Gini 指数
    """
    if not item_counts:
        return 0.0
    
    counts = sorted(item_counts.values())
    n = len(counts)
    
    if n == 0:
        return 0.0
    
    # 计算 Gini 系数
    total = sum(counts)
    if total == 0:
        return 0.0
    
    cumsum = 0
    gini_sum = 0
    
    for i, count in enumerate(counts):
        cumsum += count
        gini_sum += (2 * (i + 1) - n - 1) * count
    
    return gini_sum / (n * total)


def novelty(
    predictions: List[List[Tuple[int, int, int]]],
    item_popularity: Dict[Tuple[int, int, int], float],
    k: int = 10,
) -> float:
    """
    计算 Novelty（新颖度）
    
    推荐的物品越不热门，新颖度越高
    
    Args:
        predictions: 预测列表
        item_popularity: 物品热度（如点击率）
        k: 取前 K 个预测
    
    Returns:
        Novelty 值
    """
    if not predictions:
        return 0.0
    
    novelty_sum = 0.0
    count = 0
    
    for preds in predictions:
        top_k = preds[:k]
        
        for pred in top_k:
            pop = item_popularity.get(pred, 0.0)
            if pop > 0:
                # 使用 self-information: -log(popularity)
                novelty_sum += -math.log2(pop + 1e-10)
                count += 1
    
    if count == 0:
        return 0.0
    
    return novelty_sum / count


class MetricsCalculator:
    """
    指标计算器
    
    封装所有评估指标的计算逻辑
    """
    
    def __init__(self):
        """初始化指标计算器"""
        self.reset()
    
    def reset(self):
        """重置所有累积状态"""
        self.predictions = []
        self.ground_truth = []
        self.item_counts = defaultdict(int)
    
    def add_batch(
        self,
        predictions: List[List[Tuple[int, int, int]]],
        ground_truth: List[Tuple[int, int, int]],
    ):
        """
        添加一个批次的预测结果
        
        Args:
            predictions: 预测列表
            ground_truth: 真实标签列表
        """
        self.predictions.extend(predictions)
        self.ground_truth.extend(ground_truth)
        
        # 统计物品被推荐的次数
        for preds in predictions:
            for pred in preds[:10]:  # 只统计前 10 个
                self.item_counts[pred] += 1
    
    def compute(self, k_values: List[int] = [5, 10, 20, 50]) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            k_values: K 值列表
        
        Returns:
            指标字典
        """
        metrics = {}
        
        for k in k_values:
            metrics[f"recall@{k}"] = self.recall_at_k(
                self.predictions, self.ground_truth, k
            )
            metrics[f"ndcg@{k}"] = self.ndcg_at_k(
                self.predictions, self.ground_truth, k
            )
            metrics[f"hit_rate@{k}"] = self.hit_rate(
                self.predictions, self.ground_truth, k
            )
        
        metrics["mrr"] = self.mrr(self.predictions, self.ground_truth)
        metrics["gini"] = gini_index(self.item_counts)
        
        return metrics
    
    def recall_at_k(
        self,
        predictions: List[List[Tuple[int, int, int]]],
        ground_truth: List[Tuple[int, int, int]],
        k: int = 10,
    ) -> float:
        """计算 Recall@K"""
        return recall_at_k(predictions, ground_truth, k)
    
    def ndcg_at_k(
        self,
        predictions: List[List[Tuple[int, int, int]]],
        ground_truth: List[Tuple[int, int, int]],
        k: int = 10,
    ) -> float:
        """计算 NDCG@K"""
        return ndcg_at_k(predictions, ground_truth, k)
    
    def mrr(
        self,
        predictions: List[List[Tuple[int, int, int]]],
        ground_truth: List[Tuple[int, int, int]],
    ) -> float:
        """计算 MRR"""
        return mrr(predictions, ground_truth)
    
    def hit_rate(
        self,
        predictions: List[List[Tuple[int, int, int]]],
        ground_truth: List[Tuple[int, int, int]],
        k: int = 10,
    ) -> float:
        """计算 Hit Rate@K"""
        return hit_rate(predictions, ground_truth, k)


class OnlineMetrics:
    """
    在线指标收集器
    
    用于收集和计算在线 A/B 测试的指标
    """
    
    def __init__(self):
        """初始化在线指标收集器"""
        self.impressions = 0
        self.clicks = 0
        self.conversions = 0
        self.dwell_time_sum = 0.0
        self.revenue = 0.0
    
    def record_impression(self):
        """记录一次曝光"""
        self.impressions += 1
    
    def record_click(self):
        """记录一次点击"""
        self.clicks += 1
    
    def record_conversion(self, revenue: float = 0.0):
        """记录一次转化"""
        self.conversions += 1
        self.revenue += revenue
    
    def record_dwell_time(self, seconds: float):
        """记录停留时间"""
        self.dwell_time_sum += seconds
    
    def compute(self) -> Dict[str, float]:
        """计算在线指标"""
        metrics = {
            "impressions": self.impressions,
            "clicks": self.clicks,
            "conversions": self.conversions,
            "revenue": self.revenue,
        }
        
        if self.impressions > 0:
            metrics["ctr"] = self.clicks / self.impressions
            metrics["avg_dwell_time"] = self.dwell_time_sum / self.impressions
        
        if self.clicks > 0:
            metrics["cvr"] = self.conversions / self.clicks
        
        return metrics
    
    def reset(self):
        """重置所有指标"""
        self.impressions = 0
        self.clicks = 0
        self.conversions = 0
        self.dwell_time_sum = 0.0
        self.revenue = 0.0

