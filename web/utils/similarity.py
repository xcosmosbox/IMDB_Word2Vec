"""
相似度计算模块 (优化版)
======================

提供余弦相似度计算、Top-K 相似搜索、相似度矩阵生成等功能。

优化:
- 使用 KNN 索引加速搜索 (O(N) → O(log N))
- 预计算归一化嵌入
- 智能缓存

使用方法:
    # 快速搜索（推荐）
    results = find_similar_fast("MOV_tt0111161", k=10)
    
    # 传统方法（兼容）
    results = find_top_k_similar(query_vec, all_embeddings, tokens, k=10)
"""
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import streamlit as st
from pathlib import Path

# 导入名称映射
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.name_mapping import get_display_name, get_entity_type


# =============================================================================
# 基础相似度计算
# =============================================================================

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度
    
    余弦相似度 = (A · B) / (||A|| × ||B||)
    
    值域: [-1, 1]
    - 1 表示完全相同方向
    - 0 表示正交
    - -1 表示完全相反方向
    
    Args:
        vec_a: 第一个向量
        vec_b: 第二个向量
        
    Returns:
        余弦相似度值
    """
    # 计算点积
    dot_product = np.dot(vec_a, vec_b)
    
    # 计算范数
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    # 避免除零
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def batch_cosine_similarity(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    计算一个查询向量与多个向量的余弦相似度
    
    使用向量化操作，比逐个计算快得多。
    
    Args:
        query_vec: 查询向量，shape (embedding_dim,)
        embeddings: 嵌入矩阵，shape (n_samples, embedding_dim)
        
    Returns:
        相似度数组，shape (n_samples,)
    """
    # 归一化查询向量
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(len(embeddings))
    query_normalized = query_vec / query_norm
    
    # 归一化所有嵌入向量
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # 避免除零
    norms = np.where(norms == 0, 1, norms)
    embeddings_normalized = embeddings / norms
    
    # 计算余弦相似度 (矩阵乘法)
    similarities = np.dot(embeddings_normalized, query_normalized)
    
    return similarities


# =============================================================================
# Top-K 相似搜索
# =============================================================================

def find_top_k_similar(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    tokens: List[str],
    k: int = 10,
    exclude_self: bool = True,
    query_token: Optional[str] = None,
    entity_type_filter: Optional[str] = None,
    include_names: bool = True,
) -> List[Dict]:
    """
    找到与查询向量最相似的 K 个向量
    
    Args:
        query_vec: 查询向量
        embeddings: 所有嵌入向量
        tokens: 对应的 Token 列表
        k: 返回数量
        exclude_self: 是否排除自身（当查询来自 embeddings 时）
        query_token: 查询 Token（用于排除自身）
        entity_type_filter: 实体类型过滤器，如 "MOV", "ACT"
        include_names: 是否包含显示名称
        
    Returns:
        相似结果列表，每个元素包含:
        - token: Token 字符串
        - name: 显示名称 (如果 include_names=True)
        - similarity: 相似度值
        - rank: 排名
    """
    # 计算所有相似度
    similarities = batch_cosine_similarity(query_vec, embeddings)
    
    # 创建结果列表
    results = []
    for i, (token, sim) in enumerate(zip(tokens, similarities)):
        # 排除自身
        if exclude_self and query_token and token == query_token:
            continue
        
        # 实体类型过滤
        if entity_type_filter:
            token_type = token.split("_")[0] if "_" in token else "OTHER"
            if token_type != entity_type_filter:
                continue
        
        result = {
            "token": token,
            "similarity": float(sim),
            "index": i,
        }
        
        if include_names:
            result["name"] = get_display_name(token)
            result["type"] = get_entity_type(token)
        
        results.append(result)
    
    # 按相似度降序排序
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # 取 Top-K 并添加排名
    top_k = results[:k]
    for rank, item in enumerate(top_k, start=1):
        item["rank"] = rank
    
    return top_k


# =============================================================================
# 快速相似度搜索 (使用 KNN 索引)
# =============================================================================

def find_similar_fast(
    query_token: str,
    k: int = 10,
    entity_type_filter: Optional[str] = None,
    include_names: bool = True,
) -> List[Dict[str, Any]]:
    """
    快速相似度搜索（使用预计算的 KNN 索引）
    
    相比 find_top_k_similar，速度提升 10-100 倍。
    
    Args:
        query_token: 查询 Token
        k: 返回数量
        entity_type_filter: 实体类型过滤
        include_names: 是否包含显示名称
        
    Returns:
        相似结果列表，包含:
        - token: Token
        - name: 显示名称
        - similarity: 相似度值
        - type: 实体类型
        - rank: 排名
    """
    # 延迟导入避免循环引用
    from .precompute import find_similar_fast as _find_similar_fast
    
    results = _find_similar_fast(query_token, k=k, entity_type_filter=entity_type_filter)
    
    # 添加排名
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    
    return results


def find_similar_by_vector_fast(
    query_vec: np.ndarray,
    k: int = 10,
    exclude_indices: Optional[List[int]] = None,
    entity_type_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    使用向量快速搜索相似项（使用 KNN 索引）
    
    Args:
        query_vec: 查询向量
        k: 返回数量
        exclude_indices: 要排除的索引
        entity_type_filter: 类型过滤
        
    Returns:
        相似结果列表
    """
    from .precompute import knn_search, get_tokens_list
    
    tokens_list = get_tokens_list()
    similarities, indices = knn_search(
        query_vec,
        k=k + 10,  # 多取一些用于过滤
        exclude_indices=exclude_indices,
    )
    
    results = []
    for sim, idx in zip(similarities, indices):
        if idx >= len(tokens_list):
            continue
        
        token = tokens_list[int(idx)]
        if not token:
            continue
        
        # 类型过滤
        entity_type = get_entity_type(token)
        if entity_type_filter and entity_type != entity_type_filter:
            continue
        
        results.append({
            "token": token,
            "name": get_display_name(token),
            "type": entity_type,
            "similarity": float(sim),
            "index": int(idx),
        })
        
        if len(results) >= k:
            break
    
    # 添加排名
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    
    return results


# =============================================================================
# 相似度矩阵
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="计算相似度矩阵...")
def compute_similarity_matrix(
    embeddings: np.ndarray,
    indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    计算嵌入向量之间的相似度矩阵
    
    注意: 当向量数量很大时，矩阵会非常大，建议使用 indices 参数进行子集计算。
    
    Args:
        embeddings: 嵌入矩阵
        indices: 要计算的索引子集，None 表示全部
        
    Returns:
        相似度矩阵，shape (n, n)
    """
    if indices is not None:
        embeddings = embeddings[indices]
    
    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    
    # 计算相似度矩阵
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix


# =============================================================================
# 向量运算
# =============================================================================

def vector_arithmetic(
    embeddings: np.ndarray,
    tokens: List[str],
    token_to_id: Dict[str, int],
    positive: List[str],
    negative: List[str],
) -> Tuple[np.ndarray, List[Dict]]:
    """
    向量算术运算: sum(positive) - sum(negative)
    
    经典例子: King - Man + Woman ≈ Queen
    
    Args:
        embeddings: 嵌入矩阵
        tokens: Token 列表
        token_to_id: Token 到 ID 的映射
        positive: 正向 Token 列表（相加）
        negative: 负向 Token 列表（相减）
        
    Returns:
        (result_vector, top_k_similar) 元组
    """
    result_vec = np.zeros(embeddings.shape[1])
    
    # 加上正向向量
    for token in positive:
        if token in token_to_id:
            idx = token_to_id[token]
            result_vec += embeddings[idx]
    
    # 减去负向向量
    for token in negative:
        if token in token_to_id:
            idx = token_to_id[token]
            result_vec -= embeddings[idx]
    
    # 归一化结果向量
    norm = np.linalg.norm(result_vec)
    if norm > 0:
        result_vec = result_vec / norm
    
    # 找到最相似的 Token
    exclude_tokens = set(positive + negative)
    similarities = batch_cosine_similarity(result_vec, embeddings)
    
    results = []
    for i, (token, sim) in enumerate(zip(tokens, similarities)):
        if token in exclude_tokens:
            continue
        results.append({
            "token": token,
            "similarity": float(sim),
            "index": i,
        })
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return result_vec, results[:10]


# =============================================================================
# 聚类相似度
# =============================================================================

def compute_cluster_similarities(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> Dict[int, Dict]:
    """
    计算各聚类的内部相似度统计
    
    Args:
        embeddings: 嵌入矩阵
        cluster_labels: 聚类标签
        
    Returns:
        {cluster_id: {mean_similarity, std_similarity, size}} 字典
    """
    unique_clusters = np.unique(cluster_labels)
    cluster_stats = {}
    
    for cluster_id in unique_clusters:
        # 获取该聚类的向量
        mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[mask]
        
        if len(cluster_embeddings) < 2:
            cluster_stats[int(cluster_id)] = {
                "mean_similarity": 1.0,
                "std_similarity": 0.0,
                "size": len(cluster_embeddings),
            }
            continue
        
        # 计算聚类内相似度矩阵
        sim_matrix = compute_similarity_matrix(cluster_embeddings)
        
        # 提取上三角（排除对角线）
        upper_tri = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
        
        cluster_stats[int(cluster_id)] = {
            "mean_similarity": float(np.mean(upper_tri)),
            "std_similarity": float(np.std(upper_tri)),
            "size": len(cluster_embeddings),
        }
    
    return cluster_stats

