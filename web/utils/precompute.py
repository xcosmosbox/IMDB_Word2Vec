"""
预计算模块
==========

启动时一次性计算并缓存到文件:
1. 嵌入向量归一化矩阵 (加速相似度计算)
2. KNN 索引 (sklearn NearestNeighbors)
3. Token 列表 (全局只构建一次)
4. 2D/3D 降维坐标 (PCA/UMAP/t-SNE 多配置)

使用方法:
    from utils.precompute import (
        get_normalized_embeddings,
        get_knn_index,
        get_tokens_list,
        get_precomputed_coords,
    )
    
    # 获取归一化嵌入（加速相似度）
    norm_embeddings = get_normalized_embeddings()
    
    # 使用 KNN 索引快速搜索
    knn = get_knn_index()
    distances, indices = knn.kneighbors([query_vec])
    
    # 获取预计算的降维坐标
    coords = get_precomputed_coords("PCA", dim=3, sample_size=3000)
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle
import json

import streamlit as st
from sklearn.neighbors import NearestNeighbors

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DataFiles, DimReductionParams
from .cache_manager import (
    get_cached_or_compute,
    CACHE_DIR,
    DIM_REDUCTION_CACHE_DIR,
    KNN_CACHE_DIR,
    compute_files_hash,
    save_pickle,
    load_pickle,
)


# =============================================================================
# 降维预计算配置
# =============================================================================

# 预计算的降维配置
PRECOMPUTE_CONFIGS = [
    # PCA (最快，优先计算)
    {"method": "PCA", "dim": 2, "samples": [1000, 3000, 5000]},
    {"method": "PCA", "dim": 3, "samples": [1000, 3000, 5000]},
    # UMAP (较快)
    {"method": "UMAP", "dim": 2, "samples": [1000, 3000]},
    {"method": "UMAP", "dim": 3, "samples": [1000, 3000]},
    # t-SNE (最慢，可选)
    {"method": "t-SNE", "dim": 2, "samples": [1000, 3000]},
    {"method": "t-SNE", "dim": 3, "samples": [1000, 3000]},
]


# =============================================================================
# 基础数据加载
# =============================================================================

@st.cache_data(ttl=86400, show_spinner="加载嵌入向量...")
def _load_embeddings() -> np.ndarray:
    """加载原始嵌入向量"""
    if not DataFiles.EMBEDDINGS_NPY.exists():
        st.error(f"嵌入向量文件不存在: {DataFiles.EMBEDDINGS_NPY}")
        return np.array([])
    return np.load(DataFiles.EMBEDDINGS_NPY)


@st.cache_data(ttl=86400, show_spinner="加载 Token 映射...")
def _load_token_to_id() -> Dict[str, int]:
    """加载 Token→ID 映射"""
    if not DataFiles.TOKEN_TO_ID_JSON.exists():
        return {}
    with open(DataFiles.TOKEN_TO_ID_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=86400, show_spinner="加载 ID→Token 映射...")
def _load_id_to_token() -> Dict[int, str]:
    """加载 ID→Token 映射"""
    if not DataFiles.ID_TO_TOKEN_JSON.exists():
        return {}
    with open(DataFiles.ID_TO_TOKEN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


# =============================================================================
# 归一化嵌入
# =============================================================================

def _compute_normalized_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    计算归一化嵌入向量
    
    归一化后，余弦相似度 = 点积，大幅加速计算
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # 避免除零
    return embeddings / norms


def get_normalized_embeddings() -> np.ndarray:
    """
    获取归一化嵌入向量（带智能缓存）
    
    Returns:
        归一化后的嵌入矩阵，shape (vocab_size, embedding_dim)
    """
    embeddings = _load_embeddings()
    if len(embeddings) == 0:
        return embeddings
    
    return get_cached_or_compute(
        cache_name="normalized_embeddings",
        source_files=[DataFiles.EMBEDDINGS_NPY],
        compute_fn=lambda: _compute_normalized_embeddings(embeddings),
        cache_dir=CACHE_DIR,
        spinner_text="计算归一化嵌入...",
    )


# =============================================================================
# KNN 索引
# =============================================================================

def _build_knn_index(normalized_embeddings: np.ndarray, n_neighbors: int = 50) -> NearestNeighbors:
    """
    构建 KNN 索引
    
    Args:
        normalized_embeddings: 归一化后的嵌入矩阵
        n_neighbors: 最大邻居数
        
    Returns:
        训练好的 NearestNeighbors 对象
    """
    # 使用 brute force + cosine，对于中等规模数据效果好
    knn = NearestNeighbors(
        n_neighbors=min(n_neighbors, len(normalized_embeddings)),
        metric="cosine",
        algorithm="brute",  # brute 对 cosine 更稳定
        n_jobs=-1,  # 使用所有 CPU
    )
    knn.fit(normalized_embeddings)
    return knn


def get_knn_index(n_neighbors: int = 50) -> Optional[NearestNeighbors]:
    """
    获取 KNN 索引（带智能缓存）
    
    Args:
        n_neighbors: 最大邻居数
        
    Returns:
        训练好的 NearestNeighbors 对象
    """
    norm_embeddings = get_normalized_embeddings()
    if len(norm_embeddings) == 0:
        return None
    
    return get_cached_or_compute(
        cache_name=f"knn_index_k{n_neighbors}",
        source_files=[DataFiles.EMBEDDINGS_NPY],
        compute_fn=lambda: _build_knn_index(norm_embeddings, n_neighbors),
        cache_dir=KNN_CACHE_DIR,
        spinner_text=f"构建 KNN 索引 (k={n_neighbors})...",
    )


def knn_search(
    query_vec: np.ndarray,
    k: int = 10,
    exclude_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 KNN 索引进行快速相似度搜索
    
    Args:
        query_vec: 查询向量
        k: 返回数量
        exclude_indices: 要排除的索引列表
        
    Returns:
        (distances, indices) 元组
    """
    knn = get_knn_index(n_neighbors=max(k + 10, 50))
    if knn is None:
        return np.array([]), np.array([])
    
    # 归一化查询向量
    query_norm = np.linalg.norm(query_vec)
    if query_norm > 0:
        query_vec = query_vec / query_norm
    
    # 搜索
    distances, indices = knn.kneighbors([query_vec], n_neighbors=k + len(exclude_indices or []) + 1)
    
    # 过滤排除的索引
    if exclude_indices:
        exclude_set = set(exclude_indices)
        mask = ~np.isin(indices[0], list(exclude_set))
        indices = indices[0][mask][:k]
        distances = distances[0][mask][:k]
    else:
        indices = indices[0][:k]
        distances = distances[0][:k]
    
    # cosine distance → similarity
    similarities = 1 - distances
    
    return similarities, indices


# =============================================================================
# Token 列表
# =============================================================================

def _build_tokens_list(token_to_id: Dict[str, int], vocab_size: int) -> List[str]:
    """
    构建 Token 列表（按 ID 索引）
    
    Args:
        token_to_id: Token→ID 映射
        vocab_size: 词汇表大小
        
    Returns:
        Token 列表，tokens_list[id] = token
    """
    tokens_list = [""] * vocab_size
    for token, idx in token_to_id.items():
        if idx < vocab_size:
            tokens_list[idx] = token
    return tokens_list


def get_tokens_list() -> List[str]:
    """
    获取 Token 列表（带缓存）
    
    Returns:
        Token 列表，tokens_list[id] = token
    """
    token_to_id = _load_token_to_id()
    embeddings = _load_embeddings()
    
    if not token_to_id or len(embeddings) == 0:
        return []
    
    return get_cached_or_compute(
        cache_name="tokens_list",
        source_files=[DataFiles.TOKEN_TO_ID_JSON, DataFiles.EMBEDDINGS_NPY],
        compute_fn=lambda: _build_tokens_list(token_to_id, len(embeddings)),
        cache_dir=CACHE_DIR,
        show_spinner=False,
    )


# =============================================================================
# 降维坐标预计算
# =============================================================================

def _compute_dim_reduction(
    embeddings: np.ndarray,
    method: str,
    n_components: int,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """
    计算降维坐标
    
    Args:
        embeddings: 原始嵌入
        method: 降维方法 (PCA/UMAP/t-SNE)
        n_components: 目标维度
        sample_indices: 采样索引
        
    Returns:
        降维后的坐标，shape (n_samples, n_components)
    """
    sample_embeddings = embeddings[sample_indices]
    
    if method == "PCA":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)
        return reducer.fit_transform(sample_embeddings)
    
    elif method == "UMAP":
        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=DimReductionParams.UMAP_N_NEIGHBORS,
            min_dist=DimReductionParams.UMAP_MIN_DIST,
            random_state=DimReductionParams.UMAP_RANDOM_STATE,
        )
        return reducer.fit_transform(sample_embeddings)
    
    elif method == "t-SNE":
        from sklearn.manifold import TSNE
        import sklearn
        sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
        
        tsne_kwargs = {
            "n_components": n_components,
            "perplexity": DimReductionParams.TSNE_PERPLEXITY,
            "random_state": DimReductionParams.TSNE_RANDOM_STATE,
            "init": "pca",
        }
        
        # 兼容不同版本的 sklearn
        if sklearn_version >= (1, 5):
            tsne_kwargs["max_iter"] = DimReductionParams.TSNE_N_ITER
        else:
            tsne_kwargs["n_iter"] = DimReductionParams.TSNE_N_ITER
        
        reducer = TSNE(**tsne_kwargs)
        return reducer.fit_transform(sample_embeddings)
    
    else:
        raise ValueError(f"未知的降维方法: {method}")


def _get_sample_indices(vocab_size: int, sample_size: int, random_state: int = 42) -> np.ndarray:
    """获取固定的采样索引"""
    np.random.seed(random_state)
    return np.random.choice(vocab_size, min(sample_size, vocab_size), replace=False)


def get_precomputed_coords(
    method: str,
    dim: int,
    sample_size: int,
) -> Optional[Dict[str, Any]]:
    """
    获取预计算的降维坐标
    
    Args:
        method: 降维方法 (PCA/UMAP/t-SNE)
        dim: 目标维度 (2 或 3)
        sample_size: 采样大小
        
    Returns:
        包含 coords, indices 的字典，未找到返回 None
    """
    embeddings = _load_embeddings()
    if len(embeddings) == 0:
        return None
    
    # 获取采样索引
    sample_indices = _get_sample_indices(len(embeddings), sample_size)
    
    # 缓存名称
    cache_name = f"dim_reduction_{method}_{dim}d_{sample_size}"
    
    def compute():
        coords = _compute_dim_reduction(embeddings, method, dim, sample_indices)
        return {
            "coords": coords,
            "indices": sample_indices,
            "method": method,
            "dim": dim,
            "sample_size": sample_size,
        }
    
    return get_cached_or_compute(
        cache_name=cache_name,
        source_files=[DataFiles.EMBEDDINGS_NPY],
        compute_fn=compute,
        cache_dir=DIM_REDUCTION_CACHE_DIR,
        spinner_text=f"计算 {method} {dim}D 降维 ({sample_size} 样本)...",
    )


def precompute_all_dim_reductions(progress_callback=None) -> Dict[str, bool]:
    """
    预计算所有降维配置
    
    Args:
        progress_callback: 进度回调函数 (current, total, message)
        
    Returns:
        {config_name: success} 字典
    """
    results = {}
    total = sum(len(cfg["samples"]) for cfg in PRECOMPUTE_CONFIGS)
    current = 0
    
    for config in PRECOMPUTE_CONFIGS:
        method = config["method"]
        dim = config["dim"]
        
        for sample_size in config["samples"]:
            config_name = f"{method}_{dim}d_{sample_size}"
            current += 1
            
            if progress_callback:
                progress_callback(current, total, f"计算 {config_name}...")
            
            try:
                get_precomputed_coords(method, dim, sample_size)
                results[config_name] = True
            except Exception as e:
                print(f"预计算失败 {config_name}: {e}")
                results[config_name] = False
    
    return results


# =============================================================================
# 快速相似度搜索 (整合版)
# =============================================================================

def find_similar_fast(
    query_token: str,
    k: int = 10,
    entity_type_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    快速相似度搜索（使用 KNN 索引）
    
    Args:
        query_token: 查询 Token
        k: 返回数量
        entity_type_filter: 实体类型过滤
        
    Returns:
        相似结果列表
    """
    from .name_mapping import get_display_name, get_entity_type
    
    token_to_id = _load_token_to_id()
    tokens_list = get_tokens_list()
    embeddings = _load_embeddings()
    
    if query_token not in token_to_id:
        return []
    
    query_id = token_to_id[query_token]
    if query_id >= len(embeddings):
        return []
    
    query_vec = embeddings[query_id]
    
    # 使用 KNN 搜索
    similarities, indices = knn_search(
        query_vec,
        k=k + 10,  # 多取一些用于过滤
        exclude_indices=[query_id],
    )
    
    results = []
    for sim, idx in zip(similarities, indices):
        if idx >= len(tokens_list):
            continue
        
        token = tokens_list[idx]
        if not token:
            continue
        
        # 类型过滤
        if entity_type_filter:
            token_type = get_entity_type(token)
            if token_type != entity_type_filter:
                continue
        
        results.append({
            "token": token,
            "name": get_display_name(token),
            "similarity": float(sim),
            "index": int(idx),
        })
        
        if len(results) >= k:
            break
    
    return results


# =============================================================================
# 初始化函数
# =============================================================================

def initialize_precompute(show_progress: bool = True) -> None:
    """
    初始化预计算（应用启动时调用）
    
    预加载常用数据到缓存。
    """
    if show_progress:
        with st.spinner("初始化预计算数据..."):
            # 预加载归一化嵌入
            get_normalized_embeddings()
            
            # 预加载 KNN 索引
            get_knn_index()
            
            # 预加载 Token 列表
            get_tokens_list()
            
            # 预加载 PCA 2D/3D (最常用)
            get_precomputed_coords("PCA", 2, 3000)
            get_precomputed_coords("PCA", 3, 3000)
    else:
        get_normalized_embeddings()
        get_knn_index()
        get_tokens_list()
        get_precomputed_coords("PCA", 2, 3000)
        get_precomputed_coords("PCA", 3, 3000)

