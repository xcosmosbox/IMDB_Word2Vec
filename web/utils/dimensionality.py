"""
降维算法模块
===========

封装 PCA、UMAP、t-SNE 三种降维算法，支持缓存和参数配置。

使用方法:
    # 获取预计算或实时计算的降维结果
    coords = get_cached_reduction("pca", embeddings)
    
    # 直接计算
    coords = compute_pca(embeddings)
"""
from pathlib import Path
from typing import Optional, Literal
import hashlib

import numpy as np
import streamlit as st

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CACHE_DIR, DimReductionParams


# 降维方法类型
ReductionMethod = Literal["pca", "umap", "tsne"]


# =============================================================================
# PCA 降维
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="计算 PCA 降维...")
def compute_pca(
    embeddings: np.ndarray,
    n_components: int = DimReductionParams.PCA_N_COMPONENTS,
) -> np.ndarray:
    """
    使用 PCA 进行降维
    
    PCA (Principal Component Analysis) 是一种线性降维方法，
    通过找到数据方差最大的方向来保留最重要的信息。
    
    优点: 速度快，保留全局结构
    缺点: 线性方法，可能丢失非线性关系
    
    Args:
        embeddings: shape (n_samples, n_features) 的嵌入矩阵
        n_components: 目标维度数，默认为 2
        
    Returns:
        shape (n_samples, n_components) 的降维结果
    """
    from sklearn.decomposition import PCA
    
    # 创建 PCA 模型
    pca = PCA(n_components=n_components, random_state=42)
    
    # 执行降维
    reduced = pca.fit_transform(embeddings)
    
    # 记录解释方差比
    explained_var = pca.explained_variance_ratio_
    st.session_state["pca_explained_variance"] = explained_var
    
    return reduced


# =============================================================================
# UMAP 降维
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="计算 UMAP 降维 (可能需要几分钟)...")
def compute_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = DimReductionParams.UMAP_N_NEIGHBORS,
    min_dist: float = DimReductionParams.UMAP_MIN_DIST,
    random_state: int = DimReductionParams.UMAP_RANDOM_STATE,
) -> np.ndarray:
    """
    使用 UMAP 进行降维
    
    UMAP (Uniform Manifold Approximation and Projection) 是一种非线性降维方法，
    比 t-SNE 更快，同时保留更多全局结构。
    
    优点: 速度较快，保留局部和全局结构
    缺点: 需要调参
    
    Args:
        embeddings: shape (n_samples, n_features) 的嵌入矩阵
        n_components: 目标维度数，默认为 2
        n_neighbors: 近邻数量，影响局部结构保留
        min_dist: 点之间的最小距离，影响紧密程度
        random_state: 随机种子
        
    Returns:
        shape (n_samples, n_components) 的降维结果
    """
    import umap
    
    # 创建 UMAP 模型
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric="cosine",  # 对于文本嵌入，余弦距离通常效果更好
    )
    
    # 执行降维
    reduced = reducer.fit_transform(embeddings)
    
    return reduced


# =============================================================================
# t-SNE 降维
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="计算 t-SNE 降维 (可能需要几分钟)...")
def compute_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = DimReductionParams.TSNE_PERPLEXITY,
    max_iter: int = DimReductionParams.TSNE_N_ITER,
    random_state: int = DimReductionParams.TSNE_RANDOM_STATE,
) -> np.ndarray:
    """
    使用 t-SNE 进行降维
    
    t-SNE (t-distributed Stochastic Neighbor Embedding) 是一种非线性降维方法，
    特别擅长保留局部结构，常用于可视化高维数据。
    
    优点: 保留局部结构，可视化效果好
    缺点: 速度慢，不保留全局结构，不同运行结果可能不同
    
    Args:
        embeddings: shape (n_samples, n_features) 的嵌入矩阵
        n_components: 目标维度数，默认为 2
        perplexity: 困惑度，影响局部结构保留
        max_iter: 最大迭代次数 (新版 sklearn 使用 max_iter 替代 n_iter)
        random_state: 随机种子
        
    Returns:
        shape (n_samples, n_components) 的降维结果
    """
    from sklearn.manifold import TSNE
    import sklearn
    
    # 检查 sklearn 版本，决定使用 n_iter 还是 max_iter
    sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
    
    # sklearn >= 1.5 使用 max_iter，之前版本使用 n_iter
    if sklearn_version >= (1, 5):
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
            init="pca",
        )
    else:
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=max_iter,
            random_state=random_state,
            init="pca",
        )
    
    # 执行降维
    reduced = tsne.fit_transform(embeddings)
    
    return reduced


# =============================================================================
# 缓存管理
# =============================================================================

def _get_cache_path(method: str, data_hash: str) -> Path:
    """获取缓存文件路径"""
    return CACHE_DIR / f"{method}_{data_hash}.npy"


def _compute_data_hash(data: np.ndarray) -> str:
    """计算数据的哈希值，用于缓存标识"""
    # 使用数据的形状和部分内容计算哈希
    hash_input = f"{data.shape}_{data[:100].tobytes()}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:8]


def get_cached_reduction(
    method: ReductionMethod,
    embeddings: np.ndarray,
    force_recompute: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    获取降维结果（优先使用缓存）
    
    首次计算时会自动保存到缓存目录，后续调用直接读取缓存。
    
    Args:
        method: 降维方法 ("pca", "umap", "tsne")
        embeddings: 嵌入矩阵
        force_recompute: 是否强制重新计算
        **kwargs: 传递给降维函数的额外参数
        
    Returns:
        降维后的坐标数组
    """
    # 计算数据哈希
    data_hash = _compute_data_hash(embeddings)
    cache_path = _get_cache_path(method, data_hash)
    
    # 检查缓存
    if cache_path.exists() and not force_recompute:
        try:
            cached = np.load(cache_path)
            return cached
        except Exception:
            pass  # 缓存损坏，重新计算
    
    # 选择降维方法
    if method == "pca":
        reduced = compute_pca(embeddings, **kwargs)
    elif method == "umap":
        reduced = compute_umap(embeddings, **kwargs)
    elif method == "tsne":
        reduced = compute_tsne(embeddings, **kwargs)
    else:
        raise ValueError(f"未知的降维方法: {method}")
    
    # 保存缓存
    try:
        np.save(cache_path, reduced)
    except Exception as e:
        st.warning(f"保存缓存失败: {e}")
    
    return reduced


def clear_cache(method: Optional[str] = None) -> int:
    """
    清除降维缓存
    
    Args:
        method: 指定要清除的方法缓存，None 表示清除全部
        
    Returns:
        清除的文件数量
    """
    count = 0
    
    for cache_file in CACHE_DIR.glob("*.npy"):
        if method is None or cache_file.name.startswith(f"{method}_"):
            cache_file.unlink()
            count += 1
    
    return count


# =============================================================================
# 采样功能
# =============================================================================

def sample_embeddings(
    embeddings: np.ndarray,
    n_samples: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对嵌入矩阵进行采样
    
    当数据量过大时，可以先采样再进行降维，提高速度。
    
    Args:
        embeddings: 完整嵌入矩阵
        n_samples: 采样数量
        random_state: 随机种子
        
    Returns:
        (sampled_embeddings, sample_indices) 元组
    """
    n_total = len(embeddings)
    
    if n_samples >= n_total:
        return embeddings, np.arange(n_total)
    
    np.random.seed(random_state)
    indices = np.random.choice(n_total, n_samples, replace=False)
    indices = np.sort(indices)
    
    return embeddings[indices], indices


# =============================================================================
# 降维方法比较
# =============================================================================

def compare_reductions(
    embeddings: np.ndarray,
    methods: list[ReductionMethod] = ["pca", "umap", "tsne"],
    n_samples: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """
    计算多种降维方法的结果进行比较
    
    Args:
        embeddings: 嵌入矩阵
        methods: 要比较的方法列表
        n_samples: 采样数量，None 表示使用全部数据
        
    Returns:
        {method_name: reduced_coords} 字典
    """
    # 采样
    if n_samples is not None:
        sampled, indices = sample_embeddings(embeddings, n_samples)
    else:
        sampled = embeddings
        indices = np.arange(len(embeddings))
    
    results = {"indices": indices}
    
    for method in methods:
        results[method] = get_cached_reduction(method, sampled)
    
    return results

