"""
数据加载模块
===========

负责加载和解析所有12个导出文件，使用 Streamlit 缓存机制提升性能。

文件列表:
- embeddings.npy: 原始128维向量
- embeddings.json: 完整Token+向量
- clustering.json: 预计算t-SNE坐标+聚类标签
- word2vec.onnx: ONNX推理模型
- metadata.tsv: Token列表
- vectors.tsv: TF Projector兼容格式
- embedding_tsne.png: 静态t-SNE图
- recsys/config.json: 系统配置
- recsys/token_to_id.json: Token→ID映射
- recsys/id_to_token.json: ID→Token映射
- recsys/entity_index.json: 实体分类索引
- visualization.html: 预留空文件
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DataFiles, ENTITY_TYPE_NAMES


# =============================================================================
# 配置文件加载
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="加载系统配置...")
def load_config() -> Dict[str, Any]:
    """
    加载推荐系统配置文件 (recsys/config.json)
    
    Returns:
        配置字典，包含:
        - vocab_size: 词汇表大小
        - embedding_dim: 嵌入维度
        - entity_types: 各实体类型数量
        - similarity_metric: 相似度计算方式
        - top_k_default: 默认返回数量
        - files: 相关文件名映射
    """
    config_path = DataFiles.CONFIG_JSON
    
    if not config_path.exists():
        st.error(f"配置文件不存在: {config_path}")
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config


# =============================================================================
# 聚类数据加载
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="加载聚类数据...")
def load_clustering_data() -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """
    加载聚类可视化数据 (clustering.json)
    
    Returns:
        (points_df, clusters, metadata) 元组:
        - points_df: 包含 token, x, y, cluster, type 的 DataFrame
        - clusters: 聚类中心列表
        - metadata: 元数据字典
    """
    clustering_path = DataFiles.CLUSTERING_JSON
    
    if not clustering_path.exists():
        st.error(f"聚类数据文件不存在: {clustering_path}")
        return pd.DataFrame(), [], {}
    
    with open(clustering_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 解析点数据
    points_df = pd.DataFrame(data.get("points", []))
    
    # 添加中文类型名称
    if not points_df.empty and "type" in points_df.columns:
        points_df["type_name"] = points_df["type"].map(
            lambda x: ENTITY_TYPE_NAMES.get(x, x)
        )
    
    # 解析聚类中心
    clusters = data.get("clusters", [])
    
    # 解析元数据
    metadata = data.get("metadata", {})
    
    return points_df, clusters, metadata


# =============================================================================
# 嵌入数据加载
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="加载嵌入向量 (NPY)...")
def load_embeddings_npy() -> np.ndarray:
    """
    加载原始嵌入向量 (embeddings.npy)
    
    用于降维计算 (PCA/UMAP/t-SNE)
    
    Returns:
        shape 为 (vocab_size, embedding_dim) 的 numpy 数组
    """
    npy_path = DataFiles.EMBEDDINGS_NPY
    
    if not npy_path.exists():
        st.error(f"嵌入向量文件不存在: {npy_path}")
        return np.array([])
    
    embeddings = np.load(npy_path)
    return embeddings


@st.cache_data(ttl=3600, show_spinner="加载嵌入数据 (JSON)...")
def load_embeddings_json() -> Tuple[List[str], np.ndarray, Dict]:
    """
    加载完整嵌入数据 (embeddings.json)
    
    注意: 此文件较大 (~130MB)，首次加载可能较慢
    
    Returns:
        (tokens, embeddings, metadata) 元组:
        - tokens: Token 列表
        - embeddings: 嵌入向量数组
        - metadata: 元数据字典
    """
    json_path = DataFiles.EMBEDDINGS_JSON
    
    if not json_path.exists():
        st.error(f"嵌入JSON文件不存在: {json_path}")
        return [], np.array([]), {}
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tokens = data.get("tokens", [])
    embeddings = np.array(data.get("embeddings", []))
    metadata = data.get("metadata", {})
    
    return tokens, embeddings, metadata


# =============================================================================
# 元数据加载
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="加载元数据...")
def load_metadata() -> pd.DataFrame:
    """
    加载 Token 元数据列表 (metadata.tsv)
    
    Returns:
        包含 token 和 entity_type 列的 DataFrame
    """
    tsv_path = DataFiles.METADATA_TSV
    
    if not tsv_path.exists():
        st.error(f"元数据文件不存在: {tsv_path}")
        return pd.DataFrame()
    
    # 读取 TSV 文件 (单列，无表头)
    tokens = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.append(token)
    
    # 创建 DataFrame
    df = pd.DataFrame({"token": tokens})
    
    # 添加实体类型列
    df["entity_type"] = df["token"].apply(get_entity_type)
    df["type_name"] = df["entity_type"].map(
        lambda x: ENTITY_TYPE_NAMES.get(x, x)
    )
    
    # 添加索引列
    df["id"] = range(len(df))
    
    return df


# =============================================================================
# 映射文件加载
# =============================================================================

@st.cache_data(ttl=3600, show_spinner="加载 Token→ID 映射...")
def load_token_to_id() -> Dict[str, int]:
    """
    加载 Token 到 ID 的映射 (recsys/token_to_id.json)
    
    Returns:
        {token: id} 字典
    """
    json_path = DataFiles.TOKEN_TO_ID_JSON
    
    if not json_path.exists():
        st.error(f"Token映射文件不存在: {json_path}")
        return {}
    
    with open(json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    return mapping


@st.cache_data(ttl=3600, show_spinner="加载 ID→Token 映射...")
def load_id_to_token() -> Dict[int, str]:
    """
    加载 ID 到 Token 的映射 (recsys/id_to_token.json)
    
    Returns:
        {id: token} 字典
    """
    json_path = DataFiles.ID_TO_TOKEN_JSON
    
    if not json_path.exists():
        st.error(f"ID映射文件不存在: {json_path}")
        return {}
    
    with open(json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    # JSON 的 key 是字符串，需要转换为整数
    return {int(k): v for k, v in mapping.items()}


@st.cache_data(ttl=3600, show_spinner="加载实体索引...")
def load_entity_index() -> Dict[str, List[str]]:
    """
    加载实体分类索引 (recsys/entity_index.json)
    
    Returns:
        {entity_type: [token1, token2, ...]} 字典
    """
    json_path = DataFiles.ENTITY_INDEX_JSON
    
    if not json_path.exists():
        st.error(f"实体索引文件不存在: {json_path}")
        return {}
    
    with open(json_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    
    return index


# =============================================================================
# 辅助函数
# =============================================================================

def get_entity_type(token: str) -> str:
    """
    从 Token 中提取实体类型前缀
    
    Args:
        token: 如 "MOV_tt0111161", "ACT_nm0000001"
        
    Returns:
        实体类型前缀，如 "MOV", "ACT"
    """
    if "_" in token:
        return token.split("_")[0]
    return "OTHER"


def get_token_info(token: str) -> Dict[str, Any]:
    """
    获取单个 Token 的完整信息
    
    Args:
        token: Token 字符串
        
    Returns:
        包含 id, type, type_name 等信息的字典
    """
    token_to_id = load_token_to_id()
    entity_type = get_entity_type(token)
    
    return {
        "token": token,
        "id": token_to_id.get(token, -1),
        "entity_type": entity_type,
        "type_name": ENTITY_TYPE_NAMES.get(entity_type, entity_type),
    }


def search_tokens(query: str, limit: int = 20, include_names: bool = True) -> List[str]:
    """
    搜索 Token (支持 Token 和名称的模糊匹配)
    
    Args:
        query: 搜索关键词（可以是 Token 或真实名称）
        limit: 返回结果数量上限
        include_names: 是否同时搜索真实名称
        
    Returns:
        匹配的 Token 列表
    """
    token_to_id = load_token_to_id()
    query_lower = query.lower()
    
    # 先搜索 Token
    matches = [
        token for token in token_to_id.keys()
        if query_lower in token.lower()
    ]
    
    # 如果启用名称搜索，也搜索真实名称
    if include_names and len(matches) < limit:
        try:
            from .name_mapping import search_by_name
            name_matches = search_by_name(query, limit=limit - len(matches))
            # 合并结果，去重
            for token in name_matches:
                if token not in matches:
                    matches.append(token)
        except ImportError:
            pass
    
    return matches[:limit]


def get_embedding_vector(token: str) -> Optional[np.ndarray]:
    """
    获取单个 Token 的嵌入向量
    
    Args:
        token: Token 字符串
        
    Returns:
        128维嵌入向量，如果未找到则返回 None
    """
    token_to_id = load_token_to_id()
    
    if token not in token_to_id:
        return None
    
    token_id = token_to_id[token]
    embeddings = load_embeddings_npy()
    
    if token_id >= len(embeddings):
        return None
    
    return embeddings[token_id]


# =============================================================================
# 文件信息
# =============================================================================

def get_data_files_info() -> List[Dict[str, Any]]:
    """
    获取所有数据文件的信息
    
    Returns:
        文件信息列表，每个元素包含 name, path, exists, size 等
    """
    files_info = []
    
    # 定义所有文件
    files = [
        ("embeddings.npy", DataFiles.EMBEDDINGS_NPY),
        ("embeddings.json", DataFiles.EMBEDDINGS_JSON),
        ("clustering.json", DataFiles.CLUSTERING_JSON),
        ("word2vec.onnx", DataFiles.WORD2VEC_ONNX),
        ("metadata.tsv", DataFiles.METADATA_TSV),
        ("vectors.tsv", DataFiles.VECTORS_TSV),
        ("embedding_tsne.png", DataFiles.EMBEDDING_TSNE_PNG),
        ("config.json", DataFiles.CONFIG_JSON),
        ("token_to_id.json", DataFiles.TOKEN_TO_ID_JSON),
        ("id_to_token.json", DataFiles.ID_TO_TOKEN_JSON),
        ("entity_index.json", DataFiles.ENTITY_INDEX_JSON),
        ("visualization.html", DataFiles.VISUALIZATION_HTML),
    ]
    
    for name, path in files:
        info = {
            "name": name,
            "path": str(path),
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
            "size_mb": round(path.stat().st_size / (1024**2), 2) if path.exists() else 0,
        }
        files_info.append(info)
    
    return files_info

