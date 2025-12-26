"""
名称映射模块 (高性能版)
========================

将内部 Token (如 MOV_tt0111161, ACT_nm0000001) 映射为真实的电影名/演员名/导演名。

优化策略:
- 使用 Pickle 格式缓存（比 JSON 快 10-50 倍）
- 向量化操作替代 iterrows（快 100 倍）
- 分片缓存（电影/人员分开存储）
- Streamlit 内存缓存 + 磁盘缓存双重加速

使用方法:
    from utils.name_mapping import get_display_name, fuzzy_search, search_entities
    
    # 单个 token 转换
    name = get_display_name("MOV_tt0111161")  # -> "The Shawshank Redemption"
    
    # 模糊搜索
    results = fuzzy_search("shawshnk", limit=5)  # 容忍拼写错误
"""
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import streamlit as st

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROJECT_ROOT, ENTITY_TYPE_NAMES

# 导入缓存管理器
from .cache_manager import CACHE_DIR, compute_file_hash


# =============================================================================
# 数据文件路径
# =============================================================================

# 缓存目录 (预处理生成的 CSV 文件)
SOURCE_CACHE_DIR = PROJECT_ROOT / "imdb_word2vec" / "cache"

# 电影信息文件
MOVIES_INFO_CSV = SOURCE_CACHE_DIR / "movies_info_df.csv"

# 人员信息文件
STAFF_DF_CSV = SOURCE_CACHE_DIR / "staff_df.csv"

# 名称映射缓存目录
NAME_MAPPING_CACHE_DIR = CACHE_DIR / "name_mapping"
NAME_MAPPING_CACHE_DIR.mkdir(exist_ok=True)


# =============================================================================
# 预定义映射 (类型、年代、评分等)
# =============================================================================

# 类型映射 (英文 -> 中文)
GENRE_MAPPING = {
    "GEN_Action": "动作",
    "GEN_Adventure": "冒险",
    "GEN_Animation": "动画",
    "GEN_Biography": "传记",
    "GEN_Comedy": "喜剧",
    "GEN_Crime": "犯罪",
    "GEN_Documentary": "纪录片",
    "GEN_Drama": "剧情",
    "GEN_Family": "家庭",
    "GEN_Fantasy": "奇幻",
    "GEN_Film-Noir": "黑色电影",
    "GEN_History": "历史",
    "GEN_Horror": "恐怖",
    "GEN_Music": "音乐",
    "GEN_Musical": "歌舞",
    "GEN_Mystery": "悬疑",
    "GEN_News": "新闻",
    "GEN_Reality-TV": "真人秀",
    "GEN_Romance": "爱情",
    "GEN_Sci-Fi": "科幻",
    "GEN_Short": "短片",
    "GEN_Sport": "运动",
    "GEN_Talk-Show": "脱口秀",
    "GEN_Thriller": "惊悚",
    "GEN_War": "战争",
    "GEN_Western": "西部",
    "GEN_Adult": "成人",
    "GEN_Game-Show": "游戏节目",
}

# 年代映射
ERA_MAPPING = {
    "ERA_SILENT": "默片时代",
    "ERA_1920s": "1920年代",
    "ERA_1930s": "1930年代",
    "ERA_1940s": "1940年代",
    "ERA_1950s": "1950年代",
    "ERA_1960s": "1960年代",
    "ERA_1970s": "1970年代",
    "ERA_1980s": "1980年代",
    "ERA_1990s": "1990年代",
    "ERA_2000s": "2000年代",
    "ERA_2010s": "2010年代",
    "ERA_2020s": "2020年代",
    "ERA_UNKNOWN": "未知年代",
}

# 作品类型映射
TYPE_MAPPING = {
    "TYP_movie": "电影",
    "TYP_short": "短片",
    "TYP_tvSeries": "电视剧",
    "TYP_tvMiniSeries": "迷你剧",
    "TYP_tvMovie": "电视电影",
    "TYP_tvSpecial": "电视特辑",
    "TYP_video": "视频",
    "TYP_videoGame": "视频游戏",
    "TYP_tvEpisode": "剧集",
}


# =============================================================================
# 高性能数据构建
# =============================================================================

def _get_cache_version() -> str:
    """
    计算源文件的版本哈希，用于缓存失效判断
    """
    version_parts = []
    if MOVIES_INFO_CSV.exists():
        version_parts.append(compute_file_hash(MOVIES_INFO_CSV))
    if STAFF_DF_CSV.exists():
        version_parts.append(compute_file_hash(STAFF_DF_CSV))
    
    if not version_parts:
        return "no_source"
    
    import hashlib
    combined = hashlib.md5("_".join(version_parts).encode()).hexdigest()[:8]
    return combined


def _build_movie_mapping_fast() -> Dict[str, str]:
    """
    使用向量化操作快速构建电影映射
    
    Returns:
        {MOV_ttXXXXXX: title} 字典
    """
    if not MOVIES_INFO_CSV.exists():
        return {}
    
    try:
        # 只读取需要的列，使用 PyArrow 引擎加速
        df = pd.read_csv(
            MOVIES_INFO_CSV,
            usecols=["tconst", "title"],
            dtype=str,
            engine="pyarrow" if "pyarrow" in pd.io.parsers.readers.__dict__.get("_c_parser_defaults", {}) else "c",
            na_filter=False,  # 跳过 NA 检测，更快
        )
        
        # 向量化操作：比 iterrows 快 100 倍
        # 过滤空值
        mask = (df["tconst"] != "") & (df["title"] != "")
        df = df[mask]
        
        # 构建 token
        tokens = "MOV_" + df["tconst"]
        
        # 直接转为字典
        return dict(zip(tokens, df["title"]))
        
    except Exception as e:
        st.warning(f"加载电影名称失败: {e}")
        return {}


def _build_staff_mapping_fast() -> Dict[str, str]:
    """
    使用向量化操作快速构建人员映射
    
    Returns:
        {ACT_nmXXXX: name, DIR_nmXXXX: name, PER_nmXXXX: name} 字典
    """
    if not STAFF_DF_CSV.exists():
        return {}
    
    try:
        df = pd.read_csv(
            STAFF_DF_CSV,
            usecols=["nconst", "primaryName"],
            dtype=str,
            na_filter=False,
        )
        
        # 过滤空值
        mask = (df["nconst"] != "") & (df["primaryName"] != "")
        df = df[mask]
        
        # 使用 pandas 字符串操作（比 numpy 更健壮）
        nconsts = df["nconst"].values
        names = df["primaryName"].values
        
        # 为每个人员创建 ACT_, DIR_, PER_ 三个映射
        mapping = {}
        
        for prefix in ["ACT_", "DIR_", "PER_"]:
            # 直接使用列表推导，简单可靠
            tokens = [f"{prefix}{nc}" for nc in nconsts]
            mapping.update(dict(zip(tokens, names)))
        
        return mapping
        
    except Exception as e:
        print(f"加载人员名称失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _build_static_mapping() -> Dict[str, str]:
    """
    构建静态映射（类型、年代、评分）
    """
    mapping = {}
    mapping.update(GENRE_MAPPING)
    mapping.update(ERA_MAPPING)
    mapping.update(TYPE_MAPPING)
    
    # 评分映射
    for i in range(0, 101):
        rating = i / 10
        mapping[f"RAT_{rating}"] = f"⭐ {rating}分"
        mapping[f"RAT_{rating:.1f}"] = f"⭐ {rating:.1f}分"
    
    return mapping


def _load_or_build_mapping(cache_name: str, build_fn, version: str) -> Dict[str, str]:
    """
    加载缓存或构建映射（使用 Pickle）
    """
    cache_file = NAME_MAPPING_CACHE_DIR / f"{cache_name}_{version}.pkl"
    
    # 尝试加载缓存
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass  # 缓存损坏，重新构建
    
    # 构建映射
    mapping = build_fn()
    
    # 保存缓存
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    
    return mapping


# =============================================================================
# 分片加载函数
# =============================================================================

@st.cache_data(ttl=None, show_spinner=False)
def _load_movie_mapping() -> Dict[str, str]:
    """加载电影映射（Streamlit 内存缓存）"""
    version = _get_cache_version()
    return _load_or_build_mapping("movies", _build_movie_mapping_fast, version)


@st.cache_data(ttl=None, show_spinner=False)
def _load_staff_mapping() -> Dict[str, str]:
    """加载人员映射（Streamlit 内存缓存）"""
    version = _get_cache_version()
    return _load_or_build_mapping("staff", _build_staff_mapping_fast, version)


@st.cache_data(ttl=None, show_spinner=False)
def _load_static_mapping() -> Dict[str, str]:
    """加载静态映射"""
    return _build_static_mapping()


# =============================================================================
# 并行加载支持
# =============================================================================

def _load_all_mappings_parallel() -> Dict[str, str]:
    """
    并行加载所有映射分片
    
    使用线程池并行加载电影和人员映射，提升加载速度
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    mapping = {}
    
    # 静态映射（同步，很快）
    mapping.update(_build_static_mapping())
    
    version = _get_cache_version()
    
    # 定义加载任务
    tasks = {
        "movies": lambda: _load_or_build_mapping("movies", _build_movie_mapping_fast, version),
        "staff": lambda: _load_or_build_mapping("staff", _build_staff_mapping_fast, version),
    }
    
    # 并行执行
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fn): name for name, fn in tasks.items()}
        
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                mapping.update(result)
            except Exception as e:
                print(f"加载 {name} 映射失败: {e}")
    
    return mapping


# =============================================================================
# 主加载函数
# =============================================================================

@st.cache_data(ttl=None, show_spinner="加载名称映射...")
def load_name_mapping() -> Dict[str, str]:
    """
    加载完整的 Token → 名称 映射
    
    使用分片+并行加载策略：
    - 电影和人员映射并行加载
    - 每个分片独立缓存（Pickle 格式）
    - Streamlit 内存缓存确保只加载一次
    
    Returns:
        {token: display_name} 字典
    """
    return _load_all_mappings_parallel()


@st.cache_data(ttl=None, show_spinner=False)
def load_reverse_mapping() -> Dict[str, str]:
    """
    加载 名称 → Token 的反向映射
    
    Returns:
        {lowercase_name: token} 字典
    """
    forward = load_name_mapping()
    
    reverse = {}
    for token, name in forward.items():
        key = name.lower()
        # 优先保留电影/演员（而非类型等）
        if key not in reverse or token.startswith(("MOV_", "ACT_", "DIR_")):
            reverse[key] = token
    
    return reverse


@st.cache_data(ttl=None, show_spinner=False)
def load_search_list() -> List[Tuple[str, str, str]]:
    """
    加载搜索列表
    
    Returns:
        [(display_name, token, entity_type), ...] 列表
    """
    forward = load_name_mapping()
    
    search_list = []
    for token, name in forward.items():
        entity_type = token.split("_")[0] if "_" in token else "OTHER"
        search_list.append((name, token, entity_type))
    
    return search_list


# =============================================================================
# 查询函数
# =============================================================================

def get_display_name(token: str) -> str:
    """
    获取 Token 的显示名称
    
    Args:
        token: 内部 Token，如 "MOV_tt0111161"
        
    Returns:
        显示名称，如 "The Shawshank Redemption"
    """
    # 快速路径：检查静态映射
    if token in GENRE_MAPPING:
        return GENRE_MAPPING[token]
    if token in ERA_MAPPING:
        return ERA_MAPPING[token]
    if token in TYPE_MAPPING:
        return TYPE_MAPPING[token]
    if token.startswith("RAT_"):
        try:
            rating = float(token[4:])
            return f"⭐ {rating}分"
        except ValueError:
            pass
    
    # 加载完整映射
    mapping = load_name_mapping()
    return mapping.get(token, token)


def token_to_display(token: str, show_token: bool = False) -> str:
    """
    将 Token 转换为显示文本
    
    Args:
        token: Token 字符串
        show_token: 是否在名称后显示 token
        
    Returns:
        格式化的显示文本
    """
    name = get_display_name(token)
    if show_token and name != token:
        return f"{name} ({token})"
    return name


def get_entity_type_name(token: str) -> str:
    """
    获取实体类型的中文名称
    
    Args:
        token: Token 字符串
        
    Returns:
        类型名称，如 "电影", "演员"
    """
    prefix = token.split("_")[0] if "_" in token else "OTHER"
    return ENTITY_TYPE_NAMES.get(prefix, "未知")


# =============================================================================
# 模糊搜索
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fuzzy_search(
    query: str,
    limit: int = 10,
    threshold: int = 60,
    entity_types: Optional[List[str]] = None,
) -> List[Tuple[str, str, str, float]]:
    """
    模糊搜索实体
    
    Args:
        query: 搜索词
        limit: 返回数量限制
        threshold: 相似度阈值 (0-100)
        entity_types: 限制实体类型，如 ["MOV", "ACT"]
        
    Returns:
        [(display_name, token, entity_type, score), ...] 列表
    """
    if not query or len(query) < 1:
        return []
    
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        # 降级为精确匹配
        return exact_search(query, limit, entity_types)
    
    search_list = load_search_list()
    
    # 过滤实体类型
    if entity_types:
        search_list = [
            (name, token, etype)
            for name, token, etype in search_list
            if etype in entity_types
        ]
    
    if not search_list:
        return []
    
    # 构建搜索字典
    name_to_info = {name: (token, etype) for name, token, etype in search_list}
    names = list(name_to_info.keys())
    
    # 使用 rapidfuzz 进行模糊匹配
    results = process.extract(
        query,
        names,
        scorer=fuzz.WRatio,
        limit=limit,
        score_cutoff=threshold,
    )
    
    return [
        (name, name_to_info[name][0], name_to_info[name][1], score)
        for name, score, _ in results
    ]


def exact_search(
    query: str,
    limit: int = 10,
    entity_types: Optional[List[str]] = None,
) -> List[Tuple[str, str, str, float]]:
    """
    精确子串搜索（降级方案）
    """
    query_lower = query.lower()
    search_list = load_search_list()
    
    results = []
    for name, token, etype in search_list:
        if entity_types and etype not in entity_types:
            continue
        if query_lower in name.lower():
            # 使用简单的匹配度计算
            score = len(query) / len(name) * 100
            results.append((name, token, etype, score))
    
    # 按匹配度排序
    results.sort(key=lambda x: x[3], reverse=True)
    return results[:limit]


def search_entities(
    query: str,
    limit: int = 10,
    entity_types: Optional[List[str]] = None,
) -> List[str]:
    """
    搜索实体，返回 token 列表
    
    Args:
        query: 搜索词
        limit: 返回数量
        entity_types: 限制类型
        
    Returns:
        [token1, token2, ...] 列表
    """
    results = fuzzy_search(query, limit=limit, entity_types=entity_types)
    return [token for _, token, _, _ in results]


def get_popular_entities(
    limit: int = 10,
    entity_types: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """
    获取热门实体（用于空搜索时显示）
    
    返回电影和人员各一半
    """
    search_list = load_search_list()
    
    if entity_types:
        search_list = [
            (name, token, etype)
            for name, token, etype in search_list
            if etype in entity_types
        ]
    
    # 简单策略：返回前 N 个
    # 实际可以基于评分或其他指标排序
    return search_list[:limit]


# =============================================================================
# 反向查询
# =============================================================================

def name_to_token(name: str) -> Optional[str]:
    """
    通过名称查找 token
    
    Args:
        name: 显示名称
        
    Returns:
        对应的 token，找不到返回 None
    """
    reverse = load_reverse_mapping()
    return reverse.get(name.lower())


def batch_get_display_names(tokens: List[str]) -> Dict[str, str]:
    """
    批量获取显示名称（更高效）
    
    Args:
        tokens: token 列表
        
    Returns:
        {token: display_name} 字典
    """
    mapping = load_name_mapping()
    return {token: mapping.get(token, token) for token in tokens}


# =============================================================================
# 工具函数
# =============================================================================

def format_entity_display(
    token: str,
    include_type: bool = True,
    include_token: bool = False,
) -> str:
    """
    格式化实体显示
    
    Args:
        token: Token
        include_type: 是否包含类型标签
        include_token: 是否包含原始 token
        
    Returns:
        格式化字符串
    """
    name = get_display_name(token)
    parts = [name]
    
    if include_type:
        type_name = get_entity_type_name(token)
        parts.append(f"[{type_name}]")
    
    if include_token and name != token:
        parts.append(f"({token})")
    
    return " ".join(parts)


def get_entity_emoji(token: str) -> str:
    """
    获取实体类型的 emoji
    """
    prefix = token.split("_")[0] if "_" in token else "OTHER"
    emoji_map = {
        "MOV": "🎬",
        "ACT": "🎭",
        "DIR": "🎬",
        "PER": "👤",
        "GEN": "🏷️",
        "ERA": "📅",
        "TYP": "📁",
        "RAT": "⭐",
    }
    return emoji_map.get(prefix, "📌")


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


def search_by_name(
    query: str,
    limit: int = 10,
    entity_types: Optional[List[str]] = None,
) -> List[str]:
    """
    通过名称搜索实体（返回 token 列表）
    
    这是 fuzzy_search 的简化版本，只返回 token
    
    Args:
        query: 搜索词
        limit: 返回数量
        entity_types: 限制类型
        
    Returns:
        [token1, token2, ...] 列表
    """
    results = fuzzy_search(query, limit=limit, entity_types=entity_types)
    return [token for _, token, _, _ in results]
