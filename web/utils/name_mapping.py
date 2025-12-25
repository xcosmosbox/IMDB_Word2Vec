"""
名称映射模块
============

将内部 Token (如 MOV_tt0111161, ACT_nm0000001) 映射为真实的电影名/演员名/导演名。

数据来源:
- movies_info_df.csv: 电影信息 (tconst -> title)
- staff_df.csv: 人员信息 (nconst -> primaryName)

使用方法:
    from utils.name_mapping import get_display_name, token_to_display
    
    # 单个 token 转换
    name = get_display_name("MOV_tt0111161")  # -> "The Shawshank Redemption"
    
    # 格式化为显示文本
    display = token_to_display("MOV_tt0111161")  # -> "The Shawshank Redemption (MOV_tt0111161)"
"""
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROJECT_ROOT, ENTITY_TYPE_NAMES


# =============================================================================
# 缓存文件路径
# =============================================================================

# 缓存目录 (预处理生成的 CSV 文件)
CACHE_DIR = PROJECT_ROOT / "imdb_word2vec" / "cache"

# 电影信息文件
MOVIES_INFO_CSV = CACHE_DIR / "movies_info_df.csv"

# 人员信息文件
STAFF_DF_CSV = CACHE_DIR / "staff_df.csv"

# 名称映射缓存 (首次加载后保存为 JSON，加速后续加载)
NAME_MAPPING_CACHE = Path(__file__).parent.parent / "cache" / "name_mapping.json"


# =============================================================================
# 名称映射加载
# =============================================================================

@st.cache_data(ttl=86400, show_spinner="加载名称映射...")
def load_name_mapping() -> Dict[str, str]:
    """
    加载 Token 到真实名称的映射
    
    映射关系:
    - MOV_ttXXXXXXX -> 电影名
    - ACT_nmXXXXXXX -> 演员名
    - DIR_nmXXXXXXX -> 导演名
    - GEN_XXX -> 类型名 (中文)
    - ERA_XXXX -> 年代名 (中文)
    - RAT_X.X -> 评分
    - TYP_XXX -> 作品类型 (中文)
    
    Returns:
        {token: display_name} 字典
    """
    # 尝试从缓存加载
    if NAME_MAPPING_CACHE.exists():
        try:
            with open(NAME_MAPPING_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    
    # 构建映射
    mapping = {}
    
    # 1. 加载电影名称
    if MOVIES_INFO_CSV.exists():
        try:
            movies_df = pd.read_csv(
                MOVIES_INFO_CSV,
                usecols=["tconst", "title"],
                dtype=str,
            )
            for _, row in movies_df.iterrows():
                tconst = row["tconst"]
                title = row["title"]
                if pd.notna(tconst) and pd.notna(title):
                    mapping[f"MOV_{tconst}"] = title
        except Exception as e:
            st.warning(f"加载电影名称失败: {e}")
    
    # 2. 加载人员名称 (演员、导演、编剧)
    if STAFF_DF_CSV.exists():
        try:
            staff_df = pd.read_csv(
                STAFF_DF_CSV,
                usecols=["nconst", "primaryName", "isDirectors", "isActors"],
                dtype={"nconst": str, "primaryName": str},
            )
            for _, row in staff_df.iterrows():
                nconst = row["nconst"]
                name = row["primaryName"]
                if pd.notna(nconst) and pd.notna(name):
                    # 演员
                    mapping[f"ACT_{nconst}"] = name
                    # 导演
                    mapping[f"DIR_{nconst}"] = name
                    # 通用人员
                    mapping[f"PER_{nconst}"] = name
        except Exception as e:
            st.warning(f"加载人员名称失败: {e}")
    
    # 3. 添加类型映射 (中文)
    genre_mapping = {
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
    mapping.update(genre_mapping)
    
    # 4. 添加年代映射
    era_mapping = {
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
    mapping.update(era_mapping)
    
    # 5. 添加作品类型映射
    type_mapping = {
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
    mapping.update(type_mapping)
    
    # 6. 评分保持原样显示
    for i in range(0, 101):
        rating = i / 10
        mapping[f"RAT_{rating}"] = f"⭐ {rating}分"
        mapping[f"RAT_{rating:.1f}"] = f"⭐ {rating:.1f}分"
    
    # 保存缓存
    try:
        NAME_MAPPING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(NAME_MAPPING_CACHE, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    
    return mapping


# =============================================================================
# 便捷函数
# =============================================================================

def get_display_name(token: str) -> str:
    """
    获取 Token 的显示名称
    
    Args:
        token: 内部 Token，如 "MOV_tt0111161"
        
    Returns:
        显示名称，如 "The Shawshank Redemption"；如果找不到映射，返回原 token
    """
    mapping = load_name_mapping()
    return mapping.get(token, token)


def token_to_display(token: str, show_token: bool = False) -> str:
    """
    将 Token 转换为显示文本
    
    Args:
        token: 内部 Token
        show_token: 是否在名称后显示原始 token
        
    Returns:
        格式化的显示文本
    """
    name = get_display_name(token)
    
    if show_token and name != token:
        return f"{name} ({token})"
    return name


def get_entity_display_info(token: str) -> Dict[str, str]:
    """
    获取实体的完整显示信息
    
    Args:
        token: Token 字符串
        
    Returns:
        包含 name, type, type_name, token 的字典
    """
    from .data_loader import get_entity_type
    
    entity_type = get_entity_type(token)
    display_name = get_display_name(token)
    type_name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
    
    return {
        "token": token,
        "name": display_name,
        "type": entity_type,
        "type_name": type_name,
    }


def format_entity_label(token: str, include_type: bool = True) -> str:
    """
    格式化实体标签（用于图表显示）
    
    Args:
        token: Token
        include_type: 是否包含类型标签
        
    Returns:
        格式化的标签，如 "[电影] 肖申克的救赎"
    """
    info = get_entity_display_info(token)
    
    if include_type:
        return f"[{info['type_name']}] {info['name']}"
    return info["name"]


def batch_get_display_names(tokens: list) -> Dict[str, str]:
    """
    批量获取显示名称
    
    Args:
        tokens: Token 列表
        
    Returns:
        {token: display_name} 字典
    """
    mapping = load_name_mapping()
    return {token: mapping.get(token, token) for token in tokens}


# =============================================================================
# 搜索功能增强
# =============================================================================

@st.cache_data(ttl=3600)
def build_reverse_mapping() -> Dict[str, str]:
    """
    构建反向映射：名称 -> Token（用于搜索）
    
    Returns:
        {lowercase_name: token} 字典
    """
    mapping = load_name_mapping()
    reverse = {}
    
    for token, name in mapping.items():
        # 使用小写作为键，方便搜索
        reverse[name.lower()] = token
    
    return reverse


def search_by_name(query: str, limit: int = 20) -> list:
    """
    通过名称搜索 Token
    
    支持按电影名、演员名等搜索，而不仅仅是 Token。
    
    Args:
        query: 搜索关键词
        limit: 返回数量上限
        
    Returns:
        匹配的 Token 列表
    """
    mapping = load_name_mapping()
    query_lower = query.lower()
    
    results = []
    for token, name in mapping.items():
        if query_lower in name.lower() or query_lower in token.lower():
            results.append(token)
            if len(results) >= limit:
                break
    
    return results

