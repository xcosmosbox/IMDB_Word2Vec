"""
智能缓存管理器
==============

提供基于源文件哈希的智能缓存机制，实现：
- 首次计算后永久有效
- 源数据变化时自动重算
- 毫秒级缓存加载

使用方法:
    from utils.cache_manager import get_cached_or_compute, invalidate_cache
    
    # 自动缓存计算结果
    data = get_cached_or_compute(
        cache_name="my_data",
        source_files=[embeddings_path],
        compute_fn=lambda: expensive_computation()
    )
"""
import hashlib
import pickle
import json
import time
from pathlib import Path
from typing import Callable, List, Any, Optional, Union
from datetime import datetime

import streamlit as st


# =============================================================================
# 缓存目录配置
# =============================================================================

# 缓存根目录
CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# 降维坐标缓存目录
DIM_REDUCTION_CACHE_DIR = CACHE_DIR / "dim_reduction"
DIM_REDUCTION_CACHE_DIR.mkdir(exist_ok=True)

# KNN 索引缓存目录
KNN_CACHE_DIR = CACHE_DIR / "knn_index"
KNN_CACHE_DIR.mkdir(exist_ok=True)

# 名称映射缓存目录
NAME_MAPPING_CACHE_DIR = CACHE_DIR / "name_mapping"
NAME_MAPPING_CACHE_DIR.mkdir(exist_ok=True)

# 缓存元数据文件
CACHE_METADATA_FILE = CACHE_DIR / "cache_metadata.json"


# =============================================================================
# 哈希计算
# =============================================================================

def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """
    计算单个文件的 MD5 哈希值
    
    Args:
        file_path: 文件路径
        chunk_size: 读取块大小
        
    Returns:
        MD5 哈希字符串（前16位）
    """
    if not file_path.exists():
        return "not_found"
    
    hasher = hashlib.md5()
    
    # 对于大文件，只读取头部和尾部
    file_size = file_path.stat().st_size
    
    with open(file_path, "rb") as f:
        if file_size <= chunk_size * 2:
            # 小文件，读取全部
            hasher.update(f.read())
        else:
            # 大文件，读取头部 + 文件大小 + 尾部
            hasher.update(f.read(chunk_size))
            hasher.update(str(file_size).encode())
            f.seek(-chunk_size, 2)  # 从末尾往前
            hasher.update(f.read(chunk_size))
    
    return hasher.hexdigest()[:16]


def compute_files_hash(file_paths: List[Path]) -> str:
    """
    计算多个文件的组合哈希值
    
    Args:
        file_paths: 文件路径列表
        
    Returns:
        组合 MD5 哈希字符串（前16位）
    """
    combined_hash = hashlib.md5()
    
    for path in sorted(file_paths):  # 排序保证顺序一致
        file_hash = compute_file_hash(path)
        combined_hash.update(file_hash.encode())
    
    return combined_hash.hexdigest()[:16]


# =============================================================================
# 缓存读写
# =============================================================================

def save_pickle(file_path: Path, data: Any) -> bool:
    """
    保存数据到 pickle 文件
    
    Args:
        file_path: 保存路径
        data: 要保存的数据
        
    Returns:
        是否保存成功
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception as e:
        st.warning(f"缓存保存失败: {e}")
        return False


def load_pickle(file_path: Path) -> Optional[Any]:
    """
    从 pickle 文件加载数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的数据，失败返回 None
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"缓存加载失败: {e}")
        return None


def save_json(file_path: Path, data: Any) -> bool:
    """
    保存数据到 JSON 文件
    
    Args:
        file_path: 保存路径
        data: 要保存的数据（需要可 JSON 序列化）
        
    Returns:
        是否保存成功
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.warning(f"JSON 保存失败: {e}")
        return False


def load_json(file_path: Path) -> Optional[Any]:
    """
    从 JSON 文件加载数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的数据，失败返回 None
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return None


# =============================================================================
# 缓存元数据管理
# =============================================================================

def load_cache_metadata() -> dict:
    """加载缓存元数据"""
    if CACHE_METADATA_FILE.exists():
        data = load_json(CACHE_METADATA_FILE)
        return data if data else {}
    return {}


def save_cache_metadata(metadata: dict) -> None:
    """保存缓存元数据"""
    save_json(CACHE_METADATA_FILE, metadata)


def update_cache_metadata(cache_name: str, source_hash: str, cache_path: str) -> None:
    """更新单个缓存的元数据"""
    metadata = load_cache_metadata()
    metadata[cache_name] = {
        "source_hash": source_hash,
        "cache_path": cache_path,
        "created_at": datetime.now().isoformat(),
    }
    save_cache_metadata(metadata)


# =============================================================================
# 核心缓存函数
# =============================================================================

def get_cached_or_compute(
    cache_name: str,
    source_files: List[Union[str, Path]],
    compute_fn: Callable[[], Any],
    cache_dir: Optional[Path] = None,
    use_pickle: bool = True,
    show_spinner: bool = True,
    spinner_text: Optional[str] = None,
) -> Any:
    """
    智能缓存：如果缓存有效则加载，否则计算并保存
    
    工作流程:
    1. 计算源文件的哈希值作为版本号
    2. 检查是否存在对应版本的缓存
    3. 存在且版本匹配 → 直接加载（毫秒级）
    4. 不存在或版本不匹配 → 重新计算并保存
    
    Args:
        cache_name: 缓存名称（用于文件命名）
        source_files: 源数据文件列表（用于计算哈希）
        compute_fn: 计算函数（无参数，返回要缓存的数据）
        cache_dir: 缓存目录，默认使用 CACHE_DIR
        use_pickle: 是否使用 pickle（True）或 JSON（False）
        show_spinner: 是否显示加载指示器
        spinner_text: 自定义加载提示文本
        
    Returns:
        缓存的数据或新计算的数据
        
    Example:
        data = get_cached_or_compute(
            cache_name="normalized_embeddings",
            source_files=[embeddings_path],
            compute_fn=lambda: normalize_vectors(embeddings),
        )
    """
    # 转换路径
    source_paths = [Path(f) for f in source_files]
    
    # 计算源文件哈希
    source_hash = compute_files_hash(source_paths)
    
    # 确定缓存目录和文件
    target_dir = cache_dir or CACHE_DIR
    ext = ".pkl" if use_pickle else ".json"
    cache_path = target_dir / f"{cache_name}_{source_hash[:8]}{ext}"
    
    # 检查缓存是否存在
    if cache_path.exists():
        # 缓存命中
        if show_spinner:
            with st.spinner(f"加载缓存: {cache_name}..."):
                data = load_pickle(cache_path) if use_pickle else load_json(cache_path)
        else:
            data = load_pickle(cache_path) if use_pickle else load_json(cache_path)
        
        if data is not None:
            return data
    
    # 缓存未命中，需要计算
    default_spinner = f"计算中: {cache_name}..."
    spinner_msg = spinner_text or default_spinner
    
    if show_spinner:
        with st.spinner(spinner_msg):
            start_time = time.time()
            data = compute_fn()
            elapsed = time.time() - start_time
    else:
        start_time = time.time()
        data = compute_fn()
        elapsed = time.time() - start_time
    
    # 保存缓存
    if use_pickle:
        save_pickle(cache_path, data)
    else:
        save_json(cache_path, data)
    
    # 更新元数据
    update_cache_metadata(cache_name, source_hash, str(cache_path))
    
    return data


# =============================================================================
# 缓存管理函数
# =============================================================================

def invalidate_cache(cache_name: Optional[str] = None) -> int:
    """
    清除缓存
    
    Args:
        cache_name: 要清除的缓存名称，None 表示清除所有
        
    Returns:
        清除的文件数量
    """
    count = 0
    
    if cache_name:
        # 清除指定缓存
        for cache_file in CACHE_DIR.rglob(f"{cache_name}_*"):
            cache_file.unlink()
            count += 1
    else:
        # 清除所有缓存
        for cache_file in CACHE_DIR.rglob("*.pkl"):
            cache_file.unlink()
            count += 1
        for cache_file in CACHE_DIR.rglob("*.json"):
            if cache_file.name != "cache_metadata.json":
                cache_file.unlink()
                count += 1
    
    return count


def get_cache_info() -> dict:
    """
    获取缓存统计信息
    
    Returns:
        包含缓存大小、文件数量等信息的字典
    """
    total_size = 0
    file_count = 0
    cache_files = []
    
    for cache_file in CACHE_DIR.rglob("*"):
        if cache_file.is_file():
            size = cache_file.stat().st_size
            total_size += size
            file_count += 1
            cache_files.append({
                "name": cache_file.name,
                "size_mb": round(size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(cache_file.stat().st_mtime).isoformat(),
            })
    
    return {
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "file_count": file_count,
        "files": sorted(cache_files, key=lambda x: x["name"]),
    }


# =============================================================================
# Streamlit 缓存装饰器增强
# =============================================================================

def cached_with_file(
    cache_name: str,
    source_files: List[Union[str, Path]],
    cache_dir: Optional[Path] = None,
):
    """
    装饰器版本的智能缓存
    
    Example:
        @cached_with_file("my_data", [embeddings_path])
        def compute_something():
            return expensive_computation()
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return get_cached_or_compute(
                cache_name=cache_name,
                source_files=source_files,
                compute_fn=lambda: func(*args, **kwargs),
                cache_dir=cache_dir,
            )
        return wrapper
    return decorator

