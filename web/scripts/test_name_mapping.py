#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试名称映射加载速度
"""
import time
import sys
from pathlib import Path

# 添加 web 目录到路径
WEB_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(WEB_DIR))

# 模拟 Streamlit 环境
class MockStreamlit:
    def cache_data(self, *args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    
    def spinner(self, *args, **kwargs):
        class Context:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        return Context()
    
    def warning(self, msg):
        print(f"Warning: {msg}")


# 替换 streamlit
import streamlit
streamlit.cache_data = MockStreamlit().cache_data
streamlit.spinner = MockStreamlit().spinner
streamlit.warning = MockStreamlit().warning


def main():
    print("=" * 60)
    print("Name Mapping Load Speed Test")
    print("=" * 60)
    
    # 第一次加载（可能需要构建缓存）
    print("\n[1] First load (may build cache)...")
    start = time.time()
    
    from utils.name_mapping import load_name_mapping
    mapping = load_name_mapping()
    
    elapsed = time.time() - start
    print(f"    Time: {elapsed:.2f} seconds")
    print(f"    Entries: {len(mapping):,}")
    
    # 检查缓存文件
    from utils.cache_manager import CACHE_DIR
    name_cache_dir = CACHE_DIR / "name_mapping"
    cache_files = list(name_cache_dir.glob("*.pkl"))
    print(f"    Cache files: {len(cache_files)}")
    for f in cache_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"      - {f.name}: {size_mb:.2f} MB")
    
    # 第二次加载（从 Pickle 缓存）
    print("\n[2] Second load (from Pickle cache)...")
    
    # 清除 Streamlit 的内存缓存，模拟重新启动
    from utils import name_mapping
    import importlib
    importlib.reload(name_mapping)
    
    start = time.time()
    mapping2 = name_mapping.load_name_mapping()
    elapsed2 = time.time() - start
    
    print(f"    Time: {elapsed2:.2f} seconds")
    print(f"    Entries: {len(mapping2):,}")
    
    # 对比
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"First load:  {elapsed:.2f}s")
    print(f"Cached load: {elapsed2:.2f}s")
    if elapsed2 > 0:
        print(f"Speedup:     {elapsed/elapsed2:.1f}x")
    
    # 测试查询性能（直接使用已加载的字典）
    print("\n[3] Query performance test...")
    
    test_tokens = list(mapping2.keys())[:1000]
    
    # 方法1：直接字典查询（最快）
    start = time.time()
    for token in test_tokens:
        _ = mapping2.get(token, token)
    elapsed_dict = time.time() - start
    print(f"    Direct dict lookup (1000x): {elapsed_dict*1000:.2f} ms")
    
    # 方法2：通过函数调用（有函数调用开销）
    from utils.name_mapping import get_display_name
    # 预热（确保映射已加载）
    _ = get_display_name(test_tokens[0])
    
    start = time.time()
    for token in test_tokens:
        _ = get_display_name(token)
    elapsed_func = time.time() - start
    print(f"    get_display_name (1000x): {elapsed_func*1000:.2f} ms")
    print(f"    Per query: {elapsed_func/1000*1000000:.2f} us")


if __name__ == "__main__":
    main()

