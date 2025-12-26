"""
预计算脚本
==========

运行此脚本以预计算所有降维坐标和索引。
首次运行可能需要 5-10 分钟，之后将永久缓存。

使用方法:
    cd web
    python scripts/run_precompute.py
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime

def main():
    print("=" * 60)
    print("IMDB Word2Vec 预计算脚本")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 导入预计算模块
    from utils.precompute import (
        get_normalized_embeddings,
        get_knn_index,
        get_tokens_list,
        precompute_all_dim_reductions,
    )
    
    # 1. 归一化嵌入
    print("\n[1/4] 计算归一化嵌入...")
    start = time.time()
    norm_emb = get_normalized_embeddings()
    print(f"      完成! 形状: {norm_emb.shape}, 耗时: {time.time() - start:.2f}s")
    
    # 2. KNN 索引
    print("\n[2/4] 构建 KNN 索引...")
    start = time.time()
    knn = get_knn_index(n_neighbors=50)
    print(f"      完成! 耗时: {time.time() - start:.2f}s")
    
    # 3. Token 列表
    print("\n[3/4] 构建 Token 列表...")
    start = time.time()
    tokens = get_tokens_list()
    print(f"      完成! 数量: {len(tokens)}, 耗时: {time.time() - start:.2f}s")
    
    # 4. 降维坐标
    print("\n[4/4] 预计算降维坐标...")
    
    def progress_callback(current, total, message):
        pct = current / total * 100
        print(f"      [{current}/{total}] ({pct:.0f}%) {message}")
    
    start = time.time()
    results = precompute_all_dim_reductions(progress_callback)
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    print(f"      完成! 成功: {success_count}/{total_count}, 耗时: {time.time() - start:.2f}s")
    
    # 汇总
    print("\n" + "=" * 60)
    print("预计算完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 显示缓存目录
    from utils.cache_manager import get_cache_info
    cache_info = get_cache_info()
    print(f"\n缓存统计:")
    print(f"  - 文件数量: {cache_info['file_count']}")
    print(f"  - 总大小: {cache_info['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()

