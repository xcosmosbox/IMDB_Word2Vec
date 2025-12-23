"""
预训练样本生成器（精简版）

只存储正样本对 (target_id, context_id)，不预生成负样本。
负样本在训练时动态采样，节省 99% 磁盘空间。

存储格式: npz 压缩文件，每个文件包含:
  - targets: int32 数组
  - contexts: int32 数组（一一对应）

使用方法:
    python -m imdb_word2vec.cli pretrain
"""
from __future__ import annotations

import gc
import json
import multiprocessing as mp
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CONFIG
from .logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)


# ========== 样本生成函数 ==========

def _compute_subsample_probs(freq: np.ndarray, t: float) -> np.ndarray:
    """计算高频子采样的保留概率。"""
    total = freq.sum()
    f = freq / (total + 1e-10)
    probs = np.sqrt(t / (f + 1e-10)) + t / (f + 1e-10)
    probs = np.clip(probs, 0, 1)
    probs[freq == 0] = 0
    return probs


def _generate_positive_pairs_for_chunk(
    sequences: np.ndarray,
    window_size: int,
    subsample_probs: Optional[np.ndarray],
    rng_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    为一个序列块生成正样本对（只存储 target 和 context，不存负样本）。
    
    Returns:
        (targets, contexts) - 两个 int32 数组，一一对应
    """
    rng = np.random.default_rng(rng_seed)
    
    targets = []
    contexts = []
    
    for seq in sequences:
        # 过滤 PAD 和 UNK
        valid_tokens = [t for t in seq if t > 1]
        
        # 高频子采样
        if subsample_probs is not None:
            filtered = []
            for tok in valid_tokens:
                if tok < len(subsample_probs) and rng.random() < subsample_probs[tok]:
                    filtered.append(tok)
            valid_tokens = filtered
        
        seq_len = len(valid_tokens)
        if seq_len < 2:
            continue
        
        # 生成 skip-gram 正样本对
        for i, target_word in enumerate(valid_tokens):
            # 动态窗口大小
            actual_window = rng.integers(1, window_size + 1)
            left = max(0, i - actual_window)
            right = min(seq_len, i + actual_window + 1)
            
            for j in range(left, right):
                if j == i:
                    continue
                
                context_word = valid_tokens[j]
                targets.append(target_word)
                contexts.append(context_word)
    
    if not targets:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    
    return np.array(targets, dtype=np.int32), np.array(contexts, dtype=np.int32)


# ========== 并行 Worker ==========

def _worker_generate_and_save(args):
    """Worker: 生成一个块的正样本对并保存到磁盘。"""
    (chunk_idx, sequences, window_size, subsample_probs, rng_seed, output_dir) = args
    
    # 生成正样本对
    targets, contexts = _generate_positive_pairs_for_chunk(
        sequences, window_size, subsample_probs, rng_seed
    )
    
    if len(targets) == 0:
        return chunk_idx, 0, 0
    
    # 保存到磁盘（压缩格式）
    chunk_path = output_dir / f"chunk_{chunk_idx:04d}.npz"
    np.savez_compressed(chunk_path, targets=targets, contexts=contexts)
    
    file_size = chunk_path.stat().st_size
    return chunk_idx, len(targets), file_size


# ========== 主函数 ==========

def generate_training_samples(
    sequences_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_memory_gb: Optional[float] = None,
    force_regenerate: bool = False,
    n_workers: Optional[int] = None,
) -> Path:
    """
    生成并保存正样本对到磁盘（精简版）。
    
    只存储 (target, context) 正样本对，负样本在训练时动态生成。
    相比存储完整样本，节省 99% 磁盘空间。
    
    Args:
        sequences_path: 序列数据路径
        output_dir: 输出目录
        max_memory_gb: 每个块的最大内存
        force_regenerate: 是否强制重新生成
        n_workers: 并行进程数
    
    Returns:
        样本目录路径
    """
    if sequences_path is None:
        sequences_path = CONFIG.paths.final_mapped_path
    if output_dir is None:
        output_dir = CONFIG.paths.cache_dir / "samples"
    if max_memory_gb is None:
        max_memory_gb = CONFIG.train.max_memory_gb
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    # 检查是否已有样本
    meta_path = output_dir / "meta.json"
    if meta_path.exists() and not force_regenerate:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        logger.info("========== 发现已有预生成样本 ==========")
        logger.info("块数: %d, 总正样本对: %d", meta["num_chunks"], meta["total_pairs"])
        logger.info("磁盘占用: %.2f MB", meta.get("disk_usage_mb", 0))
        logger.info("使用 --force 参数可强制重新生成")
        return output_dir
    
    # 清空并创建目录
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("========== 开始预生成正样本对 ==========")
    logger.info("序列文件: %s", sequences_path)
    logger.info("输出目录: %s", output_dir)
    logger.info("并行进程: %d", n_workers)
    logger.info("模式: 只存储正样本对（负样本训练时动态生成）")
    
    # ========== 1. 读取序列数据 ==========
    df = pd.read_csv(sequences_path)
    sequences = df.values.astype(np.int32)
    del df
    gc.collect()
    
    num_sequences = len(sequences)
    seq_len = sequences.shape[1]
    logger.info("序列数: %d, 序列长度: %d", num_sequences, seq_len)
    
    # ========== 2. 计算词频和采样分布 ==========
    flat = sequences.reshape(-1)
    flat = flat[flat > 1]
    vocab_size = int(flat.max()) + 1
    vocab_size = min(vocab_size, CONFIG.train.vocab_limit)
    
    freq = np.bincount(flat, minlength=vocab_size)
    del flat
    gc.collect()
    
    logger.info("词表规模: %d", vocab_size)
    
    subsample_probs = None
    if CONFIG.train.subsample_t > 0:
        subsample_probs = _compute_subsample_probs(freq, CONFIG.train.subsample_t)
    
    # 保存词频分布（训练时用于负采样）
    np.save(output_dir / "freq.npy", freq)
    
    # ========== 3. 计算分块策略 ==========
    window_size = CONFIG.train.window_size
    
    # 估算每个序列产生的正样本对数
    pairs_per_seq = seq_len * window_size * 0.5
    bytes_per_pair = 8  # 2 × int32
    memory_per_seq = pairs_per_seq * bytes_per_pair
    
    seqs_per_chunk = int((max_memory_gb * 1024**3) / memory_per_seq)
    seqs_per_chunk = max(10000, min(seqs_per_chunk, num_sequences))
    
    num_chunks = (num_sequences + seqs_per_chunk - 1) // seqs_per_chunk
    
    logger.info("========== 分块策略 ==========")
    logger.info("每块序列数: %d", seqs_per_chunk)
    logger.info("数据块总数: %d", num_chunks)
    
    # 预估磁盘占用
    estimated_pairs = num_sequences * pairs_per_seq
    estimated_disk_mb = (estimated_pairs * bytes_per_pair * 0.5) / (1024**2)  # 压缩率约 50%
    logger.info("预估总正样本对: %.0fM", estimated_pairs / 1e6)
    logger.info("预估磁盘占用: %.0f MB (压缩后)", estimated_disk_mb)
    
    # ========== 4. 准备并行任务 ==========
    tasks = []
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * seqs_per_chunk
        end_idx = min(start_idx + seqs_per_chunk, num_sequences)
        chunk_sequences = sequences[start_idx:end_idx].copy()
        
        tasks.append((
            chunk_idx,
            chunk_sequences,
            window_size,
            subsample_probs,
            CONFIG.random_seed + chunk_idx,
            output_dir,
        ))
    
    # ========== 5. 并行生成 ==========
    logger.info("========== 开始并行生成 (%d workers) ==========", n_workers)
    
    total_pairs = 0
    total_size = 0
    chunk_info = []
    
    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_worker_generate_and_save, tasks),
            total=num_chunks,
            desc="生成正样本对",
        ))
    
    # 收集结果
    for chunk_idx, num_pairs, file_size in sorted(results, key=lambda x: x[0]):
        if num_pairs > 0:
            chunk_info.append({
                "chunk_idx": chunk_idx,
                "num_pairs": num_pairs,
                "file_size_kb": file_size / 1024,
            })
            total_pairs += num_pairs
            total_size += file_size
    
    # ========== 6. 保存元数据 ==========
    disk_usage_mb = total_size / (1024**2)
    
    meta = {
        "sequences_path": str(sequences_path),
        "num_sequences": num_sequences,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "window_size": window_size,
        "num_chunks": len(chunk_info),
        "total_pairs": total_pairs,
        "disk_usage_mb": disk_usage_mb,
        "format": "positive_pairs_only",  # 标记格式
        "chunks": chunk_info,
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    # ========== 7. 汇总报告 ==========
    logger.info("========== 预生成完成 ==========")
    logger.info("总正样本对: %d (%.1fM)", total_pairs, total_pairs / 1e6)
    logger.info("有效块数: %d", len(chunk_info))
    logger.info("总磁盘占用: %.2f MB", disk_usage_mb)
    logger.info("平均每块: %.2f MB", disk_usage_mb / len(chunk_info) if chunk_info else 0)
    
    # 对比节省空间
    if CONFIG.train.num_negative_samples > 0:
        old_size_mb = (total_pairs * (1 + CONFIG.train.num_negative_samples) * 4 * 3) / (1024**2) * 0.4
        savings = (1 - disk_usage_mb / old_size_mb) * 100 if old_size_mb > 0 else 0
        logger.info("对比完整样本存储: 节省 %.1f%%", savings)
    
    return output_dir
