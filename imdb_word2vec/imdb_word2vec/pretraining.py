"""
预训练样本生成器

将全量序列数据转换为 Skip-gram 训练样本，分块保存到磁盘。
这样训练时可以直接从磁盘加载，避免重复生成。

特点：
- 并行生成：利用多核 CPU 加速
- 压缩存储：使用 npz 格式，节省 50%+ 磁盘空间
- 断点续传：已生成的块不会重复生成

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


def _prepare_negative_sampling_probs(freq: np.ndarray, power: float = 0.75) -> np.ndarray:
    """准备负采样的概率分布。"""
    powered = np.power(freq.astype(np.float64), power)
    total = powered.sum()
    return powered / (total + 1e-10)


def _sample_negatives(neg_prob: np.ndarray, num_ns: int, rng: np.random.Generator) -> np.ndarray:
    """采样负样本。"""
    return rng.choice(len(neg_prob), size=num_ns, p=neg_prob, replace=True)


def _generate_samples_for_chunk(
    sequences: np.ndarray,
    window_size: int,
    num_ns: int,
    neg_prob: np.ndarray,
    subsample_probs: Optional[np.ndarray],
    rng_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """为一个序列块生成 skip-gram 样本。"""
    rng = np.random.default_rng(rng_seed)
    
    targets = []
    contexts = []
    labels = []
    
    for seq in sequences:
        valid_tokens = [t for t in seq if t > 1]
        
        if subsample_probs is not None:
            filtered = []
            for tok in valid_tokens:
                if tok < len(subsample_probs) and rng.random() < subsample_probs[tok]:
                    filtered.append(tok)
            valid_tokens = filtered
        
        seq_len = len(valid_tokens)
        if seq_len < 2:
            continue
        
        for i, target_word in enumerate(valid_tokens):
            actual_window = rng.integers(1, window_size + 1)
            left = max(0, i - actual_window)
            right = min(seq_len, i + actual_window + 1)
            
            for j in range(left, right):
                if j == i:
                    continue
                
                context_word = valid_tokens[j]
                negatives = _sample_negatives(neg_prob, num_ns, rng)
                
                context = np.concatenate(([context_word], negatives)).astype(np.int32)
                label = np.concatenate(([1], np.zeros(num_ns, dtype=np.float32)))
                
                targets.append(np.int32(target_word))
                contexts.append(context)
                labels.append(label)
    
    if not targets:
        return np.array([], dtype=np.int32), np.array([]).reshape(0, num_ns + 1), np.array([]).reshape(0, num_ns + 1)
    
    return (
        np.array(targets, dtype=np.int32),
        np.stack(contexts).astype(np.int32),
        np.stack(labels).astype(np.float32),
    )


# ========== 并行 Worker ==========

def _worker_generate_and_save(args):
    """Worker: 生成一个块的样本并保存到磁盘。"""
    (chunk_idx, sequences, window_size, num_ns, neg_prob, 
     subsample_probs, rng_seed, output_dir, use_compression) = args
    
    # 生成样本
    targets, contexts, labels = _generate_samples_for_chunk(
        sequences, window_size, num_ns, neg_prob, subsample_probs, rng_seed
    )
    
    if len(targets) == 0:
        return chunk_idx, 0, 0
    
    # 保存到磁盘
    chunk_path = output_dir / f"chunk_{chunk_idx:04d}"
    chunk_path.mkdir(exist_ok=True)
    
    if use_compression:
        # 使用 npz 压缩格式，节省约 50% 空间
        np.savez_compressed(
            chunk_path / "samples.npz",
            targets=targets,
            contexts=contexts,
            labels=labels,
        )
        file_size = (chunk_path / "samples.npz").stat().st_size
    else:
        # 使用普通 npy 格式
        np.save(chunk_path / "targets.npy", targets)
        np.save(chunk_path / "contexts.npy", contexts)
        np.save(chunk_path / "labels.npy", labels)
        file_size = sum(f.stat().st_size for f in chunk_path.glob("*.npy"))
    
    return chunk_idx, len(targets), file_size


# ========== 主函数 ==========

def generate_training_samples(
    sequences_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_memory_gb: Optional[float] = None,
    force_regenerate: bool = False,
    use_compression: bool = True,
    n_workers: Optional[int] = None,
) -> Path:
    """
    生成并保存全部训练样本到磁盘（并行版）。
    
    Args:
        sequences_path: 序列数据路径，默认从 config 获取
        output_dir: 输出目录，默认为 cache/samples/
        max_memory_gb: 每个样本块的最大内存，默认从 config 获取
        force_regenerate: 是否强制重新生成（删除已有样本）
        use_compression: 是否使用压缩存储（推荐，节省 50%+ 空间）
        n_workers: 并行进程数，默认 CPU 核心数 - 1
    
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
        logger.info("块数: %d, 总样本: %d", meta["num_chunks"], meta["total_samples"])
        logger.info("磁盘占用: %.2f GB", meta.get("disk_usage_gb", 0))
        logger.info("使用 --force 参数可强制重新生成")
        return output_dir
    
    # 清空并创建目录
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("========== 开始预生成训练样本 ==========")
    logger.info("序列文件: %s", sequences_path)
    logger.info("输出目录: %s", output_dir)
    logger.info("并行进程: %d", n_workers)
    logger.info("压缩存储: %s", "是" if use_compression else "否")
    
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
    
    neg_prob = _prepare_negative_sampling_probs(freq)
    
    # 保存采样分布（训练时需要）
    np.save(output_dir / "neg_prob.npy", neg_prob)
    if subsample_probs is not None:
        np.save(output_dir / "subsample_probs.npy", subsample_probs)
    
    # ========== 3. 计算分块策略 ==========
    window_size = CONFIG.train.window_size
    num_ns = CONFIG.train.num_negative_samples
    
    # 估算每个序列产生的样本数和内存
    samples_per_seq = seq_len * window_size * 0.5
    bytes_per_sample = (1 + num_ns + 1) * 4 + (num_ns + 1) * 4
    memory_per_seq = samples_per_seq * bytes_per_sample
    
    seqs_per_chunk = int((max_memory_gb * 1024**3) / memory_per_seq)
    seqs_per_chunk = max(1000, min(seqs_per_chunk, num_sequences))
    
    num_chunks = (num_sequences + seqs_per_chunk - 1) // seqs_per_chunk
    
    logger.info("========== 分块策略 ==========")
    logger.info("每块序列数: %d", seqs_per_chunk)
    logger.info("数据块总数: %d", num_chunks)
    logger.info("预估每块内存: %.2f GB", (seqs_per_chunk * memory_per_seq) / (1024**3))
    
    # 预估磁盘占用
    estimated_samples = num_sequences * samples_per_seq
    estimated_disk_raw = (estimated_samples * bytes_per_sample) / (1024**3)
    estimated_disk = estimated_disk_raw * (0.4 if use_compression else 1.0)  # 压缩约 60%
    logger.info("预估总样本: %.0fM", estimated_samples / 1e6)
    logger.info("预估磁盘占用: %.1f GB %s", estimated_disk, "(压缩后)" if use_compression else "")
    
    # ========== 4. 准备并行任务 ==========
    tasks = []
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * seqs_per_chunk
        end_idx = min(start_idx + seqs_per_chunk, num_sequences)
        chunk_sequences = sequences[start_idx:end_idx].copy()  # 需要 copy 才能跨进程
        
        tasks.append((
            chunk_idx,
            chunk_sequences,
            window_size,
            num_ns,
            neg_prob,
            subsample_probs,
            CONFIG.random_seed + chunk_idx,
            output_dir,
            use_compression,
        ))
    
    # ========== 5. 并行生成 ==========
    logger.info("========== 开始并行生成 (%d workers) ==========", n_workers)
    
    total_samples = 0
    total_size = 0
    chunk_info = []
    
    # 使用进程池并行处理
    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_worker_generate_and_save, tasks),
            total=num_chunks,
            desc="并行生成样本块",
        ))
    
    # 收集结果
    for chunk_idx, num_samples, file_size in sorted(results, key=lambda x: x[0]):
        if num_samples > 0:
            chunk_info.append({
                "chunk_idx": chunk_idx,
                "num_samples": num_samples,
                "file_size_mb": file_size / (1024**2),
                "path": str(output_dir / f"chunk_{chunk_idx:04d}"),
            })
            total_samples += num_samples
            total_size += file_size
    
    # ========== 6. 保存元数据 ==========
    disk_usage_gb = total_size / (1024**3)
    
    meta = {
        "sequences_path": str(sequences_path),
        "num_sequences": num_sequences,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "window_size": window_size,
        "num_ns": num_ns,
        "num_chunks": len(chunk_info),
        "total_samples": total_samples,
        "disk_usage_gb": disk_usage_gb,
        "use_compression": use_compression,
        "chunks": chunk_info,
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    # ========== 7. 汇总报告 ==========
    avg_chunk_size = disk_usage_gb / len(chunk_info) if chunk_info else 0
    avg_samples_per_chunk = total_samples / len(chunk_info) if chunk_info else 0
    
    logger.info("========== 预生成完成 ==========")
    logger.info("总样本数: %d (%.1fM)", total_samples, total_samples / 1e6)
    logger.info("有效块数: %d", len(chunk_info))
    logger.info("总磁盘占用: %.2f GB", disk_usage_gb)
    logger.info("平均每块大小: %.2f MB", avg_chunk_size * 1024)
    logger.info("平均每块样本: %.0f", avg_samples_per_chunk)
    logger.info("元数据: %s", meta_path)
    
    return output_dir
