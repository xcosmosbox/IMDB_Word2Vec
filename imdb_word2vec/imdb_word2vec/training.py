"""
Word2Vec 训练模块

支持两种训练模式：
1. 从预生成的正样本对训练（推荐，负样本动态采样）
2. 流式训练：边生成边训练（适合首次运行）

使用方法:
    # 预生成 + 训练（推荐）
    python -m imdb_word2vec.cli pretrain
    python -m imdb_word2vec.cli train --use-cache
    
    # 流式训练
    python -m imdb_word2vec.cli train
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tqdm import tqdm

from .config import CONFIG
from .logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)


# ========== 进度条回调 ==========

class TqdmProgressCallback(tf.keras.callbacks.Callback):
    """使用 tqdm 显示训练进度的回调。"""

    def __init__(self, epochs: int, desc: str = "训练中"):
        super().__init__()
        self.epochs = epochs
        self.desc = desc
        self.epoch_bar = None
        self.batch_bar = None

    def on_train_begin(self, logs=None):
        self.epoch_bar = tqdm(total=self.epochs, desc=self.desc, position=0)

    def on_epoch_begin(self, epoch, logs=None):
        steps = self.params.get("steps", 0)
        self.batch_bar = tqdm(total=steps, desc=f"Epoch {epoch+1}", position=1, leave=False)

    def on_batch_end(self, batch, logs=None):
        if self.batch_bar:
            self.batch_bar.update(1)
            if logs:
                self.batch_bar.set_postfix({
                    "loss": f"{logs.get('loss', 0):.4f}",
                    "acc": f"{logs.get('accuracy', 0):.4f}",
                })

    def on_epoch_end(self, epoch, logs=None):
        if self.batch_bar:
            self.batch_bar.close()
        if self.epoch_bar:
            self.epoch_bar.update(1)
            if logs:
                self.epoch_bar.set_postfix({
                    "loss": f"{logs.get('loss', 0):.4f}",
                    "acc": f"{logs.get('accuracy', 0):.4f}",
                })

    def on_train_end(self, logs=None):
        if self.batch_bar:
            self.batch_bar.close()
        if self.epoch_bar:
            self.epoch_bar.close()


# ========== 正样本召回率 ==========

def positive_recall(y_true, y_pred):
    """正样本召回率：正样本位置的 logit 是否是所有位置中最大的"""
    pred_max_idx = tf.argmax(y_pred, axis=-1)
    correct = tf.cast(tf.equal(pred_max_idx, 0), tf.float32)
    return tf.reduce_mean(correct)


# ========== 辅助函数 ==========

def _get_strategy() -> tf.distribute.Strategy:
    """按需创建分布式策略。"""
    if CONFIG.train.enable_distribute:
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 1:
            try:
                return tf.distribute.MirroredStrategy()
            except Exception:
                pass
    return tf.distribute.get_strategy()


# ========== Word2Vec 模型 ==========

class Word2Vec(tf.keras.Model):
    """Skip-gram 词向量模型。"""

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.target_embedding = layers.Embedding(vocab_size, embedding_dim, name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size, embedding_dim)

    def call(self, inputs):
        target, context = inputs
        target_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = tf.einsum("be,bce->bc", target_emb, context_emb)
        return dots
    
    def get_model_stats(self) -> dict:
        """获取模型参数统计信息。"""
        # 计算参数量
        target_params = self.vocab_size * self.embedding_dim
        context_params = self.vocab_size * self.embedding_dim
        total_params = target_params + context_params
        
        # 计算模型大小（float32，每个参数 4 bytes）
        model_size_bytes = total_params * 4
        model_size_mb = model_size_bytes / (1024 ** 2)
        
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "target_embedding_params": target_params,
            "context_embedding_params": context_params,
            "total_params": total_params,
            "model_size_mb": model_size_mb,
        }


def _log_model_stats(model: Word2Vec) -> None:
    """打印模型参数统计信息。"""
    stats = model.get_model_stats()
    total = stats["total_params"]
    
    # 格式化参数量
    if total >= 1e9:
        param_str = f"{total / 1e9:.2f}B"
    elif total >= 1e6:
        param_str = f"{total / 1e6:.2f}M"
    elif total >= 1e3:
        param_str = f"{total / 1e3:.2f}K"
    else:
        param_str = str(total)
    
    logger.info("========== 模型参数统计 ==========")
    logger.info("词表规模: %d", stats["vocab_size"])
    logger.info("嵌入维度: %d", stats["embedding_dim"])
    logger.info("Target Embedding: %s 参数", f"{stats['target_embedding_params']:,}")
    logger.info("Context Embedding: %s 参数", f"{stats['context_embedding_params']:,}")
    logger.info("总参数量: %s (%s)", f"{total:,}", param_str)
    logger.info("模型大小: %.2f MB (float32)", stats["model_size_mb"])


# ========== 负采样相关 ==========

def _prepare_negative_sampling_probs(freq: np.ndarray, power: float = 0.75) -> np.ndarray:
    """准备负采样的概率分布（3/4 次幂平滑）。"""
    powered = np.power(freq.astype(np.float64), power)
    total = powered.sum()
    return powered / (total + 1e-10)


def _compute_subsample_probs(freq: np.ndarray, t: float) -> np.ndarray:
    """计算高频子采样的保留概率。"""
    total = freq.sum()
    f = freq / (total + 1e-10)
    probs = np.sqrt(t / (f + 1e-10)) + t / (f + 1e-10)
    probs = np.clip(probs, 0, 1)
    probs[freq == 0] = 0
    return probs


# ========== 导出函数 ==========

def _export_all_formats(model: Word2Vec, tokens: List[str], output_dir: Path):
    """导出所有格式的词向量文件。"""
    from .export import export_all
    
    weights = model.target_embedding.get_weights()[0]
    export_all(weights, tokens, output_dir)


def _load_vocab_tokens(vocab_path: Path, vocab_size: int) -> List[str]:
    """从词表文件加载 token 列表。"""
    vocab_df = pd.read_csv(vocab_path, header=None, names=["key", "value"])
    inv = vocab_df.set_index("value")["key"].to_dict()
    tokens = [inv.get(i, f"<ID_{i}>") for i in range(vocab_size)]
    return tokens


# ========== 从磁盘加载正样本对 ==========

def _load_positive_pairs_from_disk(chunk_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """从磁盘加载一个正样本对文件。"""
    data = np.load(chunk_path)
    return data["targets"], data["contexts"]


def _add_negative_samples(
    targets: np.ndarray,
    contexts: np.ndarray,
    neg_prob: np.ndarray,
    num_ns: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为正样本对动态添加负样本。
    
    Args:
        targets: 目标词 ID 数组
        contexts: 正样本上下文词 ID 数组
        neg_prob: 负采样概率分布
        num_ns: 负样本数量
        rng: 随机数生成器
    
    Returns:
        (targets, contexts_with_neg, labels)
        contexts_with_neg: shape (n, 1 + num_ns)，第一列是正样本
        labels: shape (n, 1 + num_ns)，第一列是 1，其余是 0
    """
    n = len(targets)
    
    # 采样负样本
    negatives = rng.choice(len(neg_prob), size=(n, num_ns), p=neg_prob, replace=True)
    
    # 组合正样本和负样本
    contexts_with_neg = np.concatenate([contexts.reshape(-1, 1), negatives], axis=1).astype(np.int32)
    
    # 创建标签（第一列是 1，其余是 0）
    labels = np.zeros((n, 1 + num_ns), dtype=np.float32)
    labels[:, 0] = 1.0
    
    return targets, contexts_with_neg, labels


# ========== 从缓存训练 ==========

def train_word2vec_from_cache(
    samples_dir: Path,
    vocab_path: Path,
) -> Tuple[Path, Path]:
    """
    从预生成的正样本对目录训练 Word2Vec 模型。
    
    负样本在训练时动态生成，每个 epoch 使用不同的负样本。
    """
    logger.info("========== 从缓存样本训练 Word2Vec ==========")
    
    # 加载元数据
    meta_path = samples_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"找不到预生成样本: {meta_path}")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    vocab_size = meta["vocab_size"]
    num_chunks = meta["num_chunks"]
    total_pairs = meta["total_pairs"]
    
    logger.info("词表规模: %d", vocab_size)
    logger.info("数据块数: %d", num_chunks)
    logger.info("总正样本对: %d (%.1fM)", total_pairs, total_pairs / 1e6)
    
    # 加载词频分布
    freq_path = samples_dir / "freq.npy"
    if freq_path.exists():
        freq = np.load(freq_path)
    else:
        # 兼容旧版本
        freq = np.ones(vocab_size, dtype=np.float64)
    
    neg_prob = _prepare_negative_sampling_probs(freq)
    
    # ========== 初始化模型 ==========
    strategy = _get_strategy()
    logger.info("分布式策略: %s", strategy.__class__.__name__)
    
    batch_size = CONFIG.train.batch_size_word2vec
    num_ns = CONFIG.train.num_negative_samples
    
    logger.info("负样本数: %d (动态采样)", num_ns)
    
    with strategy.scope():
        model = Word2Vec(vocab_size=vocab_size, embedding_dim=CONFIG.train.embedding_dim)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        if mixed_precision.global_policy().compute_dtype == "float16":
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["accuracy", positive_recall]
        )
    
    # 打印模型参数统计
    _log_model_stats(model)
    
    # ========== 分块流式训练 ==========
    global_epochs = CONFIG.train.global_epochs
    epochs_per_chunk = CONFIG.train.epochs_per_chunk
    
    logger.info("全局轮数: %d, 每块训练轮数: %d", global_epochs, epochs_per_chunk)
    
    lr_callback = ReduceLROnPlateau(
        monitor="loss",
        factor=0.7,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )
    
    # 获取所有 chunk 文件
    chunk_files = sorted(samples_dir.glob("chunk_*.npz"))
    num_chunks = len(chunk_files)
    
    global_bar = tqdm(total=global_epochs * num_chunks, desc="训练总进度")
    rng = np.random.default_rng(CONFIG.random_seed)
    
    for global_epoch in range(global_epochs):
        logger.info("---------- 全局轮次 %d/%d ----------", global_epoch + 1, global_epochs)
        
        # 每轮打乱数据块顺序
        chunk_indices = np.random.permutation(num_chunks)
        
        for i, chunk_idx in enumerate(chunk_indices):
            chunk_path = chunk_files[chunk_idx]
            
            # 从磁盘加载正样本对
            targets, contexts = _load_positive_pairs_from_disk(chunk_path)
            chunk_pairs = len(targets)
            
            if chunk_pairs == 0:
                global_bar.update(1)
                continue
            
            # 动态添加负样本（每次都不同）
            targets, contexts_with_neg, labels = _add_negative_samples(
                targets, contexts, neg_prob, num_ns, rng
            )
            
            # 构建 Dataset
            steps_per_epoch = max(1, chunk_pairs // batch_size)
            shuffle_buffer = min(CONFIG.train.shuffle_buffer_size, chunk_pairs)
            
            dataset = tf.data.Dataset.from_tensor_slices(
                ((targets, contexts_with_neg), labels)
            )
            dataset = dataset.shuffle(shuffle_buffer)
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.repeat()
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            # 训练
            model.fit(
                dataset,
                epochs=epochs_per_chunk,
                steps_per_epoch=steps_per_epoch,
                verbose=0,
                callbacks=[lr_callback],
            )
            
            # 释放内存
            del targets, contexts, contexts_with_neg, labels, dataset
            gc.collect()
            
            # 更新进度
            current_loss = model.history.history.get("loss", [0])[-1] if model.history else 0
            current_acc = model.history.history.get("accuracy", [0])[-1] if model.history else 0
            global_bar.update(1)
            global_bar.set_postfix({
                "epoch": f"{global_epoch+1}/{global_epochs}",
                "chunk": f"{i+1}/{num_chunks}",
                "loss": f"{current_loss:.4f}",
                "acc": f"{current_acc:.4f}",
            })
    
    global_bar.close()
    
    # ========== 导出所有格式 ==========
    logger.info("========== 导出模型文件 ==========")
    tokens = _load_vocab_tokens(vocab_path, vocab_size)
    _export_all_formats(model, tokens, CONFIG.paths.artifacts_dir)
    
    logger.info("========== 训练完成 ==========")
    return CONFIG.paths.vectors_path, CONFIG.paths.metadata_path


# ========== 流式训练（边生成边训练） ==========

def _generate_samples_for_chunk(
    sequences: np.ndarray,
    window_size: int,
    num_ns: int,
    neg_prob: np.ndarray,
    subsample_probs: Optional[np.ndarray],
    rng_seed: int,
) -> Tuple[List, List, List]:
    """为一个序列块生成 skip-gram 样本（含负采样）。"""
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
                negatives = rng.choice(len(neg_prob), size=num_ns, p=neg_prob, replace=True)
                
                context = np.concatenate(([context_word], negatives)).astype(np.int32)
                label = np.concatenate(([1], np.zeros(num_ns, dtype=np.float32)))
                
                targets.append(np.int32(target_word))
                contexts.append(context)
                labels.append(label)
    
    return targets, contexts, labels


def train_word2vec(
    sequences_path: Path,
    vocab_path: Path,
    vocab_limit: int,
    max_sequences: Optional[int] = None,
    use_cache: bool = False,
) -> Tuple[Path, Path]:
    """
    训练 Word2Vec 模型。
    
    Args:
        sequences_path: 序列数据 CSV 路径
        vocab_path: 词汇表路径
        vocab_limit: 词表规模上限
        max_sequences: 可选的序列数上限
        use_cache: 是否使用预生成的样本缓存
    
    Returns:
        (vectors_path, metadata_path)
    """
    # 检查是否使用缓存
    samples_dir = CONFIG.paths.cache_dir / "samples"
    if use_cache or (samples_dir / "meta.json").exists():
        if (samples_dir / "meta.json").exists():
            logger.info("发现预生成样本缓存，使用缓存训练")
            return train_word2vec_from_cache(samples_dir, vocab_path)
        else:
            logger.warning("未找到预生成样本，切换到流式训练模式")
    
    logger.info("========== 开始 Word2Vec 流式训练 ==========")
    logger.info("读取序列文件: %s", sequences_path)
    
    # ========== 1. 读取序列数据 ==========
    df = pd.read_csv(sequences_path)
    sequences = df.values.astype(np.int32)
    del df
    gc.collect()
    
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    
    num_sequences = len(sequences)
    seq_len = sequences.shape[1]
    logger.info("序列数: %d, 序列长度: %d", num_sequences, seq_len)
    
    # ========== 2. 计算词频和采样分布 ==========
    flat = sequences.reshape(-1)
    flat = flat[flat > 1]
    vocab_size = int(flat.max()) + 1 if flat.size else vocab_limit
    vocab_size = min(vocab_size, vocab_limit)
    
    freq = np.bincount(flat, minlength=vocab_size)
    del flat
    gc.collect()
    
    logger.info("词表规模: %d, 有效 token 数: %d", vocab_size, (freq > 0).sum())
    
    subsample_probs = None
    if CONFIG.train.subsample_t > 0:
        subsample_probs = _compute_subsample_probs(freq, CONFIG.train.subsample_t)
    
    neg_prob = _prepare_negative_sampling_probs(freq)
    
    # ========== 3. 计算分块策略 ==========
    window_size = CONFIG.train.window_size
    num_ns = CONFIG.train.num_negative_samples
    batch_size = CONFIG.train.batch_size_word2vec
    max_memory_gb = CONFIG.train.max_memory_gb
    
    samples_per_seq = seq_len * window_size * 0.5
    bytes_per_sample = (1 + num_ns + 1) * 4 + (num_ns + 1) * 4
    memory_per_seq = samples_per_seq * bytes_per_sample
    
    seqs_per_chunk = int((max_memory_gb * 1024**3) / memory_per_seq)
    seqs_per_chunk = max(1000, min(seqs_per_chunk, num_sequences))
    
    num_chunks = (num_sequences + seqs_per_chunk - 1) // seqs_per_chunk
    
    logger.info("========== 流式训练配置 ==========")
    logger.info("目标内存限制: %.1f GB", max_memory_gb)
    logger.info("每块序列数: %d", seqs_per_chunk)
    logger.info("数据块总数: %d", num_chunks)
    logger.info("负样本数: %d", num_ns)
    
    # ========== 4. 初始化模型 ==========
    strategy = _get_strategy()
    logger.info("分布式策略: %s", strategy.__class__.__name__)
    
    with strategy.scope():
        model = Word2Vec(vocab_size=vocab_size, embedding_dim=CONFIG.train.embedding_dim)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        if mixed_precision.global_policy().compute_dtype == "float16":
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["accuracy", positive_recall]
        )
    
    # 打印模型参数统计
    _log_model_stats(model)
    
    # ========== 5. 分块流式训练 ==========
    global_epochs = CONFIG.train.global_epochs
    epochs_per_chunk = CONFIG.train.epochs_per_chunk
    
    logger.info("全局轮数: %d, 每块训练轮数: %d", global_epochs, epochs_per_chunk)
    
    lr_callback = ReduceLROnPlateau(
        monitor="loss",
        factor=0.7,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )
    
    global_bar = tqdm(total=global_epochs * num_chunks, desc="流式训练总进度")
    
    for global_epoch in range(global_epochs):
        logger.info("---------- 全局轮次 %d/%d ----------", global_epoch + 1, global_epochs)
        
        chunk_indices = np.random.permutation(num_chunks)
        
        for chunk_idx in chunk_indices:
            start_idx = chunk_idx * seqs_per_chunk
            end_idx = min(start_idx + seqs_per_chunk, num_sequences)
            chunk_sequences = sequences[start_idx:end_idx]
            
            # 生成样本
            targets, contexts, labels = _generate_samples_for_chunk(
                chunk_sequences,
                window_size,
                num_ns,
                neg_prob,
                subsample_probs,
                rng_seed=CONFIG.random_seed + global_epoch * 1000 + chunk_idx,
            )
            
            if len(targets) == 0:
                global_bar.update(1)
                continue
            
            targets_arr = np.array(targets, dtype=np.int32)
            contexts_arr = np.stack(contexts).astype(np.int32)
            labels_arr = np.stack(labels).astype(np.float32)
            
            del targets, contexts, labels
            gc.collect()
            
            chunk_samples = len(targets_arr)
            steps_per_epoch = max(1, chunk_samples // batch_size)
            shuffle_buffer = min(CONFIG.train.shuffle_buffer_size, chunk_samples)
            
            dataset = tf.data.Dataset.from_tensor_slices(
                ((targets_arr, contexts_arr), labels_arr)
            )
            dataset = dataset.shuffle(shuffle_buffer)
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.repeat()
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            model.fit(
                dataset,
                epochs=epochs_per_chunk,
                steps_per_epoch=steps_per_epoch,
                verbose=0,
                callbacks=[lr_callback],
            )
            
            del targets_arr, contexts_arr, labels_arr, dataset
            gc.collect()
            
            current_loss = model.history.history.get("loss", [0])[-1] if model.history else 0
            current_acc = model.history.history.get("accuracy", [0])[-1] if model.history else 0
            global_bar.update(1)
            global_bar.set_postfix({
                "epoch": f"{global_epoch+1}/{global_epochs}",
                "loss": f"{current_loss:.4f}",
                "acc": f"{current_acc:.4f}",
            })
    
    global_bar.close()
    
    # ========== 6. 导出所有格式 ==========
    logger.info("========== 导出模型文件 ==========")
    tokens = _load_vocab_tokens(vocab_path, vocab_size)
    _export_all_formats(model, tokens, CONFIG.paths.artifacts_dir)
    
    logger.info("========== Word2Vec 流式训练完成 ==========")
    return CONFIG.paths.vectors_path, CONFIG.paths.metadata_path
