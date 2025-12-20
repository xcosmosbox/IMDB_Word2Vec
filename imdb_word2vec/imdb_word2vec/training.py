from __future__ import annotations

import io
import multiprocessing as mp
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tqdm import tqdm

from .config import CONFIG
from .logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)


class TqdmProgressCallback(tf.keras.callbacks.Callback):
    """使用 tqdm 显示训练进度的回调。"""

    def __init__(self, epochs: int, desc: str = "Training"):
        super().__init__()
        self.epochs = epochs
        self.desc = desc
        self.pbar = None

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(total=self.epochs, desc=self.desc, unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", 0)
        acc = logs.get("accuracy", 0)
        pos_recall = logs.get("positive_recall", 0)
        self.pbar.set_postfix({
            "loss": f"{loss:.4f}",
            "acc": f"{acc:.4f}",
            "pos_recall": f"{pos_recall:.4f}"
        })
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        if self.pbar:
            self.pbar.close()


def positive_recall(y_true, y_pred):
    """
    正样本召回率：模型对正样本（第一个位置）的预测准确率。
    
    y_true: [batch, num_ns+1]，第一个位置是 1（正样本），其余是 0
    y_pred: [batch, num_ns+1]，logits
    
    正样本召回 = 正样本位置的 logit 是所有位置中最大的比例
    """
    # 获取预测中最大值的索引
    pred_max_idx = tf.argmax(y_pred, axis=-1)  # [batch]
    # 正样本在第 0 位
    correct = tf.cast(tf.equal(pred_max_idx, 0), tf.float32)
    return tf.reduce_mean(correct)


# ========== 辅助函数 ==========

def _prepare_negative_sampling_probs(freq: np.ndarray) -> np.ndarray:
    """基于 freq^0.75 构建负采样分布。"""
    freq_pow = np.power(freq, 0.75, dtype=np.float64)
    total = freq_pow.sum()
    if total == 0:
        return np.ones_like(freq_pow) / len(freq_pow)
    return freq_pow / total


def _sample_negatives(prob: np.ndarray, num_ns: int, rng: np.random.Generator) -> np.ndarray:
    """按给定分布采样负样本。"""
    indices = rng.choice(len(prob), size=num_ns, replace=True, p=prob)
    return indices.astype(np.int64)


def _compute_subsample_probs(freq: np.ndarray, t: float) -> np.ndarray:
    """计算高频子采样保留概率。"""
    total = freq.sum()
    probs = np.ones_like(freq, dtype=np.float64)
    mask = freq > 0
    freq_frac = freq[mask] / total
    probs[mask] = np.minimum(1.0, np.sqrt(t / freq_frac) + t / freq_frac)
    return probs


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
        self.target_embedding = layers.Embedding(vocab_size, embedding_dim, name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size, embedding_dim)

    def call(self, pair):
        target, context = pair
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = tf.einsum("be,bce->bc", word_emb, context_emb)
        return dots


# ========== 序列数据处理 ==========

def _process_sequences_chunk(args: Tuple) -> Tuple[List, List, List]:
    """
    并行处理序列块，生成 skip-gram 样本。
    
    Args:
        args: (sequences, window_size, num_ns, neg_prob, subsample_probs, rng_seed)
    
    Returns:
        (targets, contexts, labels)
    """
    sequences, window_size, num_ns, neg_prob, subsample_probs, rng_seed = args
    rng = np.random.default_rng(rng_seed)
    
    targets = []
    contexts = []
    labels = []
    
    for seq in sequences:
        # 过滤掉 padding (0) 和 UNK (1)
        valid_tokens = [t for t in seq if t > 1]
        
        # 可选：高频子采样
        if subsample_probs is not None:
            filtered = []
            for tok in valid_tokens:
                if tok < len(subsample_probs) and rng.random() < subsample_probs[tok]:
                    filtered.append(tok)
            valid_tokens = filtered
        
        seq_len = len(valid_tokens)
        if seq_len < 2:
            continue
        
        # 生成 skip-gram 样本
        for i, target_word in enumerate(valid_tokens):
            # 动态窗口大小
            actual_window = rng.integers(1, window_size + 1)
            left = max(0, i - actual_window)
            right = min(seq_len, i + actual_window + 1)
            
            for j in range(left, right):
                if j == i:
                    continue
                
                context_word = valid_tokens[j]
                
                # 负采样
                negatives = _sample_negatives(neg_prob, num_ns, rng)
                
                # 构造样本
                context = np.concatenate(([context_word], negatives)).astype(np.int64)
                label = np.concatenate(([1], np.zeros(num_ns, dtype=np.int64)))
                
                targets.append(np.int64(target_word))
                contexts.append(context)
                labels.append(label)
    
    return targets, contexts, labels


def _parallel_generate_samples(
    sequences: np.ndarray,
    window_size: int,
    num_ns: int,
    neg_prob: np.ndarray,
    subsample_probs: Optional[np.ndarray],
    rng_seed: int,
    n_workers: Optional[int] = None,
    chunk_size: int = 5000,
) -> Tuple[List, List, List]:
    """并行生成 skip-gram 样本。"""
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    n_seqs = len(sequences)
    n_chunks = (n_seqs + chunk_size - 1) // chunk_size
    
    # 构建任务
    tasks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_seqs)
        chunk_seqs = sequences[start:end]
        chunk_seed = rng_seed + i
        tasks.append((chunk_seqs, window_size, num_ns, neg_prob, subsample_probs, chunk_seed))
    
    logger.info("启动并行样本生成: %d 个序列块, %d 个工作进程", n_chunks, n_workers)
    
    all_targets = []
    all_contexts = []
    all_labels = []
    
    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(_process_sequences_chunk, tasks),
            total=n_chunks,
            desc=f"并行生成样本 ({n_workers} workers)",
            unit="chunk"
        ))
    
    for targets, contexts, labels in tqdm(results, desc="合并结果", unit="chunk"):
        all_targets.extend(targets)
        all_contexts.extend(contexts)
        all_labels.extend(labels)
    
    return all_targets, all_contexts, all_labels


# ========== 导出 ==========

def _export_vectors(model: Word2Vec, tokens: List[str], vectors_path: Path, metadata_path: Path) -> None:
    """导出词向量与元数据。"""
    weights = model.get_layer("w2v_embedding").get_weights()[0]
    vectors_path.parent.mkdir(parents=True, exist_ok=True)

    with io.open(vectors_path, "w", encoding="utf-8") as out_v, \
         io.open(metadata_path, "w", encoding="utf-8") as out_m:
        for idx, token in tqdm(enumerate(tokens), total=len(tokens), desc="导出词向量"):
            if idx == 0:
                continue  # 跳过 <PAD>
            if idx >= len(weights):
                break
            vec = weights[idx]
            out_v.write("\t".join([str(x) for x in vec]) + "\n")
            out_m.write(token + "\n")
    
    logger.info("词向量与元数据已导出: %s, %s", vectors_path, metadata_path)


def _load_vocab_tokens(vocab_path: Path, vocab_size: int) -> List[str]:
    """从词表文件加载 token 列表。"""
    vocab_df = pd.read_csv(vocab_path, header=None, names=["key", "value"])
    inv = vocab_df.set_index("value")["key"].to_dict()
    tokens = [inv.get(i, f"<ID_{i}>") for i in range(vocab_size)]
    return tokens


# ========== 主训练函数 ==========

def train_word2vec(
    sequences_path: Path,
    vocab_path: Path,
    vocab_limit: int,
    max_sequences: Optional[int] = None,
) -> Tuple[Path, Path]:
    """
    训练 Word2Vec 模型。
    
    Args:
        sequences_path: 序列数据 CSV 路径（每行是一个序列）
        vocab_path: 词汇表路径
        vocab_limit: 词表规模上限
        max_sequences: 可选的序列数上限
    
    Returns:
        (vectors_path, metadata_path)
    """
    logger.info("========== 开始 Word2Vec 训练 ==========")
    logger.info("读取序列文件: %s", sequences_path)
    
    # 读取序列数据
    df = pd.read_csv(sequences_path)
    sequences = df.values.astype(np.int64)
    
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    
    logger.info("序列数: %d, 序列长度: %d", len(sequences), sequences.shape[1])
    
    # 统计词频
    flat = sequences.reshape(-1)
    flat = flat[flat > 1]  # 排除 PAD 和 UNK
    vocab_size = int(flat.max()) + 1 if flat.size else vocab_limit
    vocab_size = min(vocab_size, vocab_limit)
    
    freq = np.bincount(flat, minlength=vocab_size)
    logger.info("词表规模: %d, 有效 token 数: %d", vocab_size, (freq > 0).sum())
    
    # 子采样概率
    subsample_probs = None
    if CONFIG.train.subsample_t > 0:
        subsample_probs = _compute_subsample_probs(freq, CONFIG.train.subsample_t)
    
    # 负采样分布
    neg_prob = _prepare_negative_sampling_probs(freq)
    
    # ========== 生成训练样本 ==========
    window_size = CONFIG.train.window_size
    num_ns = CONFIG.train.num_negative_samples
    batch_size = CONFIG.train.batch_size_word2vec
    
    logger.info("窗口大小: %d, 负样本数: %d", window_size, num_ns)
    
    # 并行生成样本
    all_targets, all_contexts, all_labels = _parallel_generate_samples(
        sequences=sequences,
        window_size=window_size,
        num_ns=num_ns,
        neg_prob=neg_prob,
        subsample_probs=subsample_probs,
        rng_seed=CONFIG.random_seed,
        n_workers=CONFIG.train.parallel_workers,
        chunk_size=CONFIG.train.parallel_chunk_size,
    )
    
    total_samples = len(all_targets)
    logger.info("总样本数: %d", total_samples)
    
    if total_samples == 0:
        raise ValueError("没有生成任何训练样本，请检查序列数据")
    
    # 转换为 numpy 数组
    targets_arr = np.array(all_targets, dtype=np.int64)
    contexts_arr = np.stack(all_contexts).astype(np.int64)
    labels_arr = np.stack(all_labels).astype(np.int64)
    
    # 释放列表内存
    del all_targets, all_contexts, all_labels
    
    # ========== 构建 Dataset ==========
    shuffle_buffer = CONFIG.train.shuffle_buffer_size
    steps_per_epoch = max(1, total_samples // batch_size)
    
    dataset = tf.data.Dataset.from_tensor_slices(((targets_arr, contexts_arr), labels_arr))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    logger.info("Dataset 构建完成, steps_per_epoch: %d, shuffle_buffer: %d", 
                steps_per_epoch, shuffle_buffer)
    
    # ========== 训练模型 ==========
    strategy = _get_strategy()
    logger.info("分布式策略: %s", strategy.__class__.__name__)
    
    epochs = 10
    
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
    
    # 回调
    lr_callback = ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )
    tqdm_callback = TqdmProgressCallback(epochs=epochs, desc="Word2Vec 训练")
    
    # 训练
    model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=0,
        callbacks=[tqdm_callback, lr_callback],
    )
    
    # ========== 导出词向量 ==========
    tokens = _load_vocab_tokens(vocab_path, vocab_size)
    vectors_path = CONFIG.paths.vectors_path
    metadata_path = CONFIG.paths.metadata_path
    _export_vectors(model, tokens, vectors_path, metadata_path)
    
    logger.info("========== Word2Vec 训练完成 ==========")
    return vectors_path, metadata_path
