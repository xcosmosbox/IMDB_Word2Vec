from __future__ import annotations

import io
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
        self.pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{acc:.4f}"})
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        if self.pbar:
            self.pbar.close()


def _build_token_stats(data: np.ndarray) -> np.ndarray:
    """统计 token 频次（按整数 id），用于低频裁剪与负采样分布。"""
    flat = data.reshape(-1)
    max_id = int(flat.max()) if flat.size else 0
    freq = np.bincount(flat, minlength=max_id + 1)
    return freq


def _filter_sequence(
    seq: np.ndarray,
    keep_tokens: np.ndarray,
    subsample_probs: Optional[np.ndarray],
    rng: np.random.Generator,
) -> List[int]:
    """对单条序列进行低频裁剪与高频子采样，返回保留的 token 列表。"""
    out: List[int] = []
    for tok in seq:
        if tok < 0:
            continue  # 已被丢弃的 token
        if tok >= keep_tokens.size or not keep_tokens[tok]:
            continue  # 低频或被截断的 token 直接丢弃
        if subsample_probs is not None and subsample_probs[tok] < 1.0:
            if rng.random() > subsample_probs[tok]:
                continue  # 高频子采样丢弃
        out.append(int(tok))
    return out


def _compute_subsample_probs(freq: np.ndarray, t: float) -> np.ndarray:
    """计算高频子采样保留概率，t 越小丢弃越多。"""
    total = freq.sum()
    probs = np.ones_like(freq, dtype=np.float64)
    mask = freq > 0
    freq_frac = freq[mask] / total
    probs[mask] = np.minimum(1.0, np.sqrt(t / freq_frac) + t / freq_frac)
    return probs


def _prepare_negative_sampling_probs(freq: np.ndarray) -> np.ndarray:
    """基于 freq^0.75 构建负采样分布。"""
    freq_pow = np.power(freq, 0.75, dtype=np.float64)
    total = freq_pow.sum()
    if total == 0:
        return np.ones_like(freq_pow) / len(freq_pow)  # 退化情况，均匀分布
    return freq_pow / total


def _sample_negatives(prob: np.ndarray, num_ns: int, rng: np.random.Generator) -> np.ndarray:
    """按给定分布采样负样本。"""
    indices = rng.choice(len(prob), size=num_ns, replace=True, p=prob)
    return indices.astype(np.int64)


def _skipgram_generator(
    data: np.ndarray,
    window_size: int,
    num_ns: int,
    max_sequences: Optional[int],
    seq_chunk_size: int,
    keep_tokens: np.ndarray,
    subsample_probs: Optional[np.ndarray],
    neg_sampling_prob: np.ndarray,
    rng_seed: int,
) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]:
    """流式生成 (target, context_with_negatives, labels) 样本，避免一次性占用大量内存。"""
    rng = np.random.default_rng(rng_seed)
    n_rows = data.shape[0] if max_sequences is None else min(max_sequences, data.shape[0])
    for start_row in range(0, n_rows, seq_chunk_size):
        end_row = min(start_row + seq_chunk_size, n_rows)
        chunk = data[start_row:end_row]
        for seq in chunk:
            filtered = _filter_sequence(seq, keep_tokens, subsample_probs, rng)
            seq_len = len(filtered)
            if seq_len == 0:
                continue
            for i, target_word in enumerate(filtered):
                left = max(i - window_size, 0)
                right = min(i + window_size + 1, seq_len)
                for j in range(left, right):
                    if j == i:
                        continue
                    context_word = filtered[j]
                    negatives = _sample_negatives(neg_sampling_prob, num_ns, rng)
                    context = np.concatenate(([context_word], negatives)).astype(np.int64)
                # 标签：正样本 1，其余负样本 0
                label = np.concatenate(([1], np.zeros(num_ns, dtype=np.int64)))
                yield (np.int64(target_word), context), label


def _get_strategy() -> tf.distribute.Strategy:
    """按需创建分布式策略，默认使用单设备策略。"""
    if CONFIG.train.enable_distribute:
        gpus = tf.config.list_logical_devices("GPU")
        if len(gpus) >= 1:
            try:
                return tf.distribute.MirroredStrategy()
            except Exception:
                pass
    return tf.distribute.get_strategy()


class Word2Vec(tf.keras.Model):
    """经典的 Skip-gram 词向量模型。"""

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


def _export_vectors(model: Word2Vec, tokens: List[str], vectors_path: Path, metadata_path: Path) -> None:
    """导出词向量与元数据，便于可视化与检索。"""
    weights = model.get_layer("w2v_embedding").get_weights()[0]
    vectors_path.parent.mkdir(parents=True, exist_ok=True)

    with io.open(vectors_path, "w", encoding="utf-8") as out_v, io.open(metadata_path, "w", encoding="utf-8") as out_m:
        for idx, token in tqdm(enumerate(tokens), total=len(tokens), desc="导出词向量"):
            if idx == 0:
                continue  # 0 作为填充位，跳过
            vec = weights[idx]
            out_v.write("\t".join([str(x) for x in vec]) + "\n")
            out_m.write(token + "\n")
    logger.info("词向量与元数据已导出：%s, %s", vectors_path, metadata_path)


def _load_vocab_tokens(vocab_path: Path, vocab_size: int, kept_old_ids: Optional[np.ndarray] = None) -> List[str]:
    """根据 vocab 文件构造按索引排序的 token 列表。

    Args:
        vocab_path: 词表 CSV 路径。
        vocab_size: 需要的 token 数。
        kept_old_ids: 若进行了压缩映射，提供原始 id 列表，以保证导出顺序与新 id 对齐。
    """
    vocab_df = pd.read_csv(vocab_path, header=None, names=["key", "value"])
    inv = vocab_df.set_index("value")["key"].to_dict()
    if kept_old_ids is not None:
        tokens = [inv.get(int(old_id), str(old_id)) for old_id in kept_old_ids[:vocab_size]]
    else:
        tokens = [inv.get(i, str(i)) for i in range(vocab_size)]
    return tokens


def train_word2vec(
    mapped_features_path: Path,
    vocab_path: Path,
    vocab_limit: int,
    max_sequences: Optional[int] = None,
) -> Tuple[Path, Path]:
    """使用向量化特征训练 Word2Vec，返回向量与元数据路径。

    Args:
        mapped_features_path: 向量化后的 CSV 路径（如 final_mapped_vec.csv）。
        vocab_path: 词表 CSV 路径。
        vocab_limit: 词表规模上限。
        max_sequences: 可选的序列数上限，用于小样本验证。
    """
    logger.info("读取特征文件: %s", mapped_features_path)
    mapped_df = pd.read_csv(mapped_features_path)
    data = mapped_df.values.astype(np.int64)

    # 仅截取 max_sequences（若指定）
    if max_sequences is not None:
        data = data[:max_sequences]

    # 统计频次，用于低频裁剪、词表截断、负采样分布与子采样
    freq = _build_token_stats(data)

    # 根据频次截断词表（低频裁剪 + top-K）
    vocab_limit = CONFIG.train.vocab_limit
    min_freq = CONFIG.train.min_freq
    # 选出满足频次的 token
    candidate_ids = np.nonzero(freq >= min_freq)[0]
    # 若超过 vocab_limit，按频次排序截断
    if candidate_ids.size > vocab_limit:
        topk = np.argsort(freq[candidate_ids])[::-1][:vocab_limit]
        keep_ids = candidate_ids[topk]
    else:
        keep_ids = candidate_ids

    # 将保留 token 压缩为连续 id，显著减少 embedding 尺寸
    old_to_new = np.full_like(freq, -1, dtype=np.int64)
    old_to_new[keep_ids] = np.arange(len(keep_ids), dtype=np.int64)
    data = old_to_new[data]

    # 压缩后重新统计频次（已去除丢弃 token）
    valid_tokens = data[data >= 0]
    if valid_tokens.size == 0:
        raise ValueError("过滤后没有可用的 token，请检查 min_freq 或子采样参数。")
    freq_new = np.bincount(valid_tokens, minlength=len(keep_ids))

    # 构造 keep_tokens 布尔表（已压缩，全部保留）
    keep_tokens = np.ones(len(freq_new), dtype=bool)

    # 高频子采样保留概率（压缩索引空间）
    subsample_probs = None
    if CONFIG.train.subsample_t > 0:
        subsample_probs = _compute_subsample_probs(freq_new, CONFIG.train.subsample_t)

    # 负采样分布（freq^0.75，压缩索引空间）
    neg_prob = _prepare_negative_sampling_probs(freq_new)

    # 压缩后的词表规模
    vocab_size = len(freq_new)
    logger.info("压缩后词表规模: %d", vocab_size)

    # 估算每个 epoch 的样本数（用于 steps_per_epoch）
    # 每个序列中，每个有效 token 大约产生 2*window_size 个 skip-gram 样本
    n_rows = data.shape[0]
    avg_valid_tokens_per_row = valid_tokens.size / n_rows if n_rows > 0 else 0
    window_size = CONFIG.train.window_size
    estimated_samples = int(n_rows * avg_valid_tokens_per_row * 2 * window_size * 0.5)  # 保守估计
    batch_size = CONFIG.train.batch_size_word2vec
    steps_per_epoch = max(1, estimated_samples // batch_size)
    logger.info("估算样本数: %d, steps_per_epoch: %d", estimated_samples, steps_per_epoch)

    # 构造流式 Dataset，分块生成 skip-gram 样本，降低内存占用
    seq_chunk_size = CONFIG.train.seq_chunk_size
    num_negative = CONFIG.train.num_negative_samples

    def generator():
        yield from _skipgram_generator(
            data=data,
            window_size=window_size,
            num_ns=num_negative,
            max_sequences=max_sequences,
            seq_chunk_size=seq_chunk_size,
            keep_tokens=keep_tokens,
            subsample_probs=subsample_probs,
            neg_sampling_prob=neg_prob,
            rng_seed=CONFIG.random_seed,
        )

    output_signature = (
        (
            tf.TensorSpec(shape=(), dtype=tf.int64),  # target
            tf.TensorSpec(shape=(num_negative + 1,), dtype=tf.int64),  # context+negatives
        ),
        tf.TensorSpec(shape=(num_negative + 1,), dtype=tf.int64),  # labels
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.shuffle(10_000)
    dataset = dataset.batch(CONFIG.train.batch_size_word2vec, drop_remainder=True)
    dataset = dataset.repeat()  # 允许多轮 epoch 迭代，避免数据耗尽
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # 分布式与混合精度策略
    strategy = _get_strategy()
    logger.info("分布式策略: %s, 设备数: %d", strategy.__class__.__name__, strategy.num_replicas_in_sync)

    epochs = 30
    accum_steps = max(1, CONFIG.train.accum_steps_word2vec)
    with strategy.scope():
        model = Word2Vec(vocab_size=vocab_size, embedding_dim=CONFIG.train.embedding_dim)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        base_opt = tf.keras.optimizers.Adam()
        # 混合精度下使用 LossScaleOptimizer 以稳定训练
        if mixed_precision.global_policy().compute_dtype == "float16":
            optimizer = mixed_precision.LossScaleOptimizer(base_opt)
        else:
            optimizer = base_opt

    if accum_steps > 1:
        logger.info("开启梯度累积模式: accum_steps=%d", accum_steps)
        dataset_iter = iter(dataset)
        for epoch in range(epochs):
            pbar = tqdm(range(steps_per_epoch), desc=f"Word2Vec 累积训练 Epoch {epoch+1}/{epochs}", unit="step")
            accum_grads = None
            step_in_epoch = 0
            for _ in pbar:
                try:
                    (targets, contexts), labels = next(dataset_iter)
                except StopIteration:
                    # 数据耗尽时重新创建迭代器（配合 .repeat() 使用时不会触发）
                    dataset_iter = iter(dataset)
                    (targets, contexts), labels = next(dataset_iter)

                with tf.GradientTape() as tape:
                    logits = model((targets, contexts), training=True)
                    loss = loss_fn(tf.cast(labels, tf.float32), logits) / accum_steps
                    # 混合精度场景对 loss 进行缩放
                    if isinstance(optimizer, mixed_precision.LossScaleOptimizer):
                        scaled_loss = optimizer.get_scaled_loss(loss)
                    else:
                        scaled_loss = loss
                grads = tape.gradient(scaled_loss, model.trainable_variables)
                if isinstance(optimizer, mixed_precision.LossScaleOptimizer):
                    grads = optimizer.get_unscaled_gradients(grads)
                if accum_grads is None:
                    accum_grads = [tf.zeros_like(g) if g is not None else None for g in grads]
                accum_grads = [
                    ag + g if ag is not None and g is not None else ag if g is None else g
                    for ag, g in zip(accum_grads, grads)
                ]
                step_in_epoch += 1

                if step_in_epoch % accum_steps == 0:
                    optimizer.apply_gradients(
                        ( (g, v) for g, v in zip(accum_grads, model.trainable_variables) if g is not None )
                    )
                    accum_grads = None
                pbar.set_postfix({"loss": float(loss) * accum_steps})

            # epoch 末尾若仍有残余梯度，补一次更新
            if accum_grads is not None:
                optimizer.apply_gradients(
                    ( (g, v) for g, v in zip(accum_grads, model.trainable_variables) if g is not None )
                )
    else:
        with strategy.scope():
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=["accuracy"],
            )
        # 学习率调度：当 loss 停滞时自动降低学习率
        lr_callback = ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,       # 每次降低为原来的一半
            patience=3,       # 连续 3 个 epoch 无改善则触发
            min_lr=1e-6,      # 最低学习率
            verbose=1,
        )
        tqdm_callback = TqdmProgressCallback(epochs=epochs, desc="Word2Vec 训练")
        model.fit(
            dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,  # 指定每个 epoch 的步数，配合 .repeat() 使用
            verbose=0,
            callbacks=[tqdm_callback, lr_callback],
        )

    tokens = _load_vocab_tokens(vocab_path, vocab_size, kept_old_ids=keep_ids)
    vectors_path = CONFIG.paths.vectors_path
    metadata_path = CONFIG.paths.metadata_path
    _export_vectors(model, tokens, vectors_path, metadata_path)
    return vectors_path, metadata_path


