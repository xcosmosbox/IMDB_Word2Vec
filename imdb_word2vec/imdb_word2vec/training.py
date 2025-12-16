from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from .config import CONFIG
from .logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)


def _generate_training_data(
    sequences: List[List[int]],
    window_size: int,
    num_ns: int,
    vocab_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据窗口与负采样参数生成 Skip-gram 训练数据（纯 NumPy 实现，避免 skipgrams 随机数 float Bug）。"""
    targets, contexts, labels = [], [], []
    rng = np.random.default_rng(seed)

    for sequence in sequences:
        seq_len = len(sequence)
        for i, target_word in enumerate(sequence):
            # 窗口内的正样本
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, seq_len)
            for j in range(start, end):
                if j == i:
                    continue
                context_word = sequence[j]

                # 负样本，排除当前 target/context
                negatives = []
                while len(negatives) < num_ns:
                    candidate = rng.integers(0, vocab_size)
                    if candidate != context_word and candidate != target_word:
                        negatives.append(candidate)

                context = np.array([context_word] + negatives, dtype=np.int64)
                label = np.array([1] + [0] * num_ns, dtype=np.int64)

                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

    return np.array(targets, dtype=np.int64), np.array(contexts, dtype=np.int64), np.array(labels, dtype=np.int64)


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
        for idx, token in enumerate(tokens):
            if idx == 0:
                continue  # 0 作为填充位，跳过
            vec = weights[idx]
            out_v.write("\t".join([str(x) for x in vec]) + "\n")
            out_m.write(token + "\n")
    logger.info("词向量与元数据已导出：%s, %s", vectors_path, metadata_path)


def _load_vocab_tokens(vocab_path: Path, vocab_size: int) -> List[str]:
    """根据 vocab 文件构造按索引排序的 token 列表。"""
    vocab_df = pd.read_csv(vocab_path, header=None, names=["key", "value"])
    inv = vocab_df.set_index("value")["key"].to_dict()
    tokens = [inv.get(i, str(i)) for i in range(vocab_size)]
    return tokens


def train_word2vec(
    fused_features_path: Path,
    vocab_path: Path,
    vocab_limit: int,
    max_sequences: Optional[int] = None,
) -> Tuple[Path, Path]:
    """使用融合特征训练 Word2Vec，返回向量与元数据路径。"""
    fused_df = pd.read_parquet(fused_features_path)
    data = fused_df.values.astype(int)
    if max_sequences is not None:
        data = data[:max_sequences]

    max_token = int(data.max()) if data.size > 0 else 0
    vocab_size = min(vocab_limit, max_token + 1)
    data = np.clip(data, 0, vocab_size - 1)

    sequences = data.tolist()
    targets, contexts, labels = _generate_training_data(
        sequences=sequences,
        window_size=CONFIG.train.window_size,
        num_ns=CONFIG.train.num_negative_samples,
        vocab_size=vocab_size,
        seed=CONFIG.random_seed,
    )

    logger.info(
        "Skip-gram 数据集: targets=%s contexts=%s labels=%s",
        targets.shape,
        contexts.shape,
        labels.shape,
    )

    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(10_000).batch(CONFIG.train.batch_size_word2vec, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    model = Word2Vec(vocab_size=vocab_size, embedding_dim=CONFIG.train.embedding_dim)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(dataset, epochs=25, verbose=1)

    tokens = _load_vocab_tokens(vocab_path, vocab_size)
    vectors_path = CONFIG.paths.vectors_path
    metadata_path = CONFIG.paths.metadata_path
    _export_vectors(model, tokens, vectors_path, metadata_path)
    return vectors_path, metadata_path


