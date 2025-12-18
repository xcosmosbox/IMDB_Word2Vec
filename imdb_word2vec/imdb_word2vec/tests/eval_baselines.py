from __future__ import annotations

"""基线验证与报告生成：自编码器 / Word2Vec / 拼接向量."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from ..config import CONFIG
from ..logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)


@dataclass
class EvalConfig:
    sample_rows: int = 200  # 采样行数（AE/拼接）
    sample_vocab: int = 200  # 采样 token 数（W2V）
    top_k: int = 5  # 相似度 TopK
    kmeans_k: int = 8  # 聚类簇数


def _cosine_topk(mat: np.ndarray, top_k: int) -> List[Tuple[int, List[Tuple[int, float]]]]:
    """对给定矩阵计算每个向量的 TopK 近邻（排除自身）。"""
    mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = mat_norm @ mat_norm.T
    results = []
    for i in range(mat.shape[0]):
        sims[i, i] = -1  # 排除自身
        idx = np.argpartition(-sims[i], range(top_k))[:top_k]
        idx = idx[np.argsort(-sims[i, idx])]
        results.append((i, [(int(j), float(sims[i, j])) for j in idx]))
    return results


def evaluate_word2vec(vectors_path: Path, metadata_path: Path, cfg: EvalConfig) -> str:
    """基于词向量的相似度与聚类报告。"""
    vectors = np.loadtxt(vectors_path, delimiter="\t", dtype=np.float32)
    tokens = metadata_path.read_text(encoding="utf-8").splitlines()
    vocab_size = min(len(tokens), vectors.shape[0] + 1)  # metadata 与 vectors 对齐

    # 采样 token（跳过填充 0）
    candidate_ids = np.arange(1, vocab_size)
    if candidate_ids.size > cfg.sample_vocab:
        rng = np.random.default_rng(CONFIG.random_seed)
        candidate_ids = rng.choice(candidate_ids, size=cfg.sample_vocab, replace=False)
    emb = vectors[candidate_ids - 1]  # vectors 不含填充位

    # 相似度
    topk = _cosine_topk(emb, cfg.top_k)
    sim_lines = []
    for idx, neigh in topk[:5]:  # 展示前 5 个样本
        token = tokens[candidate_ids[idx]]
        neigh_str = ", ".join(f"{tokens[candidate_ids[j]]}:{score:.3f}" for j, score in neigh)
        sim_lines.append(f"- {token} -> {neigh_str}")

    # 聚类
    kmeans = KMeans(n_clusters=min(cfg.kmeans_k, emb.shape[0]), n_init="auto", random_state=CONFIG.random_seed)
    labels = kmeans.fit_predict(emb)
    counts = pd.Series(labels).value_counts().sort_index()
    cluster_str = ", ".join(f"C{i}:{cnt}" for i, cnt in counts.items())

    report = [
        "## Word2Vec 基线",
        f"- 采样 token 数: {emb.shape[0]} / vocab_size≈{vocab_size}",
        f"- Top{cfg.top_k} 相似度示例（前5行）：",
        *sim_lines,
        f"- KMeans(k={kmeans.n_clusters}) 簇大小: {cluster_str}",
    ]
    return "\n".join(report)


def evaluate_autoencoder(fused_path: Path, cfg: EvalConfig) -> str:
    """基于自编码器输出的相似度与聚类报告。"""
    fused = pd.read_parquet(fused_path).values.astype(np.float32)
    if fused.shape[0] > cfg.sample_rows:
        fused = fused[: cfg.sample_rows]

    topk = _cosine_topk(fused, cfg.top_k)
    sim_lines = []
    for idx, neigh in topk[:5]:
        neigh_str = ", ".join(f"row{j}:{score:.3f}" for j, score in neigh)
        sim_lines.append(f"- row{idx} -> {neigh_str}")

    kmeans = KMeans(n_clusters=min(cfg.kmeans_k, fused.shape[0]), n_init="auto", random_state=CONFIG.random_seed)
    labels = kmeans.fit_predict(fused)
    counts = pd.Series(labels).value_counts().sort_index()
    cluster_str = ", ".join(f"C{i}:{cnt}" for i, cnt in counts.items())

    report = [
        "## Autoencoder 基线",
        f"- 采样行数: {fused.shape[0]}",
        f"- Top{cfg.top_k} 相似度示例（前5行）：",
        *sim_lines,
        f"- KMeans(k={kmeans.n_clusters}) 簇大小: {cluster_str}",
    ]
    return "\n".join(report)


def evaluate_combined(
    final_mapped_path: Path,
    fused_path: Path,
    vectors_path: Path,
    metadata_path: Path,
    cfg: EvalConfig,
) -> str:
    """行级拼接向量：AE 行向量 + 行内 token 的平均 W2V 向量，再做相似度与聚类."""
    fused = pd.read_parquet(fused_path).values.astype(np.float32)
    mapped = pd.read_csv(final_mapped_path).values.astype(np.int64)
    vectors = np.loadtxt(vectors_path, delimiter="\t", dtype=np.float32)

    n_rows = min(fused.shape[0], mapped.shape[0], cfg.sample_rows)
    fused = fused[:n_rows]
    mapped = mapped[:n_rows]

    # 行内 token 平均词向量（忽略 0 和超出范围的 token）
    w2v_dim = vectors.shape[1]
    mean_embs = np.zeros((n_rows, w2v_dim), dtype=np.float32)
    for i in tqdm(range(n_rows), desc="平均 W2V", disable=not CONFIG.data.enable_tqdm):
        row = mapped[i]
        row = row[(row > 0) & (row <= vectors.shape[0])]
        if row.size > 0:
            mean_embs[i] = vectors[row - 1].mean(axis=0)

    combined = np.concatenate([fused, mean_embs], axis=1)
    topk = _cosine_topk(combined, cfg.top_k)
    sim_lines = []
    for idx, neigh in topk[:5]:
        neigh_str = ", ".join(f"row{j}:{score:.3f}" for j, score in neigh)
        sim_lines.append(f"- row{idx} -> {neigh_str}")

    kmeans = KMeans(n_clusters=min(cfg.kmeans_k, combined.shape[0]), n_init="auto", random_state=CONFIG.random_seed)
    labels = kmeans.fit_predict(combined)
    counts = pd.Series(labels).value_counts().sort_index()
    cluster_str = ", ".join(f"C{i}:{cnt}" for i, cnt in counts.items())

    report = [
        "## 拼接向量基线（AE + 行内平均 W2V）",
        f"- 采样行数: {combined.shape[0]}",
        f"- Top{cfg.top_k} 相似度示例（前5行）：",
        *sim_lines,
        f"- KMeans(k={kmeans.n_clusters}) 簇大小: {cluster_str}",
    ]
    return "\n".join(report)


def write_report(sections: List[str], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = "# 基线验证报告\n\n" + "\n\n".join(sections)
    out_path.write_text(content, encoding="utf-8")
    logger.info("评估报告已生成：%s", out_path)
    return out_path


def run_eval(mode: str, cfg: EvalConfig) -> Path:
    sections = []
    if mode in ("ae", "all"):
        sections.append(evaluate_autoencoder(CONFIG.paths.fused_features_path, cfg))
    if mode in ("w2v", "all"):
        sections.append(
            evaluate_word2vec(CONFIG.paths.vectors_path, CONFIG.paths.metadata_path, cfg)
        )
    if mode in ("combo", "all"):
        sections.append(
            evaluate_combined(
                CONFIG.paths.final_mapped_path,
                CONFIG.paths.fused_features_path,
                CONFIG.paths.vectors_path,
                CONFIG.paths.metadata_path,
                cfg,
            )
        )
    return write_report(sections, CONFIG.paths.artifacts_dir / "eval_report.md")


