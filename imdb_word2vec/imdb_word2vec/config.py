from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import tensorflow as tf

# 全局随机种子，确保流程可复现
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)


@dataclass
class PathConfig:
    """路径相关配置，集中管理数据与输出位置。"""

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "imdb_data"
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "cache"
    )
    logs_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "logs"
    )
    artifacts_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "artifacts"
    )
    slices_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data_slices"
    )

    @property
    def fused_features_path(self) -> Path:
        return self.cache_dir / "fused_features.parquet"

    @property
    def final_mapped_path(self) -> Path:
        return self.cache_dir / "final_mapped_vec.csv"

    @property
    def vocab_path(self) -> Path:
        return self.cache_dir / "vocab.csv"

    @property
    def best_model_path(self) -> Path:
        return self.artifacts_dir / "best_model.keras"

    @property
    def vectors_path(self) -> Path:
        return self.artifacts_dir / "vectors.tsv"

    @property
    def metadata_path(self) -> Path:
        return self.artifacts_dir / "metadata.tsv"

    def ensure(self) -> None:
        """确保目录存在。"""
        for folder in [self.data_dir, self.cache_dir, self.logs_dir, self.artifacts_dir, self.slices_dir]:
            folder.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """数据下载与采样配置。"""

    subset_rows: Optional[int] = 100_000  # None 表示全量
    tsv_urls: Dict[str, str] = field(
        default_factory=lambda: {
            "name.basics.tsv.gz": "https://datasets.imdbws.com/name.basics.tsv.gz",
            "title.akas.tsv.gz": "https://datasets.imdbws.com/title.akas.tsv.gz",
            "title.basics.tsv.gz": "https://datasets.imdbws.com/title.basics.tsv.gz",
            "title.crew.tsv.gz": "https://datasets.imdbws.com/title.crew.tsv.gz",
            "title.episode.tsv.gz":"https://datasets.imdbws.com/title.episode.tsv.gz",
            "title.principals.tsv.gz":"https://datasets.imdbws.com/title.principals.tsv.gz",
            "title.ratings.tsv.gz": "https://datasets.imdbws.com/title.ratings.tsv.gz",
        }
    )


@dataclass
class TrainConfig:
    """训练相关配置，兼顾 GPU/CPU 回退。"""

    use_gpu: bool = field(default_factory=lambda: bool(tf.config.list_physical_devices("GPU")))
    batch_size_autoencoder: int = 2048
    epochs_autoencoder: int = 50
    autoencoder_val_split: float = 0.2
    batch_size_word2vec: int = 1024
    window_size: int = 2
    num_negative_samples: int = 10
    embedding_dim: int = 150
    vocab_limit: int = 20_000  # 用于 Word2Vec 的词表上限
    max_sequences: Optional[int] = None  # Word2Vec 训练样本上限，用于小样本快速验证


@dataclass
class Config:
    """顶层配置对象，聚合路径、数据与训练配置。"""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    random_seed: int = GLOBAL_SEED

    def setup(self) -> None:
        """创建必要目录并设置随机种子。"""
        self.paths.ensure()
        os.environ["PYTHONHASHSEED"] = str(self.random_seed)
        tf.random.set_seed(self.random_seed)


CONFIG = Config()

