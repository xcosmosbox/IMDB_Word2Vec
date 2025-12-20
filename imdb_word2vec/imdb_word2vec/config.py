from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import tensorflow as tf

# 全局随机种子，确保流程可复现
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)


def detect_device() -> Tuple[str, str]:
    """检测最佳可用设备，优先级：NVIDIA CUDA > Apple Metal > CPU。

    Returns:
        (device_string, device_name): 例如 ("/GPU:0", "NVIDIA") 或 ("/GPU:0", "Metal") 或 ("/CPU:0", "CPU")
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # 尝试获取 GPU 详情以区分 NVIDIA 和 Metal
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            device_name = gpu_details.get("device_name", "")

            # 检查是否为 NVIDIA GPU
            if "NVIDIA" in device_name.upper():
                return "/GPU:0", "NVIDIA"

            # macOS 上的 Metal GPU
            if sys.platform == "darwin":
                return "/GPU:0", "Metal"

            # 其他 GPU（可能是 AMD 等）
            return "/GPU:0", "GPU"
        except Exception:
            # 如果无法获取详情，但有 GPU 可用
            if sys.platform == "darwin":
                return "/GPU:0", "Metal"
            return "/GPU:0", "GPU"

    return "/CPU:0", "CPU"


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
        """Word2Vec 序列数据路径"""
        return self.cache_dir / "word2vec_sequences.csv"

    @property
    def tabular_features_path(self) -> Path:
        """Autoencoder 表格特征数据路径"""
        return self.cache_dir / "tabular_features.csv"

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

    # subset_rows: Optional[int] = 100_000  # None 表示全量
    subset_rows: Optional[int] = None  # None 表示全量
    enable_tqdm: bool = True  # 数据相关步骤是否开启 tqdm 进度展示
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
    """训练相关配置，兼顾 NVIDIA/Metal/CPU 多环境回退。"""

    # 设备检测结果：(device_string, device_type)
    _device_info: Tuple[str, str] = field(default_factory=detect_device)
    enable_distribute: bool = True  # 多 GPU 时启用 MirroredStrategy
    enable_mixed_precision: bool = True  # GPU/Metal 环境启用混合精度以节省显存

    # 词表与采样相关配置
    min_freq: int = 3  # 低频裁剪阈值，低于该频次的 token 将被丢弃
    subsample_t: float = 1e-4  # 高频子采样阈值，越小丢弃高频 token 越多
    vocab_limit: int = 50_000  # 用于 Word2Vec 的词表上限

    # 负采样与窗口
    window_size: int = 5
    num_negative_samples: int = 20  # 适度降低负样本，平衡速度与质量

    # 数据规模与分块
    max_sequences: Optional[int] = None  # 训练语料行数上限
    seq_chunk_size: int = 100_000  # 分块生成 skip-gram 样本的序列块大小

    # 数据集缓存与 shuffle 配置
    shuffle_buffer_size: int = 50_000  # shuffle 缓冲区大小，越大随机性越好但内存占用越多
    precache_samples: bool = True  # 是否预先生成所有样本到内存（加速训练，但需要更多内存）
    precache_max_samples: Optional[int] = 5_000_000  # 预缓存的最大样本数，None 表示全部缓存
    chunked_epoch_training: bool = True  # 内存不足时，每个 epoch 加载不同数据块（确保全量数据都被训练）

    # 并行样本生成配置
    parallel_workers: Optional[int] = None  # 并行生成样本的进程数，None 表示自动检测（CPU核心数-1）
    parallel_chunk_size: int = 10000  # 每个进程处理的序列数

    # 梯度累积（显存紧张时可开启）
    accum_steps_word2vec: int = 1  # >1 时开启梯度累积
    accum_steps_autoencoder: int = 1

    batch_size_autoencoder: int = 1024
    epochs_autoencoder: int = 10
    autoencoder_val_split: float = 0.2
    batch_size_word2vec: int = 1024
    embedding_dim: int = 128

    @property
    def device_string(self) -> str:
        """获取 TensorFlow 设备字符串，如 '/GPU:0' 或 '/CPU:0'。"""
        return self._device_info[0]

    @property
    def device_type(self) -> str:
        """获取设备类型名称：'NVIDIA', 'Metal', 'GPU', 或 'CPU'。"""
        return self._device_info[1]

    @property
    def use_gpu(self) -> bool:
        """向后兼容：是否使用 GPU。"""
        return self._device_info[1] != "CPU"


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
        # 在支持 GPU/Metal 时按需启用混合精度
        if self.train.enable_mixed_precision and self.train.use_gpu:
            try:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float32")
            except Exception:
                pass


CONFIG = Config()

