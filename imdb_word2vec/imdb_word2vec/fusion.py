from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization
from tqdm import tqdm

from .config import CONFIG
from .logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)


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
        val_loss = logs.get("val_loss", 0)
        self.pbar.set_postfix({"loss": f"{loss:.4f}", "val_loss": f"{val_loss:.4f}"})
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        if self.pbar:
            self.pbar.close()


def train_autoencoder(
    final_mapped_path: Path,
    max_rows: Optional[int] = None,
) -> Path:
    """训练自编码器以融合高维离散特征，输出稠密表示。

    Args:
        final_mapped_path: 向量化后的 CSV 路径。
        max_rows: 可选的行数上限，用于小样本验证。
    """
    df = pd.read_csv(final_mapped_path)
    if max_rows is not None:
        df = df.head(max_rows)

    data = df.values
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(-1,1))
    data_std = scaler.fit_transform(data)

    # 分布式策略与混合精度由 config.setup 设置
    strategy = _get_strategy()
    logger.info("Autoencoder 分布式策略: %s, 设备数: %d", strategy.__class__.__name__, strategy.num_replicas_in_sync)

    with strategy.scope():
        input_layer = Input(shape=(data_std.shape[1],))

        x = Dense(192, activation="gelu")(input_layer)
        x = LayerNormalization()(x)
        x = Dropout(0.1)(x)

        x = Dense(128, activation="gelu")(x)
        x = LayerNormalization()(x)
        x = Dropout(0.1)(x)

        x = Dense(64, activation="gelu")(x)  # 瓶颈
        x = LayerNormalization()(x)
        x = Dropout(0.1)(x)

        x = Dense(128, activation="gelu")(x)
        x = LayerNormalization()(x)
        x = Dropout(0.1)(x)

        x = Dense(192, activation="gelu")(x)
        x = LayerNormalization()(x)
        x = Dropout(0.11)(x)

        output_layer = Dense(data_std.shape[1], activation="tanh")(x)

        model: Model = Model(inputs=input_layer, outputs=output_layer)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=opt, loss="mse")
        logger.info("自编码器结构: 输入维度=%d", data_std.shape[1])

    ckpt_path = CONFIG.paths.best_model_path
    epochs = CONFIG.train.epochs_autoencoder

    # 使用 tqdm 回调显示训练进度
    tqdm_callback = TqdmProgressCallback(epochs=epochs, desc="Autoencoder 训练")
    # 学习率调度：当 val_loss 停滞时自动降低学习率
    lr_callback = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,       # 每次降低为原来的一半
        patience=5,       # 连续 5 个 epoch 无改善则触发
        min_lr=1e-7,      # 最低学习率
        verbose=1,
    )
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,  # 增加耐心，配合学习率衰减
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=0,  # 关闭 ModelCheckpoint 的输出，使用 tqdm
        ),
        lr_callback,
        tqdm_callback,
    ]

    # 使用策略作用域下训练
    history = model.fit(
        data_std,
        data_std,
        epochs=epochs,
        batch_size=CONFIG.train.batch_size_autoencoder,
        validation_split=CONFIG.train.autoencoder_val_split,
        callbacks=callbacks,
        verbose=0,  # 关闭 Keras 默认输出，使用 tqdm
    )
    logger.info("训练完成，最佳 val_loss=%.6f", min(history.history["val_loss"]))
    model.load_weights(ckpt_path)

    # 使用 tqdm 显示推理进度
    logger.info("正在生成融合特征...")
    n_samples = data_std.shape[0]
    batch_size = CONFIG.train.batch_size_autoencoder
    n_batches = (n_samples + batch_size - 1) // batch_size

    fused_batches = []
    with tqdm(total=n_batches, desc="推理进度", unit="batch") as pbar:
        for i in range(0, n_samples, batch_size):
            batch = data_std[i : i + batch_size]
            fused_batch = model.predict(batch, verbose=0)
            # 混合精度下输出可能为 float16，这里统一转回 float32 便于后续训练
            fused_batches.append(fused_batch.astype(np.float32))
            pbar.update(1)
    fused = np.vstack(fused_batches)

    fused_df = pd.DataFrame(fused)
    fused_path = CONFIG.paths.fused_features_path
    fused_df.to_parquet(fused_path, index=False)
    logger.info("融合特征已保存：%s", fused_path)
    return fused_path


