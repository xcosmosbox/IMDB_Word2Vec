from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input
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

    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)

    input_layer = Input(shape=(data_std.shape[1],))
    x = Dense(512, activation="relu")(input_layer)
    x = Dropout(0.25)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.25)(x)
    output_layer = Dense(data_std.shape[1], activation="relu")(x)

    model: Model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mse")
    logger.info("自编码器结构: 输入维度=%d", data_std.shape[1])

    ckpt_path = CONFIG.paths.best_model_path
    epochs = CONFIG.train.epochs_autoencoder

    # 使用 tqdm 回调显示训练进度
    tqdm_callback = TqdmProgressCallback(epochs=epochs, desc="Autoencoder 训练")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=0,  # 关闭 ModelCheckpoint 的输出，使用 tqdm
        ),
        tqdm_callback,
    ]

    # 使用检测到的设备进行训练
    device = CONFIG.train.device_string
    device_type = CONFIG.train.device_type
    logger.info("使用设备进行训练: %s (%s)", device, device_type)

    with tf.device(device):
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
                fused_batches.append(fused_batch)
                pbar.update(1)
        fused = np.vstack(fused_batches)

    fused_df = pd.DataFrame(fused)
    fused_path = CONFIG.paths.fused_features_path
    fused_df.to_parquet(fused_path, index=False)
    logger.info("融合特征已保存：%s", fused_path)
    return fused_path


