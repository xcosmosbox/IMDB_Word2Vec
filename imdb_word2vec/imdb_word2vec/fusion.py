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

from .config import CONFIG
from .logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)


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
            verbose=1,
        ),
    ]

    device = "/GPU:0" if CONFIG.train.use_gpu else "/CPU:0"
    with tf.device(device):
        history = model.fit(
            data_std,
            data_std,
            epochs=CONFIG.train.epochs_autoencoder,
            batch_size=CONFIG.train.batch_size_autoencoder,
            validation_split=CONFIG.train.autoencoder_val_split,
            callbacks=callbacks,
            verbose=1,
        )
        logger.info("训练完成，最佳 val_loss=%.6f", min(history.history["val_loss"]))
        model.load_weights(ckpt_path)
        fused = model.predict(
            data_std, batch_size=CONFIG.train.batch_size_autoencoder, verbose=1
        )

    fused_df = pd.DataFrame(fused)
    fused_path = CONFIG.paths.fused_features_path
    fused_df.to_parquet(fused_path, index=False)
    logger.info("融合特征已保存：%s", fused_path)
    return fused_path


