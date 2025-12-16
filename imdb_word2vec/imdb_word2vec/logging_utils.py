import logging
from pathlib import Path
from typing import Optional


def setup_logging(logs_dir: Path, name: str = "imdb_word2vec") -> logging.Logger:
    """初始化日志，输出到控制台与文件。

    日志格式保持简洁，便于观察流水线进度。
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(logs_dir / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取 logger，若未找到则返回 root logger。"""
    return logging.getLogger(name or "imdb_word2vec")

