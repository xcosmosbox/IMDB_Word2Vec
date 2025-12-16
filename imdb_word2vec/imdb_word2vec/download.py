from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import Dict

import requests
from tqdm import tqdm

from .config import CONFIG
from .logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)


def _download_file(url: str, target: Path) -> None:
    """下载单个压缩文件，带简单进度条。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        logger.info("已存在，跳过下载: %s", target.name)
        return

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as bar:
            with target.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    logger.info("下载完成: %s", target.name)


def _decompress_gzip(src: Path, dst: Path) -> None:
    """解压 gzip 为 tsv。"""
    if dst.exists():
        logger.info("已存在，跳过解压: %s", dst.name)
        return
    with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    logger.info("解压完成: %s -> %s", src.name, dst.name)


def _save_head_slice(tsv_path: Path, slice_dir: Path, nrows: int = 1000) -> None:
    """保存 TSV 的前 n 行到切片目录，便于快速查看，不影响后续训练。"""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("未安装 pandas，跳过切片：%s", tsv_path.name)
        return

    slice_dir.mkdir(parents=True, exist_ok=True)
    slice_path = slice_dir / tsv_path.name.replace(".tsv", "_head.tsv")
    if slice_path.exists():
        return
    try:
        df_head = pd.read_csv(tsv_path, sep="\t", nrows=nrows, low_memory=False)
        df_head.to_csv(slice_path, sep="\t", index=False)
        logger.info("已保存前 %d 行切片：%s", nrows, slice_path.name)
    except Exception as exc:
        logger.warning("切片失败 %s: %s", tsv_path.name, exc)


def download_all(tsv_urls: Dict[str, str] | None = None) -> Dict[str, Path]:
    """下载并解压所需 IMDb 数据，返回 tsv 路径字典。"""
    urls = tsv_urls or CONFIG.data.tsv_urls
    CONFIG.paths.ensure()
    paths: Dict[str, Path] = {}

    for fname, url in urls.items():
        gz_path = CONFIG.paths.data_dir / fname
        tsv_path = CONFIG.paths.data_dir / fname.replace(".gz", "")
        _download_file(url, gz_path)
        _decompress_gzip(gz_path, tsv_path)
        _save_head_slice(tsv_path, CONFIG.paths.slices_dir, nrows=1000)
        paths[fname] = tsv_path
    return paths

