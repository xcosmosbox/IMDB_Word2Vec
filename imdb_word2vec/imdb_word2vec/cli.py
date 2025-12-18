from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import CONFIG
from .download import download_all
from .feature_engineering import run_feature_engineering
from .logging_utils import setup_logging
from .preprocess import preprocess_all
from .training import train_word2vec


logger = setup_logging(CONFIG.paths.logs_dir)


def _load_cached_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """从缓存路径加载预处理后的 DataFrame。"""
    movies_info_df = pd.read_csv(CONFIG.paths.cache_dir / "movies_info_df.csv")
    staff_df = pd.read_csv(CONFIG.paths.cache_dir / "staff_df.csv")
    regional_titles_df = pd.read_csv(CONFIG.paths.cache_dir / "regional_titles_df.csv")
    return movies_info_df, staff_df, regional_titles_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMDb Word2Vec 全流程 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # download
    subparsers.add_parser("download", help="下载并解压 IMDb 数据")

    # preprocess
    preprocess_parser = subparsers.add_parser("preprocess", help="清洗并生成基础表")
    preprocess_parser.add_argument("--subset-rows", type=int, default=None, help="可选行数上限")

    # feature engineering
    fe_parser = subparsers.add_parser("fe", help="特征工程：生成 Word2Vec 序列数据")
    fe_parser.add_argument("--subset-rows", type=int, default=None, help="可选行数上限")

    # train
    train_parser = subparsers.add_parser("train", help="Word2Vec 训练")
    train_parser.add_argument("--max-seq", type=int, default=None, help="序列数上限")
    train_parser.add_argument("--vocab-limit", type=int, default=CONFIG.train.vocab_limit, help="词表上限")

    # all: 完整流程
    all_parser = subparsers.add_parser("all", help="执行完整流程: download → preprocess → fe → train")
    all_parser.add_argument("--subset-rows", type=int, default=None, help="预处理采样行数")
    all_parser.add_argument("--max-seq", type=int, default=None, help="Word2Vec 序列数上限")
    all_parser.add_argument("--vocab-limit", type=int, default=CONFIG.train.vocab_limit, help="词表上限")

    # eval baselines（可选）
    eval_parser = subparsers.add_parser("eval", help="基线验证与报告生成")
    eval_parser.add_argument(
        "--mode",
        choices=["w2v", "all"],
        default="w2v",
        help="选择评估对象",
    )
    eval_parser.add_argument("--sample-vocab", type=int, default=200, help="采样 token 数")
    eval_parser.add_argument("--top-k", type=int, default=5, help="相似度 TopK")

    return parser.parse_args()


def main() -> None:
    CONFIG.setup()
    args = parse_args()

    if args.command == "download":
        download_all()
        return

    if args.command == "preprocess":
        if args.subset_rows is not None:
            CONFIG.data.subset_rows = args.subset_rows
        paths = download_all()
        preprocess_all(paths, subset_rows=CONFIG.data.subset_rows)
        return

    if args.command == "fe":
        if args.subset_rows is not None:
            CONFIG.data.subset_rows = args.subset_rows
        movies_info_df, staff_df, regional_titles_df = _load_cached_frames()
        run_feature_engineering(movies_info_df, staff_df, regional_titles_df)
        return

    if args.command == "train":
        train_word2vec(
            CONFIG.paths.final_mapped_path,
            CONFIG.paths.vocab_path,
            vocab_limit=args.vocab_limit,
            max_sequences=args.max_seq,
        )
        return

    if args.command == "eval":
        # 简化的评估
        logger.info("评估功能待实现")
        return

    if args.command == "all":
        if args.subset_rows is not None:
            CONFIG.data.subset_rows = args.subset_rows

        # Step 1: Download
        logger.info("========== Step 1/4: 下载数据 ==========")
        paths = download_all()
        
        # Step 2: Preprocess
        logger.info("========== Step 2/4: 数据预处理 ==========")
        movies_info_df, staff_df, regional_titles_df = preprocess_all(
            paths, subset_rows=CONFIG.data.subset_rows
        )
        
        # Step 3: Feature Engineering (序列生成)
        logger.info("========== Step 3/4: 特征工程 ==========")
        run_feature_engineering(movies_info_df, staff_df, regional_titles_df)
        
        # Step 4: Train Word2Vec
        logger.info("========== Step 4/4: Word2Vec 训练 ==========")
        train_word2vec(
            CONFIG.paths.final_mapped_path,
            CONFIG.paths.vocab_path,
            vocab_limit=args.vocab_limit,
            max_sequences=args.max_seq,
        )
        
        logger.info("========== 全流程完成 ==========")
        return


if __name__ == "__main__":
    main()
