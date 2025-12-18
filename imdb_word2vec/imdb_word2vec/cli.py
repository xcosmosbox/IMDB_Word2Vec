from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import CONFIG
from .download import download_all
from .feature_engineering import run_feature_engineering
from .fusion import train_autoencoder
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
    preprocess_parser.add_argument("--subset-rows", type=int, default=None, help="可选行数上限，用于采样")

    # feature engineering
    fe_parser = subparsers.add_parser("fe", help="特征工程与向量化")
    fe_parser.add_argument("--subset-rows", type=int, default=None, help="可选行数上限，用于采样")

    # fusion
    fusion_parser = subparsers.add_parser("fusion", help="自编码器融合")
    fusion_parser.add_argument("--max-rows", type=int, default=None, help="训练行数上限，便于小样本验证")

    # train
    train_parser = subparsers.add_parser("train", help="Word2Vec 训练")
    train_parser.add_argument("--max-seq", type=int, default=None, help="序列数上限，便于小样本验证")
    train_parser.add_argument("--vocab-limit", type=int, default=CONFIG.train.vocab_limit, help="词表上限")

    # eval baselines
    eval_parser = subparsers.add_parser("eval", help="基线验证与报告生成")
    eval_parser.add_argument(
        "--mode",
        choices=["ae", "w2v", "combo", "all"],
        default="all",
        help="选择评估对象：自编码器/Word2Vec/拼接或全部",
    )
    eval_parser.add_argument("--sample-rows", type=int, default=200, help="AE/拼接的采样行数")
    eval_parser.add_argument("--sample-vocab", type=int, default=200, help="Word2Vec 采样 token 数")
    eval_parser.add_argument("--kmeans-k", type=int, default=8, help="聚类簇数")
    eval_parser.add_argument("--top-k", type=int, default=5, help="相似度 TopK")

    # all
    all_parser = subparsers.add_parser("all", help="串行执行全部步骤")
    all_parser.add_argument("--subset-rows", type=int, default=None, help="预处理采样行数")
    all_parser.add_argument("--max-rows", type=int, default=None, help="融合阶段行数上限")
    all_parser.add_argument("--max-seq", type=int, default=None, help="Word2Vec 序列数上限")
    all_parser.add_argument("--vocab-limit", type=int, default=CONFIG.train.vocab_limit, help="词表上限")

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

    if args.command == "fusion":
        train_autoencoder(CONFIG.paths.final_mapped_path, max_rows=args.max_rows)
        return

    if args.command == "train":
        train_word2vec(
            CONFIG.paths.final_mapped_path,  # 使用离散整数序列，避免自编码器输出的连续值
            CONFIG.paths.vocab_path,
            vocab_limit=args.vocab_limit,
            max_sequences=args.max_seq,
        )
        return

    if args.command == "eval":
        from .tests.eval_baselines import EvalConfig, run_eval

        cfg = EvalConfig(
            sample_rows=args.sample_rows,
            sample_vocab=args.sample_vocab,
            top_k=args.top_k,
            kmeans_k=args.kmeans_k,
        )
        report_path = run_eval(args.mode, cfg)
        logger.info("评估完成，报告输出：%s", report_path)
        return

    if args.command == "all":
        if args.subset_rows is not None:
            CONFIG.data.subset_rows = args.subset_rows

        paths = download_all()
        movies_info_df, staff_df, regional_titles_df = preprocess_all(
            paths, subset_rows=CONFIG.data.subset_rows
        )
        run_feature_engineering(movies_info_df, staff_df, regional_titles_df)
        train_autoencoder(CONFIG.paths.final_mapped_path, max_rows=args.max_rows)
        train_word2vec(
            CONFIG.paths.final_mapped_path,  # 使用离散整数序列
            CONFIG.paths.vocab_path,
            vocab_limit=args.vocab_limit,
            max_sequences=args.max_seq,
        )
        return


if __name__ == "__main__":
    main()


