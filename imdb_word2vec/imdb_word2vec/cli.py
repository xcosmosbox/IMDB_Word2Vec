from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Support both relative and direct execution
try:
    from .config import CONFIG
    from .download import download_all
    from .feature_engineering import run_feature_engineering
    from .autoencoder import train_autoencoder
    from .logging_utils import setup_logging
    from .preprocess import preprocess_all
    from .training import train_word2vec
    from .pretraining import generate_training_samples
except ImportError:
    # If relative import fails, adjust sys.path and use absolute imports
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from imdb_word2vec.config import CONFIG
    from imdb_word2vec.download import download_all
    from imdb_word2vec.feature_engineering import run_feature_engineering
    from imdb_word2vec.autoencoder import train_autoencoder
    from imdb_word2vec.logging_utils import setup_logging
    from imdb_word2vec.preprocess import preprocess_all
    from imdb_word2vec.training import train_word2vec
    from imdb_word2vec.pretraining import generate_training_samples


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
    fe_parser = subparsers.add_parser("fe", help="特征工程：生成 Word2Vec 序列和 Autoencoder 表格特征")
    fe_parser.add_argument("--subset-rows", type=int, default=None, help="可选行数上限")

    # pretrain: 预生成训练样本
    pretrain_parser = subparsers.add_parser("pretrain", help="预生成训练样本到磁盘（可选，用于加速后续训练）")
    pretrain_parser.add_argument("--force", action="store_true", help="强制重新生成（删除已有样本）")
    pretrain_parser.add_argument("--memory-gb", type=float, default=None, help="每块内存限制 (GB)")

    # autoencoder (原 fusion)
    ae_parser = subparsers.add_parser("autoencoder", help="训练 Autoencoder（推荐系统组件）")
    ae_parser.add_argument("--max-rows", type=int, default=None, help="训练行数上限")

    # train (Word2Vec)
    train_parser = subparsers.add_parser("train", help="Word2Vec 训练")
    train_parser.add_argument("--max-seq", type=int, default=None, help="序列数上限")
    train_parser.add_argument("--vocab-limit", type=int, default=CONFIG.train.vocab_limit, help="词表上限")
    train_parser.add_argument("--use-cache", action="store_true", help="使用预生成的样本缓存")

    # all: 完整流程
    all_parser = subparsers.add_parser("all", help="执行完整流程: download → preprocess → fe → autoencoder → train")
    all_parser.add_argument("--subset-rows", type=int, default=None, help="预处理采样行数")
    all_parser.add_argument("--max-seq", type=int, default=None, help="Word2Vec 序列数上限")
    all_parser.add_argument("--vocab-limit", type=int, default=CONFIG.train.vocab_limit, help="词表上限")
    all_parser.add_argument("--skip-autoencoder", action="store_true", help="跳过 Autoencoder 训练")
    all_parser.add_argument("--pretrain", action="store_true", help="使用预生成样本方式训练")

    # export: 导出模型文件
    export_parser = subparsers.add_parser("export", help="导出所有格式的模型文件")
    export_parser.add_argument("--n-samples", type=int, default=5000, help="聚类可视化采样数")
    export_parser.add_argument("--n-clusters", type=int, default=20, help="聚类数")

    # eval baselines（可选）
    eval_parser = subparsers.add_parser("eval", help="基线验证与报告生成")
    eval_parser.add_argument(
        "--mode",
        choices=["w2v", "ae", "all"],
        default="all",
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

    if args.command == "pretrain":
        memory_gb = args.memory_gb if args.memory_gb else CONFIG.train.max_memory_gb
        generate_training_samples(
            sequences_path=CONFIG.paths.final_mapped_path,
            output_dir=CONFIG.paths.cache_dir / "samples",
            max_memory_gb=memory_gb,
            force_regenerate=args.force,
        )
        return

    if args.command == "autoencoder":
        train_autoencoder(
            CONFIG.paths.tabular_features_path,
            max_rows=args.max_rows,
        )
        return

    if args.command == "train":
        train_word2vec(
            CONFIG.paths.final_mapped_path,
            CONFIG.paths.vocab_path,
            vocab_limit=args.vocab_limit,
            max_sequences=args.max_seq,
            use_cache=args.use_cache,
        )
        return

    if args.command == "export":
        import numpy as np
        from .export import export_all
        
        # 加载已训练的权重
        weights_path = CONFIG.paths.artifacts_dir / "embeddings.npy"
        if not weights_path.exists():
            logger.error("找不到训练好的权重文件: %s", weights_path)
            logger.info("请先运行 train 命令")
            return
        
        weights = np.load(weights_path)
        
        # 加载词表
        vocab_df = pd.read_csv(CONFIG.paths.vocab_path, header=None, names=["key", "value"])
        inv = vocab_df.set_index("value")["key"].to_dict()
        tokens = [inv.get(i, f"<ID_{i}>") for i in range(len(weights))]
        
        export_all(weights, tokens, CONFIG.paths.artifacts_dir)
        return

    if args.command == "eval":
        logger.info("评估功能待实现")
        return

    if args.command == "all":
        if args.subset_rows is not None:
            CONFIG.data.subset_rows = args.subset_rows

        # Step 1: Download
        logger.info("========== Step 1/5: 下载数据 ==========")
        paths = download_all()
        
        # Step 2: Preprocess
        logger.info("========== Step 2/5: 数据预处理 ==========")
        movies_info_df, staff_df, regional_titles_df = preprocess_all(
            paths, subset_rows=CONFIG.data.subset_rows
        )
        
        # Step 3: Feature Engineering
        logger.info("========== Step 3/5: 特征工程 ==========")
        run_feature_engineering(movies_info_df, staff_df, regional_titles_df)
        
        # Step 4: Autoencoder (可选)
        if not args.skip_autoencoder:
            logger.info("========== Step 4/5: Autoencoder 训练 ==========")
            train_autoencoder(CONFIG.paths.tabular_features_path)
        else:
            logger.info("========== Step 4/5: 跳过 Autoencoder ==========")
        
        # Step 5: 预生成样本（可选）+ 训练
        if args.pretrain:
            logger.info("========== Step 5a/5: 预生成训练样本 ==========")
            generate_training_samples(
                sequences_path=CONFIG.paths.final_mapped_path,
                output_dir=CONFIG.paths.cache_dir / "samples",
            )
            logger.info("========== Step 5b/5: Word2Vec 训练（从缓存）==========")
            train_word2vec(
                CONFIG.paths.final_mapped_path,
                CONFIG.paths.vocab_path,
                vocab_limit=args.vocab_limit,
                max_sequences=args.max_seq,
                use_cache=True,
            )
        else:
            logger.info("========== Step 5/5: Word2Vec 流式训练 ==========")
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
