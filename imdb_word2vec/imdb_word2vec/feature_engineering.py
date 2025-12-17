from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG
from .logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)
random.seed(CONFIG.random_seed)


def _register_column(df: pd.DataFrame, col_name: str, vocab: Dict[str, int], counter: int) -> int:
    """对单列数据进行随机顺序注册，减少顺序性偏差。"""
    values = df[col_name].astype(str).tolist()
    random.shuffle(values)
    for item in values:
        if item not in vocab:
            counter += 1
            vocab[item] = counter
    return counter


def build_vocab(
    movies_info_df: pd.DataFrame, staff_df: pd.DataFrame, regional_titles_df: pd.DataFrame
) -> Dict[str, int]:
    """构建词汇表，涵盖影片、人员、区域与标题等离散字段。"""
    vocab: Dict[str, int] = {"0": 0, "1": 1}
    counter = 1

    # 影片类型
    for col in ["genres1", "genres2", "genres3"]:
        counter = _register_column(movies_info_df, col, vocab, counter)

    # 区域与别名类型
    for col in ["region", "types", "title"]:
        counter = _register_column(regional_titles_df, col, vocab, counter)

    # 人员职业
    for col in ["primaryProfession_top1", "primaryProfession_top2", "primaryProfession_top3"]:
        counter = _register_column(staff_df, col, vocab, counter)

    # 影片与人员标识、标题及 principals/episode 信息
    counter = _register_column(movies_info_df, "tconst", vocab, counter)
    counter = _register_column(staff_df, "nconst", vocab, counter)
    counter = _register_column(movies_info_df, "title", vocab, counter)
    for col in ["principalCat1", "principalCat2", "principalCat3", "parentTconst"]:
        counter = _register_column(movies_info_df, col, vocab, counter)

    logger.info("词汇表规模：%d", len(vocab))
    vocab_path = CONFIG.paths.vocab_path
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(vocab).to_csv(vocab_path, header=False)
    logger.info("词汇表已保存：%s", vocab_path)
    return vocab


def _region_type_one_hot(regional_titles_df: pd.DataFrame) -> pd.DataFrame:
    """对区域与类型字段进行独热编码并按影片聚合。"""
    df = regional_titles_df.copy()
    df["region"] = df["region"].fillna("\\N").astype(str)
    df["types"] = df["types"].fillna("\\N").astype(str)

    region_dummies = pd.get_dummies(df["region"], prefix="region_class", dtype=int)
    type_dummies = pd.get_dummies(df["types"], prefix="movie_type", dtype=int)

    region_agg = pd.concat([df[["tconst"]], region_dummies], axis=1).groupby("tconst").max()
    type_agg = pd.concat([df[["tconst"]], type_dummies], axis=1).groupby("tconst").max()

    combined = region_agg.join(type_agg, how="outer").reset_index().fillna(0)
    # 仅对独热列转为整数，保留 tconst 为字符串
    for col in combined.columns:
        if col != "tconst":
            combined[col] = combined[col].astype(int)
    return combined


def merge_regional_features(
    movies_info_df: pd.DataFrame, regional_titles_df: pd.DataFrame
) -> pd.DataFrame:
    """为影片添加区域与类型独热特征。"""
    combined = _region_type_one_hot(regional_titles_df)
    merged = movies_info_df.merge(combined, on="tconst", how="left").fillna(0)
    logger.info("区域与类型特征列数：%d", combined.shape[1] - 1)
    return merged


def merge_staff_movies(
    movies_info_regional_df: pd.DataFrame, staff_df: pd.DataFrame
) -> pd.DataFrame:
    """将人员代表作与影片特征表关联，得到最终训练表。"""
    staff_melted = staff_df.melt(
        id_vars=["nconst"],
        value_vars=["knownForTitle1", "knownForTitle2", "knownForTitle3", "knownForTitle4"],
        var_name="knownForTitleNumber",
        value_name="knownForTitle",
    )
    staff_melted = staff_melted.dropna(subset=["knownForTitle"])
    staff_expanded = staff_melted[["nconst", "knownForTitle"]]

    merged_df = staff_expanded.merge(
        movies_info_regional_df, left_on="knownForTitle", right_on="tconst", how="inner"
    )
    merged_df = merged_df.drop(columns=["knownForTitle"])
    logger.info("人员-影片融合表行数：%d", len(merged_df))
    return merged_df


def vectorize_dataframe(df: pd.DataFrame, vocab: Dict[str, int]) -> pd.DataFrame:
    """将非整数列映射为整数索引，便于后续模型训练。"""
    for col in df.columns:
        if not pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(str).map(lambda x: vocab.get(x, x))
    return df


def run_feature_engineering(
    movies_info_df: pd.DataFrame,
    staff_df: pd.DataFrame,
    regional_titles_df: pd.DataFrame,
) -> Tuple[Path, Path]:
    """执行特征工程全流程：独热、融合、词表与向量化。"""
    movies_info_regional = merge_regional_features(movies_info_df, regional_titles_df)
    merged_df = merge_staff_movies(movies_info_regional, staff_df)

    vocab = build_vocab(movies_info_regional, staff_df, regional_titles_df)
    mapped_df = vectorize_dataframe(merged_df.copy(), vocab)

    final_mapped_path = CONFIG.paths.final_mapped_path
    mapped_df.to_csv(final_mapped_path, index=False)
    logger.info("向量化表已保存：%s", final_mapped_path)

    return final_mapped_path, CONFIG.paths.vocab_path


