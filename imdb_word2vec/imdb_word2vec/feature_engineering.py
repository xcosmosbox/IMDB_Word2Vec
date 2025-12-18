from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CONFIG
from .logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)
random.seed(CONFIG.random_seed)

# ========== 实体类型前缀定义 ==========
ENTITY_PREFIXES = {
    "movie": "MOVIE:",       # 电影 tconst
    "person": "PERSON:",     # 人员 nconst（通用）
    "actor": "ACTOR:",       # 演员
    "director": "DIRECTOR:", # 导演
    "writer": "WRITER:",     # 编剧
    "genre": "GENRE:",       # 类型
    "region": "REGION:",     # 地区
    "category": "ROLE:",     # 角色类别（actor, actress, director 等）
    "profession": "PROF:",   # 职业
    "series": "SERIES:",     # 系列（parentTconst）
}


def _add_prefix(value: str, prefix_type: str) -> str:
    """为实体值添加类型前缀。"""
    if value in ("\\N", "", "0", "1") or pd.isna(value):
        return str(value)  # 特殊值不加前缀
    prefix = ENTITY_PREFIXES.get(prefix_type, "")
    return f"{prefix}{value}"


def _add_prefix_to_column(df: pd.DataFrame, col_name: str, prefix_type: str) -> pd.DataFrame:
    """为 DataFrame 的指定列添加类型前缀。"""
    df = df.copy()
    df[col_name] = df[col_name].astype(str).apply(lambda x: _add_prefix(x, prefix_type))
    return df


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
    """构建词汇表，涵盖影片、人员、区域、标题以及 principals/episode 关系字段。"""
    vocab: Dict[str, int] = {"0": 0, "1": 1}
    counter = 1

    # 影片类型
    for col in tqdm(["genres1", "genres2", "genres3"], desc="注册影片类型", disable=not CONFIG.data.enable_tqdm):
        counter = _register_column(movies_info_df, col, vocab, counter)

    # 区域与别名类型
    for col in tqdm(["region", "types", "title"], desc="注册区域与别名", disable=not CONFIG.data.enable_tqdm):
        counter = _register_column(regional_titles_df, col, vocab, counter)

    # 人员职业
    for col in tqdm(
        ["primaryProfession_top1", "primaryProfession_top2", "primaryProfession_top3"],
        desc="注册人员职业",
        disable=not CONFIG.data.enable_tqdm,
    ):
        counter = _register_column(staff_df, col, vocab, counter)

    # 影片与人员标识、标题及 principals/episode 信息
    for col in tqdm(
        ["tconst", "title", "principalCat1", "principalCat2", "principalCat3", "parentTconst"],
        desc="注册影片标识/标题/主类别/父标题",
        disable=not CONFIG.data.enable_tqdm,
    ):
        counter = _register_column(movies_info_df, col, vocab, counter)
    counter = _register_column(staff_df, "nconst", vocab, counter)

    # ========== 从 principals 补充 category 和 nconst（带前缀）==========
    principals_path = CONFIG.paths.cache_dir / "title_principals_df.csv"
    if principals_path.exists():
        principals_df = pd.read_csv(principals_path)
        if "category" in principals_df.columns:
            # 添加角色类别前缀
            principals_prefixed = _add_prefix_to_column(principals_df, "category", "category")
            counter = _register_column(principals_prefixed, "category", vocab, counter)
            logger.info("从 principals 注册 category 值（带 ROLE: 前缀）")
        if "nconst" in principals_df.columns:
            # 添加人员前缀
            principals_prefixed = _add_prefix_to_column(principals_df, "nconst", "person")
            counter = _register_column(principals_prefixed, "nconst", vocab, counter)
            logger.info("从 principals 注册 nconst 值（带 PERSON: 前缀）")
        if "tconst" in principals_df.columns:
            # 添加电影前缀
            principals_prefixed = _add_prefix_to_column(principals_df, "tconst", "movie")
            counter = _register_column(principals_prefixed, "tconst", vocab, counter)
            logger.info("从 principals 注册 tconst 值（带 MOVIE: 前缀）")

    # ========== 从 episode 补充 parentTconst 和 tconst（带前缀）==========
    episode_path = CONFIG.paths.cache_dir / "title_episode_df.csv"
    if episode_path.exists():
        episode_df = pd.read_csv(episode_path)
        if "parentTconst" in episode_df.columns:
            # 添加系列前缀
            episode_prefixed = _add_prefix_to_column(episode_df, "parentTconst", "series")
            counter = _register_column(episode_prefixed, "parentTconst", vocab, counter)
            logger.info("从 episode 注册 parentTconst 值（带 SERIES: 前缀）")
        if "tconst" in episode_df.columns:
            # 添加电影前缀
            episode_prefixed = _add_prefix_to_column(episode_df, "tconst", "movie")
            counter = _register_column(episode_prefixed, "tconst", vocab, counter)
            logger.info("从 episode 注册 tconst 值（带 MOVIE: 前缀）")

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


def _build_principals_sequences(principals_path: Path, vocab: Dict[str, int]) -> pd.DataFrame:
    """从 title_principals 构建演员-电影共现序列，增强协同关系学习。

    每条记录生成一个序列：[nconst, tconst, category]
    这样 Word2Vec 可以学习到 "演员-电影-角色类型" 的共现关系。
    """
    if not principals_path.exists():
        logger.warning("title_principals_df.csv 不存在，跳过演员-电影序列生成")
        return pd.DataFrame()

    principals_df = pd.read_csv(principals_path)
    # 只保留演员/演员相关的记录，这些关系最有价值
    actor_roles = principals_df[principals_df["category"].isin(["actor", "actress", "self"])]

    if actor_roles.empty:
        logger.warning("principals 中无演员记录")
        return pd.DataFrame()

    # 构建序列：[nconst, tconst, category]（使用实体前缀）
    sequences = []
    for _, row in tqdm(actor_roles.iterrows(), total=len(actor_roles), desc="生成演员-电影序列", disable=not CONFIG.data.enable_tqdm):
        # 添加实体类型前缀以匹配词表
        nconst = _add_prefix(str(row["nconst"]), "person")
        tconst = _add_prefix(str(row["tconst"]), "movie")
        category = _add_prefix(str(row["category"]), "category")

        # 映射为整数索引
        seq = [
            vocab.get(nconst, 0),
            vocab.get(tconst, 0),
            vocab.get(category, 0),
        ]
        sequences.append(seq)

    seq_df = pd.DataFrame(sequences, columns=["nconst_id", "tconst_id", "category_id"])
    logger.info("生成演员-电影序列数：%d", len(seq_df))
    return seq_df


def _build_episode_sequences(episode_path: Path, vocab: Dict[str, int]) -> pd.DataFrame:
    """从 title_episode 构建剧集-系列共现序列，学习系列内作品关联。

    每条记录生成一个序列：[tconst(剧集), parentTconst(系列)]
    这样 Word2Vec 可以学习到 "剧集 ↔ 系列" 的归属关系。
    """
    if not episode_path.exists():
        logger.warning("title_episode_df.csv 不存在，跳过剧集-系列序列生成")
        return pd.DataFrame()

    episode_df = pd.read_csv(episode_path)
    # 过滤掉无效的父标题
    valid_episodes = episode_df[
        (episode_df["parentTconst"].notna()) &
        (episode_df["parentTconst"] != "\\N") &
        (episode_df["tconst"] != episode_df["parentTconst"])  # 排除自引用
    ]

    if valid_episodes.empty:
        logger.warning("episode 中无有效剧集-系列关系")
        return pd.DataFrame()

    # 构建序列：[tconst, parentTconst]（使用实体前缀）
    sequences = []
    for _, row in tqdm(valid_episodes.iterrows(), total=len(valid_episodes), desc="生成剧集-系列序列", disable=not CONFIG.data.enable_tqdm):
        # 添加实体类型前缀以匹配词表
        tconst = _add_prefix(str(row["tconst"]), "movie")
        parent_tconst = _add_prefix(str(row["parentTconst"]), "series")

        # 映射为整数索引
        seq = [
            vocab.get(tconst, 0),
            vocab.get(parent_tconst, 0),
        ]
        sequences.append(seq)

    seq_df = pd.DataFrame(sequences, columns=["tconst_id", "parentTconst_id"])
    logger.info("生成剧集-系列序列数：%d", len(seq_df))
    return seq_df


def _pad_and_align_sequences(seq_df: pd.DataFrame, target_columns: pd.Index) -> pd.DataFrame:
    """将序列 DataFrame 填充并对齐到目标列结构。"""
    if seq_df.empty:
        return pd.DataFrame()

    n_target_cols = len(target_columns)
    padded_df = seq_df.copy()

    # 填充到目标列数
    for i in range(seq_df.shape[1], n_target_cols):
        padded_df[f"pad_{i}"] = 0

    # 重命名列以匹配目标
    padded_df.columns = target_columns[:len(padded_df.columns)].tolist() + list(padded_df.columns[len(target_columns):])

    # 确保所有目标列都存在
    for col in target_columns:
        if col not in padded_df.columns:
            padded_df[col] = 0

    # 按目标列顺序排列
    padded_df = padded_df[target_columns]
    return padded_df


def _apply_entity_prefixes(
    movies_info_df: pd.DataFrame,
    staff_df: pd.DataFrame,
    regional_titles_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """为所有实体列添加类型前缀，增强语义区分度。"""
    logger.info("正在添加实体类型前缀...")

    # ========== 电影信息表 ==========
    movies = movies_info_df.copy()
    # 电影 ID
    movies = _add_prefix_to_column(movies, "tconst", "movie")
    # 类型
    for col in ["genres1", "genres2", "genres3"]:
        if col in movies.columns:
            movies = _add_prefix_to_column(movies, col, "genre")
    # 主要演职类别
    for col in ["principalCat1", "principalCat2", "principalCat3"]:
        if col in movies.columns:
            movies = _add_prefix_to_column(movies, col, "category")
    # 父系列
    if "parentTconst" in movies.columns:
        movies = _add_prefix_to_column(movies, "parentTconst", "series")

    # ========== 人员信息表 ==========
    staff = staff_df.copy()
    # 人员 ID
    staff = _add_prefix_to_column(staff, "nconst", "person")
    # 职业
    for col in ["primaryProfession_top1", "primaryProfession_top2", "primaryProfession_top3"]:
        if col in staff.columns:
            staff = _add_prefix_to_column(staff, col, "profession")
    # 代表作（电影 ID）
    for col in ["knownForTitle1", "knownForTitle2", "knownForTitle3", "knownForTitle4"]:
        if col in staff.columns:
            staff = _add_prefix_to_column(staff, col, "movie")

    # ========== 区域别名表 ==========
    regional = regional_titles_df.copy()
    regional = _add_prefix_to_column(regional, "tconst", "movie")
    if "region" in regional.columns:
        regional = _add_prefix_to_column(regional, "region", "region")

    logger.info("实体类型前缀添加完成")
    return movies, staff, regional


def run_feature_engineering(
    movies_info_df: pd.DataFrame,
    staff_df: pd.DataFrame,
    regional_titles_df: pd.DataFrame,
) -> Tuple[Path, Path]:
    """执行特征工程全流程：实体前缀、独热、融合、词表与向量化。"""

    # ========== 0. 添加实体类型前缀 ==========
    movies_info_prefixed, staff_prefixed, regional_prefixed = _apply_entity_prefixes(
        movies_info_df, staff_df, regional_titles_df
    )

    movies_info_regional = merge_regional_features(movies_info_prefixed, regional_prefixed)
    merged_df = merge_staff_movies(movies_info_regional, staff_prefixed)

    vocab = build_vocab(movies_info_regional, staff_prefixed, regional_prefixed)
    mapped_df = vectorize_dataframe(merged_df.copy(), vocab)

    base_row_count = len(mapped_df)
    logger.info("基础训练数据行数：%d", base_row_count)

    # ========== 1. 生成演员-电影共现序列并追加 ==========
    principals_path = CONFIG.paths.cache_dir / "title_principals_df.csv"
    principals_seq_df = _build_principals_sequences(principals_path, vocab)

    if not principals_seq_df.empty:
        principals_padded = _pad_and_align_sequences(principals_seq_df, mapped_df.columns)
        mapped_df = pd.concat([mapped_df, principals_padded], ignore_index=True)
        logger.info("追加演员-电影序列后总行数：%d (+%d)", len(mapped_df), len(principals_padded))

    # ========== 2. 生成剧集-系列共现序列并追加 ==========
    episode_path = CONFIG.paths.cache_dir / "title_episode_df.csv"
    episode_seq_df = _build_episode_sequences(episode_path, vocab)

    if not episode_seq_df.empty:
        episode_padded = _pad_and_align_sequences(episode_seq_df, mapped_df.columns)
        mapped_df = pd.concat([mapped_df, episode_padded], ignore_index=True)
        logger.info("追加剧集-系列序列后总行数：%d (+%d)", len(mapped_df), len(episode_padded))

    # ========== 汇总统计 ==========
    total_added = len(mapped_df) - base_row_count
    logger.info(
        "特征工程完成：基础数据 %d 行 + 关系序列 %d 行 = 总计 %d 行",
        base_row_count, total_added, len(mapped_df)
    )

    final_mapped_path = CONFIG.paths.final_mapped_path
    mapped_df.to_csv(final_mapped_path, index=False)
    logger.info("向量化表已保存：%s", final_mapped_path)

    return final_mapped_path, CONFIG.paths.vocab_path


