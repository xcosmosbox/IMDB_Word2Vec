from __future__ import annotations

import pandas as pd
from typing import Dict, Optional, Tuple

from .config import CONFIG
from .logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)


def _read_tsv(path, nrows: Optional[int] = None, low_memory: bool = True) -> pd.DataFrame:
    """统一的 TSV 读取入口。"""
    return pd.read_csv(path, sep="\t", low_memory=low_memory, nrows=nrows)


def _clean_known_for_titles(df: pd.DataFrame, movie_tconsts: set[str]) -> pd.DataFrame:
    """过滤代表作字段，仅保留电影集合内的条目。"""
    def _filter_row(row):
        titles = row["knownForTitles"].split(",")
        filtered = [t for t in titles if t in movie_tconsts]
        row["knownForTitles"] = ",".join(filtered)
        return row

    df = df.apply(_filter_row, axis=1)
    df = df[df["knownForTitles"] != ""]
    return df


def preprocess_all(
    tsv_paths: Dict[str, str],
    subset_rows: Optional[int] = CONFIG.data.subset_rows,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """完成 IMDb 基础表的清洗与拆分，返回关键 DataFrame。"""
    # 读取影片基本信息
    title_basics = _read_tsv(tsv_paths["title.basics.tsv.gz"], nrows=subset_rows, low_memory=False)
    title_basics = title_basics.drop(columns=["startYear", "endYear", "runtimeMinutes"])
    title_basics = title_basics.dropna(subset=["titleType", "primaryTitle", "originalTitle", "isAdult", "genres"])
    title_basics = title_basics[title_basics["titleType"] == "movie"].drop(columns=["titleType"])
    movie_tconsts = set(title_basics["tconst"])
    logger.info("影片数量: %d", len(movie_tconsts))

    # 人员信息
    name_basics = _read_tsv(tsv_paths["name.basics.tsv.gz"], nrows=subset_rows)
    name_basics = name_basics.drop(columns=["primaryName", "birthYear", "deathYear"])
    name_basics = name_basics.dropna(subset=["nconst", "primaryProfession", "knownForTitles"])
    name_basics = _clean_known_for_titles(name_basics, movie_tconsts)
    name_nconsts = set(name_basics["nconst"])
    logger.info("人员数量: %d", len(name_nconsts))

    # 别名与区域
    title_akas = _read_tsv(tsv_paths["title.akas.tsv.gz"], nrows=subset_rows)
    title_akas = title_akas.drop(columns=["ordering", "language", "attributes"])
    title_akas = title_akas.dropna(subset=["title", "titleId"])
    title_akas = title_akas[title_akas["titleId"].isin(movie_tconsts)]
    title_akas = title_akas.rename(columns={"titleId": "tconst"})

    # 剧组
    title_crew = _read_tsv(tsv_paths["title.crew.tsv.gz"], nrows=subset_rows)
    title_crew = title_crew[title_crew["tconst"].isin(movie_tconsts)]

    def _filter_person_ids(series: pd.Series) -> pd.Series:
        def _f(cell: str) -> str:
            if cell == "\\N":
                return cell
            ids = [i for i in cell.split(",") if i in name_nconsts]
            return ",".join(ids) if ids else "\\N"
        return series.apply(_f)

    title_crew["directors"] = _filter_person_ids(title_crew["directors"])
    title_crew["writers"] = _filter_person_ids(title_crew["writers"])
    title_crew = title_crew.rename(columns={"directors": "directors_nconst", "writers": "writers_nconst"})

    # 评分
    title_ratings = _read_tsv(tsv_paths["title.ratings.tsv.gz"], nrows=subset_rows)
    title_ratings = title_ratings.dropna(subset=["averageRating", "numVotes"])
    title_ratings = title_ratings[title_ratings["tconst"].isin(movie_tconsts)]

    # 演职员明细（principals）：用于补充演员/角色信息
    title_principals = _read_tsv(tsv_paths["title.principals.tsv.gz"], nrows=subset_rows)
    title_principals = title_principals[title_principals["tconst"].isin(movie_tconsts)]
    principals_cols = ["tconst", "nconst", "category", "characters"]
    title_principals = title_principals[principals_cols].fillna("\\N")

    # 剧集层级（episode）：如有与电影 tconst 对应的父标题，则保留映射
    title_episode = _read_tsv(tsv_paths["title.episode.tsv.gz"], nrows=subset_rows)
    title_episode = title_episode[title_episode["tconst"].isin(movie_tconsts)]
    title_episode = title_episode[["tconst", "parentTconst"]].fillna("\\N")

    # 构建影片信息表
    original_titles = title_akas[title_akas["isOriginalTitle"] == 1]
    movies_info = title_basics.merge(original_titles[["tconst", "title"]], on="tconst", how="left")
    movies_info = movies_info.merge(title_ratings[["tconst", "averageRating", "numVotes"]], on="tconst", how="left")

    avg_median = movies_info["averageRating"].median()
    votes_median = movies_info["numVotes"].median()
    movies_info["averageRating"].fillna(avg_median, inplace=True)
    movies_info["numVotes"].fillna(votes_median, inplace=True)
    movies_info["averageRating"] = movies_info["averageRating"].round().astype(int)
    movies_info["numVotes"] = movies_info["numVotes"].apply(lambda x: int(round(x / 10) * 10))
    movies_info[["genres1", "genres2", "genres3"]] = movies_info["genres"].str.split(",", expand=True).fillna("\\N")
    # 依据 principals 聚合每部影片的主类别（按出现频次排序取前三）
    principal_cat_agg = (
        title_principals.groupby("tconst")["category"]
        .agg(lambda s: pd.Series(s).value_counts().index.tolist())
        .reset_index()
    )
    for idx in range(3):
        col = f"principalCat{idx+1}"
        principal_cat_agg[col] = principal_cat_agg["category"].apply(lambda lst: lst[idx] if len(lst) > idx else "\\N")
    principal_cat_agg = principal_cat_agg.drop(columns=["category"])

    # 合并 episode 父标题信息
    movies_info = movies_info.merge(title_episode, on="tconst", how="left")
    movies_info["parentTconst"] = movies_info["parentTconst"].fillna("\\N").astype(str)

    # 合并 principals 类别聚合
    movies_info = movies_info.merge(principal_cat_agg, on="tconst", how="left")
    for idx in range(3):
        col = f"principalCat{idx+1}"
        if col not in movies_info:
            movies_info[col] = "\\N"
        movies_info[col] = movies_info[col].fillna("\\N").astype(str)

    movies_info = movies_info[
        [
            "tconst",
            "title",
            "genres1",
            "genres2",
            "genres3",
            "isAdult",
            "averageRating",
            "numVotes",
            "parentTconst",
            "principalCat1",
            "principalCat2",
            "principalCat3",
        ]
    ]
    movies_info["averageRating"] = movies_info["averageRating"].astype(str)
    movies_info["numVotes"] = movies_info["numVotes"].astype(str)

    # 人员特征表
    name_basics[["primaryProfession_top1", "primaryProfession_top2", "primaryProfession_top3"]] = (
        name_basics["primaryProfession"].str.split(",", expand=True).fillna("\\N")
    )
    name_basics[["knownForTitle1", "knownForTitle2", "knownForTitle3", "knownForTitle4"]] = (
        name_basics["knownForTitles"].str.split(",", expand=True).fillna("\\N")
    )
    name_basics["isDirectors"] = 0
    name_basics["isWriters"] = 0
    name_basics["isActors"] = 0  # 来自 principals 的演员标记

    directors = title_crew["directors_nconst"].str.split(",").explode().unique()
    name_basics.loc[name_basics["nconst"].isin(directors), "isDirectors"] = 1
    writers = title_crew["writers_nconst"].str.split(",").explode().unique()
    name_basics.loc[name_basics["nconst"].isin(writers), "isWriters"] = 1

    # 依据 principals 标记演员
    principal_actors = title_principals[title_principals["category"].isin(["actor", "actress"])]["nconst"].unique()
    name_basics.loc[name_basics["nconst"].isin(principal_actors), "isActors"] = 1

    staff_df = name_basics[
        [
            "nconst",
            "primaryProfession_top1",
            "primaryProfession_top2",
            "primaryProfession_top3",
            "knownForTitle1",
            "knownForTitle2",
            "knownForTitle3",
            "knownForTitle4",
            "isDirectors",
            "isWriters",
            "isActors",
        ]
    ]

    regional_titles_df = title_akas[["tconst", "title", "region", "types"]]

    # 落盘缓存
    movies_info.to_csv(CONFIG.paths.cache_dir / "movies_info_df.csv", index=False)
    staff_df.to_csv(CONFIG.paths.cache_dir / "staff_df.csv", index=False)
    regional_titles_df.to_csv(CONFIG.paths.cache_dir / "regional_titles_df.csv", index=False)
    title_crew.to_csv(CONFIG.paths.cache_dir / "title_crew_tsv_df.csv", index=False)
    title_principals.to_csv(CONFIG.paths.cache_dir / "title_principals_df.csv", index=False)
    title_episode.to_csv(CONFIG.paths.cache_dir / "title_episode_df.csv", index=False)

    logger.info("清洗完成，生成 movies_info/staff/regional_titles 数据集")
    return movies_info, staff_df, regional_titles_df

