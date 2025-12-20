from __future__ import annotations

import pandas as pd
from typing import Dict, Optional, Tuple

from tqdm import tqdm

from .config import CONFIG
from .logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)

# ========== 缺失值占位符 ==========
PLACEHOLDER = "UNKNOWN"  # 统一的留空词


def _read_tsv(path, nrows: Optional[int] = None, low_memory: bool = True) -> pd.DataFrame:
    """统一的 TSV 读取入口。"""
    return pd.read_csv(path, sep="\t", low_memory=low_memory, nrows=nrows)


def _year_to_era(year) -> str:
    """将年份转换为年代标签。"""
    try:
        year = int(float(year))
        if year < 1920:
            return "ERA_SILENT"      # 默片时代
        elif year < 1930:
            return "ERA_1920s"
        elif year < 1940:
            return "ERA_1930s"
        elif year < 1950:
            return "ERA_1940s"
        elif year < 1960:
            return "ERA_1950s"
        elif year < 1970:
            return "ERA_1960s"
        elif year < 1980:
            return "ERA_1970s"
        elif year < 1990:
            return "ERA_1980s"
        elif year < 2000:
            return "ERA_1990s"
        elif year < 2010:
            return "ERA_2000s"
        elif year < 2020:
            return "ERA_2010s"
        else:
            return "ERA_2020s"
    except (ValueError, TypeError):
        return "ERA_UNKNOWN"


def _clean_known_for_titles(df: pd.DataFrame, valid_tconsts: set[str]) -> pd.DataFrame:
    """过滤代表作字段，仅保留有效作品集合内的条目。"""
    def _filter_row(row):
        titles_str = str(row["knownForTitles"])
        if titles_str == "\\N" or pd.isna(titles_str):
            return row
        titles = titles_str.split(",")
        filtered = [t for t in titles if t in valid_tconsts]
        row["knownForTitles"] = ",".join(filtered) if filtered else "\\N"
        return row

    if CONFIG.data.enable_tqdm:
        tqdm.pandas(desc="过滤代表作字段")
        df = df.progress_apply(_filter_row, axis=1)
    else:
        df = df.apply(_filter_row, axis=1)
    # 不再删除空代表作的行，保留所有人员
    return df


def preprocess_all(
    tsv_paths: Dict[str, str],
    subset_rows: Optional[int] = CONFIG.data.subset_rows,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """完成 IMDb 基础表的清洗与拆分，返回关键 DataFrame。
    
    改进点：
    1. 保留年代信息 (startYear)
    2. 保留电视剧、短片等所有类型
    3. 保留早期电影，缺失信息用 UNKNOWN 填充
    4. 保留 primaryName
    5. 评分保留一位小数
    6. 添加年代分类
    """
    step_bar = tqdm(total=8, desc="预处理阶段", disable=not CONFIG.data.enable_tqdm)

    # ========== 1. 读取影片基本信息 ==========
    title_basics = _read_tsv(tsv_paths["title.basics.tsv.gz"], nrows=subset_rows, low_memory=False)
    
    # 保留年份信息，不再删除
    # 只删除 endYear 和 runtimeMinutes（使用较少）
    if "endYear" in title_basics.columns:
        title_basics = title_basics.drop(columns=["endYear"])
    if "runtimeMinutes" in title_basics.columns:
        title_basics = title_basics.drop(columns=["runtimeMinutes"])
    
    # 用占位符填充缺失值，而不是删除
    title_basics["primaryTitle"] = title_basics["primaryTitle"].fillna(PLACEHOLDER)
    title_basics["originalTitle"] = title_basics["originalTitle"].fillna(PLACEHOLDER)
    title_basics["isAdult"] = title_basics["isAdult"].fillna(0).astype(int)
    title_basics["genres"] = title_basics["genres"].fillna(PLACEHOLDER)
    title_basics["startYear"] = title_basics["startYear"].fillna("\\N")
    title_basics["titleType"] = title_basics["titleType"].fillna(PLACEHOLDER)
    
    # 保留所有类型：movie, tvSeries, short, tvMovie, tvMiniSeries 等
    # 只排除明显无关的类型
    exclude_types = ["videoGame", "tvEpisode"]  # 游戏和单集不需要
    title_basics = title_basics[~title_basics["titleType"].isin(exclude_types)]
    
    # 添加年代标签
    title_basics["era"] = title_basics["startYear"].apply(_year_to_era)
    
    valid_tconsts = set(title_basics["tconst"])
    logger.info("作品数量: %d (含电影、电视剧、短片等)", len(valid_tconsts))
    step_bar.update(1)

    # ========== 2. 人员信息 ==========
    name_basics = _read_tsv(tsv_paths["name.basics.tsv.gz"], nrows=subset_rows)
    
    # 保留 primaryName！
    # 用占位符填充缺失值
    name_basics["primaryName"] = name_basics["primaryName"].fillna(PLACEHOLDER)
    name_basics["primaryProfession"] = name_basics["primaryProfession"].fillna(PLACEHOLDER)
    name_basics["knownForTitles"] = name_basics["knownForTitles"].fillna("\\N")
    name_basics["birthYear"] = name_basics["birthYear"].fillna("\\N")
    name_basics["deathYear"] = name_basics["deathYear"].fillna("\\N")
    
    # 添加出生年代
    name_basics["birthEra"] = name_basics["birthYear"].apply(_year_to_era)
    
    # 过滤代表作（保留所有人员，即使代表作为空）
    name_basics = _clean_known_for_titles(name_basics, valid_tconsts)
    name_nconsts = set(name_basics["nconst"])
    logger.info("人员数量: %d", len(name_nconsts))
    step_bar.update(1)

    # ========== 3. 别名与区域 ==========
    title_akas = _read_tsv(tsv_paths["title.akas.tsv.gz"], nrows=subset_rows)
    if "ordering" in title_akas.columns:
        title_akas = title_akas.drop(columns=["ordering"])
    if "language" in title_akas.columns:
        title_akas = title_akas.drop(columns=["language"])
    if "attributes" in title_akas.columns:
        title_akas = title_akas.drop(columns=["attributes"])
    
    title_akas["title"] = title_akas["title"].fillna(PLACEHOLDER)
    title_akas["titleId"] = title_akas["titleId"].fillna("\\N")
    title_akas = title_akas[title_akas["titleId"].isin(valid_tconsts)]
    title_akas = title_akas.rename(columns={"titleId": "tconst"})
    step_bar.update(1)

    # ========== 4. 剧组 ==========
    title_crew = _read_tsv(tsv_paths["title.crew.tsv.gz"], nrows=subset_rows)
    title_crew = title_crew[title_crew["tconst"].isin(valid_tconsts)]
    title_crew["directors"] = title_crew["directors"].fillna("\\N")
    title_crew["writers"] = title_crew["writers"].fillna("\\N")

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

    # ========== 5. 评分 - 保留一位小数 ==========
    title_ratings = _read_tsv(tsv_paths["title.ratings.tsv.gz"], nrows=subset_rows)
    title_ratings["averageRating"] = title_ratings["averageRating"].fillna(0.0)
    title_ratings["numVotes"] = title_ratings["numVotes"].fillna(0)
    title_ratings = title_ratings[title_ratings["tconst"].isin(valid_tconsts)]
    step_bar.update(1)

    # ========== 6. 演职员明细 ==========
    title_principals = _read_tsv(tsv_paths["title.principals.tsv.gz"], nrows=subset_rows)
    title_principals = title_principals[title_principals["tconst"].isin(valid_tconsts)]
    principals_cols = ["tconst", "nconst", "category", "characters", "ordering"]
    available_cols = [c for c in principals_cols if c in title_principals.columns]
    title_principals = title_principals[available_cols].fillna("\\N")
    step_bar.update(1)

    # ========== 7. 剧集层级 ==========
    title_episode = _read_tsv(tsv_paths["title.episode.tsv.gz"], nrows=subset_rows)
    # 不再只保留电影，保留所有有效作品
    title_episode = title_episode[
        (title_episode["tconst"].isin(valid_tconsts)) | 
        (title_episode["parentTconst"].isin(valid_tconsts))
    ]
    title_episode = title_episode[["tconst", "parentTconst"]].fillna("\\N")
    step_bar.update(1)

    # ========== 8. 构建影片信息表 ==========
    original_titles = title_akas[title_akas.get("isOriginalTitle", 0) == 1]
    movies_info = title_basics.merge(
        original_titles[["tconst", "title"]], on="tconst", how="left"
    )
    # 如果没有原始标题，使用 primaryTitle
    movies_info["title"] = movies_info["title"].fillna(movies_info["primaryTitle"])
    
    movies_info = movies_info.merge(
        title_ratings[["tconst", "averageRating", "numVotes"]], on="tconst", how="left"
    )

    # 评分保留一位小数
    avg_median = movies_info["averageRating"].median()
    votes_median = movies_info["numVotes"].median()
    movies_info = movies_info.assign(
        averageRating=movies_info["averageRating"].fillna(avg_median).round(1),  # 保留一位小数
        numVotes=movies_info["numVotes"].fillna(votes_median).astype(int),
    )
    
    # 拆分类型（最多保留5个）
    genres_split = movies_info["genres"].str.split(",", expand=True)
    for i in range(5):
        col = f"genres{i+1}"
        if i < genres_split.shape[1]:
            movies_info[col] = genres_split[i].fillna("\\N")
        else:
            movies_info[col] = "\\N"
    
    # 依据 principals 聚合每部影片的所有角色类别
    principal_cat_agg = (
        title_principals.groupby("tconst")["category"]
        .agg(lambda s: pd.Series(s).value_counts().index.tolist())
        .reset_index()
    )
    # 保留更多角色类别（最多10个）
    for idx in range(10):
        col = f"principalCat{idx+1}"
        principal_cat_agg[col] = principal_cat_agg["category"].apply(
            lambda lst: lst[idx] if len(lst) > idx else "\\N"
        )
    principal_cat_agg = principal_cat_agg.drop(columns=["category"])

    # 合并 episode 父标题信息
    movies_info = movies_info.merge(title_episode, on="tconst", how="left")
    movies_info["parentTconst"] = movies_info["parentTconst"].fillna("\\N").astype(str)

    # 合并 principals 类别聚合
    movies_info = movies_info.merge(principal_cat_agg, on="tconst", how="left")
    for idx in range(10):
        col = f"principalCat{idx+1}"
        if col not in movies_info:
            movies_info[col] = "\\N"
        movies_info[col] = movies_info[col].fillna("\\N").astype(str)

    # 最终列选择
    movies_info_cols = [
        "tconst", "titleType", "title", "primaryTitle",
        "genres1", "genres2", "genres3", "genres4", "genres5",
        "isAdult", "startYear", "era",
        "averageRating", "numVotes",
        "parentTconst",
    ] + [f"principalCat{i+1}" for i in range(10)]
    
    movies_info = movies_info[[c for c in movies_info_cols if c in movies_info.columns]]

    # ========== 9. 人员特征表 ==========
    # 拆分职业
    prof_split = name_basics["primaryProfession"].str.split(",", expand=True)
    for i in range(5):
        col = f"primaryProfession_{i+1}"
        if i < prof_split.shape[1]:
            name_basics[col] = prof_split[i].fillna("\\N")
        else:
            name_basics[col] = "\\N"
    
    # 拆分代表作（最多10部）
    titles_split = name_basics["knownForTitles"].str.split(",", expand=True)
    for i in range(10):
        col = f"knownForTitle{i+1}"
        if i < titles_split.shape[1]:
            name_basics[col] = titles_split[i].fillna("\\N")
        else:
            name_basics[col] = "\\N"
    
    # 标记导演/编剧/演员
    name_basics["isDirectors"] = 0
    name_basics["isWriters"] = 0
    name_basics["isActors"] = 0

    directors = title_crew["directors_nconst"].str.split(",").explode().unique()
    name_basics.loc[name_basics["nconst"].isin(directors), "isDirectors"] = 1
    writers = title_crew["writers_nconst"].str.split(",").explode().unique()
    name_basics.loc[name_basics["nconst"].isin(writers), "isWriters"] = 1

    principal_actors = title_principals[
        title_principals["category"].isin(["actor", "actress", "self"])
    ]["nconst"].unique()
    name_basics.loc[name_basics["nconst"].isin(principal_actors), "isActors"] = 1

    staff_df_cols = (
        ["nconst", "primaryName", "birthYear", "deathYear", "birthEra"]
        + [f"primaryProfession_{i+1}" for i in range(5)]
        + [f"knownForTitle{i+1}" for i in range(10)]
        + ["isDirectors", "isWriters", "isActors"]
    )
    staff_df = name_basics[[c for c in staff_df_cols if c in name_basics.columns]]

    regional_titles_df = title_akas[["tconst", "title", "region", "types"]].copy()
    regional_titles_df = regional_titles_df.fillna("\\N")

    # ========== 10. 落盘缓存 ==========
    movies_info.to_csv(CONFIG.paths.cache_dir / "movies_info_df.csv", index=False)
    staff_df.to_csv(CONFIG.paths.cache_dir / "staff_df.csv", index=False)
    regional_titles_df.to_csv(CONFIG.paths.cache_dir / "regional_titles_df.csv", index=False)
    title_crew.to_csv(CONFIG.paths.cache_dir / "title_crew_tsv_df.csv", index=False)
    title_principals.to_csv(CONFIG.paths.cache_dir / "title_principals_df.csv", index=False)
    title_episode.to_csv(CONFIG.paths.cache_dir / "title_episode_df.csv", index=False)
    step_bar.update(1)
    step_bar.close()

    logger.info("清洗完成，生成 movies_info/staff/regional_titles 数据集")
    logger.info("  - movies_info: %d 行", len(movies_info))
    logger.info("  - staff_df: %d 行", len(staff_df))
    logger.info("  - regional_titles_df: %d 行", len(regional_titles_df))
    
    return movies_info, staff_df, regional_titles_df
