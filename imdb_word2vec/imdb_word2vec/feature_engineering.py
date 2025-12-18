from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CONFIG
from .logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)
random.seed(CONFIG.random_seed)

# ========== 实体类型前缀定义 ==========
ENTITY_PREFIXES = {
    "movie": "MOV_",        # 电影 tconst
    "person": "PER_",       # 人员 nconst（通用）
    "actor": "ACT_",        # 演员
    "director": "DIR_",     # 导演
    "writer": "WRI_",       # 编剧
    "genre": "GEN_",        # 类型
    "region": "REG_",       # 地区
    "category": "ROL_",     # 角色类别（actor, actress, director 等）
    "profession": "PRO_",   # 职业
    "series": "SER_",       # 系列（parentTconst）
    "rating": "RAT_",       # 评分等级
    "year": "YEA_",         # 年份（如果需要）
}


def _add_prefix(value: str, prefix_type: str) -> str:
    """为实体值添加类型前缀。"""
    if value in ("\\N", "", "nan", "None") or pd.isna(value):
        return None  # 返回 None 以便后续过滤
    prefix = ENTITY_PREFIXES.get(prefix_type, "")
    return f"{prefix}{value}"


# ========== 序列生成器 ==========

def _generate_person_movie_sequences(
    staff_df: pd.DataFrame,
    title_crew_df: pd.DataFrame,
    title_principals_df: pd.DataFrame,
) -> List[List[str]]:
    """
    生成 人员 → 电影 序列。
    
    序列形式: [PERSON:nm001, MOVIE:tt001, MOVIE:tt002, MOVIE:tt003, ...]
    每个人的所有作品形成一个序列，让 Word2Vec 学习到 "同一个人参与的电影是相关的"。
    """
    sequences = []
    
    # 1. 从 staff_df 的 knownForTitles 提取
    logger.info("从 staff_df 提取人员-代表作序列...")
    for _, row in tqdm(staff_df.iterrows(), total=len(staff_df), desc="人员代表作序列"):
        person = _add_prefix(str(row["nconst"]), "person")
        if person is None:
            continue
        
        movies = []
        for col in ["knownForTitle1", "knownForTitle2", "knownForTitle3", "knownForTitle4"]:
            if col in row and pd.notna(row[col]) and row[col] != "\\N":
                movie = _add_prefix(str(row[col]), "movie")
                if movie:
                    movies.append(movie)
        
        if len(movies) >= 1:
            seq = [person] + movies
            sequences.append(seq)
    
    # 2. 从 title_principals 提取更详细的演员-电影关系
    if title_principals_df is not None and not title_principals_df.empty:
        logger.info("从 principals 提取演员-电影序列...")
        # 按人员聚合所有参演电影
        person_movies = title_principals_df.groupby("nconst")["tconst"].apply(list).to_dict()
        
        for nconst, tconsts in tqdm(person_movies.items(), desc="演员参演序列"):
            person = _add_prefix(str(nconst), "person")
            if person is None:
                continue
            
            movies = [_add_prefix(str(t), "movie") for t in tconsts if t != "\\N"]
            movies = [m for m in movies if m is not None]
            
            if len(movies) >= 2:  # 至少 2 部电影才有共现意义
                seq = [person] + movies
                sequences.append(seq)
    
    # 3. 从 title_crew 提取导演/编剧-电影关系
    if title_crew_df is not None and not title_crew_df.empty:
        logger.info("从 crew 提取导演/编剧序列...")
        
        # 导演
        director_movies = {}
        for _, row in title_crew_df.iterrows():
            tconst = str(row["tconst"])
            directors_str = str(row.get("directors_nconst", "\\N"))
            if directors_str != "\\N":
                for d in directors_str.split(","):
                    if d and d != "\\N":
                        if d not in director_movies:
                            director_movies[d] = []
                        director_movies[d].append(tconst)
        
        for nconst, tconsts in director_movies.items():
            person = _add_prefix(nconst, "director")
            if person is None:
                continue
            movies = [_add_prefix(t, "movie") for t in tconsts]
            movies = [m for m in movies if m is not None]
            if len(movies) >= 2:
                seq = [person] + movies
                sequences.append(seq)
        
        # 编剧
        writer_movies = {}
        for _, row in title_crew_df.iterrows():
            tconst = str(row["tconst"])
            writers_str = str(row.get("writers_nconst", "\\N"))
            if writers_str != "\\N":
                for w in writers_str.split(","):
                    if w and w != "\\N":
                        if w not in writer_movies:
                            writer_movies[w] = []
                        writer_movies[w].append(tconst)
        
        for nconst, tconsts in writer_movies.items():
            person = _add_prefix(nconst, "writer")
            if person is None:
                continue
            movies = [_add_prefix(t, "movie") for t in tconsts]
            movies = [m for m in movies if m is not None]
            if len(movies) >= 2:
                seq = [person] + movies
                sequences.append(seq)
    
    logger.info("人员-电影序列数: %d", len(sequences))
    return sequences


def _generate_movie_context_sequences(
    movies_info_df: pd.DataFrame,
    title_principals_df: pd.DataFrame,
    title_crew_df: pd.DataFrame,
) -> List[List[str]]:
    """
    生成 电影 → 上下文 序列。
    
    序列形式: [MOVIE:tt001, GENRE:Action, GENRE:Drama, DIRECTOR:nm001, ACTOR:nm002, ...]
    让 Word2Vec 学习到 "电影的类型、导演、演员是相关的"。
    """
    sequences = []
    
    # 预处理 principals 和 crew 为字典便于查询
    movie_actors = {}
    movie_directors = {}
    
    if title_principals_df is not None and not title_principals_df.empty:
        for _, row in title_principals_df.iterrows():
            tconst = str(row["tconst"])
            nconst = str(row["nconst"])
            category = str(row.get("category", ""))
            
            if tconst not in movie_actors:
                movie_actors[tconst] = []
            
            if category in ("actor", "actress", "self"):
                movie_actors[tconst].append(nconst)
    
    if title_crew_df is not None and not title_crew_df.empty:
        for _, row in title_crew_df.iterrows():
            tconst = str(row["tconst"])
            directors_str = str(row.get("directors_nconst", "\\N"))
            if directors_str != "\\N":
                movie_directors[tconst] = [d for d in directors_str.split(",") if d and d != "\\N"]
    
    logger.info("生成电影-上下文序列...")
    for _, row in tqdm(movies_info_df.iterrows(), total=len(movies_info_df), desc="电影上下文序列"):
        tconst = str(row["tconst"])
        movie = _add_prefix(tconst, "movie")
        if movie is None:
            continue
        
        context = [movie]
        
        # 添加类型
        for col in ["genres1", "genres2", "genres3"]:
            if col in row and pd.notna(row[col]) and row[col] != "\\N":
                genre = _add_prefix(str(row[col]), "genre")
                if genre:
                    context.append(genre)
        
        # 添加评分等级（离散化）
        if "averageRating" in row:
            try:
                rating = int(float(row["averageRating"]))
                rating_token = _add_prefix(str(rating), "rating")
                if rating_token:
                    context.append(rating_token)
            except (ValueError, TypeError):
                pass
        
        # 添加导演
        if tconst in movie_directors:
            for d in movie_directors[tconst][:3]:  # 最多 3 个导演
                director = _add_prefix(d, "director")
                if director:
                    context.append(director)
        
        # 添加主要演员
        if tconst in movie_actors:
            for a in movie_actors[tconst][:5]:  # 最多 5 个演员
                actor = _add_prefix(a, "actor")
                if actor:
                    context.append(actor)
        
        if len(context) >= 3:  # 电影 + 至少 2 个上下文
            sequences.append(context)
    
    logger.info("电影-上下文序列数: %d", len(sequences))
    return sequences


def _generate_series_episode_sequences(
    title_episode_df: pd.DataFrame,
) -> List[List[str]]:
    """
    生成 系列 → 剧集 序列。
    
    序列形式: [SERIES:tt100, MOVIE:tt101, MOVIE:tt102, MOVIE:tt103]
    让 Word2Vec 学习到 "同一系列的作品是相关的"。
    """
    sequences = []
    
    if title_episode_df is None or title_episode_df.empty:
        logger.warning("无 episode 数据，跳过系列-剧集序列")
        return sequences
    
    # 按系列聚合所有剧集
    series_episodes = title_episode_df.groupby("parentTconst")["tconst"].apply(list).to_dict()
    
    logger.info("生成系列-剧集序列...")
    for parent_tconst, tconsts in tqdm(series_episodes.items(), desc="系列剧集序列"):
        if parent_tconst == "\\N" or pd.isna(parent_tconst):
            continue
        
        series = _add_prefix(str(parent_tconst), "series")
        if series is None:
            continue
        
        episodes = [_add_prefix(str(t), "movie") for t in tconsts]
        episodes = [e for e in episodes if e is not None]
        
        if len(episodes) >= 2:  # 至少 2 集才有意义
            seq = [series] + episodes
            sequences.append(seq)
    
    logger.info("系列-剧集序列数: %d", len(sequences))
    return sequences


def _generate_coactor_sequences(
    title_principals_df: pd.DataFrame,
    min_coactors: int = 2,
    max_coactors: int = 10,
) -> List[List[str]]:
    """
    生成 合作演员 序列。
    
    序列形式: [ACTOR:nm001, ACTOR:nm002, ACTOR:nm003, ...]
    同一部电影的演员放在一起，让 Word2Vec 学习到 "经常合作的演员"。
    """
    sequences = []
    
    if title_principals_df is None or title_principals_df.empty:
        logger.warning("无 principals 数据，跳过合作演员序列")
        return sequences
    
    # 按电影聚合演员
    movie_cast = (
        title_principals_df[title_principals_df["category"].isin(["actor", "actress", "self"])]
        .groupby("tconst")["nconst"]
        .apply(list)
        .to_dict()
    )
    
    logger.info("生成合作演员序列...")
    for tconst, actors in tqdm(movie_cast.items(), desc="合作演员序列"):
        if len(actors) < min_coactors:
            continue
        
        # 限制演员数量
        actors = actors[:max_coactors]
        
        actor_tokens = [_add_prefix(str(a), "actor") for a in actors]
        actor_tokens = [t for t in actor_tokens if t is not None]
        
        if len(actor_tokens) >= min_coactors:
            sequences.append(actor_tokens)
    
    logger.info("合作演员序列数: %d", len(sequences))
    return sequences


def _generate_genre_movie_sequences(
    movies_info_df: pd.DataFrame,
    max_movies_per_genre: int = 100,
) -> List[List[str]]:
    """
    生成 类型 → 电影 序列。
    
    序列形式: [GENRE:Action, MOVIE:tt001, MOVIE:tt002, ...]
    同类型的电影放在一起，让 Word2Vec 学习到 "同类型电影是相关的"。
    """
    sequences = []
    
    # 收集每个类型下的电影
    genre_movies = {}
    
    for _, row in movies_info_df.iterrows():
        tconst = str(row["tconst"])
        for col in ["genres1", "genres2", "genres3"]:
            if col in row and pd.notna(row[col]) and row[col] != "\\N":
                genre = str(row[col])
                if genre not in genre_movies:
                    genre_movies[genre] = []
                genre_movies[genre].append(tconst)
    
    logger.info("生成类型-电影序列...")
    for genre, tconsts in tqdm(genre_movies.items(), desc="类型电影序列"):
        genre_token = _add_prefix(genre, "genre")
        if genre_token is None:
            continue
        
        # 随机采样避免序列过长
        if len(tconsts) > max_movies_per_genre:
            tconsts = random.sample(tconsts, max_movies_per_genre)
        
        movies = [_add_prefix(t, "movie") for t in tconsts]
        movies = [m for m in movies if m is not None]
        
        if len(movies) >= 5:  # 至少 5 部电影
            seq = [genre_token] + movies
            sequences.append(seq)
    
    logger.info("类型-电影序列数: %d", len(sequences))
    return sequences


# ========== 词表构建 ==========

def build_vocab_from_sequences(sequences: List[List[str]]) -> Dict[str, int]:
    """从序列中构建词汇表。"""
    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    counter = 1
    
    all_tokens = set()
    for seq in sequences:
        all_tokens.update(seq)
    
    # 随机打乱以避免顺序偏差
    all_tokens_list = list(all_tokens)
    random.shuffle(all_tokens_list)
    
    for token in tqdm(all_tokens_list, desc="构建词汇表"):
        if token not in vocab:
            counter += 1
            vocab[token] = counter
    
    logger.info("词汇表规模: %d", len(vocab))
    return vocab


def save_sequences_to_csv(
    sequences: List[List[str]],
    vocab: Dict[str, int],
    output_path: Path,
    max_seq_len: int = 50,
) -> None:
    """
    将序列保存为 CSV 文件。
    
    每行是一个序列，用整数 ID 表示，短序列用 0 填充。
    """
    # 转换为整数 ID
    int_sequences = []
    for seq in tqdm(sequences, desc="转换序列为整数"):
        int_seq = [vocab.get(token, 1) for token in seq]  # 1 = <UNK>
        # 截断或填充
        if len(int_seq) > max_seq_len:
            int_seq = int_seq[:max_seq_len]
        else:
            int_seq = int_seq + [0] * (max_seq_len - len(int_seq))
        int_sequences.append(int_seq)
    
    # 保存为 CSV
    columns = [f"token_{i}" for i in range(max_seq_len)]
    df = pd.DataFrame(int_sequences, columns=columns)
    df.to_csv(output_path, index=False)
    logger.info("序列数据已保存: %s (%d 行)", output_path, len(df))


# ========== 主入口 ==========

def _generate_tabular_features(
    movies_info_df: pd.DataFrame,
    regional_titles_df: pd.DataFrame,
    title_principals_df: Optional[pd.DataFrame],
    vocab: Dict[str, int],
) -> pd.DataFrame:
    """
    生成电影级别的表格特征，用于 Autoencoder 训练。
    
    输出：每行是一部电影的特征向量（离散特征 + 聚合特征）
    """
    logger.info("生成 Autoencoder 表格特征...")
    
    # 1. 基础离散特征：类型、评分等
    tabular_df = movies_info_df.copy()
    
    # 映射离散特征为整数 ID
    for col in ["genres1", "genres2", "genres3"]:
        if col in tabular_df.columns:
            tabular_df[col] = tabular_df[col].astype(str).apply(
                lambda x: vocab.get(_add_prefix(x, "genre"), 0) if x != "\\N" else 0
            )
    
    # 评分和投票数（数值特征）
    if "averageRating" in tabular_df.columns:
        tabular_df["averageRating"] = pd.to_numeric(tabular_df["averageRating"], errors="coerce").fillna(0)
    if "numVotes" in tabular_df.columns:
        tabular_df["numVotes"] = pd.to_numeric(tabular_df["numVotes"], errors="coerce").fillna(0)
        # 对数变换以减少偏斜
        tabular_df["numVotes_log"] = np.log1p(tabular_df["numVotes"])
    
    # isAdult
    if "isAdult" in tabular_df.columns:
        tabular_df["isAdult"] = pd.to_numeric(tabular_df["isAdult"], errors="coerce").fillna(0)
    
    # 2. 区域独热编码
    if regional_titles_df is not None and not regional_titles_df.empty:
        region_df = regional_titles_df[["tconst", "region"]].copy()
        region_df["region"] = region_df["region"].fillna("\\N").astype(str)
        region_dummies = pd.get_dummies(region_df["region"], prefix="region", dtype=int)
        region_agg = pd.concat([region_df[["tconst"]], region_dummies], axis=1).groupby("tconst").max().reset_index()
        tabular_df = tabular_df.merge(region_agg, on="tconst", how="left")
    
    # 3. 主演/导演数量（从 principals 聚合）
    if title_principals_df is not None and not title_principals_df.empty:
        # 演员数量
        actor_counts = title_principals_df[
            title_principals_df["category"].isin(["actor", "actress"])
        ].groupby("tconst").size().reset_index(name="num_actors")
        tabular_df = tabular_df.merge(actor_counts, on="tconst", how="left")
        tabular_df["num_actors"] = tabular_df["num_actors"].fillna(0)
        
        # 主要角色类别
        cat_counts = title_principals_df.groupby("tconst")["category"].nunique().reset_index(name="num_role_types")
        tabular_df = tabular_df.merge(cat_counts, on="tconst", how="left")
        tabular_df["num_role_types"] = tabular_df["num_role_types"].fillna(0)
    
    # 4. 选择最终特征列
    feature_cols = []
    
    # 数值特征
    for col in ["genres1", "genres2", "genres3", "averageRating", "numVotes_log", 
                "isAdult", "num_actors", "num_role_types"]:
        if col in tabular_df.columns:
            feature_cols.append(col)
    
    # 区域独热特征
    region_cols = [c for c in tabular_df.columns if c.startswith("region_")]
    feature_cols.extend(region_cols)
    
    # 填充缺失值
    result_df = tabular_df[["tconst"] + feature_cols].copy()
    result_df = result_df.fillna(0)
    
    logger.info("表格特征维度: %d 行 × %d 列", len(result_df), len(feature_cols))
    return result_df


def run_feature_engineering(
    movies_info_df: pd.DataFrame,
    staff_df: pd.DataFrame,
    regional_titles_df: pd.DataFrame,
) -> Tuple[Path, Path, Path]:
    """
    执行特征工程：生成 Word2Vec 序列数据和 Autoencoder 表格特征。
    
    Returns:
        (sequences_path, tabular_path, vocab_path)
    """
    logger.info("========== 开始特征工程 ==========")
    
    # 加载辅助数据
    cache_dir = CONFIG.paths.cache_dir
    
    title_principals_df = None
    principals_path = cache_dir / "title_principals_df.csv"
    if principals_path.exists():
        title_principals_df = pd.read_csv(principals_path)
        logger.info("加载 principals 数据: %d 行", len(title_principals_df))
    
    title_episode_df = None
    episode_path = cache_dir / "title_episode_df.csv"
    if episode_path.exists():
        title_episode_df = pd.read_csv(episode_path)
        logger.info("加载 episode 数据: %d 行", len(title_episode_df))
    
    title_crew_df = None
    crew_path = cache_dir / "title_crew_tsv_df.csv"
    if crew_path.exists():
        title_crew_df = pd.read_csv(crew_path)
        logger.info("加载 crew 数据: %d 行", len(title_crew_df))
    
    # ==================== Part 1: Word2Vec 序列数据 ====================
    logger.info("---------- 生成 Word2Vec 序列 ----------")
    all_sequences = []
    
    # 1. 人员-电影序列（最重要！）
    person_movie_seqs = _generate_person_movie_sequences(
        staff_df, title_crew_df, title_principals_df
    )
    all_sequences.extend(person_movie_seqs)
    
    # 2. 电影-上下文序列（电影的类型、评分、导演、演员）
    movie_context_seqs = _generate_movie_context_sequences(
        movies_info_df, title_principals_df, title_crew_df
    )
    all_sequences.extend(movie_context_seqs)
    
    # 3. 系列-剧集序列
    series_episode_seqs = _generate_series_episode_sequences(title_episode_df)
    all_sequences.extend(series_episode_seqs)
    
    # 4. 合作演员序列
    coactor_seqs = _generate_coactor_sequences(title_principals_df)
    all_sequences.extend(coactor_seqs)
    
    # 5. 类型-电影序列
    genre_movie_seqs = _generate_genre_movie_sequences(movies_info_df)
    all_sequences.extend(genre_movie_seqs)
    
    logger.info("========== 序列统计 ==========")
    logger.info("人员-电影序列: %d", len(person_movie_seqs))
    logger.info("电影-上下文序列: %d", len(movie_context_seqs))
    logger.info("系列-剧集序列: %d", len(series_episode_seqs))
    logger.info("合作演员序列: %d", len(coactor_seqs))
    logger.info("类型-电影序列: %d", len(genre_movie_seqs))
    logger.info("总序列数: %d", len(all_sequences))
    
    # 打乱序列顺序
    random.shuffle(all_sequences)
    
    # 构建词汇表
    vocab = build_vocab_from_sequences(all_sequences)
    
    # 保存词汇表
    vocab_path = CONFIG.paths.vocab_path
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(vocab).to_csv(vocab_path, header=False)
    logger.info("词汇表已保存: %s", vocab_path)
    
    # 保存序列数据
    sequences_path = CONFIG.paths.final_mapped_path
    save_sequences_to_csv(all_sequences, vocab, sequences_path, max_seq_len=50)
    
    # ==================== Part 2: Autoencoder 表格特征 ====================
    logger.info("---------- 生成 Autoencoder 表格特征 ----------")
    tabular_df = _generate_tabular_features(
        movies_info_df, regional_titles_df, title_principals_df, vocab
    )
    
    tabular_path = CONFIG.paths.tabular_features_path
    tabular_df.to_csv(tabular_path, index=False)
    logger.info("表格特征已保存: %s", tabular_path)
    
    logger.info("========== 特征工程完成 ==========")
    logger.info("Word2Vec 序列: %s", sequences_path)
    logger.info("Autoencoder 表格: %s", tabular_path)
    logger.info("词汇表: %s", vocab_path)
    
    return sequences_path, tabular_path, vocab_path
