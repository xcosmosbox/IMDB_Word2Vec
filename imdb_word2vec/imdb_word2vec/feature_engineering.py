from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CONFIG
from .logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)
random.seed(CONFIG.random_seed)

# ========== 占位符与实体类型前缀 ==========
PLACEHOLDER = "UNKNOWN"
PAD_TOKEN = "<PAD>"  # 填充 token，不会产生幻觉

ENTITY_PREFIXES = {
    "movie": "MOV_",
    "person": "PER_",
    "actor": "ACT_",
    "director": "DIR_",
    "writer": "WRI_",
    "genre": "GEN_",
    "region": "REG_",
    "category": "ROL_",
    "profession": "PRO_",
    "series": "SER_",
    "rating": "RAT_",
    "era": "ERA_",
    "title_type": "TYP_",
    "name": "NAM_",
}


def _add_prefix(value: str, prefix_type: str) -> Optional[str]:
    """为实体值添加类型前缀。"""
    if value in ("\\N", "", "nan", "None", PLACEHOLDER) or pd.isna(value):
        return None
    prefix = ENTITY_PREFIXES.get(prefix_type, "")
    return f"{prefix}{value}"


def _rating_to_bucket(rating: float) -> str:
    """将评分转换为分段标签（0.5分一档）。"""
    try:
        rating = float(rating)
        # 0.5 分一档：0.0-0.5, 0.5-1.0, ..., 9.5-10.0
        bucket = round(rating * 2) / 2  # 四舍五入到 0.5
        return f"{bucket:.1f}"
    except (ValueError, TypeError):
        return "UNKNOWN"


# ========== 去重辅助 ==========

def _deduplicate_person_movies(
    staff_movies: Dict[str, Set[str]],
    principals_movies: Dict[str, Set[str]],
    crew_movies: Dict[str, Set[str]],
) -> Dict[str, List[str]]:
    """
    合并并去重同一人的所有作品。
    
    解决问题：同一人可能出现在 staff、principals、crew 三个来源中。
    """
    all_persons: Dict[str, Set[str]] = {}
    
    # 合并所有来源
    for source in [staff_movies, principals_movies, crew_movies]:
        for person, movies in source.items():
            if person not in all_persons:
                all_persons[person] = set()
            all_persons[person].update(movies)
    
    # 转为列表
    return {p: list(m) for p, m in all_persons.items()}


# ========== 序列生成器 ==========

def _generate_person_movie_sequences(
    staff_df: pd.DataFrame,
    title_crew_df: pd.DataFrame,
    title_principals_df: pd.DataFrame,
    max_movies_per_person: int = 100,
) -> List[List[str]]:
    """
    生成 人员 → 电影 序列（去重版）。
    
    改进：
    1. 合并 staff、principals、crew 三个来源后去重
    2. 限制每人最多 max_movies_per_person 部作品
    """
    # 1. 从 staff_df 收集
    staff_movies: Dict[str, Set[str]] = {}
    for _, row in tqdm(staff_df.iterrows(), total=len(staff_df), desc="收集 staff 作品"):
        nconst = str(row["nconst"])
        if nconst == "\\N":
            continue
        if nconst not in staff_movies:
            staff_movies[nconst] = set()
        for i in range(1, 11):
            col = f"knownForTitle{i}"
            if col in row and pd.notna(row[col]) and row[col] != "\\N":
                staff_movies[nconst].add(str(row[col]))
    
    # 2. 从 principals 收集
    principals_movies: Dict[str, Set[str]] = {}
    if title_principals_df is not None and not title_principals_df.empty:
        for _, row in tqdm(title_principals_df.iterrows(), total=len(title_principals_df), desc="收集 principals 作品"):
            nconst = str(row["nconst"])
            tconst = str(row["tconst"])
            if nconst == "\\N" or tconst == "\\N":
                continue
            if nconst not in principals_movies:
                principals_movies[nconst] = set()
            principals_movies[nconst].add(tconst)
    
    # 3. 从 crew 收集导演/编剧
    crew_movies: Dict[str, Set[str]] = {}
    if title_crew_df is not None and not title_crew_df.empty:
        for _, row in tqdm(title_crew_df.iterrows(), total=len(title_crew_df), desc="收集 crew 作品"):
            tconst = str(row["tconst"])
            for col in ["directors_nconst", "writers_nconst"]:
                val = str(row.get(col, "\\N"))
                if val != "\\N":
                    for nconst in val.split(","):
                        if nconst and nconst != "\\N":
                            if nconst not in crew_movies:
                                crew_movies[nconst] = set()
                            crew_movies[nconst].add(tconst)
    
    # 4. 合并去重
    all_person_movies = _deduplicate_person_movies(staff_movies, principals_movies, crew_movies)
    
    # 5. 生成序列
    sequences = []
    for nconst, tconsts in tqdm(all_person_movies.items(), desc="生成人员-电影序列"):
        person = _add_prefix(nconst, "person")
        if person is None:
            continue
        
        # 限制作品数量，防止高产演员序列过长
        if len(tconsts) > max_movies_per_person:
            tconsts = random.sample(tconsts, max_movies_per_person)
        
        movies = [_add_prefix(t, "movie") for t in tconsts]
        movies = [m for m in movies if m is not None]
        
        if len(movies) >= 1:
            seq = [person] + movies
            sequences.append(seq)
    
    logger.info("人员-电影序列数: %d (去重后)", len(sequences))
    return sequences


def _generate_movie_context_sequences(
    movies_info_df: pd.DataFrame,
    title_principals_df: pd.DataFrame,
    title_crew_df: pd.DataFrame,
    max_actors_per_movie: int = 20,
    max_directors_per_movie: int = 5,
) -> List[List[str]]:
    """
    生成 电影 → 上下文 序列。
    
    改进：
    1. 保留所有类型（最多5个）
    2. 保留更多演员（最多20个）
    3. 添加年代信息
    4. 评分分档更细
    """
    # 预处理演员和导演
    movie_actors: Dict[str, List[str]] = {}
    movie_directors: Dict[str, List[str]] = {}
    
    if title_principals_df is not None and not title_principals_df.empty:
        for _, row in title_principals_df.iterrows():
            tconst = str(row["tconst"])
            nconst = str(row["nconst"])
            category = str(row.get("category", ""))
            
            if category in ("actor", "actress", "self"):
                if tconst not in movie_actors:
                    movie_actors[tconst] = []
                movie_actors[tconst].append(nconst)
    
    if title_crew_df is not None and not title_crew_df.empty:
        for _, row in title_crew_df.iterrows():
            tconst = str(row["tconst"])
            directors_str = str(row.get("directors_nconst", "\\N"))
            if directors_str != "\\N":
                movie_directors[tconst] = [d for d in directors_str.split(",") if d and d != "\\N"]
    
    sequences = []
    for _, row in tqdm(movies_info_df.iterrows(), total=len(movies_info_df), desc="电影上下文序列"):
        tconst = str(row["tconst"])
        movie = _add_prefix(tconst, "movie")
        if movie is None:
            continue
        
        context = [movie]
        
        # 添加作品类型
        title_type = str(row.get("titleType", ""))
        if title_type and title_type != "\\N":
            type_token = _add_prefix(title_type, "title_type")
            if type_token:
                context.append(type_token)
        
        # 添加所有类型（最多5个）
        for i in range(1, 6):
            col = f"genres{i}"
            if col in row and pd.notna(row[col]) and row[col] != "\\N":
                genre = _add_prefix(str(row[col]), "genre")
                if genre:
                    context.append(genre)
        
        # 添加年代
        era = str(row.get("era", ""))
        if era and era != "\\N" and not era.startswith("ERA_"):
            era = f"ERA_{era}"
        if era and era != "\\N":
            context.append(era)
        
        # 添加评分分档（0.5分一档）
        if "averageRating" in row:
            try:
                rating_bucket = _rating_to_bucket(row["averageRating"])
                rating_token = _add_prefix(rating_bucket, "rating")
                if rating_token:
                    context.append(rating_token)
            except (ValueError, TypeError):
                pass
        
        # 添加导演
        if tconst in movie_directors:
            for d in movie_directors[tconst][:max_directors_per_movie]:
                director = _add_prefix(d, "director")
                if director:
                    context.append(director)
        
        # 添加演员
        if tconst in movie_actors:
            for a in movie_actors[tconst][:max_actors_per_movie]:
                actor = _add_prefix(a, "actor")
                if actor:
                    context.append(actor)
        
        if len(context) >= 3:
            sequences.append(context)
    
    logger.info("电影-上下文序列数: %d", len(sequences))
    return sequences


def _generate_series_episode_sequences(
    title_episode_df: pd.DataFrame,
) -> List[List[str]]:
    """生成 系列 → 剧集 序列。"""
    sequences = []
    
    if title_episode_df is None or title_episode_df.empty:
        logger.warning("无 episode 数据，跳过系列-剧集序列")
        return sequences
    
    series_episodes = title_episode_df.groupby("parentTconst")["tconst"].apply(list).to_dict()
    
    for parent_tconst, tconsts in tqdm(series_episodes.items(), desc="系列剧集序列"):
        if parent_tconst == "\\N" or pd.isna(parent_tconst):
            continue
        
        series = _add_prefix(str(parent_tconst), "series")
        if series is None:
            continue
        
        episodes = [_add_prefix(str(t), "movie") for t in tconsts]
        episodes = [e for e in episodes if e is not None]
        
        if len(episodes) >= 2:
            seq = [series] + episodes
            sequences.append(seq)
    
    logger.info("系列-剧集序列数: %d", len(sequences))
    return sequences


def _generate_coactor_sequences(
    title_principals_df: pd.DataFrame,
    min_coactors: int = 2,
    max_coactors: int = 20,
) -> List[List[str]]:
    """生成 合作演员 序列。"""
    sequences = []
    
    if title_principals_df is None or title_principals_df.empty:
        logger.warning("无 principals 数据，跳过合作演员序列")
        return sequences
    
    movie_cast = (
        title_principals_df[title_principals_df["category"].isin(["actor", "actress", "self"])]
        .groupby("tconst")["nconst"]
        .apply(list)
        .to_dict()
    )
    
    for tconst, actors in tqdm(movie_cast.items(), desc="合作演员序列"):
        if len(actors) < min_coactors:
            continue
        
        actors = actors[:max_coactors]
        
        actor_tokens = [_add_prefix(str(a), "actor") for a in actors]
        actor_tokens = [t for t in actor_tokens if t is not None]
        
        if len(actor_tokens) >= min_coactors:
            sequences.append(actor_tokens)
    
    logger.info("合作演员序列数: %d", len(sequences))
    return sequences


def _generate_genre_movie_sequences(
    movies_info_df: pd.DataFrame,
    max_movies_per_genre: Optional[int] = None,  # None = 全量
) -> List[List[str]]:
    """
    生成 类型 → 电影 序列（全量版）。
    
    改进：使用全量数据，不再采样。
    """
    sequences = []
    genre_movies: Dict[str, List[str]] = {}
    
    for _, row in movies_info_df.iterrows():
        tconst = str(row["tconst"])
        for i in range(1, 6):
            col = f"genres{i}"
            if col in row and pd.notna(row[col]) and row[col] != "\\N":
                genre = str(row[col])
                if genre not in genre_movies:
                    genre_movies[genre] = []
                genre_movies[genre].append(tconst)
    
    for genre, tconsts in tqdm(genre_movies.items(), desc="类型电影序列"):
        genre_token = _add_prefix(genre, "genre")
        if genre_token is None:
            continue
        
        # 可选采样
        if max_movies_per_genre and len(tconsts) > max_movies_per_genre:
            tconsts = random.sample(tconsts, max_movies_per_genre)
        
        movies = [_add_prefix(t, "movie") for t in tconsts]
        movies = [m for m in movies if m is not None]
        
        if len(movies) >= 5:
            seq = [genre_token] + movies
            sequences.append(seq)
    
    logger.info("类型-电影序列数: %d (全量)", len(sequences))
    return sequences


def _generate_era_movie_sequences(
    movies_info_df: pd.DataFrame,
    max_movies_per_era: Optional[int] = None,
) -> List[List[str]]:
    """生成 同年代电影 序列。"""
    sequences = []
    era_movies: Dict[str, List[str]] = {}
    
    for _, row in movies_info_df.iterrows():
        tconst = str(row["tconst"])
        era = str(row.get("era", ""))
        if era and era != "\\N" and era != "ERA_UNKNOWN":
            if era not in era_movies:
                era_movies[era] = []
            era_movies[era].append(tconst)
    
    for era, tconsts in tqdm(era_movies.items(), desc="年代电影序列"):
        if not era.startswith("ERA_"):
            era = f"ERA_{era}"
        
        if max_movies_per_era and len(tconsts) > max_movies_per_era:
            tconsts = random.sample(tconsts, max_movies_per_era)
        
        movies = [_add_prefix(t, "movie") for t in tconsts]
        movies = [m for m in movies if m is not None]
        
        if len(movies) >= 5:
            seq = [era] + movies
            sequences.append(seq)
    
    logger.info("年代-电影序列数: %d", len(sequences))
    return sequences


def _generate_rating_movie_sequences(
    movies_info_df: pd.DataFrame,
    max_movies_per_rating: Optional[int] = None,
) -> List[List[str]]:
    """生成 同评分段电影 序列。"""
    sequences = []
    rating_movies: Dict[str, List[str]] = {}
    
    for _, row in movies_info_df.iterrows():
        tconst = str(row["tconst"])
        try:
            rating = float(row.get("averageRating", 0))
            bucket = _rating_to_bucket(rating)
            if bucket != "UNKNOWN":
                if bucket not in rating_movies:
                    rating_movies[bucket] = []
                rating_movies[bucket].append(tconst)
        except (ValueError, TypeError):
            pass
    
    for bucket, tconsts in tqdm(rating_movies.items(), desc="评分电影序列"):
        rating_token = _add_prefix(bucket, "rating")
        if rating_token is None:
            continue
        
        if max_movies_per_rating and len(tconsts) > max_movies_per_rating:
            tconsts = random.sample(tconsts, max_movies_per_rating)
        
        movies = [_add_prefix(t, "movie") for t in tconsts]
        movies = [m for m in movies if m is not None]
        
        if len(movies) >= 5:
            seq = [rating_token] + movies
            sequences.append(seq)
    
    logger.info("评分-电影序列数: %d", len(sequences))
    return sequences


def _generate_director_genre_sequences(
    movies_info_df: pd.DataFrame,
    title_crew_df: pd.DataFrame,
) -> List[List[str]]:
    """生成 导演-类型偏好 序列。"""
    sequences = []
    
    if title_crew_df is None or title_crew_df.empty:
        return sequences
    
    # 收集每个导演执导的电影类型
    director_genres: Dict[str, Set[str]] = {}
    
    # 先获取电影-类型映射
    movie_genres: Dict[str, List[str]] = {}
    for _, row in movies_info_df.iterrows():
        tconst = str(row["tconst"])
        genres = []
        for i in range(1, 6):
            col = f"genres{i}"
            if col in row and pd.notna(row[col]) and row[col] != "\\N":
                genres.append(str(row[col]))
        if genres:
            movie_genres[tconst] = genres
    
    # 收集导演偏好
    for _, row in tqdm(title_crew_df.iterrows(), total=len(title_crew_df), desc="导演类型偏好"):
        tconst = str(row["tconst"])
        directors_str = str(row.get("directors_nconst", "\\N"))
        if directors_str != "\\N" and tconst in movie_genres:
            for d in directors_str.split(","):
                if d and d != "\\N":
                    if d not in director_genres:
                        director_genres[d] = set()
                    director_genres[d].update(movie_genres[tconst])
    
    # 生成序列
    for director, genres in director_genres.items():
        if len(genres) < 2:
            continue
        dir_token = _add_prefix(director, "director")
        if dir_token is None:
            continue
        
        genre_tokens = [_add_prefix(g, "genre") for g in genres]
        genre_tokens = [g for g in genre_tokens if g is not None]
        
        if genre_tokens:
            seq = [dir_token] + genre_tokens
            sequences.append(seq)
    
    logger.info("导演-类型偏好序列数: %d", len(sequences))
    return sequences


def _generate_actor_genre_sequences(
    movies_info_df: pd.DataFrame,
    title_principals_df: pd.DataFrame,
) -> List[List[str]]:
    """生成 演员-类型偏好 序列。"""
    sequences = []
    
    if title_principals_df is None or title_principals_df.empty:
        return sequences
    
    # 电影-类型映射
    movie_genres: Dict[str, List[str]] = {}
    for _, row in movies_info_df.iterrows():
        tconst = str(row["tconst"])
        genres = []
        for i in range(1, 6):
            col = f"genres{i}"
            if col in row and pd.notna(row[col]) and row[col] != "\\N":
                genres.append(str(row[col]))
        if genres:
            movie_genres[tconst] = genres
    
    # 收集演员类型偏好
    actor_genres: Dict[str, Set[str]] = {}
    actor_df = title_principals_df[title_principals_df["category"].isin(["actor", "actress", "self"])]
    
    for _, row in tqdm(actor_df.iterrows(), total=len(actor_df), desc="演员类型偏好"):
        nconst = str(row["nconst"])
        tconst = str(row["tconst"])
        if tconst in movie_genres:
            if nconst not in actor_genres:
                actor_genres[nconst] = set()
            actor_genres[nconst].update(movie_genres[tconst])
    
    # 生成序列
    for actor, genres in actor_genres.items():
        if len(genres) < 2:
            continue
        actor_token = _add_prefix(actor, "actor")
        if actor_token is None:
            continue
        
        genre_tokens = [_add_prefix(g, "genre") for g in genres]
        genre_tokens = [g for g in genre_tokens if g is not None]
        
        if genre_tokens:
            seq = [actor_token] + genre_tokens
            sequences.append(seq)
    
    logger.info("演员-类型偏好序列数: %d", len(sequences))
    return sequences


# ========== 词表构建 ==========

def build_vocab_from_sequences(sequences: List[List[str]]) -> Dict[str, int]:
    """从序列中构建词汇表。"""
    vocab: Dict[str, int] = {PAD_TOKEN: 0, "<UNK>": 1}
    counter = 1
    
    all_tokens = set()
    for seq in sequences:
        all_tokens.update(seq)
    
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
    max_seq_len: int = 100,  # 增加到 100
    vocab_limit: Optional[int] = None,
) -> None:
    """
    将序列保存为 CSV 文件（流式写入，避免内存溢出）。
    
    关键改进：token ID 重新映射，确保所有 ID 落在 [0, vocab_limit) 范围内。
    - 保留 PAD_TOKEN (0) 和 "<UNK>" (1)
    - 保留最频繁的 vocab_limit-2 个其他 token
    - 超出范围的 token 映射为 UNK (1)
    
    填充策略：使用 0 (<PAD>) 填充，这是专门的填充 token。
    """
    import csv
    
    if vocab_limit is None:
        vocab_limit = CONFIG.train.vocab_limit
    
    logger.info("========== Token 重新映射 ==========")
    logger.info("原始词汇表大小: %d", len(vocab))
    logger.info("目标词汇表大小: %d", vocab_limit)
    
    # ========== Step 1: 统计每个 token 的出现频率 ==========
    token_freq: Dict[str, int] = {}
    for seq in sequences:
        for token in seq:
            token_freq[token] = token_freq.get(token, 0) + 1
    
    # ========== Step 2: 按频率排序，保留最频繁的 vocab_limit-2 个 ==========
    # 预留 0 (PAD) 和 1 (UNK)
    sorted_tokens = sorted(
        token_freq.items(),
        key=lambda x: -x[1]  # 降序排列
    )[:vocab_limit - 2]
    
    # ========== Step 3: 构建新的 vocab（重新映射）==========
    new_vocab: Dict[str, int] = {PAD_TOKEN: 0, "<UNK>": 1}
    for idx, (token, freq) in enumerate(sorted_tokens):
        new_vocab[token] = 2 + idx  # 从 ID 2 开始
    
    logger.info("新词汇表大小: %d (保留频率最高的 %d 个 token)", 
                len(new_vocab), len(sorted_tokens))
    
    num_remapped = len(vocab) - len(new_vocab)
    logger.info("重新映射的 token 数: %d (将映射为 UNK)", num_remapped)
    
    # ========== Step 4: 流式写入序列（使用新 vocab 映射）==========
    columns = [f"token_{i}" for i in range(max_seq_len)]
    chunk_size = 100000  # 每次处理 100k 序列
    int_sequences = []
    total_rows = 0
    remapped_count = 0
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        
        for seq in tqdm(sequences, desc="转换序列为整数（重新映射）"):
            int_seq = []
            for token in seq:
                # 使用新 vocab 映射，未出现的 token 用 UNK (1)
                token_id = new_vocab.get(token, 1)
                if token not in new_vocab:
                    remapped_count += 1
                int_seq.append(token_id)
            
            # 截断或填充到 max_seq_len
            if len(int_seq) > max_seq_len:
                int_seq = int_seq[:max_seq_len]
            else:
                int_seq = int_seq + [0] * (max_seq_len - len(int_seq))
            
            int_sequences.append(int_seq)
            
            # 批量写入
            if len(int_sequences) >= chunk_size:
                writer.writerows(int_sequences)
                total_rows += len(int_sequences)
                int_sequences = []
        
        # 处理剩余
        if int_sequences:
            writer.writerows(int_sequences)
            total_rows += len(int_sequences)
    
    logger.info("========== 映射完成 ==========")
    logger.info("序列数据已保存: %s (%d 行, 长度 %d)", output_path, total_rows, max_seq_len)
    logger.info("被重新映射为 UNK 的 token 实例: %d", remapped_count)
    
    # 保存新词汇表供后续使用（覆盖原词汇表）
    vocab_path = CONFIG.paths.vocab_path
    pd.Series(new_vocab).to_csv(vocab_path, header=False)
    logger.info("更新后的词汇表已保存: %s (大小: %d)", vocab_path, len(new_vocab))


# 辅助：按组保存/加载序列，便于断点续跑
def _seq_group_path(cache_dir: Path, name: str) -> Path:
    seq_dir = cache_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    return seq_dir / f"{name}.csv"


def _save_seq_group(sequences: List[List[str]], path: Path) -> None:
    import csv
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(sequences)
    logger.info("已保存序列组: %s (%d 条)", path, len(sequences))


def _load_seq_group(path: Path) -> List[List[str]]:
    import csv
    sequences: List[List[str]] = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                sequences.append(row)
    logger.info("已加载序列组: %s (%d 条)", path, len(sequences))
    return sequences


# ========== Autoencoder 表格特征 ==========

def _generate_tabular_features(
    movies_info_df: pd.DataFrame,
    regional_titles_df: pd.DataFrame,
    title_principals_df: Optional[pd.DataFrame],
    vocab: Dict[str, int],
) -> pd.DataFrame:
    """生成电影级别的表格特征。"""
    logger.info("生成 Autoencoder 表格特征...")
    
    tabular_df = movies_info_df.copy()
    
    # 类型特征
    for i in range(1, 6):
        col = f"genres{i}"
        if col in tabular_df.columns:
            tabular_df[col] = tabular_df[col].astype(str).apply(
                lambda x: vocab.get(_add_prefix(x, "genre"), 0) if x != "\\N" else 0
            )
    
    # 评分和投票数
    if "averageRating" in tabular_df.columns:
        tabular_df["averageRating"] = pd.to_numeric(tabular_df["averageRating"], errors="coerce").fillna(0)
    if "numVotes" in tabular_df.columns:
        tabular_df["numVotes"] = pd.to_numeric(tabular_df["numVotes"], errors="coerce").fillna(0)
        tabular_df["numVotes_log"] = np.log1p(tabular_df["numVotes"])
    
    if "isAdult" in tabular_df.columns:
        tabular_df["isAdult"] = pd.to_numeric(tabular_df["isAdult"], errors="coerce").fillna(0)
    
    # 区域独热
    if regional_titles_df is not None and not regional_titles_df.empty:
        region_df = regional_titles_df[["tconst", "region"]].copy()
        region_df["region"] = region_df["region"].fillna("\\N").astype(str)
        region_dummies = pd.get_dummies(region_df["region"], prefix="region", dtype=int)
        region_agg = pd.concat([region_df[["tconst"]], region_dummies], axis=1).groupby("tconst").max().reset_index()
        tabular_df = tabular_df.merge(region_agg, on="tconst", how="left")
    
    # 演员数量
    if title_principals_df is not None and not title_principals_df.empty:
        actor_counts = title_principals_df[
            title_principals_df["category"].isin(["actor", "actress"])
        ].groupby("tconst").size().reset_index(name="num_actors")
        tabular_df = tabular_df.merge(actor_counts, on="tconst", how="left")
        tabular_df["num_actors"] = tabular_df["num_actors"].fillna(0)
        
        cat_counts = title_principals_df.groupby("tconst")["category"].nunique().reset_index(name="num_role_types")
        tabular_df = tabular_df.merge(cat_counts, on="tconst", how="left")
        tabular_df["num_role_types"] = tabular_df["num_role_types"].fillna(0)
    
    # 选择特征列
    feature_cols = []
    for col in ["genres1", "genres2", "genres3", "genres4", "genres5",
                "averageRating", "numVotes_log", "isAdult", "num_actors", "num_role_types"]:
        if col in tabular_df.columns:
            feature_cols.append(col)
    
    region_cols = [c for c in tabular_df.columns if c.startswith("region_")]
    feature_cols.extend(region_cols)
    
    result_df = tabular_df[["tconst"] + feature_cols].copy()
    result_df = result_df.fillna(0)
    
    logger.info("表格特征维度: %d 行 × %d 列", len(result_df), len(feature_cols))
    return result_df


# ========== 主入口 ==========

def run_feature_engineering(
    movies_info_df: pd.DataFrame,
    staff_df: pd.DataFrame,
    regional_titles_df: pd.DataFrame,
    resume: bool = True,
) -> Tuple[Path, Path, Path]:
    """
    执行特征工程：生成 Word2Vec 序列数据和 Autoencoder 表格特征。
    
    改进：
    1. 人员-电影序列去重
    2. 增加年代、评分、导演偏好、演员偏好序列
    3. 全量类型数据
    4. 更长的序列
    """
    logger.info("========== 开始特征工程 (增强版) ==========")
    
    cache_dir = CONFIG.paths.cache_dir
    
    # 加载辅助数据
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
    all_sequences: List[List[str]] = []

    # helper for per-group checkpoint files
    def _seq_path(name: str) -> Path:
        return _seq_group_path(cache_dir, name)

    # 1. 人员-电影序列（去重版）
    person_path = _seq_path("person_movie_seqs")
    if resume and person_path.exists():
        person_movie_seqs = _load_seq_group(person_path)
    else:
        person_movie_seqs = _generate_person_movie_sequences(
            staff_df, title_crew_df, title_principals_df,
            max_movies_per_person=100
        )
        _save_seq_group(person_movie_seqs, person_path)
    all_sequences.extend(person_movie_seqs)

    # 2. 电影-上下文序列
    movie_path = _seq_path("movie_context_seqs")
    if resume and movie_path.exists():
        movie_context_seqs = _load_seq_group(movie_path)
    else:
        movie_context_seqs = _generate_movie_context_sequences(
            movies_info_df, title_principals_df, title_crew_df,
            max_actors_per_movie=20,
            max_directors_per_movie=5
        )
        _save_seq_group(movie_context_seqs, movie_path)
    all_sequences.extend(movie_context_seqs)

    # 3. 系列-剧集序列
    series_path = _seq_path("series_episode_seqs")
    if resume and series_path.exists():
        series_episode_seqs = _load_seq_group(series_path)
    else:
        series_episode_seqs = _generate_series_episode_sequences(title_episode_df)
        _save_seq_group(series_episode_seqs, series_path)
    all_sequences.extend(series_episode_seqs)

    # 4. 合作演员序列
    coactor_path = _seq_path("coactor_seqs")
    if resume and coactor_path.exists():
        coactor_seqs = _load_seq_group(coactor_path)
    else:
        coactor_seqs = _generate_coactor_sequences(title_principals_df, max_coactors=20)
        _save_seq_group(coactor_seqs, coactor_path)
    all_sequences.extend(coactor_seqs)

    # 5. 类型-电影序列（全量）
    genre_path = _seq_path("genre_movie_seqs")
    if resume and genre_path.exists():
        genre_movie_seqs = _load_seq_group(genre_path)
    else:
        genre_movie_seqs = _generate_genre_movie_sequences(movies_info_df, max_movies_per_genre=None)
        _save_seq_group(genre_movie_seqs, genre_path)
    all_sequences.extend(genre_movie_seqs)

    # 6. 年代-电影序列（新增）
    era_path = _seq_path("era_movie_seqs")
    if resume and era_path.exists():
        era_movie_seqs = _load_seq_group(era_path)
    else:
        era_movie_seqs = _generate_era_movie_sequences(movies_info_df, max_movies_per_era=None)
        _save_seq_group(era_movie_seqs, era_path)
    all_sequences.extend(era_movie_seqs)

    # 7. 评分-电影序列（新增）
    rating_path = _seq_path("rating_movie_seqs")
    if resume and rating_path.exists():
        rating_movie_seqs = _load_seq_group(rating_path)
    else:
        rating_movie_seqs = _generate_rating_movie_sequences(movies_info_df, max_movies_per_rating=None)
        _save_seq_group(rating_movie_seqs, rating_path)
    all_sequences.extend(rating_movie_seqs)

    # 8. 导演-类型偏好序列（新增）
    director_path = _seq_path("director_genre_seqs")
    if resume and director_path.exists():
        director_genre_seqs = _load_seq_group(director_path)
    else:
        director_genre_seqs = _generate_director_genre_sequences(movies_info_df, title_crew_df)
        _save_seq_group(director_genre_seqs, director_path)
    all_sequences.extend(director_genre_seqs)

    # 9. 演员-类型偏好序列（新增）
    actor_path = _seq_path("actor_genre_seqs")
    if resume and actor_path.exists():
        actor_genre_seqs = _load_seq_group(actor_path)
    else:
        actor_genre_seqs = _generate_actor_genre_sequences(movies_info_df, title_principals_df)
        _save_seq_group(actor_genre_seqs, actor_path)
    all_sequences.extend(actor_genre_seqs)

    logger.info("========== 序列统计 ==========")
    logger.info("1. 人员-电影序列: %d", len(person_movie_seqs))
    logger.info("2. 电影-上下文序列: %d", len(movie_context_seqs))
    logger.info("3. 系列-剧集序列: %d", len(series_episode_seqs))
    logger.info("4. 合作演员序列: %d", len(coactor_seqs))
    logger.info("5. 类型-电影序列: %d", len(genre_movie_seqs))
    logger.info("6. 年代-电影序列: %d", len(era_movie_seqs))
    logger.info("7. 评分-电影序列: %d", len(rating_movie_seqs))
    logger.info("8. 导演-类型偏好: %d", len(director_genre_seqs))
    logger.info("9. 演员-类型偏好: %d", len(actor_genre_seqs))
    logger.info("总序列数: %d", len(all_sequences))

    logger.info("所有序列生成完成，开始合并...")
    random.shuffle(all_sequences)

    # 构建词汇表
    vocab = build_vocab_from_sequences(all_sequences)
    
    logger.info("原始词汇表规模: %d", len(vocab))

    # 保存映射后的序列（长度 100）+ 自动重新映射 token 到有效范围
    # 这一步会自动根据 vocab_limit 重新映射 token 并保存新的词汇表
    sequences_path = CONFIG.paths.final_mapped_path
    if resume and sequences_path.exists():
        logger.info("映射后的序列文件已存在，跳过写入: %s", sequences_path)
    else:
        save_sequences_to_csv(all_sequences, vocab, sequences_path, max_seq_len=100)
    
    # ==================== Part 2: Autoencoder 表格特征 ====================
    logger.info("---------- 生成 Autoencoder 表格特征 ----------")
    tabular_df = _generate_tabular_features(
        movies_info_df, regional_titles_df, title_principals_df, vocab
    )
    
    tabular_path = CONFIG.paths.tabular_features_path
    tabular_df.to_csv(tabular_path, index=False)
    logger.info("表格特征已保存: %s", tabular_path)
    
    logger.info("========== 特征工程完成 ==========")
    return sequences_path, tabular_path, vocab_path
