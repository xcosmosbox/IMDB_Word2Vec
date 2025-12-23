"""
嵌入质量评估测试

测试内容：
1. 相似度查询：找最相似的实体
2. 类比推理：如 "导演A之于电影A" ≈ "导演B之于?"
3. 聚类分析：同类型实体是否聚在一起
4. 可视化：t-SNE 降维可视化
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from imdb_word2vec.config import CONFIG


class EmbeddingEvaluator:
    """嵌入质量评估器"""
    
    def __init__(self, vectors_path: Path = None, metadata_path: Path = None):
        """
        加载词向量和元数据。
        
        Args:
            vectors_path: 词向量文件路径（TSV 格式）
            metadata_path: 元数据文件路径（每行一个 token）
        """
        self.vectors_path = vectors_path or CONFIG.paths.vectors_path
        self.metadata_path = metadata_path or CONFIG.paths.metadata_path
        
        self._load_embeddings()
    
    def _load_embeddings(self):
        """加载嵌入向量和元数据"""
        print(f"加载词向量: {self.vectors_path}")
        print(f"加载元数据: {self.metadata_path}")
        
        # 加载向量
        self.vectors = np.loadtxt(self.vectors_path, delimiter="\t")
        
        # 加载元数据（token 名称）
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.tokens = [line.strip() for line in f.readlines()]
        
        # 构建 token → index 映射
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        
        print(f"向量维度: {self.vectors.shape}")
        print(f"Token 数量: {len(self.tokens)}")
        
        # 按前缀分类
        self.entity_types = {}
        for token in self.tokens:
            prefix = token.split("_")[0] + "_" if "_" in token else "OTHER"
            if prefix not in self.entity_types:
                self.entity_types[prefix] = []
            self.entity_types[prefix].append(token)
        
        print("\n实体类型统计:")
        for prefix, tokens in sorted(self.entity_types.items(), key=lambda x: -len(x[1])):
            print(f"  {prefix}: {len(tokens)} 个")
    
    def get_vector(self, token: str) -> np.ndarray:
        """获取 token 的向量"""
        if token not in self.token_to_idx:
            raise ValueError(f"Token '{token}' 不在词表中")
        return self.vectors[self.token_to_idx[token]]
    
    def find_similar(self, query: str, top_k: int = 10, same_type_only: bool = False) -> list:
        """
        找与 query 最相似的 tokens。
        
        Args:
            query: 查询的 token（如 "MOV_tt0001"）
            top_k: 返回前 k 个
            same_type_only: 是否只返回同类型实体
        
        Returns:
            [(token, similarity), ...]
        """
        query_vec = self.get_vector(query).reshape(1, -1)
        
        # 计算与所有向量的相似度
        similarities = cosine_similarity(query_vec, self.vectors)[0]
        
        # 获取 top_k（排除自己）
        results = []
        sorted_indices = np.argsort(similarities)[::-1]
        
        query_prefix = query.split("_")[0] + "_" if "_" in query else ""
        
        for idx in sorted_indices:
            token = self.tokens[idx]
            if token == query:
                continue
            
            if same_type_only:
                token_prefix = token.split("_")[0] + "_" if "_" in token else ""
                if token_prefix != query_prefix:
                    continue
            
            results.append((token, similarities[idx]))
            if len(results) >= top_k:
                break
        
        return results
    
    def analogy(self, a: str, b: str, c: str, top_k: int = 5) -> list:
        """
        类比推理：a 之于 b，如同 c 之于 ?
        
        例如：DIR_nm001 : MOV_tt001 = DIR_nm002 : ?
        （导演1 的代表作是电影1，导演2 的代表作是？）
        
        计算方式：vec(b) - vec(a) + vec(c) ≈ vec(?)
        """
        vec_a = self.get_vector(a)
        vec_b = self.get_vector(b)
        vec_c = self.get_vector(c)
        
        # 类比向量
        analogy_vec = (vec_b - vec_a + vec_c).reshape(1, -1)
        
        # 计算相似度
        similarities = cosine_similarity(analogy_vec, self.vectors)[0]
        
        # 排除 a, b, c
        exclude = {a, b, c}
        results = []
        for idx in np.argsort(similarities)[::-1]:
            token = self.tokens[idx]
            if token in exclude:
                continue
            results.append((token, similarities[idx]))
            if len(results) >= top_k:
                break
        
        return results
    
    def cluster_analysis(self, entity_type: str, n_clusters: int = 5, sample_size: int = 500):
        """
        对某类实体进行聚类分析。
        
        Args:
            entity_type: 实体类型前缀（如 "MOV_", "ACT_"）
            n_clusters: 聚类数
            sample_size: 采样数量
        """
        # 获取该类型的所有 token
        if entity_type not in self.entity_types:
            raise ValueError(f"未知的实体类型: {entity_type}")
        
        tokens = self.entity_types[entity_type]
        if len(tokens) > sample_size:
            import random
            tokens = random.sample(tokens, sample_size)
        
        # 获取向量
        indices = [self.token_to_idx[t] for t in tokens]
        vectors = self.vectors[indices]
        
        # K-Means 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        
        # 打印每个簇的示例
        print(f"\n{entity_type} 聚类分析 ({n_clusters} 簇):")
        for cluster_id in range(n_clusters):
            cluster_tokens = [tokens[i] for i in range(len(tokens)) if labels[i] == cluster_id]
            print(f"  簇 {cluster_id}: {len(cluster_tokens)} 个，示例: {cluster_tokens[:5]}")
        
        return tokens, labels
    
    def visualize_tsne(self, entity_types: list = None, sample_per_type: int = 100, save_path: str = None):
        """
        使用 t-SNE 可视化嵌入。
        
        Args:
            entity_types: 要可视化的实体类型列表（如 ["MOV_", "ACT_", "DIR_"]）
            sample_per_type: 每种类型采样数量
            save_path: 保存路径（如果指定）
        """
        if entity_types is None:
            entity_types = ["MOV_", "ACT_", "DIR_", "GEN_"]
        
        all_tokens = []
        all_labels = []
        
        for etype in entity_types:
            if etype not in self.entity_types:
                print(f"跳过未知类型: {etype}")
                continue
            
            tokens = self.entity_types[etype]
            if len(tokens) > sample_per_type:
                import random
                tokens = random.sample(tokens, sample_per_type)
            
            all_tokens.extend(tokens)
            all_labels.extend([etype] * len(tokens))
        
        # 获取向量
        indices = [self.token_to_idx[t] for t in all_tokens]
        vectors = self.vectors[indices]
        
        print(f"正在进行 t-SNE 降维（{len(vectors)} 个向量）...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors) - 1))
        coords = tsne.fit_transform(vectors)
        
        # 可视化
        plt.figure(figsize=(12, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(entity_types)))
        
        for i, etype in enumerate(entity_types):
            mask = [label == etype for label in all_labels]
            plt.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[colors[i]],
                label=etype,
                alpha=0.6,
                s=30
            )
        
        plt.legend()
        plt.title("Word2Vec Embeddings (t-SNE)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"图片已保存: {save_path}")
        
        plt.show()


def run_evaluation():
    """运行完整的嵌入质量评估"""
    
    print("=" * 60)
    print("Word2Vec 嵌入质量评估")
    print("=" * 60)
    
    try:
        evaluator = EmbeddingEvaluator()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 `python -m imdb_word2vec.cli train` 生成词向量")
        return
    
    # ==================== 1. 相似度查询 ====================
    print("\n" + "=" * 60)
    print("1. 相似度查询测试")
    print("=" * 60)
    
    # 测试电影相似度
    if "MOV_" in evaluator.entity_types and len(evaluator.entity_types["MOV_"]) > 0:
        sample_movie = evaluator.entity_types["MOV_"][0]
        print(f"\n与 '{sample_movie}' 最相似的电影:")
        similar = evaluator.find_similar(sample_movie, top_k=10, same_type_only=True)
        for token, sim in similar:
            print(f"  {token}: {sim:.4f}")
    
    # 测试演员相似度
    if "ACT_" in evaluator.entity_types and len(evaluator.entity_types["ACT_"]) > 0:
        sample_actor = evaluator.entity_types["ACT_"][0]
        print(f"\n与 '{sample_actor}' 最相似的演员:")
        similar = evaluator.find_similar(sample_actor, top_k=10, same_type_only=True)
        for token, sim in similar:
            print(f"  {token}: {sim:.4f}")
    
    # 测试类型相似度
    if "GEN_" in evaluator.entity_types and len(evaluator.entity_types["GEN_"]) > 0:
        sample_genre = evaluator.entity_types["GEN_"][0]
        print(f"\n与 '{sample_genre}' 最相似的类型:")
        similar = evaluator.find_similar(sample_genre, top_k=10, same_type_only=True)
        for token, sim in similar:
            print(f"  {token}: {sim:.4f}")
    
    # ==================== 2. 跨类型查询 ====================
    print("\n" + "=" * 60)
    print("2. 跨类型相似度查询")
    print("=" * 60)
    
    if "GEN_" in evaluator.entity_types:
        for genre in ["GEN_Action", "GEN_Comedy", "GEN_Drama", "GEN_Horror"][:2]:
            if genre in evaluator.token_to_idx:
                print(f"\n与 '{genre}' 最相关的实体（跨类型）:")
                similar = evaluator.find_similar(genre, top_k=10, same_type_only=False)
                for token, sim in similar:
                    print(f"  {token}: {sim:.4f}")
    
    # ==================== 3. 聚类分析 ====================
    print("\n" + "=" * 60)
    print("3. 聚类分析")
    print("=" * 60)
    
    if "MOV_" in evaluator.entity_types and len(evaluator.entity_types["MOV_"]) >= 50:
        evaluator.cluster_analysis("MOV_", n_clusters=5, sample_size=200)
    
    # ==================== 4. t-SNE 可视化 ====================
    print("\n" + "=" * 60)
    print("4. t-SNE 可视化")
    print("=" * 60)
    
    try:
        save_path = CONFIG.paths.artifacts_dir / "embedding_tsne.png"
        evaluator.visualize_tsne(
            entity_types=["MOV_", "ACT_", "DIR_", "GEN_"],
            sample_per_type=100,
            save_path=str(save_path)
        )
    except Exception as e:
        print(f"可视化失败: {e}")
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()

