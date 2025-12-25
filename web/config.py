"""
配置模块
========

定义全局配置常量，包括:
- 数据文件路径
- 实体类型颜色方案
- 可视化参数
- 应用常量
"""
from pathlib import Path
from typing import Dict

# =============================================================================
# 路径配置
# =============================================================================

# 项目根目录 (web/ 的父目录)
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录 - 直接引用原始 artifacts 目录，不移动文件
DATA_DIR = PROJECT_ROOT / "imdb_word2vec" / "artifacts"

# 缓存目录 - 用于存储降维计算结果
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# =============================================================================
# 数据文件路径
# =============================================================================

class DataFiles:
    """数据文件路径集合"""
    
    # 嵌入数据
    EMBEDDINGS_NPY = DATA_DIR / "embeddings.npy"      # 原始128维向量
    EMBEDDINGS_JSON = DATA_DIR / "embeddings.json"    # 完整Token+向量
    
    # 聚类数据
    CLUSTERING_JSON = DATA_DIR / "clustering.json"    # 预计算t-SNE坐标
    
    # ONNX 模型
    WORD2VEC_ONNX = DATA_DIR / "word2vec.onnx"        # 在线推理模型
    
    # TSV 文件
    METADATA_TSV = DATA_DIR / "metadata.tsv"          # Token列表
    VECTORS_TSV = DATA_DIR / "vectors.tsv"            # TF Projector兼容
    
    # 图片
    EMBEDDING_TSNE_PNG = DATA_DIR / "embedding_tsne.png"  # 静态t-SNE图
    
    # 推荐系统配置
    RECSYS_DIR = DATA_DIR / "recsys"
    CONFIG_JSON = RECSYS_DIR / "config.json"          # 系统配置
    TOKEN_TO_ID_JSON = RECSYS_DIR / "token_to_id.json"  # Token→ID
    ID_TO_TOKEN_JSON = RECSYS_DIR / "id_to_token.json"  # ID→Token
    ENTITY_INDEX_JSON = RECSYS_DIR / "entity_index.json"  # 实体分类索引
    
    # 预留文件
    VISUALIZATION_HTML = DATA_DIR / "visualization.html"

# =============================================================================
# 实体类型配置
# =============================================================================

# 实体类型中英文映射
ENTITY_TYPE_NAMES: Dict[str, str] = {
    "MOV": "电影",
    "ACT": "演员",
    "DIR": "导演",
    "GEN": "类型",
    "RAT": "评分",
    "TYP": "作品类型",
    "ERA": "年代",
    "OTHER": "其他",
}

# 实体类型颜色方案 (用于可视化)
ENTITY_TYPE_COLORS: Dict[str, str] = {
    "MOV": "#ff6b6b",    # 红色 - 电影
    "ACT": "#4ecdc4",    # 青色 - 演员
    "DIR": "#45b7d1",    # 蓝色 - 导演
    "GEN": "#96ceb4",    # 绿色 - 类型
    "RAT": "#ffeaa7",    # 黄色 - 评分
    "TYP": "#fd79a8",    # 粉色 - 作品类型
    "ERA": "#a29bfe",    # 紫色 - 年代
    "OTHER": "#b2bec3",  # 灰色 - 其他
}

# =============================================================================
# 可视化参数
# =============================================================================

class VizParams:
    """可视化参数配置"""
    
    # 散点图
    SCATTER_POINT_SIZE = 6
    SCATTER_OPACITY = 0.7
    SCATTER_HEIGHT = 700
    
    # 热力图
    HEATMAP_HEIGHT = 400
    
    # 网络图
    NETWORK_HEIGHT = 600
    
    # 默认采样数量 (用于大数据集)
    DEFAULT_SAMPLE_SIZE = 5000
    
    # 相似度搜索
    DEFAULT_TOP_K = 10
    MAX_TOP_K = 50

# =============================================================================
# 降维参数
# =============================================================================

class DimReductionParams:
    """降维算法参数配置"""
    
    # t-SNE 参数
    TSNE_PERPLEXITY = 30
    TSNE_N_ITER = 1000
    TSNE_RANDOM_STATE = 42
    
    # UMAP 参数
    UMAP_N_NEIGHBORS = 15
    UMAP_MIN_DIST = 0.1
    UMAP_RANDOM_STATE = 42
    
    # PCA 参数
    PCA_N_COMPONENTS = 2

# =============================================================================
# 应用配置
# =============================================================================

class AppConfig:
    """应用配置"""
    
    # 页面标题
    APP_TITLE = "🎬 IMDB Word2Vec 可视化看板"
    
    # 页面图标
    PAGE_ICON = "🎬"
    
    # 布局
    LAYOUT = "wide"
    
    # 侧边栏状态
    INITIAL_SIDEBAR_STATE = "expanded"

