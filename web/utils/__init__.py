"""
工具模块包
=========

包含数据加载、降维、相似度计算、ONNX推理、可视化等工具函数。
"""
from .data_loader import (
    load_config,
    load_clustering_data,
    load_embeddings_npy,
    load_embeddings_json,
    load_metadata,
    load_token_to_id,
    load_id_to_token,
    load_entity_index,
    get_entity_type,
)
from .dimensionality import (
    compute_pca,
    compute_umap,
    compute_tsne,
    get_cached_reduction,
)
from .similarity import (
    cosine_similarity,
    find_top_k_similar,
    compute_similarity_matrix,
)
from .onnx_inference import (
    OnnxEmbeddingModel,
    get_embedding,
)
from .visualization import (
    create_scatter_plot,
    create_heatmap,
    create_network_graph,
    create_bar_chart,
)

