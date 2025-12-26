"""
工具模块包
=========

包含数据加载、降维、相似度计算、ONNX推理、可视化、名称映射、
智能缓存、预计算等工具函数。
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
    find_similar_fast,
    find_similar_by_vector_fast,
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
from .name_mapping import (
    get_display_name,
    token_to_display,
    get_entity_type,
    get_entity_type_name,
    fuzzy_search,
    search_entities,
    search_by_name,
    get_popular_entities,
    batch_get_display_names,
    format_entity_display,
    get_entity_emoji,
)
from .cache_manager import (
    get_cached_or_compute,
    invalidate_cache,
    get_cache_info,
)
from .precompute import (
    get_normalized_embeddings,
    get_knn_index,
    get_tokens_list,
    get_precomputed_coords,
    initialize_precompute,
)
