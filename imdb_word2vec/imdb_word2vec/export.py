"""
模型导出模块

提供多种格式的导出功能：
1. TSV 格式 - TensorFlow Embedding Projector 兼容
2. ONNX 格式 - 在线推理部署
3. JSON 格式 - 网页可视化
4. 聚类 JSON - 交互式聚类可视化
5. 推荐系统配置 - 网页部署配置

使用方法:
    python -m imdb_word2vec.cli export
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CONFIG
from .logging_utils import setup_logging


logger = setup_logging(CONFIG.paths.logs_dir)


# ========== 基础导出函数 ==========

def export_tsv(
    weights: np.ndarray,
    tokens: List[str],
    vectors_path: Path,
    metadata_path: Path,
) -> None:
    """导出 TSV 格式（TensorFlow Embedding Projector 兼容）。"""
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(vectors_path, "w") as out_v, open(metadata_path, "w") as out_m:
        for idx, token in enumerate(tokens):
            if idx == 0:  # 跳过 PAD
                continue
            vec = weights[idx]
            out_v.write("\t".join([str(x) for x in vec]) + "\n")
            out_m.write(token + "\n")
    
    logger.info("TSV 导出: %s, %s", vectors_path, metadata_path)


def export_onnx(
    vocab_size: int,
    embedding_dim: int,
    weights: np.ndarray,
    output_path: Path,
) -> None:
    """
    导出 ONNX 格式（在线推理）。
    
    ONNX 模型输入: token_ids (int64, shape: [batch_size])
    ONNX 模型输出: embeddings (float32, shape: [batch_size, embedding_dim])
    """
    try:
        import onnx
        from onnx import numpy_helper, TensorProto
        from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
    except ImportError:
        logger.warning("ONNX 未安装，跳过 ONNX 导出。安装: pip install onnx")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建 Embedding 查找表（使用 Gather 操作）
    # 输入: token_ids
    # 输出: embeddings
    
    # 定义输入
    input_ids = make_tensor_value_info("token_ids", TensorProto.INT64, [None])
    
    # 定义输出
    output_embeddings = make_tensor_value_info(
        "embeddings", TensorProto.FLOAT, [None, embedding_dim]
    )
    
    # 创建权重常量
    embedding_weights = numpy_helper.from_array(
        weights.astype(np.float32), name="embedding_weights"
    )
    
    # 创建 Gather 节点（实现 embedding lookup）
    gather_node = make_node(
        "Gather",
        inputs=["embedding_weights", "token_ids"],
        outputs=["embeddings"],
        axis=0,
    )
    
    # 创建图
    graph = make_graph(
        nodes=[gather_node],
        name="Word2VecEmbedding",
        inputs=[input_ids],
        outputs=[output_embeddings],
        initializer=[embedding_weights],
    )
    
    # 创建模型
    model = make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    model.ir_version = 8
    
    # 验证并保存
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    
    logger.info("ONNX 导出: %s (%.2f MB)", output_path, output_path.stat().st_size / (1024**2))


def export_json_embeddings(
    weights: np.ndarray,
    tokens: List[str],
    output_path: Path,
    max_tokens: int = 50000,
) -> None:
    """
    导出 JSON 格式嵌入（网页可视化）。
    
    格式: {
        "tokens": ["MOV_tt001", "ACT_nm002", ...],
        "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
        "metadata": {
            "vocab_size": 50000,
            "embedding_dim": 128,
            "entity_types": {"MOV": 10000, "ACT": 5000, ...}
        }
    }
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 统计实体类型
    entity_types: Dict[str, int] = {}
    valid_indices = []
    
    for idx, token in enumerate(tokens):
        if idx == 0:  # 跳过 PAD
            continue
        if idx >= max_tokens:
            break
        
        # 提取前缀
        prefix = token.split("_")[0] if "_" in token else "OTHER"
        entity_types[prefix] = entity_types.get(prefix, 0) + 1
        valid_indices.append(idx)
    
    # 构建输出
    output_tokens = [tokens[i] for i in valid_indices]
    output_embeddings = weights[valid_indices].tolist()
    
    data = {
        "tokens": output_tokens,
        "embeddings": output_embeddings,
        "metadata": {
            "vocab_size": len(output_tokens),
            "embedding_dim": weights.shape[1],
            "entity_types": entity_types,
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f)
    
    logger.info("JSON 嵌入导出: %s (%.2f MB)", output_path, output_path.stat().st_size / (1024**2))


def export_clustering_visualization(
    weights: np.ndarray,
    tokens: List[str],
    output_path: Path,
    n_samples: int = 5000,
    n_clusters: int = 20,
) -> None:
    """
    导出交互式聚类可视化数据（t-SNE 降维 + K-Means 聚类）。
    
    输出 JSON 格式，可直接用于 D3.js / Plotly 等网页可视化库。
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
    except ImportError:
        logger.warning("scikit-learn 未安装，跳过聚类可视化导出")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 采样
    valid_indices = [i for i in range(1, len(tokens)) if i < len(weights)]
    if len(valid_indices) > n_samples:
        sample_indices = np.random.choice(valid_indices, n_samples, replace=False)
    else:
        sample_indices = np.array(valid_indices)
    
    sample_tokens = [tokens[i] for i in sample_indices]
    sample_embeddings = weights[sample_indices]
    
    logger.info("t-SNE 降维中... (%d 样本)", len(sample_indices))
    
    # t-SNE 降维到 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords_2d = tsne.fit_transform(sample_embeddings)
    
    # K-Means 聚类
    logger.info("K-Means 聚类中... (%d 簇)", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(sample_embeddings)
    
    # 提取实体类型
    entity_types = []
    for token in sample_tokens:
        if "_" in token:
            prefix = token.split("_")[0]
        else:
            prefix = "OTHER"
        entity_types.append(prefix)
    
    # 构建输出
    points = []
    for i in range(len(sample_tokens)):
        points.append({
            "token": sample_tokens[i],
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "cluster": int(cluster_labels[i]),
            "type": entity_types[i],
        })
    
    # 聚类中心
    cluster_centers = []
    for c in range(n_clusters):
        cluster_points = [p for p in points if p["cluster"] == c]
        if cluster_points:
            center_x = np.mean([p["x"] for p in cluster_points])
            center_y = np.mean([p["y"] for p in cluster_points])
            
            # 找出该簇中最常见的实体类型
            types_in_cluster = [p["type"] for p in cluster_points]
            most_common_type = max(set(types_in_cluster), key=types_in_cluster.count)
            
            cluster_centers.append({
                "cluster_id": c,
                "center_x": float(center_x),
                "center_y": float(center_y),
                "size": len(cluster_points),
                "dominant_type": most_common_type,
            })
    
    data = {
        "points": points,
        "clusters": cluster_centers,
        "metadata": {
            "n_samples": len(points),
            "n_clusters": n_clusters,
            "embedding_dim": weights.shape[1],
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f)
    
    logger.info("聚类可视化导出: %s (%.2f MB)", output_path, output_path.stat().st_size / (1024**2))


def export_recommendation_config(
    weights: np.ndarray,
    tokens: List[str],
    output_dir: Path,
) -> None:
    """
    导出推荐系统配置文件。
    
    包含：
    1. token_to_id.json - Token 到 ID 的映射
    2. id_to_token.json - ID 到 Token 的映射
    3. entity_index.json - 按实体类型分类的索引
    4. config.json - 推荐系统配置
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Token <-> ID 映射
    token_to_id = {token: idx for idx, token in enumerate(tokens) if idx > 0}
    id_to_token = {idx: token for idx, token in enumerate(tokens) if idx > 0}
    
    with open(output_dir / "token_to_id.json", "w") as f:
        json.dump(token_to_id, f)
    
    with open(output_dir / "id_to_token.json", "w") as f:
        json.dump(id_to_token, f)
    
    # 按实体类型分类
    entity_index: Dict[str, List[str]] = {}
    for token in tokens:
        if "_" in token:
            prefix = token.split("_")[0]
        else:
            prefix = "OTHER"
        
        if prefix not in entity_index:
            entity_index[prefix] = []
        entity_index[prefix].append(token)
    
    with open(output_dir / "entity_index.json", "w") as f:
        json.dump(entity_index, f)
    
    # 推荐系统配置
    config = {
        "vocab_size": len(tokens),
        "embedding_dim": weights.shape[1],
        "entity_types": {k: len(v) for k, v in entity_index.items()},
        "similarity_metric": "cosine",
        "top_k_default": 10,
        "files": {
            "onnx_model": "word2vec.onnx",
            "embeddings_json": "embeddings.json",
            "clustering_json": "clustering.json",
            "token_to_id": "token_to_id.json",
            "id_to_token": "id_to_token.json",
            "entity_index": "entity_index.json",
        }
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("推荐系统配置导出: %s", output_dir)


def export_html_visualization(
    output_path: Path,
    clustering_json_path: str = "clustering.json",
) -> None:
    """
    导出交互式 HTML 可视化页面。
    
    使用 Plotly.js 创建可交互的聚类散点图。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Word2Vec 嵌入可视化</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            margin-bottom: 20px;
        }
        .stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .stat-card {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 15px 25px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #00d4ff;
        }
        .stat-label {
            font-size: 0.9em;
            color: #888;
            margin-top: 5px;
        }
        #chart {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .legend {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 20px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255,255,255,0.1);
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .search-box {
            max-width: 400px;
            margin: 20px auto;
        }
        .search-box input {
            width: 100%;
            padding: 12px 20px;
            border: none;
            border-radius: 25px;
            background: rgba(255,255,255,0.1);
            color: #fff;
            font-size: 1em;
            outline: none;
        }
        .search-box input::placeholder {
            color: #888;
        }
        .info {
            text-align: center;
            margin-top: 20px;
            color: #888;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Word2Vec 嵌入可视化</h1>
        
        <div class="stats" id="stats"></div>
        
        <div class="search-box">
            <input type="text" id="search" placeholder="搜索 Token (如: MOV_tt0111161)">
        </div>
        
        <div id="chart"></div>
        
        <div class="legend" id="legend"></div>
        
        <p class="info">
            使用 t-SNE 降维 + K-Means 聚类 | 
            点击数据点查看详情 | 
            滚轮缩放，拖拽平移
        </p>
    </div>
    
    <script>
        const COLORS = {
            'MOV': '#ff6b6b',
            'ACT': '#4ecdc4',
            'DIR': '#45b7d1',
            'GEN': '#96ceb4',
            'PER': '#ffeaa7',
            'RAT': '#dfe6e9',
            'ERA': '#a29bfe',
            'TYP': '#fd79a8',
            'OTHER': '#b2bec3'
        };
        
        const TYPE_NAMES = {
            'MOV': '电影',
            'ACT': '演员',
            'DIR': '导演',
            'GEN': '类型',
            'PER': '人员',
            'RAT': '评分',
            'ERA': '年代',
            'TYP': '作品类型',
            'OTHER': '其他'
        };
        
        fetch('CLUSTERING_JSON_PATH')
            .then(res => res.json())
            .then(data => {
                // 统计
                const stats = document.getElementById('stats');
                const typeCounts = {};
                data.points.forEach(p => {
                    typeCounts[p.type] = (typeCounts[p.type] || 0) + 1;
                });
                
                stats.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-value">${data.points.length.toLocaleString()}</div>
                        <div class="stat-label">样本数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.clusters.length}</div>
                        <div class="stat-label">聚类数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${Object.keys(typeCounts).length}</div>
                        <div class="stat-label">实体类型</div>
                    </div>
                `;
                
                // 图例
                const legend = document.getElementById('legend');
                legend.innerHTML = Object.entries(typeCounts)
                    .sort((a, b) => b[1] - a[1])
                    .map(([type, count]) => `
                        <div class="legend-item">
                            <div class="legend-color" style="background: ${COLORS[type] || COLORS.OTHER}"></div>
                            <span>${TYPE_NAMES[type] || type} (${count})</span>
                        </div>
                    `).join('');
                
                // 按类型分组
                const traces = [];
                const groupedByType = {};
                data.points.forEach(p => {
                    if (!groupedByType[p.type]) {
                        groupedByType[p.type] = { x: [], y: [], text: [], cluster: [] };
                    }
                    groupedByType[p.type].x.push(p.x);
                    groupedByType[p.type].y.push(p.y);
                    groupedByType[p.type].text.push(p.token);
                    groupedByType[p.type].cluster.push(p.cluster);
                });
                
                Object.entries(groupedByType).forEach(([type, points]) => {
                    traces.push({
                        x: points.x,
                        y: points.y,
                        text: points.text,
                        customdata: points.cluster,
                        mode: 'markers',
                        type: 'scatter',
                        name: TYPE_NAMES[type] || type,
                        marker: {
                            color: COLORS[type] || COLORS.OTHER,
                            size: 6,
                            opacity: 0.7,
                        },
                        hovertemplate: '<b>%{text}</b><br>聚类: %{customdata}<extra></extra>'
                    });
                });
                
                const layout = {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#e0e0e0' },
                    xaxis: {
                        showgrid: true,
                        gridcolor: 'rgba(255,255,255,0.1)',
                        zeroline: false,
                        title: 't-SNE 维度 1'
                    },
                    yaxis: {
                        showgrid: true,
                        gridcolor: 'rgba(255,255,255,0.1)',
                        zeroline: false,
                        title: 't-SNE 维度 2'
                    },
                    legend: {
                        x: 1,
                        y: 1,
                        bgcolor: 'rgba(0,0,0,0.5)',
                    },
                    hovermode: 'closest',
                    margin: { l: 50, r: 50, t: 20, b: 50 }
                };
                
                Plotly.newPlot('chart', traces, layout, { responsive: true });
                
                // 搜索功能
                const searchInput = document.getElementById('search');
                searchInput.addEventListener('input', (e) => {
                    const query = e.target.value.toLowerCase();
                    if (!query) {
                        Plotly.restyle('chart', { 'marker.opacity': 0.7 });
                        return;
                    }
                    
                    traces.forEach((trace, i) => {
                        const opacities = trace.text.map(t => 
                            t.toLowerCase().includes(query) ? 1 : 0.1
                        );
                        Plotly.restyle('chart', { 'marker.opacity': [opacities] }, [i]);
                    });
                });
            })
            .catch(err => {
                document.getElementById('chart').innerHTML = 
                    '<p style="text-align:center;padding:50px;color:#ff6b6b;">加载数据失败: ' + err + '</p>';
            });
    </script>
</body>
</html>'''.replace('CLUSTERING_JSON_PATH', clustering_json_path)
    
    with open(output_path, "w") as f:
        f.write(html_content)
    
    logger.info("HTML 可视化导出: %s", output_path)


# ========== 主导出函数 ==========

def export_all(
    weights: np.ndarray,
    tokens: List[str],
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    导出所有格式的文件。
    
    Returns:
        导出文件路径字典
    """
    if output_dir is None:
        output_dir = CONFIG.paths.artifacts_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("========== 开始导出所有格式 ==========")
    
    exported_files = {}
    
    # 1. TSV 格式（原有）
    vectors_path = output_dir / "vectors.tsv"
    metadata_path = output_dir / "metadata.tsv"
    export_tsv(weights, tokens, vectors_path, metadata_path)
    exported_files["vectors_tsv"] = vectors_path
    exported_files["metadata_tsv"] = metadata_path
    
    # 2. ONNX 格式（在线推理）
    onnx_path = output_dir / "word2vec.onnx"
    export_onnx(len(tokens), weights.shape[1], weights, onnx_path)
    if onnx_path.exists():
        exported_files["onnx"] = onnx_path
    
    # 3. JSON 嵌入（网页可视化）
    embeddings_json_path = output_dir / "embeddings.json"
    export_json_embeddings(weights, tokens, embeddings_json_path)
    exported_files["embeddings_json"] = embeddings_json_path
    
    # 4. 聚类可视化 JSON
    clustering_json_path = output_dir / "clustering.json"
    export_clustering_visualization(weights, tokens, clustering_json_path)
    exported_files["clustering_json"] = clustering_json_path
    
    # 5. 推荐系统配置
    recsys_dir = output_dir / "recsys"
    export_recommendation_config(weights, tokens, recsys_dir)
    exported_files["recsys_config"] = recsys_dir / "config.json"
    
    # 6. HTML 可视化页面
    html_path = output_dir / "visualization.html"
    export_html_visualization(html_path, "clustering.json")
    exported_files["visualization_html"] = html_path
    
    # 7. 保存原始权重（NumPy 格式）
    weights_path = output_dir / "embeddings.npy"
    np.save(weights_path, weights)
    exported_files["embeddings_npy"] = weights_path
    
    logger.info("========== 导出完成 ==========")
    logger.info("导出目录: %s", output_dir)
    
    # 打印文件清单
    total_size = 0
    for name, path in exported_files.items():
        if path.exists():
            size = path.stat().st_size / (1024**2)
            total_size += size
            logger.info("  - %s: %.2f MB", path.name, size)
    
    logger.info("总大小: %.2f MB", total_size)
    
    return exported_files

