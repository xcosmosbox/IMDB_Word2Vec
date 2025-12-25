"""
可视化工具模块
=============

基于 Plotly 的可视化函数，包括散点图、热力图、网络图、柱状图等。

使用方法:
    # 创建聚类散点图
    fig = create_scatter_plot(df, x="x", y="y", color="type")
    st.plotly_chart(fig)
"""
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 导入配置
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENTITY_TYPE_COLORS, ENTITY_TYPE_NAMES, VizParams


# =============================================================================
# 散点图
# =============================================================================

def create_scatter_plot(
    df: pd.DataFrame,
    x: str = "x",
    y: str = "y",
    color: str = "type",
    hover_data: Optional[List[str]] = None,
    title: Optional[str] = None,
    height: int = VizParams.SCATTER_HEIGHT,
    show_legend: bool = True,
) -> go.Figure:
    """
    创建交互式散点图
    
    用于展示 t-SNE/PCA/UMAP 降维结果。
    
    Args:
        df: 包含坐标数据的 DataFrame
        x: X 坐标列名
        y: Y 坐标列名
        color: 颜色分组列名
        hover_data: 悬停时显示的额外列
        title: 图表标题
        height: 图表高度
        show_legend: 是否显示图例
        
    Returns:
        Plotly Figure 对象
    """
    if hover_data is None:
        hover_data = ["token"] if "token" in df.columns else []
    
    # 创建颜色映射
    color_map = ENTITY_TYPE_COLORS.copy()
    
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        hover_data=hover_data,
        color_discrete_map=color_map,
        title=title,
    )
    
    # 更新布局
    fig.update_traces(
        marker=dict(
            size=VizParams.SCATTER_POINT_SIZE,
            opacity=VizParams.SCATTER_OPACITY,
        ),
    )
    
    fig.update_layout(
        height=height,
        showlegend=show_legend,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
        ),
        xaxis_title="维度 1",
        yaxis_title="维度 2",
        hovermode="closest",
        # 深色主题
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)",
        ),
    )
    
    return fig


def create_3d_scatter_plot(
    df: pd.DataFrame,
    x: str = "x",
    y: str = "y",
    z: str = "z",
    color: str = "type",
    title: Optional[str] = None,
    height: int = 700,
) -> go.Figure:
    """
    创建 3D 散点图
    
    Args:
        df: 包含 3D 坐标的 DataFrame
        x, y, z: 坐标列名
        color: 颜色分组列名
        title: 图表标题
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    color_map = ENTITY_TYPE_COLORS.copy()
    
    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color=color,
        color_discrete_map=color_map,
        title=title,
    )
    
    fig.update_traces(
        marker=dict(size=3, opacity=0.7),
    )
    
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )
    
    return fig


# =============================================================================
# 热力图
# =============================================================================

def create_heatmap(
    matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "相似度矩阵",
    height: int = VizParams.HEATMAP_HEIGHT,
    colorscale: str = "Viridis",
) -> go.Figure:
    """
    创建热力图
    
    用于展示相似度矩阵或向量值分布。
    
    Args:
        matrix: 2D 矩阵数据
        labels: 行/列标签
        title: 图表标题
        height: 图表高度
        colorscale: 颜色方案
        
    Returns:
        Plotly Figure 对象
    """
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale=colorscale,
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )
    
    return fig


def create_vector_heatmap(
    vector: np.ndarray,
    title: str = "嵌入向量可视化",
    height: int = 150,
) -> go.Figure:
    """
    将一维向量可视化为热力图
    
    用于展示 128 维嵌入向量的值分布。
    
    Args:
        vector: 1D 向量
        title: 图表标题
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    # 将向量reshape为多行便于显示
    n_cols = 32
    n_rows = len(vector) // n_cols
    matrix = vector[:n_rows * n_cols].reshape(n_rows, n_cols)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale="RdBu",
        zmid=0,  # 中心点为 0
        showscale=True,
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(
            title=f"维度 (共 {len(vector)} 维)",
            showticklabels=False,
        ),
        yaxis=dict(showticklabels=False),
    )
    
    return fig


# =============================================================================
# 网络图
# =============================================================================

def create_network_graph(
    nodes: List[Dict],
    edges: List[Dict],
    title: str = "关系网络",
    height: int = VizParams.NETWORK_HEIGHT,
) -> go.Figure:
    """
    创建网络关系图
    
    用于展示实体之间的相似关系。
    
    Args:
        nodes: 节点列表，每个节点包含 id, label, type, x, y
        edges: 边列表，每条边包含 source, target, weight
        title: 图表标题
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    # 创建边的轨迹
    edge_x = []
    edge_y = []
    edge_weights = []
    
    node_dict = {n["id"]: n for n in nodes}
    
    for edge in edges:
        source = node_dict.get(edge["source"])
        target = node_dict.get(edge["target"])
        
        if source and target:
            edge_x.extend([source["x"], target["x"], None])
            edge_y.extend([source["y"], target["y"], None])
            edge_weights.append(edge.get("weight", 1))
    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="rgba(150,150,150,0.5)"),
        hoverinfo="none",
    )
    
    # 创建节点的轨迹
    node_x = [n["x"] for n in nodes]
    node_y = [n["y"] for n in nodes]
    node_colors = [ENTITY_TYPE_COLORS.get(n.get("type", "OTHER"), "#b2bec3") for n in nodes]
    node_texts = [n.get("label", n["id"]) for n in nodes]
    node_sizes = [20 if n.get("is_center", False) else 10 for n in nodes]
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color="white"),
        ),
        text=node_texts,
        textposition="top center",
        textfont=dict(size=10, color="#e0e0e0"),
        hoverinfo="text",
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title=title,
        height=height,
        showlegend=False,
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    
    return fig


def create_radial_network(
    center_node: Dict,
    related_nodes: List[Dict],
    title: str = "相似关系图",
    height: int = VizParams.NETWORK_HEIGHT,
) -> go.Figure:
    """
    创建放射状网络图
    
    中心是查询节点，周围是相似节点，距离表示相似度。
    
    Args:
        center_node: 中心节点 {id, label, type}
        related_nodes: 相关节点列表 {id, label, type, similarity}
        title: 图表标题
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    import math
    
    # 计算节点位置
    nodes = []
    edges = []
    
    # 中心节点
    center_node["x"] = 0
    center_node["y"] = 0
    center_node["is_center"] = True
    nodes.append(center_node)
    
    # 周围节点
    n_related = len(related_nodes)
    for i, node in enumerate(related_nodes):
        angle = 2 * math.pi * i / n_related
        # 距离与相似度成反比
        similarity = node.get("similarity", 0.5)
        distance = 1 - similarity * 0.5  # 相似度越高，距离越近
        
        node["x"] = distance * math.cos(angle)
        node["y"] = distance * math.sin(angle)
        node["is_center"] = False
        nodes.append(node)
        
        # 创建边
        edges.append({
            "source": center_node["id"],
            "target": node["id"],
            "weight": similarity,
        })
    
    return create_network_graph(nodes, edges, title, height)


# =============================================================================
# 柱状图
# =============================================================================

def create_bar_chart(
    data: Dict[str, int],
    title: str = "统计分布",
    x_label: str = "类别",
    y_label: str = "数量",
    height: int = 400,
    color_map: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    创建柱状图
    
    Args:
        data: {类别: 数量} 字典
        title: 图表标题
        x_label: X 轴标签
        y_label: Y 轴标签
        height: 图表高度
        color_map: 颜色映射
        
    Returns:
        Plotly Figure 对象
    """
    if color_map is None:
        color_map = ENTITY_TYPE_COLORS
    
    categories = list(data.keys())
    values = list(data.values())
    colors = [color_map.get(cat, "#4ecdc4") for cat in categories]
    
    # 添加中文名称
    labels = [f"{cat} ({ENTITY_TYPE_NAMES.get(cat, cat)})" for cat in categories]
    
    fig = go.Figure(data=go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=values,
        textposition="auto",
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.1)",
        ),
    )
    
    return fig


def create_pie_chart(
    data: Dict[str, int],
    title: str = "分布占比",
    height: int = 400,
    color_map: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    创建饼图
    
    Args:
        data: {类别: 数量} 字典
        title: 图表标题
        height: 图表高度
        color_map: 颜色映射
        
    Returns:
        Plotly Figure 对象
    """
    if color_map is None:
        color_map = ENTITY_TYPE_COLORS
    
    categories = list(data.keys())
    values = list(data.values())
    colors = [color_map.get(cat, "#4ecdc4") for cat in categories]
    
    # 添加中文名称
    labels = [f"{cat} ({ENTITY_TYPE_NAMES.get(cat, cat)})" for cat in categories]
    
    fig = go.Figure(data=go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.4,  # 环形图
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )
    
    return fig


# =============================================================================
# 降维对比图
# =============================================================================

def create_comparison_plot(
    coords_dict: Dict[str, np.ndarray],
    labels: Optional[np.ndarray] = None,
    title: str = "降维方法对比",
    height: int = 400,
) -> go.Figure:
    """
    创建降维方法对比图
    
    并排显示 PCA、UMAP、t-SNE 的结果。
    
    Args:
        coords_dict: {方法名: 坐标数组} 字典
        labels: 类别标签
        title: 图表标题
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    n_methods = len(coords_dict)
    
    fig = make_subplots(
        rows=1,
        cols=n_methods,
        subplot_titles=list(coords_dict.keys()),
    )
    
    for i, (method, coords) in enumerate(coords_dict.items(), start=1):
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                color = ENTITY_TYPE_COLORS.get(str(label), "#b2bec3")
                fig.add_trace(
                    go.Scatter(
                        x=coords[mask, 0],
                        y=coords[mask, 1],
                        mode="markers",
                        marker=dict(size=4, color=color, opacity=0.7),
                        name=str(label),
                        showlegend=(i == 1),
                    ),
                    row=1,
                    col=i,
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode="markers",
                    marker=dict(size=4, opacity=0.7),
                    showlegend=False,
                ),
                row=1,
                col=i,
            )
    
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )
    
    return fig

