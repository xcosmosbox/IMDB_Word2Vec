"""
聚类分析页面 (Cluster Analysis) - 优化版
========================================

使用 clustering.json 中的预计算 t-SNE 坐标展示交互式聚类散点图。

优化:
- 完全名称化（悬停和点击显示名称）
- 使用 KNN 索引加速相似搜索

功能:
- 交互式散点图（缩放、平移、悬停）
- 按实体类型着色
- 点击数据点查看详情
- 侧边栏显示相似实体推荐

使用的数据文件:
- clustering.json: t-SNE 降维坐标 + K-Means 聚类标签
- embeddings.npy: 用于计算相似度
"""
import streamlit as st
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig, ENTITY_TYPE_NAMES, ENTITY_TYPE_COLORS
from utils.data_loader import (
    load_clustering_data,
    load_embeddings_npy,
    load_token_to_id,
    get_entity_type,
)
from utils.name_mapping import get_display_name, batch_get_display_names
from utils.similarity import find_similar_fast
from components.sidebar import render_page_header
from components.filters import render_type_filter, render_top_k_selector


# =============================================================================
# 页面配置
# =============================================================================

st.set_page_config(
    page_title="Cluster Analysis - " + AppConfig.APP_TITLE,
    page_icon="🎯",
    layout=AppConfig.LAYOUT,
)


# =============================================================================
# 页面标题
# =============================================================================

render_page_header(
    title="聚类分析",
    description="交互式 t-SNE 降维可视化，探索嵌入空间中的聚类结构。点击数据点查看详情和相似推荐。",
    icon="🎯",
)


# =============================================================================
# 加载数据
# =============================================================================

@st.cache_data
def load_all_data():
    """加载所有需要的数据"""
    points_df, clusters, metadata = load_clustering_data()
    embeddings = load_embeddings_npy()
    token_to_id = load_token_to_id()
    return points_df, clusters, metadata, embeddings, token_to_id


points_df, clusters, metadata, embeddings, token_to_id = load_all_data()

if points_df.empty:
    st.error("无法加载聚类数据，请检查 clustering.json 文件")
    st.stop()


# =============================================================================
# 添加名称列
# =============================================================================

@st.cache_data
def add_names_to_df(df):
    """为 DataFrame 添加名称列"""
    df = df.copy()
    # 批量获取名称
    names = batch_get_display_names(df["token"].tolist())
    df["name"] = df["token"].map(names)
    return df

points_df = add_names_to_df(points_df)


# =============================================================================
# 侧边栏
# =============================================================================

with st.sidebar:
    st.markdown("### 🎯 聚类分析设置")
    st.markdown("---")
    
    # 实体类型筛选
    available_types = points_df["type"].unique().tolist()
    selected_types = render_type_filter(
        available_types=available_types,
        key="cluster_type_filter",
    )
    
    st.markdown("---")
    
    # Top-K 设置
    top_k = render_top_k_selector(key="cluster_top_k", default=10)
    
    st.markdown("---")
    
    # 聚类信息
    st.markdown("#### 📊 聚类统计")
    n_clusters = len(clusters)
    st.metric("聚类数量", n_clusters)
    st.metric("总样本数", len(points_df))
    
    # 显示各类型数量
    st.markdown("#### 📌 类型分布")
    type_counts = points_df["type"].value_counts()
    for t, count in type_counts.items():
        color = ENTITY_TYPE_COLORS.get(t, "#888")
        name = ENTITY_TYPE_NAMES.get(t, t)
        st.markdown(
            f'<span style="color:{color}">●</span> {name}: {count:,}',
            unsafe_allow_html=True,
        )


# =============================================================================
# 筛选数据
# =============================================================================

if selected_types:
    filtered_df = points_df[points_df["type"].isin(selected_types)]
else:
    filtered_df = points_df

st.markdown(f"**显示 {len(filtered_df):,} 个数据点** (共 {len(points_df):,} 个)")


# =============================================================================
# 散点图
# =============================================================================

# 创建悬停文本
filtered_df = filtered_df.copy()
filtered_df["hover_text"] = filtered_df.apply(
    lambda row: f"<b>{row['name']}</b><br>类型: {ENTITY_TYPE_NAMES.get(row['type'], row['type'])}<br>聚类: #{row['cluster']}",
    axis=1
)

# 创建散点图
fig = px.scatter(
    filtered_df,
    x="x",
    y="y",
    color="type",
    color_discrete_map=ENTITY_TYPE_COLORS,
    hover_data={"name": True, "type": False, "x": False, "y": False, "cluster": True},
    custom_data=["token", "name"],
    title=f"t-SNE 聚类可视化 ({len(filtered_df):,} 个点)",
)

# 更新悬停模板
fig.update_traces(
    hovertemplate="<b>%{customdata[1]}</b><br>聚类: %{customdata[0]}<extra></extra>",
    marker=dict(size=6, opacity=0.7),
)

# 更新布局
fig.update_layout(
    height=700,
    legend_title="实体类型",
    legend=dict(
        itemsizing="constant",
    ),
    xaxis_title="t-SNE 维度 1",
    yaxis_title="t-SNE 维度 2",
)

# 更新图例标签为中文
fig.for_each_trace(
    lambda t: t.update(name=ENTITY_TYPE_NAMES.get(t.name, t.name))
)

# 使用 plotly_chart 显示，并捕获点击事件
event = st.plotly_chart(
    fig,
    use_container_width=True,
    on_select="rerun",
    key="cluster_scatter",
)


# =============================================================================
# 点击事件处理
# =============================================================================

selected_point = None
selected_token = None

if event and event.selection and event.selection.points:
    point_data = event.selection.points[0]
    point_index = point_data.get("point_index", None)
    
    if point_index is not None:
        # 获取对应的 trace（类型）
        curve_number = point_data.get("curve_number", 0)
        types_in_order = filtered_df["type"].unique().tolist()
        
        if curve_number < len(types_in_order):
            selected_type = types_in_order[curve_number]
            type_df = filtered_df[filtered_df["type"] == selected_type]
            
            if point_index < len(type_df):
                selected_point = type_df.iloc[point_index]
                selected_token = selected_point["token"]

# 显示选中的数据点信息
if selected_point is not None and selected_token:
    st.markdown("---")
    st.markdown("## 📌 选中的实体")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        name = selected_point["name"]
        cluster_id = selected_point["cluster"]
        entity_type = selected_point["type"]
        type_name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
        color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
        
        # 实体卡片
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}11);
            border-left: 4px solid {color};
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        ">
            <h3 style="margin: 0; color: {color};">{name}</h3>
            <p style="margin: 0.5rem 0 0 0; color: #888;">
                类型: {type_name}<br>
                聚类: #{cluster_id}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 坐标信息
        st.caption(f"坐标: ({selected_point['x']:.2f}, {selected_point['y']:.2f})")
    
    with col2:
        # 使用快速搜索获取相似实体
        st.markdown("### 🔗 相似实体推荐")
        
        similar_results = find_similar_fast(
            query_token=selected_token,
            k=top_k,
        )
        
        if similar_results:
            for result in similar_results:
                result_color = ENTITY_TYPE_COLORS.get(result.get("type", "OTHER"), "#888")
                result_type_name = ENTITY_TYPE_NAMES.get(result.get("type", "OTHER"), "")
                sim_pct = result["similarity"] * 100
                
                col_a, col_b, col_c = st.columns([3, 1, 1])
                with col_a:
                    st.markdown(
                        f'<span style="color:{result_color}">●</span> '
                        f'**{result["name"]}**',
                        unsafe_allow_html=True,
                    )
                with col_b:
                    st.caption(result_type_name[:4])
                with col_c:
                    st.caption(f"{sim_pct:.1f}%")
        else:
            st.info("无法获取相似实体")

else:
    st.info("💡 **提示:** 点击散点图中的数据点查看详情和相似推荐")


# =============================================================================
# 聚类中心信息
# =============================================================================

st.markdown("---")
st.markdown("## 📊 聚类中心统计")

if clusters:
    clusters_df = pd.DataFrame(clusters)
    clusters_df["dominant_type_name"] = clusters_df["dominant_type"].map(ENTITY_TYPE_NAMES)
    
    # 排序
    clusters_df = clusters_df.sort_values("size", ascending=False)
    
    # 显示表格
    st.dataframe(
        clusters_df[["cluster_id", "size", "dominant_type_name", "center_x", "center_y"]],
        column_config={
            "cluster_id": st.column_config.NumberColumn("聚类 ID", format="%d"),
            "size": st.column_config.NumberColumn("样本数", format="%d"),
            "dominant_type_name": st.column_config.TextColumn("主要类型"),
            "center_x": st.column_config.NumberColumn("中心 X", format="%.2f"),
            "center_y": st.column_config.NumberColumn("中心 Y", format="%.2f"),
        },
        use_container_width=True,
        hide_index=True,
    )
