"""
3D 嵌入投影器 - 优化版
=====================

类似 TensorFlow Embedding Projector 的 3D 交互式可视化。

优化:
- 使用预计算坐标（秒级加载）
- 完全名称化
- 搜索高亮

功能:
- 3D 空间可视化（支持旋转、缩放、平移）
- 多种降维方法切换 (PCA / UMAP / t-SNE)
- 搜索高亮
- 悬停显示详情
- 点击选中查看相似项

使用的数据文件:
- embeddings.npy: 原始嵌入向量
- 预计算缓存文件
"""
import streamlit as st
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig, ENTITY_TYPE_NAMES, ENTITY_TYPE_COLORS
from utils.data_loader import (
    load_embeddings_npy,
    load_token_to_id,
    load_id_to_token,
    get_entity_type,
)
from utils.name_mapping import get_display_name, fuzzy_search
from utils.similarity import find_similar_fast
from utils.precompute import get_precomputed_coords, get_tokens_list
from components.sidebar import render_page_header


# =============================================================================
# 页面配置
# =============================================================================

st.set_page_config(
    page_title="3D Projector - " + AppConfig.APP_TITLE,
    page_icon="🌐",
    layout="wide",
)


# =============================================================================
# 页面标题
# =============================================================================

render_page_header(
    title="3D 嵌入投影器",
    description="类似 TensorFlow Embedding Projector 的 3D 交互式可视化。支持旋转、缩放、搜索高亮。",
    icon="🌐",
)


# =============================================================================
# 加载数据
# =============================================================================

@st.cache_data
def load_data():
    """加载数据"""
    embeddings = load_embeddings_npy()
    token_to_id = load_token_to_id()
    id_to_token = load_id_to_token()
    tokens_list = get_tokens_list()
    return embeddings, token_to_id, id_to_token, tokens_list


embeddings, token_to_id, id_to_token, tokens_list = load_data()


# =============================================================================
# 侧边栏
# =============================================================================

with st.sidebar:
    st.markdown("### 🌐 3D 投影设置")
    st.markdown("---")
    
    # 降维方法
    method = st.radio(
        "降维方法",
        options=["PCA", "UMAP", "t-SNE"],
        index=0,
        help="PCA 最快 (已预计算)，t-SNE 效果最好",
    )
    
    st.markdown("---")
    
    # 采样数量
    sample_options = [1000, 3000, 5000] if method == "PCA" else [1000, 3000]
    sample_size = st.selectbox(
        "采样数量",
        options=sample_options,
        index=1 if len(sample_options) > 1 else 0,
        help="较小的采样数加载更快",
    )
    
    st.markdown("---")
    
    # 搜索
    st.markdown("#### 🔍 搜索高亮")
    search_query = st.text_input(
        "搜索",
        placeholder="输入电影名、演员名...",
        key="projector_search",
    )
    
    st.markdown("---")
    
    # 显示设置
    st.markdown("#### ⚙️ 显示设置")
    point_size = st.slider("点大小", 2, 12, 5)
    show_labels = st.checkbox("显示标签", value=False)
    
    st.markdown("---")
    
    # 类型过滤
    st.markdown("#### 📌 类型过滤")
    all_types = list(ENTITY_TYPE_NAMES.keys())
    selected_types = st.multiselect(
        "显示类型",
        options=all_types,
        default=all_types,
        format_func=lambda x: ENTITY_TYPE_NAMES.get(x, x),
    )


# =============================================================================
# 获取预计算坐标
# =============================================================================

st.markdown("## 🌐 3D 嵌入空间")

# 尝试使用预计算坐标
precomputed = get_precomputed_coords(method, dim=3, sample_size=sample_size)

if precomputed is not None:
    coords_3d = precomputed["coords"]
    sample_indices = precomputed["indices"]
    st.caption(f"✅ 使用预计算坐标 ({method} 3D, {sample_size} 样本)")
else:
    # 降级到实时计算
    st.warning(f"预计算坐标不可用，正在实时计算 {method}...")
    
    # 随机采样
    np.random.seed(42)
    sample_indices = np.random.choice(len(embeddings), min(sample_size, len(embeddings)), replace=False)
    sample_embeddings = embeddings[sample_indices]
    
    # 计算
    if method == "PCA":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=3, random_state=42)
        coords_3d = reducer.fit_transform(sample_embeddings)
    elif method == "UMAP":
        import umap
        reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
        coords_3d = reducer.fit_transform(sample_embeddings)
    else:  # t-SNE
        from sklearn.manifold import TSNE
        import sklearn
        sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
        if sklearn_version >= (1, 5):
            tsne = TSNE(n_components=3, perplexity=30, max_iter=1000, random_state=42, init="pca")
        else:
            tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42, init="pca")
        coords_3d = tsne.fit_transform(sample_embeddings)

# 获取采样点的信息
sample_tokens = [id_to_token.get(int(idx), f"UNK_{idx}") for idx in sample_indices]
sample_types = [get_entity_type(t) for t in sample_tokens]
sample_names = [get_display_name(t) for t in sample_tokens]


# =============================================================================
# 创建 DataFrame
# =============================================================================

df = pd.DataFrame({
    "x": coords_3d[:, 0],
    "y": coords_3d[:, 1],
    "z": coords_3d[:, 2],
    "token": sample_tokens,
    "name": sample_names,
    "type": sample_types,
    "index": sample_indices,
})


# =============================================================================
# 搜索过滤
# =============================================================================

df["highlighted"] = False
df["opacity"] = 0.7
df["size"] = point_size

if search_query and len(search_query) >= 2:
    query_lower = search_query.lower()
    
    # 匹配名称
    mask = df["name"].str.lower().str.contains(query_lower, na=False)
    
    # 高亮匹配项
    df.loc[mask, "highlighted"] = True
    df.loc[mask, "opacity"] = 1.0
    df.loc[mask, "size"] = point_size * 2
    
    # 降低非匹配项的可见度
    df.loc[~mask, "opacity"] = 0.1
    
    n_matches = mask.sum()
    st.info(f"🔍 找到 {n_matches} 个匹配项")


# =============================================================================
# 类型过滤
# =============================================================================

if selected_types:
    df_filtered = df[df["type"].isin(selected_types)]
else:
    df_filtered = df

st.caption(f"显示 {len(df_filtered):,} 个点 (共采样 {len(df):,} 个)")


# =============================================================================
# 创建 3D 图表
# =============================================================================

fig = go.Figure()

# 按类型分组绘制
for entity_type in df_filtered["type"].unique():
    type_df = df_filtered[df_filtered["type"] == entity_type]
    color = ENTITY_TYPE_COLORS.get(entity_type, "#888888")
    type_name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
    
    # 创建悬停文本
    hover_texts = [
        f"<b>{row['name']}</b><br>"
        f"类型: {type_name}"
        for _, row in type_df.iterrows()
    ]
    
    fig.add_trace(go.Scatter3d(
        x=type_df["x"],
        y=type_df["y"],
        z=type_df["z"],
        mode="markers+text" if show_labels else "markers",
        name=type_name,
        text=type_df["name"] if show_labels else None,
        textposition="top center",
        textfont=dict(size=8, color=color),
        hovertext=hover_texts,
        hoverinfo="text",
        marker=dict(
            size=type_df["size"],
            color=color,
            opacity=type_df["opacity"],
            line=dict(width=0.5, color="white") if search_query else None,
        ),
        customdata=type_df[["token", "index"]].values,
    ))

# 图表布局
fig.update_layout(
    scene=dict(
        xaxis=dict(
            title=f"{method} 维度 1",
            backgroundcolor="rgba(0,0,0,0)",
            gridcolor="rgba(255,255,255,0.1)",
            showbackground=True,
        ),
        yaxis=dict(
            title=f"{method} 维度 2",
            backgroundcolor="rgba(0,0,0,0)",
            gridcolor="rgba(255,255,255,0.1)",
            showbackground=True,
        ),
        zaxis=dict(
            title=f"{method} 维度 3",
            backgroundcolor="rgba(0,0,0,0)",
            gridcolor="rgba(255,255,255,0.1)",
            showbackground=True,
        ),
        bgcolor="rgba(0,0,0,0)",
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e0e0e0"),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(0,0,0,0.5)",
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    height=700,
)

# 添加相机控制
fig.update_layout(
    scene_camera=dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=1.5),
    ),
)

# 显示图表
event = st.plotly_chart(
    fig,
    use_container_width=True,
    on_select="rerun",
    key="projector_3d",
)


# =============================================================================
# 点击事件处理
# =============================================================================

selected_token = None

if event and event.selection and event.selection.points:
    point = event.selection.points[0]
    if "customdata" in point and point["customdata"]:
        selected_token = point["customdata"][0]

if selected_token:
    st.markdown("---")
    
    selected_name = get_display_name(selected_token)
    entity_type = get_entity_type(selected_token)
    type_name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
    color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
    
    st.markdown(f"## 📌 选中: {selected_name}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}11);
            border-left: 4px solid {color};
            padding: 1rem;
            border-radius: 0.5rem;
        ">
            <h4 style="margin: 0; color: {color};">{selected_name}</h4>
            <p style="margin: 0.5rem 0 0 0; color: #888;">类型: {type_name}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # 使用快速搜索获取相似项
        similar = find_similar_fast(selected_token, k=5)
        
        if similar:
            st.markdown("### 🔗 相似项")
            for item in similar:
                item_color = ENTITY_TYPE_COLORS.get(item.get("type", "OTHER"), "#888")
                st.markdown(
                    f'<span style="color:{item_color}">●</span> {item["name"]} '
                    f'<small style="color:#888">({item["similarity"]:.2%})</small>',
                    unsafe_allow_html=True,
                )


# =============================================================================
# 操作提示
# =============================================================================

st.markdown("---")
st.markdown("""
### 💡 操作提示

| 操作 | 方法 |
|------|------|
| **旋转** | 按住左键拖拽 |
| **缩放** | 滚轮滚动 |
| **平移** | 按住右键拖拽 |
| **重置视角** | 双击图表 |
| **选中点** | 单击数据点 |
""")


# =============================================================================
# 方法对比
# =============================================================================

with st.expander("📊 降维方法对比", expanded=False):
    st.markdown("""
    | 方法 | 速度 | 局部结构 | 全局结构 | 推荐场景 |
    |------|------|----------|----------|----------|
    | **PCA** | ⚡⚡⚡ 最快 | ⭐ | ⭐⭐⭐ | 快速预览 |
    | **UMAP** | ⚡⚡ 较快 | ⭐⭐⭐ | ⭐⭐ | 平衡选择 |
    | **t-SNE** | ⚡ 较慢 | ⭐⭐⭐ | ⭐ | 详细分析 |
    
    **注**: 所有方法均已预计算，切换时秒级加载。
    """)
