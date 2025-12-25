"""
降维对比页面 (Dimensionality Reduction Compare)
==============================================

对比 PCA、UMAP、t-SNE 三种降维方法的效果。

功能:
- 三种降维方法并排对比
- 参数调节
- 降维结果可视化

使用的数据文件:
- embeddings.npy: 原始嵌入向量
"""
import streamlit as st
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig, ENTITY_TYPE_NAMES, ENTITY_TYPE_COLORS, DimReductionParams, VizParams
from utils.data_loader import (
    load_embeddings_npy,
    load_token_to_id,
    load_id_to_token,
    get_entity_type,
)
from utils.dimensionality import (
    compute_pca,
    compute_umap,
    compute_tsne,
    sample_embeddings,
    get_cached_reduction,
)
from utils.visualization import create_scatter_plot, create_comparison_plot
from components.sidebar import render_page_header


# =============================================================================
# 页面配置
# =============================================================================

st.set_page_config(
    page_title="Compare - " + AppConfig.APP_TITLE,
    page_icon="📈",
    layout=AppConfig.LAYOUT,
)


# =============================================================================
# 页面标题
# =============================================================================

render_page_header(
    title="降维对比",
    description="对比 PCA、UMAP、t-SNE 三种降维方法的效果，观察不同算法对嵌入空间的投影差异。",
    icon="📈",
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
    return embeddings, token_to_id, id_to_token


embeddings, token_to_id, id_to_token = load_data()


# =============================================================================
# 侧边栏
# =============================================================================

with st.sidebar:
    st.markdown("### 📈 降维设置")
    st.markdown("---")
    
    # 采样数量
    st.markdown("#### 📊 采样设置")
    sample_size = st.slider(
        "采样数量",
        min_value=500,
        max_value=min(10000, len(embeddings)),
        value=min(3000, len(embeddings)),
        step=500,
        help="降维计算量大，建议使用采样以加快速度",
        key="sample_size",
    )
    
    st.markdown("---")
    
    # t-SNE 参数
    st.markdown("#### 🔧 t-SNE 参数")
    tsne_perplexity = st.slider(
        "Perplexity",
        min_value=5,
        max_value=50,
        value=DimReductionParams.TSNE_PERPLEXITY,
        help="困惑度，影响局部结构保留程度",
        key="tsne_perplexity",
    )
    
    st.markdown("---")
    
    # UMAP 参数
    st.markdown("#### 🔧 UMAP 参数")
    umap_n_neighbors = st.slider(
        "n_neighbors",
        min_value=5,
        max_value=50,
        value=DimReductionParams.UMAP_N_NEIGHBORS,
        help="近邻数量，影响局部结构保留",
        key="umap_n_neighbors",
    )
    
    umap_min_dist = st.slider(
        "min_dist",
        min_value=0.0,
        max_value=1.0,
        value=DimReductionParams.UMAP_MIN_DIST,
        step=0.05,
        help="最小距离，影响点的紧密程度",
        key="umap_min_dist",
    )
    
    st.markdown("---")
    
    # 算法说明
    st.markdown("#### 📖 算法说明")
    
    with st.expander("PCA"):
        st.markdown("""
        **主成分分析 (Principal Component Analysis)**
        
        - 线性降维方法
        - 速度最快
        - 保留全局方差最大方向
        - 可能丢失非线性结构
        """)
    
    with st.expander("UMAP"):
        st.markdown("""
        **统一流形近似和投影**
        
        - 非线性降维方法
        - 速度较快
        - 保留局部和全局结构
        - 适合可视化和聚类
        """)
    
    with st.expander("t-SNE"):
        st.markdown("""
        **t-分布随机邻域嵌入**
        
        - 非线性降维方法
        - 速度较慢
        - 擅长保留局部结构
        - 经典可视化方法
        """)


# =============================================================================
# 数据采样
# =============================================================================

st.markdown("## 📊 数据准备")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("原始数据量", f"{len(embeddings):,}")

with col2:
    st.metric("采样数量", f"{sample_size:,}")

with col3:
    st.metric("嵌入维度", embeddings.shape[1])

# 执行采样
sampled_embeddings, sample_indices = sample_embeddings(
    embeddings, sample_size, random_state=42
)

# 获取采样点的类型
sample_types = []
for idx in sample_indices:
    token = id_to_token.get(idx, "")
    entity_type = get_entity_type(token)
    sample_types.append(entity_type)

sample_types = np.array(sample_types)


# =============================================================================
# 降维计算
# =============================================================================

st.markdown("---")
st.markdown("## 🔄 降维计算")

# 计算按钮
if st.button("🚀 开始计算", use_container_width=True):
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    
    # 1. PCA
    status_text.markdown("**正在计算 PCA...**")
    progress_bar.progress(10)
    
    pca_coords = compute_pca(sampled_embeddings)
    results["PCA"] = pca_coords
    
    progress_bar.progress(30)
    
    # 2. UMAP
    status_text.markdown("**正在计算 UMAP... (可能需要几分钟)**")
    
    umap_coords = compute_umap(
        sampled_embeddings,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
    )
    results["UMAP"] = umap_coords
    
    progress_bar.progress(60)
    
    # 3. t-SNE
    status_text.markdown("**正在计算 t-SNE... (可能需要几分钟)**")
    
    tsne_coords = compute_tsne(
        sampled_embeddings,
        perplexity=tsne_perplexity,
    )
    results["t-SNE"] = tsne_coords
    
    progress_bar.progress(100)
    status_text.markdown("**✅ 计算完成!**")
    
    # 保存结果到 session state
    st.session_state["dim_reduction_results"] = results
    st.session_state["dim_reduction_types"] = sample_types
    st.session_state["dim_reduction_indices"] = sample_indices


# =============================================================================
# 结果展示
# =============================================================================

if "dim_reduction_results" in st.session_state:
    results = st.session_state["dim_reduction_results"]
    types = st.session_state["dim_reduction_types"]
    indices = st.session_state["dim_reduction_indices"]
    
    st.markdown("---")
    st.markdown("## 📊 降维结果对比")
    
    # 并排显示三种方法
    col1, col2, col3 = st.columns(3)
    
    for col, (method, coords) in zip([col1, col2, col3], results.items()):
        with col:
            st.markdown(f"### {method}")
            
            # 创建 DataFrame
            df = pd.DataFrame({
                "x": coords[:, 0],
                "y": coords[:, 1],
                "type": types,
            })
            
            # 创建散点图
            fig = create_scatter_plot(
                df,
                x="x",
                y="y",
                color="type",
                height=400,
                show_legend=False,
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # 图例
    st.markdown("---")
    st.markdown("#### 📌 图例")
    
    legend_cols = st.columns(len(ENTITY_TYPE_NAMES))
    for col, (entity_type, name) in zip(legend_cols, ENTITY_TYPE_NAMES.items()):
        with col:
            color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
            count = np.sum(types == entity_type)
            st.markdown(
                f'<span style="color:{color}">●</span> {name} ({count:,})',
                unsafe_allow_html=True,
            )
    
    # =============================================================================
    # 方法对比表格
    # =============================================================================
    
    st.markdown("---")
    st.markdown("## 📋 方法特性对比")
    
    comparison_data = {
        "特性": ["算法类型", "计算速度", "局部结构", "全局结构", "适用场景"],
        "PCA": ["线性", "⚡⚡⚡ 最快", "❌ 较弱", "✅ 较好", "快速预览、降噪"],
        "UMAP": ["非线性", "⚡⚡ 较快", "✅ 较好", "✅ 较好", "聚类分析、可视化"],
        "t-SNE": ["非线性", "⚡ 较慢", "✅✅ 最好", "❌ 较弱", "可视化、探索性分析"],
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info("👆 点击上方「开始计算」按钮进行降维计算")
    
    # 显示预期效果
    st.markdown("---")
    st.markdown("### 💡 预期效果")
    
    st.markdown("""
    计算完成后，您将看到:
    
    1. **PCA 结果**: 线性投影，保留最大方差方向
    2. **UMAP 结果**: 非线性投影，保留局部和全局结构
    3. **t-SNE 结果**: 非线性投影，最佳局部结构保留
    
    通过对比，您可以:
    - 观察不同算法对聚类结构的呈现差异
    - 评估哪种方法最适合您的分析需求
    - 了解参数调整对结果的影响
    """)

