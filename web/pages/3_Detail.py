"""
数据详情页面 (Data Detail)
=========================

展示实体的完整信息，包括嵌入向量的可视化。

功能:
- 搜索并选择实体
- 显示实体详细信息
- 128 维嵌入向量热力图可视化
- 向量统计信息

使用的数据文件:
- embeddings.json: 完整嵌入数据
- recsys/id_to_token.json: ID 映射
"""
import streamlit as st
from pathlib import Path
import sys
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig, ENTITY_TYPE_NAMES, ENTITY_TYPE_COLORS
from utils.data_loader import (
    load_embeddings_json,
    load_token_to_id,
    search_tokens,
    get_entity_type,
)
from utils.visualization import create_vector_heatmap, create_bar_chart
from components.sidebar import render_page_header
from components.entity_card import render_entity_card


# =============================================================================
# 页面配置
# =============================================================================

st.set_page_config(
    page_title="Data Detail - " + AppConfig.APP_TITLE,
    page_icon="📊",
    layout=AppConfig.LAYOUT,
)


# =============================================================================
# 页面标题
# =============================================================================

render_page_header(
    title="数据详情",
    description="查看实体的完整信息和 128 维嵌入向量可视化。",
    icon="📊",
)


# =============================================================================
# 加载数据
# =============================================================================

@st.cache_data(show_spinner="加载嵌入数据...")
def load_data():
    """加载嵌入数据"""
    tokens, embeddings, metadata = load_embeddings_json()
    token_to_id = load_token_to_id()
    return tokens, embeddings, metadata, token_to_id


with st.spinner("加载数据中..."):
    tokens, embeddings, metadata, token_to_id = load_data()

if len(tokens) == 0:
    st.error("无法加载嵌入数据，请检查 embeddings.json 文件")
    st.stop()


# =============================================================================
# 侧边栏
# =============================================================================

with st.sidebar:
    st.markdown("### 📊 数据详情设置")
    st.markdown("---")
    
    # 数据统计
    st.markdown("#### 📈 数据统计")
    st.metric("词汇表大小", f"{metadata.get('vocab_size', len(tokens)):,}")
    st.metric("嵌入维度", metadata.get("embedding_dim", embeddings.shape[1] if len(embeddings) > 0 else 0))
    
    st.markdown("---")
    
    # 实体类型统计
    st.markdown("#### 📌 实体类型分布")
    entity_types = metadata.get("entity_types", {})
    for entity_type, count in sorted(entity_types.items(), key=lambda x: -x[1]):
        name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
        color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
        st.markdown(
            f'<span style="color:{color}">●</span> {name}: {count:,}',
            unsafe_allow_html=True,
        )


# =============================================================================
# 搜索框
# =============================================================================

st.markdown("## 🔍 搜索实体")

search_query = st.text_input(
    "输入 Token 名称",
    placeholder="例如: MOV_tt0111161, ACT_nm0000001",
    key="detail_search",
)

# 搜索结果
if search_query and len(search_query) >= 2:
    matches = search_tokens(search_query, limit=10)
    
    if matches:
        st.markdown("**选择一个实体:**")
        
        selected = st.selectbox(
            "搜索结果",
            options=matches,
            format_func=lambda x: f"{x} ({ENTITY_TYPE_NAMES.get(get_entity_type(x), get_entity_type(x))})",
            key="detail_select",
            label_visibility="collapsed",
        )
        
        if selected:
            st.session_state["detail_token"] = selected
    else:
        st.warning("未找到匹配的 Token")


# =============================================================================
# 实体详情展示
# =============================================================================

selected_token = st.session_state.get("detail_token", None)

if selected_token and selected_token in token_to_id:
    st.markdown("---")
    st.markdown(f"## 📌 实体详情: `{selected_token}`")
    
    # 获取嵌入向量
    token_id = token_to_id[selected_token]
    
    if token_id < len(embeddings):
        embedding_vec = embeddings[token_id]
        
        # 基本信息
        col1, col2 = st.columns([1, 2])
        
        with col1:
            entity_type = get_entity_type(selected_token)
            entity_id = selected_token.split("_", 1)[1] if "_" in selected_token else selected_token
            
            st.markdown("### 基本信息")
            
            st.markdown(f"""
            - **Token:** `{selected_token}`
            - **类型:** {ENTITY_TYPE_NAMES.get(entity_type, entity_type)} ({entity_type})
            - **ID:** {entity_id}
            - **索引:** {token_id}
            """)
            
            # 向量统计
            st.markdown("### 向量统计")
            
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric("维度", len(embedding_vec))
                st.metric("最小值", f"{np.min(embedding_vec):.4f}")
                st.metric("最大值", f"{np.max(embedding_vec):.4f}")
            with stats_col2:
                st.metric("均值", f"{np.mean(embedding_vec):.4f}")
                st.metric("标准差", f"{np.std(embedding_vec):.4f}")
                st.metric("L2 范数", f"{np.linalg.norm(embedding_vec):.4f}")
        
        with col2:
            # 向量热力图
            st.markdown("### 嵌入向量可视化")
            
            fig = create_vector_heatmap(
                embedding_vec,
                title="",
                height=200,
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 向量值分布直方图
            st.markdown("### 向量值分布")
            
            import plotly.express as px
            
            hist_fig = px.histogram(
                x=embedding_vec,
                nbins=50,
                title="",
                labels={"x": "向量值", "count": "频次"},
            )
            hist_fig.update_layout(
                height=250,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            )
            st.plotly_chart(hist_fig, use_container_width=True)
        
        # =============================================================================
        # 原始向量数据
        # =============================================================================
        
        st.markdown("---")
        st.markdown("### 📋 原始向量数据")
        
        with st.expander("展开查看完整向量 (128 维)", expanded=False):
            # 格式化显示
            vector_str = ", ".join([f"{v:.6f}" for v in embedding_vec])
            st.code(f"[{vector_str}]", language="python")
            
            # 下载按钮
            import json
            vector_json = json.dumps({
                "token": selected_token,
                "embedding": embedding_vec.tolist(),
            }, indent=2)
            
            st.download_button(
                label="📥 下载向量 (JSON)",
                data=vector_json,
                file_name=f"{selected_token}_embedding.json",
                mime="application/json",
            )
    else:
        st.error("该实体的嵌入向量索引超出范围")

elif selected_token:
    st.warning(f"未找到 Token: {selected_token}")

else:
    # 显示使用说明
    st.markdown("---")
    st.markdown("""
    ### 💡 使用说明
    
    1. 在搜索框中输入实体的 Token 名称
    2. 从搜索结果中选择要查看的实体
    3. 查看实体的详细信息和向量可视化
    
    ### 📊 可视化说明
    
    - **热力图:** 将 128 维向量展示为 4×32 的网格，颜色表示值的大小
    - **直方图:** 显示向量值的分布情况
    - **统计信息:** 包括均值、标准差、范数等
    """)

    # 随机展示一个实体作为示例
    st.markdown("---")
    st.markdown("### 🎲 随机示例")
    
    if st.button("随机选择一个实体"):
        import random
        random_token = random.choice(list(token_to_id.keys()))
        st.session_state["detail_token"] = random_token
        st.rerun()

