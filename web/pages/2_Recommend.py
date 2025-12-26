"""
推荐关系页面 (Recommendation)
============================

使用 ONNX 模型进行在线推理，计算实体间的相似度并展示推荐关系。

功能:
- 搜索任意实体
- ONNX 推理获取嵌入向量
- 计算 Top-K 相似度
- 展示关系网络图
- 相似度排名表格

使用的数据文件:
- word2vec.onnx: ONNX 推理模型
- recsys/token_to_id.json: Token 映射
- embeddings.npy: 嵌入向量（用于相似度计算）
"""
import streamlit as st
from pathlib import Path
import sys
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig, ENTITY_TYPE_NAMES, ENTITY_TYPE_COLORS
from utils.data_loader import (
    load_embeddings_npy,
    load_token_to_id,
    load_id_to_token,
    search_tokens,
    get_entity_type,
)
from utils.onnx_inference import get_model, get_embedding
from utils.similarity import find_top_k_similar
from utils.visualization import create_radial_network
from components.sidebar import render_page_header
from components.filters import render_type_filter, render_top_k_selector
from components.similarity_list import render_similarity_list, render_similarity_table
from components.entity_card import render_entity_card


# =============================================================================
# 页面配置
# =============================================================================

st.set_page_config(
    page_title="Recommendation - " + AppConfig.APP_TITLE,
    page_icon="🔗",
    layout=AppConfig.LAYOUT,
)


# =============================================================================
# 页面标题
# =============================================================================

render_page_header(
    title="推荐关系",
    description="使用 ONNX 模型进行在线推理，搜索实体并获取相似推荐。",
    icon="🔗",
)


# =============================================================================
# 加载数据
# =============================================================================

@st.cache_data
def load_all_data():
    """加载所有需要的数据"""
    embeddings = load_embeddings_npy()
    token_to_id = load_token_to_id()
    id_to_token = load_id_to_token()
    return embeddings, token_to_id, id_to_token


embeddings, token_to_id, id_to_token = load_all_data()

# 构建 token 列表
tokens_list = [""] * len(embeddings)
for token, idx in token_to_id.items():
    if idx < len(tokens_list):
        tokens_list[idx] = token


# =============================================================================
# 侧边栏
# =============================================================================

with st.sidebar:
    st.markdown("### 🔗 推荐设置")
    st.markdown("---")
    
    # Top-K 设置
    top_k = render_top_k_selector(key="recommend_top_k", default=10, max_value=30)
    
    st.markdown("---")
    
    # 类型过滤
    st.markdown("#### 🔍 结果类型过滤")
    filter_type = st.selectbox(
        "只显示特定类型",
        options=["全部"] + list(ENTITY_TYPE_NAMES.keys()),
        format_func=lambda x: f"{ENTITY_TYPE_NAMES.get(x, x)}" if x != "全部" else "全部类型",
        key="recommend_type_filter",
    )
    
    st.markdown("---")
    
    # ONNX 模型信息
    st.markdown("#### ⚙️ ONNX 模型")
    
    # 检查 onnxruntime 是否可用
    try:
        import onnxruntime
        st.caption(f"onnxruntime: v{onnxruntime.__version__}")
    except ImportError:
        st.error("onnxruntime 未安装")
    
    model = get_model()
    if model.session:
        model_info = model.get_model_info()
        st.success("模型已加载")
        st.caption(f"大小: {model_info.get('model_size_mb', 0)} MB")
    else:
        st.error("模型加载失败")
        st.caption(f"模型路径: {model.model_path}")


# =============================================================================
# 搜索框
# =============================================================================

st.markdown("## 🔍 搜索实体")

col1, col2 = st.columns([3, 1])

with col1:
    search_query = st.text_input(
        "输入 Token 名称",
        placeholder="例如: MOV_tt0111161, ACT_nm0000001, DIR_nm0000229",
        key="search_input",
    )

with col2:
    search_button = st.button("🔍 搜索", use_container_width=True)

# 搜索建议
if search_query and len(search_query) >= 2:
    suggestions = search_tokens(search_query, limit=5)
    
    if suggestions:
        st.markdown("**搜索建议:**")
        cols = st.columns(min(len(suggestions), 5))
        
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                entity_type = get_entity_type(suggestion)
                color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
                
                if st.button(
                    suggestion,
                    key=f"suggest_{i}",
                    help=ENTITY_TYPE_NAMES.get(entity_type, entity_type),
                ):
                    st.session_state["selected_token"] = suggestion
                    st.rerun()


# =============================================================================
# 获取选中的 Token
# =============================================================================

selected_token = st.session_state.get("selected_token", None)

if search_button and search_query:
    # 精确匹配
    if search_query in token_to_id:
        selected_token = search_query
        st.session_state["selected_token"] = selected_token
    else:
        # 尝试模糊匹配
        matches = search_tokens(search_query, limit=1)
        if matches:
            selected_token = matches[0]
            st.session_state["selected_token"] = selected_token
        else:
            st.warning(f"未找到匹配的 Token: {search_query}")


# =============================================================================
# 推荐结果
# =============================================================================

if selected_token:
    st.markdown("---")
    st.markdown(f"## 📌 查询实体: `{selected_token}`")
    
    # 获取嵌入向量
    model = get_model()
    
    # 检查失败原因
    if model.session is None:
        st.error("⚠️ ONNX 模型未加载，无法进行推理")
        st.info("请检查 ONNX 模型文件是否存在，或查看控制台错误信息")
        query_vec = None
    elif selected_token not in model.token_to_id:
        st.error(f"⚠️ Token `{selected_token}` 不在词汇表中")
        query_vec = None
    else:
        query_vec = get_embedding(selected_token)
        if query_vec is None:
            st.error("⚠️ ONNX 推理失败，请检查控制台错误信息")
    
    if query_vec is not None:
        # 显示实体信息
        col1, col2 = st.columns([1, 2])
        
        with col1:
            render_entity_card(
                token=selected_token,
                embedding=query_vec,
                show_vector=True,
            )
        
        with col2:
            # 计算相似度
            type_filter = filter_type if filter_type != "全部" else None
            
            similar_results = find_top_k_similar(
                query_vec=query_vec,
                embeddings=embeddings,
                tokens=tokens_list,
                k=top_k,
                exclude_self=True,
                query_token=selected_token,
                entity_type_filter=type_filter,
            )
            
            if similar_results:
                st.markdown("### 📊 相似度排名")
                render_similarity_table(similar_results, title="")
        
        # =============================================================================
        # 关系网络图
        # =============================================================================
        
        st.markdown("---")
        st.markdown("## 🕸️ 关系网络图")
        
        if similar_results:
            # 构建网络图数据
            center_node = {
                "id": selected_token,
                "label": selected_token.split("_")[-1] if "_" in selected_token else selected_token,
                "type": get_entity_type(selected_token),
            }
            
            related_nodes = []
            for item in similar_results[:10]:  # 只显示前 10 个
                token = item["token"]
                related_nodes.append({
                    "id": token,
                    "label": token.split("_")[-1] if "_" in token else token,
                    "type": get_entity_type(token),
                    "similarity": item["similarity"],
                })
            
            # 创建网络图
            fig = create_radial_network(
                center_node=center_node,
                related_nodes=related_nodes,
                title="",
                height=500,
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("节点大小表示与查询实体的关系强度，距离中心越近相似度越高")
        else:
            st.info("没有找到相似实体")

else:
    # 显示使用说明
    st.markdown("---")
    st.markdown("""
    ### 💡 使用说明
    
    1. 在搜索框中输入实体的 Token 名称
    2. Token 格式为: `类型_ID`，例如:
       - `MOV_tt0111161` - 电影
       - `ACT_nm0000001` - 演员
       - `DIR_nm0000229` - 导演
    3. 点击搜索或选择建议的 Token
    4. 查看相似度排名和关系网络图
    
    ### 📝 Token 类型说明
    """)
    
    cols = st.columns(4)
    for i, (entity_type, name) in enumerate(ENTITY_TYPE_NAMES.items()):
        with cols[i % 4]:
            color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
            st.markdown(
                f'<span style="color:{color}">●</span> **{entity_type}**: {name}',
                unsafe_allow_html=True,
            )

