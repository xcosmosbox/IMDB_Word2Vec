"""
推荐关系页面 (Recommendation) - 优化版
======================================

使用 ONNX 模型进行在线推理，计算实体间的相似度并展示推荐关系。

优化:
- 完全名称化（用户不接触 Token）
- 使用 KNN 索引加速搜索
- 模糊搜索支持

功能:
- 搜索任意实体（输入名称）
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
    get_entity_type,
)
from utils.name_mapping import (
    get_display_name,
    fuzzy_search,
    get_entity_display_info,
    get_hot_entities,
)
from utils.onnx_inference import get_model, get_embedding
from utils.similarity import find_similar_fast, find_top_k_similar
from utils.visualization import create_radial_network
from components.sidebar import render_page_header
from components.filters import render_top_k_selector


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
    description="搜索电影、演员或导演，获取相似推荐。",
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

# 初始化选中状态
if "recommend_selected_token" not in st.session_state:
    st.session_state["recommend_selected_token"] = None

col1, col2 = st.columns([3, 1])

with col1:
    search_query = st.text_input(
        "搜索",
        placeholder="输入电影名、演员名或导演名...",
        key="recommend_search_input",
        help="支持模糊搜索，例如输入「肖申克」或「Shawshank」",
    )

with col2:
    search_button = st.button("🔍 搜索", use_container_width=True)

# 搜索建议
if search_query and len(search_query) >= 2:
    results = fuzzy_search(search_query, limit=6)
    
    if results:
        st.markdown("**搜索建议:**")
        cols = st.columns(min(len(results), 6))
        
        for i, result in enumerate(results):
            with cols[i]:
                entity_type = result["type"]
                color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
                
                # 显示名称（截断长名称）
                display_name = result["name"]
                if len(display_name) > 12:
                    display_name = display_name[:12] + "..."
                
                if st.button(
                    display_name,
                    key=f"suggest_{i}",
                    help=f"[{result['type_name']}] {result['name']}",
                    use_container_width=True,
                ):
                    st.session_state["recommend_selected_token"] = result["token"]
                    st.rerun()
    else:
        # 无匹配，尝试获取相近建议
        similar = fuzzy_search(search_query, limit=3, threshold=40)
        if similar:
            st.info(f"未找到精确匹配。您是否要找: {', '.join([s['name'] for s in similar])}？")
        else:
            st.warning("未找到匹配的实体，请尝试其他关键词")

# 空搜索时显示热门推荐
elif not search_query:
    st.markdown("**热门推荐:**")
    hot_entities = get_hot_entities(limit=6)
    
    if hot_entities:
        cols = st.columns(min(len(hot_entities), 6))
        for i, entity in enumerate(hot_entities):
            with cols[i]:
                display_name = entity["name"]
                if len(display_name) > 12:
                    display_name = display_name[:12] + "..."
                
                if st.button(
                    display_name,
                    key=f"hot_{i}",
                    help=f"[{entity['type_name']}] {entity['name']}",
                    use_container_width=True,
                ):
                    st.session_state["recommend_selected_token"] = entity["token"]
                    st.rerun()


# =============================================================================
# 获取选中的 Token
# =============================================================================

selected_token = st.session_state.get("recommend_selected_token", None)

if search_button and search_query:
    # 尝试精确匹配
    results = fuzzy_search(search_query, limit=1, threshold=80)
    if results:
        selected_token = results[0]["token"]
        st.session_state["recommend_selected_token"] = selected_token
    else:
        st.warning(f"未找到匹配的实体: {search_query}")


# =============================================================================
# 推荐结果
# =============================================================================

if selected_token:
    st.markdown("---")
    
    # 获取实体信息
    entity_info = get_entity_display_info(selected_token)
    entity_color = ENTITY_TYPE_COLORS.get(entity_info["type"], "#888")
    
    st.markdown(f"## 📌 查询: {entity_info['name']}")
    st.caption(f"类型: {entity_info['type_name']}")
    
    # 获取嵌入向量
    model = get_model()
    
    if model.session is None:
        st.error("⚠️ ONNX 模型未加载，无法进行推理")
        st.info("请检查 ONNX 模型文件是否存在，或查看控制台错误信息")
        query_vec = None
    elif selected_token not in model.token_to_id:
        st.error(f"⚠️ 实体 「{entity_info['name']}」 不在词汇表中")
        query_vec = None
    else:
        query_vec = get_embedding(selected_token)
        if query_vec is None:
            st.error("⚠️ ONNX 推理失败，请检查控制台错误信息")
    
    if query_vec is not None:
        # 显示实体信息和相似结果
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # 实体信息卡片
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {entity_color}22, {entity_color}11);
                border-left: 4px solid {entity_color};
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            ">
                <h3 style="margin: 0; color: {entity_color};">{entity_info['name']}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #888;">
                    类型: {entity_info['type_name']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # 清除按钮
            if st.button("🔄 清除选择", use_container_width=True):
                st.session_state["recommend_selected_token"] = None
                st.rerun()
        
        with col2:
            # 使用快速搜索
            type_filter = filter_type if filter_type != "全部" else None
            
            similar_results = find_similar_fast(
                query_token=selected_token,
                k=top_k,
                entity_type_filter=type_filter,
            )
            
            if similar_results:
                st.markdown("### 📊 相似推荐")
                
                # 使用表格显示结果
                for result in similar_results:
                    result_color = ENTITY_TYPE_COLORS.get(result.get("type", "OTHER"), "#888")
                    sim_pct = result["similarity"] * 100
                    
                    col_a, col_b, col_c = st.columns([3, 1, 1])
                    with col_a:
                        st.markdown(
                            f'<span style="color:{result_color}">●</span> '
                            f'**{result["name"]}**',
                            unsafe_allow_html=True,
                        )
                    with col_b:
                        st.caption(result.get("type", "")[:3])
                    with col_c:
                        st.caption(f"{sim_pct:.1f}%")
            else:
                st.info("没有找到相似实体")
        
        # =============================================================================
        # 关系网络图
        # =============================================================================
        
        st.markdown("---")
        st.markdown("## 🕸️ 关系网络图")
        
        if similar_results:
            # 构建网络图数据
            center_node = {
                "id": selected_token,
                "label": entity_info["name"],
                "type": entity_info["type"],
            }
            
            related_nodes = []
            for item in similar_results[:10]:  # 只显示前 10 个
                related_nodes.append({
                    "id": item["token"],
                    "label": item["name"],
                    "type": item.get("type", get_entity_type(item["token"])),
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
    
    1. **搜索**: 在搜索框中输入电影名、演员名或导演名
    2. **选择**: 点击搜索建议中的实体
    3. **查看**: 系统会显示最相似的实体和关系网络图
    
    ### 🎯 搜索示例
    
    - 电影: 「肖申克的救赎」、「Inception」、「泰坦尼克号」
    - 演员: 「Tom Hanks」、「Morgan Freeman」
    - 导演: 「Christopher Nolan」、「Steven Spielberg」
    - 类型: 「动作」、「喜剧」、「科幻」
    """)
    
    # 类型图例
    st.markdown("### 📌 实体类型")
    cols = st.columns(4)
    for i, (entity_type, name) in enumerate(ENTITY_TYPE_NAMES.items()):
        with cols[i % 4]:
            color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
            st.markdown(
                f'<span style="color:{color}">●</span> **{entity_type}**: {name}',
                unsafe_allow_html=True,
            )
