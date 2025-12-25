"""
嵌入探索页面 (Embedding Explore)
===============================

探索嵌入空间的语义关系，支持向量算术运算。

功能:
- 向量算术: A - B + C 运算
- 类比推理: King - Man + Woman ≈ Queen
- 最近邻搜索

使用的数据文件:
- embeddings.npy: 原始嵌入向量
- recsys/token_to_id.json: Token 映射
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
from utils.similarity import vector_arithmetic, find_top_k_similar
from utils.visualization import create_vector_heatmap
from components.sidebar import render_page_header
from components.similarity_list import render_similarity_list


# =============================================================================
# 页面配置
# =============================================================================

st.set_page_config(
    page_title="Embedding Explore - " + AppConfig.APP_TITLE,
    page_icon="🔬",
    layout=AppConfig.LAYOUT,
)


# =============================================================================
# 页面标题
# =============================================================================

render_page_header(
    title="嵌入探索",
    description="探索嵌入空间的语义关系，支持向量算术运算 (A - B + C)。",
    icon="🔬",
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

# 构建 token 列表
tokens_list = [""] * len(embeddings)
for token, idx in token_to_id.items():
    if idx < len(tokens_list):
        tokens_list[idx] = token


# =============================================================================
# 侧边栏
# =============================================================================

with st.sidebar:
    st.markdown("### 🔬 嵌入探索设置")
    st.markdown("---")
    
    st.markdown("#### 💡 向量算术说明")
    st.markdown("""
    向量算术可以揭示嵌入空间中的语义关系。
    
    **经典例子:**
    - King - Man + Woman ≈ Queen
    - Paris - France + Italy ≈ Rome
    
    **在 IMDB 数据中:**
    - 动作片 - 动作 + 喜剧 ≈ 喜剧片
    - 演员A - 电影A + 电影B ≈ 电影B的演员
    """)
    
    st.markdown("---")
    
    # 结果数量
    result_count = st.slider(
        "结果数量",
        min_value=5,
        max_value=30,
        value=10,
        key="explore_result_count",
    )


# =============================================================================
# 向量算术界面
# =============================================================================

st.markdown("## ➕ 向量算术运算")

st.markdown("""
输入要进行运算的 Token，格式: **正向项 - 负向项**

结果向量 = Σ(正向项) - Σ(负向项)
""")

# 正向项
st.markdown("### ➕ 正向项 (相加)")

positive_input = st.text_input(
    "输入正向 Token (用逗号分隔)",
    placeholder="例如: MOV_tt0111161, GEN_Drama",
    key="positive_input",
)

# 负向项
st.markdown("### ➖ 负向项 (相减)")

negative_input = st.text_input(
    "输入负向 Token (用逗号分隔，可选)",
    placeholder="例如: GEN_Action",
    key="negative_input",
)

# 解析输入
def parse_tokens(input_str):
    """解析输入的 Token 字符串"""
    if not input_str:
        return []
    tokens = [t.strip() for t in input_str.split(",")]
    return [t for t in tokens if t]

positive_tokens = parse_tokens(positive_input)
negative_tokens = parse_tokens(negative_input)

# 验证 Token
valid_positive = [t for t in positive_tokens if t in token_to_id]
valid_negative = [t for t in negative_tokens if t in token_to_id]

invalid_tokens = [t for t in positive_tokens + negative_tokens if t not in token_to_id]

if invalid_tokens:
    st.warning(f"以下 Token 未找到: {', '.join(invalid_tokens)}")

# 显示解析结果
if valid_positive or valid_negative:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**正向项:**")
        for token in valid_positive:
            entity_type = get_entity_type(token)
            color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
            st.markdown(f'<span style="color:{color}">●</span> `{token}`', unsafe_allow_html=True)
        if not valid_positive:
            st.caption("(无)")
    
    with col2:
        st.markdown("**负向项:**")
        for token in valid_negative:
            entity_type = get_entity_type(token)
            color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
            st.markdown(f'<span style="color:{color}">●</span> `{token}`', unsafe_allow_html=True)
        if not valid_negative:
            st.caption("(无)")


# =============================================================================
# 计算结果
# =============================================================================

if st.button("🧮 计算", use_container_width=True) and valid_positive:
    st.markdown("---")
    st.markdown("## 📊 计算结果")
    
    # 执行向量算术
    result_vec, similar_results = vector_arithmetic(
        embeddings=embeddings,
        tokens=tokens_list,
        token_to_id=token_to_id,
        positive=valid_positive,
        negative=valid_negative,
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 结果向量")
        
        # 向量热力图
        fig = create_vector_heatmap(
            result_vec,
            title="",
            height=150,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 向量统计
        st.markdown("**向量统计:**")
        st.markdown(f"- 范数: {np.linalg.norm(result_vec):.4f}")
        st.markdown(f"- 均值: {np.mean(result_vec):.4f}")
        st.markdown(f"- 标准差: {np.std(result_vec):.4f}")
    
    with col2:
        st.markdown("### 最相似的实体")
        
        # 显示结果
        render_similarity_list(
            results=similar_results[:result_count],
            title="",
            show_rank=True,
        )

elif not valid_positive and (positive_input or negative_input):
    st.info("请输入至少一个有效的正向 Token")


# =============================================================================
# 预设示例
# =============================================================================

st.markdown("---")
st.markdown("## 🎯 预设示例")

st.markdown("点击下方按钮尝试预设的向量运算:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**类型探索**")
    if st.button("动作 + 喜剧", key="example1"):
        st.session_state["positive_input"] = "GEN_Action, GEN_Comedy"
        st.session_state["negative_input"] = ""
        st.rerun()
    
    if st.button("恐怖 - 惊悚", key="example2"):
        st.session_state["positive_input"] = "GEN_Horror"
        st.session_state["negative_input"] = "GEN_Thriller"
        st.rerun()

with col2:
    st.markdown("**年代探索**")
    if st.button("90年代 + 动作", key="example3"):
        st.session_state["positive_input"] = "ERA_1990s, GEN_Action"
        st.session_state["negative_input"] = ""
        st.rerun()
    
    if st.button("2020s - 2010s", key="example4"):
        st.session_state["positive_input"] = "ERA_2020s"
        st.session_state["negative_input"] = "ERA_2010s"
        st.rerun()

with col3:
    st.markdown("**评分探索**")
    if st.button("高分 (8.5+)", key="example5"):
        st.session_state["positive_input"] = "RAT_8.5, RAT_9.0"
        st.session_state["negative_input"] = ""
        st.rerun()
    
    if st.button("高分 - 低分", key="example6"):
        st.session_state["positive_input"] = "RAT_9.0"
        st.session_state["negative_input"] = "RAT_5.0"
        st.rerun()


# =============================================================================
# Token 搜索辅助
# =============================================================================

st.markdown("---")
st.markdown("## 🔍 Token 搜索")

search_query = st.text_input(
    "搜索 Token",
    placeholder="输入关键词搜索可用的 Token...",
    key="explore_search",
)

if search_query and len(search_query) >= 2:
    matches = search_tokens(search_query, limit=20)
    
    if matches:
        st.markdown(f"**找到 {len(matches)} 个匹配:**")
        
        # 分类显示
        type_groups = {}
        for token in matches:
            entity_type = get_entity_type(token)
            if entity_type not in type_groups:
                type_groups[entity_type] = []
            type_groups[entity_type].append(token)
        
        for entity_type, tokens in type_groups.items():
            name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
            color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
            
            with st.expander(f"{name} ({len(tokens)} 个)", expanded=True):
                for token in tokens:
                    st.code(token, language=None)
    else:
        st.info("未找到匹配的 Token")

