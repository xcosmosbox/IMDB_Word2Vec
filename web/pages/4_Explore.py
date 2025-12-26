"""
嵌入探索页面 (Embedding Explore) - 优化版
=========================================

探索嵌入空间的语义关系，支持向量算术运算。

优化:
- 完全名称化（用户不接触 Token）
- 模糊搜索支持

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
    get_entity_type,
)
from utils.name_mapping import (
    get_display_name,
    fuzzy_search,
    get_entity_display_info,
)
from utils.similarity import vector_arithmetic, find_similar_by_vector_fast
from utils.visualization import create_vector_heatmap
from components.sidebar import render_page_header


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
@st.cache_data
def build_tokens_list():
    tokens_list = [""] * len(embeddings)
    for token, idx in token_to_id.items():
        if idx < len(tokens_list):
            tokens_list[idx] = token
    return tokens_list

tokens_list = build_tokens_list()


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
    - 国王 - 男人 + 女人 ≈ 女王
    - 巴黎 - 法国 + 意大利 ≈ 罗马
    
    **在 IMDB 数据中:**
    - 动作 + 喜剧 ≈ 动作喜剧电影
    - 演员A的电影 - 演员A + 演员B ≈ 演员B的电影
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
# 解析名称输入
# =============================================================================

def parse_and_resolve_names(input_str: str) -> tuple:
    """
    解析用户输入的名称，返回 (tokens, not_found, found_info)
    """
    if not input_str:
        return [], [], []
    
    names = [n.strip() for n in input_str.split(",") if n.strip()]
    
    tokens = []
    not_found = []
    found_info = []
    
    for name in names:
        # 尝试模糊匹配
        results = fuzzy_search(name, limit=1, threshold=70)
        
        if results:
            tokens.append(results[0]["token"])
            found_info.append(results[0])
        else:
            not_found.append(name)
    
    return tokens, not_found, found_info


# =============================================================================
# 向量算术界面
# =============================================================================

st.markdown("## ➕ 向量算术运算")

st.markdown("""
输入要进行运算的实体名称，用逗号分隔多个实体。

**结果向量 = Σ(正向项) - Σ(负向项)**
""")

# 检查是否有预设值
if "preset_positive" not in st.session_state:
    st.session_state["preset_positive"] = ""
if "preset_negative" not in st.session_state:
    st.session_state["preset_negative"] = ""

# 正向项
st.markdown("### ➕ 正向项 (相加)")

positive_input = st.text_input(
    "输入实体名称 (用逗号分隔)",
    value=st.session_state.get("preset_positive", ""),
    placeholder="例如: 动作, 喜剧",
    key="positive_input",
)

# 解析正向输入
positive_tokens, positive_not_found, positive_info = parse_and_resolve_names(positive_input)

# 负向项
st.markdown("### ➖ 负向项 (相减)")

negative_input = st.text_input(
    "输入实体名称 (用逗号分隔，可选)",
    value=st.session_state.get("preset_negative", ""),
    placeholder="例如: 恐怖",
    key="negative_input",
)

# 解析负向输入
negative_tokens, negative_not_found, negative_info = parse_and_resolve_names(negative_input)

# 显示解析结果
if positive_info or negative_info or positive_not_found or negative_not_found:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**正向项:**")
        if positive_info:
            for info in positive_info:
                color = ENTITY_TYPE_COLORS.get(info["type"], "#888")
                st.markdown(
                    f'<span style="color:{color}">●</span> {info["name"]} [{info["type_name"]}]',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("(无)")
        
        if positive_not_found:
            for name in positive_not_found:
                st.markdown(f'<span style="color:red">✗</span> {name} (未找到)', unsafe_allow_html=True)
                # 尝试提供建议
                similar = fuzzy_search(name, limit=2, threshold=40)
                if similar:
                    st.caption(f"  → 您是否要找: {', '.join([s['name'] for s in similar])}？")
    
    with col2:
        st.markdown("**负向项:**")
        if negative_info:
            for info in negative_info:
                color = ENTITY_TYPE_COLORS.get(info["type"], "#888")
                st.markdown(
                    f'<span style="color:{color}">●</span> {info["name"]} [{info["type_name"]}]',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("(无)")
        
        if negative_not_found:
            for name in negative_not_found:
                st.markdown(f'<span style="color:red">✗</span> {name} (未找到)', unsafe_allow_html=True)


# =============================================================================
# 计算结果
# =============================================================================

if st.button("🧮 计算", use_container_width=True) and positive_tokens:
    st.markdown("---")
    st.markdown("## 📊 计算结果")
    
    # 执行向量算术
    result_vec, similar_results = vector_arithmetic(
        embeddings=embeddings,
        tokens=tokens_list,
        token_to_id=token_to_id,
        positive=positive_tokens,
        negative=negative_tokens,
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
        
        # 显示结果（使用名称）
        for i, item in enumerate(similar_results[:result_count]):
            name = get_display_name(item["token"])
            entity_type = get_entity_type(item["token"])
            color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
            type_name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
            
            sim_pct = item["similarity"] * 100
            
            col_a, col_b, col_c = st.columns([3, 1, 1])
            with col_a:
                st.markdown(
                    f'{i+1}. <span style="color:{color}">●</span> **{name}**',
                    unsafe_allow_html=True,
                )
            with col_b:
                st.caption(type_name)
            with col_c:
                st.caption(f"{sim_pct:.1f}%")

elif not positive_tokens and (positive_input or negative_input):
    if positive_not_found and not positive_info:
        st.info("请输入有效的实体名称")
    elif not positive_input:
        st.info("请输入至少一个正向实体")


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
        st.session_state["preset_positive"] = "动作, 喜剧"
        st.session_state["preset_negative"] = ""
        st.rerun()
    
    if st.button("恐怖 - 惊悚", key="example2"):
        st.session_state["preset_positive"] = "恐怖"
        st.session_state["preset_negative"] = "惊悚"
        st.rerun()

with col2:
    st.markdown("**年代探索**")
    if st.button("1990年代 + 动作", key="example3"):
        st.session_state["preset_positive"] = "1990年代, 动作"
        st.session_state["preset_negative"] = ""
        st.rerun()
    
    if st.button("2020年代 - 2010年代", key="example4"):
        st.session_state["preset_positive"] = "2020年代"
        st.session_state["preset_negative"] = "2010年代"
        st.rerun()

with col3:
    st.markdown("**类型组合**")
    if st.button("科幻 + 爱情", key="example5"):
        st.session_state["preset_positive"] = "科幻, 爱情"
        st.session_state["preset_negative"] = ""
        st.rerun()
    
    if st.button("剧情 - 喜剧", key="example6"):
        st.session_state["preset_positive"] = "剧情"
        st.session_state["preset_negative"] = "喜剧"
        st.rerun()


# =============================================================================
# 实体搜索辅助
# =============================================================================

st.markdown("---")
st.markdown("## 🔍 实体搜索")

search_query = st.text_input(
    "搜索实体",
    placeholder="输入电影名、演员名、类型等...",
    key="explore_search",
)

if search_query and len(search_query) >= 2:
    matches = fuzzy_search(search_query, limit=20)
    
    if matches:
        st.markdown(f"**找到 {len(matches)} 个匹配:**")
        
        # 分类显示
        type_groups = {}
        for result in matches:
            entity_type = result["type"]
            if entity_type not in type_groups:
                type_groups[entity_type] = []
            type_groups[entity_type].append(result)
        
        for entity_type, results in type_groups.items():
            name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
            color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
            
            with st.expander(f"{name} ({len(results)} 个)", expanded=True):
                for result in results:
                    st.markdown(
                        f'<span style="color:{color}">●</span> {result["name"]}',
                        unsafe_allow_html=True,
                    )
    else:
        st.info("未找到匹配的实体")
