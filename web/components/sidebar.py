"""
侧边栏组件
=========

提供通用侧边栏渲染功能，包括导航、实体过滤器等。
"""
from typing import List, Optional
import streamlit as st

# 导入配置
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENTITY_TYPE_NAMES, ENTITY_TYPE_COLORS


def render_sidebar():
    """
    渲染通用侧边栏
    
    包含:
    - 应用标题
    - 快速导航
    - 数据统计
    """
    with st.sidebar:
        st.markdown("### 🎬 IMDB Word2Vec")
        st.markdown("---")
        
        # 显示实体类型图例
        st.markdown("#### 实体类型")
        for entity_type, name in ENTITY_TYPE_NAMES.items():
            color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
            st.markdown(
                f'<span style="color:{color}">●</span> {name} ({entity_type})',
                unsafe_allow_html=True,
            )


def render_entity_filter(
    available_types: Optional[List[str]] = None,
    key: str = "entity_filter",
) -> List[str]:
    """
    渲染实体类型过滤器
    
    Args:
        available_types: 可选的实体类型列表，默认显示全部
        key: Streamlit 组件的唯一键
        
    Returns:
        选中的实体类型列表
    """
    if available_types is None:
        available_types = list(ENTITY_TYPE_NAMES.keys())
    
    st.markdown("#### 🔍 实体类型过滤")
    
    # 全选/取消全选
    col1, col2 = st.columns(2)
    with col1:
        if st.button("全选", key=f"{key}_select_all"):
            st.session_state[f"{key}_selected"] = available_types.copy()
    with col2:
        if st.button("取消全选", key=f"{key}_deselect_all"):
            st.session_state[f"{key}_selected"] = []
    
    # 初始化选中状态
    if f"{key}_selected" not in st.session_state:
        st.session_state[f"{key}_selected"] = available_types.copy()
    
    # 多选框
    selected = st.multiselect(
        "选择要显示的类型",
        options=available_types,
        default=st.session_state[f"{key}_selected"],
        format_func=lambda x: f"{ENTITY_TYPE_NAMES.get(x, x)} ({x})",
        key=f"{key}_multiselect",
    )
    
    st.session_state[f"{key}_selected"] = selected
    
    return selected


def render_selected_entity(
    token: Optional[str] = None,
    similarity: Optional[float] = None,
):
    """
    在侧边栏渲染选中的实体信息
    
    Args:
        token: 选中的 Token
        similarity: 相似度（如果是从推荐列表选择的）
    """
    if token is None:
        st.info("点击图表中的数据点查看详情")
        return
    
    st.markdown("#### 📌 选中实体")
    
    # 解析 Token
    if "_" in token:
        parts = token.split("_", 1)
        entity_type = parts[0]
        entity_id = parts[1]
    else:
        entity_type = "OTHER"
        entity_id = token
    
    type_name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
    color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
    
    st.markdown(
        f'<span style="color:{color};font-size:1.2em">●</span> '
        f'<strong>{type_name}</strong>',
        unsafe_allow_html=True,
    )
    
    st.code(token, language=None)
    
    if similarity is not None:
        st.metric("相似度", f"{similarity:.4f}")


def render_page_header(
    title: str,
    description: str,
    icon: str = "📊",
):
    """
    渲染页面标题和描述
    
    Args:
        title: 页面标题
        description: 页面描述
        icon: 图标
    """
    st.markdown(f"# {icon} {title}")
    st.markdown(description)
    st.markdown("---")


def render_data_stats(stats: dict):
    """
    渲染数据统计卡片
    
    Args:
        stats: {标签: 值} 字典
    """
    cols = st.columns(len(stats))
    
    for col, (label, value) in zip(cols, stats.items()):
        with col:
            st.metric(label, value)

