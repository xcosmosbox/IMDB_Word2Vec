"""
筛选器组件
=========

提供各种筛选和搜索组件。
"""
from typing import List, Optional, Tuple
import streamlit as st

# 导入配置
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENTITY_TYPE_NAMES, ENTITY_TYPE_COLORS


def render_type_filter(
    available_types: Optional[List[str]] = None,
    default_selected: Optional[List[str]] = None,
    key: str = "type_filter",
    inline: bool = False,
) -> List[str]:
    """
    渲染实体类型筛选器
    
    Args:
        available_types: 可选类型列表
        default_selected: 默认选中的类型
        key: 组件唯一键
        inline: 是否水平排列
        
    Returns:
        选中的类型列表
    """
    if available_types is None:
        available_types = list(ENTITY_TYPE_NAMES.keys())
    
    if default_selected is None:
        default_selected = available_types.copy()
    
    if inline:
        # 水平排列的复选框
        cols = st.columns(len(available_types))
        selected = []
        
        for col, entity_type in zip(cols, available_types):
            with col:
                color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
                name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
                
                if st.checkbox(
                    f"● {name}",
                    value=entity_type in default_selected,
                    key=f"{key}_{entity_type}",
                ):
                    selected.append(entity_type)
        
        return selected
    else:
        # 多选框
        return st.multiselect(
            "实体类型",
            options=available_types,
            default=default_selected,
            format_func=lambda x: f"{ENTITY_TYPE_NAMES.get(x, x)} ({x})",
            key=key,
        )


def render_search_box(
    placeholder: str = "搜索 Token...",
    key: str = "search_box",
    suggestions: Optional[List[str]] = None,
) -> str:
    """
    渲染搜索框
    
    Args:
        placeholder: 占位符文本
        key: 组件唯一键
        suggestions: 搜索建议列表
        
    Returns:
        搜索查询字符串
    """
    query = st.text_input(
        "搜索",
        placeholder=placeholder,
        key=key,
        label_visibility="collapsed",
    )
    
    # 显示搜索建议
    if query and suggestions:
        matches = [s for s in suggestions if query.lower() in s.lower()][:5]
        if matches:
            st.caption("搜索建议:")
            for match in matches:
                if st.button(match, key=f"{key}_suggest_{match}"):
                    # 更新搜索框值
                    st.session_state[key] = match
                    st.rerun()
    
    return query


def render_slider_filter(
    label: str,
    min_value: float,
    max_value: float,
    default_value: Tuple[float, float],
    key: str,
    step: Optional[float] = None,
) -> Tuple[float, float]:
    """
    渲染范围滑块筛选器
    
    Args:
        label: 标签
        min_value: 最小值
        max_value: 最大值
        default_value: 默认范围
        key: 组件唯一键
        step: 步长
        
    Returns:
        选中的范围元组
    """
    return st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=default_value,
        step=step,
        key=key,
    )


def render_top_k_selector(
    key: str = "top_k",
    default: int = 10,
    max_value: int = 50,
) -> int:
    """
    渲染 Top-K 选择器
    
    Args:
        key: 组件唯一键
        default: 默认值
        max_value: 最大值
        
    Returns:
        选择的 K 值
    """
    return st.slider(
        "返回数量 (Top-K)",
        min_value=1,
        max_value=max_value,
        value=default,
        key=key,
    )


def render_method_selector(
    methods: List[str],
    default: str,
    key: str = "method",
    label: str = "选择方法",
) -> str:
    """
    渲染方法选择器
    
    Args:
        methods: 可选方法列表
        default: 默认选中
        key: 组件唯一键
        label: 标签
        
    Returns:
        选中的方法
    """
    return st.selectbox(
        label,
        options=methods,
        index=methods.index(default) if default in methods else 0,
        key=key,
    )


def render_pagination(
    total_items: int,
    items_per_page: int = 20,
    key: str = "page",
) -> Tuple[int, int]:
    """
    渲染分页控件
    
    Args:
        total_items: 总条目数
        items_per_page: 每页条目数
        key: 组件唯一键
        
    Returns:
        (起始索引, 结束索引) 元组
    """
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    if total_pages <= 1:
        return 0, total_items
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("◀ 上一页", key=f"{key}_prev"):
            if st.session_state.get(key, 1) > 1:
                st.session_state[key] = st.session_state.get(key, 1) - 1
    
    with col2:
        page = st.number_input(
            "页码",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.get(key, 1),
            key=f"{key}_input",
            label_visibility="collapsed",
        )
        st.session_state[key] = page
        st.caption(f"第 {page} / {total_pages} 页，共 {total_items} 条")
    
    with col3:
        if st.button("下一页 ▶", key=f"{key}_next"):
            if st.session_state.get(key, 1) < total_pages:
                st.session_state[key] = st.session_state.get(key, 1) + 1
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    return start_idx, end_idx

