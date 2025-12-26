"""
通用搜索输入组件
================

提供统一的实体搜索输入框，支持:
- 名称搜索（用户不接触 Token）
- 模糊匹配（支持拼写容错）
- 实时搜索建议
- 多选输入（用于向量算术）
- 空输入时显示热门推荐
- 无匹配时显示相近建议

使用方法:
    from components.search_input import render_search_input, render_multi_search_input
    
    # 单选搜索
    selected = render_search_input(
        label="搜索电影或演员",
        entity_types=["MOV", "ACT"],
        key="my_search",
    )
    
    # 多选搜索（用于向量算术）
    tokens = render_multi_search_input(
        label="输入实体（逗号分隔）",
        key="multi_search",
    )
"""
import streamlit as st
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ENTITY_TYPE_NAMES, ENTITY_TYPE_COLORS
from utils.name_mapping import (
    fuzzy_search,
    get_display_name,
    get_entity_display_info,
    get_hot_entities,
    get_token_by_name,
    load_name_mapping,
)


# =============================================================================
# 单选搜索组件
# =============================================================================

def render_search_input(
    label: str = "搜索实体",
    placeholder: str = "输入电影名、演员名...",
    entity_types: Optional[List[str]] = None,
    key: str = "search_input",
    help_text: Optional[str] = None,
    show_suggestions: bool = True,
    suggestion_limit: int = 5,
    min_chars: int = 2,
) -> Optional[str]:
    """
    渲染单选搜索输入框
    
    Args:
        label: 输入框标签
        placeholder: 占位文本
        entity_types: 限制搜索的实体类型，如 ["MOV", "ACT"]
        key: Streamlit 组件 key
        help_text: 帮助文本
        show_suggestions: 是否显示搜索建议
        suggestion_limit: 建议数量上限
        min_chars: 触发搜索的最少字符数
        
    Returns:
        选中的 Token，未选中返回 None
    """
    # 初始化 session state
    selected_key = f"{key}_selected"
    if selected_key not in st.session_state:
        st.session_state[selected_key] = None
    
    # 输入框
    query = st.text_input(
        label,
        placeholder=placeholder,
        key=key,
        help=help_text,
    )
    
    selected_token = st.session_state[selected_key]
    
    # 显示搜索建议
    if show_suggestions and query and len(query) >= min_chars:
        results = fuzzy_search(
            query,
            limit=suggestion_limit,
            entity_types=entity_types,
        )
        
        if results:
            st.markdown("**搜索建议:**")
            cols = st.columns(min(len(results), suggestion_limit))
            
            for i, result in enumerate(results):
                with cols[i]:
                    # 显示名称和类型
                    entity_type = result["type"]
                    color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
                    type_name = result["type_name"]
                    
                    # 按钮显示名称
                    button_label = result["name"]
                    if len(button_label) > 15:
                        button_label = button_label[:15] + "..."
                    
                    if st.button(
                        button_label,
                        key=f"{key}_suggest_{i}",
                        help=f"[{type_name}] {result['name']}",
                        use_container_width=True,
                    ):
                        st.session_state[selected_key] = result["token"]
                        st.rerun()
        
        elif query:
            # 无匹配，尝试获取相近建议
            similar = fuzzy_search(query, limit=3, threshold=40, entity_types=entity_types)
            if similar:
                st.info(f"未找到精确匹配。您是否要找: {', '.join([s['name'] for s in similar])}？")
            else:
                st.warning("未找到匹配的实体")
    
    # 显示选中的实体
    if selected_token:
        info = get_entity_display_info(selected_token)
        color = ENTITY_TYPE_COLORS.get(info["type"], "#888")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(
                f'<span style="color:{color}">●</span> 已选中: **{info["name"]}** [{info["type_name"]}]',
                unsafe_allow_html=True,
            )
        with col2:
            if st.button("✕ 清除", key=f"{key}_clear"):
                st.session_state[selected_key] = None
                st.rerun()
    
    return selected_token


def render_search_with_results(
    label: str = "搜索实体",
    placeholder: str = "输入电影名、演员名...",
    entity_types: Optional[List[str]] = None,
    key: str = "search_results",
    result_limit: int = 10,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    渲染搜索框并返回搜索结果列表
    
    Args:
        label: 输入框标签
        placeholder: 占位文本
        entity_types: 限制搜索的实体类型
        key: 组件 key
        result_limit: 结果数量上限
        
    Returns:
        (query, results) 元组
    """
    query = st.text_input(
        label,
        placeholder=placeholder,
        key=key,
    )
    
    results = []
    if query and len(query) >= 2:
        results = fuzzy_search(query, limit=result_limit, entity_types=entity_types)
    
    return query, results


# =============================================================================
# 多选搜索组件（用于向量算术）
# =============================================================================

def render_multi_search_input(
    label: str = "输入实体（用逗号分隔）",
    placeholder: str = "例如: 肖申克的救赎, 摩根·弗里曼",
    entity_types: Optional[List[str]] = None,
    key: str = "multi_search",
    help_text: Optional[str] = None,
) -> List[str]:
    """
    渲染多选搜索输入框（用于向量算术等场景）
    
    用户输入多个名称（逗号分隔），返回对应的 Token 列表。
    
    Args:
        label: 输入框标签
        placeholder: 占位文本
        entity_types: 限制搜索的实体类型
        key: 组件 key
        help_text: 帮助文本
        
    Returns:
        匹配的 Token 列表
    """
    input_text = st.text_input(
        label,
        placeholder=placeholder,
        key=key,
        help=help_text,
    )
    
    if not input_text:
        return []
    
    # 解析输入（逗号分隔）
    names = [n.strip() for n in input_text.split(",") if n.strip()]
    
    # 查找每个名称对应的 Token
    tokens = []
    not_found = []
    found_info = []
    
    for name in names:
        # 首先尝试精确匹配
        results = fuzzy_search(name, limit=1, threshold=80, entity_types=entity_types)
        
        if results:
            token = results[0]["token"]
            tokens.append(token)
            found_info.append(results[0])
        else:
            not_found.append(name)
    
    # 显示结果
    if found_info:
        st.markdown("**已识别:**")
        for info in found_info:
            color = ENTITY_TYPE_COLORS.get(info["type"], "#888")
            st.markdown(
                f'<span style="color:{color}">●</span> {info["name"]} [{info["type_name"]}]',
                unsafe_allow_html=True,
            )
    
    if not_found:
        st.warning(f"未找到: {', '.join(not_found)}")
        
        # 尝试为未找到的提供建议
        for name in not_found:
            similar = fuzzy_search(name, limit=3, threshold=40, entity_types=entity_types)
            if similar:
                suggestions = ", ".join([s["name"] for s in similar])
                st.caption(f"  → \"{name}\" 您是否要找: {suggestions}？")
    
    return tokens


def render_token_input_with_preview(
    label: str = "输入实体",
    placeholder: str = "输入名称...",
    key: str = "token_preview",
    entity_types: Optional[List[str]] = None,
) -> Optional[str]:
    """
    渲染带预览的搜索输入（显示匹配结果列表）
    
    Args:
        label: 标签
        placeholder: 占位文本
        key: 组件 key
        entity_types: 限制类型
        
    Returns:
        选中的 Token
    """
    query, results = render_search_with_results(
        label=label,
        placeholder=placeholder,
        entity_types=entity_types,
        key=key,
        result_limit=10,
    )
    
    selected_token = None
    
    if results:
        st.markdown(f"**找到 {len(results)} 个结果:**")
        
        for i, result in enumerate(results):
            color = ENTITY_TYPE_COLORS.get(result["type"], "#888")
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f'<span style="color:{color}">●</span> {result["name"]} '
                    f'<small style="color:#888">[{result["type_name"]}]</small>',
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("选择", key=f"{key}_select_{i}", use_container_width=True):
                    selected_token = result["token"]
    
    return selected_token


# =============================================================================
# 热门推荐组件
# =============================================================================

def render_hot_suggestions(
    title: str = "热门推荐",
    limit: int = 6,
    entity_type: Optional[str] = None,
    key: str = "hot_suggestions",
    on_select_key: Optional[str] = None,
) -> Optional[str]:
    """
    渲染热门推荐（用于空搜索时）
    
    Args:
        title: 标题
        limit: 推荐数量
        entity_type: 限制类型
        key: 组件 key
        on_select_key: 选中时写入的 session_state key
        
    Returns:
        选中的 Token
    """
    hot = get_hot_entities(limit=limit, entity_type=entity_type)
    
    if not hot:
        return None
    
    st.markdown(f"**{title}:**")
    
    cols = st.columns(min(len(hot), 3))
    selected = None
    
    for i, item in enumerate(hot):
        with cols[i % 3]:
            color = ENTITY_TYPE_COLORS.get(item["type"], "#888")
            
            # 显示名称
            button_label = item["name"]
            if len(button_label) > 12:
                button_label = button_label[:12] + "..."
            
            if st.button(
                button_label,
                key=f"{key}_{i}",
                help=f"[{item['type_name']}] {item['name']}",
                use_container_width=True,
            ):
                selected = item["token"]
                if on_select_key:
                    st.session_state[on_select_key] = selected
    
    return selected


# =============================================================================
# 实体选择器（下拉框版本）
# =============================================================================

def render_entity_selectbox(
    label: str = "选择实体",
    entity_types: Optional[List[str]] = None,
    key: str = "entity_select",
    default_token: Optional[str] = None,
) -> Optional[str]:
    """
    渲染实体选择下拉框
    
    注意：仅适用于数量较少的实体类型（如类型、年代等）
    
    Args:
        label: 标签
        entity_types: 限制类型（建议只用于 GEN, ERA, TYP 等）
        key: 组件 key
        default_token: 默认选中的 token
        
    Returns:
        选中的 Token
    """
    mapping = load_name_mapping()
    
    # 过滤实体
    options = []
    for token, name in mapping.items():
        token_type = token.split("_")[0] if "_" in token else "OTHER"
        if entity_types and token_type not in entity_types:
            continue
        options.append((name, token))
    
    # 按名称排序
    options.sort(key=lambda x: x[0])
    
    if not options:
        st.warning("没有可选的实体")
        return None
    
    # 查找默认索引
    default_index = 0
    if default_token:
        for i, (_, token) in enumerate(options):
            if token == default_token:
                default_index = i
                break
    
    # 渲染下拉框
    selected_name = st.selectbox(
        label,
        options=[name for name, _ in options],
        index=default_index,
        key=key,
    )
    
    # 查找对应的 token
    for name, token in options:
        if name == selected_name:
            return token
    
    return None

