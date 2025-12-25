"""
实体卡片组件
===========

用于展示单个实体的详细信息。
"""
from typing import Dict, List, Optional, Any
import streamlit as st
import numpy as np

# 导入配置
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENTITY_TYPE_NAMES, ENTITY_TYPE_COLORS
from utils.name_mapping import get_display_name


def render_entity_card(
    token: str,
    entity_info: Optional[Dict[str, Any]] = None,
    embedding: Optional[np.ndarray] = None,
    show_vector: bool = False,
):
    """
    渲染实体信息卡片
    
    Args:
        token: Token 字符串
        entity_info: 实体额外信息
        embedding: 嵌入向量
        show_vector: 是否显示向量可视化
    """
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
    
    # 获取真实名称
    display_name = get_display_name(token)
    
    # 卡片容器
    with st.container():
        # 标题行
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {color}22, {color}11);
                border-left: 4px solid {color};
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 10px;
            ">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="
                        background: {color};
                        color: white;
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 0.8em;
                    ">{type_name}</span>
                    <span style="font-size: 1.2em; font-weight: bold;">{display_name}</span>
                </div>
                <div style="margin-top: 8px; color: #888; font-size: 0.9em;">
                    IMDB ID: <code>{entity_id}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # 额外信息
        if entity_info:
            cols = st.columns(len(entity_info))
            for col, (key, value) in zip(cols, entity_info.items()):
                with col:
                    st.metric(key, value)
        
        # 向量可视化
        if show_vector and embedding is not None:
            from utils.visualization import create_vector_heatmap
            
            st.markdown("##### 嵌入向量")
            fig = create_vector_heatmap(embedding, title="", height=100)
            st.plotly_chart(fig, use_container_width=True)
            
            # 向量统计
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("维度", len(embedding))
            with col2:
                st.metric("均值", f"{np.mean(embedding):.4f}")
            with col3:
                st.metric("标准差", f"{np.std(embedding):.4f}")
            with col4:
                st.metric("范数", f"{np.linalg.norm(embedding):.4f}")


def render_entity_list(
    tokens: List[str],
    title: str = "实体列表",
    max_display: int = 10,
    on_click: Optional[callable] = None,
):
    """
    渲染实体列表
    
    Args:
        tokens: Token 列表
        title: 列表标题
        max_display: 最大显示数量
        on_click: 点击回调函数
    """
    st.markdown(f"#### {title}")
    
    for i, token in enumerate(tokens[:max_display]):
        # 解析 Token
        if "_" in token:
            entity_type = token.split("_")[0]
        else:
            entity_type = "OTHER"
        
        type_name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
        color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
        display_name = get_display_name(token)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(
                f'<span style="color:{color}">●</span> {type_name}',
                unsafe_allow_html=True,
            )
        with col2:
            if on_click:
                if st.button(display_name, key=f"entity_{i}_{token}"):
                    on_click(token)
            else:
                st.text(display_name)
    
    if len(tokens) > max_display:
        st.caption(f"... 还有 {len(tokens) - max_display} 个实体")


def render_entity_badge(token: str) -> str:
    """
    生成实体徽章 HTML
    
    Args:
        token: Token 字符串
        
    Returns:
        HTML 字符串
    """
    if "_" in token:
        entity_type = token.split("_")[0]
    else:
        entity_type = "OTHER"
    
    color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
    display_name = get_display_name(token)
    
    return f"""
    <span style="
        background: {color}22;
        border: 1px solid {color};
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85em;
    ">{display_name}</span>
    """


def render_mini_card(
    token: str,
    similarity: Optional[float] = None,
    rank: Optional[int] = None,
):
    """
    渲染迷你实体卡片（用于列表展示）
    
    Args:
        token: Token
        similarity: 相似度
        rank: 排名
    """
    if "_" in token:
        entity_type = token.split("_")[0]
    else:
        entity_type = "OTHER"
    
    type_name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
    color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
    display_name = get_display_name(token)
    
    rank_str = f"#{rank}" if rank else ""
    sim_str = f"{similarity:.4f}" if similarity else ""
    
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            margin-bottom: 5px;
        ">
            <div>
                <span style="color:{color};margin-right:8px;">●</span>
                <span style="font-weight:500;">{display_name}</span>
                <span style="color:#888;font-size:0.8em;margin-left:8px;">{type_name}</span>
            </div>
            <div>
                <span style="color:#888;margin-right:10px;">{rank_str}</span>
                <span style="color:#4ecdc4;font-weight:bold;">{sim_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

