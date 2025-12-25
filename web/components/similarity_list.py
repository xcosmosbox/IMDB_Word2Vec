"""
相似度列表组件
=============

展示相似度搜索结果的列表和表格组件。
"""
from typing import List, Dict, Optional
import streamlit as st
import pandas as pd

# 导入配置
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENTITY_TYPE_NAMES, ENTITY_TYPE_COLORS
from utils.name_mapping import get_display_name


def render_similarity_list(
    results: List[Dict],
    title: str = "相似实体",
    show_rank: bool = True,
    on_select: Optional[callable] = None,
):
    """
    渲染相似度结果列表
    
    Args:
        results: 相似度结果列表，每项包含 token, similarity, rank
        title: 列表标题
        show_rank: 是否显示排名
        on_select: 选择回调函数
    """
    if not results:
        st.info("暂无相似结果")
        return
    
    st.markdown(f"#### {title}")
    
    for item in results:
        token = item.get("token", "")
        similarity = item.get("similarity", 0)
        rank = item.get("rank", 0)
        
        # 解析实体类型
        if "_" in token:
            entity_type = token.split("_")[0]
        else:
            entity_type = "OTHER"
        
        # 获取显示名称（真实的电影名/演员名）
        display_name = get_display_name(token)
        type_name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
        color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
        
        # 相似度进度条颜色
        bar_color = "#4ecdc4" if similarity > 0.7 else "#ffeaa7" if similarity > 0.5 else "#ff6b6b"
        
        # 渲染单项
        col1, col2, col3 = st.columns([1, 3, 2])
        
        with col1:
            if show_rank:
                st.markdown(
                    f'<div style="text-align:center;font-size:1.5em;color:#888;">#{rank}</div>',
                    unsafe_allow_html=True,
                )
        
        with col2:
            st.markdown(
                f"""
                <div style="padding: 5px 0;">
                    <span style="color:{color}">●</span>
                    <strong>{display_name}</strong>
                    <span style="color:#888;font-size:0.8em;">({type_name})</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            if on_select:
                if st.button("查看详情", key=f"sim_{token}"):
                    on_select(token)
        
        with col3:
            # 相似度进度条
            st.markdown(
                f"""
                <div style="text-align:right;">
                    <span style="font-size:1.2em;font-weight:bold;color:{bar_color};">
                        {similarity:.4f}
                    </span>
                </div>
                <div style="
                    background: rgba(255,255,255,0.1);
                    border-radius: 4px;
                    height: 6px;
                    margin-top: 5px;
                ">
                    <div style="
                        background: {bar_color};
                        width: {similarity * 100}%;
                        height: 100%;
                        border-radius: 4px;
                    "></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown("---")


def render_similarity_table(
    results: List[Dict],
    title: str = "相似度排名",
    sortable: bool = True,
):
    """
    以表格形式渲染相似度结果
    
    Args:
        results: 相似度结果列表
        title: 表格标题
        sortable: 是否可排序
    """
    if not results:
        st.info("暂无结果")
        return
    
    st.markdown(f"#### {title}")
    
    # 转换为 DataFrame
    df = pd.DataFrame(results)
    
    # 添加实体类型列
    df["entity_type"] = df["token"].apply(
        lambda x: x.split("_")[0] if "_" in x else "OTHER"
    )
    df["type_name"] = df["entity_type"].map(ENTITY_TYPE_NAMES)
    
    # 获取显示名称
    df["display_name"] = df["token"].apply(get_display_name)
    
    # 格式化相似度
    df["相似度"] = df["similarity"].apply(lambda x: f"{x:.4f}")
    
    # 重命名列
    display_df = df[["rank", "display_name", "type_name", "相似度"]].copy()
    display_df.columns = ["排名", "名称", "类型", "相似度"]
    
    # 使用 st.dataframe 显示
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "排名": st.column_config.NumberColumn(
                "排名",
                format="%d",
                width="small",
            ),
            "名称": st.column_config.TextColumn(
                "名称",
                width="medium",
            ),
            "类型": st.column_config.TextColumn(
                "类型",
                width="small",
            ),
            "相似度": st.column_config.TextColumn(
                "相似度",
                width="small",
            ),
        },
    )


def render_compact_similarity_list(
    results: List[Dict],
    max_items: int = 5,
):
    """
    渲染紧凑版相似度列表（用于侧边栏）
    
    Args:
        results: 相似度结果
        max_items: 最大显示数量
    """
    for item in results[:max_items]:
        token = item.get("token", "")
        similarity = item.get("similarity", 0)
        rank = item.get("rank", 0)
        
        if "_" in token:
            entity_type = token.split("_")[0]
        else:
            entity_type = "OTHER"
        
        color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
        display_name = get_display_name(token)
        
        # 截断过长的名称
        display_text = display_name[:25] + '...' if len(display_name) > 25 else display_name
        
        st.markdown(
            f"""
            <div style="
                display: flex;
                justify-content: space-between;
                padding: 5px 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            ">
                <span>
                    <span style="color:{color}">●</span>
                    {display_text}
                </span>
                <span style="color:#4ecdc4;">{similarity:.3f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    if len(results) > max_items:
        st.caption(f"... 还有 {len(results) - max_items} 个结果")

