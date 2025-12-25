"""
数据浏览页面 (Data Browse)
=========================

可筛选、分页浏览所有实体数据。

功能:
- 按实体类型筛选
- 分页浏览
- 搜索功能
- 数据导出

使用的数据文件:
- metadata.tsv: Token 列表
- recsys/entity_index.json: 实体分类索引
"""
import streamlit as st
from pathlib import Path
import sys
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig, ENTITY_TYPE_NAMES, ENTITY_TYPE_COLORS
from utils.data_loader import (
    load_metadata,
    load_entity_index,
    load_config,
    get_entity_type,
)
from utils.visualization import create_bar_chart, create_pie_chart
from components.sidebar import render_page_header
from components.filters import render_type_filter, render_pagination


# =============================================================================
# 页面配置
# =============================================================================

st.set_page_config(
    page_title="Browse - " + AppConfig.APP_TITLE,
    page_icon="📋",
    layout=AppConfig.LAYOUT,
)


# =============================================================================
# 页面标题
# =============================================================================

render_page_header(
    title="数据浏览",
    description="浏览和筛选所有实体数据，支持按类型过滤和分页浏览。",
    icon="📋",
)


# =============================================================================
# 加载数据
# =============================================================================

@st.cache_data
def load_data():
    """加载数据"""
    metadata_df = load_metadata()
    entity_index = load_entity_index()
    config = load_config()
    return metadata_df, entity_index, config


metadata_df, entity_index, config = load_data()

if metadata_df.empty:
    st.error("无法加载元数据，请检查 metadata.tsv 文件")
    st.stop()


# =============================================================================
# 侧边栏
# =============================================================================

with st.sidebar:
    st.markdown("### 📋 数据浏览设置")
    st.markdown("---")
    
    # 数据统计
    st.markdown("#### 📊 数据统计")
    st.metric("总实体数", f"{len(metadata_df):,}")
    
    # 类型分布
    st.markdown("#### 📌 类型分布")
    type_counts = metadata_df["entity_type"].value_counts()
    
    for entity_type, count in type_counts.items():
        name = ENTITY_TYPE_NAMES.get(entity_type, entity_type)
        color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
        pct = count / len(metadata_df) * 100
        st.markdown(
            f'<span style="color:{color}">●</span> {name}: {count:,} ({pct:.1f}%)',
            unsafe_allow_html=True,
        )
    
    st.markdown("---")
    
    # 每页显示数量
    items_per_page = st.selectbox(
        "每页显示",
        options=[20, 50, 100, 200],
        index=1,
        key="items_per_page",
    )


# =============================================================================
# 筛选控件
# =============================================================================

st.markdown("## 🔍 筛选条件")

col1, col2 = st.columns([2, 1])

with col1:
    # 搜索框
    search_query = st.text_input(
        "搜索 Token",
        placeholder="输入关键词搜索...",
        key="browse_search",
    )

with col2:
    # 类型筛选
    available_types = metadata_df["entity_type"].unique().tolist()
    selected_types = st.multiselect(
        "实体类型",
        options=available_types,
        default=available_types,
        format_func=lambda x: f"{ENTITY_TYPE_NAMES.get(x, x)} ({x})",
        key="browse_type_filter",
    )


# =============================================================================
# 应用筛选
# =============================================================================

# 应用类型筛选
if selected_types:
    filtered_df = metadata_df[metadata_df["entity_type"].isin(selected_types)]
else:
    filtered_df = metadata_df

# 应用搜索筛选
if search_query:
    filtered_df = filtered_df[
        filtered_df["token"].str.contains(search_query, case=False, na=False)
    ]

st.markdown(f"**筛选结果: {len(filtered_df):,} 条** (共 {len(metadata_df):,} 条)")


# =============================================================================
# 数据表格
# =============================================================================

st.markdown("---")
st.markdown("## 📋 数据列表")

# 分页
total_items = len(filtered_df)
total_pages = (total_items + items_per_page - 1) // items_per_page

if total_pages > 1:
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("◀ 上一页", key="prev_page"):
            current_page = st.session_state.get("browse_page", 1)
            if current_page > 1:
                st.session_state["browse_page"] = current_page - 1
    
    with col2:
        current_page = st.session_state.get("browse_page", 1)
        if current_page > total_pages:
            current_page = total_pages
            st.session_state["browse_page"] = current_page
        
        st.markdown(
            f"<div style='text-align:center'>第 {current_page} / {total_pages} 页</div>",
            unsafe_allow_html=True,
        )
    
    with col3:
        if st.button("下一页 ▶", key="next_page"):
            current_page = st.session_state.get("browse_page", 1)
            if current_page < total_pages:
                st.session_state["browse_page"] = current_page + 1
else:
    current_page = 1

# 获取当前页数据
start_idx = (current_page - 1) * items_per_page
end_idx = min(start_idx + items_per_page, total_items)

page_df = filtered_df.iloc[start_idx:end_idx].copy()

# 显示表格
if not page_df.empty:
    # 添加序号列
    page_df = page_df.reset_index(drop=True)
    page_df.index = range(start_idx + 1, end_idx + 1)
    page_df.index.name = "序号"
    
    # 格式化显示
    display_df = page_df[["token", "entity_type", "type_name"]].copy()
    display_df.columns = ["Token", "类型代码", "类型名称"]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Token": st.column_config.TextColumn("Token", width="large"),
            "类型代码": st.column_config.TextColumn("类型代码", width="small"),
            "类型名称": st.column_config.TextColumn("类型名称", width="medium"),
        },
    )
else:
    st.info("没有匹配的数据")


# =============================================================================
# 数据统计图表
# =============================================================================

st.markdown("---")
st.markdown("## 📊 类型分布统计")

col1, col2 = st.columns(2)

with col1:
    # 柱状图
    type_counts_dict = type_counts.to_dict()
    fig_bar = create_bar_chart(
        type_counts_dict,
        title="各类型实体数量",
        x_label="实体类型",
        y_label="数量",
        height=400,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    # 饼图
    fig_pie = create_pie_chart(
        type_counts_dict,
        title="类型占比",
        height=400,
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# =============================================================================
# 数据导出
# =============================================================================

st.markdown("---")
st.markdown("## 💾 数据导出")

col1, col2 = st.columns(2)

with col1:
    # 导出筛选后的数据
    if st.button("📥 导出筛选结果 (CSV)", use_container_width=True):
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="下载 CSV",
            data=csv_data,
            file_name="filtered_entities.csv",
            mime="text/csv",
            key="download_filtered",
        )

with col2:
    # 导出全部数据
    if st.button("📥 导出全部数据 (CSV)", use_container_width=True):
        csv_data = metadata_df.to_csv(index=False)
        st.download_button(
            label="下载 CSV",
            data=csv_data,
            file_name="all_entities.csv",
            mime="text/csv",
            key="download_all",
        )

