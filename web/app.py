"""
IMDB Word2Vec 可视化看板 - 首页
==============================

应用主入口，展示系统概览、数据统计和快速导航。

运行方式:
    cd web
    streamlit run app.py

使用的数据文件:
    - recsys/config.json: 系统配置和统计信息
    - embedding_tsne.png: 静态 t-SNE 可视化图
"""
import streamlit as st
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import AppConfig, DataFiles, ENTITY_TYPE_NAMES, ENTITY_TYPE_COLORS
from utils.data_loader import load_config, get_data_files_info
from utils.visualization import create_bar_chart, create_pie_chart


# =============================================================================
# 页面配置
# =============================================================================

st.set_page_config(
    page_title=AppConfig.APP_TITLE,
    page_icon=AppConfig.PAGE_ICON,
    layout=AppConfig.LAYOUT,
    initial_sidebar_state=AppConfig.INITIAL_SIDEBAR_STATE,
)

# 自定义 CSS
st.markdown("""
<style>
    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 自定义样式 */
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stat-value {
        font-size: 2em;
        font-weight: bold;
        color: #00d4ff;
    }
    
    .stat-label {
        color: #888;
        font-size: 0.9em;
    }
    
    .feature-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #00d4ff;
    }
    
    .nav-button {
        display: block;
        width: 100%;
        padding: 15px;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        text-align: left;
        margin: 10px 0;
        transition: all 0.3s;
    }
    
    .nav-button:hover {
        border-color: #00d4ff;
        transform: translateX(5px);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 侧边栏
# =============================================================================

with st.sidebar:
    st.markdown("### 🎬 IMDB Word2Vec")
    st.markdown("基于 Word2Vec 的电影知识图谱嵌入可视化系统")
    st.markdown("---")
    
    # 数据文件状态
    st.markdown("#### 📂 数据文件状态")
    files_info = get_data_files_info()
    
    for file_info in files_info:
        if file_info["exists"]:
            st.markdown(
                f'✅ `{file_info["name"]}` ({file_info["size_mb"]} MB)',
            )
        else:
            st.markdown(
                f'❌ `{file_info["name"]}` (缺失)',
            )
    
    st.markdown("---")
    st.markdown("#### 📌 实体类型图例")
    for entity_type, name in ENTITY_TYPE_NAMES.items():
        color = ENTITY_TYPE_COLORS.get(entity_type, "#888")
        st.markdown(
            f'<span style="color:{color}">●</span> {name} ({entity_type})',
            unsafe_allow_html=True,
        )


# =============================================================================
# 主页面内容
# =============================================================================

# 标题
st.markdown('<div class="main-header">🎬 IMDB Word2Vec 可视化看板</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #888; margin-bottom: 30px;">
    基于 Word2Vec 的电影知识图谱嵌入向量可视化与分析系统<br>
    支持 PCA / UMAP / t-SNE 多种降维方法 | ONNX 在线推理 | 交互式探索
</div>
""", unsafe_allow_html=True)

# 加载配置
config = load_config()

# =============================================================================
# 数据统计卡片
# =============================================================================

st.markdown("## 📊 数据概览")

if config:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">{:,}</div>
            <div class="stat-label">词汇表大小</div>
        </div>
        """.format(config.get("vocab_size", 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">{}</div>
            <div class="stat-label">嵌入维度</div>
        </div>
        """.format(config.get("embedding_dim", 0)), unsafe_allow_html=True)
    
    with col3:
        entity_types = config.get("entity_types", {})
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">{}</div>
            <div class="stat-label">实体类型</div>
        </div>
        """.format(len(entity_types)), unsafe_allow_html=True)
    
    with col4:
        total_entities = sum(entity_types.values())
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">{:,}</div>
            <div class="stat-label">总实体数</div>
        </div>
        """.format(total_entities), unsafe_allow_html=True)

else:
    st.warning("无法加载配置文件，请检查数据路径")

st.markdown("---")

# =============================================================================
# 实体类型分布
# =============================================================================

st.markdown("## 📈 实体类型分布")

if config and config.get("entity_types"):
    entity_types = config["entity_types"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 柱状图
        fig_bar = create_bar_chart(
            entity_types,
            title="各类型实体数量",
            x_label="实体类型",
            y_label="数量",
            height=400,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # 饼图
        fig_pie = create_pie_chart(
            entity_types,
            title="实体类型占比",
            height=400,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# =============================================================================
# t-SNE 可视化预览
# =============================================================================

st.markdown("## 🎨 嵌入空间预览")

tsne_image_path = DataFiles.EMBEDDING_TSNE_PNG

if tsne_image_path.exists():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(
            str(tsne_image_path),
            caption="t-SNE 降维可视化 (静态预览)",
            use_container_width=True,
        )
    
    with col2:
        st.markdown("""
        ### 关于此图
        
        这是使用 **t-SNE** 算法将 128 维嵌入向量降至 2 维后的可视化结果。
        
        **t-SNE 特点:**
        - 保留局部结构
        - 相似实体聚集在一起
        - 不同类型形成不同簇
        
        **交互式探索:**
        
        前往 **🎯 聚类分析** 页面体验交互式可视化，支持:
        - 缩放、平移
        - 点击查看详情
        - 类型筛选
        """)
else:
    st.info("t-SNE 可视化图片未找到")

st.markdown("---")

# =============================================================================
# 功能导航
# =============================================================================

st.markdown("## 🧭 功能导航")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 聚类分析</h3>
        <p>交互式 t-SNE 散点图，点击数据点查看详情和推荐。</p>
        <p><strong>使用数据:</strong> clustering.json</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>🔗 推荐关系</h3>
        <p>输入实体，使用 ONNX 推理获取相似实体和关系网络图。</p>
        <p><strong>使用数据:</strong> word2vec.onnx</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 数据详情</h3>
        <p>查看实体的完整信息，包括 128 维嵌入向量可视化。</p>
        <p><strong>使用数据:</strong> embeddings.json</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>🔬 嵌入探索</h3>
        <p>向量算术运算 (A - B + C)，探索嵌入空间的语义关系。</p>
        <p><strong>使用数据:</strong> embeddings.npy</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>📈 降维对比</h3>
        <p>PCA / UMAP / t-SNE 三种降维方法并排对比。</p>
        <p><strong>使用数据:</strong> embeddings.npy</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>📋 数据浏览</h3>
        <p>可筛选、分页浏览所有实体，支持按类型过滤。</p>
        <p><strong>使用数据:</strong> metadata.tsv, entity_index.json</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# 技术信息
# =============================================================================

st.markdown("## ⚙️ 技术信息")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 技术栈
    - **Web 框架:** Streamlit
    - **可视化:** Plotly
    - **降维算法:** PCA, UMAP, t-SNE (scikit-learn, umap-learn)
    - **在线推理:** ONNX Runtime
    - **数据处理:** Pandas, NumPy
    """)

with col2:
    st.markdown("""
    ### 数据来源
    - **数据集:** IMDB 电影数据库
    - **嵌入模型:** Word2Vec (Skip-gram)
    - **实体类型:** 电影、演员、导演、类型、评分、年代等
    - **总词汇量:** {:,}
    """.format(config.get("vocab_size", 0) if config else 0))

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    IMDB Word2Vec 可视化看板 | 
    使用 ❤️ 和 Python 构建 | 
    <a href="docs/README.md" style="color: #00d4ff;">技术文档</a>
</div>
""", unsafe_allow_html=True)

