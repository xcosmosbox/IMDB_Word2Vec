# IMDB Word2Vec 可视化看板 - 技术文档

## 目录

1. [功能清单](#功能清单)
2. [技术栈说明](#技术栈说明)
3. [文件结构说明](#文件结构说明)
4. [部署指南](#部署指南)
5. [扩展说明](#扩展说明)
6. [资源利用说明](#资源利用说明)

---

## 功能清单

### 1. 首页概览 (app.py)

| 功能 | 说明 | 操作方式 |
|------|------|----------|
| 数据统计卡片 | 显示词汇表大小、嵌入维度、实体类型数量 | 自动加载显示 |
| 实体类型分布图 | 柱状图和饼图展示各类型实体数量 | 自动加载显示 |
| t-SNE 预览图 | 显示静态的 t-SNE 可视化图片 | 自动加载显示 |
| 功能导航 | 各页面功能介绍和快速入口 | 点击侧边栏导航 |

### 2. 聚类分析页 (1_🎯_聚类分析.py)

| 功能 | 说明 | 操作方式 |
|------|------|----------|
| t-SNE 散点图 | 交互式聚类可视化 | 缩放、平移、悬停查看 |
| 类型筛选 | 按实体类型过滤显示 | 侧边栏多选框 |
| 点击查看详情 | 点击数据点显示详细信息 | 点击散点图中的点 |
| 相似推荐 | 显示选中实体的相似实体列表 | 点击数据点后自动显示 |
| 聚类中心统计 | 展示各聚类的统计信息 | 页面底部表格 |

### 3. 推荐关系页 (2_🔗_推荐关系.py)

| 功能 | 说明 | 操作方式 |
|------|------|----------|
| Token 搜索 | 模糊搜索实体 | 输入关键词 |
| ONNX 推理 | 获取实体嵌入向量 | 选择实体后自动执行 |
| 相似度排名 | 显示 Top-K 相似实体及分数 | 自动计算显示 |
| 关系网络图 | 放射状网络展示相似关系 | 自动生成 |
| 类型过滤 | 只显示特定类型的相似结果 | 侧边栏选择器 |

### 4. 数据详情页 (3_📊_数据详情.py)

| 功能 | 说明 | 操作方式 |
|------|------|----------|
| 实体搜索 | 搜索并选择实体 | 输入关键词，选择结果 |
| 基本信息 | 显示 Token、类型、ID 等 | 选择实体后显示 |
| 向量热力图 | 128 维向量可视化 | 自动生成 |
| 向量统计 | 均值、标准差、范数等 | 自动计算 |
| 值分布直方图 | 向量值的分布情况 | 自动生成 |
| 原始数据下载 | 下载单个实体的向量 | 点击下载按钮 |

### 5. 嵌入探索页 (4_🔬_嵌入探索.py)

| 功能 | 说明 | 操作方式 |
|------|------|----------|
| 向量算术 | A - B + C 运算 | 输入正向和负向 Token |
| 最近邻搜索 | 查找运算结果的相似实体 | 自动计算 |
| 预设示例 | 快速尝试预设的运算 | 点击示例按钮 |
| Token 搜索 | 查找可用的 Token | 输入关键词 |

### 6. 降维对比页 (5_📈_降维对比.py)

| 功能 | 说明 | 操作方式 |
|------|------|----------|
| 采样设置 | 设置用于计算的样本数量 | 侧边栏滑块 |
| 参数调节 | t-SNE 和 UMAP 参数 | 侧边栏滑块 |
| 并排对比 | PCA/UMAP/t-SNE 结果并排显示 | 点击计算按钮 |
| 方法特性表 | 各方法的特点对比 | 页面底部表格 |

### 7. 数据浏览页 (6_📋_数据浏览.py)

| 功能 | 说明 | 操作方式 |
|------|------|----------|
| 类型筛选 | 按实体类型过滤 | 多选框 |
| 关键词搜索 | 搜索 Token | 输入关键词 |
| 分页浏览 | 分页显示数据 | 翻页按钮 |
| 数据导出 | 导出 CSV 格式 | 下载按钮 |
| 统计图表 | 类型分布柱状图和饼图 | 自动显示 |

### 8. 导出工具页 (7_💾_导出工具.py)

| 功能 | 说明 | 操作方式 |
|------|------|----------|
| 文件下载 | 下载各种格式的数据文件 | 点击下载按钮 |
| TF Projector 指南 | 使用 TensorFlow Projector 的教程 | 阅读说明 |
| ONNX 使用指南 | Python/JavaScript 使用示例 | 阅读代码示例 |
| 文件格式说明 | 各文件的格式和用途 | 展开查看 |

---

## 技术栈说明

### 核心框架

| 技术 | 版本要求 | 选择理由 |
|------|----------|----------|
| **Streamlit** | ≥1.28.0 | Python 全栈 Web 框架，开发效率高，组件丰富 |
| **Plotly** | ≥5.18.0 | 交互式可视化库，支持缩放、平移、悬停等交互 |
| **Pandas** | ≥2.0.0 | 数据处理和分析的标准库 |
| **NumPy** | ≥1.24.0 | 数值计算基础库 |

### 机器学习

| 技术 | 版本要求 | 选择理由 |
|------|----------|----------|
| **scikit-learn** | ≥1.3.0 | PCA、t-SNE 降维算法实现 |
| **umap-learn** | ≥0.5.4 | UMAP 降维算法，比 t-SNE 更快 |
| **ONNX Runtime** | ≥1.16.0 | ONNX 模型推理，支持跨平台部署 |

### 为什么选择 Streamlit？

1. **Python 全栈**: 无需前端知识，纯 Python 开发
2. **开发效率高**: 内置丰富组件，快速构建界面
3. **数据友好**: 原生支持 Pandas DataFrame、Plotly 图表
4. **热重载**: 代码修改后自动刷新页面
5. **缓存机制**: `@st.cache_data` 优化数据加载性能

---

## 文件结构说明

```
web/
├── app.py                      # 主入口，首页概览
├── requirements.txt            # Python 依赖
├── config.py                   # 全局配置（路径、颜色、参数）
│
├── pages/                      # Streamlit 多页面
│   ├── 1_🎯_聚类分析.py        # 聚类可视化
│   ├── 2_🔗_推荐关系.py        # 相似推荐
│   ├── 3_📊_数据详情.py        # 实体详情
│   ├── 4_🔬_嵌入探索.py        # 向量算术
│   ├── 5_📈_降维对比.py        # 降维对比
│   ├── 6_📋_数据浏览.py        # 数据浏览
│   └── 7_💾_导出工具.py        # 导出工具
│
├── utils/                      # 工具模块
│   ├── __init__.py             # 模块导出
│   ├── data_loader.py          # 数据加载（所有文件解析 + 缓存）
│   ├── dimensionality.py       # 降维算法（PCA/UMAP/t-SNE）
│   ├── similarity.py           # 相似度计算（余弦相似度、Top-K）
│   ├── onnx_inference.py       # ONNX 推理封装
│   └── visualization.py        # Plotly 图表生成
│
├── components/                 # UI 组件
│   ├── __init__.py             # 模块导出
│   ├── sidebar.py              # 侧边栏组件
│   ├── entity_card.py          # 实体信息卡片
│   ├── similarity_list.py      # 相似度列表
│   └── filters.py              # 筛选器组件
│
├── cache/                      # 降维结果缓存目录
│
├── .streamlit/
│   └── config.toml             # Streamlit 主题配置
│
└── docs/
    └── README.md               # 本文档
```

### 模块职责

| 模块 | 职责 |
|------|------|
| `config.py` | 集中管理配置：数据路径、颜色方案、算法参数 |
| `utils/data_loader.py` | 解析所有 12 个数据文件，使用 `@st.cache_data` 缓存 |
| `utils/dimensionality.py` | 封装 PCA/UMAP/t-SNE，支持结果缓存到文件 |
| `utils/similarity.py` | 余弦相似度、Top-K 搜索、向量算术 |
| `utils/onnx_inference.py` | ONNX Runtime 推理封装，单例模式 |
| `utils/visualization.py` | Plotly 图表生成函数 |
| `components/*` | 可复用的 UI 组件 |

---

## 部署指南

### 本地运行

```bash
# 1. 进入 web 目录
cd web

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行应用
streamlit run app.py
```

应用将在 `http://localhost:8501` 启动。

### 服务器部署

#### 方式一：直接运行

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

#### 方式二：使用 systemd（Linux）

创建 `/etc/systemd/system/streamlit-imdb.service`:

```ini
[Unit]
Description=IMDB Word2Vec Dashboard
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/web
ExecStart=/path/to/venv/bin/streamlit run app.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl enable streamlit-imdb
sudo systemctl start streamlit-imdb
```

#### 方式三：Docker 部署

创建 `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

构建并运行：

```bash
docker build -t imdb-dashboard .
docker run -p 8501:8501 -v /path/to/artifacts:/app/../imdb_word2vec/artifacts imdb-dashboard
```

### 注意事项

1. **数据路径**: 确保 `../imdb_word2vec/artifacts/` 路径下有所有数据文件
2. **内存需求**: `embeddings.json` (130MB) 加载需要足够内存
3. **首次加载**: 降维计算可能需要几分钟，结果会缓存到 `cache/` 目录

---

## 扩展说明

### 添加新页面

1. 在 `pages/` 目录下创建新文件，如 `8_🆕_新页面.py`
2. 文件名格式：`序号_图标_名称.py`
3. 使用相同的页面模板：

```python
import streamlit as st
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig
from components.sidebar import render_page_header

# 页面配置
st.set_page_config(
    page_title="新页面 - " + AppConfig.APP_TITLE,
    page_icon="🆕",
    layout=AppConfig.LAYOUT,
)

# 页面标题
render_page_header(
    title="新页面",
    description="页面描述",
    icon="🆕",
)

# 页面内容...
```

### 添加新的可视化图表

1. 在 `utils/visualization.py` 中添加新函数：

```python
def create_new_chart(data, **kwargs):
    """
    创建新图表
    
    Args:
        data: 数据
        **kwargs: 其他参数
        
    Returns:
        Plotly Figure 对象
    """
    import plotly.express as px
    
    fig = px.xxx(data, ...)
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )
    
    return fig
```

2. 在 `utils/__init__.py` 中导出

### 添加新的数据文件支持

1. 在 `config.py` 的 `DataFiles` 类中添加路径
2. 在 `utils/data_loader.py` 中添加加载函数：

```python
@st.cache_data(ttl=3600, show_spinner="加载新数据...")
def load_new_data():
    """加载新数据文件"""
    path = DataFiles.NEW_FILE
    
    if not path.exists():
        st.error(f"文件不存在: {path}")
        return None
    
    # 解析文件...
    return data
```

### 修改主题样式

编辑 `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#00d4ff"      # 主色调
backgroundColor = "#0e1117"   # 背景色
secondaryBackgroundColor = "#1a1a2e"  # 次要背景
textColor = "#e0e0e0"         # 文字颜色
```

---

## 资源利用说明

### 数据文件用途映射

| 文件 | 大小 | 用途 | 使用页面 |
|------|------|------|----------|
| **embeddings.npy** | 24MB | PCA/UMAP/t-SNE 降维计算 | 降维对比、嵌入探索 |
| **embeddings.json** | 130MB | 完整 Token + 向量数据 | 数据详情、嵌入探索 |
| **clustering.json** | 527KB | 预计算 t-SNE 坐标 + 聚类标签 | 聚类分析（快速加载） |
| **word2vec.onnx** | 24MB | 在线推理获取嵌入向量 | 推荐关系 |
| **metadata.tsv** | 739KB | 50,000 个 Token 列表 | 数据浏览、全局搜索 |
| **vectors.tsv** | 70MB | TF Projector 兼容格式 | 导出工具（下载） |
| **embedding_tsne.png** | 113KB | 静态 t-SNE 可视化 | 首页概览展示 |
| **recsys/config.json** | - | 系统配置元数据 | 首页概览、统计 |
| **recsys/token_to_id.json** | - | Token → ID 映射 | 推荐关系、嵌入探索 |
| **recsys/id_to_token.json** | - | ID → Token 映射 | 推理结果解析 |
| **recsys/entity_index.json** | - | 实体分类索引 | 数据浏览筛选 |
| **visualization.html** | 0B | 预留空文件 | 文档说明（预留用途） |

### visualization.html 预留用途

此文件当前为空，可用于以下用途：

1. **静态 HTML 导出**: 将 Plotly 图表导出为静态 HTML
2. **嵌入展示**: 在其他网页中嵌入可视化内容
3. **离线查看**: 生成可离线浏览的报告

### 性能优化建议

1. **大文件处理**:
   - `embeddings.json` (130MB) 首次加载较慢
   - 使用 `@st.cache_data` 缓存加载结果
   - 考虑使用 `embeddings.npy` 替代（加载更快）

2. **降维计算**:
   - 首次计算 UMAP/t-SNE 需要几分钟
   - 结果自动缓存到 `cache/` 目录
   - 可调整采样数量减少计算时间

3. **相似度搜索**:
   - 使用向量化操作 (`np.dot`) 加速
   - 避免逐个计算余弦相似度

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0.0 | 2024-12 | 初始版本，8 个功能页面 |

---

## 联系方式

如有问题或建议，请联系项目维护者。

