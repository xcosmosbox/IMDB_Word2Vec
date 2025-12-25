"""
导出工具页面 (Export Tools)
==========================

提供数据文件下载和 TensorFlow Projector 导入指南。

功能:
- 下载各种格式的数据文件
- TensorFlow Embedding Projector 导入教程
- 文件格式说明

使用的数据文件:
- vectors.tsv: TF Projector 兼容的向量文件
- metadata.tsv: Token 元数据
- 其他所有导出文件
"""
import streamlit as st
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig, DataFiles
from utils.data_loader import get_data_files_info
from components.sidebar import render_page_header


# =============================================================================
# 页面配置
# =============================================================================

st.set_page_config(
    page_title="Export - " + AppConfig.APP_TITLE,
    page_icon="💾",
    layout=AppConfig.LAYOUT,
)


# =============================================================================
# 页面标题
# =============================================================================

render_page_header(
    title="导出工具",
    description="下载数据文件，并了解如何在其他工具中使用这些数据。",
    icon="💾",
)


# =============================================================================
# 侧边栏
# =============================================================================

with st.sidebar:
    st.markdown("### 💾 导出工具")
    st.markdown("---")
    
    st.markdown("#### 📁 文件概览")
    
    files_info = get_data_files_info()
    total_size = sum(f["size_mb"] for f in files_info if f["exists"])
    
    st.metric("总文件数", len(files_info))
    st.metric("总大小", f"{total_size:.1f} MB")
    
    st.markdown("---")
    
    st.markdown("#### 🔗 相关链接")
    st.markdown("[TensorFlow Projector](https://projector.tensorflow.org/)")
    st.markdown("[ONNX Runtime](https://onnxruntime.ai/)")


# =============================================================================
# 文件列表
# =============================================================================

st.markdown("## 📁 可下载文件")

files_info = get_data_files_info()

# 按类型分组
file_categories = {
    "嵌入数据": ["embeddings.npy", "embeddings.json"],
    "可视化数据": ["clustering.json", "embedding_tsne.png"],
    "TensorFlow Projector": ["vectors.tsv", "metadata.tsv"],
    "ONNX 模型": ["word2vec.onnx"],
    "推荐系统配置": ["config.json", "token_to_id.json", "id_to_token.json", "entity_index.json"],
    "其他": ["visualization.html"],
}

for category, file_names in file_categories.items():
    st.markdown(f"### {category}")
    
    cols = st.columns(len(file_names))
    
    for col, file_name in zip(cols, file_names):
        with col:
            # 查找文件信息
            file_info = next((f for f in files_info if f["name"] == file_name), None)
            
            if file_info and file_info["exists"]:
                st.markdown(f"**{file_name}**")
                st.caption(f"大小: {file_info['size_mb']:.2f} MB")
                
                # 读取文件内容用于下载
                file_path = Path(file_info["path"])
                
                # 根据文件类型决定读取方式
                if file_name.endswith((".json", ".tsv", ".html")):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            file_content = f.read()
                        
                        st.download_button(
                            label=f"📥 下载",
                            data=file_content,
                            file_name=file_name,
                            mime="application/octet-stream",
                            key=f"download_{file_name}",
                        )
                    except Exception as e:
                        st.warning(f"读取失败: {e}")
                
                elif file_name.endswith((".npy", ".onnx", ".png")):
                    try:
                        with open(file_path, "rb") as f:
                            file_content = f.read()
                        
                        st.download_button(
                            label=f"📥 下载",
                            data=file_content,
                            file_name=file_name,
                            mime="application/octet-stream",
                            key=f"download_{file_name}",
                        )
                    except Exception as e:
                        st.warning(f"读取失败: {e}")
            else:
                st.markdown(f"**{file_name}**")
                st.caption("文件不存在")
                st.button("📥 下载", disabled=True, key=f"download_{file_name}")
    
    st.markdown("---")


# =============================================================================
# TensorFlow Projector 指南
# =============================================================================

st.markdown("## 📖 TensorFlow Embedding Projector 使用指南")

st.markdown("""
TensorFlow Embedding Projector 是一个强大的在线可视化工具，可以交互式地探索高维嵌入向量。

### 使用步骤

1. **下载文件**
   - 下载 `vectors.tsv` (向量文件)
   - 下载 `metadata.tsv` (元数据文件)

2. **访问 Projector**
   - 打开 [https://projector.tensorflow.org/](https://projector.tensorflow.org/)

3. **加载数据**
   - 点击左侧的 **"Load"** 按钮
   - 在 **"Step 1: Load a TSV file of vectors"** 中选择 `vectors.tsv`
   - 在 **"Step 2: Load a TSV file of metadata"** 中选择 `metadata.tsv`
   - 点击 **"Publish"** 外的按钮加载数据

4. **探索数据**
   - 使用 **PCA**、**t-SNE**、**UMAP** 切换降维方法
   - 在搜索框中搜索特定 Token
   - 点击数据点查看最近邻
""")

# 图示
st.markdown("### 界面预览")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 加载界面
    ```
    ┌─────────────────────────────┐
    │  Load data                  │
    │                             │
    │  Step 1: Load vectors.tsv   │
    │  [Choose file...]           │
    │                             │
    │  Step 2: Load metadata.tsv  │
    │  [Choose file...]           │
    │                             │
    │  [Load] [Publish]           │
    └─────────────────────────────┘
    ```
    """)

with col2:
    st.markdown("""
    #### 可视化界面
    ```
    ┌─────────────────────────────┐
    │  [PCA] [t-SNE] [UMAP]       │
    │  ┌───────────────────────┐  │
    │  │                       │  │
    │  │   ○ ○                 │  │
    │  │     ○ ○ ○             │  │
    │  │   ○     ○ ○           │  │
    │  │                       │  │
    │  └───────────────────────┘  │
    │  Search: [____________]     │
    └─────────────────────────────┘
    ```
    """)

st.markdown("---")


# =============================================================================
# ONNX 模型使用指南
# =============================================================================

st.markdown("## 📖 ONNX 模型使用指南")

st.markdown("""
`word2vec.onnx` 是导出的 ONNX 格式模型，可以在多种平台上进行推理。

### Python 使用示例
""")

st.code("""
import numpy as np
import onnxruntime as ort

# 加载模型
session = ort.InferenceSession("word2vec.onnx")

# 准备输入 (token_id)
token_ids = np.array([1, 2, 3], dtype=np.int64)

# 执行推理
outputs = session.run(None, {"token_ids": token_ids})
embeddings = outputs[0]  # shape: (3, 128)

print(f"嵌入向量形状: {embeddings.shape}")
""", language="python")

st.markdown("""
### JavaScript 使用示例 (ONNX Runtime Web)
""")

st.code("""
import * as ort from 'onnxruntime-web';

async function getEmbeddings(tokenIds) {
    // 加载模型
    const session = await ort.InferenceSession.create('word2vec.onnx');
    
    // 准备输入
    const inputTensor = new ort.Tensor('int64', 
        BigInt64Array.from(tokenIds.map(BigInt)), 
        [tokenIds.length]
    );
    
    // 执行推理
    const results = await session.run({ token_ids: inputTensor });
    const embeddings = results.embeddings.data;
    
    return embeddings;
}
""", language="javascript")

st.markdown("---")


# =============================================================================
# 文件格式说明
# =============================================================================

st.markdown("## 📋 文件格式说明")

file_formats = {
    "embeddings.npy": {
        "格式": "NumPy 二进制",
        "形状": "(vocab_size, 128)",
        "用途": "Python 中直接加载使用",
        "示例": "np.load('embeddings.npy')",
    },
    "embeddings.json": {
        "格式": "JSON",
        "结构": '{"tokens": [...], "embeddings": [...], "metadata": {...}}',
        "用途": "网页可视化、跨平台使用",
        "示例": "json.load(open('embeddings.json'))",
    },
    "clustering.json": {
        "格式": "JSON",
        "结构": '{"points": [...], "clusters": [...], "metadata": {...}}',
        "用途": "预计算的 t-SNE 坐标和聚类标签",
        "示例": "包含 x, y 坐标和 cluster 标签",
    },
    "vectors.tsv": {
        "格式": "TSV (制表符分隔)",
        "结构": "每行一个向量，128 个浮点数用制表符分隔",
        "用途": "TensorFlow Embedding Projector",
        "示例": "0.123\\t0.456\\t0.789\\t...",
    },
    "metadata.tsv": {
        "格式": "TSV",
        "结构": "每行一个 Token 名称",
        "用途": "与 vectors.tsv 配合使用",
        "示例": "MOV_tt0111161\\nACT_nm0000001\\n...",
    },
    "word2vec.onnx": {
        "格式": "ONNX",
        "输入": "token_ids (int64, [batch_size])",
        "输出": "embeddings (float32, [batch_size, 128])",
        "用途": "跨平台在线推理",
    },
}

for file_name, info in file_formats.items():
    with st.expander(f"📄 {file_name}"):
        for key, value in info.items():
            st.markdown(f"**{key}:** {value}")

