# IMDb Word2Vec 项目说明

本项目将 IMDb 公共数据集（非商业用途）转化为可训练的向量表示，包含数据下载、清洗、特征工程、自编码融合与 Word2Vec 训练的完整流水线。所有代码已模块化，支持 GPU 自动检测，如无 GPU 自动回退 CPU。默认使用可调子集以便快速验证，可切换为全量数据。

## 目录结构
- `imdb_word2vec/`：源码包，包含配置、下载、预处理、特征工程、融合与训练模块。
- `imdb_data/`：IMDb 原始 TSV 及解压结果（自动创建）。
- `cache/`：中间产物（CSV、Parquet、词汇表等）。
- `artifacts/`：模型与导出向量（`best_model.keras`、`vectors.tsv`、`metadata.tsv`）。
- `logs/`：运行日志。
- `imdb_venv/`：Python3.12 虚拟环境。

## 环境准备
```bash
cd /Users/yuxiang.feng/Desktop/imdb_wrod2vec/IMDB_Word2Vec/imdb_word2vec
python3.12 -m venv imdb_venv
source imdb_venv/bin/activate
pip install -r requirements.txt
```
如为 macOS 且有 Metal GPU，可选安装 `tensorflow-metal` 提升训练速度。

## 快速开始（子集训练）
```bash
source imdb_venv/bin/activate
python -m imdb_word2vec.cli all --subset-rows 100000 --max-rows 50000 --max-seq 50000
```
- `--subset-rows`：预处理阶段采样行数（None 为全量）。
- `--max-rows`：自编码器训练的行数上限，便于小样本验证。
- `--max-seq`：Word2Vec 训练序列数上限。
- 默认词表上限 `vocab-limit=20000`，可按需调整。

## 分步执行
```bash
python -m imdb_word2vec.cli download             # 下载并解压 TSV
python -m imdb_word2vec.cli preprocess --subset-rows 100000
python -m imdb_word2vec.cli fe
python -m imdb_word2vec.cli fusion --max-rows 50000
python -m imdb_word2vec.cli train --max-seq 50000 --vocab-limit 20000
```

## 数据来源与合规
- 数据下载自 <https://datasets.imdbws.com/>，仅限非商业用途。
- 所有中间文件均存放在本地目录，未包含再分发逻辑。

## 日志与产物
- 日志：`logs/imdb_word2vec.log`
- 中间数据：`cache/`（`movies_info_df.csv`、`staff_df.csv`、`regional_titles_df.csv`、`final_mapped_vec.csv`、`fused_features.parquet` 等）
- 模型与向量：`artifacts/best_model.keras`、`vectors.tsv`、`metadata.tsv`

## 训练提示
- GPU 可用时 TensorFlow 自动使用 GPU；如无 GPU 自动回退 CPU。
- 对全量数据训练前，请确认内存与磁盘空间充足；可通过子集参数逐步扩大规模。


