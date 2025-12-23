# 📋 IMDb Word2Vec 项目优缺点完整清单

---

## ✅ 优点清单（应保留）

---

### 1. 流式训练架构

**描述**: 将大规模数据分块处理，每次只加载一部分到内存，训练后释放。

**代码示例**:
```python
# training.py
for global_epoch in range(global_epochs):
    for chunk_idx in chunk_indices:
        targets, contexts = _load_positive_pairs_from_disk(chunk_path)
        # 训练
        model.fit(dataset, ...)
        # 释放内存
        del targets, contexts
        gc.collect()
```

**保留理由**:
| 方面 | 说明 |
|------|------|
| **内存效率** | 将 337 GB 数据压缩到 20 GB 磁盘 + 6 GB 内存运行 |
| **可扩展性** | 数据量增长 10 倍也能处理，只是时间增加 |
| **断点续训** | 每个 chunk 独立，训练中断可从上次位置继续 |
| **工业实践** | 大厂训练大模型都采用类似的分块策略 |

**未来迭代建议**: 保留并增强，添加 checkpoint 机制保存训练进度。

---

### 2. 配置集中管理

**描述**: 使用 dataclass 将所有配置项集中在 `config.py`，避免硬编码。

**代码示例**:
```python
# config.py
@dataclass
class TrainConfig:
    window_size: int = 5
    num_negative_samples: int = 5
    embedding_dim: int = 128
    # ...

@dataclass  
class Config:
    paths: PathConfig
    data: DataConfig
    train: TrainConfig
```

**保留理由**:
| 方面 | 说明 |
|------|------|
| **可维护性** | 修改参数只需改一处，不用全局搜索 |
| **可读性** | 新人一看 config.py 就知道有哪些可配置项 |
| **环境适配** | 可轻松扩展为从环境变量或 YAML 文件加载 |
| **类型安全** | dataclass 提供类型提示，IDE 可以自动补全 |

**未来迭代建议**: 保留，考虑支持从 `config.yaml` 加载配置。

---

### 3. 多设备自动检测

**描述**: 自动检测 NVIDIA CUDA、Apple Metal、CPU，无需用户手动配置。

**代码示例**:
```python
# config.py
def detect_device() -> Tuple[str, str]:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        if "NVIDIA" in device_name.upper():
            return "/GPU:0", "NVIDIA"
        if sys.platform == "darwin":
            return "/GPU:0", "Metal"
    return "/CPU:0", "CPU"
```

**保留理由**:
| 方面 | 说明 |
|------|------|
| **用户友好** | 用户不需要关心底层硬件 |
| **跨平台** | Windows/Linux/macOS 都能运行 |
| **优雅降级** | 没有 GPU 自动回退到 CPU，不会报错 |
| **零配置** | 即插即用，降低使用门槛 |

**未来迭代建议**: 保留，添加对 AMD ROCm 的支持。

---

### 4. 实体类型前缀设计

**描述**: 为不同类型的实体添加前缀，使 Word2Vec 能区分语义。

**代码示例**:
```python
# feature_engineering.py
ENTITY_PREFIXES = {
    "movie": "MOV_",    # MOV_tt0111161
    "actor": "ACT_",    # ACT_nm0000151
    "genre": "GEN_",    # GEN_Drama
    "era": "ERA_",      # ERA_1990s
}
```

**保留理由**:
| 方面 | 说明 |
|------|------|
| **语义区分** | "Drama" 作为类型 vs 作为电影名，前缀明确区分 |
| **避免碰撞** | 不同类型的 ID 可能相同，前缀避免混淆 |
| **可解释性** | 看到 `MOV_tt0111161` 立刻知道是电影 |
| **下游友好** | 推荐系统可以按前缀过滤候选集 |

**未来迭代建议**: 保留，但需要统一同一实体的前缀（一人一前缀）。

---

### 5. 只存正样本的优化策略

**描述**: 预生成阶段只存储正样本对，负样本在训练时动态采样。

**代码示例**:
```python
# pretraining.py - 只存 8 bytes/样本
np.savez_compressed(chunk_path, targets=targets, contexts=contexts)

# training.py - 动态添加负样本
negatives = rng.choice(len(neg_prob), size=(n, num_ns), p=neg_prob)
```

**保留理由**:
| 方面 | 说明 |
|------|------|
| **空间效率** | 从 172 bytes/样本降到 8 bytes/样本，节省 95% |
| **灵活性** | 可随时调整负样本数，无需重新生成数据 |
| **随机性** | 每个 epoch 使用不同负样本，泛化更好 |
| **研究表明** | 动态负采样效果不比静态差 |

**未来迭代建议**: 保留，这是核心优化。

---

### 6. 多种序列类型设计

**描述**: 设计 9 种不同的序列类型，捕获多维度关系。

**代码示例**:
```python
# feature_engineering.py
all_sequences.extend(_generate_person_movie_sequences(...))    # 人-电影
all_sequences.extend(_generate_movie_context_sequences(...))   # 电影上下文
all_sequences.extend(_generate_coactor_sequences(...))         # 合作演员
all_sequences.extend(_generate_era_movie_sequences(...))       # 同年代
all_sequences.extend(_generate_rating_movie_sequences(...))    # 同评分
all_sequences.extend(_generate_director_genre_sequences(...))  # 导演偏好
all_sequences.extend(_generate_actor_genre_sequences(...))     # 演员偏好
```

**保留理由**:
| 方面 | 说明 |
|------|------|
| **关系丰富** | 捕获人-物、物-物、人-人多种关系 |
| **特征工程** | 这是推荐系统最核心的部分 |
| **可扩展** | 容易添加新的序列类型 |
| **创新点** | 年代/评分序列是有价值的创新 |

**未来迭代建议**: 保留并扩展，考虑添加用户行为序列。

---

### 7. 完善的导出模块

**描述**: 支持多种格式导出，便于不同场景使用。

**代码示例**:
```python
# export.py
export_tsv()           # TensorFlow Projector
export_onnx()          # 在线推理
export_json_embeddings()  # 网页可视化
export_clustering_visualization()  # t-SNE 聚类
export_html_visualization()  # 交互式网页
export_recommendation_config()  # 推荐系统配置
```

**保留理由**:
| 方面 | 说明 |
|------|------|
| **部署友好** | ONNX 可部署到任何推理引擎 |
| **可视化** | HTML 可直接展示成果 |
| **调试方便** | JSON 格式便于检查和调试 |
| **标准兼容** | TSV 兼容 TensorFlow Embedding Projector |

**未来迭代建议**: 保留，考虑添加 TensorRT 导出。

---

### 8. CLI 命令行设计

**描述**: 提供完整的命令行接口，支持分步执行和完整流程。

**代码示例**:
```bash
python -m imdb_word2vec.cli download
python -m imdb_word2vec.cli preprocess
python -m imdb_word2vec.cli fe
python -m imdb_word2vec.cli pretrain
python -m imdb_word2vec.cli train --use-cache
python -m imdb_word2vec.cli export
python -m imdb_word2vec.cli all  # 一键执行
```

**保留理由**:
| 方面 | 说明 |
|------|------|
| **灵活性** | 可分步执行，便于调试单个步骤 |
| **自动化** | 可集成到 shell 脚本或 CI/CD |
| **用户友好** | 比修改代码更直观 |
| **可组合** | 不同命令可自由组合 |

**未来迭代建议**: 保留，添加 `--help` 的中文说明。

---

### 9. 模型参数统计功能

**描述**: 训练前打印模型参数量和大小。

**代码示例**:
```python
# training.py
def _log_model_stats(model: Word2Vec):
    logger.info("总参数量: %s (%s)", f"{total:,}", param_str)
    logger.info("模型大小: %.2f MB (float32)", stats["model_size_mb"])

# 输出:
# 总参数量: 12,800,000 (12.80M)
# 模型大小: 48.83 MB
```

**保留理由**:
| 方面 | 说明 |
|------|------|
| **透明度** | 用户清楚知道模型规模 |
| **对比基准** | 便于与其他模型对比 |
| **资源估算** | 可据此估算训练/推理资源 |
| **专业性** | 这是大模型时代的标准做法 |

**未来迭代建议**: 保留，添加 FLOPs 估算。

---

### 10. 日志系统规范

**描述**: 统一的日志格式，带时间戳和级别。

**代码示例**:
```python
# logging_utils.py
def setup_logging(logs_dir: Path) -> logging.Logger:
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # ...

# 输出:
# [2025-12-23 17:14:24][INFO] 词表规模: 50000
```

**保留理由**:
| 方面 | 说明 |
|------|------|
| **可追溯** | 时间戳便于定位问题 |
| **统一格式** | 所有模块日志风格一致 |
| **持久化** | 日志保存到文件，训练完可回溯 |
| **调试友好** | INFO/WARNING/ERROR 分级清晰 |

**未来迭代建议**: 保留，考虑添加 JSON 格式日志选项。

---

## ❌ 缺点清单（应修复）

---

### 1. 目标与实现不匹配

**问题描述**: 项目声称是"电影推荐系统"，实际只实现了嵌入训练。

**影响**:
```
用户期望: 输入用户 ID → 输出推荐电影列表
实际能做: 输入电影 ID → 输出相似电影（需要额外代码）

缺失组件:
├── 用户画像
├── 召回层
├── 精排层
├── 在线服务
└── 评估体系
```

**修复方案**:
```python
# 方案 A: 明确目标，改名为 "IMDb Embedding Trainer"

# 方案 B: 补全推荐系统组件
# recommender/
# ├── retrieval.py      # 召回：ANN 最近邻搜索
# ├── ranking.py        # 精排：CTR 预估模型
# ├── serving.py        # 在线服务 API
# └── evaluation.py     # 离线评估 (Recall@K, NDCG)
```

**修复优先级**: 🔴 P0（必须）

---

### 2. 测试覆盖为零

**问题描述**: `tests/` 目录只有 import 测试和路径测试，核心逻辑无测试。

**影响**:
```
风险:
├── 重构时不知道是否破坏了功能
├── 边界条件未验证
├── Bug 只能在运行时发现
└── 代码质量无法保证
```

**修复方案**:
```python
# tests/test_feature_engineering.py
def test_rating_to_bucket():
    assert _rating_to_bucket(7.3) == "7.5"
    assert _rating_to_bucket(7.2) == "7.0"
    assert _rating_to_bucket("invalid") == "UNKNOWN"

def test_add_prefix():
    assert _add_prefix("tt0111161", "movie") == "MOV_tt0111161"
    assert _add_prefix("\\N", "movie") is None

# tests/test_training.py
def test_add_negative_samples():
    targets = np.array([1, 2, 3])
    contexts = np.array([4, 5, 6])
    # 验证输出形状、类型、值范围
```

**修复优先级**: 🔴 P0（必须）

---

### 3. Autoencoder 模块孤立

**问题描述**: Autoencoder 与 Word2Vec 完全独立，生成的特征无下游消费者。

**影响**:
```python
# 当前数据流:
# feature_engineering → tabular_features.csv → autoencoder → fused_features.parquet → ❓

# fused_features.parquet 没人用！
```

**修复方案**:
```python
# 方案 A: 删除 Autoencoder（如果不需要）

# 方案 B: 与 Word2Vec 联合使用
def get_item_embedding(item_id):
    w2v_emb = word2vec.get_embedding(item_id)  # 128 维
    ae_emb = autoencoder.get_embedding(item_id)  # 64 维
    return np.concatenate([w2v_emb, ae_emb])  # 192 维

# 方案 C: 用于精排模型
class RankingModel:
    def __init__(self):
        self.w2v_embeddings = load_word2vec()
        self.ae_features = load_autoencoder()
    
    def predict(self, user_id, item_id):
        features = concat(self.w2v_embeddings[item_id], 
                         self.ae_features[item_id])
        return self.mlp(features)
```

**修复优先级**: 🟡 P1（重要）

---

### 4. 实体前缀不一致

**问题描述**: 同一个人可能有多个前缀（PER_、ACT_、DIR_）。

**影响**:
```python
# 同一个人:
# PER_nm0000151  ← 作为普通人员
# ACT_nm0000151  ← 作为演员出现
# DIR_nm0000151  ← 作为导演出现

# 问题: 这三个被视为不同实体，但实际是同一个人
# 导致: 嵌入空间分裂，无法正确计算相似度
```

**修复方案**:
```python
# 方案: 统一使用 PER_ 前缀，用属性区分角色
ENTITY_PREFIXES = {
    "movie": "MOV_",
    "person": "PER_",   # 所有人都用这个
    "genre": "GEN_",
    "era": "ERA_",
}

# 角色信息放到序列上下文中:
# [PER_nm001, ROLE_actor, MOV_tt001]
# [PER_nm001, ROLE_director, MOV_tt002]
```

**修复优先级**: 🟡 P1（重要）

---

### 5. feature_engineering.py 性能低下

**问题描述**: 大量使用 `iterrows()`，这是 pandas 最慢的遍历方式。

**影响**:
```python
# 当前代码 (慢):
for _, row in tqdm(staff_df.iterrows(), ...):  # O(n) 慢
    # 处理逻辑

# 21M 行 × 多个循环 = 数小时
```

**修复方案**:
```python
# 方案 A: 使用 groupby + apply
person_movies = (
    title_principals_df
    .groupby("nconst")["tconst"]
    .apply(list)
    .to_dict()
)

# 方案 B: 使用向量化操作
staff_df["prefixed_tconst"] = "MOV_" + staff_df["tconst"]

# 方案 C: 使用 polars (比 pandas 快 10-100 倍)
import polars as pl
df = pl.read_csv("data.csv")
result = df.groupby("nconst").agg(pl.col("tconst").alias("movies"))
```

**修复优先级**: 🟡 P1（重要）

---

### 6. 缺乏评估指标

**问题描述**: 只有 loss 和 accuracy，无推荐系统专用指标。

**影响**:
```
当前指标:
├── loss: 3.2451      ← 不知道好坏
├── accuracy: 0.43    ← 不知道好坏
└── positive_recall: 0.56  ← 不知道好坏

无法回答:
├── 推荐的电影用户会喜欢吗？
├── 相似电影真的相似吗？
└── 嵌入质量如何？
```

**修复方案**:
```python
# evaluation.py
def evaluate_embeddings(embeddings, test_pairs):
    """评估嵌入质量"""
    # 1. 相似度任务
    hit_rate_10 = calc_hit_rate(embeddings, test_pairs, k=10)
    mrr = calc_mrr(embeddings, test_pairs)
    
    # 2. 聚类质量
    silhouette = calc_silhouette_score(embeddings)
    
    # 3. 下游任务
    genre_classification_acc = eval_genre_classification(embeddings)
    
    return {
        "Hit@10": hit_rate_10,
        "MRR": mrr,
        "Silhouette": silhouette,
        "Genre_Acc": genre_classification_acc,
    }
```

**修复优先级**: 🟡 P1（重要）

---

### 7. 异常处理缺失

**问题描述**: 关键操作缺少 try-except，错误时直接崩溃。

**影响**:
```python
# 当前代码:
data = np.load(chunk_path)  # 文件损坏？直接崩溃
return data["targets"]       # key 不存在？直接崩溃

# 训练到一半崩溃，所有进度丢失
```

**修复方案**:
```python
def _load_positive_pairs_from_disk(chunk_path: Path):
    try:
        data = np.load(chunk_path)
        targets = data["targets"]
        contexts = data["contexts"]
        return targets, contexts
    except FileNotFoundError:
        logger.error("找不到文件: %s", chunk_path)
        raise
    except KeyError as e:
        logger.error("文件格式错误，缺少 key: %s", e)
        raise
    except Exception as e:
        logger.error("加载失败: %s, 错误: %s", chunk_path, e)
        raise
```

**修复优先级**: 🟠 P2（建议）

---

### 8. 魔法数字无文档

**问题描述**: 配置文件中的数值没有解释为什么选这个值。

**影响**:
```python
# config.py
subsample_t: float = 1e-4     # 为什么是 1e-4？
window_size: int = 5          # 为什么是 5？
num_negative_samples: int = 5 # 为什么是 5？
embedding_dim: int = 128      # 为什么是 128？

# 新人接手时完全不知道能不能改、改了会怎样
```

**修复方案**:
```python
@dataclass
class TrainConfig:
    # 高频子采样阈值
    # 参考: Mikolov et al. 2013, 推荐值 1e-3 到 1e-5
    # 越小越激进，丢弃更多高频词
    subsample_t: float = 1e-4
    
    # Skip-gram 窗口大小
    # 参考: 一般 5-10，窗口越大捕获越远的语义关系
    # 但计算量也越大
    window_size: int = 5
    
    # 负样本数量
    # 参考: 原论文推荐 5-20，小数据集用 5-10 足够
    # 增加可提升质量但训练变慢
    num_negative_samples: int = 5
```

**修复优先级**: 🟠 P2（建议）

---

### 9. 重复代码

**问题描述**: `TqdmProgressCallback` 在两个文件中重复定义。

**影响**:
```python
# training.py 第 39-80 行
class TqdmProgressCallback(tf.keras.callbacks.Callback):
    # ... 42 行代码

# autoencoder.py 第 34-55 行
class TqdmProgressCallback(tf.keras.callbacks.Callback):
    # ... 22 行代码

# 问题: 改一个忘了改另一个，行为不一致
```

**修复方案**:
```python
# utils.py (新文件)
class TqdmProgressCallback(tf.keras.callbacks.Callback):
    """统一的进度条回调"""
    # ... 完整实现

# training.py
from .utils import TqdmProgressCallback

# autoencoder.py
from .utils import TqdmProgressCallback
```

**修复优先级**: 🟠 P2（建议）

---

### 10. 缺少 CI/CD 和代码质量检查

**问题描述**: 没有自动化测试、格式检查、类型检查。

**影响**:
```
风险:
├── PR 合并可能引入 bug
├── 代码风格不统一
├── 类型错误在运行时才发现
└── 依赖版本不一致
```

**修复方案**:
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pip install pytest ruff mypy
      - run: ruff check .          # 代码格式
      - run: mypy imdb_word2vec/   # 类型检查
      - run: pytest tests/ -v      # 单元测试
```

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I"]

[tool.mypy]
python_version = "3.12"
warn_return_any = true
```

**修复优先级**: 🟠 P2（建议）

---

### 11. Word2Vec 算法过时

**问题描述**: 2013 年的算法，无法处理多义词和 OOV。

**影响**:
```
局限性:
├── "苹果" 只有一个向量，无法区分水果/公司
├── 新电影/新演员无法获得嵌入
├── 无法处理长文本语义
└── 质量不如预训练模型
```

**修复方案**:
```python
# 方案 A: 升级到 FastText (支持子词)
from gensim.models import FastText
model = FastText(sentences, vector_size=128, window=5)

# 方案 B: 使用 Sentence-BERT (更强语义)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["The Shawshank Redemption"])

# 方案 C: 使用 Two-Tower 模型 (专为推荐设计)
# 参考 Google 的 YouTube 推荐系统架构
```

**修复优先级**: 🔵 P3（长期）

---

## 📊 优先级总结

| 优先级 | 项目 | 预计工时 |
|--------|------|----------|
| 🔴 P0 | 明确项目目标 | 1 天 |
| 🔴 P0 | 添加单元测试 | 3 天 |
| 🟡 P1 | 整合/删除 Autoencoder | 1 天 |
| 🟡 P1 | 统一实体前缀 | 0.5 天 |
| 🟡 P1 | 优化 feature_engineering 性能 | 2 天 |
| 🟡 P1 | 添加评估指标 | 2 天 |
| 🟠 P2 | 添加异常处理 | 1 天 |
| 🟠 P2 | 文档化魔法数字 | 0.5 天 |
| 🟠 P2 | 抽取重复代码 | 0.5 天 |
| 🟠 P2 | 添加 CI/CD | 1 天 |
| 🔵 P3 | 升级嵌入算法 | 5 天 |

**总计**: 约 17.5 天工作量