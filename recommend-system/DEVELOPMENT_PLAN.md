# 生成式推荐系统 - 开发计划

## 📊 当前进度

### ✅ 已完成：基础设施层 (Go Backend)

| 模块 | 状态 | 说明 |
|------|------|------|
| `pkg/config` | ✅ 完成 | 配置管理 (Viper) |
| `pkg/logger` | ✅ 完成 | 日志系统 (Zap) |
| `pkg/database/postgres` | ✅ 完成 | PostgreSQL 连接 |
| `pkg/database/redis` | ✅ 完成 | Redis 缓存 |
| `pkg/database/milvus` | ✅ 完成 | Milvus 向量库 |
| `pkg/utils` | ✅ 完成 | 工具函数 |
| `internal/middleware` | ✅ 完成 | 认证/限流/追踪 |
| `internal/model` | ✅ 完成 | 数据模型 |
| `internal/repository` | ✅ 完成 | 数据访问层 |
| `internal/cache` | ✅ 完成 | 缓存抽象 |
| `internal/inference` | ✅ 完成 | 推理客户端 |
| `internal/service/recommend` | ✅ 完成 | 推荐服务 |
| `api/recommend/v1` | ✅ 完成 | HTTP API |
| `deployments/` | ✅ 完成 | Docker/K8s 配置 |

### 🔄 进行中：算法层 (Python)

| 模块 | 负责人 | 状态 | 预计时间 |
|------|--------|------|----------|
| `algorithm/semantic_id/` | Person A | 📝 待开发 | 1 周 |
| `algorithm/encoder/` | Person B | 📝 待开发 | 1 周 |
| `algorithm/decoder/` | Person C | 📝 待开发 | 1 周 |
| `algorithm/training/` | Person D | 📝 待开发 | 1 周 |
| `algorithm/feature_engineering/` | Person E | 📝 待开发 | 1 周 |
| `algorithm/serving/` | Person F | 📝 待开发 | 1 周 |

---

## 👥 团队分工

### 6 人并行开发方案

```
┌─────────────────────────────────────────────────────────────┐
│                    Week 1-2: 并行开发                        │
├─────────────────────────────────────────────────────────────┤
│  Person A ──→ Semantic ID 编码器 (RQ-VAE)                   │
│  Person B ──→ UGT Encoder (用户行为编码)                     │
│  Person C ──→ UGT Decoder + MoE (推荐生成)                   │
│  Person D ──→ Training Pipeline (三阶段训练)                 │
│  Person E ──→ Feature Engineering (数据处理)                 │
│  Person F ──→ Serving (模型部署)                             │
├─────────────────────────────────────────────────────────────┤
│                    Week 3: 集成测试                          │
├─────────────────────────────────────────────────────────────┤
│  A + B + C ──→ 完整 UGT 模型                                │
│  E ──→ 准备训练数据                                          │
│  D ──→ 端到端训练测试                                        │
├─────────────────────────────────────────────────────────────┤
│                    Week 4: 部署验证                          │
├─────────────────────────────────────────────────────────────┤
│  F ──→ 模型导出和性能测试                                    │
│  全员 ──→ 全链路压测                                         │
└─────────────────────────────────────────────────────────────┘
```

### 独立性保证

每个人的工作通过 **接口定义** 解耦：

```python
# algorithm/interfaces.py 定义了所有模块的接口
# 每个人只需要：
# 1. 实现自己的接口
# 2. 调用其他人的接口（而非具体实现）
```

---

## 📋 各角色任务详情

### Person A: Semantic ID 编码器

**输入**: 物品特征向量 (batch, 256)  
**输出**: 三层语义 ID (L1, L2, L3)

**核心技术**:
- RQ-VAE (残差向量量化)
- EMA 码本更新
- 码本大小: [1024, 4096, 16384]

**提示词文件**: `algorithm/prompts/person_a_semantic_id.md`

---

### Person B: UGT Encoder

**输入**: Token 序列 + 位置 + 类型  
**输出**: 用户表示向量 (batch, d_model)

**核心技术**:
- Dot-Product Aggregated Attention (ReLU 替代 Softmax)
- Group Layer Normalization (分组归一化)
- 12 层 Transformer Encoder

**提示词文件**: `algorithm/prompts/person_b_encoder.md`

---

### Person C: UGT Decoder + MoE

**输入**: 用户表示向量  
**输出**: 推荐物品的语义 ID 序列

**核心技术**:
- Mixture of Experts (16 专家, Top-4 激活)
- 交叉注意力
- 自回归生成 (Beam Search)

**提示词文件**: `algorithm/prompts/person_c_decoder.md`

---

### Person D: Training Pipeline

**输入**: 模型 + 数据  
**输出**: 训练好的权重

**核心技术**:
- 三阶段训练 (预训练→微调→偏好对齐)
- DeepSpeed ZeRO-2
- 混合精度 (FP16)

**提示词文件**: `algorithm/prompts/person_d_training.md`

---

### Person E: Feature Engineering

**输入**: 原始日志 (JSON/CSV)  
**输出**: 训练数据 (Parquet/TFRecord)

**核心技术**:
- 统一事件 Token 化
- 词表管理
- Spark 大规模处理

**提示词文件**: `algorithm/prompts/person_e_feature_engineering.md`

---

### Person F: Serving

**输入**: PyTorch 模型  
**输出**: Triton 推理服务

**核心技术**:
- ONNX 导出
- TensorRT 优化
- Triton Inference Server

**提示词文件**: `algorithm/prompts/person_f_serving.md`

---

## 🚀 快速开始

### 1. 阅读接口定义
```bash
cat algorithm/interfaces.py
```

### 2. 阅读自己的提示词
```bash
cat algorithm/prompts/person_X_xxx.md  # 替换为你的角色
```

### 3. 开始开发
```bash
cd algorithm/your_module/
# 创建代码文件
# 实现接口
# 编写测试
```

### 4. 运行测试
```bash
pytest algorithm/your_module/tests/ -v
```

---

## 📐 技术规范

### 代码风格
- Python: PEP 8
- 注释: 中文
- 文档: docstring + README

### 测试覆盖
- 目标: >= 80%
- 框架: pytest

### 提交规范
```
feat(module): 添加功能描述
fix(module): 修复问题描述
docs(module): 更新文档
test(module): 添加测试
```

---

## 📞 沟通协调

### 接口变更
如需修改 `interfaces.py`，必须：
1. 在群里通知所有相关人员
2. 等待确认后再修改
3. 同步更新所有实现

### 依赖问题
如果你的模块依赖其他人的输出：
1. 先使用 Mock 数据开发
2. 集成时再替换为真实实现

---

## 📅 里程碑

| 时间 | 里程碑 | 交付物 |
|------|--------|--------|
| Week 1 | 模块开发完成 | 各模块代码 + 单元测试 |
| Week 2 | 模块联调 | 集成测试通过 |
| Week 3 | 端到端训练 | 训练好的模型 |
| Week 4 | 部署上线 | 可用的推理服务 |

