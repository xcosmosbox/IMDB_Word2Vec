# 生成式推荐系统 - 算法模块

## 📁 目录结构

```
algorithm/
├── interfaces.py              # 公共接口定义（所有模块必须遵循）
├── requirements.txt           # Python 依赖
├── prompts/                   # 各模块开发提示词
│   ├── person_a_semantic_id.md
│   ├── person_b_encoder.md
│   ├── person_c_decoder.md
│   ├── person_d_training.md
│   ├── person_e_feature_engineering.md
│   └── person_f_serving.md
├── semantic_id/               # Person A: 语义 ID 编码器
├── encoder/                   # Person B: UGT 编码器
├── decoder/                   # Person C: UGT 解码器
├── training/                  # Person D: 训练 Pipeline
├── feature_engineering/       # Person E: 特征工程
└── serving/                   # Person F: 推理服务
```

## 👥 任务分配

| 角色 | 模块 | 职责 | 依赖 |
|------|------|------|------|
| **Person A** | `semantic_id/` | RQ-VAE 语义 ID 编码器 | 无 |
| **Person B** | `encoder/` | 用户行为编码器 | 无 |
| **Person C** | `decoder/` | 推荐生成解码器 (含 MoE) | 无 |
| **Person D** | `training/` | 三阶段训练 Pipeline | 接口定义 |
| **Person E** | `feature_engineering/` | 数据处理和 Token 化 | 无 |
| **Person F** | `serving/` | 模型导出和部署 | 接口定义 |

## 🔧 开发指南

### 1. 阅读接口定义

所有开发者必须先阅读 `interfaces.py`，理解自己模块需要实现的接口。

```python
# 示例：Person A 需要实现
class SemanticIDEncoderInterface(ABC):
    @abstractmethod
    def encode(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass
```

### 2. 阅读对应的提示词

进入 `prompts/` 目录，阅读自己角色对应的提示词文件。

### 3. 实现代码

在对应的模块目录下创建代码文件。

### 4. 编写测试

每个模块必须包含单元测试（放在 `tests/` 子目录）。

## 📐 接口概览

### SemanticIDEncoder (Person A)
```python
# 输入：物品特征向量 (batch, 256)
# 输出：三层语义 ID (L1, L2, L3)
l1, l2, l3 = encoder.encode(features)
```

### UGTEncoder (Person B)
```python
# 输入：Token 序列
# 输出：用户表示向量 (batch, d_model)
user_repr = encoder([l1_ids, l2_ids, l3_ids], positions, token_types, mask)
```

### UGTDecoder (Person C)
```python
# 输入：编码器输出
# 输出：推荐列表 [(L1, L2, L3), ...]
recommendations = decoder.generate(encoder_output, num_recommendations=20)
```

### Trainer (Person D)
```python
# 三阶段训练
trainer.train()  # 完整训练流程
```

### Tokenizer (Person E)
```python
# 输入：事件列表
# 输出：Token 化序列
tokenized = tokenizer.tokenize_events(events)
```

### ServingExporter (Person F)
```python
# 导出和部署
exporter.export_onnx(model, "model.onnx", config)
exporter.optimize_tensorrt("model.onnx", "model.plan", config)
```

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行测试
pytest tests/ -v

# 3. 训练模型
python training/scripts/train_stage1.py --config configs/stage1.yaml
```

## 📝 注意事项

1. **接口一致性**：严格按照 `interfaces.py` 定义实现
2. **代码风格**：遵循 PEP 8，使用中文注释
3. **测试覆盖**：目标覆盖率 >= 80%
4. **文档**：每个模块需要 README 和 docstring

