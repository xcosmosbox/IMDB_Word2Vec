# Person C: 特征工程 (Feature Engineering)

## 你的角色
你是一名数据工程师/ML 工程师，负责实现生成式推荐系统的 **特征工程模块**，包括特征转换、特征管道、特征注册等。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读接口定义文件：**

```
data-pipeline/interfaces.py
```

你需要实现的接口：

```python
class FeatureTransformerInterface(ABC):
    @abstractmethod
    def fit(self, data: List[Dict]) -> None:
        pass
    
    @abstractmethod
    def transform(self, data: Dict) -> Dict:
        pass
    
    @abstractmethod
    def fit_transform(self, data: List[Dict]) -> List[Dict]:
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass

class FeaturePipelineInterface(ABC):
    @abstractmethod
    def add_transformer(self, name: str, transformer) -> None:
        pass
    
    @abstractmethod
    def run(self, data: Dict) -> Dict:
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        pass
```

---

## 技术栈

- **框架**: scikit-learn, PyTorch
- **数值计算**: NumPy, Pandas
- **序列化**: Pickle, Joblib
- **向量化**: SentenceTransformers (文本嵌入)

---

## 你的任务

```
data-pipeline/feature-engineering/
├── transformers/
│   ├── __init__.py
│   ├── base.py               # 转换器基类
│   ├── numeric.py            # 数值特征转换
│   ├── categorical.py        # 类别特征转换
│   ├── temporal.py           # 时间特征转换
│   ├── text.py               # 文本特征转换
│   ├── embedding.py          # 嵌入特征转换
│   └── interaction.py        # 交叉特征转换
├── pipelines/
│   ├── __init__.py
│   ├── user_pipeline.py      # 用户特征管道
│   ├── item_pipeline.py      # 物品特征管道
│   └── context_pipeline.py   # 上下文特征管道
├── registry/
│   ├── __init__.py
│   ├── feature_registry.py   # 特征注册表
│   └── version_manager.py    # 版本管理
└── tests/
    ├── test_transformers.py
    ├── test_pipelines.py
    └── test_registry.py
```

---

## 1. 转换器基类 (transformers/base.py)

```python
"""
特征转换器基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pickle
import logging
from pathlib import Path

from ..interfaces import FeatureTransformerInterface

logger = logging.getLogger(__name__)


class BaseFeatureTransformer(FeatureTransformerInterface):
    """
    特征转换器基类
    
    所有具体转换器都应继承此类
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self._fitted = False
        self._input_features: List[str] = []
        self._output_features: List[str] = []
    
    @abstractmethod
    def fit(self, data: List[Dict[str, Any]]) -> None:
        """训练转换器"""
        pass
    
    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """转换特征"""
        pass
    
    def fit_transform(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """训练并转换"""
        self.fit(data)
        return [self.transform(d) for d in data]
    
    def save(self, path: str) -> None:
        """保存转换器状态"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'name': self.name,
            'config': self.config,
            'fitted': self._fitted,
            'input_features': self._input_features,
            'output_features': self._output_features,
            'state': self._get_state(),
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved transformer to {path}")
    
    def load(self, path: str) -> None:
        """加载转换器状态"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.name = state['name']
        self.config = state['config']
        self._fitted = state['fitted']
        self._input_features = state['input_features']
        self._output_features = state['output_features']
        self._set_state(state['state'])
        
        logger.info(f"Loaded transformer from {path}")
    
    @abstractmethod
    def _get_state(self) -> Dict[str, Any]:
        """获取内部状态（用于保存）"""
        pass
    
    @abstractmethod
    def _set_state(self, state: Dict[str, Any]) -> None:
        """设置内部状态（用于加载）"""
        pass
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted
    
    @property
    def input_features(self) -> List[str]:
        return self._input_features
    
    @property
    def output_features(self) -> List[str]:
        return self._output_features
```

---

## 2. 数值特征转换器 (transformers/numeric.py)

```python
"""
数值特征转换器
"""

from typing import Dict, Any, List, Optional
import numpy as np

from .base import BaseFeatureTransformer


class StandardScaler(BaseFeatureTransformer):
    """
    标准化转换器
    
    z = (x - mean) / std
    """
    
    def __init__(
        self,
        name: str,
        features: List[str],
        with_mean: bool = True,
        with_std: bool = True,
    ):
        super().__init__(name, {
            'features': features,
            'with_mean': with_mean,
            'with_std': with_std,
        })
        self._input_features = features
        self._output_features = [f"{f}_scaled" for f in features]
        self._mean: Dict[str, float] = {}
        self._std: Dict[str, float] = {}
    
    def fit(self, data: List[Dict[str, Any]]) -> None:
        """计算均值和标准差"""
        for feature in self._input_features:
            values = [d.get(feature, 0) for d in data if d.get(feature) is not None]
            if values:
                self._mean[feature] = np.mean(values)
                self._std[feature] = np.std(values) or 1.0  # 避免除零
        self._fitted = True
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化转换"""
        result = data.copy()
        
        for i, feature in enumerate(self._input_features):
            value = data.get(feature)
            if value is not None:
                mean = self._mean.get(feature, 0)
                std = self._std.get(feature, 1)
                
                if self.config['with_mean']:
                    value = value - mean
                if self.config['with_std']:
                    value = value / std
                
                result[self._output_features[i]] = value
        
        return result
    
    def _get_state(self) -> Dict[str, Any]:
        return {'mean': self._mean, 'std': self._std}
    
    def _set_state(self, state: Dict[str, Any]) -> None:
        self._mean = state['mean']
        self._std = state['std']


class MinMaxScaler(BaseFeatureTransformer):
    """
    最小-最大归一化
    
    x_scaled = (x - min) / (max - min)
    """
    
    def __init__(
        self,
        name: str,
        features: List[str],
        feature_range: tuple = (0, 1),
    ):
        super().__init__(name, {
            'features': features,
            'feature_range': feature_range,
        })
        self._input_features = features
        self._output_features = [f"{f}_normalized" for f in features]
        self._min: Dict[str, float] = {}
        self._max: Dict[str, float] = {}
    
    def fit(self, data: List[Dict[str, Any]]) -> None:
        for feature in self._input_features:
            values = [d.get(feature, 0) for d in data if d.get(feature) is not None]
            if values:
                self._min[feature] = min(values)
                self._max[feature] = max(values)
        self._fitted = True
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = data.copy()
        low, high = self.config['feature_range']
        
        for i, feature in enumerate(self._input_features):
            value = data.get(feature)
            if value is not None:
                min_val = self._min.get(feature, 0)
                max_val = self._max.get(feature, 1)
                
                if max_val != min_val:
                    scaled = (value - min_val) / (max_val - min_val)
                    scaled = scaled * (high - low) + low
                else:
                    scaled = low
                
                result[self._output_features[i]] = scaled
        
        return result
    
    def _get_state(self) -> Dict[str, Any]:
        return {'min': self._min, 'max': self._max}
    
    def _set_state(self, state: Dict[str, Any]) -> None:
        self._min = state['min']
        self._max = state['max']


class LogTransformer(BaseFeatureTransformer):
    """
    对数转换器
    
    用于处理长尾分布
    """
    
    def __init__(self, name: str, features: List[str], base: float = np.e):
        super().__init__(name, {'features': features, 'base': base})
        self._input_features = features
        self._output_features = [f"{f}_log" for f in features]
    
    def fit(self, data: List[Dict[str, Any]]) -> None:
        self._fitted = True
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = data.copy()
        
        for i, feature in enumerate(self._input_features):
            value = data.get(feature)
            if value is not None and value > 0:
                if self.config['base'] == np.e:
                    result[self._output_features[i]] = np.log1p(value)
                else:
                    result[self._output_features[i]] = np.log1p(value) / np.log(self.config['base'])
        
        return result
    
    def _get_state(self) -> Dict[str, Any]:
        return {}
    
    def _set_state(self, state: Dict[str, Any]) -> None:
        pass


class BucketTransformer(BaseFeatureTransformer):
    """
    分桶转换器
    
    将连续值离散化
    """
    
    def __init__(
        self,
        name: str,
        feature: str,
        boundaries: List[float],
        labels: Optional[List[str]] = None,
    ):
        super().__init__(name, {
            'feature': feature,
            'boundaries': boundaries,
            'labels': labels,
        })
        self._input_features = [feature]
        self._output_features = [f"{feature}_bucket"]
        self.boundaries = boundaries
        self.labels = labels or [f"bucket_{i}" for i in range(len(boundaries) + 1)]
    
    def fit(self, data: List[Dict[str, Any]]) -> None:
        self._fitted = True
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = data.copy()
        feature = self._input_features[0]
        value = data.get(feature)
        
        if value is not None:
            bucket_idx = 0
            for i, boundary in enumerate(self.boundaries):
                if value > boundary:
                    bucket_idx = i + 1
            result[self._output_features[0]] = self.labels[bucket_idx]
        
        return result
    
    def _get_state(self) -> Dict[str, Any]:
        return {}
    
    def _set_state(self, state: Dict[str, Any]) -> None:
        pass
```

---

## 3. 类别特征转换器 (transformers/categorical.py)

```python
"""
类别特征转换器
"""

from typing import Dict, Any, List, Optional
import numpy as np
from collections import Counter

from .base import BaseFeatureTransformer


class OneHotEncoder(BaseFeatureTransformer):
    """
    独热编码转换器
    """
    
    def __init__(
        self,
        name: str,
        feature: str,
        max_categories: int = 100,
        unknown_value: str = "__UNK__",
    ):
        super().__init__(name, {
            'feature': feature,
            'max_categories': max_categories,
            'unknown_value': unknown_value,
        })
        self._input_features = [feature]
        self._categories: List[str] = []
        self._category_to_idx: Dict[str, int] = {}
    
    def fit(self, data: List[Dict[str, Any]]) -> None:
        feature = self._input_features[0]
        counter = Counter(d.get(feature) for d in data if d.get(feature))
        
        # 取最常见的类别
        most_common = counter.most_common(self.config['max_categories'])
        self._categories = [cat for cat, _ in most_common]
        self._category_to_idx = {cat: i for i, cat in enumerate(self._categories)}
        
        self._output_features = [f"{feature}_{cat}" for cat in self._categories]
        self._fitted = True
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = data.copy()
        feature = self._input_features[0]
        value = data.get(feature)
        
        # 初始化为 0
        for output_feature in self._output_features:
            result[output_feature] = 0
        
        if value is not None and value in self._category_to_idx:
            idx = self._category_to_idx[value]
            result[self._output_features[idx]] = 1
        
        return result
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            'categories': self._categories,
            'category_to_idx': self._category_to_idx,
        }
    
    def _set_state(self, state: Dict[str, Any]) -> None:
        self._categories = state['categories']
        self._category_to_idx = state['category_to_idx']


class LabelEncoder(BaseFeatureTransformer):
    """
    标签编码转换器
    """
    
    def __init__(
        self,
        name: str,
        feature: str,
        unknown_value: int = -1,
    ):
        super().__init__(name, {
            'feature': feature,
            'unknown_value': unknown_value,
        })
        self._input_features = [feature]
        self._output_features = [f"{feature}_encoded"]
        self._label_to_idx: Dict[str, int] = {}
        self._idx_to_label: Dict[int, str] = {}
    
    def fit(self, data: List[Dict[str, Any]]) -> None:
        feature = self._input_features[0]
        unique_values = sorted(set(
            d.get(feature) for d in data if d.get(feature) is not None
        ))
        
        self._label_to_idx = {v: i for i, v in enumerate(unique_values)}
        self._idx_to_label = {i: v for v, i in self._label_to_idx.items()}
        self._fitted = True
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = data.copy()
        feature = self._input_features[0]
        value = data.get(feature)
        
        if value is not None:
            result[self._output_features[0]] = self._label_to_idx.get(
                value, self.config['unknown_value']
            )
        
        return result
    
    def inverse_transform(self, encoded: int) -> Optional[str]:
        """反向转换"""
        return self._idx_to_label.get(encoded)
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            'label_to_idx': self._label_to_idx,
            'idx_to_label': self._idx_to_label,
        }
    
    def _set_state(self, state: Dict[str, Any]) -> None:
        self._label_to_idx = state['label_to_idx']
        self._idx_to_label = state['idx_to_label']


class TargetEncoder(BaseFeatureTransformer):
    """
    目标编码转换器
    
    用目标变量的均值替换类别
    """
    
    def __init__(
        self,
        name: str,
        feature: str,
        target: str,
        smoothing: float = 1.0,
    ):
        super().__init__(name, {
            'feature': feature,
            'target': target,
            'smoothing': smoothing,
        })
        self._input_features = [feature]
        self._output_features = [f"{feature}_target_encoded"]
        self._encoding: Dict[str, float] = {}
        self._global_mean: float = 0
    
    def fit(self, data: List[Dict[str, Any]]) -> None:
        feature = self._input_features[0]
        target = self.config['target']
        smoothing = self.config['smoothing']
        
        # 计算全局均值
        target_values = [d[target] for d in data if target in d]
        self._global_mean = np.mean(target_values) if target_values else 0
        
        # 按类别聚合
        category_stats: Dict[str, List[float]] = {}
        for d in data:
            cat = d.get(feature)
            val = d.get(target)
            if cat is not None and val is not None:
                if cat not in category_stats:
                    category_stats[cat] = []
                category_stats[cat].append(val)
        
        # 平滑编码
        for cat, values in category_stats.items():
            n = len(values)
            cat_mean = np.mean(values)
            # 贝叶斯平滑
            self._encoding[cat] = (
                (n * cat_mean + smoothing * self._global_mean) /
                (n + smoothing)
            )
        
        self._fitted = True
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = data.copy()
        feature = self._input_features[0]
        value = data.get(feature)
        
        if value is not None:
            result[self._output_features[0]] = self._encoding.get(
                value, self._global_mean
            )
        
        return result
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            'encoding': self._encoding,
            'global_mean': self._global_mean,
        }
    
    def _set_state(self, state: Dict[str, Any]) -> None:
        self._encoding = state['encoding']
        self._global_mean = state['global_mean']
```

---

## 4. 特征管道 (pipelines/user_pipeline.py)

```python
"""
用户特征管道
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from ..interfaces import FeaturePipelineInterface, FeatureTransformerInterface
from ..transformers import (
    StandardScaler,
    MinMaxScaler,
    LogTransformer,
    OneHotEncoder,
    LabelEncoder,
)

logger = logging.getLogger(__name__)


class FeaturePipeline(FeaturePipelineInterface):
    """
    特征管道基类
    """
    
    def __init__(self, name: str):
        self.name = name
        self._transformers: Dict[str, FeatureTransformerInterface] = {}
        self._order: List[str] = []
    
    def add_transformer(
        self,
        name: str,
        transformer: FeatureTransformerInterface,
    ) -> None:
        """添加转换器"""
        self._transformers[name] = transformer
        self._order.append(name)
        logger.debug(f"Added transformer: {name}")
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """运行管道"""
        result = data.copy()
        
        for name in self._order:
            transformer = self._transformers[name]
            result = transformer.transform(result)
        
        return result
    
    def run_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量运行"""
        return [self.run(d) for d in data]
    
    def fit(self, data: List[Dict[str, Any]]) -> None:
        """训练所有转换器"""
        current_data = data
        
        for name in self._order:
            transformer = self._transformers[name]
            transformer.fit(current_data)
            current_data = [transformer.transform(d) for d in current_data]
            logger.info(f"Fitted transformer: {name}")
    
    def fit_transform(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """训练并转换"""
        self.fit(data)
        return self.run_batch(data)
    
    def get_feature_names(self) -> List[str]:
        """获取输出特征名"""
        feature_names = []
        for name in self._order:
            transformer = self._transformers[name]
            feature_names.extend(transformer.output_features)
        return feature_names
    
    def save(self, path: str) -> None:
        """保存管道"""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for name, transformer in self._transformers.items():
            transformer.save(str(save_dir / f"{name}.pkl"))
        
        # 保存顺序
        with open(save_dir / "order.txt", "w") as f:
            f.write("\n".join(self._order))
        
        logger.info(f"Saved pipeline to {path}")
    
    def load(self, path: str) -> None:
        """加载管道"""
        save_dir = Path(path)
        
        with open(save_dir / "order.txt", "r") as f:
            self._order = f.read().strip().split("\n")
        
        for name in self._order:
            transformer = self._transformers.get(name)
            if transformer:
                transformer.load(str(save_dir / f"{name}.pkl"))
        
        logger.info(f"Loaded pipeline from {path}")


class UserFeaturePipeline(FeaturePipeline):
    """
    用户特征管道
    
    使用示例:
        pipeline = UserFeaturePipeline.create_default()
        pipeline.fit(user_data)
        features = pipeline.run(user_record)
    """
    
    @classmethod
    def create_default(cls) -> "UserFeaturePipeline":
        """创建默认用户特征管道"""
        pipeline = cls("user_features")
        
        # 数值特征标准化
        pipeline.add_transformer(
            "age_scaler",
            StandardScaler("age_scaler", features=["age"])
        )
        
        # 行为计数对数化
        pipeline.add_transformer(
            "behavior_log",
            LogTransformer("behavior_log", features=[
                "view_count", "click_count", "purchase_count"
            ])
        )
        
        # 类别编码
        pipeline.add_transformer(
            "gender_encoder",
            LabelEncoder("gender_encoder", feature="gender")
        )
        
        pipeline.add_transformer(
            "city_encoder",
            OneHotEncoder("city_encoder", feature="city", max_categories=50)
        )
        
        return pipeline
```

---

## 注意事项

1. 确保 fit/transform 分离
2. 处理未知类别
3. 数值特征处理缺失值
4. 支持增量更新
5. 版本化管理

## 输出要求

请输出完整的可运行代码，包含：
1. 各类转换器
2. 特征管道
3. 特征注册表
4. 完整测试

