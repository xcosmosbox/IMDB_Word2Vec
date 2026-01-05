# Person D: 特征存储 (Feature Store)

## 你的角色
你是一名数据工程师，负责实现生成式推荐系统的 **特征存储模块**，包括在线特征服务、离线特征存储、特征同步等。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读接口定义文件：**

```
data-pipeline/interfaces.py
```

你需要实现的接口：

```python
class OnlineFeatureStoreInterface(ABC):
    @abstractmethod
    def get_features(self, entity_type, entity_id, feature_names) -> Dict:
        pass
    
    @abstractmethod
    def get_features_batch(self, entity_type, entity_ids, feature_names) -> Dict:
        pass
    
    @abstractmethod
    def set_features(self, entity_type, entity_id, features, ttl) -> bool:
        pass

class OfflineFeatureStoreInterface(ABC):
    @abstractmethod
    def write_features(self, features, table_name) -> int:
        pass
    
    @abstractmethod
    def read_features(self, entity_type, entity_ids, feature_names, ...) -> List:
        pass
    
    @abstractmethod
    def generate_training_data(self, label_table, feature_tables, output_path) -> str:
        pass
```

---

## 技术栈

- **在线存储**: Redis Cluster
- **离线存储**: Delta Lake / Parquet + PostgreSQL
- **向量存储**: Milvus
- **同步**: Kafka + Flink

---

## 你的任务

```
data-pipeline/feature-store/
├── online/
│   ├── __init__.py
│   ├── redis_store.py        # Redis 特征存储
│   ├── cache.py              # 本地缓存层
│   └── client.py             # 统一客户端
├── offline/
│   ├── __init__.py
│   ├── parquet_store.py      # Parquet 存储
│   ├── delta_store.py        # Delta Lake 存储
│   ├── pg_store.py           # PostgreSQL 存储
│   └── training_data.py      # 训练数据生成
├── sync/
│   ├── __init__.py
│   ├── offline_to_online.py  # 离线到在线同步
│   └── online_to_offline.py  # 在线到离线同步
├── vector/
│   ├── __init__.py
│   ├── milvus_store.py       # Milvus 向量存储
│   └── index_manager.py      # 索引管理
└── tests/
    ├── test_online_store.py
    ├── test_offline_store.py
    └── test_sync.py
```

---

## 1. Redis 在线特征存储 (online/redis_store.py)

```python
"""
Redis 在线特征存储

低延迟特征服务，用于实时推理
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging
from dataclasses import asdict

import redis
from redis.cluster import RedisCluster

from ..interfaces import OnlineFeatureStoreInterface, FeatureValue

logger = logging.getLogger(__name__)


class RedisFeatureStore(OnlineFeatureStoreInterface):
    """
    Redis 特征存储
    
    使用示例:
        store = RedisFeatureStore(config)
        
        # 设置特征
        store.set_features("user", "user_123", {
            "age": 25,
            "gender": "M",
            "embedding": [0.1, 0.2, ...],
        })
        
        # 获取特征
        features = store.get_features("user", "user_123", ["age", "gender"])
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._client = self._create_client()
        self._key_prefix = config.get('key_prefix', 'features')
        self._default_ttl = config.get('default_ttl', 86400)  # 1 day
    
    def _create_client(self):
        """创建 Redis 客户端"""
        if self.config.get('cluster', False):
            return RedisCluster(
                host=self.config['host'],
                port=self.config.get('port', 6379),
                password=self.config.get('password'),
                decode_responses=True,
            )
        else:
            return redis.Redis(
                host=self.config['host'],
                port=self.config.get('port', 6379),
                password=self.config.get('password'),
                db=self.config.get('db', 0),
                decode_responses=True,
            )
    
    def _make_key(self, entity_type: str, entity_id: str) -> str:
        """生成 Redis Key"""
        return f"{self._key_prefix}:{entity_type}:{entity_id}"
    
    def get_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        获取实体特征
        
        Args:
            entity_type: "user" 或 "item"
            entity_id: 实体 ID
            feature_names: 特征名列表
            
        Returns:
            Dict: 特征名 -> 特征值
        """
        key = self._make_key(entity_type, entity_id)
        
        if feature_names:
            values = self._client.hmget(key, feature_names)
            result = {}
            for name, value in zip(feature_names, values):
                if value is not None:
                    result[name] = self._deserialize_value(value)
            return result
        else:
            # 获取所有特征
            data = self._client.hgetall(key)
            return {k: self._deserialize_value(v) for k, v in data.items()}
    
    def get_features_batch(
        self,
        entity_type: str,
        entity_ids: List[str],
        feature_names: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """批量获取特征"""
        pipeline = self._client.pipeline()
        
        for entity_id in entity_ids:
            key = self._make_key(entity_type, entity_id)
            if feature_names:
                pipeline.hmget(key, feature_names)
            else:
                pipeline.hgetall(key)
        
        results = pipeline.execute()
        
        output = {}
        for entity_id, result in zip(entity_ids, results):
            if feature_names:
                output[entity_id] = {
                    name: self._deserialize_value(value)
                    for name, value in zip(feature_names, result)
                    if value is not None
                }
            else:
                output[entity_id] = {
                    k: self._deserialize_value(v)
                    for k, v in result.items()
                }
        
        return output
    
    def set_features(
        self,
        entity_type: str,
        entity_id: str,
        features: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """设置特征"""
        key = self._make_key(entity_type, entity_id)
        
        try:
            # 序列化特征值
            serialized = {
                name: self._serialize_value(value)
                for name, value in features.items()
            }
            
            pipeline = self._client.pipeline()
            pipeline.hset(key, mapping=serialized)
            
            if ttl:
                pipeline.expire(key, ttl)
            elif self._default_ttl:
                pipeline.expire(key, self._default_ttl)
            
            pipeline.execute()
            return True
            
        except Exception as e:
            logger.error(f"Failed to set features: {e}")
            return False
    
    def set_features_batch(
        self,
        entity_type: str,
        features_batch: Dict[str, Dict[str, Any]],
        ttl: Optional[int] = None,
    ) -> int:
        """批量设置特征"""
        success_count = 0
        pipeline = self._client.pipeline()
        
        for entity_id, features in features_batch.items():
            key = self._make_key(entity_type, entity_id)
            serialized = {
                name: self._serialize_value(value)
                for name, value in features.items()
            }
            pipeline.hset(key, mapping=serialized)
            
            if ttl or self._default_ttl:
                pipeline.expire(key, ttl or self._default_ttl)
        
        try:
            pipeline.execute()
            success_count = len(features_batch)
        except Exception as e:
            logger.error(f"Batch set failed: {e}")
        
        return success_count
    
    def delete_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
    ) -> bool:
        """删除特征"""
        key = self._make_key(entity_type, entity_id)
        
        try:
            if feature_names:
                self._client.hdel(key, *feature_names)
            else:
                self._client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete features: {e}")
            return False
    
    def _serialize_value(self, value: Any) -> str:
        """序列化值"""
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        return str(value)
    
    def _deserialize_value(self, value: str) -> Any:
        """反序列化值"""
        if value is None:
            return None
        
        # 尝试 JSON 解析
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass
        
        # 尝试数值解析
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        return value
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            self._client.ping()
            return True
        except Exception:
            return False
```

---

## 2. 离线特征存储 (offline/parquet_store.py)

```python
"""
Parquet 离线特征存储

用于特征历史存储和训练数据生成
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..interfaces import OfflineFeatureStoreInterface, FeatureValue

logger = logging.getLogger(__name__)


class ParquetFeatureStore(OfflineFeatureStoreInterface):
    """
    Parquet 离线特征存储
    
    使用示例:
        store = ParquetFeatureStore(config)
        
        # 写入特征
        store.write_features(features, "user_features")
        
        # 读取特征
        features = store.read_features(
            entity_type="user",
            entity_ids=["user_1", "user_2"],
            feature_names=["age", "gender"],
        )
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_path = Path(config.get('base_path', './feature_store'))
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.partition_by = config.get('partition_by', ['date'])
    
    def write_features(
        self,
        features: List[FeatureValue],
        table_name: str,
    ) -> int:
        """
        写入特征
        
        Args:
            features: 特征值列表
            table_name: 表名
            
        Returns:
            int: 写入行数
        """
        if not features:
            return 0
        
        # 转换为 DataFrame
        data = []
        for f in features:
            row = {
                'entity_type': f.entity_type,
                'entity_id': f.entity_id,
                'feature_name': f.feature_name,
                'value': self._serialize_value(f.value),
                'timestamp': f.timestamp,
                'date': f.timestamp.date().isoformat(),
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 写入 Parquet
        table_path = self.base_path / table_name
        
        pq.write_to_dataset(
            pa.Table.from_pandas(df),
            root_path=str(table_path),
            partition_cols=self.partition_by,
            existing_data_behavior='overwrite_or_ignore',
        )
        
        logger.info(f"Wrote {len(features)} features to {table_name}")
        return len(features)
    
    def read_features(
        self,
        entity_type: str,
        entity_ids: List[str],
        feature_names: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[FeatureValue]:
        """读取历史特征"""
        results = []
        
        # 扫描所有特征表
        for table_path in self.base_path.iterdir():
            if not table_path.is_dir():
                continue
            
            try:
                # 构建过滤条件
                filters = [
                    ('entity_type', '=', entity_type),
                ]
                
                if entity_ids:
                    filters.append(('entity_id', 'in', entity_ids))
                
                if feature_names:
                    filters.append(('feature_name', 'in', feature_names))
                
                if start_time:
                    filters.append(('date', '>=', start_time.date().isoformat()))
                
                if end_time:
                    filters.append(('date', '<', end_time.date().isoformat()))
                
                # 读取数据
                dataset = pq.ParquetDataset(
                    str(table_path),
                    filters=filters if filters else None,
                )
                
                df = dataset.read().to_pandas()
                
                for _, row in df.iterrows():
                    results.append(FeatureValue(
                        feature_name=row['feature_name'],
                        value=self._deserialize_value(row['value']),
                        timestamp=pd.to_datetime(row['timestamp']),
                        entity_id=row['entity_id'],
                        entity_type=row['entity_type'],
                    ))
                
            except Exception as e:
                logger.warning(f"Failed to read from {table_path}: {e}")
        
        return results
    
    def generate_training_data(
        self,
        label_table: str,
        feature_tables: List[str],
        output_path: str,
    ) -> str:
        """
        生成训练数据集
        
        将标签表和特征表 Join
        """
        # 读取标签表
        label_path = self.base_path / label_table
        labels_df = pq.read_table(str(label_path)).to_pandas()
        
        # 依次 Join 特征表
        result_df = labels_df
        
        for feature_table in feature_tables:
            feature_path = self.base_path / feature_table
            
            try:
                features_df = pq.read_table(str(feature_path)).to_pandas()
                
                # Pivot 特征（行转列）
                features_pivot = features_df.pivot_table(
                    index=['entity_id', 'entity_type', 'date'],
                    columns='feature_name',
                    values='value',
                    aggfunc='last',
                ).reset_index()
                
                # Join
                result_df = result_df.merge(
                    features_pivot,
                    on=['entity_id', 'entity_type', 'date'],
                    how='left',
                )
                
            except Exception as e:
                logger.warning(f"Failed to join {feature_table}: {e}")
        
        # 保存结果
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_df.to_parquet(str(output_path), index=False)
        
        logger.info(f"Generated training data: {output_path} ({len(result_df)} rows)")
        return str(output_path)
    
    def get_feature_statistics(
        self,
        feature_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, float]:
        """获取特征统计信息"""
        values = []
        
        for table_path in self.base_path.iterdir():
            if not table_path.is_dir():
                continue
            
            try:
                filters = [
                    ('feature_name', '=', feature_name),
                    ('date', '>=', start_time.date().isoformat()),
                    ('date', '<', end_time.date().isoformat()),
                ]
                
                dataset = pq.ParquetDataset(str(table_path), filters=filters)
                df = dataset.read().to_pandas()
                
                numeric_values = pd.to_numeric(df['value'], errors='coerce').dropna()
                values.extend(numeric_values.tolist())
                
            except Exception:
                continue
        
        if not values:
            return {}
        
        import numpy as np
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
        }
    
    def _serialize_value(self, value: Any) -> str:
        """序列化值"""
        import json
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)
    
    def _deserialize_value(self, value: str) -> Any:
        """反序列化值"""
        import json
        try:
            return json.loads(value)
        except:
            try:
                return float(value)
            except:
                return value
```

---

## 3. 特征同步 (sync/offline_to_online.py)

```python
"""
离线到在线特征同步

将离线计算的特征同步到在线存储
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..online.redis_store import RedisFeatureStore
from ..offline.parquet_store import ParquetFeatureStore
from ..interfaces import FeatureValue

logger = logging.getLogger(__name__)


class OfflineToOnlineSync:
    """
    离线到在线同步
    
    使用示例:
        sync = OfflineToOnlineSync(
            offline_store=parquet_store,
            online_store=redis_store,
        )
        
        sync.sync_features(
            feature_tables=["user_features", "item_features"],
            entity_type="user",
        )
    """
    
    def __init__(
        self,
        offline_store: ParquetFeatureStore,
        online_store: RedisFeatureStore,
        batch_size: int = 1000,
    ):
        self.offline_store = offline_store
        self.online_store = online_store
        self.batch_size = batch_size
    
    def sync_features(
        self,
        feature_tables: List[str],
        entity_type: str,
        feature_names: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        ttl: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        同步特征
        
        Returns:
            Dict: 同步统计信息
        """
        stats = {
            'total_features': 0,
            'synced_entities': 0,
            'failed_entities': 0,
            'start_time': datetime.now().isoformat(),
        }
        
        # 从离线存储读取特征
        all_features = []
        for table in feature_tables:
            features = self.offline_store.read_features(
                entity_type=entity_type,
                entity_ids=[],  # 读取所有
                feature_names=feature_names or [],
                start_time=start_time,
                end_time=end_time,
            )
            all_features.extend(features)
        
        stats['total_features'] = len(all_features)
        logger.info(f"Read {len(all_features)} features from offline store")
        
        # 按实体分组
        entity_features: Dict[str, Dict[str, Any]] = {}
        for f in all_features:
            if f.entity_id not in entity_features:
                entity_features[f.entity_id] = {}
            entity_features[f.entity_id][f.feature_name] = f.value
        
        # 批量写入在线存储
        batch = {}
        for entity_id, features in entity_features.items():
            batch[entity_id] = features
            
            if len(batch) >= self.batch_size:
                success = self.online_store.set_features_batch(
                    entity_type=entity_type,
                    features_batch=batch,
                    ttl=ttl,
                )
                stats['synced_entities'] += success
                stats['failed_entities'] += len(batch) - success
                batch = {}
        
        # 处理剩余批次
        if batch:
            success = self.online_store.set_features_batch(
                entity_type=entity_type,
                features_batch=batch,
                ttl=ttl,
            )
            stats['synced_entities'] += success
            stats['failed_entities'] += len(batch) - success
        
        stats['end_time'] = datetime.now().isoformat()
        logger.info(f"Sync completed: {stats}")
        
        return stats
    
    def sync_latest(
        self,
        entity_type: str,
        feature_names: List[str],
        ttl: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        同步最新特征
        
        只同步每个实体的最新特征值
        """
        # 读取最近一天的数据
        end_time = datetime.now()
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        return self.sync_features(
            feature_tables=["daily_features"],
            entity_type=entity_type,
            feature_names=feature_names,
            start_time=start_time,
            end_time=end_time,
            ttl=ttl,
        )
```

---

## 注意事项

1. Redis 使用 Pipeline 提高批量操作性能
2. Parquet 使用分区提高查询效率
3. 特征同步需要幂等性
4. 处理特征版本和时间点查询
5. 监控特征延迟和覆盖率

## 输出要求

请输出完整的可运行代码，包含：
1. Redis 在线存储
2. Parquet 离线存储
3. 特征同步
4. 向量存储
5. 完整测试

