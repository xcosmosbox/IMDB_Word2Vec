# Person B: ETL 流水线 (Extract-Transform-Load)

## 你的角色
你是一名数据工程师，负责实现生成式推荐系统的 **ETL 流水线**，包括数据抽取、转换、加载，支持批处理和流处理。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读接口定义文件：**

```
data-pipeline/interfaces.py
```

你需要实现的接口：

```python
class ExtractorInterface(ABC):
    @abstractmethod
    def extract(self, source, start_time, end_time) -> Generator:
        pass

class TransformerInterface(ABC):
    @abstractmethod
    def transform(self, data: Dict) -> Dict:
        pass
    
    @abstractmethod
    def transform_batch(self, data: List[Dict]) -> List[Dict]:
        pass

class LoaderInterface(ABC):
    @abstractmethod
    def load(self, data: Dict) -> bool:
        pass
    
    @abstractmethod
    def load_batch(self, data: List[Dict]) -> Tuple[int, int]:
        pass

class ETLPipelineInterface(ABC):
    @abstractmethod
    def run(self, source, target, transformers) -> Dict:
        pass
```

---

## 技术栈

- **批处理**: Apache Spark 3.x
- **流处理**: Apache Flink 1.x
- **调度**: Apache Airflow
- **存储**: Parquet, Delta Lake
- **Python**: PySpark, PyFlink

---

## 你的任务

```
data-pipeline/etl/
├── extractors/
│   ├── __init__.py
│   ├── base.py               # 抽取器基类
│   ├── postgres.py           # PostgreSQL 抽取器
│   ├── kafka.py              # Kafka 抽取器
│   ├── s3.py                 # S3 抽取器
│   └── api.py                # API 抽取器
├── transformers/
│   ├── __init__.py
│   ├── base.py               # 转换器基类
│   ├── cleaning.py           # 数据清洗
│   ├── enrichment.py         # 数据增强
│   ├── aggregation.py        # 数据聚合
│   └── normalization.py      # 数据标准化
├── loaders/
│   ├── __init__.py
│   ├── base.py               # 加载器基类
│   ├── postgres.py           # PostgreSQL 加载器
│   ├── parquet.py            # Parquet 加载器
│   ├── delta.py              # Delta Lake 加载器
│   └── redis.py              # Redis 加载器
├── spark/
│   ├── __init__.py
│   ├── jobs/
│   │   ├── daily_aggregation.py
│   │   ├── user_features.py
│   │   └── item_features.py
│   └── utils.py
├── flink/
│   ├── __init__.py
│   ├── jobs/
│   │   ├── realtime_features.py
│   │   └── session_windows.py
│   └── operators.py
├── pipeline/
│   ├── __init__.py
│   ├── batch_pipeline.py     # 批处理管道
│   ├── stream_pipeline.py    # 流处理管道
│   └── config.py
└── tests/
    ├── test_extractors.py
    ├── test_transformers.py
    └── test_loaders.py
```

---

## 1. 抽取器基类 (extractors/base.py)

```python
"""
数据抽取器基类

定义抽取器的通用接口和工具方法
"""

from abc import ABC, abstractmethod
from typing import Generator, Dict, Any, Optional
from datetime import datetime
import logging

from ..interfaces import ExtractorInterface

logger = logging.getLogger(__name__)


class BaseExtractor(ExtractorInterface):
    """
    抽取器基类
    
    所有具体抽取器都应继承此类
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """连接数据源"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass
    
    @abstractmethod
    def extract(
        self,
        source: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """抽取数据"""
        pass
    
    @abstractmethod
    def get_schema(self, source: str) -> Dict[str, str]:
        """获取数据源 Schema"""
        pass
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    def get_row_count(self, source: str) -> int:
        """获取数据源行数"""
        raise NotImplementedError
```

---

## 2. PostgreSQL 抽取器 (extractors/postgres.py)

```python
"""
PostgreSQL 数据抽取器
"""

from typing import Generator, Dict, Any, Optional
from datetime import datetime
import logging

import psycopg2
from psycopg2.extras import RealDictCursor

from .base import BaseExtractor

logger = logging.getLogger(__name__)


class PostgresExtractor(BaseExtractor):
    """
    PostgreSQL 抽取器
    
    使用示例:
        config = {
            "host": "localhost",
            "port": 5432,
            "database": "recommend",
            "user": "postgres",
            "password": "password",
        }
        
        with PostgresExtractor(config) as extractor:
            for row in extractor.extract("users"):
                process(row)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._conn = None
        self._cursor = None
    
    def connect(self) -> bool:
        """连接 PostgreSQL"""
        try:
            self._conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config.get('port', 5432),
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
            )
            self._cursor = self._conn.cursor(cursor_factory=RealDictCursor)
            self._connected = True
            logger.info(f"Connected to PostgreSQL: {self.config['host']}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开连接"""
        if self._cursor:
            self._cursor.close()
        if self._conn:
            self._conn.close()
        self._connected = False
        logger.info("Disconnected from PostgreSQL")
    
    def extract(
        self,
        source: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        batch_size: int = 10000,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        抽取数据
        
        Args:
            source: 表名或 SQL 查询
            start_time: 开始时间（用于增量抽取）
            end_time: 结束时间
            batch_size: 批次大小
        """
        if not self._connected:
            raise RuntimeError("Not connected to database")
        
        # 构建查询
        if source.strip().upper().startswith("SELECT"):
            query = source
        else:
            query = f"SELECT * FROM {source}"
            
            # 增量条件
            conditions = []
            params = []
            
            if start_time:
                conditions.append("created_at >= %s")
                params.append(start_time)
            if end_time:
                conditions.append("created_at < %s")
                params.append(end_time)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        logger.info(f"Extracting from: {source}")
        
        # 使用服务端游标
        cursor_name = f"extractor_{id(self)}"
        self._cursor.execute(f"DECLARE {cursor_name} CURSOR FOR {query}", params if params else None)
        
        total_rows = 0
        while True:
            self._cursor.execute(f"FETCH {batch_size} FROM {cursor_name}")
            rows = self._cursor.fetchall()
            
            if not rows:
                break
            
            for row in rows:
                total_rows += 1
                yield dict(row)
        
        self._cursor.execute(f"CLOSE {cursor_name}")
        logger.info(f"Extracted {total_rows} rows from {source}")
    
    def get_schema(self, source: str) -> Dict[str, str]:
        """获取表 Schema"""
        self._cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, (source,))
        
        return {row['column_name']: row['data_type'] for row in self._cursor.fetchall()}
    
    def get_row_count(self, source: str) -> int:
        """获取表行数"""
        self._cursor.execute(f"SELECT COUNT(*) FROM {source}")
        return self._cursor.fetchone()['count']
```

---

## 3. 转换器 (transformers/cleaning.py)

```python
"""
数据清洗转换器
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import re

from .base import BaseTransformer


class DataCleaningTransformer(BaseTransformer):
    """
    数据清洗转换器
    
    功能:
    - 去除空值
    - 类型转换
    - 格式标准化
    - 异常值处理
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.null_handling = self.config.get('null_handling', 'drop')
        self.type_mapping = self.config.get('type_mapping', {})
        self.date_format = self.config.get('date_format', '%Y-%m-%d %H:%M:%S')
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """转换单条数据"""
        result = {}
        
        for key, value in data.items():
            # 处理空值
            if value is None:
                if self.null_handling == 'drop':
                    continue
                elif self.null_handling == 'default':
                    value = self._get_default_value(key)
                # 'keep' 则保留 None
            
            # 类型转换
            if key in self.type_mapping:
                value = self._convert_type(value, self.type_mapping[key])
            
            # 字符串清洗
            if isinstance(value, str):
                value = self._clean_string(value)
            
            result[key] = value
        
        return result
    
    def transform_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量转换"""
        return [self.transform(d) for d in data if d]
    
    def get_output_schema(self) -> Dict[str, str]:
        """获取输出 Schema"""
        return self.type_mapping.copy()
    
    def _clean_string(self, value: str) -> str:
        """清洗字符串"""
        # 去除首尾空白
        value = value.strip()
        # 替换多个空格为单个
        value = re.sub(r'\s+', ' ', value)
        # 移除控制字符
        value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        return value
    
    def _convert_type(self, value: Any, target_type: str) -> Any:
        """类型转换"""
        if value is None:
            return None
        
        try:
            if target_type == 'int':
                return int(value)
            elif target_type == 'float':
                return float(value)
            elif target_type == 'str':
                return str(value)
            elif target_type == 'bool':
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes')
                return bool(value)
            elif target_type == 'datetime':
                if isinstance(value, str):
                    return datetime.strptime(value, self.date_format)
                return value
        except (ValueError, TypeError) as e:
            return None
        
        return value
    
    def _get_default_value(self, key: str) -> Any:
        """获取默认值"""
        defaults = self.config.get('defaults', {})
        return defaults.get(key)


class DuplicateRemover(BaseTransformer):
    """去重转换器"""
    
    def __init__(self, key_fields: List[str]):
        super().__init__({})
        self.key_fields = key_fields
        self._seen = set()
    
    def transform(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """去重"""
        key = tuple(data.get(f) for f in self.key_fields)
        
        if key in self._seen:
            return None
        
        self._seen.add(key)
        return data
    
    def transform_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量去重"""
        return [r for d in data if (r := self.transform(d)) is not None]
    
    def reset(self):
        """重置去重状态"""
        self._seen.clear()
```

---

## 4. 批处理管道 (pipeline/batch_pipeline.py)

```python
"""
批处理 ETL 管道
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import time

from ..interfaces import (
    ExtractorInterface,
    TransformerInterface,
    LoaderInterface,
    ETLPipelineInterface,
)

logger = logging.getLogger(__name__)


class BatchETLPipeline(ETLPipelineInterface):
    """
    批处理 ETL 管道
    
    使用示例:
        extractor = PostgresExtractor(db_config)
        transformer = DataCleaningTransformer(clean_config)
        loader = ParquetLoader(output_config)
        
        pipeline = BatchETLPipeline(
            extractor=extractor,
            loader=loader,
            batch_size=10000,
        )
        
        stats = pipeline.run(
            source="user_behaviors",
            target="clean_behaviors.parquet",
            transformers=[transformer],
        )
    """
    
    def __init__(
        self,
        extractor: ExtractorInterface,
        loader: LoaderInterface,
        batch_size: int = 10000,
        error_threshold: float = 0.05,
    ):
        self.extractor = extractor
        self.loader = loader
        self.batch_size = batch_size
        self.error_threshold = error_threshold
        self._status = "idle"
        self._stats = {}
    
    def run(
        self,
        source: str,
        target: str,
        transformers: List[TransformerInterface],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        运行 ETL 管道
        
        Returns:
            Dict: 执行统计信息
        """
        self._status = "running"
        start = time.time()
        
        stats = {
            "source": source,
            "target": target,
            "start_time": datetime.now().isoformat(),
            "extracted": 0,
            "transformed": 0,
            "loaded": 0,
            "failed": 0,
            "errors": [],
        }
        
        try:
            # 创建目标表
            schema = self.extractor.get_schema(source)
            for transformer in transformers:
                if hasattr(transformer, 'get_output_schema'):
                    schema.update(transformer.get_output_schema())
            self.loader.create_table_if_not_exists(schema)
            
            batch = []
            
            for record in self.extractor.extract(source, start_time, end_time):
                stats["extracted"] += 1
                
                # 应用转换
                try:
                    transformed = record
                    for transformer in transformers:
                        transformed = transformer.transform(transformed)
                        if transformed is None:
                            break
                    
                    if transformed:
                        stats["transformed"] += 1
                        batch.append(transformed)
                    
                except Exception as e:
                    stats["failed"] += 1
                    stats["errors"].append(str(e))
                    logger.warning(f"Transform error: {e}")
                
                # 批量加载
                if len(batch) >= self.batch_size:
                    success, fail = self.loader.load_batch(batch)
                    stats["loaded"] += success
                    stats["failed"] += fail
                    batch = []
                    
                    # 检查错误率
                    error_rate = stats["failed"] / max(stats["extracted"], 1)
                    if error_rate > self.error_threshold:
                        raise RuntimeError(f"Error rate {error_rate:.2%} exceeds threshold")
            
            # 加载剩余数据
            if batch:
                success, fail = self.loader.load_batch(batch)
                stats["loaded"] += success
                stats["failed"] += fail
            
            stats["status"] = "success"
            
        except Exception as e:
            stats["status"] = "failed"
            stats["error"] = str(e)
            logger.error(f"Pipeline failed: {e}")
            raise
        
        finally:
            stats["end_time"] = datetime.now().isoformat()
            stats["duration_seconds"] = time.time() - start
            self._status = "idle"
            self._stats = stats
        
        logger.info(f"Pipeline completed: {stats}")
        return stats
    
    def schedule(self, cron_expression: str) -> str:
        """调度管道（需要集成 Airflow）"""
        # TODO: 生成 Airflow DAG
        raise NotImplementedError("Use Airflow for scheduling")
    
    def get_status(self) -> Dict[str, Any]:
        """获取管道状态"""
        return {
            "status": self._status,
            "stats": self._stats,
        }
```

---

## 5. Spark 作业 (spark/jobs/daily_aggregation.py)

```python
"""
Spark 日聚合作业

聚合用户行为数据
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, timedelta
import argparse


def create_spark_session(app_name: str) -> SparkSession:
    """创建 Spark Session"""
    return (SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate())


def daily_user_aggregation(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    date: str,
) -> dict:
    """
    用户日行为聚合
    
    输入: 原始行为事件
    输出: 用户日聚合特征
    """
    # 读取数据
    df = (spark.read.parquet(input_path)
        .filter(F.col("date") == date))
    
    # 用户日聚合
    user_daily = df.groupBy("user_id", "date").agg(
        # 行为计数
        F.count("*").alias("total_events"),
        F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("view_count"),
        F.sum(F.when(F.col("event_type") == "click", 1).otherwise(0)).alias("click_count"),
        F.sum(F.when(F.col("event_type") == "like", 1).otherwise(0)).alias("like_count"),
        F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("purchase_count"),
        
        # 物品统计
        F.countDistinct("item_id").alias("unique_items"),
        F.collect_set("item_id").alias("viewed_items"),
        
        # 时间统计
        F.min("timestamp").alias("first_event_time"),
        F.max("timestamp").alias("last_event_time"),
        
        # 分类统计
        F.collect_set("category").alias("categories"),
        F.countDistinct("category").alias("unique_categories"),
    )
    
    # 计算衍生特征
    user_daily = user_daily.withColumn(
        "click_through_rate",
        F.col("click_count") / F.greatest(F.col("view_count"), F.lit(1))
    ).withColumn(
        "session_duration_hours",
        (F.unix_timestamp("last_event_time") - F.unix_timestamp("first_event_time")) / 3600
    )
    
    # 写入
    (user_daily.write
        .mode("overwrite")
        .partitionBy("date")
        .parquet(output_path))
    
    # 返回统计
    stats = {
        "date": date,
        "users_processed": user_daily.count(),
        "output_path": output_path,
    }
    
    return stats


def daily_item_aggregation(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    date: str,
) -> dict:
    """
    物品日聚合
    """
    df = (spark.read.parquet(input_path)
        .filter(F.col("date") == date))
    
    item_daily = df.groupBy("item_id", "date").agg(
        F.count("*").alias("total_events"),
        F.countDistinct("user_id").alias("unique_users"),
        F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("view_count"),
        F.sum(F.when(F.col("event_type") == "click", 1).otherwise(0)).alias("click_count"),
        F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("purchase_count"),
        F.avg("rating").alias("avg_rating"),
    )
    
    (item_daily.write
        .mode("overwrite")
        .partitionBy("date")
        .parquet(output_path))
    
    return {"date": date, "items_processed": item_daily.count()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Processing date (YYYY-MM-DD)")
    parser.add_argument("--input", required=True, help="Input path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--type", choices=["user", "item"], default="user")
    args = parser.parse_args()
    
    spark = create_spark_session("DailyAggregation")
    
    try:
        if args.type == "user":
            stats = daily_user_aggregation(spark, args.input, args.output, args.date)
        else:
            stats = daily_item_aggregation(spark, args.input, args.output, args.date)
        print(f"Job completed: {stats}")
    finally:
        spark.stop()
```

---

## 注意事项

1. 使用增量抽取减少数据量
2. 批量处理提高效率
3. 错误处理和重试机制
4. 监控管道执行状态
5. 支持断点续传

## 输出要求

请输出完整的可运行代码，包含：
1. 各类抽取器
2. 转换器
3. 加载器
4. Spark/Flink 作业
5. 完整测试

