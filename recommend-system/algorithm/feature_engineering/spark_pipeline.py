"""
Spark 大规模数据处理模块

使用 Apache Spark 处理 TB 级用户行为日志，包括：
- 原始日志预处理
- 用户序列构建
- 词表统计与构建
- 训练数据生成
- 数据验证与质量检查
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class SparkFeaturePipeline:
    """
    Spark 特征工程 Pipeline
    
    使用 Spark 处理大规模用户行为数据，生成训练所需的序列数据。
    
    Attributes:
        spark: SparkSession 实例
        app_name: Spark 应用名称
        
    Example:
        >>> pipeline = SparkFeaturePipeline("RecommendFeatureEngineering")
        >>> pipeline.process_raw_logs(
        ...     input_path="hdfs://data/raw_logs/",
        ...     output_path="hdfs://data/processed/",
        ...     semantic_id_mapping_path="hdfs://data/semantic_ids/"
        ... )
    """
    
    def __init__(
        self, 
        app_name: str = "RecommendFeatureEngineering",
        spark_config: Optional[Dict[str, str]] = None,
    ):
        """
        初始化 Spark Pipeline
        
        Args:
            app_name: Spark 应用名称
            spark_config: 额外的 Spark 配置
        """
        self.app_name = app_name
        self._spark = None
        self._spark_config = spark_config or {}
        
        # 默认 Spark 配置
        self._default_config = {
            "spark.sql.shuffle.partitions": "200",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        }
    
    @property
    def spark(self):
        """
        延迟初始化 SparkSession
        
        只有在实际使用时才初始化 Spark，避免在不需要 Spark 的场景下产生依赖。
        """
        if self._spark is None:
            try:
                from pyspark.sql import SparkSession
                
                builder = SparkSession.builder.appName(self.app_name)
                
                # 应用默认配置
                for key, value in self._default_config.items():
                    builder = builder.config(key, value)
                
                # 应用用户配置
                for key, value in self._spark_config.items():
                    builder = builder.config(key, value)
                
                self._spark = builder.getOrCreate()
                logger.info(f"SparkSession 已创建: {self.app_name}")
                
            except ImportError:
                raise ImportError(
                    "PySpark 未安装。请运行: pip install pyspark"
                )
        
        return self._spark
    
    def process_raw_logs(
        self,
        input_path: str,
        output_path: str,
        semantic_id_mapping_path: Optional[str] = None,
        input_format: str = "json",
        output_format: str = "parquet",
        min_sequence_length: int = 5,
        max_sequence_length: int = 500,
    ) -> Dict[str, Any]:
        """
        处理原始日志
        
        完整的处理流程：
        1. 读取原始日志
        2. 关联语义 ID 映射
        3. 添加时间特征
        4. 按用户分组构建序列
        5. 过滤并保存
        
        Args:
            input_path: 原始日志路径（支持本地/HDFS/S3）
            output_path: 输出路径
            semantic_id_mapping_path: 物品语义 ID 映射表路径
            input_format: 输入格式 (json/parquet/csv)
            output_format: 输出格式 (parquet/tfrecord)
            min_sequence_length: 最小序列长度
            max_sequence_length: 最大序列长度
            
        Returns:
            处理统计信息
        """
        from pyspark.sql import functions as F
        
        logger.info(f"开始处理原始日志: {input_path}")
        start_time = datetime.now()
        
        # 1. 读取原始日志
        df = self._read_data(input_path, input_format)
        initial_count = df.count()
        logger.info(f"读取原始日志: {initial_count} 条")
        
        # 2. 关联语义 ID 映射（如果提供）
        if semantic_id_mapping_path and os.path.exists(semantic_id_mapping_path):
            semantic_df = self._read_data(semantic_id_mapping_path, "parquet")
            df = df.join(semantic_df, on="item_id", how="left")
            logger.info("已关联语义 ID 映射")
        
        # 3. 添加时间特征
        df = self._add_time_features(df)
        
        # 4. 按用户分组构建序列
        sequences_df = self._build_user_sequences(df, max_sequence_length)
        
        # 5. 过滤太短的序列
        sequences_df = sequences_df.filter(F.col("seq_length") >= min_sequence_length)
        
        # 6. 保存结果
        self._write_data(sequences_df, output_path, output_format)
        
        # 统计信息
        final_count = sequences_df.count()
        end_time = datetime.now()
        
        stats = {
            "input_path": input_path,
            "output_path": output_path,
            "input_events": initial_count,
            "output_sequences": final_count,
            "processing_time": str(end_time - start_time),
            "timestamp": end_time.isoformat(),
        }
        
        logger.info(f"处理完成: {final_count} 个用户序列")
        return stats
    
    def _read_data(self, path: str, format: str):
        """读取数据"""
        if format == "json":
            return self.spark.read.json(path)
        elif format == "parquet":
            return self.spark.read.parquet(path)
        elif format == "csv":
            return self.spark.read.csv(path, header=True, inferSchema=True)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def _write_data(self, df, path: str, format: str) -> None:
        """写入数据"""
        if format == "parquet":
            df.write.parquet(path, mode="overwrite")
        elif format == "json":
            df.write.json(path, mode="overwrite")
        else:
            raise ValueError(f"不支持的输出格式: {format}")
    
    def _add_time_features(self, df):
        """
        添加时间特征
        
        从时间戳提取：
        - hour: 小时
        - day_of_week: 星期几
        - time_bucket: 时间段
        - is_weekend: 是否周末
        """
        from pyspark.sql import functions as F
        
        return df \
            .withColumn(
                "event_time",
                F.from_unixtime(F.col("timestamp"))
            ) \
            .withColumn(
                "hour",
                F.hour("event_time")
            ) \
            .withColumn(
                "day_of_week",
                F.dayofweek("event_time")
            ) \
            .withColumn(
                "time_bucket",
                F.when(F.col("hour") < 6, "night")
                 .when(F.col("hour") < 12, "morning")
                 .when(F.col("hour") < 18, "afternoon")
                 .otherwise("evening")
            ) \
            .withColumn(
                "is_weekend",
                F.when(F.col("day_of_week").isin([1, 7]), True).otherwise(False)
            )
    
    def _build_user_sequences(self, df, max_length: int = 500):
        """
        构建用户序列
        
        将每个用户的事件按时间排序并收集为数组。
        """
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window
        
        # 按用户和时间排序
        window = Window.partitionBy("user_id").orderBy("timestamp")
        
        # 添加序列位置
        df = df.withColumn("seq_position", F.row_number().over(window))
        
        # 只保留最近的 max_length 个事件
        df = df.filter(F.col("seq_position") <= max_length)
        
        # 构建事件结构
        event_struct = F.struct(
            F.col("item_id"),
            F.col("action"),
            F.col("timestamp"),
            F.col("time_bucket"),
            F.coalesce(F.col("semantic_l1"), F.lit(0)).alias("semantic_l1"),
            F.coalesce(F.col("semantic_l2"), F.lit(0)).alias("semantic_l2"),
            F.coalesce(F.col("semantic_l3"), F.lit(0)).alias("semantic_l3"),
            F.coalesce(F.col("device"), F.lit("unknown")).alias("device"),
        )
        
        # 按用户分组并收集事件
        sequences_df = df.groupBy("user_id").agg(
            F.sort_array(
                F.collect_list(
                    F.struct(F.col("timestamp"), event_struct.alias("event"))
                ),
                asc=True
            ).alias("sorted_events"),
            F.count("*").alias("seq_length"),
            F.min("timestamp").alias("first_event_time"),
            F.max("timestamp").alias("last_event_time"),
        )
        
        # 提取排序后的事件
        sequences_df = sequences_df.withColumn(
            "events",
            F.transform(F.col("sorted_events"), lambda x: x["event"])
        ).drop("sorted_events")
        
        return sequences_df
    
    def build_vocabulary(
        self,
        input_path: str,
        output_path: str,
        min_freq: int = 5,
        max_vocab_size: int = 500000,
    ) -> Dict[str, Any]:
        """
        构建词表
        
        从处理后的序列数据中统计 Token 频率并构建词表。
        
        Args:
            input_path: 序列数据路径
            output_path: 词表输出路径
            min_freq: 最小频率阈值
            max_vocab_size: 词表最大容量
            
        Returns:
            词表统计信息
        """
        from pyspark.sql import functions as F
        
        logger.info(f"开始构建词表: {input_path}")
        
        df = self.spark.read.parquet(input_path)
        
        # 展开事件
        events_df = df.select(F.explode("events").alias("event"))
        
        # 提取各类 Token
        # 行为 Token
        action_tokens = events_df.select(
            F.concat(F.lit("ACTION_"), F.col("event.action")).alias("token")
        )
        
        # 物品 Token
        item_tokens = events_df.select(
            F.concat(
                F.lit("ITEM_"),
                F.col("event.semantic_l1").cast("string"),
                F.lit("_"),
                F.col("event.semantic_l2").cast("string"),
                F.lit("_"),
                F.col("event.semantic_l3").cast("string"),
            ).alias("token")
        )
        
        # 时间 Token
        time_tokens = events_df.select(
            F.concat(F.lit("TIME_"), F.col("event.time_bucket")).alias("token")
        )
        
        # 设备 Token
        device_tokens = events_df.select(
            F.concat(F.lit("DEVICE_"), F.col("event.device")).alias("token")
        )
        
        # 合并所有 Token
        all_tokens = action_tokens.union(item_tokens).union(time_tokens).union(device_tokens)
        
        # 统计频率
        vocab_df = all_tokens \
            .groupBy("token") \
            .count() \
            .filter(F.col("count") >= min_freq) \
            .orderBy(F.desc("count")) \
            .limit(max_vocab_size)
        
        # 添加 ID
        vocab_df = vocab_df.withColumn(
            "token_id",
            F.row_number().over(Window.orderBy(F.desc("count"))) + 4  # 预留特殊 Token ID
        )
        
        # 保存
        vocab_df.write.csv(output_path, header=True, mode="overwrite")
        
        # 统计信息
        total_tokens = vocab_df.count()
        
        stats = {
            "output_path": output_path,
            "total_tokens": total_tokens,
            "min_freq": min_freq,
        }
        
        logger.info(f"词表构建完成: {total_tokens} 个 Token")
        return stats
    
    def generate_training_samples(
        self,
        input_path: str,
        output_path: str,
        window_size: int = 100,
        stride: int = 50,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Dict[str, Any]:
        """
        生成训练样本
        
        从用户序列中按滑动窗口生成训练样本，并划分训练/验证/测试集。
        
        Args:
            input_path: 序列数据路径
            output_path: 输出路径
            window_size: 滑动窗口大小
            stride: 滑动步长
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            生成统计信息
        """
        from pyspark.sql import functions as F
        from pyspark.sql.types import ArrayType, StructType, StructField, StringType, LongType
        
        logger.info(f"开始生成训练样本: {input_path}")
        
        df = self.spark.read.parquet(input_path)
        
        # 定义滑动窗口 UDF
        @F.udf(returnType=ArrayType(
            StructType([
                StructField("input_events", ArrayType(
                    StructType([
                        StructField("item_id", StringType()),
                        StructField("action", StringType()),
                        StructField("timestamp", LongType()),
                    ])
                )),
                StructField("target_item_id", StringType()),
                StructField("target_action", StringType()),
            ])
        ))
        def create_samples(events):
            if events is None or len(events) < 2:
                return []
            
            samples = []
            for i in range(0, len(events) - 1, stride):
                end = min(i + window_size, len(events) - 1)
                
                input_events = [
                    {
                        "item_id": e["item_id"],
                        "action": e["action"],
                        "timestamp": e["timestamp"],
                    }
                    for e in events[i:end]
                ]
                
                target = events[end]
                
                samples.append({
                    "input_events": input_events,
                    "target_item_id": target["item_id"],
                    "target_action": target["action"],
                })
            
            return samples
        
        # 生成样本
        samples_df = df.withColumn(
            "samples",
            create_samples(F.col("events"))
        ).select(
            F.col("user_id"),
            F.explode("samples").alias("sample")
        ).select(
            F.col("user_id"),
            F.col("sample.input_events").alias("input_events"),
            F.col("sample.target_item_id").alias("target_item_id"),
            F.col("sample.target_action").alias("target_action"),
        )
        
        # 添加随机数用于划分
        samples_df = samples_df.withColumn("rand", F.rand(seed=42))
        
        # 划分数据集
        train_df = samples_df.filter(F.col("rand") < train_ratio)
        val_df = samples_df.filter(
            (F.col("rand") >= train_ratio) & 
            (F.col("rand") < train_ratio + val_ratio)
        )
        test_df = samples_df.filter(F.col("rand") >= train_ratio + val_ratio)
        
        # 保存
        train_df.drop("rand").write.parquet(f"{output_path}/train", mode="overwrite")
        val_df.drop("rand").write.parquet(f"{output_path}/val", mode="overwrite")
        test_df.drop("rand").write.parquet(f"{output_path}/test", mode="overwrite")
        
        # 统计信息
        stats = {
            "output_path": output_path,
            "train_samples": train_df.count(),
            "val_samples": val_df.count(),
            "test_samples": test_df.count(),
            "window_size": window_size,
            "stride": stride,
        }
        
        logger.info(f"训练样本生成完成: {stats['train_samples']} 训练 / {stats['val_samples']} 验证 / {stats['test_samples']} 测试")
        return stats
    
    def validate_data(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        验证数据质量
        
        检查数据中的问题：
        - 缺失值
        - 异常值
        - 数据分布
        
        Args:
            input_path: 数据路径
            output_path: 报告输出路径
            
        Returns:
            验证报告
        """
        from pyspark.sql import functions as F
        
        logger.info(f"开始数据验证: {input_path}")
        
        df = self.spark.read.parquet(input_path)
        
        # 基本统计
        total_users = df.count()
        
        # 序列长度分布
        length_stats = df.select(
            F.min("seq_length").alias("min_length"),
            F.max("seq_length").alias("max_length"),
            F.avg("seq_length").alias("avg_length"),
            F.stddev("seq_length").alias("std_length"),
        ).collect()[0]
        
        # 时间范围
        events_df = df.select(F.explode("events").alias("event"))
        time_stats = events_df.select(
            F.min("event.timestamp").alias("min_time"),
            F.max("event.timestamp").alias("max_time"),
        ).collect()[0]
        
        # 行为分布
        action_dist = events_df.groupBy("event.action").count().collect()
        action_distribution = {row["action"]: row["count"] for row in action_dist}
        
        # 物品覆盖率
        unique_items = events_df.select("event.item_id").distinct().count()
        total_events = events_df.count()
        
        report = {
            "input_path": input_path,
            "total_users": total_users,
            "total_events": total_events,
            "unique_items": unique_items,
            "sequence_length": {
                "min": length_stats["min_length"],
                "max": length_stats["max_length"],
                "avg": float(length_stats["avg_length"]) if length_stats["avg_length"] else 0,
                "std": float(length_stats["std_length"]) if length_stats["std_length"] else 0,
            },
            "time_range": {
                "min": datetime.fromtimestamp(time_stats["min_time"]).isoformat() if time_stats["min_time"] else None,
                "max": datetime.fromtimestamp(time_stats["max_time"]).isoformat() if time_stats["max_time"] else None,
            },
            "action_distribution": action_distribution,
            "avg_events_per_user": total_events / total_users if total_users > 0 else 0,
        }
        
        # 保存报告
        if output_path:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"验证报告已保存到: {output_path}")
        
        logger.info(f"数据验证完成: {total_users} 用户, {total_events} 事件")
        return report
    
    def compute_item_statistics(
        self,
        input_path: str,
        output_path: str,
    ) -> Dict[str, Any]:
        """
        计算物品统计信息
        
        统计每个物品的：
        - 出现次数
        - 唯一用户数
        - 行为分布
        
        Args:
            input_path: 序列数据路径
            output_path: 输出路径
            
        Returns:
            统计信息
        """
        from pyspark.sql import functions as F
        
        logger.info(f"开始计算物品统计: {input_path}")
        
        df = self.spark.read.parquet(input_path)
        
        # 展开事件并关联用户
        events_df = df.select(
            F.col("user_id"),
            F.explode("events").alias("event")
        ).select(
            F.col("user_id"),
            F.col("event.item_id").alias("item_id"),
            F.col("event.action").alias("action"),
        )
        
        # 计算物品统计
        item_stats = events_df.groupBy("item_id").agg(
            F.count("*").alias("total_events"),
            F.countDistinct("user_id").alias("unique_users"),
            F.sum(F.when(F.col("action") == "click", 1).otherwise(0)).alias("click_count"),
            F.sum(F.when(F.col("action") == "buy", 1).otherwise(0)).alias("buy_count"),
            F.sum(F.when(F.col("action") == "view", 1).otherwise(0)).alias("view_count"),
        )
        
        # 计算转化率
        item_stats = item_stats.withColumn(
            "click_rate",
            F.when(F.col("view_count") > 0, F.col("click_count") / F.col("view_count")).otherwise(0)
        ).withColumn(
            "conversion_rate",
            F.when(F.col("click_count") > 0, F.col("buy_count") / F.col("click_count")).otherwise(0)
        )
        
        # 保存
        item_stats.write.parquet(output_path, mode="overwrite")
        
        total_items = item_stats.count()
        
        stats = {
            "output_path": output_path,
            "total_items": total_items,
        }
        
        logger.info(f"物品统计完成: {total_items} 个物品")
        return stats
    
    def stop(self) -> None:
        """停止 SparkSession"""
        if self._spark is not None:
            self._spark.stop()
            self._spark = None
            logger.info("SparkSession 已停止")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop()
    
    def __repr__(self) -> str:
        return f"SparkFeaturePipeline(app_name='{self.app_name}')"


# 导入 Window 以便在 UDF 外部使用
try:
    from pyspark.sql.window import Window
except ImportError:
    Window = None

