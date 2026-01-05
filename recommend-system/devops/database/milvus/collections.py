"""
Milvus 向量数据库集合管理

本模块提供 Milvus 集合的创建、配置和管理功能。

项目：生成式推荐系统
模块：数据库管理
版本：1.0.0
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException,
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """一致性级别"""
    STRONG = "Strong"
    BOUNDED = "Bounded"
    SESSION = "Session"
    EVENTUALLY = "Eventually"


@dataclass
class CollectionConfig:
    """集合配置"""
    name: str
    description: str = ""
    dim: int = 256  # 向量维度
    primary_field: str = "id"
    vector_field: str = "embedding"
    consistency_level: ConsistencyLevel = ConsistencyLevel.BOUNDED
    shards_num: int = 2
    partition_key_field: Optional[str] = None
    enable_dynamic_field: bool = True
    extra_fields: List[Dict[str, Any]] = field(default_factory=list)


class MilvusManager:
    """
    Milvus 集合管理器
    
    提供 Milvus 向量数据库的连接和集合管理功能。
    
    使用示例:
        manager = MilvusManager()
        manager.connect()
        
        # 创建物品嵌入集合
        collection = manager.create_item_embeddings_collection(dim=256)
        
        # 获取集合统计
        stats = manager.get_stats("item_embeddings")
        
        manager.disconnect()
    """
    
    # 预定义集合配置
    PREDEFINED_COLLECTIONS = {
        "item_embeddings": CollectionConfig(
            name="item_embeddings",
            description="Item embeddings for similarity search",
            dim=256,
            primary_field="item_id",
            vector_field="embedding",
            extra_fields=[
                {"name": "item_type", "dtype": DataType.VARCHAR, "max_length": 32},
                {"name": "category", "dtype": DataType.VARCHAR, "max_length": 100},
                {"name": "created_at", "dtype": DataType.INT64},
                {"name": "updated_at", "dtype": DataType.INT64},
            ]
        ),
        "user_embeddings": CollectionConfig(
            name="user_embeddings",
            description="User embeddings for personalization",
            dim=256,
            primary_field="user_id",
            vector_field="embedding",
            extra_fields=[
                {"name": "segment", "dtype": DataType.VARCHAR, "max_length": 50},
                {"name": "updated_at", "dtype": DataType.INT64},
            ]
        ),
        "semantic_id_codebook": CollectionConfig(
            name="semantic_id_codebook",
            description="Semantic ID codebook vectors",
            dim=256,
            primary_field="code_id",
            vector_field="code_vector",
            extra_fields=[
                {"name": "level", "dtype": DataType.INT8},
                {"name": "description", "dtype": DataType.VARCHAR, "max_length": 500},
                {"name": "usage_count", "dtype": DataType.INT64},
            ]
        ),
    }
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        alias: str = "default",
        timeout: float = 30.0,
    ):
        """
        初始化 Milvus 管理器
        
        Args:
            host: Milvus 服务器地址
            port: Milvus 服务器端口
            user: 用户名（可选）
            password: 密码（可选）
            alias: 连接别名
            timeout: 连接超时时间（秒）
        """
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        self.user = user or os.getenv("MILVUS_USER", "")
        self.password = password or os.getenv("MILVUS_PASSWORD", "")
        self.alias = alias
        self.timeout = timeout
        self._connected = False
        self._collections: Dict[str, Collection] = {}
    
    def connect(self, retry_times: int = 3, retry_interval: float = 2.0) -> bool:
        """
        连接到 Milvus
        
        Args:
            retry_times: 重试次数
            retry_interval: 重试间隔（秒）
            
        Returns:
            是否连接成功
        """
        if self._connected:
            logger.info(f"Already connected to Milvus at {self.host}:{self.port}")
            return True
        
        for attempt in range(retry_times):
            try:
                connect_params = {
                    "alias": self.alias,
                    "host": self.host,
                    "port": self.port,
                    "timeout": self.timeout,
                }
                
                if self.user and self.password:
                    connect_params["user"] = self.user
                    connect_params["password"] = self.password
                
                connections.connect(**connect_params)
                self._connected = True
                logger.info(f"Connected to Milvus at {self.host}:{self.port}")
                return True
                
            except MilvusException as e:
                logger.warning(
                    f"Connection attempt {attempt + 1}/{retry_times} failed: {e}"
                )
                if attempt < retry_times - 1:
                    time.sleep(retry_interval)
        
        logger.error(f"Failed to connect to Milvus after {retry_times} attempts")
        return False
    
    def disconnect(self) -> None:
        """断开 Milvus 连接"""
        if self._connected:
            connections.disconnect(self.alias)
            self._connected = False
            self._collections.clear()
            logger.info("Disconnected from Milvus")
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected
    
    def _ensure_connected(self) -> None:
        """确保已连接"""
        if not self._connected:
            if not self.connect():
                raise RuntimeError("Not connected to Milvus")
    
    def create_collection(
        self,
        config: CollectionConfig,
        drop_if_exists: bool = False,
    ) -> Collection:
        """
        创建集合
        
        Args:
            config: 集合配置
            drop_if_exists: 如果已存在是否删除
            
        Returns:
            创建的集合对象
        """
        self._ensure_connected()
        
        # 检查是否已存在
        if utility.has_collection(config.name):
            if drop_if_exists:
                logger.info(f"Dropping existing collection: {config.name}")
                utility.drop_collection(config.name)
            else:
                logger.info(f"Collection {config.name} already exists")
                return Collection(config.name)
        
        # 定义字段
        fields = [
            FieldSchema(
                name=config.primary_field,
                dtype=DataType.VARCHAR,
                max_length=64,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name=config.vector_field,
                dtype=DataType.FLOAT_VECTOR,
                dim=config.dim,
            ),
        ]
        
        # 添加额外字段
        for field_config in config.extra_fields:
            field_params = {
                "name": field_config["name"],
                "dtype": field_config["dtype"],
            }
            if "max_length" in field_config:
                field_params["max_length"] = field_config["max_length"]
            if "is_partition_key" in field_config:
                field_params["is_partition_key"] = field_config["is_partition_key"]
            
            fields.append(FieldSchema(**field_params))
        
        # 创建 Schema
        schema = CollectionSchema(
            fields=fields,
            description=config.description,
            enable_dynamic_field=config.enable_dynamic_field,
        )
        
        # 创建集合
        collection = Collection(
            name=config.name,
            schema=schema,
            consistency_level=config.consistency_level.value,
            shards_num=config.shards_num,
        )
        
        self._collections[config.name] = collection
        logger.info(f"Created collection: {config.name}")
        
        return collection
    
    def create_item_embeddings_collection(
        self,
        dim: int = 256,
        drop_if_exists: bool = False,
    ) -> Collection:
        """
        创建物品嵌入集合
        
        Args:
            dim: 向量维度
            drop_if_exists: 是否删除已存在的集合
            
        Returns:
            集合对象
        """
        config = CollectionConfig(
            name="item_embeddings",
            description="Item embeddings for similarity search in recommendation system",
            dim=dim,
            primary_field="item_id",
            vector_field="embedding",
            consistency_level=ConsistencyLevel.BOUNDED,
            shards_num=2,
            extra_fields=[
                {"name": "item_type", "dtype": DataType.VARCHAR, "max_length": 32},
                {"name": "category", "dtype": DataType.VARCHAR, "max_length": 100},
                {"name": "semantic_id_l1", "dtype": DataType.INT32},
                {"name": "semantic_id_l2", "dtype": DataType.INT32},
                {"name": "semantic_id_l3", "dtype": DataType.INT32},
                {"name": "created_at", "dtype": DataType.INT64},
                {"name": "updated_at", "dtype": DataType.INT64},
            ]
        )
        
        return self.create_collection(config, drop_if_exists)
    
    def create_user_embeddings_collection(
        self,
        dim: int = 256,
        drop_if_exists: bool = False,
    ) -> Collection:
        """
        创建用户嵌入集合
        
        Args:
            dim: 向量维度
            drop_if_exists: 是否删除已存在的集合
            
        Returns:
            集合对象
        """
        config = CollectionConfig(
            name="user_embeddings",
            description="User embeddings for personalized recommendations",
            dim=dim,
            primary_field="user_id",
            vector_field="embedding",
            consistency_level=ConsistencyLevel.BOUNDED,
            extra_fields=[
                {"name": "segment", "dtype": DataType.VARCHAR, "max_length": 50},
                {"name": "behavior_count", "dtype": DataType.INT64},
                {"name": "is_cold_start", "dtype": DataType.BOOL},
                {"name": "updated_at", "dtype": DataType.INT64},
            ]
        )
        
        return self.create_collection(config, drop_if_exists)
    
    def create_query_cache_collection(
        self,
        dim: int = 256,
        drop_if_exists: bool = False,
    ) -> Collection:
        """
        创建查询缓存集合（用于存储查询向量的预计算结果）
        
        Args:
            dim: 向量维度
            drop_if_exists: 是否删除已存在的集合
            
        Returns:
            集合对象
        """
        config = CollectionConfig(
            name="query_cache",
            description="Pre-computed query results cache",
            dim=dim,
            primary_field="query_id",
            vector_field="query_vector",
            consistency_level=ConsistencyLevel.EVENTUALLY,
            extra_fields=[
                {"name": "result_ids", "dtype": DataType.VARCHAR, "max_length": 2000},
                {"name": "result_scores", "dtype": DataType.VARCHAR, "max_length": 1000},
                {"name": "ttl", "dtype": DataType.INT64},
                {"name": "created_at", "dtype": DataType.INT64},
            ]
        )
        
        return self.create_collection(config, drop_if_exists)
    
    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """
        获取集合对象
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合对象，不存在则返回 None
        """
        self._ensure_connected()
        
        if collection_name in self._collections:
            return self._collections[collection_name]
        
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            self._collections[collection_name] = collection
            return collection
        
        return None
    
    def drop_collection(self, collection_name: str) -> bool:
        """
        删除集合
        
        Args:
            collection_name: 集合名称
            
        Returns:
            是否删除成功
        """
        self._ensure_connected()
        
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            self._collections.pop(collection_name, None)
            logger.info(f"Dropped collection: {collection_name}")
            return True
        
        logger.warning(f"Collection not found: {collection_name}")
        return False
    
    def list_collections(self) -> List[str]:
        """
        列出所有集合
        
        Returns:
            集合名称列表
        """
        self._ensure_connected()
        return utility.list_collections()
    
    def load_collection(
        self,
        collection_name: str,
        replica_number: int = 1,
    ) -> bool:
        """
        加载集合到内存
        
        Args:
            collection_name: 集合名称
            replica_number: 副本数量
            
        Returns:
            是否加载成功
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            logger.error(f"Collection not found: {collection_name}")
            return False
        
        try:
            collection.load(replica_number=replica_number)
            logger.info(f"Loaded collection: {collection_name}")
            return True
        except MilvusException as e:
            logger.error(f"Failed to load collection {collection_name}: {e}")
            return False
    
    def release_collection(self, collection_name: str) -> bool:
        """
        释放集合内存
        
        Args:
            collection_name: 集合名称
            
        Returns:
            是否释放成功
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return False
        
        try:
            collection.release()
            logger.info(f"Released collection: {collection_name}")
            return True
        except MilvusException as e:
            logger.error(f"Failed to release collection {collection_name}: {e}")
            return False
    
    def get_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            统计信息字典
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return {"error": "Collection not found"}
        
        try:
            # 刷新以获取最新数据
            collection.flush()
            
            return {
                "name": collection_name,
                "description": collection.description,
                "num_entities": collection.num_entities,
                "schema": self._schema_to_dict(collection.schema),
                "indexes": [str(idx) for idx in collection.indexes],
                "is_loaded": utility.load_state(collection_name).name,
                "shards_num": collection.num_shards if hasattr(collection, 'num_shards') else None,
            }
        except MilvusException as e:
            return {"error": str(e)}
    
    def _schema_to_dict(self, schema: CollectionSchema) -> Dict[str, Any]:
        """将 Schema 转换为字典"""
        fields = []
        for field in schema.fields:
            field_info = {
                "name": field.name,
                "dtype": field.dtype.name,
                "is_primary": field.is_primary,
            }
            if hasattr(field, "max_length") and field.max_length:
                field_info["max_length"] = field.max_length
            if hasattr(field, "dim") and field.dim:
                field_info["dim"] = field.dim
            fields.append(field_info)
        
        return {
            "description": schema.description,
            "fields": fields,
            "enable_dynamic_field": schema.enable_dynamic_field,
        }
    
    def get_loading_progress(self, collection_name: str) -> Dict[str, Any]:
        """
        获取集合加载进度
        
        Args:
            collection_name: 集合名称
            
        Returns:
            加载进度信息
        """
        self._ensure_connected()
        
        try:
            progress = utility.loading_progress(collection_name)
            return {
                "collection": collection_name,
                "progress": progress,
            }
        except MilvusException as e:
            return {"error": str(e)}


def init_all_collections(
    manager: Optional[MilvusManager] = None,
    dim: int = 256,
) -> Dict[str, Collection]:
    """
    初始化所有推荐系统需要的集合
    
    Args:
        manager: Milvus 管理器实例
        dim: 向量维度
        
    Returns:
        集合名称到集合对象的映射
    """
    if manager is None:
        manager = MilvusManager()
    
    if not manager.is_connected():
        manager.connect()
    
    collections = {}
    
    try:
        # 创建物品嵌入集合
        collections["item_embeddings"] = manager.create_item_embeddings_collection(dim=dim)
        
        # 创建用户嵌入集合
        collections["user_embeddings"] = manager.create_user_embeddings_collection(dim=dim)
        
        # 创建查询缓存集合
        collections["query_cache"] = manager.create_query_cache_collection(dim=dim)
        
        logger.info(f"Initialized {len(collections)} collections")
        
        return collections
        
    except Exception as e:
        logger.error(f"Failed to initialize collections: {e}")
        raise


if __name__ == "__main__":
    # 测试代码
    manager = MilvusManager()
    
    try:
        manager.connect()
        
        # 初始化所有集合
        collections = init_all_collections(manager, dim=256)
        
        # 打印统计信息
        print("\n=== Collection Stats ===")
        for name in collections:
            stats = manager.get_stats(name)
            print(f"\n{name}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
    finally:
        manager.disconnect()

