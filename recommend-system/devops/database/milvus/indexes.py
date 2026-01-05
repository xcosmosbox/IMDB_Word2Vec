"""
Milvus 向量索引管理

本模块提供 Milvus 索引的创建、管理和优化功能。

项目：生成式推荐系统
模块：数据库管理
版本：1.0.0
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum

from pymilvus import (
    Collection,
    utility,
    MilvusException,
)

from .collections import MilvusManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexType(Enum):
    """索引类型枚举"""
    # Flat (精确搜索)
    FLAT = "FLAT"
    
    # IVF 系列 (适合中等规模数据)
    IVF_FLAT = "IVF_FLAT"
    IVF_SQ8 = "IVF_SQ8"
    IVF_PQ = "IVF_PQ"
    
    # HNSW (高召回率，推荐使用)
    HNSW = "HNSW"
    
    # Annoy
    ANNOY = "ANNOY"
    
    # DiskANN (适合大规模数据)
    DISKANN = "DISKANN"
    
    # GPU 索引
    GPU_IVF_FLAT = "GPU_IVF_FLAT"
    GPU_IVF_PQ = "GPU_IVF_PQ"
    
    # SCANN
    SCANN = "SCANN"


class MetricType(Enum):
    """距离度量类型"""
    # 内积 (推荐用于归一化向量)
    IP = "IP"
    
    # L2 欧氏距离
    L2 = "L2"
    
    # 余弦相似度
    COSINE = "COSINE"


@dataclass
class IndexConfig:
    """索引配置"""
    index_type: IndexType
    metric_type: MetricType
    params: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为 Milvus 索引参数格式"""
        return {
            "index_type": self.index_type.value,
            "metric_type": self.metric_type.value,
            "params": self.params,
        }


class IndexManager:
    """
    Milvus 索引管理器
    
    提供向量索引的创建、删除和优化功能。
    
    使用示例:
        manager = MilvusManager()
        manager.connect()
        
        index_manager = IndexManager(manager)
        
        # 创建 HNSW 索引
        index_manager.create_index(
            "item_embeddings",
            "embedding",
            IndexType.HNSW,
            MetricType.IP
        )
        
        manager.disconnect()
    """
    
    # 预定义索引配置
    PRESET_CONFIGS = {
        # 小规模数据（< 100万）：精确搜索
        "small_exact": IndexConfig(
            index_type=IndexType.FLAT,
            metric_type=MetricType.IP,
            params={},
        ),
        
        # 中等规模数据（100万 - 1000万）：IVF_FLAT
        "medium_ivf": IndexConfig(
            index_type=IndexType.IVF_FLAT,
            metric_type=MetricType.IP,
            params={"nlist": 1024},
        ),
        
        # 大规模数据（1000万 - 1亿）：HNSW
        "large_hnsw": IndexConfig(
            index_type=IndexType.HNSW,
            metric_type=MetricType.IP,
            params={"M": 16, "efConstruction": 256},
        ),
        
        # 超大规模数据（> 1亿）：DiskANN
        "xlarge_diskann": IndexConfig(
            index_type=IndexType.DISKANN,
            metric_type=MetricType.IP,
            params={},
        ),
        
        # 高召回率场景：HNSW with high parameters
        "high_recall": IndexConfig(
            index_type=IndexType.HNSW,
            metric_type=MetricType.COSINE,
            params={"M": 32, "efConstruction": 512},
        ),
        
        # 低延迟场景：IVF_SQ8
        "low_latency": IndexConfig(
            index_type=IndexType.IVF_SQ8,
            metric_type=MetricType.IP,
            params={"nlist": 2048},
        ),
        
        # GPU 加速场景
        "gpu_accelerated": IndexConfig(
            index_type=IndexType.GPU_IVF_FLAT,
            metric_type=MetricType.IP,
            params={"nlist": 1024},
        ),
    }
    
    def __init__(self, milvus_manager: MilvusManager):
        """
        初始化索引管理器
        
        Args:
            milvus_manager: Milvus 管理器实例
        """
        self.manager = milvus_manager
    
    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_type: IndexType = IndexType.HNSW,
        metric_type: MetricType = MetricType.IP,
        params: Optional[Dict[str, Any]] = None,
        preset: Optional[str] = None,
    ) -> bool:
        """
        创建向量索引
        
        Args:
            collection_name: 集合名称
            field_name: 向量字段名称
            index_type: 索引类型
            metric_type: 距离度量类型
            params: 索引参数
            preset: 预设配置名称
            
        Returns:
            是否创建成功
        """
        collection = self.manager.get_collection(collection_name)
        if collection is None:
            logger.error(f"Collection not found: {collection_name}")
            return False
        
        # 使用预设配置
        if preset and preset in self.PRESET_CONFIGS:
            config = self.PRESET_CONFIGS[preset]
            index_params = config.to_dict()
            logger.info(f"Using preset config: {preset}")
        else:
            # 自定义配置
            default_params = self._get_default_params(index_type)
            if params:
                default_params.update(params)
            
            index_params = {
                "index_type": index_type.value,
                "metric_type": metric_type.value,
                "params": default_params,
            }
        
        try:
            # 检查是否已存在索引
            if self.has_index(collection_name, field_name):
                logger.info(f"Dropping existing index on {collection_name}.{field_name}")
                collection.drop_index()
            
            # 创建索引
            collection.create_index(
                field_name=field_name,
                index_params=index_params,
            )
            
            logger.info(
                f"Created index on {collection_name}.{field_name}: "
                f"type={index_params['index_type']}, metric={index_params['metric_type']}"
            )
            return True
            
        except MilvusException as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def _get_default_params(self, index_type: IndexType) -> Dict[str, Any]:
        """获取索引类型的默认参数"""
        defaults = {
            IndexType.FLAT: {},
            IndexType.IVF_FLAT: {"nlist": 1024},
            IndexType.IVF_SQ8: {"nlist": 1024},
            IndexType.IVF_PQ: {"nlist": 1024, "m": 8, "nbits": 8},
            IndexType.HNSW: {"M": 16, "efConstruction": 256},
            IndexType.ANNOY: {"n_trees": 8},
            IndexType.DISKANN: {},
            IndexType.GPU_IVF_FLAT: {"nlist": 1024},
            IndexType.GPU_IVF_PQ: {"nlist": 1024, "m": 8, "nbits": 8},
            IndexType.SCANN: {"nlist": 1024},
        }
        return defaults.get(index_type, {})
    
    def drop_index(self, collection_name: str, field_name: str = "embedding") -> bool:
        """
        删除索引
        
        Args:
            collection_name: 集合名称
            field_name: 字段名称
            
        Returns:
            是否删除成功
        """
        collection = self.manager.get_collection(collection_name)
        if collection is None:
            logger.error(f"Collection not found: {collection_name}")
            return False
        
        try:
            collection.drop_index()
            logger.info(f"Dropped index on {collection_name}")
            return True
        except MilvusException as e:
            logger.error(f"Failed to drop index: {e}")
            return False
    
    def has_index(self, collection_name: str, field_name: str = "embedding") -> bool:
        """
        检查是否存在索引
        
        Args:
            collection_name: 集合名称
            field_name: 字段名称
            
        Returns:
            是否存在索引
        """
        collection = self.manager.get_collection(collection_name)
        if collection is None:
            return False
        
        return len(collection.indexes) > 0
    
    def get_index_info(self, collection_name: str) -> Dict[str, Any]:
        """
        获取索引信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            索引信息字典
        """
        collection = self.manager.get_collection(collection_name)
        if collection is None:
            return {"error": "Collection not found"}
        
        indexes = []
        for idx in collection.indexes:
            indexes.append({
                "field_name": idx.field_name,
                "index_type": idx.params.get("index_type"),
                "metric_type": idx.params.get("metric_type"),
                "params": idx.params.get("params", {}),
            })
        
        return {
            "collection": collection_name,
            "indexes": indexes,
        }
    
    def create_all_indexes(
        self,
        dim: int = 256,
        preset: str = "large_hnsw",
    ) -> Dict[str, bool]:
        """
        为所有推荐系统集合创建索引
        
        Args:
            dim: 向量维度
            preset: 预设配置名称
            
        Returns:
            集合名称到创建结果的映射
        """
        collections_config = {
            "item_embeddings": "embedding",
            "user_embeddings": "embedding",
            "query_cache": "query_vector",
        }
        
        results = {}
        
        for collection_name, field_name in collections_config.items():
            if self.manager.get_collection(collection_name):
                results[collection_name] = self.create_index(
                    collection_name,
                    field_name,
                    preset=preset,
                )
            else:
                results[collection_name] = False
                logger.warning(f"Collection not found: {collection_name}")
        
        return results
    
    def rebuild_index(
        self,
        collection_name: str,
        field_name: str = "embedding",
        new_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        重建索引
        
        Args:
            collection_name: 集合名称
            field_name: 字段名称
            new_params: 新的索引参数
            
        Returns:
            是否重建成功
        """
        # 获取当前索引信息
        info = self.get_index_info(collection_name)
        if "error" in info:
            logger.error(f"Cannot get index info: {info['error']}")
            return False
        
        current_indexes = info.get("indexes", [])
        if not current_indexes:
            logger.warning(f"No index found on {collection_name}")
            return False
        
        current_index = current_indexes[0]
        
        # 合并参数
        index_type = IndexType(current_index["index_type"])
        metric_type = MetricType(current_index["metric_type"])
        params = current_index.get("params", {})
        
        if new_params:
            params.update(new_params)
        
        # 释放集合
        self.manager.release_collection(collection_name)
        
        # 删除旧索引
        self.drop_index(collection_name, field_name)
        
        # 创建新索引
        success = self.create_index(
            collection_name,
            field_name,
            index_type,
            metric_type,
            params,
        )
        
        # 重新加载集合
        if success:
            self.manager.load_collection(collection_name)
        
        return success
    
    def get_index_building_progress(self, collection_name: str) -> Dict[str, Any]:
        """
        获取索引构建进度
        
        Args:
            collection_name: 集合名称
            
        Returns:
            构建进度信息
        """
        try:
            progress = utility.index_building_progress(collection_name)
            return {
                "collection": collection_name,
                "progress": progress,
            }
        except MilvusException as e:
            return {"error": str(e)}
    
    def recommend_index_config(
        self,
        collection_name: str,
        expected_data_size: int,
        query_latency_requirement_ms: float = 50.0,
        recall_requirement: float = 0.95,
    ) -> IndexConfig:
        """
        根据需求推荐索引配置
        
        Args:
            collection_name: 集合名称
            expected_data_size: 预期数据规模
            query_latency_requirement_ms: 查询延迟要求（毫秒）
            recall_requirement: 召回率要求
            
        Returns:
            推荐的索引配置
        """
        # 根据数据规模选择基础索引类型
        if expected_data_size < 100_000:
            # 小规模：精确搜索
            base_config = self.PRESET_CONFIGS["small_exact"]
            
        elif expected_data_size < 10_000_000:
            # 中等规模：IVF
            if query_latency_requirement_ms < 10:
                base_config = self.PRESET_CONFIGS["low_latency"]
            else:
                base_config = self.PRESET_CONFIGS["medium_ivf"]
                
        elif expected_data_size < 100_000_000:
            # 大规模：HNSW
            if recall_requirement > 0.98:
                base_config = self.PRESET_CONFIGS["high_recall"]
            else:
                base_config = self.PRESET_CONFIGS["large_hnsw"]
                
        else:
            # 超大规模：DiskANN
            base_config = self.PRESET_CONFIGS["xlarge_diskann"]
        
        logger.info(
            f"Recommended index for {collection_name}: "
            f"type={base_config.index_type.value}, "
            f"data_size={expected_data_size:,}"
        )
        
        return base_config


def setup_indexes(
    manager: Optional[MilvusManager] = None,
    preset: str = "large_hnsw",
) -> Dict[str, bool]:
    """
    设置所有集合的索引
    
    Args:
        manager: Milvus 管理器实例
        preset: 预设配置名称
        
    Returns:
        设置结果
    """
    if manager is None:
        manager = MilvusManager()
        manager.connect()
    
    index_manager = IndexManager(manager)
    return index_manager.create_all_indexes(preset=preset)


if __name__ == "__main__":
    # 测试代码
    manager = MilvusManager()
    
    try:
        manager.connect()
        
        index_manager = IndexManager(manager)
        
        # 创建索引
        results = index_manager.create_all_indexes(preset="large_hnsw")
        
        print("\n=== Index Creation Results ===")
        for name, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {name}")
        
        # 获取索引信息
        print("\n=== Index Information ===")
        for name in results.keys():
            info = index_manager.get_index_info(name)
            print(f"\n{name}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        
    finally:
        manager.disconnect()

