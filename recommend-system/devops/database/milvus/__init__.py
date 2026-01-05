"""
Milvus 向量数据库管理模块

本模块提供 Milvus 向量数据库的管理功能，包括：
- 集合管理（创建、删除、配置）
- 索引管理（创建、重建、优化）
- 数据维护（压缩、清理、备份）

项目：生成式推荐系统
模块：数据库管理
版本：1.0.0
"""

from .collections import MilvusManager, CollectionConfig
from .indexes import IndexManager, IndexType
from .maintenance import MaintenanceManager

__all__ = [
    "MilvusManager",
    "CollectionConfig",
    "IndexManager",
    "IndexType",
    "MaintenanceManager",
]

__version__ = "1.0.0"

