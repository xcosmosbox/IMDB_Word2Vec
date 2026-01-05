"""
Pytest 配置文件

提供测试 fixtures 和配置。
"""

import os
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Generator, Dict, Any


# 设置测试环境变量
os.environ.setdefault("MILVUS_HOST", "localhost")
os.environ.setdefault("MILVUS_PORT", "19530")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "recommend_test")
os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "test_password")


@pytest.fixture
def mock_milvus_connection():
    """Mock Milvus 连接"""
    with patch("pymilvus.connections") as mock_conn:
        mock_conn.connect = Mock(return_value=None)
        mock_conn.disconnect = Mock(return_value=None)
        yield mock_conn


@pytest.fixture
def mock_milvus_utility():
    """Mock Milvus utility 模块"""
    with patch("pymilvus.utility") as mock_utility:
        mock_utility.has_collection = Mock(return_value=False)
        mock_utility.list_collections = Mock(return_value=[])
        mock_utility.drop_collection = Mock(return_value=None)
        mock_utility.load_state = Mock(return_value=Mock(name="NotLoad"))
        mock_utility.loading_progress = Mock(return_value={"loading_progress": "100%"})
        mock_utility.index_building_progress = Mock(return_value={"index_building_progress": "100%"})
        yield mock_utility


@pytest.fixture
def mock_milvus_collection():
    """Mock Milvus Collection"""
    with patch("pymilvus.Collection") as mock_coll_class:
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.description = "Test description"
        mock_collection.num_entities = 1000
        mock_collection.schema = MagicMock()
        mock_collection.schema.fields = []
        mock_collection.schema.description = "Test schema"
        mock_collection.schema.enable_dynamic_field = True
        mock_collection.indexes = []
        mock_collection.partitions = []
        mock_collection.flush = Mock(return_value=None)
        mock_collection.load = Mock(return_value=None)
        mock_collection.release = Mock(return_value=None)
        mock_collection.create_index = Mock(return_value=None)
        mock_collection.drop_index = Mock(return_value=None)
        mock_collection.compact = Mock(return_value=12345)
        mock_collection.get_compaction_state = Mock(
            return_value=Mock(state=Mock(name="Completed"))
        )
        mock_collection.delete = Mock(return_value=Mock(delete_count=100))
        
        mock_coll_class.return_value = mock_collection
        yield mock_coll_class


@pytest.fixture
def milvus_manager(mock_milvus_connection, mock_milvus_utility):
    """创建 MilvusManager 实例用于测试"""
    from milvus.collections import MilvusManager
    
    manager = MilvusManager(
        host="localhost",
        port="19530",
    )
    manager._connected = True
    
    yield manager
    
    manager._connected = False


@pytest.fixture
def index_manager(milvus_manager):
    """创建 IndexManager 实例用于测试"""
    from milvus.indexes import IndexManager
    
    return IndexManager(milvus_manager)


@pytest.fixture
def maintenance_manager(milvus_manager):
    """创建 MaintenanceManager 实例用于测试"""
    from milvus.maintenance import MaintenanceManager
    
    return MaintenanceManager(milvus_manager, backup_dir="/tmp/test_backups")


@pytest.fixture
def sample_collection_config():
    """示例集合配置"""
    from milvus.collections import CollectionConfig, ConsistencyLevel
    from pymilvus import DataType
    
    return CollectionConfig(
        name="test_embeddings",
        description="Test embeddings collection",
        dim=256,
        primary_field="id",
        vector_field="embedding",
        consistency_level=ConsistencyLevel.BOUNDED,
        shards_num=2,
        extra_fields=[
            {"name": "category", "dtype": DataType.VARCHAR, "max_length": 100},
            {"name": "created_at", "dtype": DataType.INT64},
        ]
    )


@pytest.fixture
def mock_postgres_connection():
    """Mock PostgreSQL 连接"""
    with patch("psycopg2.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone = Mock(return_value=(1,))
        mock_cursor.fetchall = Mock(return_value=[(1,), (2,), (3,)])
        mock_cursor.execute = Mock(return_value=None)
        mock_conn.cursor = Mock(return_value=mock_cursor)
        mock_connect.return_value = mock_conn
        yield mock_connect


@pytest.fixture
def temp_backup_dir(tmp_path):
    """创建临时备份目录"""
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    return backup_dir

