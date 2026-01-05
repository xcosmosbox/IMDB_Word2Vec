"""
Milvus 集合管理单元测试

测试 MilvusManager 类的各项功能。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMilvusManager:
    """MilvusManager 测试类"""
    
    def test_init_with_defaults(self):
        """测试默认初始化"""
        with patch("milvus.collections.connections"):
            from milvus.collections import MilvusManager
            
            manager = MilvusManager()
            
            assert manager.host == "localhost"
            assert manager.port == "19530"
            assert manager.alias == "default"
            assert not manager._connected
    
    def test_init_with_custom_params(self):
        """测试自定义参数初始化"""
        with patch("milvus.collections.connections"):
            from milvus.collections import MilvusManager
            
            manager = MilvusManager(
                host="milvus.example.com",
                port="19531",
                user="test_user",
                password="test_pass",
                alias="custom",
                timeout=60.0,
            )
            
            assert manager.host == "milvus.example.com"
            assert manager.port == "19531"
            assert manager.user == "test_user"
            assert manager.password == "test_pass"
            assert manager.alias == "custom"
            assert manager.timeout == 60.0
    
    def test_connect_success(self, mock_milvus_connection):
        """测试成功连接"""
        from milvus.collections import MilvusManager
        
        manager = MilvusManager()
        result = manager.connect()
        
        assert result is True
        assert manager._connected is True
        mock_milvus_connection.connect.assert_called_once()
    
    def test_connect_already_connected(self, mock_milvus_connection):
        """测试重复连接"""
        from milvus.collections import MilvusManager
        
        manager = MilvusManager()
        manager._connected = True
        
        result = manager.connect()
        
        assert result is True
        mock_milvus_connection.connect.assert_not_called()
    
    def test_disconnect(self, mock_milvus_connection):
        """测试断开连接"""
        from milvus.collections import MilvusManager
        
        manager = MilvusManager()
        manager._connected = True
        manager._collections = {"test": Mock()}
        
        manager.disconnect()
        
        assert manager._connected is False
        assert len(manager._collections) == 0
        mock_milvus_connection.disconnect.assert_called_once()
    
    def test_is_connected(self):
        """测试连接状态检查"""
        with patch("milvus.collections.connections"):
            from milvus.collections import MilvusManager
            
            manager = MilvusManager()
            
            assert manager.is_connected() is False
            
            manager._connected = True
            assert manager.is_connected() is True
    
    def test_list_collections(self, mock_milvus_connection, mock_milvus_utility):
        """测试列出集合"""
        from milvus.collections import MilvusManager
        
        mock_milvus_utility.list_collections.return_value = ["coll1", "coll2", "coll3"]
        
        manager = MilvusManager()
        manager._connected = True
        
        collections = manager.list_collections()
        
        assert len(collections) == 3
        assert "coll1" in collections
    
    def test_create_item_embeddings_collection(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试创建物品嵌入集合"""
        from milvus.collections import MilvusManager
        
        mock_milvus_utility.has_collection.return_value = False
        
        manager = MilvusManager()
        manager._connected = True
        
        collection = manager.create_item_embeddings_collection(dim=256)
        
        assert collection is not None
        mock_milvus_collection.assert_called()
    
    def test_create_user_embeddings_collection(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试创建用户嵌入集合"""
        from milvus.collections import MilvusManager
        
        mock_milvus_utility.has_collection.return_value = False
        
        manager = MilvusManager()
        manager._connected = True
        
        collection = manager.create_user_embeddings_collection(dim=256)
        
        assert collection is not None
    
    def test_create_collection_already_exists(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
        sample_collection_config,
    ):
        """测试创建已存在的集合"""
        from milvus.collections import MilvusManager
        
        mock_milvus_utility.has_collection.return_value = True
        
        manager = MilvusManager()
        manager._connected = True
        
        collection = manager.create_collection(sample_collection_config)
        
        assert collection is not None
        # 不应该删除已存在的集合
        mock_milvus_utility.drop_collection.assert_not_called()
    
    def test_create_collection_drop_if_exists(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
        sample_collection_config,
    ):
        """测试删除已存在集合后创建"""
        from milvus.collections import MilvusManager
        
        mock_milvus_utility.has_collection.return_value = True
        
        manager = MilvusManager()
        manager._connected = True
        
        collection = manager.create_collection(
            sample_collection_config,
            drop_if_exists=True,
        )
        
        assert collection is not None
        mock_milvus_utility.drop_collection.assert_called_once()
    
    def test_get_collection_cached(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试获取缓存的集合"""
        from milvus.collections import MilvusManager
        
        manager = MilvusManager()
        manager._connected = True
        
        mock_coll = Mock()
        manager._collections["test_collection"] = mock_coll
        
        collection = manager.get_collection("test_collection")
        
        assert collection is mock_coll
    
    def test_get_collection_not_exists(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
    ):
        """测试获取不存在的集合"""
        from milvus.collections import MilvusManager
        
        mock_milvus_utility.has_collection.return_value = False
        
        manager = MilvusManager()
        manager._connected = True
        
        collection = manager.get_collection("nonexistent")
        
        assert collection is None
    
    def test_drop_collection(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
    ):
        """测试删除集合"""
        from milvus.collections import MilvusManager
        
        mock_milvus_utility.has_collection.return_value = True
        
        manager = MilvusManager()
        manager._connected = True
        manager._collections["test_collection"] = Mock()
        
        result = manager.drop_collection("test_collection")
        
        assert result is True
        assert "test_collection" not in manager._collections
        mock_milvus_utility.drop_collection.assert_called_once()
    
    def test_drop_collection_not_exists(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
    ):
        """测试删除不存在的集合"""
        from milvus.collections import MilvusManager
        
        mock_milvus_utility.has_collection.return_value = False
        
        manager = MilvusManager()
        manager._connected = True
        
        result = manager.drop_collection("nonexistent")
        
        assert result is False
    
    def test_load_collection(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试加载集合"""
        from milvus.collections import MilvusManager
        
        mock_milvus_utility.has_collection.return_value = True
        
        manager = MilvusManager()
        manager._connected = True
        
        result = manager.load_collection("test_collection")
        
        assert result is True
    
    def test_release_collection(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试释放集合"""
        from milvus.collections import MilvusManager
        
        mock_milvus_utility.has_collection.return_value = True
        
        manager = MilvusManager()
        manager._connected = True
        
        result = manager.release_collection("test_collection")
        
        assert result is True
    
    def test_get_stats(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试获取集合统计"""
        from milvus.collections import MilvusManager
        
        mock_milvus_utility.has_collection.return_value = True
        mock_milvus_utility.load_state.return_value = Mock(name="Loaded")
        
        manager = MilvusManager()
        manager._connected = True
        
        stats = manager.get_stats("test_collection")
        
        assert "name" in stats
        assert "num_entities" in stats
        assert "schema" in stats


class TestCollectionConfig:
    """CollectionConfig 测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        from milvus.collections import CollectionConfig
        
        config = CollectionConfig(name="test")
        
        assert config.name == "test"
        assert config.dim == 256
        assert config.primary_field == "id"
        assert config.vector_field == "embedding"
        assert config.shards_num == 2
    
    def test_custom_config(self):
        """测试自定义配置"""
        from milvus.collections import CollectionConfig, ConsistencyLevel
        
        config = CollectionConfig(
            name="custom",
            description="Custom collection",
            dim=512,
            primary_field="custom_id",
            vector_field="custom_embedding",
            consistency_level=ConsistencyLevel.STRONG,
            shards_num=4,
        )
        
        assert config.name == "custom"
        assert config.dim == 512
        assert config.consistency_level == ConsistencyLevel.STRONG
        assert config.shards_num == 4


class TestInitAllCollections:
    """init_all_collections 函数测试"""
    
    def test_init_all_collections(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试初始化所有集合"""
        from milvus.collections import init_all_collections, MilvusManager
        
        mock_milvus_utility.has_collection.return_value = False
        
        manager = MilvusManager()
        manager._connected = True
        
        collections = init_all_collections(manager, dim=256)
        
        assert "item_embeddings" in collections
        assert "user_embeddings" in collections
        assert "query_cache" in collections

