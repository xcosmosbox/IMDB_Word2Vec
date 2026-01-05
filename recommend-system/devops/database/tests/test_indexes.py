"""
Milvus 索引管理单元测试

测试 IndexManager 类的各项功能。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestIndexType:
    """IndexType 枚举测试"""
    
    def test_index_types(self):
        """测试索引类型枚举值"""
        from milvus.indexes import IndexType
        
        assert IndexType.FLAT.value == "FLAT"
        assert IndexType.IVF_FLAT.value == "IVF_FLAT"
        assert IndexType.HNSW.value == "HNSW"
        assert IndexType.DISKANN.value == "DISKANN"


class TestMetricType:
    """MetricType 枚举测试"""
    
    def test_metric_types(self):
        """测试距离度量类型枚举值"""
        from milvus.indexes import MetricType
        
        assert MetricType.IP.value == "IP"
        assert MetricType.L2.value == "L2"
        assert MetricType.COSINE.value == "COSINE"


class TestIndexConfig:
    """IndexConfig 测试类"""
    
    def test_to_dict(self):
        """测试转换为字典"""
        from milvus.indexes import IndexConfig, IndexType, MetricType
        
        config = IndexConfig(
            index_type=IndexType.HNSW,
            metric_type=MetricType.IP,
            params={"M": 16, "efConstruction": 256},
        )
        
        result = config.to_dict()
        
        assert result["index_type"] == "HNSW"
        assert result["metric_type"] == "IP"
        assert result["params"]["M"] == 16
        assert result["params"]["efConstruction"] == 256


class TestIndexManager:
    """IndexManager 测试类"""
    
    def test_init(self, milvus_manager):
        """测试初始化"""
        from milvus.indexes import IndexManager
        
        index_manager = IndexManager(milvus_manager)
        
        assert index_manager.manager is milvus_manager
    
    def test_preset_configs(self):
        """测试预设配置"""
        from milvus.indexes import IndexManager, IndexType
        
        assert "small_exact" in IndexManager.PRESET_CONFIGS
        assert "medium_ivf" in IndexManager.PRESET_CONFIGS
        assert "large_hnsw" in IndexManager.PRESET_CONFIGS
        assert "xlarge_diskann" in IndexManager.PRESET_CONFIGS
        assert "high_recall" in IndexManager.PRESET_CONFIGS
        assert "low_latency" in IndexManager.PRESET_CONFIGS
        
        # 验证预设配置内容
        hnsw_config = IndexManager.PRESET_CONFIGS["large_hnsw"]
        assert hnsw_config.index_type == IndexType.HNSW
    
    def test_create_index_with_preset(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试使用预设创建索引"""
        from milvus.collections import MilvusManager
        from milvus.indexes import IndexManager
        
        mock_milvus_utility.has_collection.return_value = True
        
        manager = MilvusManager()
        manager._connected = True
        
        index_manager = IndexManager(manager)
        result = index_manager.create_index(
            "test_collection",
            "embedding",
            preset="large_hnsw",
        )
        
        assert result is True
    
    def test_create_index_custom(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试使用自定义参数创建索引"""
        from milvus.collections import MilvusManager
        from milvus.indexes import IndexManager, IndexType, MetricType
        
        mock_milvus_utility.has_collection.return_value = True
        
        manager = MilvusManager()
        manager._connected = True
        
        index_manager = IndexManager(manager)
        result = index_manager.create_index(
            "test_collection",
            "embedding",
            index_type=IndexType.IVF_FLAT,
            metric_type=MetricType.L2,
            params={"nlist": 2048},
        )
        
        assert result is True
    
    def test_create_index_collection_not_found(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
    ):
        """测试创建索引时集合不存在"""
        from milvus.collections import MilvusManager
        from milvus.indexes import IndexManager
        
        mock_milvus_utility.has_collection.return_value = False
        
        manager = MilvusManager()
        manager._connected = True
        
        index_manager = IndexManager(manager)
        result = index_manager.create_index(
            "nonexistent",
            "embedding",
        )
        
        assert result is False
    
    def test_drop_index(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试删除索引"""
        from milvus.collections import MilvusManager
        from milvus.indexes import IndexManager
        
        mock_milvus_utility.has_collection.return_value = True
        
        manager = MilvusManager()
        manager._connected = True
        
        index_manager = IndexManager(manager)
        result = index_manager.drop_index("test_collection")
        
        assert result is True
    
    def test_has_index(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试检查索引是否存在"""
        from milvus.collections import MilvusManager
        from milvus.indexes import IndexManager
        
        mock_milvus_utility.has_collection.return_value = True
        
        # 设置集合有索引
        mock_collection = mock_milvus_collection.return_value
        mock_collection.indexes = [Mock()]
        
        manager = MilvusManager()
        manager._connected = True
        
        index_manager = IndexManager(manager)
        result = index_manager.has_index("test_collection")
        
        assert result is True
    
    def test_has_index_no_index(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试检查索引不存在"""
        from milvus.collections import MilvusManager
        from milvus.indexes import IndexManager
        
        mock_milvus_utility.has_collection.return_value = True
        
        # 设置集合无索引
        mock_collection = mock_milvus_collection.return_value
        mock_collection.indexes = []
        
        manager = MilvusManager()
        manager._connected = True
        
        index_manager = IndexManager(manager)
        result = index_manager.has_index("test_collection")
        
        assert result is False
    
    def test_get_index_info(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试获取索引信息"""
        from milvus.collections import MilvusManager
        from milvus.indexes import IndexManager
        
        mock_milvus_utility.has_collection.return_value = True
        
        # 设置模拟索引
        mock_index = Mock()
        mock_index.field_name = "embedding"
        mock_index.params = {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 16},
        }
        
        mock_collection = mock_milvus_collection.return_value
        mock_collection.indexes = [mock_index]
        
        manager = MilvusManager()
        manager._connected = True
        
        index_manager = IndexManager(manager)
        info = index_manager.get_index_info("test_collection")
        
        assert "collection" in info
        assert "indexes" in info
        assert len(info["indexes"]) == 1
    
    def test_create_all_indexes(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试为所有集合创建索引"""
        from milvus.collections import MilvusManager
        from milvus.indexes import IndexManager
        
        mock_milvus_utility.has_collection.return_value = True
        mock_milvus_utility.list_collections.return_value = [
            "item_embeddings",
            "user_embeddings",
            "query_cache",
        ]
        
        manager = MilvusManager()
        manager._connected = True
        
        index_manager = IndexManager(manager)
        results = index_manager.create_all_indexes(preset="large_hnsw")
        
        assert "item_embeddings" in results
        assert "user_embeddings" in results
        assert "query_cache" in results
    
    def test_recommend_index_config_small(self):
        """测试推荐小规模数据索引配置"""
        from milvus.collections import MilvusManager
        from milvus.indexes import IndexManager, IndexType
        
        with patch("milvus.indexes.MilvusManager"):
            manager = Mock()
            manager.is_connected.return_value = True
            
            index_manager = IndexManager(manager)
            config = index_manager.recommend_index_config(
                "test_collection",
                expected_data_size=50000,
            )
            
            assert config.index_type == IndexType.FLAT
    
    def test_recommend_index_config_medium(self):
        """测试推荐中等规模数据索引配置"""
        from milvus.collections import MilvusManager
        from milvus.indexes import IndexManager, IndexType
        
        with patch("milvus.indexes.MilvusManager"):
            manager = Mock()
            manager.is_connected.return_value = True
            
            index_manager = IndexManager(manager)
            config = index_manager.recommend_index_config(
                "test_collection",
                expected_data_size=5000000,
            )
            
            assert config.index_type in [IndexType.IVF_FLAT, IndexType.IVF_SQ8]
    
    def test_recommend_index_config_large(self):
        """测试推荐大规模数据索引配置"""
        from milvus.collections import MilvusManager
        from milvus.indexes import IndexManager, IndexType
        
        with patch("milvus.indexes.MilvusManager"):
            manager = Mock()
            manager.is_connected.return_value = True
            
            index_manager = IndexManager(manager)
            config = index_manager.recommend_index_config(
                "test_collection",
                expected_data_size=50000000,
            )
            
            assert config.index_type == IndexType.HNSW
    
    def test_get_default_params(self):
        """测试获取默认索引参数"""
        from milvus.indexes import IndexManager, IndexType
        
        manager = Mock()
        index_manager = IndexManager(manager)
        
        flat_params = index_manager._get_default_params(IndexType.FLAT)
        assert flat_params == {}
        
        ivf_params = index_manager._get_default_params(IndexType.IVF_FLAT)
        assert "nlist" in ivf_params
        
        hnsw_params = index_manager._get_default_params(IndexType.HNSW)
        assert "M" in hnsw_params
        assert "efConstruction" in hnsw_params

