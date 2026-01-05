"""
Milvus 维护管理单元测试

测试 MaintenanceManager 类的各项功能。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import json

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHealthCheckResult:
    """HealthCheckResult 测试类"""
    
    def test_to_dict(self):
        """测试转换为字典"""
        from milvus.maintenance import HealthCheckResult
        
        result = HealthCheckResult(
            is_healthy=True,
            checks={"connection": True, "collections": True},
            details={"collection_count": 3},
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )
        
        data = result.to_dict()
        
        assert data["is_healthy"] is True
        assert data["checks"]["connection"] is True
        assert data["details"]["collection_count"] == 3
        assert "2024-01-01" in data["timestamp"]


class TestCompactionResult:
    """CompactionResult 测试类"""
    
    def test_to_dict(self):
        """测试转换为字典"""
        from milvus.maintenance import CompactionResult
        
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 5, 0)
        
        result = CompactionResult(
            collection_name="test_collection",
            compaction_id=12345,
            state="completed",
            start_time=start,
            end_time=end,
            duration_seconds=300.0,
        )
        
        data = result.to_dict()
        
        assert data["collection_name"] == "test_collection"
        assert data["compaction_id"] == 12345
        assert data["state"] == "completed"
        assert data["duration_seconds"] == 300.0


class TestMaintenanceManager:
    """MaintenanceManager 测试类"""
    
    def test_init(self, milvus_manager, temp_backup_dir):
        """测试初始化"""
        from milvus.maintenance import MaintenanceManager
        
        maintenance = MaintenanceManager(
            milvus_manager,
            backup_dir=str(temp_backup_dir),
        )
        
        assert maintenance.manager is milvus_manager
        assert str(temp_backup_dir) in str(maintenance.backup_dir)
    
    def test_health_check_healthy(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试健康检查 - 健康状态"""
        from milvus.collections import MilvusManager
        from milvus.maintenance import MaintenanceManager
        
        mock_milvus_utility.has_collection.return_value = True
        mock_milvus_utility.list_collections.return_value = [
            "item_embeddings",
            "user_embeddings",
            "query_cache",
        ]
        mock_milvus_utility.load_state.return_value = Mock(name="Loaded")
        
        # 设置集合有索引
        mock_collection = mock_milvus_collection.return_value
        mock_collection.indexes = [Mock()]
        mock_collection.num_entities = 1000
        
        manager = MilvusManager()
        manager._connected = True
        
        maintenance = MaintenanceManager(manager)
        result = maintenance.health_check()
        
        assert result.is_healthy is True
        assert "connection" in result.checks
    
    def test_health_check_not_connected(self, mock_milvus_connection, mock_milvus_utility):
        """测试健康检查 - 未连接"""
        from milvus.collections import MilvusManager
        from milvus.maintenance import MaintenanceManager
        
        # 模拟连接失败
        mock_milvus_connection.connect.side_effect = Exception("Connection failed")
        
        manager = MilvusManager()
        manager._connected = False
        
        maintenance = MaintenanceManager(manager)
        result = maintenance.health_check()
        
        assert result.checks.get("connection", True) is False or result.is_healthy is False
    
    def test_compact_collection(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试压缩集合"""
        from milvus.collections import MilvusManager
        from milvus.maintenance import MaintenanceManager
        
        mock_milvus_utility.has_collection.return_value = True
        
        manager = MilvusManager()
        manager._connected = True
        
        maintenance = MaintenanceManager(manager)
        result = maintenance.compact_collection(
            "test_collection",
            wait_for_completion=True,
        )
        
        assert result.collection_name == "test_collection"
        assert result.state == "completed"
    
    def test_compact_collection_not_found(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
    ):
        """测试压缩不存在的集合"""
        from milvus.collections import MilvusManager
        from milvus.maintenance import MaintenanceManager
        
        mock_milvus_utility.has_collection.return_value = False
        
        manager = MilvusManager()
        manager._connected = True
        
        maintenance = MaintenanceManager(manager)
        
        with pytest.raises(ValueError, match="Collection not found"):
            maintenance.compact_collection("nonexistent")
    
    def test_flush_collection(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试刷新集合"""
        from milvus.collections import MilvusManager
        from milvus.maintenance import MaintenanceManager
        
        mock_milvus_utility.has_collection.return_value = True
        
        manager = MilvusManager()
        manager._connected = True
        
        maintenance = MaintenanceManager(manager)
        result = maintenance.flush_collection("test_collection")
        
        assert result is True
    
    def test_flush_all_collections(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试刷新所有集合"""
        from milvus.collections import MilvusManager
        from milvus.maintenance import MaintenanceManager
        
        mock_milvus_utility.has_collection.return_value = True
        mock_milvus_utility.list_collections.return_value = ["coll1", "coll2"]
        
        manager = MilvusManager()
        manager._connected = True
        
        maintenance = MaintenanceManager(manager)
        results = maintenance.flush_all_collections()
        
        assert "coll1" in results
        assert "coll2" in results
    
    def test_get_collection_statistics(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试获取集合统计"""
        from milvus.collections import MilvusManager
        from milvus.maintenance import MaintenanceManager
        
        mock_milvus_utility.has_collection.return_value = True
        mock_milvus_utility.load_state.return_value = Mock(name="Loaded")
        
        manager = MilvusManager()
        manager._connected = True
        
        maintenance = MaintenanceManager(manager)
        stats = maintenance.get_collection_statistics("test_collection")
        
        assert "name" in stats
        assert "num_entities" in stats
        assert "schema" in stats
    
    def test_get_all_statistics(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试获取所有集合统计"""
        from milvus.collections import MilvusManager
        from milvus.maintenance import MaintenanceManager
        
        mock_milvus_utility.has_collection.return_value = True
        mock_milvus_utility.list_collections.return_value = ["coll1", "coll2"]
        mock_milvus_utility.load_state.return_value = Mock(name="Loaded")
        
        manager = MilvusManager()
        manager._connected = True
        
        maintenance = MaintenanceManager(manager)
        stats = maintenance.get_all_statistics()
        
        assert "coll1" in stats
        assert "coll2" in stats
    
    def test_cleanup_old_data(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试清理过期数据"""
        from milvus.collections import MilvusManager
        from milvus.maintenance import MaintenanceManager
        
        mock_milvus_utility.has_collection.return_value = True
        
        manager = MilvusManager()
        manager._connected = True
        
        maintenance = MaintenanceManager(manager)
        deleted = maintenance.cleanup_old_data(
            "test_collection",
            "updated_at",
            1704067200,  # 2024-01-01
        )
        
        assert deleted == 100  # Mock 返回值
    
    def test_export_collection_metadata(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
        temp_backup_dir,
    ):
        """测试导出集合元数据"""
        from milvus.collections import MilvusManager
        from milvus.maintenance import MaintenanceManager
        
        mock_milvus_utility.has_collection.return_value = True
        mock_milvus_utility.load_state.return_value = Mock(name="Loaded")
        
        manager = MilvusManager()
        manager._connected = True
        
        maintenance = MaintenanceManager(manager, backup_dir=str(temp_backup_dir))
        
        output_path = maintenance.export_collection_metadata("test_collection")
        
        assert output_path is not None
        assert os.path.exists(output_path)
        
        # 验证内容
        with open(output_path) as f:
            data = json.load(f)
            assert "name" in data
    
    def test_run_maintenance_routine(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试运行维护例程"""
        from milvus.collections import MilvusManager
        from milvus.maintenance import MaintenanceManager
        
        mock_milvus_utility.has_collection.return_value = True
        mock_milvus_utility.list_collections.return_value = [
            "item_embeddings",
            "user_embeddings",
        ]
        mock_milvus_utility.load_state.return_value = Mock(name="Loaded")
        
        # 设置集合有索引
        mock_collection = mock_milvus_collection.return_value
        mock_collection.indexes = [Mock()]
        
        manager = MilvusManager()
        manager._connected = True
        
        maintenance = MaintenanceManager(manager)
        results = maintenance.run_maintenance_routine()
        
        assert "health_check" in results
        assert "flush" in results
        assert "compact" in results
        assert "statistics" in results
        assert results.get("success", False) is True


class TestHelperFunctions:
    """辅助函数测试"""
    
    def test_run_health_check(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试 run_health_check 函数"""
        from milvus.maintenance import run_health_check
        from milvus.collections import MilvusManager
        
        mock_milvus_utility.has_collection.return_value = True
        mock_milvus_utility.list_collections.return_value = []
        mock_milvus_utility.load_state.return_value = Mock(name="Loaded")
        
        # 设置集合有索引
        mock_collection = mock_milvus_collection.return_value
        mock_collection.indexes = [Mock()]
        
        manager = MilvusManager()
        manager._connected = True
        
        result = run_health_check(manager)
        
        assert result is not None
        assert hasattr(result, "is_healthy")
    
    def test_run_maintenance(
        self,
        mock_milvus_connection,
        mock_milvus_utility,
        mock_milvus_collection,
    ):
        """测试 run_maintenance 函数"""
        from milvus.maintenance import run_maintenance
        from milvus.collections import MilvusManager
        
        mock_milvus_utility.has_collection.return_value = True
        mock_milvus_utility.list_collections.return_value = ["test"]
        mock_milvus_utility.load_state.return_value = Mock(name="Loaded")
        
        # 设置集合有索引
        mock_collection = mock_milvus_collection.return_value
        mock_collection.indexes = [Mock()]
        
        manager = MilvusManager()
        manager._connected = True
        
        result = run_maintenance(manager)
        
        assert result is not None
        assert "health_check" in result

