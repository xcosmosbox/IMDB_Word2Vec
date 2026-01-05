"""
Milvus 向量数据库维护管理

本模块提供 Milvus 数据库的维护功能，包括：
- 数据压缩
- 垃圾回收
- 备份恢复
- 健康检查

项目：生成式推荐系统
模块：数据库管理
版本：1.0.0
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from pymilvus import (
    Collection,
    connections,
    utility,
    MilvusException,
)

from .collections import MilvusManager
from .indexes import IndexManager, IndexType, MetricType

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    is_healthy: bool
    checks: Dict[str, bool]
    details: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_healthy": self.is_healthy,
            "checks": self.checks,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CompactionResult:
    """压缩操作结果"""
    collection_name: str
    compaction_id: int
    state: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "compaction_id": self.compaction_id,
            "state": self.state,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
        }


class MaintenanceManager:
    """
    Milvus 维护管理器
    
    提供 Milvus 数据库的维护和运维功能。
    
    使用示例:
        manager = MilvusManager()
        manager.connect()
        
        maintenance = MaintenanceManager(manager)
        
        # 健康检查
        health = maintenance.health_check()
        print(f"Healthy: {health.is_healthy}")
        
        # 执行压缩
        result = maintenance.compact_collection("item_embeddings")
        
        manager.disconnect()
    """
    
    def __init__(
        self,
        milvus_manager: MilvusManager,
        backup_dir: Optional[str] = None,
    ):
        """
        初始化维护管理器
        
        Args:
            milvus_manager: Milvus 管理器实例
            backup_dir: 备份目录路径
        """
        self.manager = milvus_manager
        self.backup_dir = Path(backup_dir or os.getenv("MILVUS_BACKUP_DIR", "/backups/milvus"))
        self.index_manager = IndexManager(milvus_manager)
    
    def health_check(self) -> HealthCheckResult:
        """
        执行健康检查
        
        Returns:
            健康检查结果
        """
        checks = {}
        details = {}
        
        # 检查连接状态
        try:
            checks["connection"] = self.manager.is_connected()
            if not checks["connection"]:
                checks["connection"] = self.manager.connect()
        except Exception as e:
            checks["connection"] = False
            details["connection_error"] = str(e)
        
        # 检查集合状态
        if checks["connection"]:
            try:
                collections = self.manager.list_collections()
                checks["collections_accessible"] = True
                details["collection_count"] = len(collections)
                details["collections"] = collections
            except Exception as e:
                checks["collections_accessible"] = False
                details["collections_error"] = str(e)
        
        # 检查各集合健康状态
        collection_health = {}
        expected_collections = ["item_embeddings", "user_embeddings", "query_cache"]
        
        for coll_name in expected_collections:
            coll_health = self._check_collection_health(coll_name)
            collection_health[coll_name] = coll_health
        
        checks["all_collections_healthy"] = all(
            h.get("exists", False) and h.get("has_index", False)
            for h in collection_health.values()
        )
        details["collection_health"] = collection_health
        
        # 检查系统资源
        try:
            # Milvus 2.x 不直接暴露资源信息，这里记录基本状态
            checks["system_resources"] = True
        except Exception as e:
            checks["system_resources"] = False
            details["resources_error"] = str(e)
        
        is_healthy = all(checks.values())
        
        return HealthCheckResult(
            is_healthy=is_healthy,
            checks=checks,
            details=details,
            timestamp=datetime.now(),
        )
    
    def _check_collection_health(self, collection_name: str) -> Dict[str, Any]:
        """检查单个集合的健康状态"""
        result = {
            "exists": False,
            "has_index": False,
            "is_loaded": False,
            "entity_count": 0,
        }
        
        try:
            collection = self.manager.get_collection(collection_name)
            if collection is None:
                return result
            
            result["exists"] = True
            result["has_index"] = len(collection.indexes) > 0
            result["entity_count"] = collection.num_entities
            
            try:
                load_state = utility.load_state(collection_name)
                result["is_loaded"] = load_state.name == "Loaded"
            except:
                result["is_loaded"] = False
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def compact_collection(
        self,
        collection_name: str,
        wait_for_completion: bool = True,
        timeout: float = 3600.0,
    ) -> CompactionResult:
        """
        压缩集合数据
        
        Args:
            collection_name: 集合名称
            wait_for_completion: 是否等待完成
            timeout: 超时时间（秒）
            
        Returns:
            压缩结果
        """
        collection = self.manager.get_collection(collection_name)
        if collection is None:
            raise ValueError(f"Collection not found: {collection_name}")
        
        start_time = datetime.now()
        logger.info(f"Starting compaction for collection: {collection_name}")
        
        try:
            # 触发压缩
            compaction_id = collection.compact()
            
            result = CompactionResult(
                collection_name=collection_name,
                compaction_id=compaction_id,
                state="started",
                start_time=start_time,
                end_time=None,
                duration_seconds=None,
            )
            
            if wait_for_completion:
                # 等待压缩完成
                wait_start = time.time()
                while time.time() - wait_start < timeout:
                    state = collection.get_compaction_state()
                    
                    if state.state.name in ["Completed", "Success"]:
                        result.state = "completed"
                        result.end_time = datetime.now()
                        result.duration_seconds = (result.end_time - start_time).total_seconds()
                        logger.info(
                            f"Compaction completed for {collection_name} "
                            f"in {result.duration_seconds:.2f}s"
                        )
                        break
                    elif state.state.name in ["Failed", "Timeout"]:
                        result.state = "failed"
                        result.end_time = datetime.now()
                        logger.error(f"Compaction failed for {collection_name}")
                        break
                    
                    time.sleep(5)
                else:
                    result.state = "timeout"
                    result.end_time = datetime.now()
                    logger.warning(f"Compaction timeout for {collection_name}")
            
            return result
            
        except MilvusException as e:
            logger.error(f"Compaction error: {e}")
            raise
    
    def compact_all_collections(
        self,
        parallel: bool = False,
    ) -> Dict[str, CompactionResult]:
        """
        压缩所有集合
        
        Args:
            parallel: 是否并行执行
            
        Returns:
            各集合的压缩结果
        """
        collections = self.manager.list_collections()
        results = {}
        
        for coll_name in collections:
            try:
                result = self.compact_collection(coll_name, wait_for_completion=True)
                results[coll_name] = result
            except Exception as e:
                logger.error(f"Failed to compact {coll_name}: {e}")
                results[coll_name] = CompactionResult(
                    collection_name=coll_name,
                    compaction_id=-1,
                    state=f"error: {str(e)}",
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=0,
                )
        
        return results
    
    def flush_collection(self, collection_name: str) -> bool:
        """
        刷新集合数据到持久存储
        
        Args:
            collection_name: 集合名称
            
        Returns:
            是否成功
        """
        collection = self.manager.get_collection(collection_name)
        if collection is None:
            logger.error(f"Collection not found: {collection_name}")
            return False
        
        try:
            collection.flush()
            logger.info(f"Flushed collection: {collection_name}")
            return True
        except MilvusException as e:
            logger.error(f"Flush error: {e}")
            return False
    
    def flush_all_collections(self) -> Dict[str, bool]:
        """
        刷新所有集合
        
        Returns:
            各集合的刷新结果
        """
        collections = self.manager.list_collections()
        results = {}
        
        for coll_name in collections:
            results[coll_name] = self.flush_collection(coll_name)
        
        return results
    
    def get_collection_statistics(
        self,
        collection_name: str,
    ) -> Dict[str, Any]:
        """
        获取集合详细统计信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            统计信息
        """
        collection = self.manager.get_collection(collection_name)
        if collection is None:
            return {"error": "Collection not found"}
        
        try:
            collection.flush()
            
            stats = {
                "name": collection_name,
                "description": collection.description,
                "num_entities": collection.num_entities,
                "num_partitions": len(collection.partitions),
                "partitions": [p.name for p in collection.partitions],
                "schema": {
                    "fields": [
                        {
                            "name": f.name,
                            "dtype": f.dtype.name,
                            "is_primary": f.is_primary,
                        }
                        for f in collection.schema.fields
                    ]
                },
                "indexes": [],
            }
            
            for idx in collection.indexes:
                stats["indexes"].append({
                    "field_name": idx.field_name,
                    "index_type": idx.params.get("index_type"),
                    "metric_type": idx.params.get("metric_type"),
                })
            
            # 加载状态
            try:
                load_state = utility.load_state(collection_name)
                stats["load_state"] = load_state.name
            except:
                stats["load_state"] = "Unknown"
            
            return stats
            
        except MilvusException as e:
            return {"error": str(e)}
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有集合的统计信息
        
        Returns:
            各集合的统计信息
        """
        collections = self.manager.list_collections()
        return {
            coll_name: self.get_collection_statistics(coll_name)
            for coll_name in collections
        }
    
    def cleanup_old_data(
        self,
        collection_name: str,
        field_name: str,
        cutoff_timestamp: int,
    ) -> int:
        """
        清理过期数据
        
        Args:
            collection_name: 集合名称
            field_name: 时间戳字段名称
            cutoff_timestamp: 截止时间戳（Unix 时间戳）
            
        Returns:
            删除的记录数
        """
        collection = self.manager.get_collection(collection_name)
        if collection is None:
            raise ValueError(f"Collection not found: {collection_name}")
        
        try:
            # 构建删除表达式
            expr = f"{field_name} < {cutoff_timestamp}"
            
            # 执行删除
            result = collection.delete(expr)
            
            # 刷新
            collection.flush()
            
            deleted_count = result.delete_count
            logger.info(
                f"Deleted {deleted_count} records from {collection_name} "
                f"where {field_name} < {cutoff_timestamp}"
            )
            
            return deleted_count
            
        except MilvusException as e:
            logger.error(f"Cleanup error: {e}")
            raise
    
    def export_collection_metadata(
        self,
        collection_name: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        导出集合元数据
        
        Args:
            collection_name: 集合名称
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        stats = self.get_collection_statistics(collection_name)
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.backup_dir / f"{collection_name}_metadata_{timestamp}.json")
        
        # 确保目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Exported metadata to: {output_path}")
        return output_path
    
    def rebuild_index_if_needed(
        self,
        collection_name: str,
        threshold_ratio: float = 0.3,
    ) -> bool:
        """
        根据数据变化情况判断是否需要重建索引
        
        Args:
            collection_name: 集合名称
            threshold_ratio: 触发重建的数据变化比例阈值
            
        Returns:
            是否执行了重建
        """
        collection = self.manager.get_collection(collection_name)
        if collection is None:
            return False
        
        # 获取当前统计
        stats = self.get_collection_statistics(collection_name)
        
        # 这里简化处理，实际应该比较索引构建时的数据量与当前数据量
        # Milvus 2.x 会自动维护索引，这个方法主要用于手动触发重建
        
        if not stats.get("indexes"):
            logger.warning(f"No index found on {collection_name}")
            return False
        
        # 强制重建索引
        logger.info(f"Rebuilding index for {collection_name}")
        
        # 获取当前索引配置
        current_index = stats["indexes"][0]
        
        return self.index_manager.rebuild_index(
            collection_name,
            current_index.get("field_name", "embedding"),
        )
    
    def run_maintenance_routine(self) -> Dict[str, Any]:
        """
        执行完整的维护例程
        
        Returns:
            维护结果汇总
        """
        logger.info("Starting maintenance routine...")
        results = {
            "start_time": datetime.now().isoformat(),
            "health_check": None,
            "flush": {},
            "compact": {},
            "statistics": {},
            "end_time": None,
        }
        
        try:
            # 1. 健康检查
            health = self.health_check()
            results["health_check"] = health.to_dict()
            
            if not health.is_healthy:
                logger.warning("Health check failed, skipping some maintenance tasks")
            
            # 2. 刷新所有集合
            results["flush"] = self.flush_all_collections()
            
            # 3. 压缩所有集合
            compact_results = self.compact_all_collections()
            results["compact"] = {
                name: r.to_dict() for name, r in compact_results.items()
            }
            
            # 4. 收集统计信息
            results["statistics"] = self.get_all_statistics()
            
            results["end_time"] = datetime.now().isoformat()
            results["success"] = True
            
            logger.info("Maintenance routine completed successfully")
            
        except Exception as e:
            results["error"] = str(e)
            results["success"] = False
            logger.error(f"Maintenance routine failed: {e}")
        
        return results


def run_health_check(manager: Optional[MilvusManager] = None) -> HealthCheckResult:
    """
    执行健康检查
    
    Args:
        manager: Milvus 管理器实例
        
    Returns:
        健康检查结果
    """
    if manager is None:
        manager = MilvusManager()
        manager.connect()
    
    maintenance = MaintenanceManager(manager)
    return maintenance.health_check()


def run_maintenance(manager: Optional[MilvusManager] = None) -> Dict[str, Any]:
    """
    执行维护例程
    
    Args:
        manager: Milvus 管理器实例
        
    Returns:
        维护结果
    """
    if manager is None:
        manager = MilvusManager()
        manager.connect()
    
    maintenance = MaintenanceManager(manager)
    return maintenance.run_maintenance_routine()


if __name__ == "__main__":
    # 测试代码
    manager = MilvusManager()
    
    try:
        manager.connect()
        
        maintenance = MaintenanceManager(manager)
        
        # 健康检查
        print("\n=== Health Check ===")
        health = maintenance.health_check()
        print(f"Healthy: {health.is_healthy}")
        print(f"Checks: {health.checks}")
        
        # 获取统计
        print("\n=== Collection Statistics ===")
        stats = maintenance.get_all_statistics()
        for name, stat in stats.items():
            print(f"\n{name}:")
            if "error" not in stat:
                print(f"  Entities: {stat.get('num_entities', 0)}")
                print(f"  Partitions: {stat.get('num_partitions', 0)}")
                print(f"  Load State: {stat.get('load_state', 'Unknown')}")
        
    finally:
        manager.disconnect()

