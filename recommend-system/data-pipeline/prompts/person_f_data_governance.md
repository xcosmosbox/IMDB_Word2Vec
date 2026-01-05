# Person F: 数据治理 (Data Governance)

## 你的角色
你是一名数据工程师，负责实现生成式推荐系统的 **数据治理模块**，包括数据目录、数据血缘、访问控制、审计日志等。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读接口定义文件：**

```
data-pipeline/interfaces.py
```

你需要实现的接口：

```python
class DataCatalogInterface(ABC):
    @abstractmethod
    def register_dataset(self, name, schema, metadata) -> str:
        pass
    
    @abstractmethod
    def search_datasets(self, query, filters) -> List[Dict]:
        pass

class DataLineageInterface(ABC):
    @abstractmethod
    def record_lineage(self, source_nodes, target_node, edges) -> str:
        pass
    
    @abstractmethod
    def get_upstream(self, node_id, depth) -> Tuple[List, List]:
        pass
    
    @abstractmethod
    def get_downstream(self, node_id, depth) -> Tuple[List, List]:
        pass

class DataAccessControlInterface(ABC):
    @abstractmethod
    def grant_access(self, user_id, dataset_name, permissions) -> bool:
        pass
    
    @abstractmethod
    def check_access(self, user_id, dataset_name, permission) -> bool:
        pass
```

---

## 技术栈

- **数据目录**: PostgreSQL + Elasticsearch
- **血缘追踪**: Neo4j (图数据库)
- **访问控制**: RBAC + ABAC
- **可视化**: D3.js / Mermaid

---

## 你的任务

```
data-pipeline/data-governance/
├── catalog/
│   ├── __init__.py
│   ├── data_catalog.py       # 数据目录
│   ├── metadata_store.py     # 元数据存储
│   └── search.py             # 搜索功能
├── lineage/
│   ├── __init__.py
│   ├── lineage_tracker.py    # 血缘追踪
│   ├── graph_store.py        # 图存储
│   └── visualizer.py         # 可视化
├── access/
│   ├── __init__.py
│   ├── access_control.py     # 访问控制
│   ├── policies.py           # 策略定义
│   └── audit_log.py          # 审计日志
└── tests/
    ├── test_catalog.py
    ├── test_lineage.py
    └── test_access.py
```

---

## 1. 数据目录 (catalog/data_catalog.py)

```python
"""
数据目录

管理数据集的元数据和发现
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
import uuid
import json
import logging

from ..interfaces import DataCatalogInterface

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """数据集元数据"""
    id: str
    name: str
    schema: Dict[str, str]
    description: str = ""
    owner: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 统计信息
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    last_updated_data: Optional[datetime] = None
    
    # 分类
    domain: str = ""  # 业务领域
    sensitivity: str = "internal"  # public, internal, confidential, restricted


class DataCatalog(DataCatalogInterface):
    """
    数据目录
    
    使用示例:
        catalog = DataCatalog(storage)
        
        # 注册数据集
        dataset_id = catalog.register_dataset(
            name="user_behaviors",
            schema={"user_id": "string", "item_id": "string", ...},
            metadata={"source": "kafka", "format": "parquet"},
        )
        
        # 搜索数据集
        results = catalog.search_datasets("user behavior")
    """
    
    def __init__(self, storage_backend=None):
        self._storage = storage_backend or InMemoryStorage()
    
    def register_dataset(
        self,
        name: str,
        schema: Dict[str, str],
        metadata: Dict[str, Any],
    ) -> str:
        """
        注册数据集
        
        Returns:
            str: 数据集 ID
        """
        # 检查是否已存在
        existing = self._storage.get_by_name(name)
        if existing:
            # 更新现有数据集
            existing.schema = schema
            existing.metadata.update(metadata)
            existing.updated_at = datetime.now()
            self._storage.save(existing)
            logger.info(f"Updated dataset: {name}")
            return existing.id
        
        # 创建新数据集
        dataset = Dataset(
            id=str(uuid.uuid4()),
            name=name,
            schema=schema,
            metadata=metadata,
            description=metadata.get('description', ''),
            owner=metadata.get('owner', ''),
            tags=metadata.get('tags', []),
            domain=metadata.get('domain', ''),
            sensitivity=metadata.get('sensitivity', 'internal'),
        )
        
        self._storage.save(dataset)
        logger.info(f"Registered dataset: {name} ({dataset.id})")
        
        return dataset.id
    
    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """获取数据集信息"""
        dataset = self._storage.get_by_name(name)
        if dataset:
            return asdict(dataset)
        return None
    
    def get_dataset_by_id(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """通过 ID 获取数据集"""
        dataset = self._storage.get_by_id(dataset_id)
        if dataset:
            return asdict(dataset)
        return None
    
    def search_datasets(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索数据集
        
        Args:
            query: 搜索关键词
            filters: 过滤条件 (domain, owner, tags, sensitivity)
            
        Returns:
            List[Dict]: 匹配的数据集列表
        """
        all_datasets = self._storage.get_all()
        results = []
        
        query_lower = query.lower()
        
        for dataset in all_datasets:
            # 关键词匹配
            if query:
                match = (
                    query_lower in dataset.name.lower() or
                    query_lower in dataset.description.lower() or
                    any(query_lower in tag.lower() for tag in dataset.tags)
                )
                if not match:
                    continue
            
            # 过滤条件
            if filters:
                if filters.get('domain') and dataset.domain != filters['domain']:
                    continue
                if filters.get('owner') and dataset.owner != filters['owner']:
                    continue
                if filters.get('sensitivity') and dataset.sensitivity != filters['sensitivity']:
                    continue
                if filters.get('tags'):
                    if not any(tag in dataset.tags for tag in filters['tags']):
                        continue
            
            results.append(asdict(dataset))
        
        # 按相关性排序（简单实现）
        results.sort(key=lambda x: x['name'].lower().startswith(query_lower), reverse=True)
        
        return results
    
    def add_tags(self, dataset_name: str, tags: List[str]) -> None:
        """添加标签"""
        dataset = self._storage.get_by_name(dataset_name)
        if dataset:
            for tag in tags:
                if tag not in dataset.tags:
                    dataset.tags.append(tag)
            dataset.updated_at = datetime.now()
            self._storage.save(dataset)
    
    def update_metadata(
        self,
        dataset_name: str,
        metadata: Dict[str, Any],
    ) -> None:
        """更新元数据"""
        dataset = self._storage.get_by_name(dataset_name)
        if dataset:
            dataset.metadata.update(metadata)
            dataset.updated_at = datetime.now()
            self._storage.save(dataset)
    
    def update_statistics(
        self,
        dataset_name: str,
        row_count: Optional[int] = None,
        size_bytes: Optional[int] = None,
    ) -> None:
        """更新统计信息"""
        dataset = self._storage.get_by_name(dataset_name)
        if dataset:
            if row_count is not None:
                dataset.row_count = row_count
            if size_bytes is not None:
                dataset.size_bytes = size_bytes
            dataset.last_updated_data = datetime.now()
            dataset.updated_at = datetime.now()
            self._storage.save(dataset)
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """删除数据集"""
        return self._storage.delete_by_name(dataset_name)
    
    def get_all_domains(self) -> List[str]:
        """获取所有领域"""
        all_datasets = self._storage.get_all()
        domains = set(d.domain for d in all_datasets if d.domain)
        return sorted(domains)
    
    def get_all_tags(self) -> List[str]:
        """获取所有标签"""
        all_datasets = self._storage.get_all()
        tags = set()
        for d in all_datasets:
            tags.update(d.tags)
        return sorted(tags)


class InMemoryStorage:
    """内存存储（开发用）"""
    
    def __init__(self):
        self._datasets: Dict[str, Dataset] = {}
        self._name_index: Dict[str, str] = {}
    
    def save(self, dataset: Dataset) -> None:
        self._datasets[dataset.id] = dataset
        self._name_index[dataset.name] = dataset.id
    
    def get_by_id(self, dataset_id: str) -> Optional[Dataset]:
        return self._datasets.get(dataset_id)
    
    def get_by_name(self, name: str) -> Optional[Dataset]:
        dataset_id = self._name_index.get(name)
        if dataset_id:
            return self._datasets.get(dataset_id)
        return None
    
    def get_all(self) -> List[Dataset]:
        return list(self._datasets.values())
    
    def delete_by_name(self, name: str) -> bool:
        dataset_id = self._name_index.get(name)
        if dataset_id:
            del self._datasets[dataset_id]
            del self._name_index[name]
            return True
        return False
```

---

## 2. 数据血缘 (lineage/lineage_tracker.py)

```python
"""
数据血缘追踪

记录和查询数据流转关系
"""

from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid
import logging
import json

from ..interfaces import (
    DataLineageInterface,
    LineageNode,
    LineageEdge,
)

logger = logging.getLogger(__name__)


class LineageTracker(DataLineageInterface):
    """
    数据血缘追踪器
    
    使用示例:
        tracker = LineageTracker()
        
        # 记录血缘
        source = LineageNode(node_id="kafka_topic", node_type="source", name="user_events")
        target = LineageNode(node_id="clean_events", node_type="transform", name="cleaned_data")
        edge = LineageEdge(source_id="kafka_topic", target_id="clean_events", transform_type="ETL")
        
        tracker.record_lineage([source], target, [edge])
        
        # 查询上下游
        upstream = tracker.get_upstream("clean_events")
        downstream = tracker.get_downstream("kafka_topic")
    """
    
    def __init__(self, graph_store=None):
        self._store = graph_store or InMemoryGraphStore()
    
    def record_lineage(
        self,
        source_nodes: List[LineageNode],
        target_node: LineageNode,
        edges: List[LineageEdge],
    ) -> str:
        """
        记录血缘关系
        
        Returns:
            str: 血缘记录 ID
        """
        lineage_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # 保存节点
        for node in source_nodes:
            self._store.add_node(node, timestamp)
        
        self._store.add_node(target_node, timestamp)
        
        # 保存边
        for edge in edges:
            self._store.add_edge(edge, lineage_id, timestamp)
        
        logger.info(f"Recorded lineage: {[n.name for n in source_nodes]} -> {target_node.name}")
        
        return lineage_id
    
    def get_upstream(
        self,
        node_id: str,
        depth: int = 1,
    ) -> Tuple[List[LineageNode], List[LineageEdge]]:
        """
        获取上游节点
        
        Args:
            node_id: 目标节点 ID
            depth: 追溯深度
            
        Returns:
            Tuple: (节点列表, 边列表)
        """
        visited_nodes = set()
        visited_edges = set()
        
        nodes = []
        edges = []
        
        def traverse(current_id: str, current_depth: int):
            if current_depth > depth or current_id in visited_nodes:
                return
            
            visited_nodes.add(current_id)
            
            # 获取当前节点
            node = self._store.get_node(current_id)
            if node:
                nodes.append(node)
            
            # 获取上游边
            upstream_edges = self._store.get_incoming_edges(current_id)
            
            for edge in upstream_edges:
                edge_key = (edge.source_id, edge.target_id)
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    edges.append(edge)
                    traverse(edge.source_id, current_depth + 1)
        
        traverse(node_id, 0)
        
        return nodes, edges
    
    def get_downstream(
        self,
        node_id: str,
        depth: int = 1,
    ) -> Tuple[List[LineageNode], List[LineageEdge]]:
        """获取下游节点"""
        visited_nodes = set()
        visited_edges = set()
        
        nodes = []
        edges = []
        
        def traverse(current_id: str, current_depth: int):
            if current_depth > depth or current_id in visited_nodes:
                return
            
            visited_nodes.add(current_id)
            
            node = self._store.get_node(current_id)
            if node:
                nodes.append(node)
            
            downstream_edges = self._store.get_outgoing_edges(current_id)
            
            for edge in downstream_edges:
                edge_key = (edge.source_id, edge.target_id)
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    edges.append(edge)
                    traverse(edge.target_id, current_depth + 1)
        
        traverse(node_id, 0)
        
        return nodes, edges
    
    def get_impact_analysis(
        self,
        node_id: str,
    ) -> Dict[str, Any]:
        """
        影响分析
        
        分析如果该节点变更，会影响哪些下游
        """
        downstream_nodes, downstream_edges = self.get_downstream(node_id, depth=10)
        
        # 按类型分组
        impact_by_type = {}
        for node in downstream_nodes:
            if node.node_id == node_id:
                continue
            node_type = node.node_type
            if node_type not in impact_by_type:
                impact_by_type[node_type] = []
            impact_by_type[node_type].append({
                'id': node.node_id,
                'name': node.name,
                'metadata': node.metadata,
            })
        
        return {
            'source_node': node_id,
            'total_impacted': len(downstream_nodes) - 1,
            'impact_by_type': impact_by_type,
            'edges': len(downstream_edges),
        }
    
    def visualize(self, node_id: str, depth: int = 3) -> str:
        """
        生成血缘可视化
        
        返回 Mermaid 图格式
        """
        # 获取上下游
        upstream_nodes, upstream_edges = self.get_upstream(node_id, depth)
        downstream_nodes, downstream_edges = self.get_downstream(node_id, depth)
        
        all_nodes = {n.node_id: n for n in upstream_nodes + downstream_nodes}
        all_edges = upstream_edges + downstream_edges
        
        # 生成 Mermaid 图
        lines = ["graph LR"]
        
        # 节点样式
        for node in all_nodes.values():
            style = ""
            if node.node_type == "source":
                style = ":::source"
            elif node.node_type == "sink":
                style = ":::sink"
            elif node.node_id == node_id:
                style = ":::current"
            
            lines.append(f"    {node.node_id}[{node.name}]{style}")
        
        # 边
        for edge in all_edges:
            label = edge.transform_type or ""
            if label:
                lines.append(f"    {edge.source_id} -->|{label}| {edge.target_id}")
            else:
                lines.append(f"    {edge.source_id} --> {edge.target_id}")
        
        # 样式定义
        lines.extend([
            "",
            "    classDef source fill:#90EE90;",
            "    classDef sink fill:#87CEEB;",
            "    classDef current fill:#FFD700;",
        ])
        
        return "\n".join(lines)


class InMemoryGraphStore:
    """内存图存储（开发用）"""
    
    def __init__(self):
        self._nodes: Dict[str, LineageNode] = {}
        self._edges: List[LineageEdge] = []
    
    def add_node(self, node: LineageNode, timestamp: datetime) -> None:
        self._nodes[node.node_id] = node
    
    def add_edge(self, edge: LineageEdge, lineage_id: str, timestamp: datetime) -> None:
        self._edges.append(edge)
    
    def get_node(self, node_id: str) -> Optional[LineageNode]:
        return self._nodes.get(node_id)
    
    def get_incoming_edges(self, node_id: str) -> List[LineageEdge]:
        return [e for e in self._edges if e.target_id == node_id]
    
    def get_outgoing_edges(self, node_id: str) -> List[LineageEdge]:
        return [e for e in self._edges if e.source_id == node_id]
```

---

## 3. 访问控制 (access/access_control.py)

```python
"""
数据访问控制

基于 RBAC + ABAC 的访问控制
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..interfaces import DataAccessControlInterface

logger = logging.getLogger(__name__)


class Permission(Enum):
    """权限类型"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class AccessGrant:
    """访问授权"""
    user_id: str
    dataset_name: str
    permissions: List[Permission]
    granted_by: str
    granted_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessLog:
    """访问日志"""
    user_id: str
    dataset_name: str
    permission: Permission
    action: str  # "grant", "revoke", "access", "deny"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataAccessControl(DataAccessControlInterface):
    """
    数据访问控制
    
    使用示例:
        access = DataAccessControl()
        
        # 授权
        access.grant_access("user_1", "user_behaviors", [Permission.READ])
        
        # 检查权限
        if access.check_access("user_1", "user_behaviors", Permission.READ):
            # 允许访问
            pass
    """
    
    def __init__(self, storage=None):
        self._storage = storage or InMemoryAccessStorage()
        self._audit_log: List[AccessLog] = []
    
    def grant_access(
        self,
        user_id: str,
        dataset_name: str,
        permissions: List[str],
        granted_by: str = "system",
        expires_at: Optional[datetime] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        授予访问权限
        
        Args:
            user_id: 用户 ID
            dataset_name: 数据集名称
            permissions: 权限列表 ["read", "write", ...]
            granted_by: 授权人
            expires_at: 过期时间
            conditions: 访问条件 (ABAC)
        """
        try:
            perm_enums = [Permission(p) for p in permissions]
            
            grant = AccessGrant(
                user_id=user_id,
                dataset_name=dataset_name,
                permissions=perm_enums,
                granted_by=granted_by,
                expires_at=expires_at,
                conditions=conditions or {},
            )
            
            self._storage.save_grant(grant)
            
            # 记录审计日志
            for perm in perm_enums:
                self._log_access(user_id, dataset_name, perm, "grant")
            
            logger.info(f"Granted {permissions} to {user_id} on {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to grant access: {e}")
            return False
    
    def revoke_access(
        self,
        user_id: str,
        dataset_name: str,
        permissions: Optional[List[str]] = None,
    ) -> bool:
        """撤销访问权限"""
        try:
            grant = self._storage.get_grant(user_id, dataset_name)
            
            if not grant:
                return True  # 没有授权，无需撤销
            
            if permissions is None:
                # 撤销所有权限
                self._storage.delete_grant(user_id, dataset_name)
                for perm in grant.permissions:
                    self._log_access(user_id, dataset_name, perm, "revoke")
            else:
                # 撤销指定权限
                perm_enums = [Permission(p) for p in permissions]
                grant.permissions = [p for p in grant.permissions if p not in perm_enums]
                
                if grant.permissions:
                    self._storage.save_grant(grant)
                else:
                    self._storage.delete_grant(user_id, dataset_name)
                
                for perm in perm_enums:
                    self._log_access(user_id, dataset_name, perm, "revoke")
            
            logger.info(f"Revoked {permissions or 'all'} from {user_id} on {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke access: {e}")
            return False
    
    def check_access(
        self,
        user_id: str,
        dataset_name: str,
        permission: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        检查访问权限
        
        Args:
            user_id: 用户 ID
            dataset_name: 数据集名称
            permission: 权限
            context: 访问上下文 (用于 ABAC 条件检查)
        """
        grant = self._storage.get_grant(user_id, dataset_name)
        
        if not grant:
            self._log_access(user_id, dataset_name, Permission(permission), "deny")
            return False
        
        # 检查过期
        if grant.expires_at and datetime.now() > grant.expires_at:
            self._log_access(user_id, dataset_name, Permission(permission), "deny")
            return False
        
        # 检查权限
        perm_enum = Permission(permission)
        if perm_enum not in grant.permissions:
            # ADMIN 权限可以做任何事
            if Permission.ADMIN not in grant.permissions:
                self._log_access(user_id, dataset_name, perm_enum, "deny")
                return False
        
        # 检查条件 (ABAC)
        if grant.conditions and context:
            if not self._check_conditions(grant.conditions, context):
                self._log_access(user_id, dataset_name, perm_enum, "deny")
                return False
        
        self._log_access(user_id, dataset_name, perm_enum, "access")
        return True
    
    def _check_conditions(
        self,
        conditions: Dict[str, Any],
        context: Dict[str, Any],
    ) -> bool:
        """检查 ABAC 条件"""
        for key, expected in conditions.items():
            actual = context.get(key)
            
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        
        return True
    
    def _log_access(
        self,
        user_id: str,
        dataset_name: str,
        permission: Permission,
        action: str,
    ) -> None:
        """记录访问日志"""
        log = AccessLog(
            user_id=user_id,
            dataset_name=dataset_name,
            permission=permission,
            action=action,
        )
        self._audit_log.append(log)
    
    def get_access_log(
        self,
        dataset_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """获取访问日志"""
        return [
            {
                'user_id': log.user_id,
                'dataset_name': log.dataset_name,
                'permission': log.permission.value,
                'action': log.action,
                'timestamp': log.timestamp.isoformat(),
            }
            for log in self._audit_log
            if (log.dataset_name == dataset_name and
                start_time <= log.timestamp <= end_time)
        ]
    
    def get_user_permissions(
        self,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """获取用户所有权限"""
        grants = self._storage.get_grants_by_user(user_id)
        
        return [
            {
                'dataset_name': g.dataset_name,
                'permissions': [p.value for p in g.permissions],
                'granted_at': g.granted_at.isoformat(),
                'expires_at': g.expires_at.isoformat() if g.expires_at else None,
            }
            for g in grants
        ]


class InMemoryAccessStorage:
    """内存访问存储"""
    
    def __init__(self):
        self._grants: Dict[str, AccessGrant] = {}
    
    def _make_key(self, user_id: str, dataset_name: str) -> str:
        return f"{user_id}:{dataset_name}"
    
    def save_grant(self, grant: AccessGrant) -> None:
        key = self._make_key(grant.user_id, grant.dataset_name)
        self._grants[key] = grant
    
    def get_grant(self, user_id: str, dataset_name: str) -> Optional[AccessGrant]:
        key = self._make_key(user_id, dataset_name)
        return self._grants.get(key)
    
    def delete_grant(self, user_id: str, dataset_name: str) -> None:
        key = self._make_key(user_id, dataset_name)
        self._grants.pop(key, None)
    
    def get_grants_by_user(self, user_id: str) -> List[AccessGrant]:
        return [g for g in self._grants.values() if g.user_id == user_id]
```

---

## 注意事项

1. 数据目录支持全文搜索
2. 血缘追踪支持深度遍历
3. 访问控制支持 RBAC + ABAC
4. 所有操作记录审计日志
5. 支持权限过期和条件访问

## 输出要求

请输出完整的可运行代码，包含：
1. 数据目录
2. 数据血缘
3. 访问控制
4. 审计日志
5. 完整测试

