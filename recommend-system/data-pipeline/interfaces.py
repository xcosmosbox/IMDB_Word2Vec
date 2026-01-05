"""
数据管道层接口定义

这是数据管道层的核心接口定义文件，所有模块开发者必须遵循这些接口。
通过接口驱动开发，实现模块间解耦和可插拔设计。

使用方式：
    - 各模块实现对应接口
    - 依赖注入时使用接口类型
    - 测试时使用 Mock 实现
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np


# =============================================================================
# 基础数据类型
# =============================================================================

class EventType(Enum):
    """事件类型"""
    VIEW = "view"
    CLICK = "click"
    LIKE = "like"
    DISLIKE = "dislike"
    PURCHASE = "purchase"
    SHARE = "share"
    RATING = "rating"
    SEARCH = "search"
    ADD_TO_CART = "add_to_cart"


class DataQualityLevel(Enum):
    """数据质量级别"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RawEvent:
    """原始事件"""
    event_id: str
    event_type: EventType
    user_id: str
    item_id: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedEvent:
    """处理后的事件"""
    event_id: str
    event_type: EventType
    user_id: str
    item_id: str
    timestamp: datetime
    features: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    quality_score: float = 1.0


@dataclass
class Feature:
    """特征定义"""
    name: str
    dtype: str
    description: str
    source: str
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)


@dataclass
class FeatureValue:
    """特征值"""
    feature_name: str
    value: Any
    timestamp: datetime
    entity_id: str
    entity_type: str  # "user" | "item"


@dataclass
class DataQualityReport:
    """数据质量报告"""
    check_name: str
    level: DataQualityLevel
    passed: bool
    message: str
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LineageNode:
    """血缘节点"""
    node_id: str
    node_type: str  # "source" | "transform" | "sink"
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageEdge:
    """血缘边"""
    source_id: str
    target_id: str
    transform_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 数据采集接口 (Person A)
# =============================================================================

class EventCollectorInterface(ABC):
    """
    事件采集器接口
    
    负责从各种来源采集原始事件
    """
    
    @abstractmethod
    def collect(self) -> Generator[RawEvent, None, None]:
        """
        采集事件流
        
        Yields:
            RawEvent: 原始事件
        """
        pass
    
    @abstractmethod
    def validate_event(self, event: RawEvent) -> bool:
        """验证事件格式"""
        pass
    
    @abstractmethod
    def get_offset(self) -> str:
        """获取当前消费位置"""
        pass
    
    @abstractmethod
    def commit_offset(self, offset: str) -> None:
        """提交消费位置"""
        pass


class EventPublisherInterface(ABC):
    """
    事件发布器接口
    
    负责将事件发布到消息队列
    """
    
    @abstractmethod
    def publish(self, event: RawEvent) -> bool:
        """发布事件"""
        pass
    
    @abstractmethod
    def publish_batch(self, events: List[RawEvent]) -> Tuple[int, int]:
        """
        批量发布事件
        
        Returns:
            Tuple[int, int]: (成功数, 失败数)
        """
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """刷新缓冲区"""
        pass


# =============================================================================
# ETL 接口 (Person B)
# =============================================================================

class ExtractorInterface(ABC):
    """
    数据抽取器接口
    """
    
    @abstractmethod
    def extract(
        self,
        source: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        抽取数据
        
        Args:
            source: 数据源标识
            start_time: 开始时间
            end_time: 结束时间
            
        Yields:
            Dict: 原始数据记录
        """
        pass
    
    @abstractmethod
    def get_schema(self, source: str) -> Dict[str, str]:
        """获取数据源 Schema"""
        pass


class TransformerInterface(ABC):
    """
    数据转换器接口
    """
    
    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换单条数据
        
        Args:
            data: 输入数据
            
        Returns:
            Dict: 转换后的数据
        """
        pass
    
    @abstractmethod
    def transform_batch(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """批量转换"""
        pass
    
    @abstractmethod
    def get_output_schema(self) -> Dict[str, str]:
        """获取输出 Schema"""
        pass


class LoaderInterface(ABC):
    """
    数据加载器接口
    """
    
    @abstractmethod
    def load(self, data: Dict[str, Any]) -> bool:
        """加载单条数据"""
        pass
    
    @abstractmethod
    def load_batch(self, data: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        批量加载
        
        Returns:
            Tuple[int, int]: (成功数, 失败数)
        """
        pass
    
    @abstractmethod
    def create_table_if_not_exists(self, schema: Dict[str, str]) -> bool:
        """创建目标表"""
        pass


class ETLPipelineInterface(ABC):
    """
    ETL 管道接口
    """
    
    @abstractmethod
    def run(
        self,
        source: str,
        target: str,
        transformers: List[TransformerInterface],
    ) -> Dict[str, Any]:
        """
        运行 ETL 管道
        
        Returns:
            Dict: 执行统计信息
        """
        pass
    
    @abstractmethod
    def schedule(self, cron_expression: str) -> str:
        """调度管道"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取管道状态"""
        pass


# =============================================================================
# 特征工程接口 (Person C)
# =============================================================================

class FeatureTransformerInterface(ABC):
    """
    特征转换器接口
    """
    
    @abstractmethod
    def fit(self, data: List[Dict[str, Any]]) -> None:
        """训练转换器"""
        pass
    
    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """转换特征"""
        pass
    
    @abstractmethod
    def fit_transform(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """训练并转换"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """保存转换器状态"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """加载转换器状态"""
        pass


class FeaturePipelineInterface(ABC):
    """
    特征管道接口
    """
    
    @abstractmethod
    def add_transformer(
        self,
        name: str,
        transformer: FeatureTransformerInterface
    ) -> None:
        """添加转换器"""
        pass
    
    @abstractmethod
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """运行管道"""
        pass
    
    @abstractmethod
    def run_batch(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """批量运行"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """获取输出特征名"""
        pass


# =============================================================================
# 特征存储接口 (Person D)
# =============================================================================

class OnlineFeatureStoreInterface(ABC):
    """
    在线特征存储接口
    
    低延迟读取，用于实时推理
    """
    
    @abstractmethod
    def get_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        获取实体特征
        
        Args:
            entity_type: "user" 或 "item"
            entity_id: 实体 ID
            feature_names: 特征名列表
            
        Returns:
            Dict: 特征名 -> 特征值
        """
        pass
    
    @abstractmethod
    def get_features_batch(
        self,
        entity_type: str,
        entity_ids: List[str],
        feature_names: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """批量获取特征"""
        pass
    
    @abstractmethod
    def set_features(
        self,
        entity_type: str,
        entity_id: str,
        features: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """设置特征"""
        pass
    
    @abstractmethod
    def delete_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
    ) -> bool:
        """删除特征"""
        pass


class OfflineFeatureStoreInterface(ABC):
    """
    离线特征存储接口
    
    用于特征历史存储和训练数据生成
    """
    
    @abstractmethod
    def write_features(
        self,
        features: List[FeatureValue],
        table_name: str,
    ) -> int:
        """写入特征"""
        pass
    
    @abstractmethod
    def read_features(
        self,
        entity_type: str,
        entity_ids: List[str],
        feature_names: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[FeatureValue]:
        """读取历史特征"""
        pass
    
    @abstractmethod
    def generate_training_data(
        self,
        label_table: str,
        feature_tables: List[str],
        output_path: str,
    ) -> str:
        """生成训练数据集"""
        pass
    
    @abstractmethod
    def get_feature_statistics(
        self,
        feature_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, float]:
        """获取特征统计信息"""
        pass


# =============================================================================
# 数据质量接口 (Person E)
# =============================================================================

class DataValidatorInterface(ABC):
    """
    数据验证器接口
    """
    
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> DataQualityReport:
        """验证单条数据"""
        pass
    
    @abstractmethod
    def validate_batch(
        self,
        data: List[Dict[str, Any]]
    ) -> List[DataQualityReport]:
        """批量验证"""
        pass
    
    @abstractmethod
    def add_rule(
        self,
        rule_name: str,
        rule_func: callable,
        level: DataQualityLevel,
    ) -> None:
        """添加验证规则"""
        pass
    
    @abstractmethod
    def get_rules(self) -> List[str]:
        """获取所有规则"""
        pass


class DataQualityMonitorInterface(ABC):
    """
    数据质量监控接口
    """
    
    @abstractmethod
    def start_monitoring(self) -> None:
        """启动监控"""
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> None:
        """停止监控"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """获取质量指标"""
        pass
    
    @abstractmethod
    def set_alert_threshold(
        self,
        metric_name: str,
        threshold: float,
        comparison: str,  # "gt" | "lt" | "eq"
    ) -> None:
        """设置告警阈值"""
        pass
    
    @abstractmethod
    def get_alerts(self) -> List[Dict[str, Any]]:
        """获取告警"""
        pass


class DataProfilerInterface(ABC):
    """
    数据剖析器接口
    """
    
    @abstractmethod
    def profile(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        剖析数据
        
        Returns:
            Dict: 包含统计信息、分布、异常检测等
        """
        pass
    
    @abstractmethod
    def compare_profiles(
        self,
        profile1: Dict[str, Any],
        profile2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """比较两个数据剖析结果"""
        pass
    
    @abstractmethod
    def detect_drift(
        self,
        baseline_profile: Dict[str, Any],
        current_profile: Dict[str, Any],
    ) -> Dict[str, float]:
        """检测数据漂移"""
        pass


# =============================================================================
# 数据治理接口 (Person F)
# =============================================================================

class DataCatalogInterface(ABC):
    """
    数据目录接口
    """
    
    @abstractmethod
    def register_dataset(
        self,
        name: str,
        schema: Dict[str, str],
        metadata: Dict[str, Any],
    ) -> str:
        """注册数据集"""
        pass
    
    @abstractmethod
    def get_dataset(self, name: str) -> Dict[str, Any]:
        """获取数据集信息"""
        pass
    
    @abstractmethod
    def search_datasets(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """搜索数据集"""
        pass
    
    @abstractmethod
    def add_tags(self, dataset_name: str, tags: List[str]) -> None:
        """添加标签"""
        pass
    
    @abstractmethod
    def update_metadata(
        self,
        dataset_name: str,
        metadata: Dict[str, Any],
    ) -> None:
        """更新元数据"""
        pass


class DataLineageInterface(ABC):
    """
    数据血缘接口
    """
    
    @abstractmethod
    def record_lineage(
        self,
        source_nodes: List[LineageNode],
        target_node: LineageNode,
        edges: List[LineageEdge],
    ) -> str:
        """记录血缘关系"""
        pass
    
    @abstractmethod
    def get_upstream(
        self,
        node_id: str,
        depth: int = 1,
    ) -> Tuple[List[LineageNode], List[LineageEdge]]:
        """获取上游节点"""
        pass
    
    @abstractmethod
    def get_downstream(
        self,
        node_id: str,
        depth: int = 1,
    ) -> Tuple[List[LineageNode], List[LineageEdge]]:
        """获取下游节点"""
        pass
    
    @abstractmethod
    def get_impact_analysis(
        self,
        node_id: str,
    ) -> Dict[str, Any]:
        """影响分析"""
        pass
    
    @abstractmethod
    def visualize(self, node_id: str) -> str:
        """生成血缘可视化"""
        pass


class DataAccessControlInterface(ABC):
    """
    数据访问控制接口
    """
    
    @abstractmethod
    def grant_access(
        self,
        user_id: str,
        dataset_name: str,
        permissions: List[str],  # ["read", "write", "admin"]
    ) -> bool:
        """授予访问权限"""
        pass
    
    @abstractmethod
    def revoke_access(
        self,
        user_id: str,
        dataset_name: str,
        permissions: Optional[List[str]] = None,
    ) -> bool:
        """撤销访问权限"""
        pass
    
    @abstractmethod
    def check_access(
        self,
        user_id: str,
        dataset_name: str,
        permission: str,
    ) -> bool:
        """检查访问权限"""
        pass
    
    @abstractmethod
    def get_access_log(
        self,
        dataset_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """获取访问日志"""
        pass

