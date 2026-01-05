"""
Locust 负载测试配置

基于 devops/interfaces.yaml 中定义的 SLA 契约:
- 可用性: 99.9%
- P50 延迟: 50ms
- P99 延迟: 200ms
- 错误率: 0.1%
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Environment(Enum):
    """测试环境枚举"""
    LOCAL = "local"
    DEV = "dev"
    PROD = "prod"


@dataclass
class ServiceConfig:
    """服务配置"""
    host: str
    http_port: int
    grpc_port: Optional[int] = None
    metrics_port: Optional[int] = None


@dataclass
class EnvironmentConfig:
    """环境配置"""
    name: str
    recommend_service: ServiceConfig
    user_service: ServiceConfig
    item_service: ServiceConfig
    inference_service: ServiceConfig
    
    @property
    def base_url(self) -> str:
        return f"http://{self.recommend_service.host}:{self.recommend_service.http_port}"


# ============================================================================
# 环境配置
# ============================================================================

ENVIRONMENTS: Dict[str, EnvironmentConfig] = {
    "local": EnvironmentConfig(
        name="local",
        recommend_service=ServiceConfig("localhost", 8080, 9090, 9091),
        user_service=ServiceConfig("localhost", 8081, 9091, 9092),
        item_service=ServiceConfig("localhost", 8082, 9092, 9093),
        inference_service=ServiceConfig("localhost", 50051, metrics_port=9094),
    ),
    "dev": EnvironmentConfig(
        name="dev",
        recommend_service=ServiceConfig("recommend-service.recommend-dev", 8080, 9090, 9091),
        user_service=ServiceConfig("user-service.recommend-dev", 8081, 9091, 9092),
        item_service=ServiceConfig("item-service.recommend-dev", 8082, 9092, 9093),
        inference_service=ServiceConfig("ugt-inference.recommend-dev", 50051, metrics_port=9094),
    ),
    "prod": EnvironmentConfig(
        name="prod",
        recommend_service=ServiceConfig("recommend-service.recommend-prod", 8080, 9090, 9091),
        user_service=ServiceConfig("user-service.recommend-prod", 8081, 9091, 9092),
        item_service=ServiceConfig("item-service.recommend-prod", 8082, 9092, 9093),
        inference_service=ServiceConfig("ugt-inference.recommend-prod", 50051, metrics_port=9094),
    ),
}


def get_environment() -> EnvironmentConfig:
    """获取当前环境配置"""
    env_name = os.getenv("TEST_ENV", "local")
    return ENVIRONMENTS.get(env_name, ENVIRONMENTS["local"])


# ============================================================================
# SLA 配置 (来自 interfaces.yaml)
# ============================================================================

@dataclass
class SLAConfig:
    """SLA 配置"""
    # 可用性目标
    availability: float = 0.999  # 99.9%
    
    # 延迟目标 (毫秒)
    p50_latency: int = 50
    p90_latency: int = 100
    p95_latency: int = 150
    p99_latency: int = 200
    
    # 错误率目标
    error_rate: float = 0.001  # 0.1%


SLA = SLAConfig()


# ============================================================================
# 负载场景配置 (来自 interfaces.yaml)
# ============================================================================

@dataclass
class LoadScenario:
    """负载场景配置"""
    name: str
    users: int
    spawn_rate: float
    duration: str
    rps_target: int


LOAD_SCENARIOS = {
    "baseline": LoadScenario(
        name="baseline",
        users=100,
        spawn_rate=10,
        duration="5m",
        rps_target=100,
    ),
    "stress": LoadScenario(
        name="stress",
        users=1000,
        spawn_rate=50,
        duration="10m",
        rps_target=1000,
    ),
    "spike": LoadScenario(
        name="spike",
        users=5000,
        spawn_rate=500,
        duration="2m",
        rps_target=5000,
    ),
}


# ============================================================================
# API 端点配置
# ============================================================================

@dataclass
class APIEndpoints:
    """API 端点配置"""
    # 健康检查
    health: str = "/health"
    ready: str = "/ready"
    
    # 推荐服务
    recommend: str = "/api/v1/recommend"
    recommend_personalized: str = "/api/v1/recommend/personalized"
    recommend_similar: str = "/api/v1/recommend/similar"
    recommend_coldstart: str = "/api/v1/recommend/coldstart"
    
    # 用户服务
    user_profile: str = "/api/v1/users/{user_id}"
    user_history: str = "/api/v1/users/{user_id}/history"
    user_preferences: str = "/api/v1/users/{user_id}/preferences"
    
    # 物品服务
    item_detail: str = "/api/v1/items/{item_id}"
    item_search: str = "/api/v1/items/search"
    item_categories: str = "/api/v1/items/categories"
    item_popular: str = "/api/v1/items/popular"
    
    # 反馈服务
    feedback: str = "/api/v1/feedback"
    feedback_batch: str = "/api/v1/feedback/batch"
    
    # 管理 API
    admin_dashboard: str = "/api/admin/v1/dashboard"
    admin_users: str = "/api/admin/v1/users"
    admin_items: str = "/api/admin/v1/items"
    admin_metrics: str = "/api/admin/v1/metrics"


ENDPOINTS = APIEndpoints()


# ============================================================================
# 测试数据配置
# ============================================================================

@dataclass
class TestDataConfig:
    """测试数据配置"""
    # 用户 ID 范围
    user_id_min: int = 1
    user_id_max: int = 100000
    
    # 物品 ID 范围
    item_id_min: int = 1
    item_id_max: int = 1000000
    
    # 搜索关键词
    search_terms: List[str] = field(default_factory=lambda: [
        "action", "comedy", "drama", "thriller", "sci-fi",
        "horror", "romance", "documentary", "animation", "adventure",
        "mystery", "fantasy", "crime", "biography", "history",
    ])
    
    # 推荐场景
    recommend_scenes: List[str] = field(default_factory=lambda: [
        "home", "search", "detail", "cart", "checkout",
    ])
    
    # 反馈动作类型
    feedback_actions: List[str] = field(default_factory=lambda: [
        "view", "click", "like", "share", "purchase", "skip",
    ])
    
    # 分页默认值
    default_limit: int = 20
    max_limit: int = 100


TEST_DATA = TestDataConfig()


# ============================================================================
# 请求头配置
# ============================================================================

DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "Locust-LoadTest/1.0",
}


def get_auth_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """获取认证请求头"""
    headers = DEFAULT_HEADERS.copy()
    token = api_key or os.getenv("API_KEY", "test-api-key")
    headers["Authorization"] = f"Bearer {token}"
    return headers


# ============================================================================
# 任务权重配置
# ============================================================================

@dataclass
class TaskWeights:
    """任务权重配置"""
    # 用户行为任务
    get_recommendations: int = 5
    search_items: int = 3
    submit_feedback: int = 2
    get_item_detail: int = 1
    
    # 管理员任务
    get_dashboard: int = 1
    list_users: int = 1


TASK_WEIGHTS = TaskWeights()


# ============================================================================
# 报告配置
# ============================================================================

@dataclass
class ReportConfig:
    """报告配置"""
    output_dir: str = "./results"
    formats: List[str] = field(default_factory=lambda: ["html", "json", "csv"])
    include_charts: bool = True
    include_percentiles: List[int] = field(default_factory=lambda: [50, 90, 95, 99])


REPORT_CONFIG = ReportConfig()


# ============================================================================
# 日志配置
# ============================================================================

@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None


LOG_CONFIG = LogConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
)

