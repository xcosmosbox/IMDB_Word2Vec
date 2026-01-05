"""
Locust 配置模块单元测试
"""

import os
import sys
import pytest
from unittest.mock import patch

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'load', 'locust'))

from config import (
    Environment,
    ServiceConfig,
    EnvironmentConfig,
    SLAConfig,
    LoadScenario,
    APIEndpoints,
    TestDataConfig,
    ENVIRONMENTS,
    SLA,
    LOAD_SCENARIOS,
    ENDPOINTS,
    TEST_DATA,
    get_environment,
    get_auth_headers,
    DEFAULT_HEADERS,
)


class TestServiceConfig:
    """ServiceConfig 测试"""
    
    def test_create_service_config(self):
        """测试创建服务配置"""
        config = ServiceConfig(
            host="localhost",
            http_port=8080,
            grpc_port=9090,
            metrics_port=9091
        )
        
        assert config.host == "localhost"
        assert config.http_port == 8080
        assert config.grpc_port == 9090
        assert config.metrics_port == 9091
    
    def test_service_config_optional_ports(self):
        """测试可选端口"""
        config = ServiceConfig(host="localhost", http_port=8080)
        
        assert config.grpc_port is None
        assert config.metrics_port is None


class TestEnvironmentConfig:
    """EnvironmentConfig 测试"""
    
    def test_create_environment_config(self):
        """测试创建环境配置"""
        recommend_service = ServiceConfig("localhost", 8080)
        user_service = ServiceConfig("localhost", 8081)
        item_service = ServiceConfig("localhost", 8082)
        inference_service = ServiceConfig("localhost", 50051)
        
        config = EnvironmentConfig(
            name="test",
            recommend_service=recommend_service,
            user_service=user_service,
            item_service=item_service,
            inference_service=inference_service,
        )
        
        assert config.name == "test"
        assert config.recommend_service.host == "localhost"
    
    def test_base_url_property(self):
        """测试 base_url 属性"""
        config = ENVIRONMENTS["local"]
        expected_url = f"http://{config.recommend_service.host}:{config.recommend_service.http_port}"
        
        assert config.base_url == expected_url


class TestEnvironments:
    """环境配置字典测试"""
    
    def test_environments_exist(self):
        """测试环境配置存在"""
        assert "local" in ENVIRONMENTS
        assert "dev" in ENVIRONMENTS
        assert "prod" in ENVIRONMENTS
    
    def test_local_environment(self):
        """测试本地环境配置"""
        local = ENVIRONMENTS["local"]
        
        assert local.name == "local"
        assert local.recommend_service.host == "localhost"
        assert local.recommend_service.http_port == 8080
    
    def test_dev_environment(self):
        """测试开发环境配置"""
        dev = ENVIRONMENTS["dev"]
        
        assert dev.name == "dev"
        assert "recommend-dev" in dev.recommend_service.host


class TestGetEnvironment:
    """get_environment 函数测试"""
    
    def test_default_environment(self):
        """测试默认环境"""
        with patch.dict(os.environ, {}, clear=True):
            env = get_environment()
            assert env.name == "local"
    
    def test_environment_from_env_var(self):
        """测试从环境变量获取环境"""
        with patch.dict(os.environ, {"TEST_ENV": "dev"}):
            env = get_environment()
            assert env.name == "dev"
    
    def test_unknown_environment_fallback(self):
        """测试未知环境回退到 local"""
        with patch.dict(os.environ, {"TEST_ENV": "unknown"}):
            env = get_environment()
            assert env.name == "local"


class TestSLAConfig:
    """SLAConfig 测试"""
    
    def test_default_sla_values(self):
        """测试默认 SLA 值"""
        assert SLA.availability == 0.999
        assert SLA.p50_latency == 50
        assert SLA.p99_latency == 200
        assert SLA.error_rate == 0.001
    
    def test_create_custom_sla(self):
        """测试创建自定义 SLA"""
        custom_sla = SLAConfig(
            availability=0.99,
            p50_latency=100,
            p99_latency=500,
            error_rate=0.01,
        )
        
        assert custom_sla.availability == 0.99
        assert custom_sla.p50_latency == 100


class TestLoadScenario:
    """LoadScenario 测试"""
    
    def test_baseline_scenario(self):
        """测试基线场景配置"""
        baseline = LOAD_SCENARIOS["baseline"]
        
        assert baseline.name == "baseline"
        assert baseline.rps_target == 100
        assert baseline.duration == "5m"
    
    def test_stress_scenario(self):
        """测试压力测试场景配置"""
        stress = LOAD_SCENARIOS["stress"]
        
        assert stress.name == "stress"
        assert stress.rps_target == 1000
        assert stress.users == 1000
    
    def test_spike_scenario(self):
        """测试峰值测试场景配置"""
        spike = LOAD_SCENARIOS["spike"]
        
        assert spike.name == "spike"
        assert spike.rps_target == 5000


class TestAPIEndpoints:
    """APIEndpoints 测试"""
    
    def test_health_endpoint(self):
        """测试健康检查端点"""
        assert ENDPOINTS.health == "/health"
        assert ENDPOINTS.ready == "/ready"
    
    def test_recommend_endpoints(self):
        """测试推荐端点"""
        assert ENDPOINTS.recommend == "/api/v1/recommend"
        assert ENDPOINTS.recommend_similar == "/api/v1/recommend/similar"
    
    def test_user_endpoints(self):
        """测试用户端点"""
        assert "{user_id}" in ENDPOINTS.user_profile
        assert "{user_id}" in ENDPOINTS.user_history
    
    def test_item_endpoints(self):
        """测试物品端点"""
        assert "{item_id}" in ENDPOINTS.item_detail
        assert ENDPOINTS.item_search == "/api/v1/items/search"


class TestTestDataConfig:
    """TestDataConfig 测试"""
    
    def test_user_id_range(self):
        """测试用户 ID 范围"""
        assert TEST_DATA.user_id_min == 1
        assert TEST_DATA.user_id_max == 100000
    
    def test_item_id_range(self):
        """测试物品 ID 范围"""
        assert TEST_DATA.item_id_min == 1
        assert TEST_DATA.item_id_max == 1000000
    
    def test_search_terms(self):
        """测试搜索词列表"""
        assert len(TEST_DATA.search_terms) > 0
        assert "action" in TEST_DATA.search_terms
    
    def test_feedback_actions(self):
        """测试反馈动作列表"""
        assert "click" in TEST_DATA.feedback_actions
        assert "view" in TEST_DATA.feedback_actions
        assert "like" in TEST_DATA.feedback_actions


class TestGetAuthHeaders:
    """get_auth_headers 函数测试"""
    
    def test_default_headers(self):
        """测试默认请求头"""
        headers = get_auth_headers()
        
        assert "Content-Type" in headers
        assert "Authorization" in headers
        assert headers["Content-Type"] == "application/json"
    
    def test_custom_api_key(self):
        """测试自定义 API 密钥"""
        headers = get_auth_headers("custom-key")
        
        assert "Bearer custom-key" in headers["Authorization"]
    
    def test_api_key_from_env(self):
        """测试从环境变量获取 API 密钥"""
        with patch.dict(os.environ, {"API_KEY": "env-api-key"}):
            headers = get_auth_headers()
            assert "Bearer env-api-key" in headers["Authorization"]


class TestDefaultHeaders:
    """DEFAULT_HEADERS 测试"""
    
    def test_content_type(self):
        """测试 Content-Type"""
        assert DEFAULT_HEADERS["Content-Type"] == "application/json"
    
    def test_accept(self):
        """测试 Accept"""
        assert DEFAULT_HEADERS["Accept"] == "application/json"
    
    def test_user_agent(self):
        """测试 User-Agent"""
        assert "Locust" in DEFAULT_HEADERS["User-Agent"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

