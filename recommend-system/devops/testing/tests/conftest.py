"""
Pytest 配置和共享 Fixtures
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'load', 'locust'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reports'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmark'))


# =============================================================================
# 测试数据 Fixtures
# =============================================================================

@pytest.fixture
def sample_k6_data():
    """K6 测试数据"""
    return {
        "state": {
            "testRunDurationMs": 300000,
            "isStdErrTty": False,
        },
        "metrics": {
            "http_reqs": {
                "values": {
                    "count": 10000,
                    "rate": 33.33,
                }
            },
            "http_req_duration": {
                "values": {
                    "avg": 45.5,
                    "min": 10.0,
                    "max": 500.0,
                    "p(50)": 40.0,
                    "p(90)": 80.0,
                    "p(95)": 120.0,
                    "p(99)": 180.0,
                },
                "thresholds": {
                    "p(95)<200": {"ok": True},
                }
            },
            "http_req_failed": {
                "values": {
                    "rate": 0.0005,
                }
            },
            "vus_max": {
                "values": {
                    "max": 100,
                }
            }
        }
    }


@pytest.fixture
def sample_locust_data():
    """Locust 测试数据"""
    return {
        "start_time": 1704067200,
        "last_request_timestamp": 1704067500,
        "stats": [
            {
                "name": "Aggregated",
                "num_requests": 5000,
                "num_failures": 5,
                "total_rps": 16.67,
                "avg_response_time": 55.0,
                "min_response_time": 15.0,
                "max_response_time": 450.0,
                "response_time_percentile_50": 45.0,
                "response_time_percentile_90": 90.0,
                "response_time_percentile_95": 130.0,
                "response_time_percentile_99": 200.0,
            }
        ]
    }


@pytest.fixture
def temp_k6_file(sample_k6_data):
    """创建临时 K6 JSON 文件"""
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='_baseline.json',
        delete=False
    ) as f:
        json.dump(sample_k6_data, f)
        yield f.name
    
    # 清理
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def temp_locust_file(sample_locust_data):
    """创建临时 Locust JSON 文件"""
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='_locust.json',
        delete=False
    ) as f:
        json.dump(sample_locust_data, f)
        yield f.name
    
    # 清理
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def temp_output_dir():
    """创建临时输出目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # 清理
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


# =============================================================================
# 环境变量 Fixtures
# =============================================================================

@pytest.fixture
def clean_env():
    """清理测试相关环境变量"""
    env_vars = ['TEST_ENV', 'API_KEY', 'BASE_URL']
    original = {k: os.environ.get(k) for k in env_vars}
    
    # 清理
    for k in env_vars:
        if k in os.environ:
            del os.environ[k]
    
    yield
    
    # 恢复
    for k, v in original.items():
        if v is not None:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]


@pytest.fixture
def mock_env():
    """模拟测试环境变量"""
    os.environ['TEST_ENV'] = 'local'
    os.environ['API_KEY'] = 'test-api-key'
    os.environ['BASE_URL'] = 'http://localhost:8080'
    
    yield
    
    del os.environ['TEST_ENV']
    del os.environ['API_KEY']
    del os.environ['BASE_URL']


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_http_response():
    """模拟 HTTP 响应"""
    from unittest.mock import MagicMock
    
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": {
            "recommendations": [
                {"item_id": "item_1", "score": 0.95},
                {"item_id": "item_2", "score": 0.90},
            ]
        }
    }
    response.elapsed.total_seconds.return_value = 0.05
    
    return response


@pytest.fixture
def mock_error_response():
    """模拟错误 HTTP 响应"""
    from unittest.mock import MagicMock
    
    response = MagicMock()
    response.status_code = 500
    response.json.return_value = {"error": "Internal Server Error"}
    response.elapsed.total_seconds.return_value = 0.1
    
    return response


# =============================================================================
# 辅助函数
# =============================================================================

def pytest_configure(config):
    """Pytest 配置"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    if config.getoption("-m"):
        return
    
    # 默认跳过慢速测试，除非明确指定
    skip_slow = pytest.mark.skip(reason="use -m slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

