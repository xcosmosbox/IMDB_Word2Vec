"""
pytest 配置文件

提供测试 fixtures 和配置
"""

import os
import sys
from pathlib import Path

import pytest

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def logging_dir():
    """返回日志系统目录路径"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def interfaces_path(logging_dir):
    """返回接口定义文件路径"""
    return logging_dir.parent / "interfaces.yaml"


@pytest.fixture(scope="session")
def loki_config_path(logging_dir):
    """返回 Loki 配置文件路径"""
    return logging_dir / "loki" / "loki-config.yaml"


@pytest.fixture(scope="session")
def promtail_config_path(logging_dir):
    """返回 Promtail 配置文件路径"""
    return logging_dir / "promtail" / "promtail-config.yaml"


@pytest.fixture(scope="session")
def fluentd_config_path(logging_dir):
    """返回 Fluentd 配置文件路径"""
    return logging_dir / "fluentd" / "fluent.conf"


@pytest.fixture(scope="session")
def alert_rules_path(logging_dir):
    """返回告警规则文件路径"""
    return logging_dir / "loki" / "rules" / "alerts.yaml"


@pytest.fixture(scope="session")
def dashboards_dir(logging_dir):
    """返回 Grafana Dashboard 目录路径"""
    return logging_dir / "grafana" / "dashboards"


# =============================================================================
# 测试数据
# =============================================================================

@pytest.fixture
def sample_valid_log():
    """返回有效的日志记录样本"""
    return {
        'timestamp': '2025-01-05T10:30:00.123456Z',
        'level': 'INFO',
        'service': 'recommend-service',
        'trace_id': 'trace-abc-123',
        'message': 'Processing recommendation request',
        'user_id': 'user_12345',
        'request_id': 'req_67890',
        'duration_ms': 45.2
    }


@pytest.fixture
def sample_error_log():
    """返回错误日志记录样本"""
    return {
        'timestamp': '2025-01-05T10:30:00.123456Z',
        'level': 'ERROR',
        'service': 'recommend-service',
        'trace_id': 'trace-def-456',
        'message': 'Database connection failed',
        'error_stack': 'ConnectionError: timeout\n  at connect(db.go:150)'
    }


@pytest.fixture
def sample_inference_log():
    """返回推理日志记录样本"""
    return {
        'timestamp': '2025-01-05T10:30:00.123456Z',
        'level': 'INFO',
        'service': 'ugt-inference',
        'trace_id': 'trace-ghi-789',
        'message': 'Inference completed',
        'model': 'ugt-v1',
        'batch_size': 32,
        'latency_ms': 85.5,
        'gpu_memory_mb': 8192,
        'status': 'success'
    }


# =============================================================================
# 测试配置
# =============================================================================

def pytest_configure(config):
    """pytest 配置钩子"""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    # 如果没有指定 -m 选项，跳过 slow 测试
    if not config.option.markexpr:
        skip_slow = pytest.mark.skip(reason="需要 -m slow 选项来运行慢测试")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

