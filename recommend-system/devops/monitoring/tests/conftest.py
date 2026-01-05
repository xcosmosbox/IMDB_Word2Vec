"""
Pytest 配置文件

提供测试 fixtures 和通用配置
"""

import os
import sys
from pathlib import Path

import pytest

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def monitoring_base_path():
    """返回监控配置根目录"""
    return Path(__file__).parent.parent


@pytest.fixture
def prometheus_path(monitoring_base_path):
    """返回 Prometheus 配置目录"""
    return monitoring_base_path / "prometheus"


@pytest.fixture
def grafana_path(monitoring_base_path):
    """返回 Grafana 配置目录"""
    return monitoring_base_path / "grafana"


@pytest.fixture
def alertmanager_path(monitoring_base_path):
    """返回 AlertManager 配置目录"""
    return monitoring_base_path / "alertmanager"


@pytest.fixture
def interfaces_config():
    """加载接口定义"""
    interfaces_file = Path(__file__).parent.parent.parent / "interfaces.yaml"
    if interfaces_file.exists():
        import yaml
        with open(interfaces_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return None

