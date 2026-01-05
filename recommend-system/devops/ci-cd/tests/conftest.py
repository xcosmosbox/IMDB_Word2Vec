"""
pytest 配置文件

提供测试夹具和公共配置
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_project_structure(temp_dir):
    """创建模拟的项目结构"""
    # 创建目录结构
    dirs = [
        "cmd/recommend-service",
        "cmd/user-service",
        "cmd/item-service",
        "algorithm",
        "frontend/user-app",
        "frontend/admin",
        "deployments/docker",
        "devops/kubernetes/base",
        "devops/kubernetes/overlays/dev",
        "devops/kubernetes/overlays/prod",
        "bin",
    ]
    
    for d in dirs:
        (temp_dir / d).mkdir(parents=True, exist_ok=True)
    
    # 创建模拟文件
    files = {
        "go.mod": "module recommend-system\ngo 1.21\n",
        "cmd/recommend-service/main.go": "package main\nfunc main() {}\n",
        "cmd/user-service/main.go": "package main\nfunc main() {}\n",
        "cmd/item-service/main.go": "package main\nfunc main() {}\n",
        "algorithm/requirements.txt": "torch>=2.0.0\n",
        "frontend/user-app/package.json": '{"name": "user-app", "version": "1.0.0"}\n',
        "frontend/admin/package.json": '{"name": "admin", "version": "1.0.0"}\n',
        "deployments/docker/Dockerfile": "FROM golang:1.21\n",
    }
    
    for path, content in files.items():
        file_path = temp_dir / path
        file_path.write_text(content)
    
    return temp_dir


@pytest.fixture
def workflows_dir():
    """返回工作流目录路径"""
    return Path(__file__).parent.parent / ".github" / "workflows"


@pytest.fixture
def scripts_dir():
    """返回脚本目录路径"""
    return Path(__file__).parent.parent / "scripts"


def pytest_configure(config):
    """pytest 配置"""
    config.addinivalue_line(
        "markers", "integration: 集成测试标记"
    )
    config.addinivalue_line(
        "markers", "slow: 慢速测试标记"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集行为"""
    # 根据标记跳过测试
    if not config.getoption("--run-slow", default=False):
        skip_slow = pytest.mark.skip(reason="需要 --run-slow 选项运行")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="运行慢速测试"
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="运行集成测试"
    )

