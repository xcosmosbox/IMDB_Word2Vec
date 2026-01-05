#!/usr/bin/env python3
"""
GitHub Actions Workflow 配置验证测试

测试工作流配置的正确性和完整性
运行: pytest tests/test_workflows.py -v
"""

import os
import pytest
import yaml
from pathlib import Path


# 获取工作流目录
WORKFLOWS_DIR = Path(__file__).parent.parent / ".github" / "workflows"


def load_workflow(name: str) -> dict:
    """加载工作流文件"""
    workflow_path = WORKFLOWS_DIR / name
    if not workflow_path.exists():
        pytest.skip(f"工作流文件不存在: {name}")
    
    with open(workflow_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestCIWorkflow:
    """CI 工作流测试"""
    
    @pytest.fixture
    def workflow(self):
        return load_workflow("ci.yaml")
    
    def test_workflow_name(self, workflow):
        """测试工作流名称"""
        assert workflow["name"] == "CI"
    
    def test_trigger_events(self, workflow):
        """测试触发事件"""
        assert "on" in workflow
        on = workflow["on"]
        assert "push" in on
        assert "pull_request" in on
    
    def test_push_branches(self, workflow):
        """测试推送分支"""
        branches = workflow["on"]["push"]["branches"]
        assert "main" in branches
        assert "develop" in branches
    
    def test_required_env_variables(self, workflow):
        """测试必需的环境变量"""
        env = workflow.get("env", {})
        assert "GO_VERSION" in env
        assert "PYTHON_VERSION" in env
        assert "NODE_VERSION" in env
    
    def test_required_jobs(self, workflow):
        """测试必需的任务"""
        jobs = workflow["jobs"]
        required_jobs = ["go-lint", "go-test", "go-build", 
                        "python-lint", "python-test",
                        "frontend-lint", "frontend-test", "frontend-build"]
        for job in required_jobs:
            assert job in jobs, f"缺少必需的任务: {job}"
    
    def test_go_test_services(self, workflow):
        """测试 Go 测试任务的服务配置"""
        go_test = workflow["jobs"]["go-test"]
        services = go_test.get("services", {})
        assert "postgres" in services
        assert "redis" in services
    
    def test_job_dependencies(self, workflow):
        """测试任务依赖关系"""
        jobs = workflow["jobs"]
        
        # go-test 依赖 go-lint
        assert "go-lint" in jobs["go-test"].get("needs", [])
        
        # go-build 依赖 go-test
        assert "go-test" in jobs["go-build"].get("needs", [])


class TestCDDevWorkflow:
    """开发环境部署工作流测试"""
    
    @pytest.fixture
    def workflow(self):
        return load_workflow("cd-dev.yaml")
    
    def test_workflow_name(self, workflow):
        """测试工作流名称"""
        assert "Development" in workflow["name"]
    
    def test_trigger_on_develop_branch(self, workflow):
        """测试在 develop 分支触发"""
        on = workflow["on"]
        if "push" in on:
            assert "develop" in on["push"]["branches"]
    
    def test_has_deploy_job(self, workflow):
        """测试部署任务存在"""
        assert "deploy" in workflow["jobs"]
    
    def test_deploy_uses_kubectl(self, workflow):
        """测试部署使用 kubectl"""
        deploy = workflow["jobs"]["deploy"]
        steps = deploy["steps"]
        
        kubectl_setup = any(
            "azure/setup-kubectl" in step.get("uses", "") 
            for step in steps
        )
        assert kubectl_setup, "部署任务应该包含 kubectl 设置"
    
    def test_slack_notification(self, workflow):
        """测试 Slack 通知"""
        deploy = workflow["jobs"]["deploy"]
        steps = deploy["steps"]
        
        has_slack = any(
            "action-slack" in step.get("uses", "")
            for step in steps
        )
        assert has_slack, "应该包含 Slack 通知"


class TestCDProdWorkflow:
    """生产环境部署工作流测试"""
    
    @pytest.fixture
    def workflow(self):
        return load_workflow("cd-prod.yaml")
    
    def test_workflow_name(self, workflow):
        """测试工作流名称"""
        assert "Production" in workflow["name"]
    
    def test_trigger_on_release(self, workflow):
        """测试在 release 事件触发"""
        on = workflow["on"]
        assert "release" in on
    
    def test_has_security_scan(self, workflow):
        """测试安全扫描任务"""
        jobs = workflow["jobs"]
        assert "security-scan" in jobs
    
    def test_has_canary_deployment(self, workflow):
        """测试金丝雀部署"""
        jobs = workflow["jobs"]
        assert "canary-deploy" in jobs
    
    def test_has_rollback_job(self, workflow):
        """测试回滚任务"""
        jobs = workflow["jobs"]
        assert "rollback" in jobs
    
    def test_deploy_needs_security_scan(self, workflow):
        """测试部署依赖安全扫描"""
        canary = workflow["jobs"].get("canary-deploy", {})
        needs = canary.get("needs", [])
        
        # 检查是否有安全相关的前置任务
        has_security_prereq = any(
            "security" in n or "scan" in n or "validation" in n 
            for n in needs
        )
        assert has_security_prereq, "金丝雀部署应该依赖安全扫描"
    
    def test_environment_protection(self, workflow):
        """测试环境保护"""
        deploy = workflow["jobs"].get("full-deploy", {})
        assert deploy.get("environment") == "production"


class TestSecurityScanWorkflow:
    """安全扫描工作流测试"""
    
    @pytest.fixture
    def workflow(self):
        return load_workflow("security-scan.yaml")
    
    def test_workflow_name(self, workflow):
        """测试工作流名称"""
        assert "Security" in workflow["name"]
    
    def test_scheduled_run(self, workflow):
        """测试定时运行"""
        on = workflow["on"]
        assert "schedule" in on
    
    def test_dependency_scans(self, workflow):
        """测试依赖扫描任务"""
        jobs = workflow["jobs"]
        
        # 应该有各语言的依赖扫描
        assert "go-dependencies" in jobs or any("go" in j and "depend" in j for j in jobs)
        assert "python-dependencies" in jobs or any("python" in j and "depend" in j for j in jobs)
    
    def test_code_scan(self, workflow):
        """测试代码扫描"""
        jobs = workflow["jobs"]
        assert "code-scan" in jobs
    
    def test_secret_detection(self, workflow):
        """测试密钥检测"""
        jobs = workflow["jobs"]
        assert "secret-scan" in jobs
    
    def test_container_scan(self, workflow):
        """测试容器扫描"""
        jobs = workflow["jobs"]
        assert "container-scan" in jobs


class TestReleaseWorkflow:
    """版本发布工作流测试"""
    
    @pytest.fixture
    def workflow(self):
        return load_workflow("release.yaml")
    
    def test_workflow_name(self, workflow):
        """测试工作流名称"""
        assert "Release" in workflow["name"]
    
    def test_manual_trigger(self, workflow):
        """测试手动触发"""
        on = workflow["on"]
        assert "workflow_dispatch" in on
    
    def test_version_type_input(self, workflow):
        """测试版本类型输入"""
        inputs = workflow["on"]["workflow_dispatch"]["inputs"]
        assert "version_type" in inputs
        
        options = inputs["version_type"]["options"]
        assert "patch" in options
        assert "minor" in options
        assert "major" in options
    
    def test_has_test_job(self, workflow):
        """测试包含测试任务"""
        jobs = workflow["jobs"]
        assert "test" in jobs
    
    def test_has_build_job(self, workflow):
        """测试包含构建任务"""
        jobs = workflow["jobs"]
        assert "build" in jobs
    
    def test_has_release_job(self, workflow):
        """测试包含发布任务"""
        jobs = workflow["jobs"]
        assert "release" in jobs


class TestWorkflowIntegrity:
    """工作流完整性测试"""
    
    def test_all_workflows_exist(self):
        """测试所有必需的工作流存在"""
        required_workflows = [
            "ci.yaml",
            "cd-dev.yaml",
            "cd-prod.yaml",
            "security-scan.yaml",
            "release.yaml"
        ]
        
        for workflow in required_workflows:
            workflow_path = WORKFLOWS_DIR / workflow
            assert workflow_path.exists(), f"缺少工作流文件: {workflow}"
    
    def test_all_workflows_valid_yaml(self):
        """测试所有工作流是有效的 YAML"""
        for workflow_file in WORKFLOWS_DIR.glob("*.yaml"):
            try:
                with open(workflow_file, "r", encoding="utf-8") as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"无效的 YAML 文件 {workflow_file.name}: {e}")
    
    def test_consistent_secrets_usage(self):
        """测试密钥使用的一致性"""
        required_secrets = [
            "DOCKER_REGISTRY_URL",
            "DOCKER_USERNAME",
            "DOCKER_PASSWORD",
            "KUBECONFIG_DEV",
            "KUBECONFIG_PROD",
            "SLACK_WEBHOOK_URL"
        ]
        
        for workflow_file in WORKFLOWS_DIR.glob("*.yaml"):
            with open(workflow_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 检查使用的密钥格式是否正确
            for secret in required_secrets:
                if secret in content:
                    assert f"secrets.{secret}" in content, \
                        f"{workflow_file.name} 中 {secret} 的引用格式不正确"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

