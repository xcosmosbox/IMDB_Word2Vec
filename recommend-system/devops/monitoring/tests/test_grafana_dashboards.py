"""
Grafana 仪表板验证测试

测试内容:
1. JSON 语法正确性
2. Dashboard 结构完整性
3. Panel 配置有效性
4. 数据源配置正确性
"""

import json
import os
import unittest
from pathlib import Path
from typing import Dict, List

import yaml


class TestGrafanaDatasources(unittest.TestCase):
    """Grafana 数据源配置测试"""

    @classmethod
    def setUpClass(cls):
        """加载配置"""
        cls.base_path = Path(__file__).parent.parent
        cls.datasources_path = cls.base_path / "grafana" / "provisioning" / "datasources"
        
        config_file = cls.datasources_path / "prometheus.yaml"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                cls.config = yaml.safe_load(f)
        else:
            cls.config = None

    def test_datasources_file_exists(self):
        """测试数据源配置文件存在"""
        config_file = self.datasources_path / "prometheus.yaml"
        self.assertTrue(config_file.exists(), "prometheus.yaml 不存在")

    def test_datasources_valid_yaml(self):
        """测试数据源配置是有效的 YAML"""
        self.assertIsNotNone(self.config)

    def test_has_prometheus_datasource(self):
        """测试包含 Prometheus 数据源"""
        datasources = self.config.get('datasources', [])
        prometheus_ds = [ds for ds in datasources if ds.get('type') == 'prometheus']
        self.assertGreater(len(prometheus_ds), 0, "缺少 Prometheus 数据源")

    def test_prometheus_is_default(self):
        """测试 Prometheus 是默认数据源"""
        datasources = self.config.get('datasources', [])
        for ds in datasources:
            if ds.get('name') == 'Prometheus':
                self.assertTrue(ds.get('isDefault', False))
                break


class TestGrafanaDashboardProvisioning(unittest.TestCase):
    """Grafana 仪表板 Provisioning 配置测试"""

    @classmethod
    def setUpClass(cls):
        """加载配置"""
        cls.base_path = Path(__file__).parent.parent
        cls.dashboards_path = cls.base_path / "grafana" / "provisioning" / "dashboards"
        
        config_file = cls.dashboards_path / "default.yaml"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                cls.config = yaml.safe_load(f)
        else:
            cls.config = None

    def test_provisioning_file_exists(self):
        """测试 Provisioning 配置文件存在"""
        config_file = self.dashboards_path / "default.yaml"
        self.assertTrue(config_file.exists(), "default.yaml 不存在")

    def test_provisioning_valid_yaml(self):
        """测试配置是有效的 YAML"""
        self.assertIsNotNone(self.config)

    def test_has_providers(self):
        """测试包含 providers"""
        self.assertIn('providers', self.config)
        self.assertIsInstance(self.config['providers'], list)


class TestGrafanaDashboards(unittest.TestCase):
    """Grafana 仪表板测试"""

    @classmethod
    def setUpClass(cls):
        """加载所有仪表板"""
        cls.base_path = Path(__file__).parent.parent
        cls.dashboards_path = cls.base_path / "grafana" / "dashboards"
        
        cls.dashboards: Dict[str, dict] = {}
        
        # 加载所有 JSON 仪表板
        if cls.dashboards_path.exists():
            for json_file in cls.dashboards_path.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        cls.dashboards[json_file.stem] = json.load(f)
                except json.JSONDecodeError:
                    cls.dashboards[json_file.stem] = None

    def test_overview_dashboard_exists(self):
        """测试总览仪表板存在"""
        self.assertIn('overview', self.dashboards)
        self.assertIsNotNone(self.dashboards['overview'])

    def test_services_dashboard_exists(self):
        """测试服务监控仪表板存在"""
        self.assertIn('services', self.dashboards)
        self.assertIsNotNone(self.dashboards['services'])

    def test_inference_dashboard_exists(self):
        """测试推理监控仪表板存在"""
        self.assertIn('inference', self.dashboards)
        self.assertIsNotNone(self.dashboards['inference'])

    def test_database_dashboard_exists(self):
        """测试数据库监控仪表板存在"""
        self.assertIn('database', self.dashboards)
        self.assertIsNotNone(self.dashboards['database'])

    def test_dashboards_have_title(self):
        """测试每个仪表板都有标题"""
        for name, dashboard in self.dashboards.items():
            if dashboard:
                self.assertIn('title', dashboard, f"{name} 缺少 title")

    def test_dashboards_have_uid(self):
        """测试每个仪表板都有 UID"""
        for name, dashboard in self.dashboards.items():
            if dashboard:
                self.assertIn('uid', dashboard, f"{name} 缺少 uid")

    def test_dashboards_have_panels(self):
        """测试每个仪表板都有面板"""
        for name, dashboard in self.dashboards.items():
            if dashboard:
                self.assertIn('panels', dashboard, f"{name} 缺少 panels")
                self.assertIsInstance(dashboard['panels'], list)
                self.assertGreater(
                    len(dashboard['panels']), 
                    0, 
                    f"{name} 没有面板"
                )

    def test_dashboards_have_tags(self):
        """测试每个仪表板都有标签"""
        for name, dashboard in self.dashboards.items():
            if dashboard:
                self.assertIn('tags', dashboard, f"{name} 缺少 tags")
                self.assertIn(
                    'recommend-system', 
                    dashboard['tags'],
                    f"{name} 缺少 recommend-system 标签"
                )

    def test_overview_dashboard_has_key_metrics(self):
        """测试总览仪表板包含关键指标"""
        dashboard = self.dashboards.get('overview')
        if dashboard:
            panel_titles = [p.get('title', '') for p in dashboard['panels']]
            
            # 检查关键面板存在
            key_panels = ['在线服务', 'QPS', 'P99', '错误率']
            for key in key_panels:
                found = any(key in title for title in panel_titles)
                self.assertTrue(found, f"总览仪表板缺少 {key} 相关面板")

    def test_inference_dashboard_has_gpu_metrics(self):
        """测试推理仪表板包含 GPU 指标"""
        dashboard = self.dashboards.get('inference')
        if dashboard:
            panel_titles = [p.get('title', '') for p in dashboard['panels']]
            
            # 检查 GPU 相关面板
            gpu_related = any('GPU' in title for title in panel_titles)
            self.assertTrue(gpu_related, "推理仪表板缺少 GPU 相关面板")

    def test_database_dashboard_has_db_metrics(self):
        """测试数据库仪表板包含数据库指标"""
        dashboard = self.dashboards.get('database')
        if dashboard:
            panel_titles = [p.get('title', '') for p in dashboard['panels']]
            
            # 检查 PostgreSQL 相关面板
            postgres_related = any('Postgres' in title or 'PostgreSQL' in title 
                                   or '连接' in title 
                                   for title in panel_titles)
            self.assertTrue(postgres_related, "数据库仪表板缺少 PostgreSQL 相关面板")
            
            # 检查 Redis 相关面板
            redis_related = any('Redis' in title or '内存' in title 
                               for title in panel_titles)
            self.assertTrue(redis_related, "数据库仪表板缺少 Redis 相关面板")

    def test_panels_have_datasource(self):
        """测试面板都有数据源配置"""
        for name, dashboard in self.dashboards.items():
            if dashboard:
                for panel in dashboard['panels']:
                    # 跳过 row 类型的面板
                    if panel.get('type') == 'row':
                        continue
                    
                    # 检查有 datasource 或 targets
                    has_datasource = 'datasource' in panel
                    has_targets = 'targets' in panel and len(panel.get('targets', [])) > 0
                    
                    self.assertTrue(
                        has_datasource or has_targets,
                        f"{name} 的面板 {panel.get('title', 'unknown')} 缺少数据源配置"
                    )

    def test_dashboard_refresh_interval(self):
        """测试仪表板有刷新间隔"""
        for name, dashboard in self.dashboards.items():
            if dashboard:
                self.assertIn('refresh', dashboard, f"{name} 缺少 refresh 设置")


class TestGrafanaDashboardVariables(unittest.TestCase):
    """Grafana 仪表板变量测试"""

    @classmethod
    def setUpClass(cls):
        """加载仪表板"""
        cls.base_path = Path(__file__).parent.parent
        cls.dashboards_path = cls.base_path / "grafana" / "dashboards"
        
        cls.dashboards: Dict[str, dict] = {}
        if cls.dashboards_path.exists():
            for json_file in cls.dashboards_path.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        cls.dashboards[json_file.stem] = json.load(f)
                except json.JSONDecodeError:
                    pass

    def test_dashboards_have_templating(self):
        """测试仪表板有模板变量"""
        for name, dashboard in self.dashboards.items():
            if dashboard:
                self.assertIn('templating', dashboard, f"{name} 缺少 templating")

    def test_services_dashboard_has_service_variable(self):
        """测试服务仪表板有服务变量"""
        dashboard = self.dashboards.get('services')
        if dashboard:
            templating = dashboard.get('templating', {})
            variables = templating.get('list', [])
            variable_names = [v.get('name', '') for v in variables]
            
            self.assertIn('service', variable_names, "服务仪表板缺少 service 变量")

    def test_inference_dashboard_has_model_variable(self):
        """测试推理仪表板有模型变量"""
        dashboard = self.dashboards.get('inference')
        if dashboard:
            templating = dashboard.get('templating', {})
            variables = templating.get('list', [])
            variable_names = [v.get('name', '') for v in variables]
            
            self.assertIn('model', variable_names, "推理仪表板缺少 model 变量")


if __name__ == '__main__':
    unittest.main()

