"""
AlertManager 配置验证测试

测试内容:
1. YAML 语法正确性
2. 路由配置完整性
3. 接收者配置有效性
4. 抑制规则有效性
"""

import os
import unittest
from pathlib import Path

import yaml


class TestAlertManagerConfig(unittest.TestCase):
    """AlertManager 配置测试"""

    @classmethod
    def setUpClass(cls):
        """加载配置文件"""
        cls.base_path = Path(__file__).parent.parent
        cls.alertmanager_path = cls.base_path / "alertmanager"
        
        config_file = cls.alertmanager_path / "alertmanager.yaml"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                cls.config = yaml.safe_load(f)
        else:
            cls.config = None

    def test_config_file_exists(self):
        """测试配置文件存在"""
        config_file = self.alertmanager_path / "alertmanager.yaml"
        self.assertTrue(config_file.exists(), "alertmanager.yaml 不存在")

    def test_config_valid_yaml(self):
        """测试配置是有效的 YAML"""
        self.assertIsNotNone(self.config, "无法解析 alertmanager.yaml")

    def test_config_has_global(self):
        """测试配置包含 global 部分"""
        self.assertIn('global', self.config)

    def test_config_has_route(self):
        """测试配置包含 route 部分"""
        self.assertIn('route', self.config)
        route = self.config['route']
        
        # 验证必要字段
        self.assertIn('receiver', route)
        self.assertIn('group_by', route)

    def test_config_has_receivers(self):
        """测试配置包含 receivers 部分"""
        self.assertIn('receivers', self.config)
        receivers = self.config['receivers']
        
        self.assertIsInstance(receivers, list)
        self.assertGreater(len(receivers), 0)

    def test_default_receiver_exists(self):
        """测试默认接收者存在"""
        route = self.config['route']
        default_receiver = route.get('receiver')
        
        receiver_names = [r['name'] for r in self.config['receivers']]
        self.assertIn(default_receiver, receiver_names)

    def test_critical_receiver_exists(self):
        """测试 critical 接收者存在"""
        receiver_names = [r['name'] for r in self.config['receivers']]
        self.assertIn('critical-alerts', receiver_names)

    def test_warning_receiver_exists(self):
        """测试 warning 接收者存在"""
        receiver_names = [r['name'] for r in self.config['receivers']]
        self.assertIn('warning-alerts', receiver_names)

    def test_routes_reference_valid_receivers(self):
        """测试所有路由引用的接收者都存在"""
        receiver_names = [r['name'] for r in self.config['receivers']]
        
        def check_routes(routes):
            for route in routes:
                if 'receiver' in route:
                    self.assertIn(
                        route['receiver'], 
                        receiver_names,
                        f"路由引用了不存在的接收者: {route['receiver']}"
                    )
                if 'routes' in route:
                    check_routes(route['routes'])
        
        if 'routes' in self.config['route']:
            check_routes(self.config['route']['routes'])

    def test_inhibit_rules_exist(self):
        """测试抑制规则存在"""
        self.assertIn('inhibit_rules', self.config)
        self.assertIsInstance(self.config['inhibit_rules'], list)
        self.assertGreater(len(self.config['inhibit_rules']), 0)

    def test_inhibit_rules_have_required_fields(self):
        """测试抑制规则有必要字段"""
        for rule in self.config['inhibit_rules']:
            self.assertTrue(
                'source_match' in rule or 'source_match_re' in rule,
                "抑制规则缺少 source_match 或 source_match_re"
            )
            self.assertTrue(
                'target_match' in rule or 'target_match_re' in rule,
                "抑制规则缺少 target_match 或 target_match_re"
            )

    def test_service_down_inhibit_rule(self):
        """测试 ServiceDown 抑制规则存在"""
        has_service_down_rule = False
        for rule in self.config['inhibit_rules']:
            source_match = rule.get('source_match', {})
            if source_match.get('alertname') == 'ServiceDown':
                has_service_down_rule = True
                break
        
        self.assertTrue(has_service_down_rule, "缺少 ServiceDown 抑制规则")


class TestAlertManagerTemplates(unittest.TestCase):
    """AlertManager 模板测试"""

    @classmethod
    def setUpClass(cls):
        """设置路径"""
        cls.base_path = Path(__file__).parent.parent
        cls.templates_path = cls.base_path / "alertmanager" / "templates"

    def test_templates_directory_exists(self):
        """测试模板目录存在"""
        self.assertTrue(self.templates_path.exists(), "templates 目录不存在")

    def test_slack_template_exists(self):
        """测试 Slack 模板存在"""
        slack_file = self.templates_path / "slack.tmpl"
        self.assertTrue(slack_file.exists(), "slack.tmpl 不存在")

    def test_slack_template_has_title_definition(self):
        """测试 Slack 模板包含 title 定义"""
        slack_file = self.templates_path / "slack.tmpl"
        with open(slack_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('define "slack.title"', content)

    def test_slack_template_has_text_definition(self):
        """测试 Slack 模板包含 text 定义"""
        slack_file = self.templates_path / "slack.tmpl"
        with open(slack_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('define "slack.text"', content)

    def test_slack_template_has_color_definition(self):
        """测试 Slack 模板包含 color 定义"""
        slack_file = self.templates_path / "slack.tmpl"
        with open(slack_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('define "slack.color"', content)


if __name__ == '__main__':
    unittest.main()

