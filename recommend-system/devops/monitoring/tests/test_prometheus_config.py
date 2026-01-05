"""
Prometheus 配置验证测试

测试内容:
1. YAML 语法正确性
2. 配置结构完整性
3. 告警规则有效性
4. 记录规则有效性
"""

import json
import os
import re
import unittest
from pathlib import Path
from typing import Any, Dict, List

import yaml


class TestPrometheusConfig(unittest.TestCase):
    """Prometheus 配置测试"""

    @classmethod
    def setUpClass(cls):
        """加载配置文件"""
        cls.base_path = Path(__file__).parent.parent
        cls.prometheus_path = cls.base_path / "prometheus"
        
        # 加载主配置
        config_file = cls.prometheus_path / "prometheus.yaml"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                cls.prometheus_config = yaml.safe_load(f)
        else:
            cls.prometheus_config = None

    def test_prometheus_config_exists(self):
        """测试主配置文件存在"""
        config_file = self.prometheus_path / "prometheus.yaml"
        self.assertTrue(config_file.exists(), "prometheus.yaml 文件不存在")

    def test_prometheus_config_valid_yaml(self):
        """测试主配置是有效的 YAML"""
        self.assertIsNotNone(self.prometheus_config, "无法解析 prometheus.yaml")

    def test_prometheus_has_global_section(self):
        """测试配置包含 global 部分"""
        self.assertIn('global', self.prometheus_config)
        global_config = self.prometheus_config['global']
        
        # 验证必要字段
        self.assertIn('scrape_interval', global_config)
        self.assertIn('evaluation_interval', global_config)

    def test_prometheus_has_scrape_configs(self):
        """测试配置包含 scrape_configs"""
        self.assertIn('scrape_configs', self.prometheus_config)
        scrape_configs = self.prometheus_config['scrape_configs']
        
        self.assertIsInstance(scrape_configs, list)
        self.assertGreater(len(scrape_configs), 0)

    def test_prometheus_has_alerting(self):
        """测试配置包含告警配置"""
        self.assertIn('alerting', self.prometheus_config)

    def test_scrape_config_has_required_fields(self):
        """测试每个 scrape_config 都有必要字段"""
        for config in self.prometheus_config['scrape_configs']:
            self.assertIn('job_name', config, f"scrape_config 缺少 job_name")

    def test_service_ports_match_interface(self):
        """测试服务端口符合接口契约"""
        # 根据 interfaces.yaml 定义的端口
        expected_ports = {
            'recommend-service': 9091,
            'user-service': 9092,
            'item-service': 9093,
            'ugt-inference': 9094,
        }
        
        for config in self.prometheus_config['scrape_configs']:
            job_name = config.get('job_name', '')
            if job_name in expected_ports:
                static_configs = config.get('static_configs', [])
                if static_configs:
                    targets = static_configs[0].get('targets', [])
                    if targets:
                        # 提取端口
                        target = targets[0]
                        port = int(target.split(':')[-1])
                        self.assertEqual(
                            port, 
                            expected_ports[job_name],
                            f"{job_name} 端口不匹配"
                        )


class TestAlertingRules(unittest.TestCase):
    """告警规则测试"""

    @classmethod
    def setUpClass(cls):
        """加载告警规则"""
        cls.base_path = Path(__file__).parent.parent
        cls.rules_path = cls.base_path / "prometheus" / "rules"
        
        # 加载告警规则
        alerting_file = cls.rules_path / "alerting-rules.yaml"
        if alerting_file.exists():
            with open(alerting_file, 'r', encoding='utf-8') as f:
                cls.alerting_rules = yaml.safe_load(f)
        else:
            cls.alerting_rules = None

    def test_alerting_rules_file_exists(self):
        """测试告警规则文件存在"""
        alerting_file = self.rules_path / "alerting-rules.yaml"
        self.assertTrue(alerting_file.exists(), "alerting-rules.yaml 不存在")

    def test_alerting_rules_valid_yaml(self):
        """测试告警规则是有效的 YAML"""
        self.assertIsNotNone(self.alerting_rules)

    def test_alerting_rules_has_groups(self):
        """测试告警规则包含 groups"""
        self.assertIn('groups', self.alerting_rules)
        self.assertIsInstance(self.alerting_rules['groups'], list)
        self.assertGreater(len(self.alerting_rules['groups']), 0)

    def test_critical_alerts_exist(self):
        """测试关键告警规则存在 (按 interfaces.yaml 要求)"""
        required_alerts = ['ServiceDown', 'HighErrorRate', 'HighLatency']
        
        all_alerts = []
        for group in self.alerting_rules['groups']:
            for rule in group.get('rules', []):
                if 'alert' in rule:
                    all_alerts.append(rule['alert'])
        
        for alert_name in required_alerts:
            self.assertIn(
                alert_name, 
                all_alerts, 
                f"缺少必要告警规则: {alert_name}"
            )

    def test_warning_alerts_exist(self):
        """测试警告告警规则存在"""
        required_alerts = ['HighMemoryUsage', 'HighCPUUsage']
        
        all_alerts = []
        for group in self.alerting_rules['groups']:
            for rule in group.get('rules', []):
                if 'alert' in rule:
                    all_alerts.append(rule['alert'])
        
        for alert_name in required_alerts:
            self.assertIn(
                alert_name, 
                all_alerts, 
                f"缺少必要告警规则: {alert_name}"
            )

    def test_alert_rule_has_required_fields(self):
        """测试每个告警规则都有必要字段"""
        required_fields = ['alert', 'expr', 'labels', 'annotations']
        
        for group in self.alerting_rules['groups']:
            for rule in group.get('rules', []):
                if 'alert' in rule:  # 只检查告警规则，不检查记录规则
                    for field in required_fields:
                        self.assertIn(
                            field, 
                            rule, 
                            f"告警 {rule['alert']} 缺少字段: {field}"
                        )

    def test_alert_has_severity_label(self):
        """测试每个告警都有 severity 标签"""
        for group in self.alerting_rules['groups']:
            for rule in group.get('rules', []):
                if 'alert' in rule:
                    labels = rule.get('labels', {})
                    self.assertIn(
                        'severity', 
                        labels, 
                        f"告警 {rule['alert']} 缺少 severity 标签"
                    )
                    # 验证 severity 值
                    self.assertIn(
                        labels['severity'],
                        ['critical', 'warning', 'info'],
                        f"告警 {rule['alert']} 的 severity 值无效"
                    )

    def test_alert_has_summary_annotation(self):
        """测试每个告警都有 summary 注解"""
        for group in self.alerting_rules['groups']:
            for rule in group.get('rules', []):
                if 'alert' in rule:
                    annotations = rule.get('annotations', {})
                    self.assertIn(
                        'summary', 
                        annotations, 
                        f"告警 {rule['alert']} 缺少 summary 注解"
                    )


class TestRecordingRules(unittest.TestCase):
    """记录规则测试"""

    @classmethod
    def setUpClass(cls):
        """加载记录规则"""
        cls.base_path = Path(__file__).parent.parent
        cls.rules_path = cls.base_path / "prometheus" / "rules"
        
        recording_file = cls.rules_path / "recording-rules.yaml"
        if recording_file.exists():
            with open(recording_file, 'r', encoding='utf-8') as f:
                cls.recording_rules = yaml.safe_load(f)
        else:
            cls.recording_rules = None

    def test_recording_rules_file_exists(self):
        """测试记录规则文件存在"""
        recording_file = self.rules_path / "recording-rules.yaml"
        self.assertTrue(recording_file.exists(), "recording-rules.yaml 不存在")

    def test_recording_rules_valid_yaml(self):
        """测试记录规则是有效的 YAML"""
        self.assertIsNotNone(self.recording_rules)

    def test_recording_rules_has_groups(self):
        """测试记录规则包含 groups"""
        self.assertIn('groups', self.recording_rules)

    def test_recording_rule_naming_convention(self):
        """测试记录规则命名约定 (namespace:metric:aggregation)"""
        pattern = re.compile(r'^[a-z_]+:[a-z_]+:[a-z0-9_]+$')
        
        for group in self.recording_rules['groups']:
            for rule in group.get('rules', []):
                if 'record' in rule:
                    record_name = rule['record']
                    self.assertTrue(
                        pattern.match(record_name),
                        f"记录规则 {record_name} 不符合命名约定"
                    )

    def test_recording_rule_has_expr(self):
        """测试每个记录规则都有 expr"""
        for group in self.recording_rules['groups']:
            for rule in group.get('rules', []):
                if 'record' in rule:
                    self.assertIn(
                        'expr', 
                        rule, 
                        f"记录规则 {rule['record']} 缺少 expr"
                    )


class TestScrapeConfigs(unittest.TestCase):
    """抓取配置测试"""

    @classmethod
    def setUpClass(cls):
        """加载抓取配置"""
        cls.base_path = Path(__file__).parent.parent
        cls.scrape_path = cls.base_path / "prometheus" / "scrape-configs"

    def test_kubernetes_config_exists(self):
        """测试 Kubernetes 抓取配置存在"""
        k8s_file = self.scrape_path / "kubernetes.yaml"
        self.assertTrue(k8s_file.exists(), "kubernetes.yaml 不存在")

    def test_custom_config_exists(self):
        """测试自定义抓取配置存在"""
        custom_file = self.scrape_path / "custom.yaml"
        self.assertTrue(custom_file.exists(), "custom.yaml 不存在")

    def test_kubernetes_config_valid_yaml(self):
        """测试 Kubernetes 配置是有效的 YAML"""
        k8s_file = self.scrape_path / "kubernetes.yaml"
        with open(k8s_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self.assertIsNotNone(config)

    def test_custom_config_valid_yaml(self):
        """测试自定义配置是有效的 YAML"""
        custom_file = self.scrape_path / "custom.yaml"
        with open(custom_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self.assertIsNotNone(config)


if __name__ == '__main__':
    unittest.main()

