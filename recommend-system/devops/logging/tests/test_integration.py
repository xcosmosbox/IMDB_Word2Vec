"""
日志系统集成测试

测试各组件之间的集成和配置一致性
"""

import os
import json
import unittest
from pathlib import Path
from typing import Dict, Any, List, Set

import yaml


# 获取项目路径
LOGGING_DIR = Path(__file__).parent.parent
DEVOPS_DIR = LOGGING_DIR.parent
INTERFACES_PATH = DEVOPS_DIR / "interfaces.yaml"


class TestLoggingIntegration(unittest.TestCase):
    """日志系统集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """加载所有配置文件"""
        # 加载接口定义
        with open(INTERFACES_PATH, 'r', encoding='utf-8') as f:
            cls.interfaces = yaml.safe_load(f)
        
        # 加载 Loki 配置
        loki_config_path = LOGGING_DIR / "loki" / "loki-config.yaml"
        with open(loki_config_path, 'r', encoding='utf-8') as f:
            cls.loki_config = yaml.safe_load(f)
        
        # 加载 Promtail 配置
        promtail_config_path = LOGGING_DIR / "promtail" / "promtail-config.yaml"
        with open(promtail_config_path, 'r', encoding='utf-8') as f:
            cls.promtail_config = yaml.safe_load(f)
        
        # 加载 Fluentd 配置
        fluentd_config_path = LOGGING_DIR / "fluentd" / "fluent.conf"
        with open(fluentd_config_path, 'r', encoding='utf-8') as f:
            cls.fluentd_config = f.read()
        
        # 加载告警规则
        alerts_path = LOGGING_DIR / "loki" / "rules" / "alerts.yaml"
        with open(alerts_path, 'r', encoding='utf-8') as f:
            cls.alert_rules = yaml.safe_load(f)
    
    def test_promtail_connects_to_loki(self):
        """测试 Promtail 连接到正确的 Loki 端点"""
        loki_port = self.loki_config['server']['http_listen_port']
        
        for client in self.promtail_config['clients']:
            url = client['url']
            self.assertIn('loki', url)
            self.assertIn(str(loki_port), url)
    
    def test_fluentd_connects_to_loki(self):
        """测试 Fluentd 连接到正确的 Loki 端点"""
        loki_port = self.loki_config['server']['http_listen_port']
        
        self.assertIn('loki', self.fluentd_config)
        self.assertIn(str(loki_port), self.fluentd_config)
    
    def test_consistent_log_fields(self):
        """测试日志字段在所有配置中一致"""
        # 从接口定义获取必需字段
        required_fields = set(
            self.interfaces['logging']['format']['required_fields']
        )
        
        # 检查 Promtail 配置提取的字段
        promtail_fields = set()
        for scrape_config in self.promtail_config['scrape_configs']:
            for stage in scrape_config.get('pipeline_stages', []):
                if 'json' in stage:
                    promtail_fields.update(stage['json'].get('expressions', {}).keys())
        
        # 验证必需字段被提取
        missing_fields = required_fields - promtail_fields
        self.assertEqual(len(missing_fields), 0,
                        f"Promtail 配置缺少必需字段: {missing_fields}")
    
    def test_consistent_log_levels(self):
        """测试日志级别在所有配置中一致"""
        valid_levels = set(self.interfaces['logging']['levels'])
        
        # 检查告警规则中使用的日志级别
        for group in self.alert_rules.get('groups', []):
            for rule in group.get('rules', []):
                expr = rule.get('expr', '')
                
                # 提取表达式中的日志级别
                for level in ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL']:
                    if f'level="{level}"' in expr or f"level='{level}'" in expr:
                        self.assertIn(level, valid_levels,
                                    f"告警规则使用了无效的日志级别: {level}")
    
    def test_consistent_labels(self):
        """测试标签在所有配置中一致"""
        valid_labels = set(self.interfaces['logging']['labels'])
        
        # 检查 Promtail 配置的标签
        promtail_labels = set()
        for scrape_config in self.promtail_config['scrape_configs']:
            for stage in scrape_config.get('pipeline_stages', []):
                if 'labels' in stage:
                    labels_config = stage['labels']
                    if isinstance(labels_config, dict):
                        promtail_labels.update(labels_config.keys())
                    elif isinstance(labels_config, list):
                        promtail_labels.update(labels_config)
        
        # 验证使用的标签是有效的
        for label in promtail_labels:
            if label in valid_labels:
                self.assertIn(label, valid_labels)
    
    def test_retention_consistency(self):
        """测试保留策略一致性"""
        # Loki 的保留期
        loki_retention = self.loki_config['table_manager']['retention_period']
        
        # 保留期应该是合理的值 (至少 7 天)
        retention_hours = int(loki_retention.replace('h', ''))
        self.assertGreaterEqual(retention_hours, 168,
                               "保留期应该至少 7 天 (168 小时)")


class TestKubernetesManifests(unittest.TestCase):
    """Kubernetes 清单一致性测试"""
    
    @classmethod
    def setUpClass(cls):
        """加载所有 Kubernetes 清单"""
        cls.manifests = {}
        
        # 加载 Loki 部署清单
        loki_deployment_path = LOGGING_DIR / "loki" / "deployment.yaml"
        with open(loki_deployment_path, 'r', encoding='utf-8') as f:
            cls.manifests['loki'] = list(yaml.safe_load_all(f))
        
        # 加载 Promtail DaemonSet
        promtail_daemonset_path = LOGGING_DIR / "promtail" / "daemonset.yaml"
        with open(promtail_daemonset_path, 'r', encoding='utf-8') as f:
            cls.manifests['promtail'] = list(yaml.safe_load_all(f))
        
        # 加载 Fluentd DaemonSet
        fluentd_daemonset_path = LOGGING_DIR / "fluentd" / "daemonset.yaml"
        with open(fluentd_daemonset_path, 'r', encoding='utf-8') as f:
            cls.manifests['fluentd'] = list(yaml.safe_load_all(f))
    
    def test_consistent_namespace(self):
        """测试命名空间一致性"""
        expected_namespace = "recommend-prod"
        
        for component, manifests in self.manifests.items():
            for manifest in manifests:
                if manifest and 'metadata' in manifest:
                    namespace = manifest['metadata'].get('namespace')
                    if namespace:
                        self.assertEqual(namespace, expected_namespace,
                                       f"{component} 的命名空间应该是 {expected_namespace}")
    
    def test_consistent_labels(self):
        """测试标签一致性"""
        for component, manifests in self.manifests.items():
            for manifest in manifests:
                if manifest and 'metadata' in manifest:
                    labels = manifest['metadata'].get('labels', {})
                    
                    # 检查组件标签
                    if 'component' in labels:
                        self.assertEqual(labels['component'], 'logging',
                                       f"{component} 的 component 标签应该是 logging")
    
    def test_service_discovery_labels(self):
        """测试服务发现标签"""
        for component, manifests in self.manifests.items():
            daemonsets = [m for m in manifests 
                        if m and m.get('kind') in ('DaemonSet', 'Deployment')]
            
            for ds in daemonsets:
                template = ds['spec']['template']
                annotations = template['metadata'].get('annotations', {})
                
                # 检查 Prometheus 抓取注解
                if 'prometheus.io/scrape' in annotations:
                    self.assertEqual(annotations['prometheus.io/scrape'], 'true')
    
    def test_resource_requests_and_limits(self):
        """测试资源请求和限制配置"""
        for component, manifests in self.manifests.items():
            workloads = [m for m in manifests 
                        if m and m.get('kind') in ('DaemonSet', 'Deployment')]
            
            for workload in workloads:
                containers = workload['spec']['template']['spec']['containers']
                
                for container in containers:
                    resources = container.get('resources', {})
                    
                    # 检查资源请求
                    self.assertIn('requests', resources,
                                f"{component}/{container['name']} 应该配置资源请求")
                    
                    # 检查资源限制
                    self.assertIn('limits', resources,
                                f"{component}/{container['name']} 应该配置资源限制")
    
    def test_probes_configured(self):
        """测试健康检查探针配置"""
        for component, manifests in self.manifests.items():
            workloads = [m for m in manifests 
                        if m and m.get('kind') in ('DaemonSet', 'Deployment')]
            
            for workload in workloads:
                containers = workload['spec']['template']['spec']['containers']
                
                for container in containers:
                    # 检查就绪探针
                    self.assertIn('readinessProbe', container,
                                f"{component}/{container['name']} 应该配置就绪探针")
                    
                    # 检查存活探针
                    self.assertIn('livenessProbe', container,
                                f"{component}/{container['name']} 应该配置存活探针")


class TestAlertRulesIntegration(unittest.TestCase):
    """告警规则集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """加载告警规则和接口定义"""
        with open(INTERFACES_PATH, 'r', encoding='utf-8') as f:
            cls.interfaces = yaml.safe_load(f)
        
        alerts_path = LOGGING_DIR / "loki" / "rules" / "alerts.yaml"
        with open(alerts_path, 'r', encoding='utf-8') as f:
            cls.alert_rules = yaml.safe_load(f)
    
    def test_critical_alerts_exist(self):
        """测试关键告警存在"""
        critical_alert_names = set()
        
        for group in self.alert_rules.get('groups', []):
            for rule in group.get('rules', []):
                if rule.get('labels', {}).get('severity') == 'critical':
                    critical_alert_names.add(rule['alert'])
        
        # 应该有关键告警
        self.assertTrue(len(critical_alert_names) > 0,
                       "应该至少有一个 critical 级别的告警")
        
        # 检查必要的关键告警
        expected_critical_alerts = {'FatalLogDetected', 'PanicDetected'}
        for expected in expected_critical_alerts:
            self.assertIn(expected, critical_alert_names,
                        f"应该有 {expected} 告警")
    
    def test_alert_annotations(self):
        """测试告警注解完整性"""
        for group in self.alert_rules.get('groups', []):
            for rule in group.get('rules', []):
                annotations = rule.get('annotations', {})
                
                # 检查必需的注解
                self.assertIn('summary', annotations,
                            f"告警 {rule['alert']} 应该有 summary 注解")
                self.assertIn('description', annotations,
                            f"告警 {rule['alert']} 应该有 description 注解")
    
    def test_alert_for_duration(self):
        """测试告警触发持续时间配置"""
        for group in self.alert_rules.get('groups', []):
            for rule in group.get('rules', []):
                # for 字段定义告警触发前的等待时间
                self.assertIn('for', rule,
                            f"告警 {rule['alert']} 应该配置 for 字段")


class TestGrafanaDashboardIntegration(unittest.TestCase):
    """Grafana Dashboard 集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """加载 Dashboard 文件"""
        dashboards_dir = LOGGING_DIR / "grafana" / "dashboards"
        cls.dashboards = {}
        
        for dashboard_file in dashboards_dir.glob("*.json"):
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                cls.dashboards[dashboard_file.name] = json.load(f)
    
    def test_dashboards_use_loki(self):
        """测试 Dashboard 使用 Loki 数据源"""
        for name, dashboard in self.dashboards.items():
            panels = dashboard.get('panels', [])
            
            loki_used = False
            for panel in panels:
                targets = panel.get('targets', [])
                for target in targets:
                    datasource = target.get('datasource', {})
                    if isinstance(datasource, dict):
                        if datasource.get('type') == 'loki' or datasource.get('uid') == 'loki':
                            loki_used = True
                            break
                if loki_used:
                    break
            
            self.assertTrue(loki_used, f"Dashboard {name} 应该使用 Loki 数据源")
    
    def test_dashboards_have_refresh(self):
        """测试 Dashboard 配置自动刷新"""
        for name, dashboard in self.dashboards.items():
            refresh = dashboard.get('refresh')
            self.assertIsNotNone(refresh, f"Dashboard {name} 应该配置自动刷新")
    
    def test_dashboards_have_time_range(self):
        """测试 Dashboard 配置时间范围"""
        for name, dashboard in self.dashboards.items():
            time = dashboard.get('time', {})
            self.assertIn('from', time, f"Dashboard {name} 应该配置开始时间")
            self.assertIn('to', time, f"Dashboard {name} 应该配置结束时间")
    
    def test_dashboards_unique_uids(self):
        """测试 Dashboard UID 唯一"""
        uids = []
        for name, dashboard in self.dashboards.items():
            uid = dashboard.get('uid')
            self.assertIsNotNone(uid, f"Dashboard {name} 应该有 UID")
            self.assertNotIn(uid, uids, f"Dashboard UID {uid} 重复")
            uids.append(uid)


class TestEndToEndScenarios(unittest.TestCase):
    """端到端场景测试"""
    
    def test_error_log_flow(self):
        """测试错误日志流程"""
        # 模拟错误日志
        error_log = {
            'timestamp': '2025-01-05T10:30:00.123456Z',
            'level': 'ERROR',
            'service': 'recommend-service',
            'trace_id': 'trace-123',
            'message': 'Database connection failed',
            'error_stack': 'ConnectionError: timeout'
        }
        
        # 验证日志格式
        required_fields = {'timestamp', 'level', 'service', 'trace_id', 'message'}
        for field in required_fields:
            self.assertIn(field, error_log)
        
        # 验证日志级别
        self.assertEqual(error_log['level'], 'ERROR')
    
    def test_inference_log_flow(self):
        """测试推理日志流程"""
        # 模拟推理日志
        inference_log = {
            'timestamp': '2025-01-05T10:30:00.123456Z',
            'level': 'INFO',
            'service': 'ugt-inference',
            'trace_id': 'trace-456',
            'message': 'Inference completed',
            'model': 'ugt-v1',
            'batch_size': 32,
            'latency_ms': 85.5,
            'status': 'success'
        }
        
        # 验证日志格式
        required_fields = {'timestamp', 'level', 'service', 'trace_id', 'message'}
        for field in required_fields:
            self.assertIn(field, inference_log)
        
        # 验证推理特定字段
        self.assertIn('model', inference_log)
        self.assertIn('latency_ms', inference_log)
    
    def test_trace_correlation(self):
        """测试跨服务追踪关联"""
        # 模拟同一请求的多个日志
        trace_id = 'trace-789'
        
        logs = [
            {
                'timestamp': '2025-01-05T10:30:00.000Z',
                'level': 'INFO',
                'service': 'recommend-service',
                'trace_id': trace_id,
                'message': 'Received recommendation request'
            },
            {
                'timestamp': '2025-01-05T10:30:00.050Z',
                'level': 'INFO',
                'service': 'user-service',
                'trace_id': trace_id,
                'message': 'Fetching user profile'
            },
            {
                'timestamp': '2025-01-05T10:30:00.100Z',
                'level': 'INFO',
                'service': 'ugt-inference',
                'trace_id': trace_id,
                'message': 'Running inference'
            },
            {
                'timestamp': '2025-01-05T10:30:00.200Z',
                'level': 'INFO',
                'service': 'recommend-service',
                'trace_id': trace_id,
                'message': 'Returning recommendations'
            }
        ]
        
        # 验证所有日志有相同的 trace_id
        for log in logs:
            self.assertEqual(log['trace_id'], trace_id)
        
        # 验证日志按时间排序
        timestamps = [log['timestamp'] for log in logs]
        self.assertEqual(timestamps, sorted(timestamps))


if __name__ == '__main__':
    unittest.main()

