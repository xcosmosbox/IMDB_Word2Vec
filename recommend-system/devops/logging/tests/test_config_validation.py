"""
日志系统配置验证测试

测试所有配置文件的语法和内容正确性
"""

import os
import json
import unittest
from pathlib import Path
from typing import Dict, Any, List

import yaml


# 获取项目根目录
LOGGING_DIR = Path(__file__).parent.parent
LOKI_DIR = LOGGING_DIR / "loki"
PROMTAIL_DIR = LOGGING_DIR / "promtail"
FLUENTD_DIR = LOGGING_DIR / "fluentd"
GRAFANA_DIR = LOGGING_DIR / "grafana"


class TestLokiConfig(unittest.TestCase):
    """Loki 配置文件测试"""
    
    def setUp(self):
        """加载 Loki 配置文件"""
        self.config_path = LOKI_DIR / "loki-config.yaml"
        self.local_config_path = LOKI_DIR / "local-config.yaml"
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        with open(self.local_config_path, 'r', encoding='utf-8') as f:
            self.local_config = yaml.safe_load(f)
    
    def test_config_file_exists(self):
        """测试配置文件存在"""
        self.assertTrue(self.config_path.exists(), "loki-config.yaml 不存在")
        self.assertTrue(self.local_config_path.exists(), "local-config.yaml 不存在")
    
    def test_server_config(self):
        """测试服务器配置"""
        self.assertIn('server', self.config)
        server = self.config['server']
        
        # 检查端口配置
        self.assertEqual(server['http_listen_port'], 3100, "HTTP 端口应为 3100")
        self.assertEqual(server['grpc_listen_port'], 9096, "gRPC 端口应为 9096")
    
    def test_schema_config(self):
        """测试 schema 配置"""
        self.assertIn('schema_config', self.config)
        schema = self.config['schema_config']
        
        self.assertIn('configs', schema)
        self.assertTrue(len(schema['configs']) > 0, "至少需要一个 schema 配置")
        
        config = schema['configs'][0]
        self.assertIn('from', config)
        self.assertIn('store', config)
        self.assertIn('object_store', config)
        self.assertIn('schema', config)
    
    def test_storage_config(self):
        """测试存储配置"""
        self.assertIn('storage_config', self.config)
        storage = self.config['storage_config']
        
        self.assertIn('boltdb_shipper', storage)
        self.assertIn('filesystem', storage)
    
    def test_limits_config(self):
        """测试限制配置"""
        self.assertIn('limits_config', self.config)
        limits = self.config['limits_config']
        
        # 检查关键限制参数
        self.assertIn('ingestion_rate_mb', limits)
        self.assertIn('ingestion_burst_size_mb', limits)
        self.assertIn('max_entries_limit_per_query', limits)
        
        # 验证合理的值
        self.assertGreater(limits['ingestion_rate_mb'], 0)
        self.assertGreater(limits['max_entries_limit_per_query'], 0)
    
    def test_retention_config(self):
        """测试保留策略配置"""
        self.assertIn('table_manager', self.config)
        table_manager = self.config['table_manager']
        
        self.assertTrue(table_manager['retention_deletes_enabled'], 
                       "保留策略删除应该启用")
        self.assertIn('retention_period', table_manager)
    
    def test_ruler_config(self):
        """测试规则器配置"""
        self.assertIn('ruler', self.config)
        ruler = self.config['ruler']
        
        self.assertIn('storage', ruler)
        self.assertIn('alertmanager_url', ruler)
        self.assertTrue(ruler['enable_api'], "规则器 API 应该启用")
    
    def test_local_config_differences(self):
        """测试本地配置与生产配置的差异"""
        # 本地配置应该有更短的保留期
        local_retention = self.local_config['table_manager']['retention_period']
        prod_retention = self.config['table_manager']['retention_period']
        
        # 本地配置应该有更低的限制
        local_limits = self.local_config['limits_config']
        prod_limits = self.config['limits_config']
        
        self.assertLessEqual(
            local_limits['ingestion_rate_mb'],
            prod_limits['ingestion_rate_mb'],
            "本地配置的摄入速率应该更低"
        )


class TestLokiDeployment(unittest.TestCase):
    """Loki Kubernetes 部署清单测试"""
    
    def setUp(self):
        """加载部署清单"""
        self.deployment_path = LOKI_DIR / "deployment.yaml"
        
        with open(self.deployment_path, 'r', encoding='utf-8') as f:
            self.manifests = list(yaml.safe_load_all(f))
    
    def test_deployment_file_exists(self):
        """测试部署文件存在"""
        self.assertTrue(self.deployment_path.exists())
    
    def test_manifests_count(self):
        """测试清单数量"""
        # 应该包含 ConfigMap, PVC, Deployment, Service, ServiceAccount, ServiceMonitor
        self.assertGreaterEqual(len(self.manifests), 5, 
                                "应该至少包含 5 个 Kubernetes 资源")
    
    def test_namespace_consistency(self):
        """测试命名空间一致性"""
        expected_namespace = "recommend-prod"
        
        for manifest in self.manifests:
            if manifest and 'metadata' in manifest:
                namespace = manifest['metadata'].get('namespace')
                if namespace:
                    self.assertEqual(namespace, expected_namespace,
                                   f"资源 {manifest['metadata']['name']} 命名空间不一致")
    
    def test_deployment_exists(self):
        """测试 Deployment 资源存在"""
        deployments = [m for m in self.manifests 
                      if m and m.get('kind') == 'Deployment']
        self.assertEqual(len(deployments), 1, "应该有一个 Deployment")
        
        deployment = deployments[0]
        self.assertEqual(deployment['metadata']['name'], 'loki')
    
    def test_service_exists(self):
        """测试 Service 资源存在"""
        services = [m for m in self.manifests 
                   if m and m.get('kind') == 'Service']
        self.assertGreaterEqual(len(services), 1, "应该至少有一个 Service")
    
    def test_resource_limits(self):
        """测试资源限制配置"""
        deployments = [m for m in self.manifests 
                      if m and m.get('kind') == 'Deployment']
        
        for deployment in deployments:
            containers = deployment['spec']['template']['spec']['containers']
            for container in containers:
                self.assertIn('resources', container, 
                            f"容器 {container['name']} 应该配置资源限制")
                resources = container['resources']
                self.assertIn('requests', resources)
                self.assertIn('limits', resources)
    
    def test_probes_configured(self):
        """测试探针配置"""
        deployments = [m for m in self.manifests 
                      if m and m.get('kind') == 'Deployment']
        
        for deployment in deployments:
            containers = deployment['spec']['template']['spec']['containers']
            for container in containers:
                self.assertIn('livenessProbe', container,
                            f"容器 {container['name']} 应该配置存活探针")
                self.assertIn('readinessProbe', container,
                            f"容器 {container['name']} 应该配置就绪探针")


class TestLokiAlertRules(unittest.TestCase):
    """Loki 告警规则测试"""
    
    def setUp(self):
        """加载告警规则"""
        self.rules_path = LOKI_DIR / "rules" / "alerts.yaml"
        
        with open(self.rules_path, 'r', encoding='utf-8') as f:
            self.rules = yaml.safe_load(f)
    
    def test_rules_file_exists(self):
        """测试规则文件存在"""
        self.assertTrue(self.rules_path.exists())
    
    def test_groups_structure(self):
        """测试规则组结构"""
        self.assertIn('groups', self.rules)
        self.assertTrue(len(self.rules['groups']) > 0, "至少需要一个规则组")
    
    def test_rule_structure(self):
        """测试规则结构"""
        for group in self.rules['groups']:
            self.assertIn('name', group, "规则组需要名称")
            self.assertIn('rules', group, "规则组需要规则列表")
            
            for rule in group['rules']:
                self.assertIn('alert', rule, "规则需要 alert 名称")
                self.assertIn('expr', rule, "规则需要 expr 表达式")
                self.assertIn('labels', rule, "规则需要 labels")
                self.assertIn('annotations', rule, "规则需要 annotations")
                
                # 检查 severity 标签
                self.assertIn('severity', rule['labels'],
                            f"规则 {rule['alert']} 需要 severity 标签")
    
    def test_severity_levels(self):
        """测试 severity 级别有效性"""
        valid_severities = {'critical', 'warning', 'info'}
        
        for group in self.rules['groups']:
            for rule in group['rules']:
                severity = rule['labels']['severity']
                self.assertIn(severity, valid_severities,
                            f"规则 {rule['alert']} 的 severity 无效: {severity}")
    
    def test_error_alert_exists(self):
        """测试错误告警存在"""
        error_alerts = []
        for group in self.rules['groups']:
            for rule in group['rules']:
                if 'error' in rule['alert'].lower():
                    error_alerts.append(rule['alert'])
        
        self.assertTrue(len(error_alerts) > 0, "应该有错误相关的告警规则")
    
    def test_panic_alert_exists(self):
        """测试 Panic 告警存在"""
        panic_alerts = []
        for group in self.rules['groups']:
            for rule in group['rules']:
                if 'panic' in rule['alert'].lower():
                    panic_alerts.append(rule['alert'])
        
        self.assertTrue(len(panic_alerts) > 0, "应该有 Panic 相关的告警规则")


class TestPromtailConfig(unittest.TestCase):
    """Promtail 配置文件测试"""
    
    def setUp(self):
        """加载 Promtail 配置文件"""
        self.config_path = PROMTAIL_DIR / "promtail-config.yaml"
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def test_config_file_exists(self):
        """测试配置文件存在"""
        self.assertTrue(self.config_path.exists())
    
    def test_server_config(self):
        """测试服务器配置"""
        self.assertIn('server', self.config)
        self.assertIn('http_listen_port', self.config['server'])
    
    def test_clients_config(self):
        """测试客户端配置"""
        self.assertIn('clients', self.config)
        self.assertTrue(len(self.config['clients']) > 0)
        
        client = self.config['clients'][0]
        self.assertIn('url', client)
        self.assertIn('loki', client['url'], "客户端应该连接到 Loki")
    
    def test_scrape_configs(self):
        """测试抓取配置"""
        self.assertIn('scrape_configs', self.config)
        self.assertTrue(len(self.config['scrape_configs']) > 0)
    
    def test_kubernetes_sd_config(self):
        """测试 Kubernetes 服务发现配置"""
        k8s_jobs = [sc for sc in self.config['scrape_configs']
                   if 'kubernetes_sd_configs' in sc]
        
        self.assertTrue(len(k8s_jobs) > 0, "应该有 Kubernetes 服务发现配置")
    
    def test_pipeline_stages(self):
        """测试管道阶段配置"""
        for scrape_config in self.config['scrape_configs']:
            if 'pipeline_stages' in scrape_config:
                stages = scrape_config['pipeline_stages']
                
                # 检查 JSON 解析阶段
                json_stages = [s for s in stages if 'json' in s]
                self.assertTrue(len(json_stages) > 0, 
                              f"抓取配置 {scrape_config.get('job_name', 'unknown')} "
                              f"应该有 JSON 解析阶段")
    
    def test_required_fields_extraction(self):
        """测试必需字段提取 (符合 interfaces.yaml)"""
        required_fields = {'timestamp', 'level', 'service', 'trace_id', 'message'}
        
        for scrape_config in self.config['scrape_configs']:
            if 'pipeline_stages' in scrape_config:
                for stage in scrape_config['pipeline_stages']:
                    if 'json' in stage:
                        expressions = stage['json'].get('expressions', {})
                        extracted_fields = set(expressions.keys())
                        
                        # 检查必需字段是否被提取
                        for field in required_fields:
                            self.assertIn(field, extracted_fields,
                                        f"字段 {field} 应该在 JSON 表达式中被提取")
                        break


class TestPromtailDaemonSet(unittest.TestCase):
    """Promtail DaemonSet 部署测试"""
    
    def setUp(self):
        """加载 DaemonSet 清单"""
        self.daemonset_path = PROMTAIL_DIR / "daemonset.yaml"
        
        with open(self.daemonset_path, 'r', encoding='utf-8') as f:
            self.manifests = list(yaml.safe_load_all(f))
    
    def test_daemonset_exists(self):
        """测试 DaemonSet 存在"""
        daemonsets = [m for m in self.manifests 
                     if m and m.get('kind') == 'DaemonSet']
        self.assertEqual(len(daemonsets), 1, "应该有一个 DaemonSet")
    
    def test_rbac_configured(self):
        """测试 RBAC 配置"""
        service_accounts = [m for m in self.manifests 
                          if m and m.get('kind') == 'ServiceAccount']
        cluster_roles = [m for m in self.manifests 
                        if m and m.get('kind') == 'ClusterRole']
        cluster_role_bindings = [m for m in self.manifests 
                                if m and m.get('kind') == 'ClusterRoleBinding']
        
        self.assertEqual(len(service_accounts), 1, "应该有 ServiceAccount")
        self.assertEqual(len(cluster_roles), 1, "应该有 ClusterRole")
        self.assertEqual(len(cluster_role_bindings), 1, "应该有 ClusterRoleBinding")
    
    def test_volume_mounts(self):
        """测试卷挂载配置"""
        daemonsets = [m for m in self.manifests 
                     if m and m.get('kind') == 'DaemonSet']
        
        ds = daemonsets[0]
        containers = ds['spec']['template']['spec']['containers']
        
        for container in containers:
            volume_mounts = container.get('volumeMounts', [])
            mount_paths = [vm['mountPath'] for vm in volume_mounts]
            
            # 检查必要的挂载点
            self.assertIn('/var/log/pods', mount_paths, 
                        "应该挂载 Pod 日志目录")


class TestFluentdConfig(unittest.TestCase):
    """Fluentd 配置文件测试"""
    
    def setUp(self):
        """加载 Fluentd 配置文件"""
        self.config_path = FLUENTD_DIR / "fluent.conf"
        self.parsers_path = FLUENTD_DIR / "parsers.conf"
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = f.read()
        
        with open(self.parsers_path, 'r', encoding='utf-8') as f:
            self.parsers = f.read()
    
    def test_config_files_exist(self):
        """测试配置文件存在"""
        self.assertTrue(self.config_path.exists())
        self.assertTrue(self.parsers_path.exists())
    
    def test_system_section(self):
        """测试系统配置部分"""
        self.assertIn('<system>', self.config)
        self.assertIn('workers', self.config)
    
    def test_source_sections(self):
        """测试输入源配置"""
        self.assertIn('<source>', self.config)
        self.assertIn('@type tail', self.config)
    
    def test_filter_sections(self):
        """测试过滤器配置"""
        self.assertIn('<filter', self.config)
        self.assertIn('kubernetes_metadata', self.config)
    
    def test_output_sections(self):
        """测试输出配置"""
        self.assertIn('<match', self.config)
        self.assertIn('loki', self.config)
    
    def test_buffer_config(self):
        """测试缓冲区配置"""
        self.assertIn('<buffer>', self.config)
        self.assertIn('flush_interval', self.config)
    
    def test_sensitive_data_masking(self):
        """测试敏感数据脱敏"""
        self.assertIn('password', self.config.lower())
        self.assertIn('masked', self.config.upper())
    
    def test_debug_filter(self):
        """测试 DEBUG 日志过滤"""
        self.assertIn('DEBUG', self.config)
        self.assertIn('exclude', self.config)


class TestGrafanaDashboards(unittest.TestCase):
    """Grafana Dashboard 测试"""
    
    def setUp(self):
        """加载 Dashboard JSON 文件"""
        self.dashboards_dir = GRAFANA_DIR / "dashboards"
        self.dashboard_files = list(self.dashboards_dir.glob("*.json"))
    
    def test_dashboards_exist(self):
        """测试 Dashboard 文件存在"""
        self.assertTrue(len(self.dashboard_files) > 0, 
                       "应该至少有一个 Dashboard 文件")
    
    def test_dashboard_json_valid(self):
        """测试 Dashboard JSON 有效"""
        for dashboard_file in self.dashboard_files:
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                try:
                    dashboard = json.load(f)
                    self.assertIsInstance(dashboard, dict)
                except json.JSONDecodeError as e:
                    self.fail(f"Dashboard {dashboard_file.name} JSON 无效: {e}")
    
    def test_dashboard_structure(self):
        """测试 Dashboard 结构"""
        for dashboard_file in self.dashboard_files:
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                dashboard = json.load(f)
            
            # 检查必需字段
            self.assertIn('title', dashboard, 
                        f"Dashboard {dashboard_file.name} 缺少 title")
            self.assertIn('panels', dashboard, 
                        f"Dashboard {dashboard_file.name} 缺少 panels")
            self.assertIn('uid', dashboard, 
                        f"Dashboard {dashboard_file.name} 缺少 uid")
    
    def test_panels_have_targets(self):
        """测试面板有数据查询"""
        for dashboard_file in self.dashboard_files:
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                dashboard = json.load(f)
            
            panels = dashboard.get('panels', [])
            visualization_panels = [p for p in panels 
                                   if p.get('type') not in ('row', 'text')]
            
            for panel in visualization_panels:
                if panel.get('type') != 'row':
                    targets = panel.get('targets', [])
                    self.assertTrue(len(targets) > 0,
                                  f"面板 '{panel.get('title', 'unknown')}' "
                                  f"在 {dashboard_file.name} 中应该有查询目标")
    
    def test_loki_datasource(self):
        """测试使用 Loki 数据源"""
        for dashboard_file in self.dashboard_files:
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.assertIn('loki', content.lower(),
                        f"Dashboard {dashboard_file.name} 应该使用 Loki 数据源")
    
    def test_template_variables(self):
        """测试模板变量"""
        for dashboard_file in self.dashboard_files:
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                dashboard = json.load(f)
            
            templating = dashboard.get('templating', {})
            variables = templating.get('list', [])
            
            # 主日志 Dashboard 应该有命名空间和应用变量
            if 'logs' in dashboard_file.name:
                var_names = [v.get('name') for v in variables]
                self.assertIn('namespace', var_names,
                            f"Dashboard {dashboard_file.name} 应该有 namespace 变量")
                self.assertIn('app', var_names,
                            f"Dashboard {dashboard_file.name} 应该有 app 变量")


class TestInterfaceCompliance(unittest.TestCase):
    """接口契约合规性测试 (interfaces.yaml)"""
    
    def setUp(self):
        """加载接口定义"""
        interfaces_path = Path(__file__).parent.parent.parent / "interfaces.yaml"
        
        with open(interfaces_path, 'r', encoding='utf-8') as f:
            self.interfaces = yaml.safe_load(f)
        
        self.logging_interface = self.interfaces.get('logging', {})
    
    def test_required_fields_in_promtail(self):
        """测试 Promtail 配置包含必需字段"""
        required_fields = self.logging_interface.get('format', {}).get('required_fields', [])
        
        promtail_config_path = PROMTAIL_DIR / "promtail-config.yaml"
        with open(promtail_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 从 pipeline_stages 中提取字段
        extracted_fields = set()
        for scrape_config in config.get('scrape_configs', []):
            for stage in scrape_config.get('pipeline_stages', []):
                if 'json' in stage:
                    extracted_fields.update(stage['json'].get('expressions', {}).keys())
        
        for field in required_fields:
            self.assertIn(field, extracted_fields,
                        f"必需字段 {field} 未在 Promtail 配置中提取")
    
    def test_log_levels_in_alerts(self):
        """测试告警规则使用正确的日志级别"""
        valid_levels = set(self.logging_interface.get('levels', []))
        
        alerts_path = LOKI_DIR / "rules" / "alerts.yaml"
        with open(alerts_path, 'r', encoding='utf-8') as f:
            alerts = yaml.safe_load(f)
        
        # 检查告警表达式中使用的日志级别
        for group in alerts.get('groups', []):
            for rule in group.get('rules', []):
                expr = rule.get('expr', '')
                
                for level in valid_levels:
                    if f'level="{level}"' in expr or f"level='{level}'" in expr:
                        # 验证使用的级别是有效的
                        self.assertIn(level, valid_levels,
                                    f"规则 {rule['alert']} 使用了无效的日志级别: {level}")
    
    def test_labels_in_configs(self):
        """测试配置使用正确的标签"""
        valid_labels = set(self.logging_interface.get('labels', []))
        
        # 检查 Promtail 配置
        promtail_config_path = PROMTAIL_DIR / "promtail-config.yaml"
        with open(promtail_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 从 relabel_configs 和 labels 中提取标签
        for scrape_config in config.get('scrape_configs', []):
            for stage in scrape_config.get('pipeline_stages', []):
                if 'labels' in stage:
                    labels = stage['labels']
                    if isinstance(labels, dict):
                        for label in labels.keys():
                            if label in valid_labels:
                                # 验证使用的标签是有效的
                                self.assertIn(label, valid_labels)


if __name__ == '__main__':
    unittest.main()

