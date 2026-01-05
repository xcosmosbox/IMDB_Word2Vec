#!/usr/bin/env python3
"""
性能测试报告生成器

从 K6 或 Locust 的 JSON 输出生成 HTML 报告。

使用方法:
    python generate-report.py results.json -o report.html
    python generate-report.py results.json --format html,json,junit
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import xml.etree.ElementTree as ET

try:
    from jinja2 import Template, Environment, FileSystemLoader
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


# =============================================================================
# 配置
# =============================================================================

@dataclass
class SLAConfig:
    """SLA 配置"""
    availability: float = 0.999  # 99.9%
    p50_latency: float = 50  # ms
    p99_latency: float = 200  # ms
    error_rate: float = 0.001  # 0.1%


SLA = SLAConfig()


# =============================================================================
# 数据提取器
# =============================================================================

class K6DataExtractor:
    """K6 数据提取器"""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.metrics = data.get('metrics', {})
        self.state = data.get('state', {})
    
    def get_metric_values(self, metric_name: str) -> Dict[str, float]:
        """获取指标值"""
        metric = self.metrics.get(metric_name, {})
        return metric.get('values', {})
    
    def get_duration(self) -> float:
        """获取测试持续时间 (秒)"""
        return self.state.get('testRunDurationMs', 0) / 1000
    
    def get_total_requests(self) -> int:
        """获取总请求数"""
        return self.get_metric_values('http_reqs').get('count', 0)
    
    def get_rps(self) -> float:
        """获取每秒请求数"""
        return self.get_metric_values('http_reqs').get('rate', 0)
    
    def get_latency_stats(self) -> Dict[str, float]:
        """获取延迟统计"""
        values = self.get_metric_values('http_req_duration')
        return {
            'avg': values.get('avg', 0),
            'min': values.get('min', 0),
            'max': values.get('max', 0),
            'p50': values.get('p(50)', 0),
            'p90': values.get('p(90)', 0),
            'p95': values.get('p(95)', 0),
            'p99': values.get('p(99)', 0),
        }
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        return self.get_metric_values('http_req_failed').get('rate', 0)
    
    def get_thresholds(self) -> List[Dict[str, Any]]:
        """获取阈值检查结果"""
        results = []
        for name, metric in self.metrics.items():
            if metric.get('thresholds'):
                for t_name, t_val in metric['thresholds'].items():
                    results.append({
                        'name': f"{name}: {t_name}",
                        'passed': t_val.get('ok', False),
                    })
        return results
    
    def get_vus_max(self) -> int:
        """获取最大 VU 数"""
        return int(self.get_metric_values('vus_max').get('max', 0))


class LocustDataExtractor:
    """Locust 数据提取器"""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.stats = data.get('stats', [])
        self.total = next((s for s in self.stats if s.get('name') == 'Aggregated'), {})
    
    def get_duration(self) -> float:
        """获取测试持续时间"""
        return self.data.get('last_request_timestamp', 0) - self.data.get('start_time', 0)
    
    def get_total_requests(self) -> int:
        """获取总请求数"""
        return self.total.get('num_requests', 0)
    
    def get_rps(self) -> float:
        """获取每秒请求数"""
        return self.total.get('total_rps', 0)
    
    def get_latency_stats(self) -> Dict[str, float]:
        """获取延迟统计"""
        return {
            'avg': self.total.get('avg_response_time', 0),
            'min': self.total.get('min_response_time', 0),
            'max': self.total.get('max_response_time', 0),
            'p50': self.total.get('response_time_percentile_50', 0),
            'p90': self.total.get('response_time_percentile_90', 0),
            'p95': self.total.get('response_time_percentile_95', 0),
            'p99': self.total.get('response_time_percentile_99', 0),
        }
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        total = self.total.get('num_requests', 0)
        failures = self.total.get('num_failures', 0)
        return failures / total if total > 0 else 0


# =============================================================================
# 报告生成器
# =============================================================================

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.data = self._load_data()
        self.extractor = self._create_extractor()
    
    def _load_data(self) -> Dict[str, Any]:
        """加载数据文件"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_extractor(self):
        """创建数据提取器"""
        # 根据数据格式判断来源
        if 'metrics' in self.data and 'state' in self.data:
            return K6DataExtractor(self.data)
        elif 'stats' in self.data:
            return LocustDataExtractor(self.data)
        else:
            # 默认使用 K6 提取器
            return K6DataExtractor(self.data)
    
    def generate_html(self, output_file: str, template_file: Optional[str] = None):
        """生成 HTML 报告"""
        if not HAS_JINJA2:
            print("Error: jinja2 is required for HTML report generation.")
            print("Install with: pip install jinja2")
            return False
        
        # 加载模板
        if template_file:
            template_path = Path(template_file)
            env = Environment(loader=FileSystemLoader(template_path.parent))
            template = env.get_template(template_path.name)
        else:
            # 使用内置模板
            template = Template(self._get_default_template())
        
        # 准备数据
        latency = self.extractor.get_latency_stats()
        error_rate = self.extractor.get_error_rate() * 100
        
        # SLA 检查
        sla_checks = [
            {
                'name': '可用性',
                'target': '≥ 99.9%',
                'actual': f"{(1 - self.extractor.get_error_rate()) * 100:.2f}%",
                'passed': self.extractor.get_error_rate() < SLA.error_rate,
            },
            {
                'name': 'P50 延迟',
                'target': '≤ 50ms',
                'actual': f"{latency['p50']:.2f}ms",
                'passed': latency['p50'] <= SLA.p50_latency,
            },
            {
                'name': 'P99 延迟',
                'target': '≤ 200ms',
                'actual': f"{latency['p99']:.2f}ms",
                'passed': latency['p99'] <= SLA.p99_latency,
            },
            {
                'name': '错误率',
                'target': '≤ 0.1%',
                'actual': f"{error_rate:.4f}%",
                'passed': self.extractor.get_error_rate() <= SLA.error_rate,
            },
        ]
        
        # 阈值检查
        thresholds = []
        if hasattr(self.extractor, 'get_thresholds'):
            thresholds = self.extractor.get_thresholds()
        
        # 模拟 RPS 趋势数据
        duration = self.extractor.get_duration()
        rps = self.extractor.get_rps()
        rps_labels = [f"{i}s" for i in range(0, int(duration) + 1, max(1, int(duration // 10)))]
        rps_values = [rps * (0.8 + 0.4 * (i / len(rps_labels))) for i in range(len(rps_labels))]
        
        # 渲染模板
        context = {
            'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_type': self._detect_test_type(),
            'environment': os.getenv('TEST_ENV', 'local'),
            'duration': self.extractor.get_duration(),
            'total_requests': self.extractor.get_total_requests(),
            'rps': self.extractor.get_rps(),
            'avg_latency': latency['avg'],
            'error_rate': error_rate,
            'p50': latency['p50'],
            'p90': latency['p90'],
            'p95': latency['p95'],
            'p99': latency['p99'],
            'max_latency': latency['max'],
            'sla_checks': sla_checks,
            'thresholds': thresholds,
            'rps_trend_labels': rps_labels,
            'rps_trend_values': rps_values,
        }
        
        # 添加自定义过滤器
        def format_number(value):
            return "{:,}".format(int(value))
        
        if hasattr(template, 'globals'):
            template.globals['format_number'] = format_number
        
        html = template.render(**context)
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"HTML report generated: {output_file}")
        return True
    
    def generate_json(self, output_file: str):
        """生成 JSON 报告"""
        latency = self.extractor.get_latency_stats()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_type': self._detect_test_type(),
            'duration': self.extractor.get_duration(),
            'summary': {
                'total_requests': self.extractor.get_total_requests(),
                'rps': self.extractor.get_rps(),
                'error_rate': self.extractor.get_error_rate(),
            },
            'latency': latency,
            'sla_passed': self._check_sla_passed(latency),
            'raw_data': self.data,
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"JSON report generated: {output_file}")
        return True
    
    def generate_junit(self, output_file: str):
        """生成 JUnit XML 报告 (用于 CI/CD 集成)"""
        latency = self.extractor.get_latency_stats()
        error_rate = self.extractor.get_error_rate()
        
        # 创建 XML 结构
        testsuite = ET.Element('testsuite')
        testsuite.set('name', 'Performance Tests')
        testsuite.set('tests', '4')
        testsuite.set('time', str(self.extractor.get_duration()))
        
        failures = 0
        
        # SLA 测试用例
        test_cases = [
            ('availability', '可用性', (1 - error_rate) >= SLA.availability,
             f"Availability: {(1 - error_rate) * 100:.2f}%"),
            ('p50_latency', 'P50 延迟', latency['p50'] <= SLA.p50_latency,
             f"P50 Latency: {latency['p50']:.2f}ms"),
            ('p99_latency', 'P99 延迟', latency['p99'] <= SLA.p99_latency,
             f"P99 Latency: {latency['p99']:.2f}ms"),
            ('error_rate', '错误率', error_rate <= SLA.error_rate,
             f"Error Rate: {error_rate * 100:.4f}%"),
        ]
        
        for name, display_name, passed, message in test_cases:
            testcase = ET.SubElement(testsuite, 'testcase')
            testcase.set('name', display_name)
            testcase.set('classname', 'PerformanceTest')
            
            if not passed:
                failures += 1
                failure = ET.SubElement(testcase, 'failure')
                failure.set('message', f"SLA violation: {message}")
        
        testsuite.set('failures', str(failures))
        
        # 写入文件
        tree = ET.ElementTree(testsuite)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        print(f"JUnit report generated: {output_file}")
        return True
    
    def _detect_test_type(self) -> str:
        """检测测试类型"""
        filename = Path(self.data_file).stem.lower()
        if 'baseline' in filename:
            return 'Baseline Test'
        elif 'stress' in filename:
            return 'Stress Test'
        elif 'spike' in filename:
            return 'Spike Test'
        else:
            return 'Load Test'
    
    def _check_sla_passed(self, latency: Dict[str, float]) -> bool:
        """检查 SLA 是否通过"""
        error_rate = self.extractor.get_error_rate()
        return (
            error_rate <= SLA.error_rate and
            latency['p50'] <= SLA.p50_latency and
            latency['p99'] <= SLA.p99_latency
        )
    
    def _get_default_template(self) -> str:
        """获取默认模板"""
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Performance Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .metric { display: inline-block; margin: 10px; padding: 20px; background: #f0f0f0; border-radius: 8px; }
        .metric-value { font-size: 24px; font-weight: bold; }
        .pass { color: green; }
        .fail { color: red; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1>Performance Test Report</h1>
    <p>Generated: {{ report_time }} | Type: {{ test_type }} | Duration: {{ duration }}s</p>
    
    <div class="metric">
        <div class="metric-value">{{ total_requests }}</div>
        <div>Total Requests</div>
    </div>
    <div class="metric">
        <div class="metric-value">{{ rps | round(2) }}</div>
        <div>RPS</div>
    </div>
    <div class="metric">
        <div class="metric-value">{{ avg_latency | round(2) }}ms</div>
        <div>Avg Latency</div>
    </div>
    <div class="metric">
        <div class="metric-value">{{ error_rate | round(4) }}%</div>
        <div>Error Rate</div>
    </div>
    
    <h2>SLA Checks</h2>
    <table>
        <tr><th>Metric</th><th>Target</th><th>Actual</th><th>Status</th></tr>
        {% for check in sla_checks %}
        <tr>
            <td>{{ check.name }}</td>
            <td>{{ check.target }}</td>
            <td>{{ check.actual }}</td>
            <td class="{{ 'pass' if check.passed else 'fail' }}">
                {{ 'PASS' if check.passed else 'FAIL' }}
            </td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>'''


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate performance test reports')
    parser.add_argument('input', help='Input JSON file from K6 or Locust')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-f', '--format', default='html',
                        help='Output format: html, json, junit, all (comma-separated)')
    parser.add_argument('-t', '--template', help='Custom HTML template file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # 创建报告生成器
    generator = ReportGenerator(args.input)
    
    # 解析格式
    formats = [f.strip().lower() for f in args.format.split(',')]
    if 'all' in formats:
        formats = ['html', 'json', 'junit']
    
    # 生成基础输出文件名
    input_path = Path(args.input)
    base_output = args.output if args.output else str(input_path.with_suffix(''))
    
    # 生成报告
    success = True
    for fmt in formats:
        if fmt == 'html':
            output_file = f"{base_output}.html" if not args.output else args.output
            if not generator.generate_html(output_file, args.template):
                success = False
        elif fmt == 'json':
            output_file = f"{base_output}_report.json"
            if not generator.generate_json(output_file):
                success = False
        elif fmt == 'junit':
            output_file = f"{base_output}_junit.xml"
            if not generator.generate_junit(output_file):
                success = False
        else:
            print(f"Unknown format: {fmt}")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

