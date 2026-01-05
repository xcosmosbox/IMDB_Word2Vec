"""
报告生成器单元测试
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reports'))

from generate_report import (
    SLAConfig,
    SLA,
    K6DataExtractor,
    LocustDataExtractor,
    ReportGenerator,
)


# =============================================================================
# 测试数据
# =============================================================================

SAMPLE_K6_DATA = {
    "state": {
        "testRunDurationMs": 300000,
        "isStdErrTty": False,
    },
    "metrics": {
        "http_reqs": {
            "values": {
                "count": 10000,
                "rate": 33.33,
            }
        },
        "http_req_duration": {
            "values": {
                "avg": 45.5,
                "min": 10.0,
                "max": 500.0,
                "p(50)": 40.0,
                "p(90)": 80.0,
                "p(95)": 120.0,
                "p(99)": 180.0,
            },
            "thresholds": {
                "p(95)<200": {"ok": True},
            }
        },
        "http_req_failed": {
            "values": {
                "rate": 0.0005,
            }
        },
        "vus_max": {
            "values": {
                "max": 100,
            }
        }
    }
}

SAMPLE_LOCUST_DATA = {
    "start_time": 1704067200,
    "last_request_timestamp": 1704067500,
    "stats": [
        {
            "name": "Aggregated",
            "num_requests": 5000,
            "num_failures": 5,
            "total_rps": 16.67,
            "avg_response_time": 55.0,
            "min_response_time": 15.0,
            "max_response_time": 450.0,
            "response_time_percentile_50": 45.0,
            "response_time_percentile_90": 90.0,
            "response_time_percentile_95": 130.0,
            "response_time_percentile_99": 200.0,
        }
    ]
}


# =============================================================================
# SLAConfig 测试
# =============================================================================

class TestSLAConfig:
    """SLAConfig 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        sla = SLAConfig()
        
        assert sla.availability == 0.999
        assert sla.p50_latency == 50
        assert sla.p99_latency == 200
        assert sla.error_rate == 0.001
    
    def test_custom_values(self):
        """测试自定义值"""
        sla = SLAConfig(
            availability=0.99,
            p50_latency=100,
            p99_latency=500,
            error_rate=0.01,
        )
        
        assert sla.availability == 0.99
        assert sla.p99_latency == 500
    
    def test_global_sla_instance(self):
        """测试全局 SLA 实例"""
        assert SLA.availability == 0.999
        assert SLA.p99_latency == 200


# =============================================================================
# K6DataExtractor 测试
# =============================================================================

class TestK6DataExtractor:
    """K6DataExtractor 测试"""
    
    @pytest.fixture
    def extractor(self):
        """创建提取器实例"""
        return K6DataExtractor(SAMPLE_K6_DATA)
    
    def test_get_duration(self, extractor):
        """测试获取持续时间"""
        duration = extractor.get_duration()
        assert duration == 300.0  # 300000ms = 300s
    
    def test_get_total_requests(self, extractor):
        """测试获取总请求数"""
        total = extractor.get_total_requests()
        assert total == 10000
    
    def test_get_rps(self, extractor):
        """测试获取 RPS"""
        rps = extractor.get_rps()
        assert rps == 33.33
    
    def test_get_latency_stats(self, extractor):
        """测试获取延迟统计"""
        latency = extractor.get_latency_stats()
        
        assert latency['avg'] == 45.5
        assert latency['p50'] == 40.0
        assert latency['p99'] == 180.0
        assert latency['max'] == 500.0
    
    def test_get_error_rate(self, extractor):
        """测试获取错误率"""
        error_rate = extractor.get_error_rate()
        assert error_rate == 0.0005
    
    def test_get_thresholds(self, extractor):
        """测试获取阈值"""
        thresholds = extractor.get_thresholds()
        
        assert len(thresholds) > 0
        assert any(t['name'].startswith('http_req_duration') for t in thresholds)
    
    def test_get_vus_max(self, extractor):
        """测试获取最大 VU 数"""
        vus_max = extractor.get_vus_max()
        assert vus_max == 100
    
    def test_get_metric_values_missing(self, extractor):
        """测试获取不存在的指标"""
        values = extractor.get_metric_values('nonexistent_metric')
        assert values == {}


# =============================================================================
# LocustDataExtractor 测试
# =============================================================================

class TestLocustDataExtractor:
    """LocustDataExtractor 测试"""
    
    @pytest.fixture
    def extractor(self):
        """创建提取器实例"""
        return LocustDataExtractor(SAMPLE_LOCUST_DATA)
    
    def test_get_duration(self, extractor):
        """测试获取持续时间"""
        duration = extractor.get_duration()
        assert duration == 300  # 1704067500 - 1704067200
    
    def test_get_total_requests(self, extractor):
        """测试获取总请求数"""
        total = extractor.get_total_requests()
        assert total == 5000
    
    def test_get_rps(self, extractor):
        """测试获取 RPS"""
        rps = extractor.get_rps()
        assert rps == 16.67
    
    def test_get_latency_stats(self, extractor):
        """测试获取延迟统计"""
        latency = extractor.get_latency_stats()
        
        assert latency['avg'] == 55.0
        assert latency['p50'] == 45.0
        assert latency['p99'] == 200.0
    
    def test_get_error_rate(self, extractor):
        """测试获取错误率"""
        error_rate = extractor.get_error_rate()
        expected = 5 / 5000  # 5 failures / 5000 requests
        assert error_rate == expected


# =============================================================================
# ReportGenerator 测试
# =============================================================================

class TestReportGenerator:
    """ReportGenerator 测试"""
    
    @pytest.fixture
    def temp_k6_file(self):
        """创建临时 K6 数据文件"""
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            delete=False
        ) as f:
            json.dump(SAMPLE_K6_DATA, f)
            return f.name
    
    @pytest.fixture
    def temp_locust_file(self):
        """创建临时 Locust 数据文件"""
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            delete=False
        ) as f:
            json.dump(SAMPLE_LOCUST_DATA, f)
            return f.name
    
    def test_load_k6_data(self, temp_k6_file):
        """测试加载 K6 数据"""
        generator = ReportGenerator(temp_k6_file)
        
        assert isinstance(generator.extractor, K6DataExtractor)
        os.unlink(temp_k6_file)
    
    def test_load_locust_data(self, temp_locust_file):
        """测试加载 Locust 数据"""
        generator = ReportGenerator(temp_locust_file)
        
        assert isinstance(generator.extractor, LocustDataExtractor)
        os.unlink(temp_locust_file)
    
    def test_detect_test_type_baseline(self, temp_k6_file):
        """测试检测基线测试类型"""
        # 重命名文件以包含 'baseline'
        baseline_file = temp_k6_file.replace('.json', '_baseline.json')
        os.rename(temp_k6_file, baseline_file)
        
        generator = ReportGenerator(baseline_file)
        test_type = generator._detect_test_type()
        
        assert test_type == 'Baseline Test'
        os.unlink(baseline_file)
    
    def test_detect_test_type_stress(self):
        """测试检测压力测试类型"""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='_stress.json',
            delete=False
        ) as f:
            json.dump(SAMPLE_K6_DATA, f)
            stress_file = f.name
        
        generator = ReportGenerator(stress_file)
        test_type = generator._detect_test_type()
        
        assert test_type == 'Stress Test'
        os.unlink(stress_file)
    
    def test_check_sla_passed_success(self, temp_k6_file):
        """测试 SLA 检查 - 通过"""
        generator = ReportGenerator(temp_k6_file)
        latency = generator.extractor.get_latency_stats()
        
        # K6 示例数据应该通过 SLA
        passed = generator._check_sla_passed(latency)
        assert passed is True
        os.unlink(temp_k6_file)
    
    def test_check_sla_passed_failure(self):
        """测试 SLA 检查 - 失败"""
        # 创建一个超过 SLA 的数据
        bad_data = {
            "state": {"testRunDurationMs": 60000},
            "metrics": {
                "http_reqs": {"values": {"count": 1000, "rate": 16.67}},
                "http_req_duration": {
                    "values": {
                        "avg": 100,
                        "p50": 80,
                        "p99": 500,  # 超过 200ms SLA
                        "min": 10,
                        "max": 1000,
                        "p90": 150,
                        "p95": 300,
                    }
                },
                "http_req_failed": {"values": {"rate": 0.05}},  # 超过 0.1% SLA
            }
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as f:
            json.dump(bad_data, f)
            bad_file = f.name
        
        generator = ReportGenerator(bad_file)
        latency = generator.extractor.get_latency_stats()
        
        passed = generator._check_sla_passed(latency)
        assert passed is False
        os.unlink(bad_file)
    
    def test_generate_json_report(self, temp_k6_file):
        """测试生成 JSON 报告"""
        generator = ReportGenerator(temp_k6_file)
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as f:
            output_file = f.name
        
        result = generator.generate_json(output_file)
        
        assert result is True
        assert os.path.exists(output_file)
        
        # 验证 JSON 内容
        with open(output_file, 'r') as f:
            report = json.load(f)
        
        assert 'timestamp' in report
        assert 'summary' in report
        assert 'latency' in report
        
        os.unlink(temp_k6_file)
        os.unlink(output_file)
    
    def test_generate_junit_report(self, temp_k6_file):
        """测试生成 JUnit 报告"""
        generator = ReportGenerator(temp_k6_file)
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.xml',
            delete=False
        ) as f:
            output_file = f.name
        
        result = generator.generate_junit(output_file)
        
        assert result is True
        assert os.path.exists(output_file)
        
        # 验证 XML 内容
        with open(output_file, 'r') as f:
            content = f.read()
        
        assert 'testsuite' in content
        assert 'testcase' in content
        
        os.unlink(temp_k6_file)
        os.unlink(output_file)


# =============================================================================
# 集成测试
# =============================================================================

class TestReportGeneratorIntegration:
    """报告生成器集成测试"""
    
    def test_full_report_generation_flow(self):
        """测试完整报告生成流程"""
        # 创建测试数据文件
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='_baseline.json',
            delete=False
        ) as f:
            json.dump(SAMPLE_K6_DATA, f)
            input_file = f.name
        
        try:
            generator = ReportGenerator(input_file)
            
            # 检查数据加载
            assert generator.extractor.get_total_requests() == 10000
            
            # 生成 JSON 报告
            json_output = input_file.replace('.json', '_report.json')
            generator.generate_json(json_output)
            assert os.path.exists(json_output)
            
            # 生成 JUnit 报告
            junit_output = input_file.replace('.json', '_junit.xml')
            generator.generate_junit(junit_output)
            assert os.path.exists(junit_output)
            
            # 清理
            os.unlink(json_output)
            os.unlink(junit_output)
        finally:
            os.unlink(input_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

