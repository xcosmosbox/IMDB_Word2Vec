"""
推理基准测试模块单元测试
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import asdict

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmark'))

from inference_benchmark import (
    BenchmarkConfig,
    LatencyStats,
    BenchmarkResult,
    InferenceClient,
    InferenceBenchmark,
    SLAChecker,
)


# =============================================================================
# BenchmarkConfig 测试
# =============================================================================

class TestBenchmarkConfig:
    """BenchmarkConfig 测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = BenchmarkConfig()
        
        assert config.host == "localhost"
        assert config.port == 50051
        assert config.http_port == 8080
        assert config.duration == 60
        assert config.warmup == 10
        assert config.concurrency == 10
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = BenchmarkConfig(
            host="api.example.com",
            port=9090,
            duration=120,
            concurrency=20,
        )
        
        assert config.host == "api.example.com"
        assert config.port == 9090
        assert config.duration == 120
        assert config.concurrency == 20
    
    def test_batch_sizes_default(self):
        """测试默认批量大小"""
        config = BenchmarkConfig()
        
        assert 1 in config.batch_sizes
        assert 8 in config.batch_sizes
    
    def test_sequence_lengths_default(self):
        """测试默认序列长度"""
        config = BenchmarkConfig()
        
        assert 64 in config.sequence_lengths
        assert 128 in config.sequence_lengths


# =============================================================================
# LatencyStats 测试
# =============================================================================

class TestLatencyStats:
    """LatencyStats 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        stats = LatencyStats()
        
        assert stats.count == 0
        assert stats.avg == 0.0
        assert stats.min == float('inf')
        assert stats.max == 0.0
    
    def test_with_values(self):
        """测试设置值"""
        stats = LatencyStats(
            count=100,
            avg=45.5,
            min=10.0,
            max=200.0,
            p50=40.0,
            p90=80.0,
            p95=120.0,
            p99=180.0,
        )
        
        assert stats.count == 100
        assert stats.avg == 45.5
        assert stats.p99 == 180.0


# =============================================================================
# BenchmarkResult 测试
# =============================================================================

class TestBenchmarkResult:
    """BenchmarkResult 测试"""
    
    def test_create_result(self):
        """测试创建结果"""
        result = BenchmarkResult(
            name="Test Benchmark",
            batch_size=8,
            sequence_length=128,
            protocol="http",
            duration=60.0,
            total_requests=1000,
            success_count=995,
            fail_count=5,
            rps=16.67,
            throughput_samples_per_sec=133.36,
            latency=LatencyStats(count=995, avg=50.0, p99=150.0),
        )
        
        assert result.name == "Test Benchmark"
        assert result.batch_size == 8
        assert result.success_count == 995
        assert result.rps == 16.67
    
    def test_result_to_dict(self):
        """测试结果转字典"""
        result = BenchmarkResult(
            name="Test",
            batch_size=1,
            sequence_length=64,
            protocol="http",
            duration=10.0,
            total_requests=100,
            success_count=100,
            fail_count=0,
            rps=10.0,
            throughput_samples_per_sec=10.0,
            latency=LatencyStats(),
        )
        
        result_dict = asdict(result)
        
        assert isinstance(result_dict, dict)
        assert result_dict['name'] == "Test"
        assert 'latency' in result_dict


# =============================================================================
# InferenceClient 测试
# =============================================================================

class TestInferenceClient:
    """InferenceClient 测试"""
    
    @pytest.fixture
    def client(self):
        """创建客户端实例"""
        config = BenchmarkConfig()
        return InferenceClient(config)
    
    def test_create_client(self, client):
        """测试创建客户端"""
        assert client.http_base_url == "http://localhost:8080"
    
    @patch('inference_benchmark.requests')
    def test_http_inference_success(self, mock_requests, client):
        """测试 HTTP 推理成功"""
        # 模拟响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.elapsed.total_seconds.return_value = 0.05
        mock_requests.post.return_value = mock_response
        
        result = client.http_inference("user_1", [1, 2, 3], 20)
        
        assert result['success'] is True
        assert result['status_code'] == 200
        assert result['latency_ms'] == 50.0
    
    @patch('inference_benchmark.requests')
    def test_http_inference_failure(self, mock_requests, client):
        """测试 HTTP 推理失败"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.elapsed.total_seconds.return_value = 0.1
        mock_requests.post.return_value = mock_response
        
        result = client.http_inference("user_1", [1, 2, 3], 20)
        
        assert result['success'] is False
        assert result['status_code'] == 500
    
    def test_grpc_inference(self, client):
        """测试 gRPC 推理 (模拟)"""
        result = client.grpc_inference("user_1", [1, 2, 3], 20)
        
        assert 'success' in result
        assert 'latency_ms' in result
        assert result['latency_ms'] > 0


# =============================================================================
# InferenceBenchmark 测试
# =============================================================================

class TestInferenceBenchmark:
    """InferenceBenchmark 测试"""
    
    @pytest.fixture
    def benchmark(self):
        """创建基准测试实例"""
        config = BenchmarkConfig(
            duration=1,  # 短时间测试
            warmup=0,
            concurrency=2,
            batch_sizes=[1],
            sequence_lengths=[64],
            use_http=False,
            use_grpc=True,
        )
        return InferenceBenchmark(config)
    
    def test_generate_test_data(self, benchmark):
        """测试生成测试数据"""
        data = benchmark.generate_test_data(batch_size=4, sequence_length=128)
        
        assert len(data) == 4
        assert all('user_id' in d for d in data)
        assert all('sequence' in d for d in data)
        assert all(len(d['sequence']) == 128 for d in data)
    
    def test_calculate_latency_stats(self, benchmark):
        """测试计算延迟统计"""
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        
        stats = benchmark._calculate_latency_stats(latencies)
        
        assert stats.count == 10
        assert stats.avg == 55.0
        assert stats.min == 10.0
        assert stats.max == 100.0
        assert stats.p50 == 50.0
    
    def test_calculate_latency_stats_empty(self, benchmark):
        """测试计算空延迟统计"""
        stats = benchmark._calculate_latency_stats([])
        
        assert stats.count == 0
        assert stats.avg == 0.0


# =============================================================================
# SLAChecker 测试
# =============================================================================

class TestSLAChecker:
    """SLAChecker 测试"""
    
    @pytest.fixture
    def passing_results(self):
        """创建通过 SLA 的结果"""
        return [
            BenchmarkResult(
                name="Test 1",
                batch_size=1,
                sequence_length=64,
                protocol="http",
                duration=60.0,
                total_requests=1000,
                success_count=999,  # 99.9% 成功率
                fail_count=1,
                rps=100,
                throughput_samples_per_sec=100,
                latency=LatencyStats(
                    count=999,
                    avg=30.0,
                    min=10.0,
                    max=150.0,
                    p50=25.0,  # < 50ms
                    p90=50.0,
                    p95=80.0,
                    p99=150.0,  # < 200ms
                ),
            )
        ]
    
    @pytest.fixture
    def failing_results(self):
        """创建失败 SLA 的结果"""
        return [
            BenchmarkResult(
                name="Test 2",
                batch_size=1,
                sequence_length=64,
                protocol="http",
                duration=60.0,
                total_requests=1000,
                success_count=900,  # 90% 成功率 (低于 99.9%)
                fail_count=100,
                rps=50,  # 低于 100 RPS
                throughput_samples_per_sec=50,
                latency=LatencyStats(
                    count=900,
                    avg=100.0,
                    min=20.0,
                    max=1000.0,
                    p50=80.0,  # > 50ms
                    p90=200.0,
                    p95=400.0,
                    p99=500.0,  # > 200ms
                ),
            )
        ]
    
    def test_check_passing(self, passing_results):
        """测试 SLA 检查通过"""
        checker = SLAChecker(passing_results)
        result = checker.check()
        
        assert result['passed'] is True
        assert len(result['violations']) == 0
    
    def test_check_failing(self, failing_results):
        """测试 SLA 检查失败"""
        checker = SLAChecker(failing_results)
        result = checker.check()
        
        assert result['passed'] is False
        assert len(result['violations']) > 0
    
    def test_check_violations_details(self, failing_results):
        """测试 SLA 违规详情"""
        checker = SLAChecker(failing_results)
        result = checker.check()
        
        violations = result['violations']
        
        # 检查是否包含预期的违规
        violation_metrics = [v['metric'] for v in violations]
        
        assert 'P50 Latency' in violation_metrics
        assert 'P99 Latency' in violation_metrics
    
    def test_sla_targets(self, passing_results):
        """测试 SLA 目标值"""
        checker = SLAChecker(passing_results)
        
        assert checker.sla['p50_latency_ms'] == 50
        assert checker.sla['p99_latency_ms'] == 200
        assert checker.sla['success_rate_pct'] == 99.9
        assert checker.sla['min_rps'] == 100


# =============================================================================
# 边界条件测试
# =============================================================================

class TestEdgeCases:
    """边界条件测试"""
    
    def test_empty_results_sla_check(self):
        """测试空结果 SLA 检查"""
        checker = SLAChecker([])
        result = checker.check()
        
        assert result['passed'] is True
        assert result['violations'] == []
    
    def test_zero_requests(self):
        """测试零请求结果"""
        result = BenchmarkResult(
            name="Zero Test",
            batch_size=1,
            sequence_length=64,
            protocol="http",
            duration=60.0,
            total_requests=0,
            success_count=0,
            fail_count=0,
            rps=0,
            throughput_samples_per_sec=0,
            latency=LatencyStats(),
        )
        
        checker = SLAChecker([result])
        check_result = checker.check()
        
        # 零请求不应该导致崩溃
        assert 'passed' in check_result
    
    def test_single_latency_value(self):
        """测试单个延迟值"""
        config = BenchmarkConfig()
        benchmark = InferenceBenchmark(config)
        
        stats = benchmark._calculate_latency_stats([50.0])
        
        assert stats.count == 1
        assert stats.avg == 50.0
        assert stats.p50 == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

