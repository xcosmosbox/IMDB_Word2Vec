"""
性能基准测试模块单元测试

Author: Person F (MLOps Engineer)
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serving.config import BenchmarkConfig, ExportConfig
from serving.benchmark import (
    BenchmarkResult,
    TritonBenchmark,
    run_benchmark,
    MockTritonClient,
    create_mock_benchmark,
)


class TestBenchmarkResult:
    """BenchmarkResult 类测试"""
    
    def test_default_values(self):
        """测试默认值"""
        result = BenchmarkResult()
        
        assert result.throughput == 0.0
        assert result.latency_p50 == 0.0
        assert result.latency_p99 == 0.0
        assert result.total_requests == 0
        assert result.successful_requests == 0
        assert result.failed_requests == 0
    
    def test_custom_values(self):
        """测试自定义值"""
        result = BenchmarkResult(
            throughput=1000.0,
            latency_p50=5.0,
            latency_p99=15.0,
            total_requests=10000,
            successful_requests=9999,
            failed_requests=1,
            batch_size=32,
            seq_length=512,
        )
        
        assert result.throughput == 1000.0
        assert result.latency_p50 == 5.0
        assert result.latency_p99 == 15.0
        assert result.batch_size == 32
        assert result.seq_length == 512
    
    def test_to_dict(self):
        """测试转换为字典"""
        result = BenchmarkResult(
            throughput=500.0,
            latency_p99=10.0,
        )
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert d["throughput"] == 500.0
        assert d["latency_p99"] == 10.0
        assert "batch_size" in d
        assert "seq_length" in d
    
    def test_summary(self):
        """测试生成摘要"""
        result = BenchmarkResult(
            throughput=1000.0,
            latency_p50=5.0,
            latency_p90=8.0,
            latency_p95=10.0,
            latency_p99=15.0,
            latency_avg=6.0,
            latency_min=2.0,
            latency_max=20.0,
            total_requests=10000,
            successful_requests=9990,
            failed_requests=10,
            total_time=10.0,
            batch_size=16,
            seq_length=256,
        )
        
        summary = result.summary()
        
        assert isinstance(summary, str)
        assert "1000.00 req/s" in summary
        assert "P99" in summary
        assert "15.00" in summary
        assert "99.90%" in summary  # 成功率


class TestMockTritonClient:
    """MockTritonClient 类测试"""
    
    def test_init(self):
        """测试初始化"""
        client = MockTritonClient("localhost:8001")
        
        assert client.url == "localhost:8001"
        assert client.latency_range == (5.0, 15.0)
    
    def test_init_custom_latency(self):
        """测试自定义延迟范围"""
        client = MockTritonClient("localhost:8001", latency_range=(1.0, 5.0))
        
        assert client.latency_range == (1.0, 5.0)
    
    def test_infer_returns_dict(self):
        """测试推理返回字典"""
        client = MockTritonClient("localhost:8001", latency_range=(0.1, 0.2))
        
        result = client.infer("model", [])
        
        assert isinstance(result, dict)
        assert "recommendations" in result
        assert "scores" in result
    
    def test_infer_simulates_latency(self):
        """测试推理模拟延迟"""
        client = MockTritonClient("localhost:8001", latency_range=(50.0, 100.0))
        
        start = time.time()
        client.infer("model", [])
        elapsed = (time.time() - start) * 1000  # ms
        
        # 延迟应该在范围内
        assert elapsed >= 40  # 留一些误差
        assert elapsed <= 150
    
    def test_is_server_ready(self):
        """测试服务器就绪检查"""
        client = MockTritonClient("localhost:8001")
        
        assert client.is_server_ready() is True
    
    def test_is_model_ready(self):
        """测试模型就绪检查"""
        client = MockTritonClient("localhost:8001")
        
        assert client.is_model_ready("any_model") is True


class TestTritonBenchmark:
    """TritonBenchmark 类测试"""
    
    @pytest.fixture
    def benchmark_config(self):
        """创建测试配置"""
        return BenchmarkConfig(
            num_warmup_requests=10,
            num_requests=100,
            concurrency=1,
        )
    
    @pytest.fixture
    def export_config(self):
        """创建导出配置"""
        return ExportConfig(model_name="test_model")
    
    @pytest.fixture
    def benchmark(self, benchmark_config, export_config):
        """创建基准测试器"""
        return TritonBenchmark(benchmark_config, export_config)
    
    def test_init(self, benchmark_config, export_config):
        """测试初始化"""
        benchmark = TritonBenchmark(benchmark_config, export_config)
        
        assert benchmark.config == benchmark_config
        assert benchmark.export_config == export_config
    
    def test_init_with_defaults(self):
        """测试使用默认配置初始化"""
        benchmark = TritonBenchmark()
        
        assert benchmark.config is not None
        assert benchmark.export_config is not None
    
    @patch('serving.benchmark.TritonBenchmark._get_client')
    def test_run_calculates_metrics(self, mock_get_client, benchmark):
        """测试运行计算指标"""
        # 创建模拟客户端
        mock_client = MockTritonClient("localhost:8001", latency_range=(1.0, 2.0))
        mock_get_client.return_value = mock_client
        
        # 模拟 _create_inputs 方法
        with patch.object(benchmark, '_create_inputs', return_value=[]):
            result = benchmark.run("localhost:8001", "test_model", 1, 10)
        
        assert isinstance(result, BenchmarkResult)
        assert result.total_requests == 100
        assert result.throughput > 0
        assert result.latency_p99 > 0
    
    @patch('serving.benchmark.TritonBenchmark._get_client')
    def test_run_handles_failures(self, mock_get_client, benchmark):
        """测试运行处理失败"""
        mock_client = Mock()
        mock_client.infer.side_effect = Exception("Connection error")
        mock_get_client.return_value = mock_client
        
        with patch.object(benchmark, '_create_inputs', return_value=[]):
            result = benchmark.run("localhost:8001", "test_model")
        
        assert result.failed_requests == 100
        assert result.successful_requests == 0
    
    def test_generate_report(self, benchmark):
        """测试生成报告"""
        results = {
            "bs1_seq32": BenchmarkResult(
                throughput=1000.0,
                latency_p99=10.0,
                total_requests=100,
                successful_requests=100,
            ),
            "bs8_seq64": BenchmarkResult(
                throughput=800.0,
                latency_p99=15.0,
                total_requests=100,
                successful_requests=95,
            ),
        }
        
        report = benchmark.generate_report(results)
        
        assert isinstance(report, str)
        assert "bs1_seq32" in report
        assert "bs8_seq64" in report
        assert "吞吐量" in report or "req/s" in report
    
    def test_generate_report_saves_to_file(self, benchmark):
        """测试报告保存到文件"""
        import tempfile
        
        results = {
            "test": BenchmarkResult(throughput=100.0),
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = f.name
        
        try:
            benchmark.generate_report(results, output_path)
            
            assert os.path.exists(output_path)
            with open(output_path, 'r') as f:
                content = f.read()
            assert "test" in content
        finally:
            os.unlink(output_path)


class TestRunBenchmark:
    """run_benchmark 便捷函数测试"""
    
    @patch('serving.benchmark.TritonBenchmark.run')
    def test_returns_metrics_dict(self, mock_run):
        """测试返回指标字典"""
        mock_run.return_value = BenchmarkResult(
            throughput=500.0,
            latency_p50=5.0,
            latency_p90=8.0,
            latency_p99=12.0,
        )
        
        result = run_benchmark("localhost:8001", "model", 1000)
        
        assert isinstance(result, dict)
        assert result["throughput"] == 500.0
        assert result["latency_p50"] == 5.0
        assert result["latency_p90"] == 8.0
        assert result["latency_p99"] == 12.0
    
    @patch('serving.benchmark.TritonBenchmark.run')
    def test_passes_parameters(self, mock_run):
        """测试传递参数"""
        mock_run.return_value = BenchmarkResult()
        
        run_benchmark(
            "localhost:8001",
            "test_model",
            num_requests=5000,
            batch_size=16,
            seq_length=256
        )
        
        mock_run.assert_called_once_with(
            "localhost:8001",
            "test_model",
            16,
            256
        )


class TestCreateMockBenchmark:
    """create_mock_benchmark 函数测试"""
    
    def test_returns_tuple(self):
        """测试返回元组"""
        benchmark, client = create_mock_benchmark()
        
        assert isinstance(benchmark, TritonBenchmark)
        assert isinstance(client, MockTritonClient)
    
    def test_uses_parameters(self):
        """测试使用参数"""
        benchmark, client = create_mock_benchmark(
            num_requests=50,
            latency_range=(2.0, 4.0)
        )
        
        assert benchmark.config.num_requests == 50
        assert client.latency_range == (2.0, 4.0)


class TestTritonBenchmarkCreateInputs:
    """TritonBenchmark._create_inputs 方法测试"""
    
    def test_requires_tritonclient(self):
        """测试需要 tritonclient"""
        benchmark = TritonBenchmark()
        
        # 如果 tritonclient 不可用，应该抛出 ImportError
        with patch.dict('sys.modules', {'tritonclient.http': None}):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                benchmark._create_inputs(1, 10)


class TestTritonBenchmarkSingleRequest:
    """TritonBenchmark._single_request 方法测试"""
    
    def test_returns_success_and_latency(self):
        """测试返回成功和延迟"""
        benchmark = TritonBenchmark()
        
        mock_client = Mock()
        mock_client.infer.return_value = {}
        
        success, latency = benchmark._single_request(mock_client, "model", [])
        
        assert success is True
        assert latency > 0
    
    def test_returns_failure_on_exception(self):
        """测试异常时返回失败"""
        benchmark = TritonBenchmark()
        
        mock_client = Mock()
        mock_client.infer.side_effect = Exception("Error")
        
        success, latency = benchmark._single_request(mock_client, "model", [])
        
        assert success is False
        assert latency == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

