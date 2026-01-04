"""
性能基准测试模块

对部署的 Triton 推理服务进行性能测试。

特性：
1. 延迟测试 - P50/P90/P95/P99 延迟
2. 吞吐量测试 - QPS 测量
3. 并发测试 - 多客户端并发请求
4. 多维度测试 - 不同批次大小和序列长度

Author: Person F (MLOps Engineer)
"""

import os
import time
import logging
import statistics
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np

from .config import BenchmarkConfig, ExportConfig, ModelInputSpec

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """
    基准测试结果
    
    Attributes:
        throughput: 吞吐量（请求/秒）
        latency_p50: P50 延迟（毫秒）
        latency_p90: P90 延迟（毫秒）
        latency_p95: P95 延迟（毫秒）
        latency_p99: P99 延迟（毫秒）
        latency_avg: 平均延迟（毫秒）
        latency_min: 最小延迟（毫秒）
        latency_max: 最大延迟（毫秒）
        total_requests: 总请求数
        successful_requests: 成功请求数
        failed_requests: 失败请求数
        total_time: 总测试时间（秒）
        batch_size: 测试批次大小
        seq_length: 测试序列长度
    """
    throughput: float = 0.0
    latency_p50: float = 0.0
    latency_p90: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_avg: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0.0
    batch_size: int = 1
    seq_length: int = 100
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            "throughput": self.throughput,
            "latency_p50": self.latency_p50,
            "latency_p90": self.latency_p90,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "latency_avg": self.latency_avg,
            "latency_min": self.latency_min,
            "latency_max": self.latency_max,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_time": self.total_time,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
        }
    
    def summary(self) -> str:
        """生成结果摘要"""
        success_rate = (self.successful_requests / self.total_requests * 100
                       if self.total_requests > 0 else 0)
        
        return f"""
性能测试结果摘要:
================
配置: batch_size={self.batch_size}, seq_length={self.seq_length}
请求统计:
  - 总请求数: {self.total_requests}
  - 成功请求: {self.successful_requests} ({success_rate:.2f}%)
  - 失败请求: {self.failed_requests}
  - 总测试时间: {self.total_time:.2f}s

吞吐量: {self.throughput:.2f} req/s

延迟统计 (ms):
  - P50: {self.latency_p50:.2f}
  - P90: {self.latency_p90:.2f}
  - P95: {self.latency_p95:.2f}
  - P99: {self.latency_p99:.2f}
  - 平均: {self.latency_avg:.2f}
  - 最小: {self.latency_min:.2f}
  - 最大: {self.latency_max:.2f}
"""


class TritonBenchmark:
    """
    Triton 性能基准测试器
    
    对 Triton Inference Server 进行性能测试。
    
    Example:
        >>> benchmark = TritonBenchmark(config)
        >>> result = benchmark.run("localhost:8001", "ugt_recommend")
        >>> print(result.summary())
    """
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        export_config: Optional[ExportConfig] = None
    ):
        """
        初始化基准测试器
        
        Args:
            config: 基准测试配置
            export_config: 导出配置（用于获取模型输入规格）
        """
        self.config = config or BenchmarkConfig()
        self.export_config = export_config or ExportConfig()
        self.input_spec = ModelInputSpec()
        
        # 延迟初始化 Triton 客户端
        self._client = None
        self._client_lock = threading.Lock()
    
    def _get_client(self, triton_url: str):
        """获取 Triton 客户端（延迟初始化）"""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    try:
                        import tritonclient.http as httpclient
                        self._client = httpclient.InferenceServerClient(
                            url=triton_url,
                            verbose=False
                        )
                    except ImportError:
                        raise ImportError(
                            "tritonclient 未安装。请安装: "
                            "pip install tritonclient[http]"
                        )
        return self._client
    
    def _create_inputs(
        self,
        batch_size: int,
        seq_length: int
    ) -> List:
        """
        创建测试输入
        
        Args:
            batch_size: 批次大小
            seq_length: 序列长度
        
        Returns:
            Triton 输入列表
        """
        import tritonclient.http as httpclient
        
        l1_size, l2_size, l3_size = self.export_config.codebook_sizes
        
        inputs = []
        
        # 创建各个输入
        input_specs = [
            (self.input_spec.encoder_l1_ids, l1_size, "INT64"),
            (self.input_spec.encoder_l2_ids, l2_size, "INT64"),
            (self.input_spec.encoder_l3_ids, l3_size, "INT64"),
        ]
        
        for name, max_val, dtype in input_specs:
            inp = httpclient.InferInput(name, [batch_size, seq_length], dtype)
            data = np.random.randint(0, max_val, (batch_size, seq_length)).astype(np.int64)
            inp.set_data_from_numpy(data)
            inputs.append(inp)
        
        # 位置输入
        positions = httpclient.InferInput(
            self.input_spec.encoder_positions,
            [batch_size, seq_length],
            "INT64"
        )
        pos_data = np.tile(np.arange(seq_length), (batch_size, 1)).astype(np.int64)
        positions.set_data_from_numpy(pos_data)
        inputs.append(positions)
        
        # Token 类型输入
        token_types = httpclient.InferInput(
            self.input_spec.encoder_token_types,
            [batch_size, seq_length],
            "INT64"
        )
        type_data = np.zeros((batch_size, seq_length), dtype=np.int64)
        token_types.set_data_from_numpy(type_data)
        inputs.append(token_types)
        
        # 掩码输入
        mask = httpclient.InferInput(
            self.input_spec.encoder_mask,
            [batch_size, seq_length],
            "FP32"
        )
        mask_data = np.ones((batch_size, seq_length), dtype=np.float32)
        mask.set_data_from_numpy(mask_data)
        inputs.append(mask)
        
        return inputs
    
    def _single_request(
        self,
        client,
        model_name: str,
        inputs: List
    ) -> Tuple[bool, float]:
        """
        执行单次推理请求
        
        Args:
            client: Triton 客户端
            model_name: 模型名称
            inputs: 输入列表
        
        Returns:
            Tuple[成功标志, 延迟(毫秒)]
        """
        try:
            start_time = time.perf_counter()
            client.infer(model_name, inputs)
            latency = (time.perf_counter() - start_time) * 1000
            return True, latency
        except Exception as e:
            logger.debug(f"请求失败: {e}")
            return False, 0.0
    
    def run(
        self,
        triton_url: str,
        model_name: str,
        batch_size: int = 1,
        seq_length: int = 100
    ) -> BenchmarkResult:
        """
        运行基准测试
        
        Args:
            triton_url: Triton Server URL
            model_name: 模型名称
            batch_size: 批次大小
            seq_length: 序列长度
        
        Returns:
            BenchmarkResult 对象
        """
        client = self._get_client(triton_url)
        
        # 创建测试输入
        inputs = self._create_inputs(batch_size, seq_length)
        
        logger.info(f"开始基准测试: model={model_name}, "
                   f"batch_size={batch_size}, seq_length={seq_length}")
        
        # 预热
        logger.info(f"预热 {self.config.num_warmup_requests} 次请求...")
        for _ in range(self.config.num_warmup_requests):
            try:
                client.infer(model_name, inputs)
            except Exception as e:
                logger.warning(f"预热请求失败: {e}")
        
        # 正式测试
        logger.info(f"开始正式测试 {self.config.num_requests} 次请求...")
        latencies = []
        successful = 0
        failed = 0
        
        start_time = time.perf_counter()
        
        if self.config.concurrency > 1:
            # 并发测试
            with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
                futures = [
                    executor.submit(self._single_request, client, model_name, inputs)
                    for _ in range(self.config.num_requests)
                ]
                
                for future in as_completed(futures):
                    success, latency = future.result()
                    if success:
                        successful += 1
                        latencies.append(latency)
                    else:
                        failed += 1
        else:
            # 串行测试
            for _ in range(self.config.num_requests):
                success, latency = self._single_request(client, model_name, inputs)
                if success:
                    successful += 1
                    latencies.append(latency)
                else:
                    failed += 1
        
        total_time = time.perf_counter() - start_time
        
        # 计算统计数据
        result = BenchmarkResult(
            batch_size=batch_size,
            seq_length=seq_length,
            total_requests=self.config.num_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
        )
        
        if latencies:
            latencies.sort()
            result.throughput = successful / total_time
            result.latency_avg = statistics.mean(latencies)
            result.latency_min = min(latencies)
            result.latency_max = max(latencies)
            
            # 计算百分位数
            result.latency_p50 = np.percentile(latencies, 50)
            result.latency_p90 = np.percentile(latencies, 90)
            result.latency_p95 = np.percentile(latencies, 95)
            result.latency_p99 = np.percentile(latencies, 99)
        
        logger.info(f"测试完成: throughput={result.throughput:.2f} req/s, "
                   f"P99={result.latency_p99:.2f}ms")
        
        return result
    
    def run_sweep(
        self,
        triton_url: str,
        model_name: str
    ) -> Dict[str, BenchmarkResult]:
        """
        运行完整的参数扫描测试
        
        测试不同批次大小和序列长度的组合。
        
        Args:
            triton_url: Triton Server URL
            model_name: 模型名称
        
        Returns:
            测试结果字典，键为 "batch_size_seq_length"
        """
        results = {}
        
        for batch_size in self.config.test_batch_sizes:
            for seq_length in self.config.test_seq_lengths:
                key = f"bs{batch_size}_seq{seq_length}"
                logger.info(f"\n{'='*50}")
                logger.info(f"测试配置: {key}")
                logger.info(f"{'='*50}")
                
                result = self.run(triton_url, model_name, batch_size, seq_length)
                results[key] = result
                
                print(result.summary())
        
        return results
    
    def generate_report(
        self,
        results: Dict[str, BenchmarkResult],
        output_path: Optional[str] = None
    ) -> str:
        """
        生成测试报告
        
        Args:
            results: 测试结果字典
            output_path: 报告输出路径（可选）
        
        Returns:
            报告内容字符串
        """
        lines = [
            "=" * 80,
            "UGT 模型性能测试报告",
            "=" * 80,
            "",
            f"测试配置:",
            f"  - 预热请求数: {self.config.num_warmup_requests}",
            f"  - 测试请求数: {self.config.num_requests}",
            f"  - 并发数: {self.config.concurrency}",
            "",
            "=" * 80,
            "测试结果汇总",
            "=" * 80,
            "",
            f"{'配置':<20} {'吞吐量(req/s)':<15} {'P50(ms)':<10} "
            f"{'P90(ms)':<10} {'P99(ms)':<10} {'成功率':<10}",
            "-" * 80,
        ]
        
        for key, result in results.items():
            success_rate = (result.successful_requests / result.total_requests * 100
                          if result.total_requests > 0 else 0)
            lines.append(
                f"{key:<20} {result.throughput:<15.2f} {result.latency_p50:<10.2f} "
                f"{result.latency_p90:<10.2f} {result.latency_p99:<10.2f} "
                f"{success_rate:<10.1f}%"
            )
        
        lines.extend([
            "",
            "=" * 80,
            "性能分析",
            "=" * 80,
            "",
        ])
        
        # 找出最佳配置
        if results:
            best_throughput = max(results.items(), key=lambda x: x[1].throughput)
            best_latency = min(results.items(), key=lambda x: x[1].latency_p99)
            
            lines.extend([
                f"最高吞吐量配置: {best_throughput[0]} "
                f"({best_throughput[1].throughput:.2f} req/s)",
                f"最低 P99 延迟配置: {best_latency[0]} "
                f"({best_latency[1].latency_p99:.2f} ms)",
                "",
            ])
            
            # 检查是否满足目标延迟
            target_latency = self.export_config.target_latency_ms
            meeting_target = [
                k for k, v in results.items()
                if v.latency_p99 <= target_latency
            ]
            
            if meeting_target:
                lines.append(f"满足目标延迟 (<{target_latency}ms) 的配置: {', '.join(meeting_target)}")
            else:
                lines.append(f"⚠️ 没有配置满足目标延迟 (<{target_latency}ms)")
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"报告已保存到: {output_path}")
        
        return report


def run_benchmark(
    triton_url: str,
    model_name: str,
    num_requests: int = 10000,
    batch_size: int = 1,
    seq_length: int = 100
) -> Dict[str, float]:
    """
    便捷函数：运行性能基准测试
    
    Args:
        triton_url: Triton Server URL
        model_name: 模型名称
        num_requests: 请求数量
        batch_size: 批次大小
        seq_length: 序列长度
    
    Returns:
        性能指标字典
    
    Example:
        >>> metrics = run_benchmark("localhost:8001", "ugt_recommend")
        >>> print(f"P99: {metrics['latency_p99']:.2f}ms")
    """
    config = BenchmarkConfig(num_requests=num_requests)
    benchmark = TritonBenchmark(config)
    result = benchmark.run(triton_url, model_name, batch_size, seq_length)
    
    return {
        "throughput": result.throughput,
        "latency_p50": result.latency_p50,
        "latency_p90": result.latency_p90,
        "latency_p99": result.latency_p99,
    }


class MockTritonClient:
    """
    模拟 Triton 客户端
    
    用于测试目的，不需要实际的 Triton Server。
    """
    
    def __init__(self, url: str, latency_range: Tuple[float, float] = (5.0, 15.0)):
        """
        初始化模拟客户端
        
        Args:
            url: Triton Server URL（不使用）
            latency_range: 模拟延迟范围（毫秒）
        """
        self.url = url
        self.latency_range = latency_range
    
    def infer(self, model_name: str, inputs: List) -> Dict[str, Any]:
        """
        模拟推理请求
        
        Args:
            model_name: 模型名称
            inputs: 输入列表
        
        Returns:
            模拟的输出
        """
        # 模拟推理延迟
        latency = np.random.uniform(*self.latency_range) / 1000  # 转换为秒
        time.sleep(latency)
        
        # 返回模拟输出
        return {
            "recommendations": np.zeros((1, 50, 3), dtype=np.int64),
            "scores": np.zeros((1, 50), dtype=np.float32),
        }
    
    def is_server_ready(self) -> bool:
        """检查服务器是否就绪"""
        return True
    
    def is_model_ready(self, model_name: str) -> bool:
        """检查模型是否就绪"""
        return True


def create_mock_benchmark(
    num_requests: int = 100,
    latency_range: Tuple[float, float] = (5.0, 15.0)
) -> Tuple[TritonBenchmark, MockTritonClient]:
    """
    创建用于测试的模拟基准测试环境
    
    Args:
        num_requests: 请求数量
        latency_range: 模拟延迟范围
    
    Returns:
        Tuple[基准测试器, 模拟客户端]
    """
    config = BenchmarkConfig(
        num_requests=num_requests,
        num_warmup_requests=10
    )
    benchmark = TritonBenchmark(config)
    mock_client = MockTritonClient("localhost:8001", latency_range)
    
    return benchmark, mock_client

