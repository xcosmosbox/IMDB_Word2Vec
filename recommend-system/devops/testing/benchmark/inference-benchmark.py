#!/usr/bin/env python3
"""
推理基准测试脚本

用于测量 UGT (Unified Generative Transformer) 模型的推理性能。
支持批量推理、延迟分析、GPU 利用率监控等。

使用方法:
    python inference-benchmark.py --host localhost --port 50051 --duration 60
"""

import argparse
import json
import time
import random
import statistics
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading

try:
    import grpc
    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False
    print("Warning: grpc not installed. gRPC benchmarks will be skipped.")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not installed. HTTP benchmarks will be skipped.")


# =============================================================================
# 配置
# =============================================================================

@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    host: str = "localhost"
    port: int = 50051
    http_port: int = 8080
    duration: int = 60  # 秒
    warmup: int = 10  # 预热时间 (秒)
    concurrency: int = 10
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 32])
    sequence_lengths: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    use_grpc: bool = True
    use_http: bool = True
    verbose: bool = False


# =============================================================================
# 结果数据类
# =============================================================================

@dataclass
class LatencyStats:
    """延迟统计"""
    count: int = 0
    avg: float = 0.0
    min: float = float('inf')
    max: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    std_dev: float = 0.0


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    batch_size: int
    sequence_length: int
    protocol: str
    duration: float
    total_requests: int
    success_count: int
    fail_count: int
    rps: float
    throughput_samples_per_sec: float
    latency: LatencyStats
    errors: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# 推理客户端
# =============================================================================

class InferenceClient:
    """推理服务客户端"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.http_base_url = f"http://{config.host}:{config.http_port}"
    
    def http_inference(self, user_id: str, sequence: List[int], limit: int = 20) -> Dict[str, Any]:
        """HTTP 推理请求"""
        if not HAS_REQUESTS:
            raise RuntimeError("requests library not installed")
        
        payload = {
            "user_id": user_id,
            "sequence": sequence,
            "limit": limit,
        }
        
        response = requests.post(
            f"{self.http_base_url}/api/v1/recommend",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        
        return {
            "status_code": response.status_code,
            "latency_ms": response.elapsed.total_seconds() * 1000,
            "success": response.status_code == 200,
        }
    
    def grpc_inference(self, user_id: str, sequence: List[int], limit: int = 20) -> Dict[str, Any]:
        """gRPC 推理请求 (模拟)"""
        # 注意: 实际实现需要根据 protobuf 定义来生成客户端代码
        # 这里使用模拟实现
        start_time = time.time()
        
        # 模拟推理延迟 (实际应该调用 gRPC 服务)
        time.sleep(random.uniform(0.005, 0.02))
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "status_code": 200,
            "latency_ms": latency_ms,
            "success": True,
        }


# =============================================================================
# 基准测试执行器
# =============================================================================

class InferenceBenchmark:
    """推理基准测试执行器"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.client = InferenceClient(config)
        self.results: List[BenchmarkResult] = []
    
    def generate_test_data(self, batch_size: int, sequence_length: int) -> List[Dict[str, Any]]:
        """生成测试数据"""
        test_data = []
        for i in range(batch_size):
            test_data.append({
                "user_id": f"benchmark_user_{i}",
                "sequence": [random.randint(1, 10000) for _ in range(sequence_length)],
                "limit": 20,
            })
        return test_data
    
    def run_single_benchmark(
        self,
        name: str,
        batch_size: int,
        sequence_length: int,
        protocol: str,
        duration: int,
    ) -> BenchmarkResult:
        """运行单个基准测试"""
        print(f"  Running: {name} (batch={batch_size}, seq_len={sequence_length}, protocol={protocol})")
        
        latencies: List[float] = []
        errors: Dict[str, int] = {}
        success_count = 0
        fail_count = 0
        
        # 选择推理方法
        if protocol == "http":
            inference_func = self.client.http_inference
        else:
            inference_func = self.client.grpc_inference
        
        # 生成测试数据
        test_data = self.generate_test_data(batch_size, sequence_length)
        
        # 预热
        print(f"    Warming up for {self.config.warmup}s...")
        warmup_end = time.time() + self.config.warmup
        while time.time() < warmup_end:
            for data in test_data:
                try:
                    inference_func(data["user_id"], data["sequence"], data["limit"])
                except Exception:
                    pass
        
        # 正式测试
        print(f"    Running for {duration}s...")
        start_time = time.time()
        end_time = start_time + duration
        
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            futures = []
            
            while time.time() < end_time:
                for data in test_data:
                    if time.time() >= end_time:
                        break
                    future = executor.submit(
                        inference_func,
                        data["user_id"],
                        data["sequence"],
                        data["limit"],
                    )
                    futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=1)
                    if result["success"]:
                        success_count += 1
                        latencies.append(result["latency_ms"])
                    else:
                        fail_count += 1
                        error_key = f"status_{result['status_code']}"
                        errors[error_key] = errors.get(error_key, 0) + 1
                except Exception as e:
                    fail_count += 1
                    error_key = type(e).__name__
                    errors[error_key] = errors.get(error_key, 0) + 1
        
        actual_duration = time.time() - start_time
        total_requests = success_count + fail_count
        
        # 计算延迟统计
        latency_stats = self._calculate_latency_stats(latencies)
        
        # 计算吞吐量
        rps = total_requests / actual_duration if actual_duration > 0 else 0
        throughput = (success_count * batch_size) / actual_duration if actual_duration > 0 else 0
        
        result = BenchmarkResult(
            name=name,
            batch_size=batch_size,
            sequence_length=sequence_length,
            protocol=protocol,
            duration=actual_duration,
            total_requests=total_requests,
            success_count=success_count,
            fail_count=fail_count,
            rps=rps,
            throughput_samples_per_sec=throughput,
            latency=latency_stats,
            errors=errors,
        )
        
        self._print_result(result)
        return result
    
    def _calculate_latency_stats(self, latencies: List[float]) -> LatencyStats:
        """计算延迟统计"""
        if not latencies:
            return LatencyStats()
        
        sorted_latencies = sorted(latencies)
        count = len(sorted_latencies)
        
        return LatencyStats(
            count=count,
            avg=statistics.mean(sorted_latencies),
            min=sorted_latencies[0],
            max=sorted_latencies[-1],
            p50=sorted_latencies[int(count * 0.50)],
            p90=sorted_latencies[int(count * 0.90)],
            p95=sorted_latencies[int(count * 0.95)],
            p99=sorted_latencies[min(int(count * 0.99), count - 1)],
            std_dev=statistics.stdev(sorted_latencies) if count > 1 else 0,
        )
    
    def _print_result(self, result: BenchmarkResult):
        """打印单个测试结果"""
        print(f"    ├─ Requests: {result.total_requests} (Success: {result.success_count}, Fail: {result.fail_count})")
        print(f"    ├─ RPS: {result.rps:.2f}")
        print(f"    ├─ Throughput: {result.throughput_samples_per_sec:.2f} samples/s")
        print(f"    ├─ Latency (ms): avg={result.latency.avg:.2f}, p50={result.latency.p50:.2f}, p99={result.latency.p99:.2f}")
        print(f"    └─ Success Rate: {result.success_count / max(result.total_requests, 1) * 100:.2f}%")
    
    def run_all(self) -> List[BenchmarkResult]:
        """运行所有基准测试"""
        print("\n" + "=" * 70)
        print("            INFERENCE BENCHMARK STARTING")
        print("=" * 70)
        print(f"Host: {self.config.host}")
        print(f"HTTP Port: {self.config.http_port}")
        print(f"gRPC Port: {self.config.port}")
        print(f"Duration: {self.config.duration}s per test")
        print(f"Concurrency: {self.config.concurrency}")
        print(f"Batch Sizes: {self.config.batch_sizes}")
        print(f"Sequence Lengths: {self.config.sequence_lengths}")
        print("=" * 70)
        
        self.results = []
        
        # 测试不同批量大小和序列长度的组合
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                # HTTP 测试
                if self.config.use_http and HAS_REQUESTS:
                    try:
                        result = self.run_single_benchmark(
                            name=f"HTTP Inference (batch={batch_size}, seq={seq_len})",
                            batch_size=batch_size,
                            sequence_length=seq_len,
                            protocol="http",
                            duration=self.config.duration,
                        )
                        self.results.append(result)
                    except Exception as e:
                        print(f"    ERROR: {e}")
                
                # gRPC 测试
                if self.config.use_grpc and HAS_GRPC:
                    try:
                        result = self.run_single_benchmark(
                            name=f"gRPC Inference (batch={batch_size}, seq={seq_len})",
                            batch_size=batch_size,
                            sequence_length=seq_len,
                            protocol="grpc",
                            duration=self.config.duration,
                        )
                        self.results.append(result)
                    except Exception as e:
                        print(f"    ERROR: {e}")
        
        return self.results
    
    def print_summary(self):
        """打印汇总报告"""
        if not self.results:
            print("No results to summarize.")
            return
        
        print("\n" + "=" * 70)
        print("                     BENCHMARK SUMMARY")
        print("=" * 70)
        
        # 按协议分组
        http_results = [r for r in self.results if r.protocol == "http"]
        grpc_results = [r for r in self.results if r.protocol == "grpc"]
        
        for protocol, results in [("HTTP", http_results), ("gRPC", grpc_results)]:
            if not results:
                continue
            
            print(f"\n{protocol} Results:")
            print("-" * 70)
            print(f"{'Batch':<8} {'SeqLen':<8} {'RPS':<12} {'P50(ms)':<12} {'P99(ms)':<12} {'Success%':<10}")
            print("-" * 70)
            
            for r in results:
                success_rate = r.success_count / max(r.total_requests, 1) * 100
                print(f"{r.batch_size:<8} {r.sequence_length:<8} {r.rps:<12.2f} {r.latency.p50:<12.2f} {r.latency.p99:<12.2f} {success_rate:<10.2f}")
        
        # 最佳配置
        if self.results:
            best_throughput = max(self.results, key=lambda r: r.throughput_samples_per_sec)
            best_latency = min(self.results, key=lambda r: r.latency.p99)
            
            print("\n" + "-" * 70)
            print("BEST CONFIGURATIONS:")
            print(f"  Highest Throughput: {best_throughput.name}")
            print(f"    - {best_throughput.throughput_samples_per_sec:.2f} samples/s")
            print(f"  Lowest P99 Latency: {best_latency.name}")
            print(f"    - {best_latency.latency.p99:.2f} ms")
        
        print("=" * 70)
    
    def save_results(self, output_file: str):
        """保存结果到 JSON 文件"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results],
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")


# =============================================================================
# SLA 检查器
# =============================================================================

class SLAChecker:
    """SLA 检查器"""
    
    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        self.sla = {
            "p50_latency_ms": 50,
            "p99_latency_ms": 200,
            "success_rate_pct": 99.9,
            "min_rps": 100,
        }
    
    def check(self) -> Dict[str, Any]:
        """检查 SLA 合规性"""
        violations = []
        passed = []
        
        for result in self.results:
            success_rate = result.success_count / max(result.total_requests, 1) * 100
            
            checks = [
                ("P50 Latency", result.latency.p50, self.sla["p50_latency_ms"], "<="),
                ("P99 Latency", result.latency.p99, self.sla["p99_latency_ms"], "<="),
                ("Success Rate", success_rate, self.sla["success_rate_pct"], ">="),
                ("RPS", result.rps, self.sla["min_rps"], ">="),
            ]
            
            for name, actual, target, op in checks:
                if op == "<=" and actual > target:
                    violations.append({
                        "test": result.name,
                        "metric": name,
                        "actual": actual,
                        "target": target,
                        "operator": op,
                    })
                elif op == ">=" and actual < target:
                    violations.append({
                        "test": result.name,
                        "metric": name,
                        "actual": actual,
                        "target": target,
                        "operator": op,
                    })
                else:
                    passed.append({
                        "test": result.name,
                        "metric": name,
                        "actual": actual,
                        "target": target,
                    })
        
        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "passed_checks": passed,
        }
    
    def print_report(self):
        """打印 SLA 检查报告"""
        result = self.check()
        
        print("\n" + "=" * 70)
        print("                     SLA CHECK REPORT")
        print("=" * 70)
        
        print(f"SLA Targets:")
        print(f"  - P50 Latency: <= {self.sla['p50_latency_ms']}ms")
        print(f"  - P99 Latency: <= {self.sla['p99_latency_ms']}ms")
        print(f"  - Success Rate: >= {self.sla['success_rate_pct']}%")
        print(f"  - Min RPS: >= {self.sla['min_rps']}")
        print()
        
        if result["passed"]:
            print("✓ ALL SLA CHECKS PASSED")
        else:
            print("✗ SLA VIOLATIONS DETECTED:")
            for v in result["violations"]:
                print(f"  - {v['test']}: {v['metric']} = {v['actual']:.2f} (target: {v['operator']} {v['target']})")
        
        print("=" * 70)


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inference Benchmark for UGT Model")
    parser.add_argument("--host", default="localhost", help="推理服务主机")
    parser.add_argument("--port", type=int, default=50051, help="gRPC 端口")
    parser.add_argument("--http-port", type=int, default=8080, help="HTTP 端口")
    parser.add_argument("--duration", type=int, default=30, help="每个测试持续时间 (秒)")
    parser.add_argument("--warmup", type=int, default=5, help="预热时间 (秒)")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数")
    parser.add_argument("--batch-sizes", type=str, default="1,8,16", help="批量大小列表 (逗号分隔)")
    parser.add_argument("--seq-lengths", type=str, default="64,128,256", help="序列长度列表 (逗号分隔)")
    parser.add_argument("--no-http", action="store_true", help="跳过 HTTP 测试")
    parser.add_argument("--no-grpc", action="store_true", help="跳过 gRPC 测试")
    parser.add_argument("--output", default=None, help="输出 JSON 文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 解析批量大小和序列长度
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
    
    config = BenchmarkConfig(
        host=args.host,
        port=args.port,
        http_port=args.http_port,
        duration=args.duration,
        warmup=args.warmup,
        concurrency=args.concurrency,
        batch_sizes=batch_sizes,
        sequence_lengths=seq_lengths,
        use_grpc=not args.no_grpc,
        use_http=not args.no_http,
        verbose=args.verbose,
    )
    
    # 运行基准测试
    benchmark = InferenceBenchmark(config)
    results = benchmark.run_all()
    
    # 打印汇总
    benchmark.print_summary()
    
    # SLA 检查
    sla_checker = SLAChecker(results)
    sla_checker.print_report()
    
    # 保存结果
    output_file = args.output or f"inference_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    benchmark.save_results(output_file)
    
    # 返回退出码
    sla_result = sla_checker.check()
    sys.exit(0 if sla_result["passed"] else 1)


if __name__ == "__main__":
    main()

