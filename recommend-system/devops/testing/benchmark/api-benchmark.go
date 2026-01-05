// Package benchmark 提供 API 基准测试工具
//
// 用于测量推荐系统各 API 端点的性能基准线。
// 支持并发测试、延迟分布分析、吞吐量测量等。
//
// 使用方法:
//
//	go run api-benchmark.go -url http://localhost:8080 -duration 60s -concurrency 10
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// =============================================================================
// 配置
// =============================================================================

// Config 基准测试配置
type Config struct {
	BaseURL     string        // 目标服务 URL
	Duration    time.Duration // 测试持续时间
	Concurrency int           // 并发数
	RateLimit   int           // 每秒请求限制 (0 = 不限制)
	WarmupTime  time.Duration // 预热时间
	APIKey      string        // API 密钥
	Verbose     bool          // 详细输出
}

// DefaultConfig 返回默认配置
func DefaultConfig() *Config {
	return &Config{
		BaseURL:     "http://localhost:8080",
		Duration:    60 * time.Second,
		Concurrency: 10,
		RateLimit:   0,
		WarmupTime:  5 * time.Second,
		APIKey:      "test-api-key",
		Verbose:     false,
	}
}

// =============================================================================
// 结果统计
// =============================================================================

// Stats 统计结果
type Stats struct {
	Name           string
	TotalRequests  int64
	SuccessCount   int64
	FailCount      int64
	Latencies      []float64 // 毫秒
	StartTime      time.Time
	EndTime        time.Time
	BytesSent      int64
	BytesReceived  int64
	StatusCodes    map[int]int64
	Errors         map[string]int64
	mu             sync.Mutex
}

// NewStats 创建新的统计对象
func NewStats(name string) *Stats {
	return &Stats{
		Name:        name,
		Latencies:   make([]float64, 0, 10000),
		StatusCodes: make(map[int]int64),
		Errors:      make(map[string]int64),
	}
}

// RecordRequest 记录请求结果
func (s *Stats) RecordRequest(latencyMs float64, success bool, statusCode int, bytesSent, bytesReceived int64, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	atomic.AddInt64(&s.TotalRequests, 1)
	s.Latencies = append(s.Latencies, latencyMs)
	s.BytesSent += bytesSent
	s.BytesReceived += bytesReceived
	s.StatusCodes[statusCode]++

	if success {
		atomic.AddInt64(&s.SuccessCount, 1)
	} else {
		atomic.AddInt64(&s.FailCount, 1)
		if err != nil {
			s.Errors[err.Error()]++
		}
	}
}

// Calculate 计算统计指标
func (s *Stats) Calculate() *BenchmarkResult {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.Latencies) == 0 {
		return &BenchmarkResult{Name: s.Name}
	}

	// 排序延迟用于计算百分位数
	sorted := make([]float64, len(s.Latencies))
	copy(sorted, s.Latencies)
	sort.Float64s(sorted)

	duration := s.EndTime.Sub(s.StartTime).Seconds()
	if duration == 0 {
		duration = 1
	}

	result := &BenchmarkResult{
		Name:           s.Name,
		Duration:       duration,
		TotalRequests:  s.TotalRequests,
		SuccessCount:   s.SuccessCount,
		FailCount:      s.FailCount,
		RPS:            float64(s.TotalRequests) / duration,
		SuccessRate:    float64(s.SuccessCount) / float64(s.TotalRequests) * 100,
		AvgLatency:     average(sorted),
		MinLatency:     sorted[0],
		MaxLatency:     sorted[len(sorted)-1],
		P50Latency:     percentile(sorted, 50),
		P90Latency:     percentile(sorted, 90),
		P95Latency:     percentile(sorted, 95),
		P99Latency:     percentile(sorted, 99),
		StdDev:         stdDev(sorted),
		BytesSent:      s.BytesSent,
		BytesReceived:  s.BytesReceived,
		ThroughputMBps: float64(s.BytesReceived) / duration / 1024 / 1024,
		StatusCodes:    s.StatusCodes,
		Errors:         s.Errors,
	}

	return result
}

// BenchmarkResult 基准测试结果
type BenchmarkResult struct {
	Name           string
	Duration       float64
	TotalRequests  int64
	SuccessCount   int64
	FailCount      int64
	RPS            float64
	SuccessRate    float64
	AvgLatency     float64
	MinLatency     float64
	MaxLatency     float64
	P50Latency     float64
	P90Latency     float64
	P95Latency     float64
	P99Latency     float64
	StdDev         float64
	BytesSent      int64
	BytesReceived  int64
	ThroughputMBps float64
	StatusCodes    map[int]int64
	Errors         map[string]int64
}

// =============================================================================
// 基准测试执行器
// =============================================================================

// Benchmark 基准测试执行器
type Benchmark struct {
	config *Config
	client *http.Client
}

// NewBenchmark 创建基准测试执行器
func NewBenchmark(config *Config) *Benchmark {
	return &Benchmark{
		config: config,
		client: &http.Client{
			Timeout: 30 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 100,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

// RunAll 运行所有基准测试
func (b *Benchmark) RunAll() []*BenchmarkResult {
	results := make([]*BenchmarkResult, 0)

	fmt.Println("\n╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║              API Benchmark Starting                              ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Printf("Target: %s\n", b.config.BaseURL)
	fmt.Printf("Duration: %s\n", b.config.Duration)
	fmt.Printf("Concurrency: %d\n", b.config.Concurrency)
	fmt.Println()

	// 健康检查
	if !b.healthCheck() {
		fmt.Println("❌ Health check failed!")
		return results
	}
	fmt.Println("✓ Health check passed")
	fmt.Println()

	// 运行各端点基准测试
	endpoints := []struct {
		name     string
		method   string
		path     string
		body     interface{}
	}{
		{"健康检查", "GET", "/health", nil},
		{"获取推荐", "POST", "/api/v1/recommend", map[string]interface{}{
			"user_id": "benchmark_user",
			"limit":   20,
			"scene":   "home",
		}},
		{"搜索物品", "GET", "/api/v1/items/search?q=action&limit=20", nil},
		{"物品详情", "GET", "/api/v1/items/item_1", nil},
		{"提交反馈", "POST", "/api/v1/feedback", map[string]interface{}{
			"user_id": "benchmark_user",
			"item_id": "item_1",
			"action":  "click",
		}},
	}

	for _, ep := range endpoints {
		fmt.Printf("Testing: %s (%s %s)\n", ep.name, ep.method, ep.path)
		result := b.runEndpoint(ep.name, ep.method, ep.path, ep.body)
		results = append(results, result)
		b.printResult(result)
		fmt.Println()
	}

	return results
}

// runEndpoint 运行单个端点基准测试
func (b *Benchmark) runEndpoint(name, method, path string, body interface{}) *BenchmarkResult {
	stats := NewStats(name)
	ctx, cancel := context.WithTimeout(context.Background(), b.config.Duration+b.config.WarmupTime)
	defer cancel()

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, b.config.Concurrency)

	// 预热
	fmt.Printf("  Warming up for %s...\n", b.config.WarmupTime)
	warmupCtx, warmupCancel := context.WithTimeout(ctx, b.config.WarmupTime)
	for i := 0; i < b.config.Concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-warmupCtx.Done():
					return
				default:
					b.makeRequest(method, path, body)
				}
			}
		}()
	}
	<-warmupCtx.Done()
	warmupCancel()
	wg.Wait()

	// 正式测试
	fmt.Printf("  Running benchmark for %s...\n", b.config.Duration)
	stats.StartTime = time.Now()

	testCtx, testCancel := context.WithTimeout(ctx, b.config.Duration)
	defer testCancel()

	for i := 0; i < b.config.Concurrency*2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-testCtx.Done():
					return
				case semaphore <- struct{}{}:
					start := time.Now()
					resp, bytesSent, bytesReceived, err := b.makeRequest(method, path, body)

					latency := time.Since(start).Seconds() * 1000
					success := err == nil && resp != nil && resp.StatusCode >= 200 && resp.StatusCode < 300
					statusCode := 0
					if resp != nil {
						statusCode = resp.StatusCode
					}

					stats.RecordRequest(latency, success, statusCode, bytesSent, bytesReceived, err)
					<-semaphore
				}
			}
		}()
	}

	<-testCtx.Done()
	wg.Wait()
	stats.EndTime = time.Now()

	return stats.Calculate()
}

// makeRequest 发起请求
func (b *Benchmark) makeRequest(method, path string, body interface{}) (*http.Response, int64, int64, error) {
	url := b.config.BaseURL + path

	var reqBody io.Reader
	var bytesSent int64
	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			return nil, 0, 0, err
		}
		reqBody = bytes.NewBuffer(jsonData)
		bytesSent = int64(len(jsonData))
	}

	req, err := http.NewRequest(method, url, reqBody)
	if err != nil {
		return nil, bytesSent, 0, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+b.config.APIKey)

	resp, err := b.client.Do(req)
	if err != nil {
		return nil, bytesSent, 0, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	bytesReceived := int64(len(respBody))

	return resp, bytesSent, bytesReceived, nil
}

// healthCheck 健康检查
func (b *Benchmark) healthCheck() bool {
	resp, _, _, err := b.makeRequest("GET", "/health", nil)
	return err == nil && resp != nil && resp.StatusCode == 200
}

// printResult 打印结果
func (b *Benchmark) printResult(r *BenchmarkResult) {
	fmt.Printf("  ┌─────────────────────────────────────────────────────────────────┐\n")
	fmt.Printf("  │ %-63s │\n", r.Name)
	fmt.Printf("  ├─────────────────────────────────────────────────────────────────┤\n")
	fmt.Printf("  │ Total Requests:   %-45d │\n", r.TotalRequests)
	fmt.Printf("  │ Success Rate:     %-43.2f%% │\n", r.SuccessRate)
	fmt.Printf("  │ RPS:              %-45.2f │\n", r.RPS)
	fmt.Printf("  │ Avg Latency:      %-43.2fms │\n", r.AvgLatency)
	fmt.Printf("  │ P50 Latency:      %-43.2fms │\n", r.P50Latency)
	fmt.Printf("  │ P95 Latency:      %-43.2fms │\n", r.P95Latency)
	fmt.Printf("  │ P99 Latency:      %-43.2fms │\n", r.P99Latency)
	fmt.Printf("  │ Max Latency:      %-43.2fms │\n", r.MaxLatency)
	fmt.Printf("  │ Throughput:       %-41.2f MB/s │\n", r.ThroughputMBps)
	fmt.Printf("  └─────────────────────────────────────────────────────────────────┘\n")
}

// =============================================================================
// 辅助函数
// =============================================================================

func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func percentile(sortedValues []float64, p float64) float64 {
	if len(sortedValues) == 0 {
		return 0
	}
	index := int(float64(len(sortedValues)-1) * p / 100)
	return sortedValues[index]
}

func stdDev(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	avg := average(values)
	sumSquares := 0.0
	for _, v := range values {
		sumSquares += (v - avg) * (v - avg)
	}
	return math.Sqrt(sumSquares / float64(len(values)))
}

// =============================================================================
// 主函数
// =============================================================================

func main() {
	config := DefaultConfig()

	flag.StringVar(&config.BaseURL, "url", config.BaseURL, "目标服务 URL")
	flag.DurationVar(&config.Duration, "duration", config.Duration, "测试持续时间")
	flag.IntVar(&config.Concurrency, "concurrency", config.Concurrency, "并发数")
	flag.IntVar(&config.RateLimit, "rate", config.RateLimit, "每秒请求限制 (0 = 不限制)")
	flag.DurationVar(&config.WarmupTime, "warmup", config.WarmupTime, "预热时间")
	flag.StringVar(&config.APIKey, "key", config.APIKey, "API 密钥")
	flag.BoolVar(&config.Verbose, "v", config.Verbose, "详细输出")
	flag.Parse()

	benchmark := NewBenchmark(config)
	results := benchmark.RunAll()

	// 输出 JSON 结果
	if len(results) > 0 {
		fmt.Println("\n═══════════════════════════════════════════════════════════════════")
		fmt.Println("                         JSON Results")
		fmt.Println("═══════════════════════════════════════════════════════════════════")

		jsonData, err := json.MarshalIndent(results, "", "  ")
		if err == nil {
			fmt.Println(string(jsonData))

			// 保存到文件
			outputFile := fmt.Sprintf("benchmark_results_%s.json", time.Now().Format("20060102_150405"))
			if err := os.WriteFile(outputFile, jsonData, 0644); err == nil {
				fmt.Printf("\nResults saved to: %s\n", outputFile)
			}
		}
	}
}

