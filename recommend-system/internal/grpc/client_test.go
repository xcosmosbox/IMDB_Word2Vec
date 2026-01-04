// Package grpc 客户端单元测试
package grpc

import (
	"context"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// TestDefaultClientConfig 测试默认客户端配置
func TestDefaultClientConfig(t *testing.T) {
	cfg := DefaultClientConfig()

	if cfg.Timeout != 30*time.Second {
		t.Errorf("expected timeout 30s, got %v", cfg.Timeout)
	}

	if cfg.MaxRetries != 3 {
		t.Errorf("expected max retries 3, got %d", cfg.MaxRetries)
	}

	if cfg.RetryBackoff != 100*time.Millisecond {
		t.Errorf("expected retry backoff 100ms, got %v", cfg.RetryBackoff)
	}

	if cfg.MaxRecvMsgSize != 4*1024*1024 {
		t.Errorf("expected MaxRecvMsgSize 4MB, got %d", cfg.MaxRecvMsgSize)
	}

	if cfg.MaxSendMsgSize != 4*1024*1024 {
		t.Errorf("expected MaxSendMsgSize 4MB, got %d", cfg.MaxSendMsgSize)
	}

	if !cfg.Insecure {
		t.Error("expected Insecure to be true")
	}
}

// TestClientsCreation 测试客户端集合创建
func TestClientsCreation(t *testing.T) {
	// 创建一个临时服务器
	lis, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}
	defer lis.Close()

	grpcServer := grpc.NewServer()
	go func() {
		_ = grpcServer.Serve(lis)
	}()
	defer grpcServer.Stop()

	// 等待服务器启动
	time.Sleep(100 * time.Millisecond)

	// 创建客户端
	cfg := DefaultClientConfig()
	cfg.RecommendServiceAddr = lis.Addr().String()

	clients, err := NewClients(cfg)
	if err != nil {
		t.Fatalf("failed to create clients: %v", err)
	}
	defer clients.Close()

	// 验证推荐服务客户端已创建
	if clients.Recommend == nil {
		t.Error("expected Recommend client to be created")
	}

	// 验证其他客户端未创建（因为地址为空）
	if clients.User != nil {
		t.Error("expected User client to be nil")
	}

	if clients.Item != nil {
		t.Error("expected Item client to be nil")
	}
}

// TestClientsClose 测试客户端关闭
func TestClientsClose(t *testing.T) {
	// 创建一个临时服务器
	lis, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}
	defer lis.Close()

	grpcServer := grpc.NewServer()
	go func() {
		_ = grpcServer.Serve(lis)
	}()
	defer grpcServer.Stop()

	// 等待服务器启动
	time.Sleep(100 * time.Millisecond)

	// 创建客户端
	cfg := DefaultClientConfig()
	cfg.RecommendServiceAddr = lis.Addr().String()

	clients, err := NewClients(cfg)
	if err != nil {
		t.Fatalf("failed to create clients: %v", err)
	}

	// 关闭客户端
	err = clients.Close()
	if err != nil {
		t.Errorf("failed to close clients: %v", err)
	}

	// 再次关闭应该是安全的
	err = clients.Close()
	if err != nil {
		t.Logf("second close returned error (expected): %v", err)
	}
}

// TestClientsGetConnState 测试获取连接状态
func TestClientsGetConnState(t *testing.T) {
	// 创建一个临时服务器
	lis, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}
	defer lis.Close()

	grpcServer := grpc.NewServer()
	go func() {
		_ = grpcServer.Serve(lis)
	}()
	defer grpcServer.Stop()

	// 等待服务器启动
	time.Sleep(100 * time.Millisecond)

	// 创建客户端
	cfg := DefaultClientConfig()
	cfg.RecommendServiceAddr = lis.Addr().String()

	clients, err := NewClients(cfg)
	if err != nil {
		t.Fatalf("failed to create clients: %v", err)
	}
	defer clients.Close()

	// 获取连接状态
	states := clients.GetConnState()
	if len(states) != 1 {
		t.Errorf("expected 1 connection, got %d", len(states))
	}

	// 验证状态值存在
	for addr, state := range states {
		t.Logf("Connection %s: %s", addr, state)
	}
}

// TestWithTimeout 测试超时上下文创建
func TestWithTimeout(t *testing.T) {
	ctx := context.Background()
	timeout := 5 * time.Second

	newCtx, cancel := WithTimeout(ctx, timeout)
	defer cancel()

	deadline, ok := newCtx.Deadline()
	if !ok {
		t.Error("expected context to have deadline")
	}

	// 验证 deadline 在合理范围内
	expectedDeadline := time.Now().Add(timeout)
	diff := deadline.Sub(expectedDeadline)
	if diff > time.Second || diff < -time.Second {
		t.Errorf("deadline not in expected range, diff: %v", diff)
	}
}

// TestWithDeadline 测试截止时间上下文创建
func TestWithDeadline(t *testing.T) {
	ctx := context.Background()
	deadline := time.Now().Add(10 * time.Second)

	newCtx, cancel := WithDeadline(ctx, deadline)
	defer cancel()

	actualDeadline, ok := newCtx.Deadline()
	if !ok {
		t.Error("expected context to have deadline")
	}

	// 验证 deadline 相等
	if !actualDeadline.Equal(deadline) {
		t.Errorf("expected deadline %v, got %v", deadline, actualDeadline)
	}
}

// TestTimeoutUnaryInterceptor 测试超时拦截器
func TestTimeoutUnaryInterceptor(t *testing.T) {
	timeout := 5 * time.Second
	interceptor := timeoutUnaryInterceptor(timeout)

	if interceptor == nil {
		t.Error("expected interceptor to be non-nil")
	}
}

// TestRetryUnaryInterceptor 测试重试拦截器
func TestRetryUnaryInterceptor(t *testing.T) {
	maxRetries := 3
	backoff := 100 * time.Millisecond
	interceptor := retryUnaryInterceptor(maxRetries, backoff)

	if interceptor == nil {
		t.Error("expected interceptor to be non-nil")
	}
}

// TestIsRetriable 测试可重试判断
func TestIsRetriable(t *testing.T) {
	// 目前实现总是返回 false
	if isRetriable(nil) {
		t.Error("expected nil error to be non-retriable")
	}
}

// TestSingleClientCreation 测试单服务客户端创建
func TestSingleClientCreation(t *testing.T) {
	// 创建一个临时服务器
	lis, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}
	defer lis.Close()

	grpcServer := grpc.NewServer()
	go func() {
		_ = grpcServer.Serve(lis)
	}()
	defer grpcServer.Stop()

	// 等待服务器启动
	time.Sleep(100 * time.Millisecond)

	cfg := DefaultClientConfig()

	// 测试 RecommendClient
	recommendClient, err := NewRecommendClient(lis.Addr().String(), cfg)
	if err != nil {
		t.Fatalf("failed to create recommend client: %v", err)
	}
	defer recommendClient.Close()

	// 测试 UserClient
	userClient, err := NewUserClient(lis.Addr().String(), cfg)
	if err != nil {
		t.Fatalf("failed to create user client: %v", err)
	}
	defer userClient.Close()

	// 测试 ItemClient
	itemClient, err := NewItemClient(lis.Addr().String(), cfg)
	if err != nil {
		t.Fatalf("failed to create item client: %v", err)
	}
	defer itemClient.Close()
}

// TestClientConnectionFailure 测试连接失败场景
func TestClientConnectionFailure(t *testing.T) {
	cfg := DefaultClientConfig()
	cfg.RecommendServiceAddr = "invalid-address:12345"

	// 注意：gRPC 默认使用延迟连接，所以创建客户端时可能不会立即失败
	clients, err := NewClients(cfg)
	if err != nil {
		// 如果立即失败，这是预期的
		t.Logf("Client creation failed as expected: %v", err)
		return
	}
	defer clients.Close()

	// 如果创建成功，验证连接状态
	states := clients.GetConnState()
	t.Logf("Connection states: %v", states)
}

// TestClientBuildDialOptions 测试构建连接选项
func TestClientBuildDialOptions(t *testing.T) {
	cfg := DefaultClientConfig()
	clients := &Clients{config: cfg}

	opts := clients.buildDialOptions()

	// 验证选项数量
	if len(opts) == 0 {
		t.Error("expected dial options to be non-empty")
	}

	// 验证不安全传输选项
	if !cfg.Insecure {
		t.Error("expected Insecure to be true in config")
	}
}

// BenchmarkClientCreation 基准测试客户端创建
func BenchmarkClientCreation(b *testing.B) {
	// 创建一个临时服务器
	lis, err := net.Listen("tcp", ":0")
	if err != nil {
		b.Fatalf("failed to create listener: %v", err)
	}
	defer lis.Close()

	grpcServer := grpc.NewServer()
	go func() {
		_ = grpcServer.Serve(lis)
	}()
	defer grpcServer.Stop()

	// 等待服务器启动
	time.Sleep(100 * time.Millisecond)

	cfg := DefaultClientConfig()
	addr := lis.Addr().String()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		conn, err := grpc.Dial(
			addr,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
		)
		if err != nil {
			b.Fatalf("failed to dial: %v", err)
		}
		conn.Close()
	}
}

// BenchmarkWithTimeout 基准测试超时上下文创建
func BenchmarkWithTimeout(b *testing.B) {
	ctx := context.Background()
	timeout := 5 * time.Second

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, cancel := WithTimeout(ctx, timeout)
		cancel()
	}
}

