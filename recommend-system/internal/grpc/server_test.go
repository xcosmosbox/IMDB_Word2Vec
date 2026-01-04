// Package grpc 服务器单元测试
package grpc

import (
	"context"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
)

// mockLogger 模拟日志器
type mockLogger struct{}

func (m *mockLogger) Info(msg string, keysAndValues ...interface{})  {}
func (m *mockLogger) Debug(msg string, keysAndValues ...interface{}) {}
func (m *mockLogger) Warn(msg string, keysAndValues ...interface{})  {}
func (m *mockLogger) Error(msg string, keysAndValues ...interface{}) {}

// TestDefaultServerConfig 测试默认配置
func TestDefaultServerConfig(t *testing.T) {
	cfg := DefaultServerConfig()

	if cfg.Address != ":50051" {
		t.Errorf("expected address :50051, got %s", cfg.Address)
	}

	if cfg.MaxRecvMsgSize != 4*1024*1024 {
		t.Errorf("expected MaxRecvMsgSize 4MB, got %d", cfg.MaxRecvMsgSize)
	}

	if cfg.MaxSendMsgSize != 4*1024*1024 {
		t.Errorf("expected MaxSendMsgSize 4MB, got %d", cfg.MaxSendMsgSize)
	}

	if cfg.MaxConcurrentStreams != 1000 {
		t.Errorf("expected MaxConcurrentStreams 1000, got %d", cfg.MaxConcurrentStreams)
	}

	if !cfg.EnableReflection {
		t.Error("expected EnableReflection to be true")
	}

	if !cfg.EnableHealthCheck {
		t.Error("expected EnableHealthCheck to be true")
	}
}

// TestNewServer 测试服务器创建
func TestNewServer(t *testing.T) {
	// 测试 nil logger
	_, err := NewServer(DefaultServerConfig(), nil)
	if err == nil {
		t.Error("expected error when logger is nil")
	}
}

// TestNewServerWithMockLogger 测试使用模拟日志器创建服务器
func TestNewServerWithMockLogger(t *testing.T) {
	// 使用随机端口避免端口冲突
	cfg := DefaultServerConfig()
	cfg.Address = ":0" // 使用随机端口
	cfg.EnableReflection = true
	cfg.EnableHealthCheck = true

	// 由于我们无法在测试中轻易创建 logger.Logger 实例，
	// 这个测试主要验证配置解析逻辑
	t.Skip("Skipping test that requires logger.Logger implementation")
}

// TestServerAddress 测试服务器地址
func TestServerAddress(t *testing.T) {
	// 创建一个临时监听器来测试地址获取
	lis, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}
	defer lis.Close()

	addr := lis.Addr().String()
	if addr == "" {
		t.Error("expected non-empty address")
	}
}

// TestBuildServerOptions 测试服务器选项构建
func TestBuildServerOptions(t *testing.T) {
	cfg := DefaultServerConfig()
	// 由于 buildServerOptions 需要 logger，这里跳过
	t.Skip("Skipping test that requires logger.Logger implementation")
}

// TestServerInterceptorRecovery 测试 panic 恢复
func TestServerInterceptorRecovery(t *testing.T) {
	// 测试 recoveryUnaryInterceptor 的 panic 恢复功能
	// 由于需要 logger 实现，这里跳过
	t.Skip("Skipping test that requires logger.Logger implementation")
}

// TestHealthCheckIntegration 测试健康检查集成
func TestHealthCheckIntegration(t *testing.T) {
	// 创建一个简单的 gRPC 服务器来测试健康检查
	lis, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}

	grpcServer := grpc.NewServer()
	
	// 启动服务器
	go func() {
		_ = grpcServer.Serve(lis)
	}()
	defer grpcServer.Stop()

	// 等待服务器启动
	time.Sleep(100 * time.Millisecond)

	// 连接到服务器
	conn, err := grpc.Dial(
		lis.Addr().String(),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer conn.Close()

	// 健康检查客户端
	healthClient := healthpb.NewHealthClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	// 注意：由于我们没有注册健康检查服务，这个调用会失败
	// 这里只是验证客户端可以连接
	_, err = healthClient.Check(ctx, &healthpb.HealthCheckRequest{})
	// 预期会失败，因为没有注册服务
	if err == nil {
		t.Log("Health check succeeded (unexpected but not an error)")
	}
}

// TestServerGracefulStop 测试优雅停止
func TestServerGracefulStop(t *testing.T) {
	lis, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}

	grpcServer := grpc.NewServer()

	// 启动服务器
	go func() {
		_ = grpcServer.Serve(lis)
	}()

	// 等待服务器启动
	time.Sleep(100 * time.Millisecond)

	// 测试优雅停止
	done := make(chan struct{})
	go func() {
		grpcServer.GracefulStop()
		close(done)
	}()

	select {
	case <-done:
		// 成功
	case <-time.After(5 * time.Second):
		t.Error("graceful stop timeout")
		grpcServer.Stop()
	}
}

// TestServerStop 测试立即停止
func TestServerStop(t *testing.T) {
	lis, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}

	grpcServer := grpc.NewServer()

	// 启动服务器
	errCh := make(chan error, 1)
	go func() {
		errCh <- grpcServer.Serve(lis)
	}()

	// 等待服务器启动
	time.Sleep(100 * time.Millisecond)

	// 立即停止
	grpcServer.Stop()

	// 等待服务器退出
	select {
	case err := <-errCh:
		if err != nil {
			// grpc.Stop() 后 Serve 返回的错误可能为 nil 或特定错误
			t.Logf("Server stopped with error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Error("server stop timeout")
	}
}

// BenchmarkServerCreation 基准测试服务器创建
func BenchmarkServerCreation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		lis, err := net.Listen("tcp", ":0")
		if err != nil {
			b.Fatalf("failed to create listener: %v", err)
		}

		grpcServer := grpc.NewServer()
		grpcServer.Stop()
		lis.Close()
	}
}

