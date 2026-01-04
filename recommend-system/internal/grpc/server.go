// Package grpc 实现 gRPC 服务器和客户端
//
// 该包提供了 gRPC 服务器的创建、配置和管理功能，支持：
//   - 服务注册
//   - 拦截器（日志、追踪、限流等）
//   - 健康检查
//   - 优雅关闭
package grpc

import (
	"context"
	"fmt"
	"net"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/reflection"

	"recommend-system/pkg/logger"
)

// =============================================================================
// 服务器配置
// =============================================================================

// ServerConfig gRPC 服务器配置
type ServerConfig struct {
	// Address 监听地址，如 ":50051"
	Address string

	// MaxRecvMsgSize 最大接收消息大小（字节），默认 4MB
	MaxRecvMsgSize int

	// MaxSendMsgSize 最大发送消息大小（字节），默认 4MB
	MaxSendMsgSize int

	// MaxConcurrentStreams 最大并发流数量
	MaxConcurrentStreams uint32

	// ConnectionTimeout 连接超时时间
	ConnectionTimeout time.Duration

	// KeepAliveTime 保活探测间隔
	KeepAliveTime time.Duration

	// KeepAliveTimeout 保活探测超时
	KeepAliveTimeout time.Duration

	// EnableReflection 是否启用反射（调试用）
	EnableReflection bool

	// EnableHealthCheck 是否启用健康检查
	EnableHealthCheck bool
}

// DefaultServerConfig 返回默认配置
func DefaultServerConfig() ServerConfig {
	return ServerConfig{
		Address:              ":50051",
		MaxRecvMsgSize:       4 * 1024 * 1024,  // 4MB
		MaxSendMsgSize:       4 * 1024 * 1024,  // 4MB
		MaxConcurrentStreams: 1000,
		ConnectionTimeout:    120 * time.Second,
		KeepAliveTime:        30 * time.Second,
		KeepAliveTimeout:     10 * time.Second,
		EnableReflection:     true,
		EnableHealthCheck:    true,
	}
}

// =============================================================================
// gRPC 服务器
// =============================================================================

// Server gRPC 服务器
type Server struct {
	config       ServerConfig
	grpcServer   *grpc.Server
	listener     net.Listener
	logger       *logger.Logger
	healthServer *health.Server
}

// NewServer 创建 gRPC 服务器
func NewServer(cfg ServerConfig, log *logger.Logger) (*Server, error) {
	if log == nil {
		return nil, fmt.Errorf("logger is required")
	}

	// 创建监听器
	lis, err := net.Listen("tcp", cfg.Address)
	if err != nil {
		return nil, fmt.Errorf("failed to listen on %s: %w", cfg.Address, err)
	}

	// 构建服务器选项
	opts := buildServerOptions(cfg, log)

	// 创建 gRPC 服务器
	grpcServer := grpc.NewServer(opts...)

	server := &Server{
		config:     cfg,
		grpcServer: grpcServer,
		listener:   lis,
		logger:     log,
	}

	// 启用反射
	if cfg.EnableReflection {
		reflection.Register(grpcServer)
		log.Info("gRPC reflection enabled")
	}

	// 启用健康检查
	if cfg.EnableHealthCheck {
		server.healthServer = health.NewServer()
		healthpb.RegisterHealthServer(grpcServer, server.healthServer)
		server.healthServer.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
		log.Info("gRPC health check enabled")
	}

	return server, nil
}

// buildServerOptions 构建服务器选项
func buildServerOptions(cfg ServerConfig, log *logger.Logger) []grpc.ServerOption {
	opts := []grpc.ServerOption{
		grpc.MaxRecvMsgSize(cfg.MaxRecvMsgSize),
		grpc.MaxSendMsgSize(cfg.MaxSendMsgSize),
		grpc.MaxConcurrentStreams(cfg.MaxConcurrentStreams),
		grpc.ConnectionTimeout(cfg.ConnectionTimeout),
		grpc.KeepaliveParams(keepalive.ServerParameters{
			Time:    cfg.KeepAliveTime,
			Timeout: cfg.KeepAliveTimeout,
		}),
		grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
			MinTime:             5 * time.Second,
			PermitWithoutStream: true,
		}),
		// 拦截器
		grpc.ChainUnaryInterceptor(
			recoveryUnaryInterceptor(log),
			loggingUnaryInterceptor(log),
			metricsUnaryInterceptor(),
		),
		grpc.ChainStreamInterceptor(
			recoveryStreamInterceptor(log),
			loggingStreamInterceptor(log),
			metricsStreamInterceptor(),
		),
	}

	return opts
}

// =============================================================================
// 服务器方法
// =============================================================================

// GetGRPCServer 获取底层 gRPC 服务器实例
func (s *Server) GetGRPCServer() *grpc.Server {
	return s.grpcServer
}

// RegisterService 注册 gRPC 服务
// 使用示例:
//
//	server.RegisterService(&recommendv1.RecommendService_ServiceDesc, recommendService)
func (s *Server) RegisterService(sd *grpc.ServiceDesc, impl interface{}) {
	s.grpcServer.RegisterService(sd, impl)
	s.logger.Info("registered gRPC service", "service", sd.ServiceName)

	// 更新健康状态
	if s.healthServer != nil {
		s.healthServer.SetServingStatus(sd.ServiceName, healthpb.HealthCheckResponse_SERVING)
	}
}

// SetServiceStatus 设置服务健康状态
func (s *Server) SetServiceStatus(service string, serving bool) {
	if s.healthServer == nil {
		return
	}
	status := healthpb.HealthCheckResponse_NOT_SERVING
	if serving {
		status = healthpb.HealthCheckResponse_SERVING
	}
	s.healthServer.SetServingStatus(service, status)
}

// Start 启动服务器（阻塞）
func (s *Server) Start() error {
	s.logger.Info("gRPC server starting", "address", s.listener.Addr().String())
	return s.grpcServer.Serve(s.listener)
}

// StartAsync 异步启动服务器
func (s *Server) StartAsync() <-chan error {
	errCh := make(chan error, 1)
	go func() {
		if err := s.Start(); err != nil {
			errCh <- err
		}
		close(errCh)
	}()
	return errCh
}

// Stop 立即停止服务器
func (s *Server) Stop() {
	s.logger.Info("gRPC server stopping")
	if s.healthServer != nil {
		s.healthServer.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)
	}
	s.grpcServer.Stop()
}

// GracefulStop 优雅停止服务器
func (s *Server) GracefulStop() {
	s.logger.Info("gRPC server gracefully stopping")
	if s.healthServer != nil {
		s.healthServer.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)
	}
	s.grpcServer.GracefulStop()
}

// GracefulStopWithTimeout 带超时的优雅停止
func (s *Server) GracefulStopWithTimeout(timeout time.Duration) {
	s.logger.Info("gRPC server gracefully stopping with timeout", "timeout", timeout)
	if s.healthServer != nil {
		s.healthServer.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)
	}

	done := make(chan struct{})
	go func() {
		s.grpcServer.GracefulStop()
		close(done)
	}()

	select {
	case <-done:
		s.logger.Info("gRPC server gracefully stopped")
	case <-time.After(timeout):
		s.logger.Warn("gRPC server graceful stop timeout, forcing stop")
		s.grpcServer.Stop()
	}
}

// Address 返回服务器监听地址
func (s *Server) Address() string {
	return s.listener.Addr().String()
}

// =============================================================================
// 拦截器实现
// =============================================================================

// recoveryUnaryInterceptor panic 恢复拦截器（一元 RPC）
func recoveryUnaryInterceptor(log *logger.Logger) grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (resp interface{}, err error) {
		defer func() {
			if r := recover(); r != nil {
				log.Error("gRPC panic recovered",
					"method", info.FullMethod,
					"panic", r,
				)
				err = fmt.Errorf("internal server error")
			}
		}()
		return handler(ctx, req)
	}
}

// loggingUnaryInterceptor 日志拦截器（一元 RPC）
func loggingUnaryInterceptor(log *logger.Logger) grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		start := time.Now()

		// 调用处理器
		resp, err := handler(ctx, req)

		// 记录日志
		duration := time.Since(start)
		if err != nil {
			log.Error("gRPC request failed",
				"method", info.FullMethod,
				"duration", duration,
				"error", err,
			)
		} else {
			log.Debug("gRPC request completed",
				"method", info.FullMethod,
				"duration", duration,
			)
		}

		return resp, err
	}
}

// metricsUnaryInterceptor 指标拦截器（一元 RPC）
func metricsUnaryInterceptor() grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		start := time.Now()
		resp, err := handler(ctx, req)
		_ = time.Since(start)
		// TODO: 记录 Prometheus 指标
		// grpcRequestTotal.WithLabelValues(info.FullMethod).Inc()
		// grpcRequestDuration.WithLabelValues(info.FullMethod).Observe(duration.Seconds())
		return resp, err
	}
}

// recoveryStreamInterceptor panic 恢复拦截器（流式 RPC）
func recoveryStreamInterceptor(log *logger.Logger) grpc.StreamServerInterceptor {
	return func(
		srv interface{},
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) (err error) {
		defer func() {
			if r := recover(); r != nil {
				log.Error("gRPC stream panic recovered",
					"method", info.FullMethod,
					"panic", r,
				)
				err = fmt.Errorf("internal server error")
			}
		}()
		return handler(srv, ss)
	}
}

// loggingStreamInterceptor 日志拦截器（流式 RPC）
func loggingStreamInterceptor(log *logger.Logger) grpc.StreamServerInterceptor {
	return func(
		srv interface{},
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		start := time.Now()
		log.Debug("gRPC stream started", "method", info.FullMethod)

		err := handler(srv, ss)

		duration := time.Since(start)
		if err != nil {
			log.Error("gRPC stream failed",
				"method", info.FullMethod,
				"duration", duration,
				"error", err,
			)
		} else {
			log.Debug("gRPC stream completed",
				"method", info.FullMethod,
				"duration", duration,
			)
		}

		return err
	}
}

// metricsStreamInterceptor 指标拦截器（流式 RPC）
func metricsStreamInterceptor() grpc.StreamServerInterceptor {
	return func(
		srv interface{},
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		start := time.Now()
		err := handler(srv, ss)
		_ = time.Since(start)
		// TODO: 记录 Prometheus 指标
		return err
	}
}

