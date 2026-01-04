// Package grpc 实现 gRPC 客户端
//
// 该文件提供了 gRPC 客户端的创建和管理功能，包括：
//   - 连接管理
//   - 重试策略
//   - 负载均衡
//   - 连接池
package grpc

import (
	"context"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"

	recommendv1 "recommend-system/proto/recommend/v1"
	userv1 "recommend-system/proto/user/v1"
	itemv1 "recommend-system/proto/item/v1"
)

// =============================================================================
// 客户端配置
// =============================================================================

// ClientConfig gRPC 客户端配置
type ClientConfig struct {
	// RecommendServiceAddr 推荐服务地址
	RecommendServiceAddr string

	// UserServiceAddr 用户服务地址
	UserServiceAddr string

	// ItemServiceAddr 物品服务地址
	ItemServiceAddr string

	// Timeout 默认请求超时时间
	Timeout time.Duration

	// MaxRetries 最大重试次数
	MaxRetries int

	// RetryBackoff 重试间隔基础值
	RetryBackoff time.Duration

	// KeepAliveTime 保活探测间隔
	KeepAliveTime time.Duration

	// KeepAliveTimeout 保活探测超时
	KeepAliveTimeout time.Duration

	// MaxRecvMsgSize 最大接收消息大小
	MaxRecvMsgSize int

	// MaxSendMsgSize 最大发送消息大小
	MaxSendMsgSize int

	// Insecure 是否使用不安全连接（无 TLS）
	Insecure bool
}

// DefaultClientConfig 返回默认客户端配置
func DefaultClientConfig() ClientConfig {
	return ClientConfig{
		Timeout:          30 * time.Second,
		MaxRetries:       3,
		RetryBackoff:     100 * time.Millisecond,
		KeepAliveTime:    30 * time.Second,
		KeepAliveTimeout: 10 * time.Second,
		MaxRecvMsgSize:   4 * 1024 * 1024, // 4MB
		MaxSendMsgSize:   4 * 1024 * 1024, // 4MB
		Insecure:         true,
	}
}

// =============================================================================
// gRPC 客户端集合
// =============================================================================

// Clients gRPC 客户端集合
type Clients struct {
	// Recommend 推荐服务客户端
	Recommend recommendv1.RecommendServiceClient

	// User 用户服务客户端
	User userv1.UserServiceClient

	// Item 物品服务客户端
	Item itemv1.ItemServiceClient

	// 内部字段
	config ClientConfig
	conns  []*grpc.ClientConn
	mu     sync.RWMutex
}

// NewClients 创建客户端集合
func NewClients(cfg ClientConfig) (*Clients, error) {
	clients := &Clients{
		config: cfg,
		conns:  make([]*grpc.ClientConn, 0),
	}

	// 连接推荐服务
	if cfg.RecommendServiceAddr != "" {
		conn, err := clients.createConnection(cfg.RecommendServiceAddr)
		if err != nil {
			clients.Close()
			return nil, fmt.Errorf("failed to connect recommend service: %w", err)
		}
		clients.conns = append(clients.conns, conn)
		clients.Recommend = recommendv1.NewRecommendServiceClient(conn)
	}

	// 连接用户服务
	if cfg.UserServiceAddr != "" {
		conn, err := clients.createConnection(cfg.UserServiceAddr)
		if err != nil {
			clients.Close()
			return nil, fmt.Errorf("failed to connect user service: %w", err)
		}
		clients.conns = append(clients.conns, conn)
		clients.User = userv1.NewUserServiceClient(conn)
	}

	// 连接物品服务
	if cfg.ItemServiceAddr != "" {
		conn, err := clients.createConnection(cfg.ItemServiceAddr)
		if err != nil {
			clients.Close()
			return nil, fmt.Errorf("failed to connect item service: %w", err)
		}
		clients.conns = append(clients.conns, conn)
		clients.Item = itemv1.NewItemServiceClient(conn)
	}

	return clients, nil
}

// createConnection 创建单个 gRPC 连接
func (c *Clients) createConnection(addr string) (*grpc.ClientConn, error) {
	opts := c.buildDialOptions()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, addr, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to dial %s: %w", addr, err)
	}

	return conn, nil
}

// buildDialOptions 构建连接选项
func (c *Clients) buildDialOptions() []grpc.DialOption {
	opts := []grpc.DialOption{
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(c.config.MaxRecvMsgSize),
			grpc.MaxCallSendMsgSize(c.config.MaxSendMsgSize),
		),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                c.config.KeepAliveTime,
			Timeout:             c.config.KeepAliveTimeout,
			PermitWithoutStream: true,
		}),
		grpc.WithChainUnaryInterceptor(
			timeoutUnaryInterceptor(c.config.Timeout),
			retryUnaryInterceptor(c.config.MaxRetries, c.config.RetryBackoff),
		),
	}

	// 安全选项
	if c.config.Insecure {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	return opts
}

// Close 关闭所有连接
func (c *Clients) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	var lastErr error
	for _, conn := range c.conns {
		if err := conn.Close(); err != nil {
			lastErr = err
		}
	}
	c.conns = nil
	return lastErr
}

// IsConnected 检查指定服务是否已连接
func (c *Clients) IsConnected(service string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	for _, conn := range c.conns {
		if conn.Target() == service {
			return conn.GetState() == connectivity.Ready
		}
	}
	return false
}

// GetConnState 获取连接状态
func (c *Clients) GetConnState() map[string]string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	states := make(map[string]string)
	for _, conn := range c.conns {
		states[conn.Target()] = conn.GetState().String()
	}
	return states
}

// =============================================================================
// 单服务客户端
// =============================================================================

// RecommendClient 推荐服务客户端封装
type RecommendClient struct {
	client recommendv1.RecommendServiceClient
	conn   *grpc.ClientConn
	config ClientConfig
}

// NewRecommendClient 创建推荐服务客户端
func NewRecommendClient(addr string, cfg ClientConfig) (*RecommendClient, error) {
	clients := &Clients{config: cfg}
	conn, err := clients.createConnection(addr)
	if err != nil {
		return nil, err
	}

	return &RecommendClient{
		client: recommendv1.NewRecommendServiceClient(conn),
		conn:   conn,
		config: cfg,
	}, nil
}

// GetRecommendations 获取推荐
func (c *RecommendClient) GetRecommendations(ctx context.Context, req *recommendv1.GetRecommendationsRequest) (*recommendv1.GetRecommendationsResponse, error) {
	return c.client.GetRecommendations(ctx, req)
}

// GetSimilarItems 获取相似物品
func (c *RecommendClient) GetSimilarItems(ctx context.Context, req *recommendv1.GetSimilarItemsRequest) (*recommendv1.GetSimilarItemsResponse, error) {
	return c.client.GetSimilarItems(ctx, req)
}

// SubmitFeedback 提交反馈
func (c *RecommendClient) SubmitFeedback(ctx context.Context, req *recommendv1.SubmitFeedbackRequest) (*recommendv1.SubmitFeedbackResponse, error) {
	return c.client.SubmitFeedback(ctx, req)
}

// Close 关闭连接
func (c *RecommendClient) Close() error {
	return c.conn.Close()
}

// UserClient 用户服务客户端封装
type UserClient struct {
	client userv1.UserServiceClient
	conn   *grpc.ClientConn
	config ClientConfig
}

// NewUserClient 创建用户服务客户端
func NewUserClient(addr string, cfg ClientConfig) (*UserClient, error) {
	clients := &Clients{config: cfg}
	conn, err := clients.createConnection(addr)
	if err != nil {
		return nil, err
	}

	return &UserClient{
		client: userv1.NewUserServiceClient(conn),
		conn:   conn,
		config: cfg,
	}, nil
}

// GetUser 获取用户
func (c *UserClient) GetUser(ctx context.Context, req *userv1.GetUserRequest) (*userv1.GetUserResponse, error) {
	return c.client.GetUser(ctx, req)
}

// CreateUser 创建用户
func (c *UserClient) CreateUser(ctx context.Context, req *userv1.CreateUserRequest) (*userv1.CreateUserResponse, error) {
	return c.client.CreateUser(ctx, req)
}

// GetUserProfile 获取用户画像
func (c *UserClient) GetUserProfile(ctx context.Context, req *userv1.GetUserProfileRequest) (*userv1.GetUserProfileResponse, error) {
	return c.client.GetUserProfile(ctx, req)
}

// RecordBehavior 记录用户行为
func (c *UserClient) RecordBehavior(ctx context.Context, req *userv1.RecordBehaviorRequest) (*userv1.RecordBehaviorResponse, error) {
	return c.client.RecordBehavior(ctx, req)
}

// Close 关闭连接
func (c *UserClient) Close() error {
	return c.conn.Close()
}

// ItemClient 物品服务客户端封装
type ItemClient struct {
	client itemv1.ItemServiceClient
	conn   *grpc.ClientConn
	config ClientConfig
}

// NewItemClient 创建物品服务客户端
func NewItemClient(addr string, cfg ClientConfig) (*ItemClient, error) {
	clients := &Clients{config: cfg}
	conn, err := clients.createConnection(addr)
	if err != nil {
		return nil, err
	}

	return &ItemClient{
		client: itemv1.NewItemServiceClient(conn),
		conn:   conn,
		config: cfg,
	}, nil
}

// GetItem 获取物品
func (c *ItemClient) GetItem(ctx context.Context, req *itemv1.GetItemRequest) (*itemv1.GetItemResponse, error) {
	return c.client.GetItem(ctx, req)
}

// CreateItem 创建物品
func (c *ItemClient) CreateItem(ctx context.Context, req *itemv1.CreateItemRequest) (*itemv1.CreateItemResponse, error) {
	return c.client.CreateItem(ctx, req)
}

// BatchGetItems 批量获取物品
func (c *ItemClient) BatchGetItems(ctx context.Context, req *itemv1.BatchGetItemsRequest) (*itemv1.BatchGetItemsResponse, error) {
	return c.client.BatchGetItems(ctx, req)
}

// SearchItems 搜索物品
func (c *ItemClient) SearchItems(ctx context.Context, req *itemv1.SearchItemsRequest) (*itemv1.SearchItemsResponse, error) {
	return c.client.SearchItems(ctx, req)
}

// GetSimilarItems 获取相似物品
func (c *ItemClient) GetSimilarItems(ctx context.Context, req *itemv1.GetSimilarItemsRequest) (*itemv1.GetSimilarItemsResponse, error) {
	return c.client.GetSimilarItems(ctx, req)
}

// Close 关闭连接
func (c *ItemClient) Close() error {
	return c.conn.Close()
}

// =============================================================================
// 客户端拦截器
// =============================================================================

// timeoutUnaryInterceptor 超时拦截器
func timeoutUnaryInterceptor(timeout time.Duration) grpc.UnaryClientInterceptor {
	return func(
		ctx context.Context,
		method string,
		req, reply interface{},
		cc *grpc.ClientConn,
		invoker grpc.UnaryInvoker,
		opts ...grpc.CallOption,
	) error {
		// 如果 context 没有设置 deadline，则添加默认超时
		if _, ok := ctx.Deadline(); !ok {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, timeout)
			defer cancel()
		}
		return invoker(ctx, method, req, reply, cc, opts...)
	}
}

// retryUnaryInterceptor 重试拦截器
func retryUnaryInterceptor(maxRetries int, backoff time.Duration) grpc.UnaryClientInterceptor {
	return func(
		ctx context.Context,
		method string,
		req, reply interface{},
		cc *grpc.ClientConn,
		invoker grpc.UnaryInvoker,
		opts ...grpc.CallOption,
	) error {
		var lastErr error
		for i := 0; i <= maxRetries; i++ {
			err := invoker(ctx, method, req, reply, cc, opts...)
			if err == nil {
				return nil
			}
			lastErr = err

			// 检查是否可重试
			if !isRetriable(err) {
				return err
			}

			// 最后一次不需要等待
			if i < maxRetries {
				// 指数退避
				wait := backoff * time.Duration(1<<uint(i))
				select {
				case <-ctx.Done():
					return ctx.Err()
				case <-time.After(wait):
				}
			}
		}
		return lastErr
	}
}

// isRetriable 判断错误是否可重试
func isRetriable(err error) bool {
	// TODO: 根据 gRPC 错误码判断是否可重试
	// 例如：UNAVAILABLE, RESOURCE_EXHAUSTED 等可重试
	// INVALID_ARGUMENT, NOT_FOUND 等不可重试
	return false
}

// =============================================================================
// 工具函数
// =============================================================================

// WithTimeout 创建带超时的上下文
func WithTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(ctx, timeout)
}

// WithDeadline 创建带截止时间的上下文
func WithDeadline(ctx context.Context, deadline time.Time) (context.Context, context.CancelFunc) {
	return context.WithDeadline(ctx, deadline)
}

