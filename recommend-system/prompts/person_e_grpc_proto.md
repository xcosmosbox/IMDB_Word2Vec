# Person E: gRPC & Proto（服务间通信）

## 你的角色
你是一名 Go 后端工程师，负责实现生成式推荐系统的 **gRPC 服务定义** 和 **服务间通信** 模块。

## 背景知识

微服务架构中，服务间通信使用 gRPC 比 HTTP REST 更高效：
- 二进制协议，更小的传输体积
- 双向流式通信
- 强类型接口定义
- 代码自动生成

## 你的任务

实现以下模块：

```
recommend-system/
├── proto/
│   ├── recommend/v1/
│   │   └── recommend.proto      # 推荐服务 Proto
│   ├── user/v1/
│   │   └── user.proto           # 用户服务 Proto
│   └── item/v1/
│       └── item.proto           # 物品服务 Proto
├── internal/grpc/
│   ├── server.go                # gRPC 服务器
│   └── client.go                # gRPC 客户端
└── scripts/
    └── gen_proto.sh             # Proto 生成脚本
```

---

## 1. proto/recommend/v1/recommend.proto

```protobuf
syntax = "proto3";

package recommend.v1;

option go_package = "recommend-system/proto/recommend/v1;recommendv1";

import "google/protobuf/timestamp.proto";

// 推荐服务
service RecommendService {
    // 获取推荐列表
    rpc GetRecommendations(GetRecommendationsRequest) returns (GetRecommendationsResponse);
    
    // 获取相似物品
    rpc GetSimilarItems(GetSimilarItemsRequest) returns (GetSimilarItemsResponse);
    
    // 提交反馈
    rpc SubmitFeedback(SubmitFeedbackRequest) returns (SubmitFeedbackResponse);
    
    // 流式推荐（实时推荐流）
    rpc StreamRecommendations(StreamRecommendationsRequest) returns (stream RecommendationEvent);
}

// 获取推荐请求
message GetRecommendationsRequest {
    string user_id = 1;
    int32 limit = 2;
    Context context = 3;
    repeated string exclude_items = 4;
    string scene = 5;  // home, search, detail
}

// 推荐上下文
message Context {
    string device = 1;
    string os = 2;
    string location = 3;
    string page_context = 4;
    google.protobuf.Timestamp timestamp = 5;
}

// 获取推荐响应
message GetRecommendationsResponse {
    repeated Recommendation recommendations = 1;
    string request_id = 2;
    string strategy = 3;
}

// 推荐项
message Recommendation {
    string item_id = 1;
    float score = 2;
    string reason = 3;
    SemanticID semantic_id = 4;
}

// 语义 ID
message SemanticID {
    int32 l1 = 1;
    int32 l2 = 2;
    int32 l3 = 3;
}

// 获取相似物品请求
message GetSimilarItemsRequest {
    string item_id = 1;
    int32 limit = 2;
}

// 获取相似物品响应
message GetSimilarItemsResponse {
    repeated SimilarItem items = 1;
}

// 相似物品
message SimilarItem {
    string item_id = 1;
    float similarity = 2;
}

// 提交反馈请求
message SubmitFeedbackRequest {
    string user_id = 1;
    string item_id = 2;
    string action = 3;  // click, like, dislike, buy
    string request_id = 4;
    google.protobuf.Timestamp timestamp = 5;
}

// 提交反馈响应
message SubmitFeedbackResponse {
    bool success = 1;
}

// 流式推荐请求
message StreamRecommendationsRequest {
    string user_id = 1;
    Context context = 2;
}

// 推荐事件
message RecommendationEvent {
    oneof event {
        Recommendation recommendation = 1;
        string heartbeat = 2;
    }
}
```

---

## 2. proto/user/v1/user.proto

```protobuf
syntax = "proto3";

package user.v1;

option go_package = "recommend-system/proto/user/v1;userv1";

import "google/protobuf/timestamp.proto";

// 用户服务
service UserService {
    // 获取用户
    rpc GetUser(GetUserRequest) returns (GetUserResponse);
    
    // 批量获取用户
    rpc BatchGetUsers(BatchGetUsersRequest) returns (BatchGetUsersResponse);
    
    // 获取用户行为
    rpc GetUserBehaviors(GetUserBehaviorsRequest) returns (GetUserBehaviorsResponse);
    
    // 获取用户画像
    rpc GetUserProfile(GetUserProfileRequest) returns (GetUserProfileResponse);
}

// 获取用户请求
message GetUserRequest {
    string user_id = 1;
}

// 获取用户响应
message GetUserResponse {
    User user = 1;
}

// 用户
message User {
    string id = 1;
    string name = 2;
    string email = 3;
    int32 age = 4;
    string gender = 5;
    google.protobuf.Timestamp created_at = 6;
    google.protobuf.Timestamp updated_at = 7;
}

// 批量获取用户请求
message BatchGetUsersRequest {
    repeated string user_ids = 1;
}

// 批量获取用户响应
message BatchGetUsersResponse {
    repeated User users = 1;
}

// 获取用户行为请求
message GetUserBehaviorsRequest {
    string user_id = 1;
    int32 limit = 2;
}

// 获取用户行为响应
message GetUserBehaviorsResponse {
    repeated UserBehavior behaviors = 1;
}

// 用户行为
message UserBehavior {
    string user_id = 1;
    string item_id = 2;
    string action = 3;
    google.protobuf.Timestamp timestamp = 4;
    map<string, string> context = 5;
}

// 获取用户画像请求
message GetUserProfileRequest {
    string user_id = 1;
}

// 获取用户画像响应
message GetUserProfileResponse {
    UserProfile profile = 1;
}

// 用户画像
message UserProfile {
    User user = 1;
    int32 total_actions = 2;
    repeated CategoryScore preferred_categories = 3;
    repeated int32 active_hours = 4;
    google.protobuf.Timestamp last_active = 5;
}

// 类别得分
message CategoryScore {
    string category = 1;
    float score = 2;
}
```

---

## 3. proto/item/v1/item.proto

```protobuf
syntax = "proto3";

package item.v1;

option go_package = "recommend-system/proto/item/v1;itemv1";

import "google/protobuf/timestamp.proto";
import "google/protobuf/struct.proto";

// 物品服务
service ItemService {
    // 获取物品
    rpc GetItem(GetItemRequest) returns (GetItemResponse);
    
    // 批量获取物品
    rpc BatchGetItems(BatchGetItemsRequest) returns (BatchGetItemsResponse);
    
    // 搜索物品
    rpc SearchItems(SearchItemsRequest) returns (SearchItemsResponse);
    
    // 获取物品统计
    rpc GetItemStats(GetItemStatsRequest) returns (GetItemStatsResponse);
}

// 获取物品请求
message GetItemRequest {
    string item_id = 1;
}

// 获取物品响应
message GetItemResponse {
    Item item = 1;
}

// 物品
message Item {
    string id = 1;
    string type = 2;
    string title = 3;
    string description = 4;
    string category = 5;
    repeated string tags = 6;
    google.protobuf.Struct metadata = 7;
    string status = 8;
    google.protobuf.Timestamp created_at = 9;
    google.protobuf.Timestamp updated_at = 10;
}

// 批量获取物品请求
message BatchGetItemsRequest {
    repeated string item_ids = 1;
}

// 批量获取物品响应
message BatchGetItemsResponse {
    repeated Item items = 1;
}

// 搜索物品请求
message SearchItemsRequest {
    string query = 1;
    int32 limit = 2;
    string type = 3;
    string category = 4;
}

// 搜索物品响应
message SearchItemsResponse {
    repeated Item items = 1;
    int64 total = 2;
}

// 获取物品统计请求
message GetItemStatsRequest {
    string item_id = 1;
}

// 获取物品统计响应
message GetItemStatsResponse {
    ItemStats stats = 1;
}

// 物品统计
message ItemStats {
    string item_id = 1;
    int64 view_count = 2;
    int64 click_count = 3;
    int64 like_count = 4;
    int64 share_count = 5;
    double avg_rating = 6;
}
```

---

## 4. internal/grpc/server.go

```go
package grpc

import (
    "context"
    "net"
    
    "google.golang.org/grpc"
    "google.golang.org/grpc/reflection"
    
    recommendv1 "recommend-system/proto/recommend/v1"
    userv1 "recommend-system/proto/user/v1"
    itemv1 "recommend-system/proto/item/v1"
    "recommend-system/pkg/logger"
)

// Server gRPC 服务器
type Server struct {
    grpcServer *grpc.Server
    listener   net.Listener
    logger     *logger.Logger
}

// ServerConfig 服务器配置
type ServerConfig struct {
    Address string
}

// NewServer 创建 gRPC 服务器
func NewServer(cfg ServerConfig, logger *logger.Logger) (*Server, error) {
    lis, err := net.Listen("tcp", cfg.Address)
    if err != nil {
        return nil, err
    }
    
    // 创建 gRPC 服务器
    grpcServer := grpc.NewServer(
        grpc.UnaryInterceptor(UnaryServerInterceptor(logger)),
        grpc.StreamInterceptor(StreamServerInterceptor(logger)),
    )
    
    // 启用 reflection（方便调试）
    reflection.Register(grpcServer)
    
    return &Server{
        grpcServer: grpcServer,
        listener:   lis,
        logger:     logger,
    }, nil
}

// RegisterRecommendService 注册推荐服务
func (s *Server) RegisterRecommendService(svc recommendv1.RecommendServiceServer) {
    recommendv1.RegisterRecommendServiceServer(s.grpcServer, svc)
}

// RegisterUserService 注册用户服务
func (s *Server) RegisterUserService(svc userv1.UserServiceServer) {
    userv1.RegisterUserServiceServer(s.grpcServer, svc)
}

// RegisterItemService 注册物品服务
func (s *Server) RegisterItemService(svc itemv1.ItemServiceServer) {
    itemv1.RegisterItemServiceServer(s.grpcServer, svc)
}

// Start 启动服务器
func (s *Server) Start() error {
    s.logger.Info("gRPC server starting", "address", s.listener.Addr().String())
    return s.grpcServer.Serve(s.listener)
}

// Stop 停止服务器
func (s *Server) Stop() {
    s.grpcServer.GracefulStop()
}

// UnaryServerInterceptor 一元 RPC 拦截器
func UnaryServerInterceptor(logger *logger.Logger) grpc.UnaryServerInterceptor {
    return func(
        ctx context.Context,
        req interface{},
        info *grpc.UnaryServerInfo,
        handler grpc.UnaryHandler,
    ) (interface{}, error) {
        // 记录请求
        logger.Info("gRPC request", "method", info.FullMethod)
        
        // 调用处理器
        resp, err := handler(ctx, req)
        
        if err != nil {
            logger.Error("gRPC error", "method", info.FullMethod, "error", err)
        }
        
        return resp, err
    }
}

// StreamServerInterceptor 流式 RPC 拦截器
func StreamServerInterceptor(logger *logger.Logger) grpc.StreamServerInterceptor {
    return func(
        srv interface{},
        ss grpc.ServerStream,
        info *grpc.StreamServerInfo,
        handler grpc.StreamHandler,
    ) error {
        logger.Info("gRPC stream", "method", info.FullMethod)
        return handler(srv, ss)
    }
}
```

---

## 5. internal/grpc/client.go

```go
package grpc

import (
    "context"
    "time"
    
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
    
    recommendv1 "recommend-system/proto/recommend/v1"
    userv1 "recommend-system/proto/user/v1"
    itemv1 "recommend-system/proto/item/v1"
)

// ClientConfig 客户端配置
type ClientConfig struct {
    RecommendServiceAddr string
    UserServiceAddr      string
    ItemServiceAddr      string
    Timeout              time.Duration
}

// Clients gRPC 客户端集合
type Clients struct {
    Recommend recommendv1.RecommendServiceClient
    User      userv1.UserServiceClient
    Item      itemv1.ItemServiceClient
    
    conns []*grpc.ClientConn
}

// NewClients 创建客户端集合
func NewClients(cfg ClientConfig) (*Clients, error) {
    clients := &Clients{
        conns: make([]*grpc.ClientConn, 0),
    }
    
    // 连接推荐服务
    if cfg.RecommendServiceAddr != "" {
        conn, err := grpc.Dial(
            cfg.RecommendServiceAddr,
            grpc.WithTransportCredentials(insecure.NewCredentials()),
        )
        if err != nil {
            return nil, err
        }
        clients.conns = append(clients.conns, conn)
        clients.Recommend = recommendv1.NewRecommendServiceClient(conn)
    }
    
    // 连接用户服务
    if cfg.UserServiceAddr != "" {
        conn, err := grpc.Dial(
            cfg.UserServiceAddr,
            grpc.WithTransportCredentials(insecure.NewCredentials()),
        )
        if err != nil {
            return nil, err
        }
        clients.conns = append(clients.conns, conn)
        clients.User = userv1.NewUserServiceClient(conn)
    }
    
    // 连接物品服务
    if cfg.ItemServiceAddr != "" {
        conn, err := grpc.Dial(
            cfg.ItemServiceAddr,
            grpc.WithTransportCredentials(insecure.NewCredentials()),
        )
        if err != nil {
            return nil, err
        }
        clients.conns = append(clients.conns, conn)
        clients.Item = itemv1.NewItemServiceClient(conn)
    }
    
    return clients, nil
}

// Close 关闭所有连接
func (c *Clients) Close() error {
    for _, conn := range c.conns {
        if err := conn.Close(); err != nil {
            return err
        }
    }
    return nil
}

// WithTimeout 创建带超时的上下文
func WithTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
    return context.WithTimeout(ctx, timeout)
}
```

---

## 6. scripts/gen_proto.sh

```bash
#!/bin/bash

# Proto 生成脚本
# 需要安装: protoc, protoc-gen-go, protoc-gen-go-grpc

set -e

PROTO_DIR="proto"
OUT_DIR="."

echo "Generating Go code from proto files..."

# 安装依赖（如果需要）
# go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
# go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# 生成推荐服务
protoc \
    --proto_path=${PROTO_DIR} \
    --go_out=${OUT_DIR} \
    --go_opt=paths=source_relative \
    --go-grpc_out=${OUT_DIR} \
    --go-grpc_opt=paths=source_relative \
    ${PROTO_DIR}/recommend/v1/recommend.proto

# 生成用户服务
protoc \
    --proto_path=${PROTO_DIR} \
    --go_out=${OUT_DIR} \
    --go_opt=paths=source_relative \
    --go-grpc_out=${OUT_DIR} \
    --go-grpc_opt=paths=source_relative \
    ${PROTO_DIR}/user/v1/user.proto

# 生成物品服务
protoc \
    --proto_path=${PROTO_DIR} \
    --go_out=${OUT_DIR} \
    --go_opt=paths=source_relative \
    --go-grpc_out=${OUT_DIR} \
    --go-grpc_opt=paths=source_relative \
    ${PROTO_DIR}/item/v1/item.proto

echo "Proto generation completed!"
```

---

## 注意事项

1. **版本管理**: Proto 文件放在 v1/ 目录下，便于版本迭代
2. **向后兼容**: 添加字段时使用新的 field number
3. **性能**: 批量接口减少 RPC 调用次数
4. **超时**: 客户端调用时设置合理的超时时间

## 输出要求

请输出完整的代码，包含：
1. 所有 Proto 文件
2. gRPC 服务端和客户端代码
3. 生成脚本
4. 使用示例

