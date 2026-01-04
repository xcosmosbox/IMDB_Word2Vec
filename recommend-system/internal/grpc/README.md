# gRPC & Proto 模块说明文档

> Person E: gRPC 服务间通信模块

## 目录

- [概述](#概述)
- [架构设计](#架构设计)
- [目录结构](#目录结构)
- [Proto 文件](#proto-文件)
- [服务器实现](#服务器实现)
- [客户端实现](#客户端实现)
- [类型转换](#类型转换)
- [使用指南](#使用指南)
- [测试](#测试)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

---

## 概述

本模块实现了生成式推荐系统的 gRPC 服务间通信层，提供：

- **Proto 定义**：推荐服务、用户服务、物品服务的接口定义
- **gRPC 服务器**：高性能、可配置的服务器实现
- **gRPC 客户端**：支持连接池、重试、超时的客户端
- **类型转换**：Proto 消息与业务类型的双向转换

### 核心特性

| 特性 | 说明 |
|------|------|
| 高性能 | 基于 gRPC 二进制协议，传输效率高 |
| 类型安全 | Proto 强类型定义，自动代码生成 |
| 流式支持 | 支持服务端流式推送（如实时推荐） |
| 可扩展 | 拦截器机制支持日志、限流、追踪 |
| 健康检查 | 内置 gRPC 健康检查协议支持 |

---

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        外部客户端 / Gateway                       │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        gRPC 服务层                               │
│  ┌─────────────────┬─────────────────┬─────────────────┐       │
│  │ RecommendService│   UserService   │   ItemService   │       │
│  │   (推荐服务)     │   (用户服务)     │   (物品服务)     │       │
│  └────────┬────────┴────────┬────────┴────────┬────────┘       │
│           │                 │                 │                 │
│           └────────────────┼────────────────┘                 │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    拦截器链                              │   │
│  │  [Recovery] → [Logging] → [Metrics] → [Auth] → ...      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        业务逻辑层                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   interfaces.go                          │   │
│  │  UserService / ItemService / RecommendService / ...      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Proto 与 interfaces.go 映射

| interfaces.go | Proto 消息 | 说明 |
|---------------|------------|------|
| `User` | `userv1.User` | 用户信息 |
| `Item` | `itemv1.Item` | 物品信息 |
| `Recommendation` | `recommendv1.Recommendation` | 推荐项 |
| `UserBehavior` | `userv1.UserBehavior` | 用户行为 |
| `RecommendRequest` | `recommendv1.GetRecommendationsRequest` | 推荐请求 |
| `RecommendResponse` | `recommendv1.GetRecommendationsResponse` | 推荐响应 |
| `Feedback` | `recommendv1.SubmitFeedbackRequest` | 用户反馈 |

---

## 目录结构

```
recommend-system/
├── proto/                          # Proto 定义
│   ├── recommend/v1/
│   │   └── recommend.proto         # 推荐服务 Proto
│   ├── user/v1/
│   │   └── user.proto              # 用户服务 Proto
│   └── item/v1/
│       └── item.proto              # 物品服务 Proto
├── internal/grpc/                  # gRPC 实现
│   ├── server.go                   # 服务器实现
│   ├── client.go                   # 客户端实现
│   ├── converter.go                # 类型转换
│   ├── server_test.go              # 服务器测试
│   ├── client_test.go              # 客户端测试
│   ├── converter_test.go           # 转换器测试
│   └── README.md                   # 本文档
└── scripts/
    ├── gen_proto.sh                # Proto 生成脚本 (Linux/macOS)
    └── gen_proto.ps1               # Proto 生成脚本 (Windows)
```

---

## Proto 文件

### 推荐服务 (recommend.proto)

```protobuf
service RecommendService {
    // 获取个性化推荐
    rpc GetRecommendations(GetRecommendationsRequest) 
        returns (GetRecommendationsResponse);
    
    // 获取相似物品
    rpc GetSimilarItems(GetSimilarItemsRequest) 
        returns (GetSimilarItemsResponse);
    
    // 提交用户反馈
    rpc SubmitFeedback(SubmitFeedbackRequest) 
        returns (SubmitFeedbackResponse);
    
    // 流式推荐
    rpc StreamRecommendations(StreamRecommendationsRequest) 
        returns (stream RecommendationEvent);
}
```

### 用户服务 (user.proto)

```protobuf
service UserService {
    // 用户 CRUD
    rpc GetUser(GetUserRequest) returns (GetUserResponse);
    rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
    rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse);
    rpc DeleteUser(DeleteUserRequest) returns (DeleteUserResponse);
    
    // 批量操作
    rpc BatchGetUsers(BatchGetUsersRequest) returns (BatchGetUsersResponse);
    
    // 行为相关
    rpc RecordBehavior(RecordBehaviorRequest) returns (RecordBehaviorResponse);
    rpc GetUserBehaviors(GetUserBehaviorsRequest) returns (GetUserBehaviorsResponse);
    rpc GetUserProfile(GetUserProfileRequest) returns (GetUserProfileResponse);
}
```

### 物品服务 (item.proto)

```protobuf
service ItemService {
    // 物品 CRUD
    rpc GetItem(GetItemRequest) returns (GetItemResponse);
    rpc CreateItem(CreateItemRequest) returns (CreateItemResponse);
    rpc UpdateItem(UpdateItemRequest) returns (UpdateItemResponse);
    rpc DeleteItem(DeleteItemRequest) returns (DeleteItemResponse);
    
    // 批量与搜索
    rpc BatchGetItems(BatchGetItemsRequest) returns (BatchGetItemsResponse);
    rpc ListItems(ListItemsRequest) returns (ListItemsResponse);
    rpc SearchItems(SearchItemsRequest) returns (SearchItemsResponse);
    
    // 相似物品
    rpc GetSimilarItems(GetSimilarItemsRequest) returns (GetSimilarItemsResponse);
    
    // 统计
    rpc GetItemStats(GetItemStatsRequest) returns (GetItemStatsResponse);
}
```

### 生成 Go 代码

**Linux/macOS:**

```bash
cd recommend-system
chmod +x scripts/gen_proto.sh
./scripts/gen_proto.sh
```

**Windows PowerShell:**

```powershell
cd recommend-system
.\scripts\gen_proto.ps1
```

生成的文件：

```
proto/
├── recommend/v1/
│   ├── recommend.pb.go         # 消息定义
│   └── recommend_grpc.pb.go    # 服务定义
├── user/v1/
│   ├── user.pb.go
│   └── user_grpc.pb.go
└── item/v1/
    ├── item.pb.go
    └── item_grpc.pb.go
```

---

## 服务器实现

### 基本使用

```go
package main

import (
    "recommend-system/internal/grpc"
    "recommend-system/pkg/logger"
    recommendv1 "recommend-system/proto/recommend/v1"
)

func main() {
    // 创建日志器
    log := logger.NewLogger()
    
    // 创建服务器配置
    cfg := grpc.DefaultServerConfig()
    cfg.Address = ":50051"
    
    // 创建服务器
    server, err := grpc.NewServer(cfg, log)
    if err != nil {
        panic(err)
    }
    
    // 注册服务
    server.RegisterService(
        &recommendv1.RecommendService_ServiceDesc,
        NewRecommendServiceImpl(),
    )
    
    // 启动服务器
    if err := server.Start(); err != nil {
        panic(err)
    }
}
```

### 配置选项

```go
type ServerConfig struct {
    // 监听地址
    Address string // 默认 ":50051"
    
    // 消息大小限制
    MaxRecvMsgSize int // 默认 4MB
    MaxSendMsgSize int // 默认 4MB
    
    // 并发限制
    MaxConcurrentStreams uint32 // 默认 1000
    
    // 连接管理
    ConnectionTimeout time.Duration // 默认 120s
    KeepAliveTime     time.Duration // 默认 30s
    KeepAliveTimeout  time.Duration // 默认 10s
    
    // 功能开关
    EnableReflection  bool // 默认 true（调试用）
    EnableHealthCheck bool // 默认 true
}
```

### 优雅关闭

```go
// 优雅关闭（等待所有请求完成）
server.GracefulStop()

// 带超时的优雅关闭
server.GracefulStopWithTimeout(30 * time.Second)

// 立即关闭
server.Stop()
```

### 健康检查

服务器自动启用 gRPC 健康检查协议：

```go
// 设置服务状态
server.SetServiceStatus("recommend.v1.RecommendService", true)

// 客户端检查
import healthpb "google.golang.org/grpc/health/grpc_health_v1"

healthClient := healthpb.NewHealthClient(conn)
resp, err := healthClient.Check(ctx, &healthpb.HealthCheckRequest{
    Service: "recommend.v1.RecommendService",
})
```

---

## 客户端实现

### 客户端集合

```go
package main

import (
    "context"
    "recommend-system/internal/grpc"
    recommendv1 "recommend-system/proto/recommend/v1"
)

func main() {
    // 创建配置
    cfg := grpc.DefaultClientConfig()
    cfg.RecommendServiceAddr = "localhost:50051"
    cfg.UserServiceAddr = "localhost:50052"
    cfg.ItemServiceAddr = "localhost:50053"
    
    // 创建客户端集合
    clients, err := grpc.NewClients(cfg)
    if err != nil {
        panic(err)
    }
    defer clients.Close()
    
    // 使用推荐服务
    resp, err := clients.Recommend.GetRecommendations(
        context.Background(),
        &recommendv1.GetRecommendationsRequest{
            UserId: "user123",
            Limit:  20,
        },
    )
}
```

### 单服务客户端

```go
// 推荐服务客户端
recommendClient, err := grpc.NewRecommendClient("localhost:50051", cfg)
defer recommendClient.Close()

resp, err := recommendClient.GetRecommendations(ctx, req)

// 用户服务客户端
userClient, err := grpc.NewUserClient("localhost:50052", cfg)
defer userClient.Close()

// 物品服务客户端
itemClient, err := grpc.NewItemClient("localhost:50053", cfg)
defer itemClient.Close()
```

### 配置选项

```go
type ClientConfig struct {
    // 服务地址
    RecommendServiceAddr string
    UserServiceAddr      string
    ItemServiceAddr      string
    
    // 超时与重试
    Timeout      time.Duration // 默认 30s
    MaxRetries   int           // 默认 3
    RetryBackoff time.Duration // 默认 100ms
    
    // 连接管理
    KeepAliveTime    time.Duration // 默认 30s
    KeepAliveTimeout time.Duration // 默认 10s
    
    // 消息大小
    MaxRecvMsgSize int // 默认 4MB
    MaxSendMsgSize int // 默认 4MB
    
    // 安全
    Insecure bool // 默认 true（无 TLS）
}
```

### 连接状态监控

```go
// 检查连接状态
states := clients.GetConnState()
for addr, state := range states {
    fmt.Printf("%s: %s\n", addr, state)
}

// 检查特定服务
if clients.IsConnected("localhost:50051") {
    // 已连接
}
```

---

## 类型转换

### 转换函数

```go
import (
    "recommend-system/internal/grpc"
    "recommend-system/internal/interfaces"
)

// 业务类型 → Proto
protoUser := grpc.UserToProto(user)
protoItem := grpc.ItemToProto(item)
protoRec := grpc.RecommendationToProto(rec)

// Proto → 业务类型
user := grpc.ProtoToUser(protoUser)
item := grpc.ProtoToItem(protoItem)
rec := grpc.ProtoToRecommendation(protoRec)

// 批量转换
protoUsers := grpc.UsersToProto(users)
users := grpc.ProtoToUsers(protoUsers)

protoItems := grpc.ItemsToProto(items)
items := grpc.ProtoToItems(protoItems)

// 请求/响应转换
protoReq := grpc.RecommendRequestToProto(req)
req := grpc.ProtoToRecommendRequest(protoReq)

protoResp := grpc.RecommendResponseToProto(resp)
resp := grpc.ProtoToRecommendResponse(protoResp)
```

### 时间转换

```go
import "google.golang.org/protobuf/types/known/timestamppb"

// time.Time → Timestamp
ts := grpc.TimeToProto(time.Now())

// Timestamp → time.Time
t := grpc.ProtoToTime(ts)
```

---

## 使用指南

### 实现 gRPC 服务

1. **实现服务接口**

```go
package service

import (
    "context"
    
    "recommend-system/internal/interfaces"
    recommendv1 "recommend-system/proto/recommend/v1"
    grpcpkg "recommend-system/internal/grpc"
)

type RecommendServiceImpl struct {
    recommendv1.UnimplementedRecommendServiceServer
    
    svc interfaces.RecommendService
}

func NewRecommendServiceImpl(svc interfaces.RecommendService) *RecommendServiceImpl {
    return &RecommendServiceImpl{svc: svc}
}

func (s *RecommendServiceImpl) GetRecommendations(
    ctx context.Context,
    req *recommendv1.GetRecommendationsRequest,
) (*recommendv1.GetRecommendationsResponse, error) {
    // 转换请求
    bizReq := grpcpkg.ProtoToRecommendRequest(req)
    
    // 调用业务逻辑
    bizResp, err := s.svc.GetRecommendations(ctx, bizReq)
    if err != nil {
        return nil, err
    }
    
    // 转换响应
    return grpcpkg.RecommendResponseToProto(bizResp), nil
}
```

2. **注册服务**

```go
server.RegisterService(
    &recommendv1.RecommendService_ServiceDesc,
    NewRecommendServiceImpl(recommendService),
)
```

### 添加新接口

1. 在对应的 `.proto` 文件中添加 RPC 方法和消息定义
2. 运行 `gen_proto.sh` 或 `gen_proto.ps1` 重新生成代码
3. 在 `converter.go` 中添加类型转换函数
4. 实现服务接口方法

---

## 测试

### 运行测试

```bash
cd recommend-system

# 运行所有测试
go test ./internal/grpc/...

# 运行特定测试
go test ./internal/grpc/ -run TestUserToProto

# 运行基准测试
go test ./internal/grpc/ -bench=.

# 生成覆盖率报告
go test ./internal/grpc/... -coverprofile=coverage.out
go tool cover -html=coverage.out
```

### 测试文件

| 文件 | 说明 |
|------|------|
| `server_test.go` | 服务器配置、创建、启停测试 |
| `client_test.go` | 客户端配置、连接、拦截器测试 |
| `converter_test.go` | 类型转换正确性测试 |

---

## 最佳实践

### 1. 版本管理

Proto 文件放在 `v1/` 目录下，便于版本迭代：

```
proto/recommend/v1/recommend.proto
proto/recommend/v2/recommend.proto  # 新版本
```

### 2. 向后兼容

- 添加新字段使用新的 field number
- 不要删除或修改已有字段
- 使用 `reserved` 标记废弃字段

```protobuf
message User {
    string id = 1;
    string name = 2;
    // 废弃字段
    reserved 3;
    reserved "old_field";
    // 新字段
    string new_field = 4;
}
```

### 3. 超时设置

```go
// 客户端设置超时
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

resp, err := client.GetRecommendations(ctx, req)
```

### 4. 错误处理

```go
import "google.golang.org/grpc/status"

// 服务端返回错误
return nil, status.Error(codes.NotFound, "user not found")

// 客户端处理错误
resp, err := client.GetUser(ctx, req)
if err != nil {
    st, ok := status.FromError(err)
    if ok {
        switch st.Code() {
        case codes.NotFound:
            // 处理未找到
        case codes.Unavailable:
            // 处理服务不可用
        }
    }
}
```

### 5. 批量接口

使用批量接口减少 RPC 调用次数：

```go
// 推荐使用
resp, err := client.BatchGetItems(ctx, &itemv1.BatchGetItemsRequest{
    ItemIds: []string{"item1", "item2", "item3"},
})

// 不推荐：多次调用
for _, id := range itemIds {
    resp, err := client.GetItem(ctx, &itemv1.GetItemRequest{ItemId: id})
}
```

---

## 常见问题

### Q1: 如何生成 Proto 代码？

```bash
# 安装依赖
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# 生成代码
./scripts/gen_proto.sh
```

### Q2: 连接失败怎么办？

1. 检查服务器是否启动
2. 检查端口是否被占用
3. 检查防火墙设置
4. 查看连接状态：`clients.GetConnState()`

### Q3: 如何启用 TLS？

```go
// 服务端
import "google.golang.org/grpc/credentials"

creds, err := credentials.NewServerTLSFromFile("cert.pem", "key.pem")
grpcServer := grpc.NewServer(grpc.Creds(creds))

// 客户端
creds, err := credentials.NewClientTLSFromFile("cert.pem", "")
conn, err := grpc.Dial(addr, grpc.WithTransportCredentials(creds))
```

### Q4: 如何添加认证？

```go
// 使用 metadata 传递 token
import "google.golang.org/grpc/metadata"

md := metadata.Pairs("authorization", "Bearer "+token)
ctx := metadata.NewOutgoingContext(context.Background(), md)
resp, err := client.GetRecommendations(ctx, req)
```

### Q5: 性能优化建议？

1. **连接复用**：使用 `Clients` 复用连接
2. **批量操作**：使用 `BatchGet*` 接口
3. **压缩**：启用 gRPC 压缩
4. **连接池**：生产环境配置连接池

---

## 相关文档

- [interfaces.go](../interfaces/interfaces.go) - 接口定义
- [生成式推荐系统架构设计](../../../docs/生成式推荐系统架构设计.md) - 整体架构
- [gRPC 官方文档](https://grpc.io/docs/)
- [Protocol Buffers 语言指南](https://developers.google.com/protocol-buffers/docs/proto3)

---

## 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-01-04 | 初始版本，实现基础 gRPC 服务 |

