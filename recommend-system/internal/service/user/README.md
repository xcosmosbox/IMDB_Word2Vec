# 用户服务（User Service）

## 概述

用户服务是生成式推荐系统的核心微服务之一，负责用户相关的所有操作，包括：

- **用户管理**：用户的创建、查询、更新、删除（CRUD）
- **行为记录**：记录和查询用户与物品的交互行为
- **用户画像**：生成和维护用户画像信息

本模块严格遵循 `interfaces.UserService` 接口定义，采用依赖注入实现模块解耦。

---

## 目录结构

```
recommend-system/
├── api/user/v1/
│   ├── handler.go           # HTTP API 处理器
│   └── handler_test.go      # Handler 单元测试
├── cmd/user-service/
│   └── main.go              # 服务入口
└── internal/service/user/
    ├── service.go           # 核心业务逻辑
    ├── adapter.go           # 仓库和缓存适配器
    ├── mocks_test.go        # 测试 Mock 实现
    ├── service_test.go      # 服务层单元测试
    └── README.md            # 本文档
```

---

## 快速开始

### 1. 前置条件

- Go 1.21+
- PostgreSQL 14+
- Redis 7+

### 2. 配置文件

编辑 `configs/config.yaml`：

```yaml
server:
  name: user-service
  http_port: 8082
  mode: release  # debug, release, test

database:
  host: localhost
  port: 5432
  user: postgres
  password: your_password
  dbname: recommend_db
  sslmode: disable

redis:
  addrs:
    - localhost:6379
  password: ""
  db: 0

log:
  level: info
  format: json
```

### 3. 启动服务

```bash
# 开发模式
go run cmd/user-service/main.go -config=configs/config.yaml

# 或者使用环境变量
CONFIG_PATH=configs/config.yaml go run cmd/user-service/main.go
```

### 4. 验证服务

```bash
# 健康检查
curl http://localhost:8082/health

# 就绪检查
curl http://localhost:8082/ready
```

---

## API 接口

### 基础信息

| 属性 | 值 |
|------|-----|
| 基础路径 | `/api/v1` |
| 内容类型 | `application/json` |
| 服务端口 | `8082` (默认) |

### 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/users` | 创建用户 |
| GET | `/api/v1/users/:id` | 获取用户信息 |
| PUT | `/api/v1/users/:id` | 更新用户信息 |
| DELETE | `/api/v1/users/:id` | 删除用户 |
| GET | `/api/v1/users/:id/behaviors` | 获取用户行为历史 |
| POST | `/api/v1/users/:id/behaviors` | 记录用户行为 |
| GET | `/api/v1/users/:id/profile` | 获取用户画像 |

### 接口详情

#### 1. 创建用户

```bash
POST /api/v1/users
```

**请求体：**

```json
{
    "name": "张三",
    "email": "zhangsan@example.com",
    "age": 25,
    "gender": "male"
}
```

**响应：**

```json
{
    "code": 0,
    "message": "user created successfully",
    "data": {
        "id": "u_20250104150405_a1b2c3d4",
        "name": "张三",
        "email": "zhangsan@example.com",
        "age": 25,
        "gender": "male",
        "created_at": "2025-01-04T15:04:05Z",
        "updated_at": "2025-01-04T15:04:05Z"
    }
}
```

#### 2. 获取用户信息

```bash
GET /api/v1/users/:id
```

**响应：**

```json
{
    "code": 0,
    "data": {
        "id": "u_20250104150405_a1b2c3d4",
        "name": "张三",
        "email": "zhangsan@example.com",
        "age": 25,
        "gender": "male",
        "metadata": {
            "level": "vip"
        },
        "created_at": "2025-01-04T15:04:05Z",
        "updated_at": "2025-01-04T15:04:05Z"
    }
}
```

#### 3. 更新用户信息

```bash
PUT /api/v1/users/:id
```

**请求体（部分更新）：**

```json
{
    "name": "李四",
    "age": 30
}
```

#### 4. 删除用户

```bash
DELETE /api/v1/users/:id
```

**响应：**

```json
{
    "code": 0,
    "message": "user deleted successfully"
}
```

#### 5. 记录用户行为

```bash
POST /api/v1/users/:id/behaviors
```

**请求体：**

```json
{
    "item_id": "movie_tt0111161",
    "action": "view",
    "context": {
        "device_type": "mobile",
        "platform": "ios",
        "source": "recommend"
    }
}
```

**支持的行为类型：**

| 行为 | 说明 |
|------|------|
| `view` | 浏览 |
| `click` | 点击 |
| `like` | 点赞 |
| `dislike` | 点踩 |
| `favorite` | 收藏 |
| `share` | 分享 |
| `comment` | 评论 |
| `purchase` | 购买 |
| `rate` | 评分 |
| `play` | 播放 |
| `complete` | 完成观看/阅读 |

#### 6. 获取用户行为历史

```bash
GET /api/v1/users/:id/behaviors?limit=100
```

**查询参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `limit` | int | 20 | 返回数量，最大 100 |

#### 7. 获取用户画像

```bash
GET /api/v1/users/:id/profile
```

**响应：**

```json
{
    "code": 0,
    "data": {
        "user": { ... },
        "total_actions": 150,
        "preferred_types": {
            "view": 80,
            "click": 45,
            "like": 25
        },
        "active_hours": {
            "10": 20,
            "14": 35,
            "20": 45
        },
        "last_active": "2025-01-04T15:30:00Z"
    }
}
```

---

## 架构设计

### 分层架构

```
┌─────────────────────────────────────────────┐
│               HTTP Handler                   │  ← API 层
│            (api/user/v1)                     │
└─────────────────────┬───────────────────────┘
                      │
┌─────────────────────▼───────────────────────┐
│               User Service                   │  ← 业务逻辑层
│         (internal/service/user)              │
└─────────────────────┬───────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
│ Repository  │ │   Cache   │ │  Logger   │  ← 基础设施层
└─────────────┘ └───────────┘ └───────────┘
```

### 接口驱动

所有核心组件都通过接口定义（`internal/interfaces/interfaces.go`）：

```go
// UserService 用户服务接口
type UserService interface {
    GetUser(ctx context.Context, userID string) (*User, error)
    CreateUser(ctx context.Context, req *CreateUserRequest) (*User, error)
    UpdateUser(ctx context.Context, userID string, req *UpdateUserRequest) (*User, error)
    DeleteUser(ctx context.Context, userID string) error
    RecordBehavior(ctx context.Context, req *RecordBehaviorRequest) error
    GetUserBehaviors(ctx context.Context, userID string, limit int) ([]*UserBehavior, error)
    GetUserProfile(ctx context.Context, userID string) (*UserProfile, error)
}
```

### 缓存策略

| 数据类型 | 缓存 Key 前缀 | TTL | 说明 |
|----------|--------------|-----|------|
| 用户信息 | `user:` | 30分钟 | 更新时主动清除 |
| 用户画像 | `user:profile:` | 15分钟 | 更新时主动清除 |
| 行为历史 | `user:behavior:` | 5分钟 | 新增时清除 |

---

## 开发指南

### 添加新功能

1. **定义接口**（如需要）：在 `interfaces.go` 中添加新方法
2. **实现业务逻辑**：在 `service.go` 中实现
3. **添加 HTTP 端点**：在 `handler.go` 中添加处理函数
4. **编写测试**：添加对应的单元测试

### 示例：添加用户标签功能

```go
// 1. 在 interfaces.go 中添加方法签名
type UserService interface {
    // ... existing methods
    AddUserTags(ctx context.Context, userID string, tags []string) error
    GetUserTags(ctx context.Context, userID string) ([]string, error)
}

// 2. 在 service.go 中实现
func (s *Service) AddUserTags(ctx context.Context, userID string, tags []string) error {
    // 实现逻辑
}

// 3. 在 handler.go 中添加端点
func (h *Handler) AddUserTags(c *gin.Context) {
    // 处理请求
}
```

### 测试

```bash
# 运行所有测试
go test ./internal/service/user/... -v

# 运行特定测试
go test ./internal/service/user/... -run TestService_GetUser -v

# 生成覆盖率报告
go test ./internal/service/user/... -cover -coverprofile=coverage.out
go tool cover -html=coverage.out

# 运行 API Handler 测试
go test ./api/user/v1/... -v
```

### Mock 使用

测试时使用 Mock 实现：

```go
func TestMyFeature(t *testing.T) {
    // 创建 Mock
    mockRepo := NewMockUserRepository()
    mockCache := NewMockCache()
    
    // 配置 Mock 行为
    mockRepo.GetByIDFunc = func(ctx context.Context, userID string) (*interfaces.User, error) {
        return &interfaces.User{ID: userID, Name: "Test"}, nil
    }
    
    // 创建服务
    service := NewService(mockRepo, mockCache, nil)
    
    // 执行测试
    user, err := service.GetUser(context.Background(), "user_001")
    
    // 验证结果
    assert.NoError(t, err)
    assert.Equal(t, "Test", user.Name)
}
```

---

## 错误处理

### 错误码

| 错误码 | HTTP 状态码 | 说明 |
|--------|------------|------|
| 0 | 200/201 | 成功 |
| 400 | 400 | 请求参数错误 |
| 404 | 404 | 资源不存在 |
| 500 | 500 | 服务器内部错误 |

### 错误响应格式

```json
{
    "code": 404,
    "message": "user not found"
}
```

### 自定义错误

```go
var (
    ErrUserNotFound    = fmt.Errorf("user not found")
    ErrInvalidRequest  = fmt.Errorf("invalid request")
    ErrUserAlreadyExists = fmt.Errorf("user already exists")
)
```

---

## 性能优化

### 缓存优化

- 使用多级缓存（本地 + Redis）
- 缓存穿透保护
- 主动缓存失效

### 数据库优化

- 使用连接池
- 索引优化
- 批量操作支持

### 并发安全

- 所有 Mock 实现使用 sync.RWMutex
- 服务层方法是并发安全的

---

## 监控与日志

### 日志格式

```json
{
    "timestamp": "2025-01-04T15:04:05.000Z",
    "level": "info",
    "caller": "user/service.go:98",
    "message": "user created successfully",
    "user_id": "u_20250104150405_a1b2c3d4"
}
```

### 健康检查端点

| 端点 | 说明 |
|------|------|
| `/health` | 服务存活检查 |
| `/ready` | 服务就绪检查（含数据库和 Redis） |

---

## 数据库表结构

### users 表

```sql
CREATE TABLE users (
    id VARCHAR(64) PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(32),
    avatar_url VARCHAR(512),
    status INT DEFAULT 1,
    preferences JSONB,
    tags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);
```

### user_behaviors 表

```sql
CREATE TABLE user_behaviors (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    item_id VARCHAR(64) NOT NULL,
    item_type VARCHAR(32),
    action VARCHAR(32) NOT NULL,
    value FLOAT,
    context JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_behaviors_user_id ON user_behaviors(user_id);
CREATE INDEX idx_behaviors_timestamp ON user_behaviors(timestamp DESC);
CREATE INDEX idx_behaviors_user_item ON user_behaviors(user_id, item_id);
```

### user_profiles 表

```sql
CREATE TABLE user_profiles (
    user_id VARCHAR(64) PRIMARY KEY REFERENCES users(id),
    demographics JSONB,
    interests JSONB,
    behavior_stats JSONB,
    content_preferences JSONB,
    recent_items JSONB,
    long_term_interests JSONB,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

---

## 与其他服务的协作

### 依赖的服务

| 服务 | 用途 |
|------|------|
| PostgreSQL | 数据持久化 |
| Redis | 缓存 |

### 被依赖的服务

| 服务 | 说明 |
|------|------|
| 推荐服务 | 获取用户画像和行为 |
| 特征服务 | 获取用户特征 |
| 冷启动服务 | 新用户处理 |

### 事件发布（未来扩展）

```go
// 用户创建事件
type UserCreatedEvent struct {
    UserID    string    `json:"user_id"`
    Email     string    `json:"email"`
    CreatedAt time.Time `json:"created_at"`
}

// 行为记录事件
type BehaviorRecordedEvent struct {
    UserID    string    `json:"user_id"`
    ItemID    string    `json:"item_id"`
    Action    string    `json:"action"`
    Timestamp time.Time `json:"timestamp"`
}
```

---

## 常见问题

### Q: 如何扩展用户属性？

使用 `Metadata` 字段存储自定义属性：

```go
user.Metadata = map[string]string{
    "vip_level": "gold",
    "source": "app",
}
```

### Q: 如何处理高并发行为记录？

建议：
1. 使用消息队列异步处理
2. 批量写入数据库
3. 使用 Redis 做写缓冲

### Q: 缓存和数据库不一致怎么办？

- 更新时主动清除缓存（Cache Aside 模式）
- 设置合理的 TTL
- 关键数据直接查库

---

## 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0.0 | 2025-01-04 | 初始版本，实现基础 CRUD 和行为记录 |

---

## 联系方式

如有问题，请联系：
- 负责人：Person A
- 接口定义：`internal/interfaces/interfaces.go`
- 相关文档：`docs/生成式推荐系统架构设计.md`

