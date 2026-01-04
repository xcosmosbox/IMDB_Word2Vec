# Person A: User Service（用户服务）

## 你的角色
你是一名 Go 后端工程师，负责实现生成式推荐系统的 **用户服务** 模块。

## 背景知识

用户服务是推荐系统的核心服务之一，负责：
- 用户信息的 CRUD 操作
- 用户行为记录与查询
- 用户画像管理
- 用户偏好设置

## 你的任务

实现以下模块：

```
recommend-system/
├── api/user/v1/
│   └── handler.go           # HTTP API 处理器
├── cmd/user-service/
│   └── main.go              # 服务入口
└── internal/service/user/
    └── service.go           # 业务逻辑
```

---

## 1. internal/service/user/service.go

```go
package user

import (
    "context"
    "time"
    
    "recommend-system/internal/model"
    "recommend-system/internal/repository"
    "recommend-system/internal/cache"
    "recommend-system/pkg/logger"
)

// UserService 用户服务
type UserService struct {
    userRepo repository.UserRepo
    cache    cache.Cache
    logger   *logger.Logger
}

// NewUserService 创建用户服务
func NewUserService(
    userRepo repository.UserRepo,
    cache cache.Cache,
    logger *logger.Logger,
) *UserService {
    return &UserService{
        userRepo: userRepo,
        cache:    cache,
        logger:   logger,
    }
}

// GetUser 获取用户信息
func (s *UserService) GetUser(ctx context.Context, userID string) (*model.User, error) {
    // 1. 先查缓存
    cacheKey := "user:" + userID
    var user model.User
    err := s.cache.Get(ctx, cacheKey, &user)
    if err == nil {
        return &user, nil
    }
    
    // 2. 查数据库
    userPtr, err := s.userRepo.GetByID(ctx, userID)
    if err != nil {
        return nil, err
    }
    
    // 3. 写入缓存
    _ = s.cache.Set(ctx, cacheKey, userPtr, 30*time.Minute)
    
    return userPtr, nil
}

// CreateUser 创建用户
func (s *UserService) CreateUser(ctx context.Context, req *CreateUserRequest) (*model.User, error) {
    user := &model.User{
        ID:        generateUserID(),
        Name:      req.Name,
        Email:     req.Email,
        Age:       req.Age,
        Gender:    req.Gender,
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    if err := s.userRepo.Create(ctx, user); err != nil {
        return nil, err
    }
    
    return user, nil
}

// UpdateUser 更新用户信息
func (s *UserService) UpdateUser(ctx context.Context, userID string, req *UpdateUserRequest) (*model.User, error) {
    user, err := s.userRepo.GetByID(ctx, userID)
    if err != nil {
        return nil, err
    }
    
    // 更新字段
    if req.Name != "" {
        user.Name = req.Name
    }
    if req.Email != "" {
        user.Email = req.Email
    }
    if req.Age > 0 {
        user.Age = req.Age
    }
    user.UpdatedAt = time.Now()
    
    if err := s.userRepo.Update(ctx, user); err != nil {
        return nil, err
    }
    
    // 清除缓存
    _ = s.cache.Delete(ctx, "user:"+userID)
    
    return user, nil
}

// DeleteUser 删除用户
func (s *UserService) DeleteUser(ctx context.Context, userID string) error {
    if err := s.userRepo.Delete(ctx, userID); err != nil {
        return err
    }
    
    // 清除缓存
    _ = s.cache.Delete(ctx, "user:"+userID)
    
    return nil
}

// RecordBehavior 记录用户行为
func (s *UserService) RecordBehavior(ctx context.Context, req *RecordBehaviorRequest) error {
    behavior := &model.UserBehavior{
        UserID:    req.UserID,
        ItemID:    req.ItemID,
        Action:    req.Action,
        Timestamp: time.Now(),
        Context:   req.Context,
    }
    
    return s.userRepo.AddBehavior(ctx, behavior)
}

// GetUserBehaviors 获取用户行为历史
func (s *UserService) GetUserBehaviors(ctx context.Context, userID string, limit int) ([]*model.UserBehavior, error) {
    return s.userRepo.GetBehaviors(ctx, userID, limit)
}

// GetUserProfile 获取用户画像
func (s *UserService) GetUserProfile(ctx context.Context, userID string) (*UserProfile, error) {
    // 获取用户基本信息
    user, err := s.GetUser(ctx, userID)
    if err != nil {
        return nil, err
    }
    
    // 获取用户行为统计
    behaviors, err := s.userRepo.GetBehaviors(ctx, userID, 1000)
    if err != nil {
        return nil, err
    }
    
    // 计算用户偏好
    profile := &UserProfile{
        User:           user,
        TotalActions:   len(behaviors),
        PreferredTypes: s.calculatePreferredTypes(behaviors),
        ActiveHours:    s.calculateActiveHours(behaviors),
        LastActive:     s.getLastActiveTime(behaviors),
    }
    
    return profile, nil
}

// 请求/响应结构体
type CreateUserRequest struct {
    Name   string `json:"name" binding:"required"`
    Email  string `json:"email" binding:"required,email"`
    Age    int    `json:"age"`
    Gender string `json:"gender"`
}

type UpdateUserRequest struct {
    Name   string `json:"name"`
    Email  string `json:"email"`
    Age    int    `json:"age"`
    Gender string `json:"gender"`
}

type RecordBehaviorRequest struct {
    UserID  string            `json:"user_id" binding:"required"`
    ItemID  string            `json:"item_id" binding:"required"`
    Action  string            `json:"action" binding:"required"`
    Context map[string]string `json:"context"`
}

type UserProfile struct {
    User           *model.User       `json:"user"`
    TotalActions   int               `json:"total_actions"`
    PreferredTypes map[string]int    `json:"preferred_types"`
    ActiveHours    map[int]int       `json:"active_hours"`
    LastActive     time.Time         `json:"last_active"`
}

// 辅助函数
func generateUserID() string {
    // 使用 UUID 或 snowflake 生成唯一 ID
    return "u_" + time.Now().Format("20060102150405") + "_" + randomString(8)
}

func randomString(n int) string {
    // 生成随机字符串
    // 实现细节...
    return ""
}

func (s *UserService) calculatePreferredTypes(behaviors []*model.UserBehavior) map[string]int {
    types := make(map[string]int)
    // 统计各类型行为次数
    // 实现细节...
    return types
}

func (s *UserService) calculateActiveHours(behaviors []*model.UserBehavior) map[int]int {
    hours := make(map[int]int)
    for _, b := range behaviors {
        hour := b.Timestamp.Hour()
        hours[hour]++
    }
    return hours
}

func (s *UserService) getLastActiveTime(behaviors []*model.UserBehavior) time.Time {
    if len(behaviors) == 0 {
        return time.Time{}
    }
    return behaviors[0].Timestamp
}
```

---

## 2. api/user/v1/handler.go

```go
package v1

import (
    "net/http"
    
    "github.com/gin-gonic/gin"
    
    "recommend-system/internal/service/user"
)

// UserHandler 用户 API 处理器
type UserHandler struct {
    userService *user.UserService
}

// NewUserHandler 创建处理器
func NewUserHandler(userService *user.UserService) *UserHandler {
    return &UserHandler{userService: userService}
}

// RegisterRoutes 注册路由
func (h *UserHandler) RegisterRoutes(r *gin.RouterGroup) {
    users := r.Group("/users")
    {
        users.POST("", h.CreateUser)
        users.GET("/:id", h.GetUser)
        users.PUT("/:id", h.UpdateUser)
        users.DELETE("/:id", h.DeleteUser)
        users.GET("/:id/behaviors", h.GetUserBehaviors)
        users.POST("/:id/behaviors", h.RecordBehavior)
        users.GET("/:id/profile", h.GetUserProfile)
    }
}

// CreateUser 创建用户
// POST /api/v1/users
func (h *UserHandler) CreateUser(c *gin.Context) {
    var req user.CreateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    result, err := h.userService.CreateUser(c.Request.Context(), &req)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusCreated, gin.H{
        "code": 0,
        "data": result,
    })
}

// GetUser 获取用户
// GET /api/v1/users/:id
func (h *UserHandler) GetUser(c *gin.Context) {
    userID := c.Param("id")
    
    result, err := h.userService.GetUser(c.Request.Context(), userID)
    if err != nil {
        c.JSON(http.StatusNotFound, gin.H{"error": "user not found"})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{
        "code": 0,
        "data": result,
    })
}

// UpdateUser 更新用户
// PUT /api/v1/users/:id
func (h *UserHandler) UpdateUser(c *gin.Context) {
    userID := c.Param("id")
    
    var req user.UpdateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    result, err := h.userService.UpdateUser(c.Request.Context(), userID, &req)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{
        "code": 0,
        "data": result,
    })
}

// DeleteUser 删除用户
// DELETE /api/v1/users/:id
func (h *UserHandler) DeleteUser(c *gin.Context) {
    userID := c.Param("id")
    
    if err := h.userService.DeleteUser(c.Request.Context(), userID); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{
        "code": 0,
        "message": "deleted",
    })
}

// GetUserBehaviors 获取用户行为历史
// GET /api/v1/users/:id/behaviors
func (h *UserHandler) GetUserBehaviors(c *gin.Context) {
    userID := c.Param("id")
    limit := 100 // 默认限制
    
    behaviors, err := h.userService.GetUserBehaviors(c.Request.Context(), userID, limit)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{
        "code": 0,
        "data": behaviors,
    })
}

// RecordBehavior 记录用户行为
// POST /api/v1/users/:id/behaviors
func (h *UserHandler) RecordBehavior(c *gin.Context) {
    userID := c.Param("id")
    
    var req user.RecordBehaviorRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    req.UserID = userID
    
    if err := h.userService.RecordBehavior(c.Request.Context(), &req); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusCreated, gin.H{
        "code": 0,
        "message": "recorded",
    })
}

// GetUserProfile 获取用户画像
// GET /api/v1/users/:id/profile
func (h *UserHandler) GetUserProfile(c *gin.Context) {
    userID := c.Param("id")
    
    profile, err := h.userService.GetUserProfile(c.Request.Context(), userID)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{
        "code": 0,
        "data": profile,
    })
}
```

---

## 3. cmd/user-service/main.go

```go
package main

import (
    "context"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"
    
    "github.com/gin-gonic/gin"
    
    "recommend-system/pkg/config"
    "recommend-system/pkg/logger"
    "recommend-system/pkg/database"
    "recommend-system/internal/cache"
    "recommend-system/internal/repository"
    "recommend-system/internal/middleware"
    "recommend-system/internal/service/user"
    userv1 "recommend-system/api/user/v1"
)

func main() {
    // 加载配置
    cfg, err := config.LoadConfig("configs/config.yaml")
    if err != nil {
        log.Fatalf("Failed to load config: %v", err)
    }
    
    // 初始化日志
    zapLogger := logger.InitLogger(cfg.Logger.Level)
    
    // 初始化数据库
    db, err := database.InitPostgres(cfg.Database)
    if err != nil {
        log.Fatalf("Failed to connect database: %v", err)
    }
    
    // 初始化 Redis
    redisClient, err := database.InitRedis(cfg.Redis)
    if err != nil {
        log.Fatalf("Failed to connect redis: %v", err)
    }
    
    // 初始化 Repository
    userRepo := repository.NewUserRepository(db)
    
    // 初始化 Cache
    redisCache := cache.NewRedisCache(redisClient)
    
    // 初始化 Service
    userService := user.NewUserService(userRepo, redisCache, zapLogger)
    
    // 初始化 Handler
    userHandler := userv1.NewUserHandler(userService)
    
    // 设置 Gin
    router := gin.New()
    router.Use(gin.Recovery())
    router.Use(middleware.RequestLogger(zapLogger))
    router.Use(middleware.CORS())
    
    // 健康检查
    router.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{"status": "ok"})
    })
    
    // 注册路由
    apiV1 := router.Group("/api/v1")
    userHandler.RegisterRoutes(apiV1)
    
    // 启动服务器
    srv := &http.Server{
        Addr:    ":8082",
        Handler: router,
    }
    
    go func() {
        log.Printf("User service starting on :8082")
        if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            log.Fatalf("Failed to start server: %v", err)
        }
    }()
    
    // 优雅关闭
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit
    
    log.Println("Shutting down server...")
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    if err := srv.Shutdown(ctx); err != nil {
        log.Fatalf("Server forced to shutdown: %v", err)
    }
    
    log.Println("Server exited")
}
```

---

## API 接口规范

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/users | 创建用户 |
| GET | /api/v1/users/:id | 获取用户 |
| PUT | /api/v1/users/:id | 更新用户 |
| DELETE | /api/v1/users/:id | 删除用户 |
| GET | /api/v1/users/:id/behaviors | 获取行为历史 |
| POST | /api/v1/users/:id/behaviors | 记录行为 |
| GET | /api/v1/users/:id/profile | 获取用户画像 |

---

## 测试用例

```go
// tests/unit/user_service_test.go
func TestUserService_CreateUser(t *testing.T) {
    // Mock 依赖
    mockRepo := mocks.NewMockUserRepo()
    mockCache := mocks.NewMockCache()
    
    service := user.NewUserService(mockRepo, mockCache, nil)
    
    req := &user.CreateUserRequest{
        Name:  "Test User",
        Email: "test@example.com",
        Age:   25,
    }
    
    result, err := service.CreateUser(context.Background(), req)
    
    assert.NoError(t, err)
    assert.Equal(t, "Test User", result.Name)
}
```

---

## 注意事项

1. **缓存策略**: 用户信息使用 30 分钟缓存，更新时删除缓存
2. **行为记录**: 异步写入，避免阻塞主流程
3. **用户画像**: 可考虑定时计算并缓存，而非实时计算
4. **安全性**: 敏感信息（如密码）需要加密存储

## 输出要求

请输出完整的可运行代码，包含：
1. 所有 Go 文件
2. 详细的中文注释
3. 单元测试
4. API 文档

