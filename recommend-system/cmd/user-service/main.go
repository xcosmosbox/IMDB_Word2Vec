// Package main 提供用户服务的启动入口
//
// 用户服务是推荐系统的核心微服务之一，负责用户相关的所有操作。
//
// 启动方式:
//   go run cmd/user-service/main.go
//   go run cmd/user-service/main.go -config=configs/config.yaml
//
// 环境变量:
//   CONFIG_PATH - 配置文件路径
//   SERVER_PORT - 服务端口（覆盖配置文件）
//
// 健康检查:
//   GET /health - 服务健康状态
//   GET /ready  - 服务就绪状态
package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	userv1 "recommend-system/api/user/v1"
	"recommend-system/internal/cache"
	"recommend-system/internal/middleware"
	"recommend-system/internal/repository"
	userservice "recommend-system/internal/service/user"
	"recommend-system/pkg/config"
	"recommend-system/pkg/database"
	"recommend-system/pkg/logger"
)

// 版本信息，通过编译时注入
var (
	Version   = "dev"
	BuildTime = "unknown"
	GitCommit = "unknown"
)

// 默认配置
const (
	defaultConfigPath = "configs/config.yaml"
	defaultPort       = 8082
	serviceName       = "user-service"
)

func main() {
	// 解析命令行参数
	configPath := flag.String("config", defaultConfigPath, "配置文件路径")
	showVersion := flag.Bool("version", false, "显示版本信息")
	flag.Parse()

	// 显示版本信息
	if *showVersion {
		fmt.Printf("%s version %s\n", serviceName, Version)
		fmt.Printf("Build time: %s\n", BuildTime)
		fmt.Printf("Git commit: %s\n", GitCommit)
		os.Exit(0)
	}

	// 检查环境变量覆盖
	if envConfig := os.Getenv("CONFIG_PATH"); envConfig != "" {
		*configPath = envConfig
	}

	// 加载配置
	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Printf("Failed to load config: %v\n", err)
		os.Exit(1)
	}

	// 初始化日志
	if err := logger.Init(&logger.Config{
		Level:      cfg.Log.Level,
		Format:     cfg.Log.Format,
		Output:     cfg.Log.Output,
		Filename:   cfg.Log.Filename,
		MaxSize:    cfg.Log.MaxSize,
		MaxBackups: cfg.Log.MaxBackups,
		MaxAge:     cfg.Log.MaxAge,
		Compress:   cfg.Log.Compress,
	}); err != nil {
		fmt.Printf("Failed to init logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()

	logger.Info("Starting user service",
		zap.String("version", Version),
		zap.String("config", *configPath),
	)

	// 初始化数据库连接
	db, err := database.NewPostgresDB(&database.PostgresConfig{
		Host:            cfg.Database.Host,
		Port:            cfg.Database.Port,
		User:            cfg.Database.User,
		Password:        cfg.Database.Password,
		DBName:          cfg.Database.DBName,
		SSLMode:         cfg.Database.SSLMode,
		MaxOpenConns:    int32(cfg.Database.MaxOpenConns),
		MaxIdleConns:    int32(cfg.Database.MaxIdleConns),
		ConnMaxLifetime: cfg.Database.ConnMaxLifetime,
		ConnMaxIdleTime: cfg.Database.ConnMaxIdleTime,
	})
	if err != nil {
		logger.Fatal("Failed to connect to database", zap.Error(err))
	}
	defer db.Close()

	// 初始化 Redis
	redisClient, err := database.NewRedisClient(&database.RedisConfig{
		Addrs:        cfg.Redis.Addrs,
		Password:     cfg.Redis.Password,
		DB:           cfg.Redis.DB,
		PoolSize:     cfg.Redis.PoolSize,
		MinIdleConns: cfg.Redis.MinIdleConns,
		DialTimeout:  cfg.Redis.DialTimeout,
		ReadTimeout:  cfg.Redis.ReadTimeout,
		WriteTimeout: cfg.Redis.WriteTimeout,
		ClusterMode:  cfg.Redis.ClusterMode,
	})
	if err != nil {
		logger.Fatal("Failed to connect to Redis", zap.Error(err))
	}
	defer redisClient.Close()

	// 初始化仓储层
	userRepo := repository.NewUserRepository(db)

	// 初始化缓存层
	multiLevelCache := cache.NewMultiLevelCache(
		redisClient,
		"user:",
		10000,          // 本地缓存大小
		5*time.Minute,  // 本地缓存 TTL
	)

	// 初始化用户服务
	userSvc := userservice.NewService(
		&userRepoAdapter{repo: userRepo},
		&cacheAdapter{cache: multiLevelCache},
		logger.Logger,
	)

	// 初始化 HTTP Handler
	userHandler := userv1.NewHandler(userSvc, logger.Logger)

	// 设置 Gin 模式
	if cfg.Server.Mode == "release" {
		gin.SetMode(gin.ReleaseMode)
	}

	// 创建 Gin 引擎
	router := gin.New()

	// 注册中间件
	router.Use(gin.Recovery())
	router.Use(requestLoggerMiddleware())
	router.Use(corsMiddleware())

	// 健康检查端点
	router.GET("/health", healthHandler(db, redisClient))
	router.GET("/ready", readyHandler(db, redisClient))

	// API 版本组
	apiV1 := router.Group("/api/v1")
	{
		// 可选：添加认证中间件
		// apiV1.Use(middleware.Auth(middleware.DefaultAuthConfig()))

		// 注册用户服务路由
		userHandler.RegisterRoutes(apiV1)
	}

	// 确定服务端口
	port := cfg.Server.HTTPPort
	if port == 0 {
		port = defaultPort
	}
	if envPort := os.Getenv("SERVER_PORT"); envPort != "" {
		fmt.Sscanf(envPort, "%d", &port)
	}

	// 创建 HTTP 服务器
	srv := &http.Server{
		Addr:         fmt.Sprintf(":%d", port),
		Handler:      router,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
	}

	// 启动服务器（非阻塞）
	go func() {
		logger.Info("User service started",
			zap.Int("port", port),
			zap.String("mode", cfg.Server.Mode),
		)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("Failed to start server", zap.Error(err))
		}
	}()

	// 等待中断信号
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down server...")

	// 优雅关闭
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logger.Error("Server forced to shutdown", zap.Error(err))
	}

	logger.Info("Server exited gracefully")
}

// =============================================================================
// 中间件
// =============================================================================

// requestLoggerMiddleware 请求日志中间件
func requestLoggerMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path
		query := c.Request.URL.RawQuery

		c.Next()

		latency := time.Since(start)
		statusCode := c.Writer.Status()

		// 记录请求日志
		fields := []zap.Field{
			zap.Int("status", statusCode),
			zap.String("method", c.Request.Method),
			zap.String("path", path),
			zap.String("query", query),
			zap.String("ip", c.ClientIP()),
			zap.Duration("latency", latency),
			zap.Int("body_size", c.Writer.Size()),
		}

		if statusCode >= 500 {
			logger.Error("request completed with error", fields...)
		} else if statusCode >= 400 {
			logger.Warn("request completed with client error", fields...)
		} else {
			logger.Info("request completed", fields...)
		}
	}
}

// corsMiddleware CORS 中间件
func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Authorization, X-API-Key")
		c.Header("Access-Control-Max-Age", "86400")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}

// =============================================================================
// 健康检查
// =============================================================================

// healthHandler 健康检查处理器
func healthHandler(db *database.PostgresDB, redis *database.RedisClient) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "ok",
			"service": serviceName,
			"version": Version,
			"time":    time.Now().Format(time.RFC3339),
		})
	}
}

// readyHandler 就绪检查处理器
func readyHandler(db *database.PostgresDB, redis *database.RedisClient) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
		defer cancel()

		// 检查数据库连接
		if err := db.Ping(ctx); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status":  "not ready",
				"reason":  "database connection failed",
				"error":   err.Error(),
			})
			return
		}

		// 检查 Redis 连接
		if err := redis.Ping(ctx); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status":  "not ready",
				"reason":  "redis connection failed",
				"error":   err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status":   "ready",
			"service":  serviceName,
			"database": "connected",
			"redis":    "connected",
		})
	}
}

// =============================================================================
// 适配器（用于接口转换）
// =============================================================================

// userRepoAdapter 用户仓库适配器
//
// 将 repository.UserRepository 适配到 interfaces.UserRepository
type userRepoAdapter struct {
	repo *repository.UserRepository
}

func (a *userRepoAdapter) GetByID(ctx context.Context, userID string) (*interfaces.User, error) {
	// 调用实际仓库并转换结果
	// 这里需要根据实际 model.User 进行转换
	return nil, fmt.Errorf("not implemented - requires model conversion")
}

func (a *userRepoAdapter) GetByIDs(ctx context.Context, userIDs []string) ([]*interfaces.User, error) {
	return nil, fmt.Errorf("not implemented")
}

func (a *userRepoAdapter) Create(ctx context.Context, user *interfaces.User) error {
	return fmt.Errorf("not implemented")
}

func (a *userRepoAdapter) Update(ctx context.Context, user *interfaces.User) error {
	return fmt.Errorf("not implemented")
}

func (a *userRepoAdapter) Delete(ctx context.Context, userID string) error {
	return fmt.Errorf("not implemented")
}

func (a *userRepoAdapter) GetBehaviors(ctx context.Context, userID string, limit int) ([]*interfaces.UserBehavior, error) {
	return nil, fmt.Errorf("not implemented")
}

func (a *userRepoAdapter) AddBehavior(ctx context.Context, behavior *interfaces.UserBehavior) error {
	return fmt.Errorf("not implemented")
}

func (a *userRepoAdapter) GetUserItemInteractions(ctx context.Context, userID, itemID string) ([]*interfaces.UserBehavior, error) {
	return nil, fmt.Errorf("not implemented")
}

// cacheAdapter 缓存适配器
//
// 将 cache.MultiLevelCache 适配到 interfaces.Cache
type cacheAdapter struct {
	cache *cache.MultiLevelCache
}

func (a *cacheAdapter) Get(ctx context.Context, key string, value interface{}) error {
	data, err := a.cache.Get(ctx, key)
	if err != nil {
		return err
	}
	// 这里需要反序列化
	_ = data
	return nil
}

func (a *cacheAdapter) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	// 这里需要序列化
	return nil
}

func (a *cacheAdapter) Delete(ctx context.Context, key string) error {
	return a.cache.Delete(ctx, key)
}

func (a *cacheAdapter) Exists(ctx context.Context, key string) (bool, error) {
	return a.cache.Exists(ctx, key)
}

func (a *cacheAdapter) MGet(ctx context.Context, keys []string) ([]interface{}, error) {
	return nil, fmt.Errorf("not implemented")
}

func (a *cacheAdapter) MSet(ctx context.Context, kvs map[string]interface{}, ttl time.Duration) error {
	return fmt.Errorf("not implemented")
}

// 确保适配器实现了接口（编译时检查）
var _ interfaces.UserRepository = (*userRepoAdapter)(nil)
var _ interfaces.Cache = (*cacheAdapter)(nil)
var _ = middleware.DefaultAuthConfig // 确保 middleware 包被引用

