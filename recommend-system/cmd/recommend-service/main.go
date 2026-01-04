// Package main 推荐服务入口
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
	"recommend-system/internal/cache"
	"recommend-system/internal/inference"
	"recommend-system/internal/middleware"
	"recommend-system/internal/repository"
	recommendSvc "recommend-system/internal/service/recommend"
	recommendAPI "recommend-system/api/recommend/v1"
	"recommend-system/pkg/config"
	"recommend-system/pkg/database"
	"recommend-system/pkg/logger"
	"go.uber.org/zap"
)

var (
	configPath = flag.String("config", "configs/config.yaml", "配置文件路径")
	version    = "1.0.0"
	buildTime  = "unknown"
)

func main() {
	flag.Parse()

	// 加载配置
	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Printf("Failed to load config: %v\n", err)
		os.Exit(1)
	}

	// 初始化日志
	if err := logger.Init(&logger.Config{
		Level:   cfg.Log.Level,
		Format:  cfg.Log.Format,
		Output:  cfg.Log.Output,
	}); err != nil {
		fmt.Printf("Failed to init logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()

	logger.Info("Starting recommend service",
		zap.String("version", version),
		zap.String("build_time", buildTime),
	)

	// 初始化数据库连接
	postgresDB, err := database.NewPostgresDB(&database.PostgresConfig{
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
		logger.Fatal("Failed to connect to PostgreSQL", zap.Error(err))
	}
	defer postgresDB.Close()

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

	// 初始化 Milvus
	milvusClient, err := database.NewMilvusClient(&database.MilvusConfig{
		Address:    cfg.Milvus.Address,
		Port:       cfg.Milvus.Port,
		User:       cfg.Milvus.User,
		Password:   cfg.Milvus.Password,
		Database:   cfg.Milvus.Database,
		Collection: cfg.Milvus.Collection,
	})
	if err != nil {
		logger.Warn("Failed to connect to Milvus, vector search will be disabled", zap.Error(err))
	} else {
		defer milvusClient.Close()
	}

	// 创建缓存
	multiLevelCache := cache.NewMultiLevelCache(redisClient, "rec:", 10000, 5*time.Minute)

	// 创建推理客户端
	var inferClient inference.Client
	if cfg.Inference.TritonURL != "" {
		inferClient = inference.NewTritonClient(&inference.TritonConfig{
			BaseURL:      cfg.Inference.TritonURL,
			ModelName:    cfg.Inference.ModelName,
			ModelVersion: cfg.Inference.ModelVersion,
			Timeout:      cfg.Inference.Timeout,
		})
	} else {
		// 使用模拟客户端
		inferClient = inference.NewMockClient(10 * time.Millisecond)
		logger.Warn("Using mock inference client")
	}

	// 创建仓储
	userRepo := repository.NewUserRepository(postgresDB)
	itemRepo := repository.NewItemRepository(postgresDB)
	recommendRepo := repository.NewRecommendRepository(postgresDB)

	// 创建服务
	recommendService := recommendSvc.NewService(
		userRepo,
		itemRepo,
		recommendRepo,
		inferClient,
		milvusClient,
		multiLevelCache,
		&recommendSvc.Config{
			DefaultSize:         20,
			MaxSize:             100,
			CandidateSize:       500,
			ExposureFilterHours: 24,
			ColdStartThreshold:  10,
			ModelVersion:        "v1.0.0",
			MilvusCollection:    cfg.Milvus.Collection,
		},
	)

	// 创建 HTTP 服务器
	if cfg.Server.Mode == "release" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()

	// 中间件
	router.Use(middleware.Recovery())
	router.Use(middleware.CORS())
	router.Use(middleware.Tracing(cfg.Server.Name))
	router.Use(middleware.RequestLogger())

	// 限流
	rateLimiter := middleware.NewRateLimiter(middleware.DefaultRateLimitConfig())
	router.Use(middleware.RateLimit(rateLimiter))

	// 认证 (API 组)
	authConfig := middleware.DefaultAuthConfig()
	apiGroup := router.Group("/api/v1")
	apiGroup.Use(middleware.Auth(authConfig))

	// 注册路由
	recommendHandler := recommendAPI.NewHandler(recommendService)
	recommendHandler.RegisterRoutes(apiGroup)

	// 健康检查
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "healthy",
			"service": cfg.Server.Name,
			"version": version,
		})
	})

	// 就绪检查
	router.GET("/ready", func(c *gin.Context) {
		// 检查依赖
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		if err := postgresDB.Ping(ctx); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status": "not ready",
				"error":  "database connection failed",
			})
			return
		}

		if err := redisClient.Ping(ctx); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status": "not ready",
				"error":  "redis connection failed",
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{"status": "ready"})
	})

	// 启动服务器
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Server.HTTPPort),
		Handler:      router,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
	}

	// 优雅关闭
	go func() {
		logger.Info("HTTP server starting",
			zap.Int("port", cfg.Server.HTTPPort),
		)

		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("HTTP server failed", zap.Error(err))
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

	if err := server.Shutdown(ctx); err != nil {
		logger.Error("Server forced to shutdown", zap.Error(err))
	}

	logger.Info("Server exited")
}

