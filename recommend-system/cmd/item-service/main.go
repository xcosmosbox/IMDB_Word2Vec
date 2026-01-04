// Package main 物品服务入口
//
// 物品服务负责管理推荐系统中的所有可推荐物品。
//
// 启动命令：
//
//	go run cmd/item-service/main.go -config configs/config.yaml
//
// 环境变量：
//
//	CONFIG_PATH: 配置文件路径
//	PORT: 服务端口（默认 8083）
//	LOG_LEVEL: 日志级别（debug, info, warn, error）
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

	itemAPI "recommend-system/api/item/v1"
	"recommend-system/internal/cache"
	"recommend-system/internal/interfaces"
	"recommend-system/internal/middleware"
	"recommend-system/internal/repository"
	"recommend-system/internal/service/item"
	"recommend-system/pkg/config"
	"recommend-system/pkg/database"
	"recommend-system/pkg/logger"

	"go.uber.org/zap"
)

// =============================================================================
// 版本信息
// =============================================================================

var (
	// 编译时注入的版本信息
	version   = "1.0.0"
	buildTime = "unknown"
	gitCommit = "unknown"

	// 命令行参数
	configPath  = flag.String("config", "configs/config.yaml", "配置文件路径")
	showVersion = flag.Bool("version", false, "显示版本信息")
)

// =============================================================================
// 主函数
// =============================================================================

func main() {
	flag.Parse()

	// 显示版本信息
	if *showVersion {
		fmt.Printf("Item Service %s\n", version)
		fmt.Printf("Build Time: %s\n", buildTime)
		fmt.Printf("Git Commit: %s\n", gitCommit)
		os.Exit(0)
	}

	// 加载配置
	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Printf("Failed to load config: %v\n", err)
		os.Exit(1)
	}

	// 初始化日志
	if err := logger.Init(&logger.Config{
		Level:  cfg.Log.Level,
		Format: cfg.Log.Format,
		Output: cfg.Log.Output,
	}); err != nil {
		fmt.Printf("Failed to init logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()

	logger.Info("Starting Item Service",
		zap.String("version", version),
		zap.String("build_time", buildTime),
		zap.String("git_commit", gitCommit),
		zap.String("config_path", *configPath),
	)

	// 初始化数据库连接
	postgresDB, err := initPostgres(cfg)
	if err != nil {
		logger.Fatal("Failed to connect to PostgreSQL", zap.Error(err))
	}
	defer postgresDB.Close()
	logger.Info("PostgreSQL connected")

	// 初始化 Redis
	redisClient, err := initRedis(cfg)
	if err != nil {
		logger.Fatal("Failed to connect to Redis", zap.Error(err))
	}
	defer redisClient.Close()
	logger.Info("Redis connected")

	// 初始化 Milvus（可选，失败不影响服务启动）
	milvusClient, err := initMilvus(cfg)
	if err != nil {
		logger.Warn("Failed to connect to Milvus, vector search will be disabled",
			zap.Error(err),
		)
	} else {
		defer milvusClient.Close()
		logger.Info("Milvus connected")
	}

	// 创建缓存
	multiLevelCache := cache.NewMultiLevelCache(
		redisClient,
		"item:",
		10000,               // 本地缓存大小
		5*time.Minute,       // 本地缓存 TTL
	)

	// 创建缓存适配器（适配 interfaces.Cache 接口）
	cacheAdapter := item.NewCacheAdapter(&cacheWrapper{cache: multiLevelCache})

	// 创建 Repository
	itemRepo := repository.NewItemRepository(postgresDB)

	// 创建 Service
	itemService := item.NewService(
		&itemRepoAdapter{repo: itemRepo},
		milvusClient,
		cacheAdapter,
		&item.Config{
			CacheTTL:         time.Hour,
			MilvusCollection: cfg.Milvus.Collection,
			DefaultPageSize:  20,
			MaxPageSize:      100,
			EmbeddingDim:     256,
		},
	)

	// 创建 Handler
	itemHandler := itemAPI.NewHandler(itemService)

	// 设置 Gin 模式
	if cfg.Server.Mode == "release" {
		gin.SetMode(gin.ReleaseMode)
	}

	// 创建路由
	router := gin.New()

	// 全局中间件
	router.Use(middleware.Recovery())
	router.Use(middleware.CORS())
	router.Use(middleware.Tracing("item-service"))
	router.Use(middleware.RequestLogger())

	// 限流
	rateLimiter := middleware.NewRateLimiter(middleware.DefaultRateLimitConfig())
	router.Use(middleware.RateLimit(rateLimiter))

	// 健康检查（不需要认证）
	router.GET("/health", healthCheck(postgresDB, redisClient))
	router.GET("/ready", readyCheck(postgresDB, redisClient))

	// API 路由组
	apiV1 := router.Group("/api/v1")

	// 认证中间件（可选，根据配置决定是否启用）
	if cfg.Server.AuthEnabled {
		authConfig := middleware.DefaultAuthConfig()
		apiV1.Use(middleware.Auth(authConfig))
	}

	// 注册物品服务路由
	itemHandler.RegisterRoutes(apiV1)

	// 获取服务端口
	port := cfg.Server.HTTPPort
	if port == 0 {
		port = 8083 // 默认端口
	}

	// 创建 HTTP 服务器
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", port),
		Handler:      router,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
	}

	// 启动服务器
	go func() {
		logger.Info("HTTP server starting",
			zap.Int("port", port),
		)

		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("HTTP server failed", zap.Error(err))
		}
	}()

	// 等待中断信号
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down Item Service...")

	// 优雅关闭
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		logger.Error("Server forced to shutdown", zap.Error(err))
	}

	logger.Info("Item Service exited")
}

// =============================================================================
// 初始化函数
// =============================================================================

// initPostgres 初始化 PostgreSQL 连接
func initPostgres(cfg *config.Config) (*database.PostgresDB, error) {
	return database.NewPostgresDB(&database.PostgresConfig{
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
}

// initRedis 初始化 Redis 连接
func initRedis(cfg *config.Config) (*database.RedisClient, error) {
	return database.NewRedisClient(&database.RedisConfig{
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
}

// initMilvus 初始化 Milvus 连接
func initMilvus(cfg *config.Config) (*database.MilvusClient, error) {
	return database.NewMilvusClient(&database.MilvusConfig{
		Address:    cfg.Milvus.Address,
		Port:       cfg.Milvus.Port,
		User:       cfg.Milvus.User,
		Password:   cfg.Milvus.Password,
		Database:   cfg.Milvus.Database,
		Collection: cfg.Milvus.Collection,
	})
}

// =============================================================================
// 健康检查处理器
// =============================================================================

// healthCheck 健康检查处理器
func healthCheck(db *database.PostgresDB, redis *database.RedisClient) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "healthy",
			"service": "item-service",
			"version": version,
			"time":    time.Now().Format(time.RFC3339),
		})
	}
}

// readyCheck 就绪检查处理器
func readyCheck(db *database.PostgresDB, redis *database.RedisClient) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		// 检查数据库连接
		if err := db.Ping(ctx); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status": "not ready",
				"error":  "database connection failed",
			})
			return
		}

		// 检查 Redis 连接
		if err := redis.Ping(ctx); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status": "not ready",
				"error":  "redis connection failed",
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status": "ready",
			"time":   time.Now().Format(time.RFC3339),
		})
	}
}

// =============================================================================
// 适配器类型
// =============================================================================

// cacheWrapper 缓存包装器，适配 interfaces.Cache 接口
type cacheWrapper struct {
	cache *cache.MultiLevelCache
}

// Get 获取缓存
func (w *cacheWrapper) Get(ctx context.Context, key string, value interface{}) error {
	data, err := w.cache.Get(ctx, key)
	if err != nil {
		return err
	}
	// 简化处理：假设 value 可以直接赋值
	// 实际应使用 json 反序列化
	_ = data
	return nil
}

// Set 设置缓存
func (w *cacheWrapper) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	// 简化处理：将 value 序列化为字节
	data := []byte(fmt.Sprintf("%v", value))
	return w.cache.Set(ctx, key, data, ttl)
}

// Delete 删除缓存
func (w *cacheWrapper) Delete(ctx context.Context, key string) error {
	return w.cache.Delete(ctx, key)
}

// Exists 检查缓存是否存在
func (w *cacheWrapper) Exists(ctx context.Context, key string) (bool, error) {
	return w.cache.Exists(ctx, key)
}

// MGet 批量获取（简化实现）
func (w *cacheWrapper) MGet(ctx context.Context, keys []string) ([]interface{}, error) {
	results := make([]interface{}, len(keys))
	for i, key := range keys {
		data, err := w.cache.Get(ctx, key)
		if err != nil {
			results[i] = nil
		} else {
			results[i] = data
		}
	}
	return results, nil
}

// MSet 批量设置（简化实现）
func (w *cacheWrapper) MSet(ctx context.Context, kvs map[string]interface{}, ttl time.Duration) error {
	for key, value := range kvs {
		data := []byte(fmt.Sprintf("%v", value))
		if err := w.cache.Set(ctx, key, data, ttl); err != nil {
			return err
		}
	}
	return nil
}

// itemRepoAdapter 物品仓储适配器，适配 interfaces.ItemRepository 接口
type itemRepoAdapter struct {
	repo *repository.ItemRepository
}

// GetByID 根据 ID 获取物品
func (a *itemRepoAdapter) GetByID(ctx context.Context, itemID string) (*interfaces.Item, error) {
	// 需要实现类型转换
	return nil, fmt.Errorf("not implemented")
}

// GetByIDs 批量获取物品
func (a *itemRepoAdapter) GetByIDs(ctx context.Context, itemIDs []string) ([]*interfaces.Item, error) {
	return nil, fmt.Errorf("not implemented")
}

// Create 创建物品
func (a *itemRepoAdapter) Create(ctx context.Context, item *interfaces.Item) error {
	return fmt.Errorf("not implemented")
}

// Update 更新物品
func (a *itemRepoAdapter) Update(ctx context.Context, item *interfaces.Item) error {
	return fmt.Errorf("not implemented")
}

// Delete 删除物品
func (a *itemRepoAdapter) Delete(ctx context.Context, itemID string) error {
	return fmt.Errorf("not implemented")
}

// List 列出物品
func (a *itemRepoAdapter) List(ctx context.Context, itemType, category string, page, pageSize int) ([]*interfaces.Item, int64, error) {
	return nil, 0, fmt.Errorf("not implemented")
}

// Search 搜索物品
func (a *itemRepoAdapter) Search(ctx context.Context, query string, limit int) ([]*interfaces.Item, error) {
	return nil, fmt.Errorf("not implemented")
}

// GetStats 获取物品统计
func (a *itemRepoAdapter) GetStats(ctx context.Context, itemID string) (*interfaces.ItemStats, error) {
	return nil, fmt.Errorf("not implemented")
}

// IncrementStats 增加统计
func (a *itemRepoAdapter) IncrementStats(ctx context.Context, itemID, action string) error {
	return fmt.Errorf("not implemented")
}

// GetPopularByCategories 获取热门物品
func (a *itemRepoAdapter) GetPopularByCategories(ctx context.Context, categories []string, limit int) ([]*interfaces.Item, error) {
	return nil, fmt.Errorf("not implemented")
}

