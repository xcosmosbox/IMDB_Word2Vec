// Package config 提供统一的配置管理
package config

import (
	"fmt"
	"strings"
	"time"

	"github.com/spf13/viper"
)

// Config 全局配置结构
type Config struct {
	Server    ServerConfig    `mapstructure:"server"`
	Database  DatabaseConfig  `mapstructure:"database"`
	Redis     RedisConfig     `mapstructure:"redis"`
	Milvus    MilvusConfig    `mapstructure:"milvus"`
	Kafka     KafkaConfig     `mapstructure:"kafka"`
	Inference InferenceConfig `mapstructure:"inference"`
	LLM       LLMConfig       `mapstructure:"llm"`
	Log       LogConfig       `mapstructure:"log"`
	Metrics   MetricsConfig   `mapstructure:"metrics"`
	Tracing   TracingConfig   `mapstructure:"tracing"`
}

// ServerConfig 服务器配置
type ServerConfig struct {
	Name            string        `mapstructure:"name"`
	HTTPPort        int           `mapstructure:"http_port"`
	GRPCPort        int           `mapstructure:"grpc_port"`
	MetricsPort     int           `mapstructure:"metrics_port"`
	ReadTimeout     time.Duration `mapstructure:"read_timeout"`
	WriteTimeout    time.Duration `mapstructure:"write_timeout"`
	ShutdownTimeout time.Duration `mapstructure:"shutdown_timeout"`
	Mode            string        `mapstructure:"mode"` // debug, release, test
}

// DatabaseConfig PostgreSQL 配置
type DatabaseConfig struct {
	Host            string        `mapstructure:"host"`
	Port            int           `mapstructure:"port"`
	User            string        `mapstructure:"user"`
	Password        string        `mapstructure:"password"`
	DBName          string        `mapstructure:"dbname"`
	SSLMode         string        `mapstructure:"sslmode"`
	MaxOpenConns    int           `mapstructure:"max_open_conns"`
	MaxIdleConns    int           `mapstructure:"max_idle_conns"`
	ConnMaxLifetime time.Duration `mapstructure:"conn_max_lifetime"`
	ConnMaxIdleTime time.Duration `mapstructure:"conn_max_idle_time"`
}

// DSN 返回数据库连接字符串
func (c *DatabaseConfig) DSN() string {
	return fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		c.Host, c.Port, c.User, c.Password, c.DBName, c.SSLMode,
	)
}

// RedisConfig Redis 配置
type RedisConfig struct {
	Addrs        []string      `mapstructure:"addrs"`
	Password     string        `mapstructure:"password"`
	DB           int           `mapstructure:"db"`
	PoolSize     int           `mapstructure:"pool_size"`
	MinIdleConns int           `mapstructure:"min_idle_conns"`
	DialTimeout  time.Duration `mapstructure:"dial_timeout"`
	ReadTimeout  time.Duration `mapstructure:"read_timeout"`
	WriteTimeout time.Duration `mapstructure:"write_timeout"`
	ClusterMode  bool          `mapstructure:"cluster_mode"`
}

// MilvusConfig Milvus 向量数据库配置
type MilvusConfig struct {
	Address    string `mapstructure:"address"`
	Port       int    `mapstructure:"port"`
	User       string `mapstructure:"user"`
	Password   string `mapstructure:"password"`
	Database   string `mapstructure:"database"`
	Collection string `mapstructure:"collection"`
}

// KafkaConfig Kafka 配置
type KafkaConfig struct {
	Brokers       []string `mapstructure:"brokers"`
	ConsumerGroup string   `mapstructure:"consumer_group"`
	Topics        struct {
		UserEvents        string `mapstructure:"user_events"`
		RecommendationLog string `mapstructure:"recommendation_log"`
		ItemUpdates       string `mapstructure:"item_updates"`
	} `mapstructure:"topics"`
}

// InferenceConfig 推理服务配置
type InferenceConfig struct {
	TritonURL       string        `mapstructure:"triton_url"`
	ModelName       string        `mapstructure:"model_name"`
	ModelVersion    string        `mapstructure:"model_version"`
	Timeout         time.Duration `mapstructure:"timeout"`
	MaxBatchSize    int           `mapstructure:"max_batch_size"`
	EnableFallback  bool          `mapstructure:"enable_fallback"`
	FallbackTimeout time.Duration `mapstructure:"fallback_timeout"`
}

// LLMConfig 大模型配置
type LLMConfig struct {
	Provider    string        `mapstructure:"provider"` // openai, anthropic, local
	APIKey      string        `mapstructure:"api_key"`
	APIBase     string        `mapstructure:"api_base"`
	Model       string        `mapstructure:"model"`
	MaxTokens   int           `mapstructure:"max_tokens"`
	Temperature float64       `mapstructure:"temperature"`
	Timeout     time.Duration `mapstructure:"timeout"`
	MaxRetries  int           `mapstructure:"max_retries"`
}

// LogConfig 日志配置
type LogConfig struct {
	Level      string `mapstructure:"level"`       // debug, info, warn, error
	Format     string `mapstructure:"format"`      // json, console
	Output     string `mapstructure:"output"`      // stdout, file
	Filename   string `mapstructure:"filename"`    // 日志文件路径
	MaxSize    int    `mapstructure:"max_size"`    // MB
	MaxBackups int    `mapstructure:"max_backups"` // 保留数量
	MaxAge     int    `mapstructure:"max_age"`     // 保留天数
	Compress   bool   `mapstructure:"compress"`    // 是否压缩
}

// MetricsConfig 监控指标配置
type MetricsConfig struct {
	Enabled bool   `mapstructure:"enabled"`
	Path    string `mapstructure:"path"`
}

// TracingConfig 链路追踪配置
type TracingConfig struct {
	Enabled     bool    `mapstructure:"enabled"`
	ServiceName string  `mapstructure:"service_name"`
	Endpoint    string  `mapstructure:"endpoint"`
	SampleRate  float64 `mapstructure:"sample_rate"`
}

// GlobalConfig 全局配置实例
var GlobalConfig *Config

// Load 加载配置文件
func Load(configPath string) (*Config, error) {
	v := viper.New()

	// 设置配置文件
	v.SetConfigFile(configPath)
	v.SetConfigType("yaml")

	// 环境变量支持
	v.AutomaticEnv()
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))

	// 设置默认值
	setDefaults(v)

	// 读取配置
	if err := v.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	// 解析配置
	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	GlobalConfig = &cfg
	return &cfg, nil
}

// setDefaults 设置默认配置值
func setDefaults(v *viper.Viper) {
	// Server defaults
	v.SetDefault("server.name", "recommend-service")
	v.SetDefault("server.http_port", 8080)
	v.SetDefault("server.grpc_port", 9090)
	v.SetDefault("server.metrics_port", 9091)
	v.SetDefault("server.read_timeout", "30s")
	v.SetDefault("server.write_timeout", "30s")
	v.SetDefault("server.shutdown_timeout", "10s")
	v.SetDefault("server.mode", "release")

	// Database defaults
	v.SetDefault("database.host", "localhost")
	v.SetDefault("database.port", 5432)
	v.SetDefault("database.sslmode", "disable")
	v.SetDefault("database.max_open_conns", 100)
	v.SetDefault("database.max_idle_conns", 10)
	v.SetDefault("database.conn_max_lifetime", "1h")
	v.SetDefault("database.conn_max_idle_time", "10m")

	// Redis defaults
	v.SetDefault("redis.addrs", []string{"localhost:6379"})
	v.SetDefault("redis.db", 0)
	v.SetDefault("redis.pool_size", 100)
	v.SetDefault("redis.min_idle_conns", 10)
	v.SetDefault("redis.dial_timeout", "5s")
	v.SetDefault("redis.read_timeout", "3s")
	v.SetDefault("redis.write_timeout", "3s")
	v.SetDefault("redis.cluster_mode", false)

	// Milvus defaults
	v.SetDefault("milvus.address", "localhost")
	v.SetDefault("milvus.port", 19530)
	v.SetDefault("milvus.database", "default")

	// Inference defaults
	v.SetDefault("inference.timeout", "100ms")
	v.SetDefault("inference.max_batch_size", 64)
	v.SetDefault("inference.enable_fallback", true)
	v.SetDefault("inference.fallback_timeout", "50ms")

	// LLM defaults
	v.SetDefault("llm.provider", "openai")
	v.SetDefault("llm.max_tokens", 1024)
	v.SetDefault("llm.temperature", 0.7)
	v.SetDefault("llm.timeout", "30s")
	v.SetDefault("llm.max_retries", 3)

	// Log defaults
	v.SetDefault("log.level", "info")
	v.SetDefault("log.format", "json")
	v.SetDefault("log.output", "stdout")
	v.SetDefault("log.max_size", 100)
	v.SetDefault("log.max_backups", 3)
	v.SetDefault("log.max_age", 7)
	v.SetDefault("log.compress", true)

	// Metrics defaults
	v.SetDefault("metrics.enabled", true)
	v.SetDefault("metrics.path", "/metrics")

	// Tracing defaults
	v.SetDefault("tracing.enabled", true)
	v.SetDefault("tracing.sample_rate", 0.1)
}

// MustLoad 加载配置，失败则 panic
func MustLoad(configPath string) *Config {
	cfg, err := Load(configPath)
	if err != nil {
		panic(err)
	}
	return cfg
}

