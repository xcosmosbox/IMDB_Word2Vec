// Package logger 提供统一的日志管理
package logger

import (
	"context"
	"os"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// Logger 全局日志实例
var Logger *zap.Logger

// SugaredLogger 语法糖日志实例
var SugaredLogger *zap.SugaredLogger

// Config 日志配置
type Config struct {
	Level      string // debug, info, warn, error
	Format     string // json, console
	Output     string // stdout, file
	Filename   string
	MaxSize    int  // MB
	MaxBackups int
	MaxAge     int // days
	Compress   bool
}

// Init 初始化日志
func Init(cfg *Config) error {
	level := parseLevel(cfg.Level)

	// 编码器配置
	encoderConfig := zapcore.EncoderConfig{
		TimeKey:        "timestamp",
		LevelKey:       "level",
		NameKey:        "logger",
		CallerKey:      "caller",
		FunctionKey:    zapcore.OmitKey,
		MessageKey:     "message",
		StacktraceKey:  "stacktrace",
		LineEnding:     zapcore.DefaultLineEnding,
		EncodeLevel:    zapcore.LowercaseLevelEncoder,
		EncodeTime:     zapcore.ISO8601TimeEncoder,
		EncodeDuration: zapcore.SecondsDurationEncoder,
		EncodeCaller:   zapcore.ShortCallerEncoder,
	}

	// 选择编码器
	var encoder zapcore.Encoder
	if cfg.Format == "console" {
		encoder = zapcore.NewConsoleEncoder(encoderConfig)
	} else {
		encoder = zapcore.NewJSONEncoder(encoderConfig)
	}

	// 选择输出
	var writeSyncer zapcore.WriteSyncer
	if cfg.Output == "file" && cfg.Filename != "" {
		file, err := os.OpenFile(cfg.Filename, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			return err
		}
		writeSyncer = zapcore.AddSync(file)
	} else {
		writeSyncer = zapcore.AddSync(os.Stdout)
	}

	// 创建 Core
	core := zapcore.NewCore(encoder, writeSyncer, level)

	// 创建 Logger
	Logger = zap.New(core,
		zap.AddCaller(),
		zap.AddCallerSkip(1),
		zap.AddStacktrace(zapcore.ErrorLevel),
	)

	SugaredLogger = Logger.Sugar()

	return nil
}

// parseLevel 解析日志级别
func parseLevel(level string) zapcore.Level {
	switch level {
	case "debug":
		return zapcore.DebugLevel
	case "info":
		return zapcore.InfoLevel
	case "warn":
		return zapcore.WarnLevel
	case "error":
		return zapcore.ErrorLevel
	default:
		return zapcore.InfoLevel
	}
}

// Sync 刷新日志缓冲区
func Sync() error {
	if Logger != nil {
		return Logger.Sync()
	}
	return nil
}

// WithContext 返回带有 context 信息的 logger
func WithContext(ctx context.Context) *zap.Logger {
	if Logger == nil {
		return zap.NewNop()
	}

	// 从 context 中提取追踪信息
	fields := extractContextFields(ctx)
	return Logger.With(fields...)
}

// extractContextFields 从 context 提取字段
func extractContextFields(ctx context.Context) []zap.Field {
	fields := make([]zap.Field, 0, 4)

	// 提取 trace_id
	if traceID := ctx.Value("trace_id"); traceID != nil {
		fields = append(fields, zap.String("trace_id", traceID.(string)))
	}

	// 提取 span_id
	if spanID := ctx.Value("span_id"); spanID != nil {
		fields = append(fields, zap.String("span_id", spanID.(string)))
	}

	// 提取 user_id
	if userID := ctx.Value("user_id"); userID != nil {
		fields = append(fields, zap.String("user_id", userID.(string)))
	}

	// 提取 request_id
	if requestID := ctx.Value("request_id"); requestID != nil {
		fields = append(fields, zap.String("request_id", requestID.(string)))
	}

	return fields
}

// Debug 输出 debug 级别日志
func Debug(msg string, fields ...zap.Field) {
	if Logger != nil {
		Logger.Debug(msg, fields...)
	}
}

// Info 输出 info 级别日志
func Info(msg string, fields ...zap.Field) {
	if Logger != nil {
		Logger.Info(msg, fields...)
	}
}

// Warn 输出 warn 级别日志
func Warn(msg string, fields ...zap.Field) {
	if Logger != nil {
		Logger.Warn(msg, fields...)
	}
}

// Error 输出 error 级别日志
func Error(msg string, fields ...zap.Field) {
	if Logger != nil {
		Logger.Error(msg, fields...)
	}
}

// Fatal 输出 fatal 级别日志并退出
func Fatal(msg string, fields ...zap.Field) {
	if Logger != nil {
		Logger.Fatal(msg, fields...)
	}
}

// With 返回带有额外字段的 logger
func With(fields ...zap.Field) *zap.Logger {
	if Logger != nil {
		return Logger.With(fields...)
	}
	return zap.NewNop()
}

// RequestLogger 请求日志结构
type RequestLogger struct {
	StartTime  time.Time
	Method     string
	Path       string
	ClientIP   string
	UserAgent  string
	RequestID  string
	TraceID    string
	UserID     string
	StatusCode int
	Latency    time.Duration
	BodySize   int
	Error      string
}

// Log 输出请求日志
func (r *RequestLogger) Log() {
	fields := []zap.Field{
		zap.String("method", r.Method),
		zap.String("path", r.Path),
		zap.String("client_ip", r.ClientIP),
		zap.String("user_agent", r.UserAgent),
		zap.String("request_id", r.RequestID),
		zap.Int("status_code", r.StatusCode),
		zap.Duration("latency", r.Latency),
		zap.Int("body_size", r.BodySize),
	}

	if r.TraceID != "" {
		fields = append(fields, zap.String("trace_id", r.TraceID))
	}
	if r.UserID != "" {
		fields = append(fields, zap.String("user_id", r.UserID))
	}

	if r.StatusCode >= 500 {
		if r.Error != "" {
			fields = append(fields, zap.String("error", r.Error))
		}
		Error("request completed with error", fields...)
	} else if r.StatusCode >= 400 {
		Warn("request completed with client error", fields...)
	} else {
		Info("request completed", fields...)
	}
}

