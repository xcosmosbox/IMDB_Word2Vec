package middleware

import (
	"context"
	"time"

	"github.com/gin-gonic/gin"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
	"recommend-system/pkg/logger"
	"go.uber.org/zap"
)

// TracingConfig 追踪配置
type TracingConfig struct {
	ServiceName string
	Enabled     bool
}

// Tracing 链路追踪中间件
func Tracing(serviceName string) gin.HandlerFunc {
	tracer := otel.Tracer(serviceName)

	return func(c *gin.Context) {
		// 从请求头提取 context
		ctx := otel.GetTextMapPropagator().Extract(
			c.Request.Context(),
			propagation.HeaderCarrier(c.Request.Header),
		)

		// 创建 span
		spanName := c.Request.Method + " " + c.FullPath()
		ctx, span := tracer.Start(ctx, spanName,
			trace.WithSpanKind(trace.SpanKindServer),
		)
		defer span.End()

		// 设置请求属性
		span.SetAttributes(
			attribute.String("http.method", c.Request.Method),
			attribute.String("http.url", c.Request.URL.String()),
			attribute.String("http.host", c.Request.Host),
			attribute.String("http.user_agent", c.Request.UserAgent()),
			attribute.String("net.peer.ip", c.ClientIP()),
		)

		// 获取 trace_id 和 span_id
		spanCtx := span.SpanContext()
		traceID := spanCtx.TraceID().String()
		spanID := spanCtx.SpanID().String()

		// 设置到上下文
		c.Set("trace_id", traceID)
		c.Set("span_id", spanID)
		ctx = context.WithValue(ctx, "trace_id", traceID)
		ctx = context.WithValue(ctx, "span_id", spanID)

		// 设置响应头
		c.Header("X-Trace-ID", traceID)

		// 更新请求上下文
		c.Request = c.Request.WithContext(ctx)

		// 记录开始时间
		startTime := time.Now()

		// 处理请求
		c.Next()

		// 记录响应属性
		latency := time.Since(startTime)
		statusCode := c.Writer.Status()

		span.SetAttributes(
			attribute.Int("http.status_code", statusCode),
			attribute.Int64("http.response_size", int64(c.Writer.Size())),
			attribute.Int64("http.latency_ms", latency.Milliseconds()),
		)

		// 根据状态码设置 span 状态
		if statusCode >= 400 {
			span.SetAttributes(attribute.Bool("error", true))
		}
	}
}

// SpanFromContext 从上下文获取 span
func SpanFromContext(ctx context.Context) trace.Span {
	return trace.SpanFromContext(ctx)
}

// StartSpan 创建新的 span
func StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	tracer := otel.Tracer("recommend-system")
	return tracer.Start(ctx, name, opts...)
}

// RequestLogger 请求日志中间件
func RequestLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 开始时间
		startTime := time.Now()

		// 请求路径
		path := c.Request.URL.Path
		query := c.Request.URL.RawQuery

		// 处理请求
		c.Next()

		// 计算延迟
		latency := time.Since(startTime)

		// 获取追踪信息
		traceID, _ := c.Get("trace_id")
		userID := GetUserID(c)

		// 记录日志
		fields := []zap.Field{
			zap.String("method", c.Request.Method),
			zap.String("path", path),
			zap.String("query", query),
			zap.Int("status", c.Writer.Status()),
			zap.Duration("latency", latency),
			zap.String("client_ip", c.ClientIP()),
			zap.String("user_agent", c.Request.UserAgent()),
		}

		if traceID != nil {
			fields = append(fields, zap.String("trace_id", traceID.(string)))
		}
		if userID != "" {
			fields = append(fields, zap.String("user_id", userID))
		}

		// 根据状态码选择日志级别
		status := c.Writer.Status()
		if status >= 500 {
			// 添加错误信息
			if len(c.Errors) > 0 {
				fields = append(fields, zap.String("error", c.Errors.String()))
			}
			logger.Error("request completed with server error", fields...)
		} else if status >= 400 {
			logger.Warn("request completed with client error", fields...)
		} else {
			logger.Info("request completed", fields...)
		}
	}
}

// Recovery 恢复中间件
func Recovery() gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			if err := recover(); err != nil {
				// 获取追踪信息
				traceID, _ := c.Get("trace_id")
				userID := GetUserID(c)

				fields := []zap.Field{
					zap.Any("panic", err),
					zap.String("method", c.Request.Method),
					zap.String("path", c.Request.URL.Path),
					zap.String("client_ip", c.ClientIP()),
				}

				if traceID != nil {
					fields = append(fields, zap.String("trace_id", traceID.(string)))
				}
				if userID != "" {
					fields = append(fields, zap.String("user_id", userID))
				}

				logger.Error("panic recovered", fields...)

				c.AbortWithStatusJSON(500, gin.H{
					"code":    500,
					"message": "internal server error",
				})
			}
		}()

		c.Next()
	}
}

// CORS 跨域中间件
func CORS() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Authorization, X-API-Key, X-Trace-ID")
		c.Header("Access-Control-Expose-Headers", "X-Trace-ID")
		c.Header("Access-Control-Max-Age", "86400")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}

