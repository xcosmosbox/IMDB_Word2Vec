// Package llm 提供大语言模型客户端的接口定义和通用工具
//
// 本包定义了与 LLM 交互的标准接口，支持多种后端实现（OpenAI、本地模型等）。
// 所有实现都需要遵循 interfaces.LLMClient 接口规范。
//
// 使用方式：
//
//	client := llm.NewOpenAIClient(config)
//	response, err := client.Chat(ctx, messages, llm.WithMaxTokens(256))
package llm

import (
	"context"
	"errors"
	"sync"
	"time"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// 错误定义
// =============================================================================

var (
	// ErrEmptyPrompt 空提示词错误
	ErrEmptyPrompt = errors.New("prompt cannot be empty")

	// ErrEmptyMessages 空消息列表错误
	ErrEmptyMessages = errors.New("messages cannot be empty")

	// ErrAPIKeyRequired API Key 必需错误
	ErrAPIKeyRequired = errors.New("API key is required")

	// ErrRequestTimeout 请求超时错误
	ErrRequestTimeout = errors.New("request timeout")

	// ErrRateLimitExceeded 速率限制错误
	ErrRateLimitExceeded = errors.New("rate limit exceeded")

	// ErrModelNotAvailable 模型不可用错误
	ErrModelNotAvailable = errors.New("model not available")

	// ErrInvalidResponse 无效响应错误
	ErrInvalidResponse = errors.New("invalid response from LLM")

	// ErrContextCanceled 上下文取消错误
	ErrContextCanceled = errors.New("context canceled")
)

// =============================================================================
// 配置结构
// =============================================================================

// ClientConfig LLM 客户端通用配置
type ClientConfig struct {
	// APIKey API 密钥
	APIKey string `json:"api_key" yaml:"api_key"`

	// BaseURL API 基础 URL
	BaseURL string `json:"base_url" yaml:"base_url"`

	// Model 默认模型名称
	Model string `json:"model" yaml:"model"`

	// EmbeddingModel 嵌入模型名称
	EmbeddingModel string `json:"embedding_model" yaml:"embedding_model"`

	// Timeout 请求超时时间
	Timeout time.Duration `json:"timeout" yaml:"timeout"`

	// MaxRetries 最大重试次数
	MaxRetries int `json:"max_retries" yaml:"max_retries"`

	// RetryDelay 重试延迟
	RetryDelay time.Duration `json:"retry_delay" yaml:"retry_delay"`

	// MaxConcurrency 最大并发数
	MaxConcurrency int `json:"max_concurrency" yaml:"max_concurrency"`
}

// DefaultClientConfig 返回默认配置
func DefaultClientConfig() ClientConfig {
	return ClientConfig{
		BaseURL:        "https://api.openai.com/v1",
		Model:          "gpt-3.5-turbo",
		EmbeddingModel: "text-embedding-ada-002",
		Timeout:        30 * time.Second,
		MaxRetries:     3,
		RetryDelay:     time.Second,
		MaxConcurrency: 10,
	}
}

// Validate 验证配置
func (c *ClientConfig) Validate() error {
	if c.APIKey == "" {
		return ErrAPIKeyRequired
	}
	return nil
}

// =============================================================================
// 选项应用辅助函数
// =============================================================================

// ApplyOptions 应用选项到 LLMOptions
func ApplyOptions(opts ...interfaces.LLMOption) *interfaces.LLMOptions {
	options := DefaultLLMOptions()
	for _, opt := range opts {
		opt(options)
	}
	return options
}

// DefaultLLMOptions 返回默认 LLM 选项
func DefaultLLMOptions() *interfaces.LLMOptions {
	return &interfaces.LLMOptions{
		MaxTokens:   256,
		Temperature: 0.7,
		Model:       "gpt-3.5-turbo",
	}
}

// =============================================================================
// 缓存包装器
// =============================================================================

// CachedClient 带缓存的 LLM 客户端包装器
// 用于缓存 LLM 响应，减少重复调用成本
type CachedClient struct {
	client interfaces.LLMClient
	cache  map[string]cachedResponse
	mu     sync.RWMutex
	ttl    time.Duration
}

// cachedResponse 缓存的响应
type cachedResponse struct {
	response  string
	embedding []float32
	expiresAt time.Time
}

// NewCachedClient 创建带缓存的客户端
func NewCachedClient(client interfaces.LLMClient, ttl time.Duration) *CachedClient {
	c := &CachedClient{
		client: client,
		cache:  make(map[string]cachedResponse),
		ttl:    ttl,
	}

	// 启动缓存清理协程
	go c.cleanupLoop()

	return c
}

// Complete 文本补全（带缓存）
func (c *CachedClient) Complete(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
	cacheKey := "complete:" + prompt

	// 尝试从缓存获取
	if response, found := c.getFromCache(cacheKey); found {
		return response.response, nil
	}

	// 调用底层客户端
	response, err := c.client.Complete(ctx, prompt, opts...)
	if err != nil {
		return "", err
	}

	// 存入缓存
	c.setCache(cacheKey, cachedResponse{
		response:  response,
		expiresAt: time.Now().Add(c.ttl),
	})

	return response, nil
}

// Embed 文本嵌入（带缓存）
func (c *CachedClient) Embed(ctx context.Context, text string) ([]float32, error) {
	cacheKey := "embed:" + text

	// 尝试从缓存获取
	if response, found := c.getFromCache(cacheKey); found {
		return response.embedding, nil
	}

	// 调用底层客户端
	embedding, err := c.client.Embed(ctx, text)
	if err != nil {
		return nil, err
	}

	// 存入缓存
	c.setCache(cacheKey, cachedResponse{
		embedding: embedding,
		expiresAt: time.Now().Add(c.ttl),
	})

	return embedding, nil
}

// Chat 对话式交互（不缓存，因为可能涉及上下文）
func (c *CachedClient) Chat(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
	return c.client.Chat(ctx, messages, opts...)
}

// getFromCache 从缓存获取
func (c *CachedClient) getFromCache(key string) (cachedResponse, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	item, exists := c.cache[key]
	if !exists {
		return cachedResponse{}, false
	}

	if time.Now().After(item.expiresAt) {
		return cachedResponse{}, false
	}

	return item, true
}

// setCache 设置缓存
func (c *CachedClient) setCache(key string, value cachedResponse) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache[key] = value
}

// cleanupLoop 定期清理过期缓存
func (c *CachedClient) cleanupLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		c.mu.Lock()
		now := time.Now()
		for key, item := range c.cache {
			if now.After(item.expiresAt) {
				delete(c.cache, key)
			}
		}
		c.mu.Unlock()
	}
}

// =============================================================================
// 重试包装器
// =============================================================================

// RetryClient 带重试机制的 LLM 客户端包装器
type RetryClient struct {
	client     interfaces.LLMClient
	maxRetries int
	retryDelay time.Duration
}

// NewRetryClient 创建带重试的客户端
func NewRetryClient(client interfaces.LLMClient, maxRetries int, retryDelay time.Duration) *RetryClient {
	return &RetryClient{
		client:     client,
		maxRetries: maxRetries,
		retryDelay: retryDelay,
	}
}

// Complete 文本补全（带重试）
func (r *RetryClient) Complete(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
	var lastErr error
	for i := 0; i <= r.maxRetries; i++ {
		response, err := r.client.Complete(ctx, prompt, opts...)
		if err == nil {
			return response, nil
		}

		lastErr = err

		// 检查是否应该重试
		if !r.shouldRetry(err) {
			return "", err
		}

		// 检查上下文是否已取消
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(r.retryDelay * time.Duration(i+1)):
			// 指数退避
		}
	}

	return "", lastErr
}

// Embed 文本嵌入（带重试）
func (r *RetryClient) Embed(ctx context.Context, text string) ([]float32, error) {
	var lastErr error
	for i := 0; i <= r.maxRetries; i++ {
		embedding, err := r.client.Embed(ctx, text)
		if err == nil {
			return embedding, nil
		}

		lastErr = err

		if !r.shouldRetry(err) {
			return nil, err
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(r.retryDelay * time.Duration(i+1)):
		}
	}

	return nil, lastErr
}

// Chat 对话式交互（带重试）
func (r *RetryClient) Chat(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
	var lastErr error
	for i := 0; i <= r.maxRetries; i++ {
		response, err := r.client.Chat(ctx, messages, opts...)
		if err == nil {
			return response, nil
		}

		lastErr = err

		if !r.shouldRetry(err) {
			return "", err
		}

		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(r.retryDelay * time.Duration(i+1)):
		}
	}

	return "", lastErr
}

// shouldRetry 判断是否应该重试
func (r *RetryClient) shouldRetry(err error) bool {
	// 某些错误不应该重试
	if errors.Is(err, ErrEmptyPrompt) ||
		errors.Is(err, ErrEmptyMessages) ||
		errors.Is(err, ErrAPIKeyRequired) ||
		errors.Is(err, context.Canceled) {
		return false
	}
	return true
}

// =============================================================================
// 并发限制器
// =============================================================================

// RateLimitedClient 带速率限制的 LLM 客户端包装器
type RateLimitedClient struct {
	client  interfaces.LLMClient
	limiter chan struct{}
}

// NewRateLimitedClient 创建带速率限制的客户端
func NewRateLimitedClient(client interfaces.LLMClient, maxConcurrency int) *RateLimitedClient {
	return &RateLimitedClient{
		client:  client,
		limiter: make(chan struct{}, maxConcurrency),
	}
}

// Complete 文本补全（带并发限制）
func (r *RateLimitedClient) Complete(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
	select {
	case r.limiter <- struct{}{}:
		defer func() { <-r.limiter }()
		return r.client.Complete(ctx, prompt, opts...)
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

// Embed 文本嵌入（带并发限制）
func (r *RateLimitedClient) Embed(ctx context.Context, text string) ([]float32, error) {
	select {
	case r.limiter <- struct{}{}:
		defer func() { <-r.limiter }()
		return r.client.Embed(ctx, text)
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// Chat 对话式交互（带并发限制）
func (r *RateLimitedClient) Chat(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
	select {
	case r.limiter <- struct{}{}:
		defer func() { <-r.limiter }()
		return r.client.Chat(ctx, messages, opts...)
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

// =============================================================================
// 辅助函数
// =============================================================================

// BuildSystemMessage 构建系统消息
func BuildSystemMessage(content string) interfaces.Message {
	return interfaces.Message{
		Role:    "system",
		Content: content,
	}
}

// BuildUserMessage 构建用户消息
func BuildUserMessage(content string) interfaces.Message {
	return interfaces.Message{
		Role:    "user",
		Content: content,
	}
}

// BuildAssistantMessage 构建助手消息
func BuildAssistantMessage(content string) interfaces.Message {
	return interfaces.Message{
		Role:    "assistant",
		Content: content,
	}
}

// BuildMessages 快捷构建消息列表
func BuildMessages(system, user string) []interfaces.Message {
	messages := make([]interfaces.Message, 0, 2)
	if system != "" {
		messages = append(messages, BuildSystemMessage(system))
	}
	if user != "" {
		messages = append(messages, BuildUserMessage(user))
	}
	return messages
}

