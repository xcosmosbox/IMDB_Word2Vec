// Package llm 提供大语言模型客户端实现
//
// 本文件实现了本地模型客户端，支持：
// - Ollama 本地推理
// - 自定义 HTTP 推理服务
// - ONNX Runtime 推理（预留接口）
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// Ollama 客户端
// =============================================================================

// OllamaClient Ollama 本地模型客户端
// 实现 interfaces.LLMClient 接口
type OllamaClient struct {
	// baseURL Ollama 服务地址
	baseURL string

	// model 默认模型名称
	model string

	// embeddingModel 嵌入模型
	embeddingModel string

	// httpClient HTTP 客户端
	httpClient *http.Client
}

// OllamaConfig Ollama 客户端配置
type OllamaConfig struct {
	// BaseURL Ollama 服务地址，默认为 http://localhost:11434
	BaseURL string `json:"base_url" yaml:"base_url"`

	// Model 默认使用的模型，如 llama2, qwen
	Model string `json:"model" yaml:"model"`

	// EmbeddingModel 嵌入模型
	EmbeddingModel string `json:"embedding_model" yaml:"embedding_model"`

	// Timeout 请求超时时间
	Timeout time.Duration `json:"timeout" yaml:"timeout"`
}

// DefaultOllamaConfig 返回默认 Ollama 配置
func DefaultOllamaConfig() OllamaConfig {
	return OllamaConfig{
		BaseURL:        "http://localhost:11434",
		Model:          "llama2",
		EmbeddingModel: "nomic-embed-text",
		Timeout:        120 * time.Second, // 本地模型通常需要更长时间
	}
}

// NewOllamaClient 创建 Ollama 客户端
//
// 参数：
//   - cfg: Ollama 配置
//
// 返回：
//   - *OllamaClient: 客户端实例
func NewOllamaClient(cfg OllamaConfig) *OllamaClient {
	// 设置默认值
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://localhost:11434"
	}
	if cfg.Model == "" {
		cfg.Model = "llama2"
	}
	if cfg.EmbeddingModel == "" {
		cfg.EmbeddingModel = "nomic-embed-text"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 120 * time.Second
	}

	return &OllamaClient{
		baseURL:        cfg.BaseURL,
		model:          cfg.Model,
		embeddingModel: cfg.EmbeddingModel,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
	}
}

// Complete 文本补全
func (c *OllamaClient) Complete(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
	if prompt == "" {
		return "", ErrEmptyPrompt
	}

	options := ApplyOptions(opts...)

	// 使用配置的模型或选项中指定的模型
	model := c.model
	if options.Model != "" {
		model = options.Model
	}

	// 构建 Ollama 请求
	reqBody := ollamaGenerateRequest{
		Model:  model,
		Prompt: prompt,
		Stream: false,
		Options: ollamaOptions{
			Temperature: options.Temperature,
			NumPredict:  options.MaxTokens,
		},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return "", ErrContextCanceled
		}
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Ollama API error (status %d)", resp.StatusCode)
	}

	var result ollamaGenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Response, nil
}

// Chat 对话式交互
func (c *OllamaClient) Chat(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
	if len(messages) == 0 {
		return "", ErrEmptyMessages
	}

	options := ApplyOptions(opts...)

	model := c.model
	if options.Model != "" {
		model = options.Model
	}

	// 构建 Ollama chat 请求
	reqBody := ollamaChatRequest{
		Model:    model,
		Messages: convertToOllamaMessages(messages),
		Stream:   false,
		Options: ollamaOptions{
			Temperature: options.Temperature,
			NumPredict:  options.MaxTokens,
		},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return "", ErrContextCanceled
		}
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Ollama API error (status %d)", resp.StatusCode)
	}

	var result ollamaChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Message.Content, nil
}

// Embed 文本嵌入
func (c *OllamaClient) Embed(ctx context.Context, text string) ([]float32, error) {
	if text == "" {
		return nil, ErrEmptyPrompt
	}

	// 构建 Ollama 嵌入请求
	reqBody := ollamaEmbedRequest{
		Model:  c.embeddingModel,
		Prompt: text,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ErrContextCanceled
		}
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Ollama API error (status %d)", resp.StatusCode)
	}

	var result ollamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// 转换 float64 到 float32
	embedding := make([]float32, len(result.Embedding))
	for i, v := range result.Embedding {
		embedding[i] = float32(v)
	}

	return embedding, nil
}

// =============================================================================
// Ollama 请求/响应结构
// =============================================================================

type ollamaGenerateRequest struct {
	Model   string        `json:"model"`
	Prompt  string        `json:"prompt"`
	Stream  bool          `json:"stream"`
	Options ollamaOptions `json:"options,omitempty"`
}

type ollamaOptions struct {
	Temperature float64 `json:"temperature,omitempty"`
	NumPredict  int     `json:"num_predict,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
}

type ollamaGenerateResponse struct {
	Model     string `json:"model"`
	Response  string `json:"response"`
	Done      bool   `json:"done"`
	CreatedAt string `json:"created_at"`
}

type ollamaChatRequest struct {
	Model    string           `json:"model"`
	Messages []ollamaMessage  `json:"messages"`
	Stream   bool             `json:"stream"`
	Options  ollamaOptions    `json:"options,omitempty"`
}

type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaChatResponse struct {
	Model   string        `json:"model"`
	Message ollamaMessage `json:"message"`
	Done    bool          `json:"done"`
}

type ollamaEmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type ollamaEmbedResponse struct {
	Embedding []float64 `json:"embedding"`
}

// convertToOllamaMessages 转换消息格式
func convertToOllamaMessages(messages []interfaces.Message) []ollamaMessage {
	result := make([]ollamaMessage, len(messages))
	for i, m := range messages {
		result[i] = ollamaMessage{
			Role:    m.Role,
			Content: m.Content,
		}
	}
	return result
}

// =============================================================================
// 自定义 HTTP 推理客户端
// =============================================================================

// HTTPInferenceClient 自定义 HTTP 推理客户端
// 用于连接自部署的推理服务
type HTTPInferenceClient struct {
	// baseURL 推理服务地址
	baseURL string

	// apiKey 可选的 API 密钥
	apiKey string

	// chatEndpoint 对话端点
	chatEndpoint string

	// embedEndpoint 嵌入端点
	embedEndpoint string

	// httpClient HTTP 客户端
	httpClient *http.Client
}

// HTTPInferenceConfig 自定义推理客户端配置
type HTTPInferenceConfig struct {
	// BaseURL 推理服务地址
	BaseURL string `json:"base_url" yaml:"base_url"`

	// APIKey 可选的 API 密钥
	APIKey string `json:"api_key" yaml:"api_key"`

	// ChatEndpoint 对话端点路径
	ChatEndpoint string `json:"chat_endpoint" yaml:"chat_endpoint"`

	// EmbedEndpoint 嵌入端点路径
	EmbedEndpoint string `json:"embed_endpoint" yaml:"embed_endpoint"`

	// Timeout 请求超时时间
	Timeout time.Duration `json:"timeout" yaml:"timeout"`
}

// NewHTTPInferenceClient 创建自定义推理客户端
func NewHTTPInferenceClient(cfg HTTPInferenceConfig) (*HTTPInferenceClient, error) {
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("base URL is required")
	}

	if cfg.ChatEndpoint == "" {
		cfg.ChatEndpoint = "/v1/chat"
	}
	if cfg.EmbedEndpoint == "" {
		cfg.EmbedEndpoint = "/v1/embed"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}

	return &HTTPInferenceClient{
		baseURL:       cfg.BaseURL,
		apiKey:        cfg.APIKey,
		chatEndpoint:  cfg.ChatEndpoint,
		embedEndpoint: cfg.EmbedEndpoint,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
	}, nil
}

// Complete 文本补全
func (c *HTTPInferenceClient) Complete(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
	messages := []interfaces.Message{
		{Role: "user", Content: prompt},
	}
	return c.Chat(ctx, messages, opts...)
}

// Chat 对话式交互
func (c *HTTPInferenceClient) Chat(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
	if len(messages) == 0 {
		return "", ErrEmptyMessages
	}

	options := ApplyOptions(opts...)

	// 构建通用请求格式
	reqBody := map[string]interface{}{
		"messages":    messages,
		"max_tokens":  options.MaxTokens,
		"temperature": options.Temperature,
	}
	if options.Model != "" {
		reqBody["model"] = options.Model
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+c.chatEndpoint, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return "", ErrContextCanceled
		}
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("inference API error (status %d)", resp.StatusCode)
	}

	// 尝试解析通用响应格式
	var result struct {
		Response string `json:"response"`
		Content  string `json:"content"`
		Text     string `json:"text"`
		Choices  []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			Text string `json:"text"`
		} `json:"choices"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	// 尝试多种响应格式
	if result.Response != "" {
		return result.Response, nil
	}
	if result.Content != "" {
		return result.Content, nil
	}
	if result.Text != "" {
		return result.Text, nil
	}
	if len(result.Choices) > 0 {
		if result.Choices[0].Message.Content != "" {
			return result.Choices[0].Message.Content, nil
		}
		if result.Choices[0].Text != "" {
			return result.Choices[0].Text, nil
		}
	}

	return "", ErrInvalidResponse
}

// Embed 文本嵌入
func (c *HTTPInferenceClient) Embed(ctx context.Context, text string) ([]float32, error) {
	if text == "" {
		return nil, ErrEmptyPrompt
	}

	reqBody := map[string]interface{}{
		"input": text,
		"text":  text, // 兼容不同格式
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+c.embedEndpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ErrContextCanceled
		}
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("inference API error (status %d)", resp.StatusCode)
	}

	// 尝试解析多种响应格式
	var result struct {
		Embedding []float32 `json:"embedding"`
		Vector    []float32 `json:"vector"`
		Data      []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Embedding) > 0 {
		return result.Embedding, nil
	}
	if len(result.Vector) > 0 {
		return result.Vector, nil
	}
	if len(result.Data) > 0 && len(result.Data[0].Embedding) > 0 {
		return result.Data[0].Embedding, nil
	}

	return nil, ErrInvalidResponse
}

// =============================================================================
// Mock 客户端（用于测试）
// =============================================================================

// MockClient Mock LLM 客户端
// 用于单元测试和开发调试
type MockClient struct {
	// CompleteFunc 自定义 Complete 函数
	CompleteFunc func(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error)

	// EmbedFunc 自定义 Embed 函数
	EmbedFunc func(ctx context.Context, text string) ([]float32, error)

	// ChatFunc 自定义 Chat 函数
	ChatFunc func(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error)

	// DefaultResponse 默认响应
	DefaultResponse string

	// DefaultEmbedding 默认嵌入向量
	DefaultEmbedding []float32

	// ShouldError 是否返回错误
	ShouldError bool

	// Error 要返回的错误
	Error error
}

// NewMockClient 创建 Mock 客户端
func NewMockClient() *MockClient {
	return &MockClient{
		DefaultResponse:  "这是一个模拟响应",
		DefaultEmbedding: make([]float32, 768), // 默认 768 维向量
	}
}

// Complete Mock 文本补全
func (m *MockClient) Complete(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
	if m.ShouldError {
		if m.Error != nil {
			return "", m.Error
		}
		return "", fmt.Errorf("mock error")
	}

	if m.CompleteFunc != nil {
		return m.CompleteFunc(ctx, prompt, opts...)
	}

	return m.DefaultResponse, nil
}

// Embed Mock 文本嵌入
func (m *MockClient) Embed(ctx context.Context, text string) ([]float32, error) {
	if m.ShouldError {
		if m.Error != nil {
			return nil, m.Error
		}
		return nil, fmt.Errorf("mock error")
	}

	if m.EmbedFunc != nil {
		return m.EmbedFunc(ctx, text)
	}

	return m.DefaultEmbedding, nil
}

// Chat Mock 对话
func (m *MockClient) Chat(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
	if m.ShouldError {
		if m.Error != nil {
			return "", m.Error
		}
		return "", fmt.Errorf("mock error")
	}

	if m.ChatFunc != nil {
		return m.ChatFunc(ctx, messages, opts...)
	}

	return m.DefaultResponse, nil
}

// SetResponse 设置默认响应
func (m *MockClient) SetResponse(response string) {
	m.DefaultResponse = response
}

// SetEmbedding 设置默认嵌入向量
func (m *MockClient) SetEmbedding(embedding []float32) {
	m.DefaultEmbedding = embedding
}

// SetError 设置错误
func (m *MockClient) SetError(err error) {
	m.ShouldError = true
	m.Error = err
}

// ClearError 清除错误
func (m *MockClient) ClearError() {
	m.ShouldError = false
	m.Error = nil
}

