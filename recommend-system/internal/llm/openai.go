// Package llm 提供大语言模型客户端实现
//
// 本文件实现了 OpenAI 兼容的 LLM 客户端。
// 支持 OpenAI API 以及兼容接口（如 Azure OpenAI、国内代理等）。
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// OpenAI 客户端
// =============================================================================

// OpenAIClient OpenAI API 客户端
// 实现 interfaces.LLMClient 接口
type OpenAIClient struct {
	// apiKey API 密钥
	apiKey string

	// baseURL API 基础 URL
	baseURL string

	// model 默认模型
	model string

	// embeddingModel 嵌入模型
	embeddingModel string

	// httpClient HTTP 客户端
	httpClient *http.Client
}

// OpenAIConfig OpenAI 客户端配置
type OpenAIConfig struct {
	// APIKey API 密钥（必需）
	APIKey string `json:"api_key" yaml:"api_key"`

	// BaseURL API 基础 URL，默认为 OpenAI 官方地址
	BaseURL string `json:"base_url" yaml:"base_url"`

	// Model 默认使用的模型
	Model string `json:"model" yaml:"model"`

	// EmbeddingModel 嵌入模型
	EmbeddingModel string `json:"embedding_model" yaml:"embedding_model"`

	// Timeout 请求超时时间
	Timeout time.Duration `json:"timeout" yaml:"timeout"`
}

// DefaultOpenAIConfig 返回默认 OpenAI 配置
func DefaultOpenAIConfig() OpenAIConfig {
	return OpenAIConfig{
		BaseURL:        "https://api.openai.com/v1",
		Model:          "gpt-3.5-turbo",
		EmbeddingModel: "text-embedding-ada-002",
		Timeout:        30 * time.Second,
	}
}

// NewOpenAIClient 创建 OpenAI 客户端
//
// 参数：
//   - cfg: OpenAI 配置
//
// 返回：
//   - *OpenAIClient: 客户端实例
//   - error: 如果配置无效则返回错误
func NewOpenAIClient(cfg OpenAIConfig) (*OpenAIClient, error) {
	if cfg.APIKey == "" {
		return nil, ErrAPIKeyRequired
	}

	// 设置默认值
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	if cfg.Model == "" {
		cfg.Model = "gpt-3.5-turbo"
	}
	if cfg.EmbeddingModel == "" {
		cfg.EmbeddingModel = "text-embedding-ada-002"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 30 * time.Second
	}

	return &OpenAIClient{
		apiKey:         cfg.APIKey,
		baseURL:        cfg.BaseURL,
		model:          cfg.Model,
		embeddingModel: cfg.EmbeddingModel,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
	}, nil
}

// Complete 文本补全
//
// 将提示词发送给 LLM 进行补全。内部实现为单轮对话。
//
// 参数：
//   - ctx: 上下文
//   - prompt: 提示词
//   - opts: 可选参数
//
// 返回：
//   - string: 补全结果
//   - error: 如果失败则返回错误
func (c *OpenAIClient) Complete(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
	if prompt == "" {
		return "", ErrEmptyPrompt
	}

	// 将补全请求转换为对话请求
	messages := []interfaces.Message{
		{Role: "user", Content: prompt},
	}

	return c.Chat(ctx, messages, opts...)
}

// Chat 对话式交互
//
// 发送多轮对话消息给 LLM。
//
// 参数：
//   - ctx: 上下文
//   - messages: 消息列表
//   - opts: 可选参数
//
// 返回：
//   - string: LLM 响应内容
//   - error: 如果失败则返回错误
func (c *OpenAIClient) Chat(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
	if len(messages) == 0 {
		return "", ErrEmptyMessages
	}

	// 应用选项
	options := ApplyOptions(opts...)

	// 使用配置的模型或选项中指定的模型
	model := c.model
	if options.Model != "" {
		model = options.Model
	}

	// 构建请求体
	reqBody := chatRequest{
		Model:       model,
		Messages:    convertMessages(messages),
		MaxTokens:   options.MaxTokens,
		Temperature: options.Temperature,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	// 创建 HTTP 请求
	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	// 发送请求
	resp, err := c.httpClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return "", ErrContextCanceled
		}
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// 检查状态码
	if resp.StatusCode != http.StatusOK {
		return "", c.handleErrorResponse(resp)
	}

	// 解析响应
	var result chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Choices) == 0 {
		return "", ErrInvalidResponse
	}

	return result.Choices[0].Message.Content, nil
}

// Embed 文本嵌入
//
// 将文本转换为向量表示。
//
// 参数：
//   - ctx: 上下文
//   - text: 待嵌入的文本
//
// 返回：
//   - []float32: 嵌入向量
//   - error: 如果失败则返回错误
func (c *OpenAIClient) Embed(ctx context.Context, text string) ([]float32, error) {
	if text == "" {
		return nil, ErrEmptyPrompt
	}

	// 构建请求体
	reqBody := embeddingRequest{
		Model: c.embeddingModel,
		Input: text,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// 创建 HTTP 请求
	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	// 发送请求
	resp, err := c.httpClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ErrContextCanceled
		}
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// 检查状态码
	if resp.StatusCode != http.StatusOK {
		return nil, c.handleErrorResponse(resp)
	}

	// 解析响应
	var result embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Data) == 0 {
		return nil, ErrInvalidResponse
	}

	return result.Data[0].Embedding, nil
}

// handleErrorResponse 处理错误响应
func (c *OpenAIClient) handleErrorResponse(resp *http.Response) error {
	bodyBytes, _ := io.ReadAll(resp.Body)

	// 尝试解析错误响应
	var errResp errorResponse
	if err := json.Unmarshal(bodyBytes, &errResp); err == nil && errResp.Error.Message != "" {
		// 根据状态码返回特定错误
		switch resp.StatusCode {
		case http.StatusTooManyRequests:
			return fmt.Errorf("%w: %s", ErrRateLimitExceeded, errResp.Error.Message)
		case http.StatusUnauthorized:
			return fmt.Errorf("%w: %s", ErrAPIKeyRequired, errResp.Error.Message)
		case http.StatusServiceUnavailable:
			return fmt.Errorf("%w: %s", ErrModelNotAvailable, errResp.Error.Message)
		default:
			return fmt.Errorf("OpenAI API error (status %d): %s", resp.StatusCode, errResp.Error.Message)
		}
	}

	return fmt.Errorf("OpenAI API error (status %d): %s", resp.StatusCode, string(bodyBytes))
}

// =============================================================================
// 请求/响应结构
// =============================================================================

// chatRequest 对话请求
type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	TopP        float64       `json:"top_p,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
}

// chatMessage 对话消息
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// chatResponse 对话响应
type chatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// embeddingRequest 嵌入请求
type embeddingRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

// embeddingResponse 嵌入响应
type embeddingResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// errorResponse 错误响应
type errorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param"`
		Code    string `json:"code"`
	} `json:"error"`
}

// =============================================================================
// 辅助函数
// =============================================================================

// convertMessages 转换消息格式
func convertMessages(messages []interfaces.Message) []chatMessage {
	result := make([]chatMessage, len(messages))
	for i, m := range messages {
		result[i] = chatMessage{
			Role:    m.Role,
			Content: m.Content,
		}
	}
	return result
}

// =============================================================================
// Azure OpenAI 客户端
// =============================================================================

// AzureOpenAIClient Azure OpenAI 客户端
type AzureOpenAIClient struct {
	*OpenAIClient

	// deploymentID 部署 ID
	deploymentID string

	// apiVersion API 版本
	apiVersion string
}

// AzureOpenAIConfig Azure OpenAI 配置
type AzureOpenAIConfig struct {
	// APIKey Azure API 密钥
	APIKey string `json:"api_key" yaml:"api_key"`

	// Endpoint Azure 端点 URL
	Endpoint string `json:"endpoint" yaml:"endpoint"`

	// DeploymentID 模型部署 ID
	DeploymentID string `json:"deployment_id" yaml:"deployment_id"`

	// APIVersion API 版本
	APIVersion string `json:"api_version" yaml:"api_version"`

	// Timeout 请求超时时间
	Timeout time.Duration `json:"timeout" yaml:"timeout"`
}

// NewAzureOpenAIClient 创建 Azure OpenAI 客户端
func NewAzureOpenAIClient(cfg AzureOpenAIConfig) (*AzureOpenAIClient, error) {
	if cfg.APIKey == "" {
		return nil, ErrAPIKeyRequired
	}

	if cfg.Endpoint == "" {
		return nil, fmt.Errorf("endpoint is required for Azure OpenAI")
	}

	if cfg.DeploymentID == "" {
		return nil, fmt.Errorf("deployment ID is required for Azure OpenAI")
	}

	if cfg.APIVersion == "" {
		cfg.APIVersion = "2023-05-15"
	}

	if cfg.Timeout == 0 {
		cfg.Timeout = 30 * time.Second
	}

	baseClient := &OpenAIClient{
		apiKey:  cfg.APIKey,
		baseURL: cfg.Endpoint,
		model:   cfg.DeploymentID,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
	}

	return &AzureOpenAIClient{
		OpenAIClient: baseClient,
		deploymentID: cfg.DeploymentID,
		apiVersion:   cfg.APIVersion,
	}, nil
}

// Chat Azure OpenAI 对话
func (c *AzureOpenAIClient) Chat(ctx context.Context, messages []interfaces.Message, opts ...interfaces.LLMOption) (string, error) {
	if len(messages) == 0 {
		return "", ErrEmptyMessages
	}

	options := ApplyOptions(opts...)

	// 构建请求体
	reqBody := chatRequest{
		Messages:    convertMessages(messages),
		MaxTokens:   options.MaxTokens,
		Temperature: options.Temperature,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	// Azure OpenAI 的 URL 格式不同
	url := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s",
		c.baseURL, c.deploymentID, c.apiVersion)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return "", ErrContextCanceled
		}
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", c.handleErrorResponse(resp)
	}

	var result chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Choices) == 0 {
		return "", ErrInvalidResponse
	}

	return result.Choices[0].Message.Content, nil
}

// Embed Azure OpenAI 嵌入
func (c *AzureOpenAIClient) Embed(ctx context.Context, text string) ([]float32, error) {
	if text == "" {
		return nil, ErrEmptyPrompt
	}

	reqBody := embeddingRequest{
		Input: text,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Azure OpenAI 嵌入 URL
	url := fmt.Sprintf("%s/openai/deployments/%s/embeddings?api-version=%s",
		c.baseURL, c.deploymentID, c.apiVersion)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ErrContextCanceled
		}
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.handleErrorResponse(resp)
	}

	var result embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Data) == 0 {
		return nil, ErrInvalidResponse
	}

	return result.Data[0].Embedding, nil
}

