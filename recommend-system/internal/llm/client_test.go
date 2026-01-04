package llm

import (
	"context"
	"testing"
	"time"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// ApplyOptions 测试
// =============================================================================

func TestApplyOptions(t *testing.T) {
	tests := []struct {
		name     string
		opts     []interfaces.LLMOption
		expected *interfaces.LLMOptions
	}{
		{
			name: "default options",
			opts: nil,
			expected: &interfaces.LLMOptions{
				MaxTokens:   256,
				Temperature: 0.7,
				Model:       "gpt-3.5-turbo",
			},
		},
		{
			name: "with max tokens",
			opts: []interfaces.LLMOption{
				interfaces.WithMaxTokens(512),
			},
			expected: &interfaces.LLMOptions{
				MaxTokens:   512,
				Temperature: 0.7,
				Model:       "gpt-3.5-turbo",
			},
		},
		{
			name: "with temperature",
			opts: []interfaces.LLMOption{
				interfaces.WithTemperature(0.3),
			},
			expected: &interfaces.LLMOptions{
				MaxTokens:   256,
				Temperature: 0.3,
				Model:       "gpt-3.5-turbo",
			},
		},
		{
			name: "with model",
			opts: []interfaces.LLMOption{
				interfaces.WithModel("gpt-4"),
			},
			expected: &interfaces.LLMOptions{
				MaxTokens:   256,
				Temperature: 0.7,
				Model:       "gpt-4",
			},
		},
		{
			name: "with multiple options",
			opts: []interfaces.LLMOption{
				interfaces.WithMaxTokens(1024),
				interfaces.WithTemperature(0.5),
				interfaces.WithModel("gpt-4-turbo"),
			},
			expected: &interfaces.LLMOptions{
				MaxTokens:   1024,
				Temperature: 0.5,
				Model:       "gpt-4-turbo",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ApplyOptions(tt.opts...)

			if result.MaxTokens != tt.expected.MaxTokens {
				t.Errorf("MaxTokens = %d, want %d", result.MaxTokens, tt.expected.MaxTokens)
			}
			if result.Temperature != tt.expected.Temperature {
				t.Errorf("Temperature = %f, want %f", result.Temperature, tt.expected.Temperature)
			}
			if result.Model != tt.expected.Model {
				t.Errorf("Model = %s, want %s", result.Model, tt.expected.Model)
			}
		})
	}
}

// =============================================================================
// BuildMessages 测试
// =============================================================================

func TestBuildMessages(t *testing.T) {
	tests := []struct {
		name     string
		system   string
		user     string
		expected int
	}{
		{
			name:     "both system and user",
			system:   "You are a helpful assistant",
			user:     "Hello",
			expected: 2,
		},
		{
			name:     "only user",
			system:   "",
			user:     "Hello",
			expected: 1,
		},
		{
			name:     "only system",
			system:   "You are a helpful assistant",
			user:     "",
			expected: 1,
		},
		{
			name:     "neither",
			system:   "",
			user:     "",
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messages := BuildMessages(tt.system, tt.user)

			if len(messages) != tt.expected {
				t.Errorf("len(messages) = %d, want %d", len(messages), tt.expected)
			}
		})
	}
}

func TestBuildSystemMessage(t *testing.T) {
	content := "You are a helpful assistant"
	msg := BuildSystemMessage(content)

	if msg.Role != "system" {
		t.Errorf("Role = %s, want system", msg.Role)
	}
	if msg.Content != content {
		t.Errorf("Content = %s, want %s", msg.Content, content)
	}
}

func TestBuildUserMessage(t *testing.T) {
	content := "Hello, world!"
	msg := BuildUserMessage(content)

	if msg.Role != "user" {
		t.Errorf("Role = %s, want user", msg.Role)
	}
	if msg.Content != content {
		t.Errorf("Content = %s, want %s", msg.Content, content)
	}
}

func TestBuildAssistantMessage(t *testing.T) {
	content := "Hello! How can I help you?"
	msg := BuildAssistantMessage(content)

	if msg.Role != "assistant" {
		t.Errorf("Role = %s, want assistant", msg.Role)
	}
	if msg.Content != content {
		t.Errorf("Content = %s, want %s", msg.Content, content)
	}
}

// =============================================================================
// CachedClient 测试
// =============================================================================

func TestCachedClient_Complete(t *testing.T) {
	mock := NewMockClient()
	mock.DefaultResponse = "cached response"

	cached := NewCachedClient(mock, time.Hour)

	ctx := context.Background()
	prompt := "test prompt"

	// 第一次调用
	response1, err := cached.Complete(ctx, prompt)
	if err != nil {
		t.Fatalf("first call failed: %v", err)
	}
	if response1 != "cached response" {
		t.Errorf("response1 = %s, want 'cached response'", response1)
	}

	// 修改 mock 响应
	mock.DefaultResponse = "new response"

	// 第二次调用应该返回缓存的结果
	response2, err := cached.Complete(ctx, prompt)
	if err != nil {
		t.Fatalf("second call failed: %v", err)
	}
	if response2 != "cached response" {
		t.Errorf("response2 = %s, want 'cached response' (cached)", response2)
	}
}

func TestCachedClient_Embed(t *testing.T) {
	mock := NewMockClient()
	mock.DefaultEmbedding = []float32{1.0, 2.0, 3.0}

	cached := NewCachedClient(mock, time.Hour)

	ctx := context.Background()
	text := "test text"

	// 第一次调用
	embedding1, err := cached.Embed(ctx, text)
	if err != nil {
		t.Fatalf("first call failed: %v", err)
	}
	if len(embedding1) != 3 {
		t.Errorf("len(embedding1) = %d, want 3", len(embedding1))
	}

	// 修改 mock 响应
	mock.DefaultEmbedding = []float32{4.0, 5.0, 6.0}

	// 第二次调用应该返回缓存的结果
	embedding2, err := cached.Embed(ctx, text)
	if err != nil {
		t.Fatalf("second call failed: %v", err)
	}
	if embedding2[0] != 1.0 {
		t.Errorf("embedding2[0] = %f, want 1.0 (cached)", embedding2[0])
	}
}

// =============================================================================
// RetryClient 测试
// =============================================================================

func TestRetryClient_Complete_Success(t *testing.T) {
	mock := NewMockClient()
	mock.DefaultResponse = "success"

	retry := NewRetryClient(mock, 3, time.Millisecond)

	ctx := context.Background()
	response, err := retry.Complete(ctx, "test")

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if response != "success" {
		t.Errorf("response = %s, want 'success'", response)
	}
}

func TestRetryClient_Complete_RetryOnError(t *testing.T) {
	callCount := 0
	mock := NewMockClient()
	mock.CompleteFunc = func(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
		callCount++
		if callCount < 3 {
			return "", ErrRequestTimeout
		}
		return "success after retry", nil
	}

	retry := NewRetryClient(mock, 3, time.Millisecond)

	ctx := context.Background()
	response, err := retry.Complete(ctx, "test")

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if response != "success after retry" {
		t.Errorf("response = %s, want 'success after retry'", response)
	}
	if callCount != 3 {
		t.Errorf("callCount = %d, want 3", callCount)
	}
}

func TestRetryClient_Complete_NoRetryOnCertainErrors(t *testing.T) {
	callCount := 0
	mock := NewMockClient()
	mock.CompleteFunc = func(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
		callCount++
		return "", ErrEmptyPrompt
	}

	retry := NewRetryClient(mock, 3, time.Millisecond)

	ctx := context.Background()
	_, err := retry.Complete(ctx, "test")

	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if callCount != 1 {
		t.Errorf("callCount = %d, want 1 (no retry)", callCount)
	}
}

// =============================================================================
// RateLimitedClient 测试
// =============================================================================

func TestRateLimitedClient_Complete(t *testing.T) {
	mock := NewMockClient()
	mock.DefaultResponse = "limited response"

	limited := NewRateLimitedClient(mock, 2)

	ctx := context.Background()

	// 测试正常调用
	response, err := limited.Complete(ctx, "test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if response != "limited response" {
		t.Errorf("response = %s, want 'limited response'", response)
	}
}

func TestRateLimitedClient_ConcurrencyLimit(t *testing.T) {
	callCount := 0
	maxConcurrent := 0
	currentConcurrent := 0

	mock := NewMockClient()
	mock.CompleteFunc = func(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
		currentConcurrent++
		if currentConcurrent > maxConcurrent {
			maxConcurrent = currentConcurrent
		}
		callCount++
		time.Sleep(10 * time.Millisecond)
		currentConcurrent--
		return "response", nil
	}

	limited := NewRateLimitedClient(mock, 2)

	ctx := context.Background()

	// 启动多个并发请求
	done := make(chan struct{}, 5)
	for i := 0; i < 5; i++ {
		go func() {
			limited.Complete(ctx, "test")
			done <- struct{}{}
		}()
	}

	// 等待所有请求完成
	for i := 0; i < 5; i++ {
		<-done
	}

	if callCount != 5 {
		t.Errorf("callCount = %d, want 5", callCount)
	}
	if maxConcurrent > 2 {
		t.Errorf("maxConcurrent = %d, want <= 2", maxConcurrent)
	}
}

// =============================================================================
// ClientConfig 测试
// =============================================================================

func TestClientConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		config  ClientConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: ClientConfig{
				APIKey: "test-api-key",
			},
			wantErr: false,
		},
		{
			name: "missing API key",
			config: ClientConfig{
				APIKey: "",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestDefaultClientConfig(t *testing.T) {
	cfg := DefaultClientConfig()

	if cfg.BaseURL == "" {
		t.Error("BaseURL should not be empty")
	}
	if cfg.Model == "" {
		t.Error("Model should not be empty")
	}
	if cfg.Timeout == 0 {
		t.Error("Timeout should not be zero")
	}
	if cfg.MaxRetries < 0 {
		t.Error("MaxRetries should be non-negative")
	}
}

