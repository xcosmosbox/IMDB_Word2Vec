package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"recommend-system/internal/interfaces"
)

// =============================================================================
// OpenAIClient 创建测试
// =============================================================================

func TestNewOpenAIClient(t *testing.T) {
	tests := []struct {
		name    string
		config  OpenAIConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: OpenAIConfig{
				APIKey: "test-api-key",
			},
			wantErr: false,
		},
		{
			name: "missing API key",
			config: OpenAIConfig{
				APIKey: "",
			},
			wantErr: true,
		},
		{
			name: "full config",
			config: OpenAIConfig{
				APIKey:         "test-api-key",
				BaseURL:        "https://custom.openai.com/v1",
				Model:          "gpt-4",
				EmbeddingModel: "text-embedding-3-large",
				Timeout:        60 * time.Second,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewOpenAIClient(tt.config)

			if (err != nil) != tt.wantErr {
				t.Errorf("NewOpenAIClient() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && client == nil {
				t.Error("expected client to be non-nil")
			}
		})
	}
}

func TestDefaultOpenAIConfig(t *testing.T) {
	cfg := DefaultOpenAIConfig()

	if cfg.BaseURL == "" {
		t.Error("BaseURL should have default value")
	}
	if cfg.Model == "" {
		t.Error("Model should have default value")
	}
	if cfg.EmbeddingModel == "" {
		t.Error("EmbeddingModel should have default value")
	}
	if cfg.Timeout == 0 {
		t.Error("Timeout should have default value")
	}
}

// =============================================================================
// OpenAIClient Complete 测试
// =============================================================================

func TestOpenAIClient_Complete(t *testing.T) {
	// 创建模拟服务器
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 验证请求头
		if r.Header.Get("Authorization") != "Bearer test-api-key" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}

		// 返回模拟响应
		response := chatResponse{
			ID:    "test-id",
			Model: "gpt-3.5-turbo",
			Choices: []struct {
				Index   int `json:"index"`
				Message struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			}{
				{
					Index: 0,
					Message: struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					}{
						Role:    "assistant",
						Content: "Hello! How can I help you?",
					},
					FinishReason: "stop",
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client, err := NewOpenAIClient(OpenAIConfig{
		APIKey:  "test-api-key",
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	ctx := context.Background()
	response, err := client.Complete(ctx, "Hello")

	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}

	if response != "Hello! How can I help you?" {
		t.Errorf("Complete() = %s, want 'Hello! How can I help you?'", response)
	}
}

func TestOpenAIClient_Complete_EmptyPrompt(t *testing.T) {
	client, err := NewOpenAIClient(OpenAIConfig{
		APIKey: "test-api-key",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	ctx := context.Background()
	_, err = client.Complete(ctx, "")

	if err != ErrEmptyPrompt {
		t.Errorf("Complete() error = %v, want ErrEmptyPrompt", err)
	}
}

// =============================================================================
// OpenAIClient Chat 测试
// =============================================================================

func TestOpenAIClient_Chat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 解析请求体
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		// 验证消息
		if len(req.Messages) < 1 {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		response := chatResponse{
			ID:    "test-id",
			Model: req.Model,
			Choices: []struct {
				Index   int `json:"index"`
				Message struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			}{
				{
					Index: 0,
					Message: struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					}{
						Role:    "assistant",
						Content: "I understand your message.",
					},
					FinishReason: "stop",
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client, err := NewOpenAIClient(OpenAIConfig{
		APIKey:  "test-api-key",
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	ctx := context.Background()
	messages := []interfaces.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Hello!"},
	}

	response, err := client.Chat(ctx, messages)

	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if response != "I understand your message." {
		t.Errorf("Chat() = %s, want 'I understand your message.'", response)
	}
}

func TestOpenAIClient_Chat_EmptyMessages(t *testing.T) {
	client, err := NewOpenAIClient(OpenAIConfig{
		APIKey: "test-api-key",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	ctx := context.Background()
	_, err = client.Chat(ctx, []interfaces.Message{})

	if err != ErrEmptyMessages {
		t.Errorf("Chat() error = %v, want ErrEmptyMessages", err)
	}
}

func TestOpenAIClient_Chat_WithOptions(t *testing.T) {
	var receivedReq chatRequest

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedReq)

		response := chatResponse{
			Choices: []struct {
				Index   int `json:"index"`
				Message struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			}{
				{
					Message: struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					}{
						Role:    "assistant",
						Content: "response",
					},
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client, _ := NewOpenAIClient(OpenAIConfig{
		APIKey:  "test-api-key",
		BaseURL: server.URL,
	})

	ctx := context.Background()
	messages := []interfaces.Message{
		{Role: "user", Content: "test"},
	}

	client.Chat(ctx, messages,
		interfaces.WithMaxTokens(1024),
		interfaces.WithTemperature(0.5),
		interfaces.WithModel("gpt-4"),
	)

	if receivedReq.MaxTokens != 1024 {
		t.Errorf("MaxTokens = %d, want 1024", receivedReq.MaxTokens)
	}
	if receivedReq.Temperature != 0.5 {
		t.Errorf("Temperature = %f, want 0.5", receivedReq.Temperature)
	}
	if receivedReq.Model != "gpt-4" {
		t.Errorf("Model = %s, want gpt-4", receivedReq.Model)
	}
}

// =============================================================================
// OpenAIClient Embed 测试
// =============================================================================

func TestOpenAIClient_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := embeddingResponse{
			Object: "list",
			Data: []struct {
				Object    string    `json:"object"`
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			}{
				{
					Object:    "embedding",
					Index:     0,
					Embedding: []float32{0.1, 0.2, 0.3, 0.4, 0.5},
				},
			},
			Model: "text-embedding-ada-002",
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client, err := NewOpenAIClient(OpenAIConfig{
		APIKey:  "test-api-key",
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	ctx := context.Background()
	embedding, err := client.Embed(ctx, "test text")

	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}

	if len(embedding) != 5 {
		t.Errorf("len(embedding) = %d, want 5", len(embedding))
	}

	if embedding[0] != 0.1 {
		t.Errorf("embedding[0] = %f, want 0.1", embedding[0])
	}
}

func TestOpenAIClient_Embed_EmptyText(t *testing.T) {
	client, err := NewOpenAIClient(OpenAIConfig{
		APIKey: "test-api-key",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	ctx := context.Background()
	_, err = client.Embed(ctx, "")

	if err != ErrEmptyPrompt {
		t.Errorf("Embed() error = %v, want ErrEmptyPrompt", err)
	}
}

// =============================================================================
// OpenAIClient 错误处理测试
// =============================================================================

func TestOpenAIClient_HandleErrorResponse(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		body       string
		wantErr    error
	}{
		{
			name:       "rate limit",
			statusCode: http.StatusTooManyRequests,
			body:       `{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}`,
			wantErr:    ErrRateLimitExceeded,
		},
		{
			name:       "unauthorized",
			statusCode: http.StatusUnauthorized,
			body:       `{"error": {"message": "Invalid API key", "type": "auth_error"}}`,
			wantErr:    ErrAPIKeyRequired,
		},
		{
			name:       "service unavailable",
			statusCode: http.StatusServiceUnavailable,
			body:       `{"error": {"message": "Model overloaded", "type": "server_error"}}`,
			wantErr:    ErrModelNotAvailable,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				w.Write([]byte(tt.body))
			}))
			defer server.Close()

			client, _ := NewOpenAIClient(OpenAIConfig{
				APIKey:  "test-api-key",
				BaseURL: server.URL,
			})

			ctx := context.Background()
			_, err := client.Complete(ctx, "test")

			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestOpenAIClient_EmptyChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := chatResponse{
			ID:      "test-id",
			Choices: []struct {
				Index   int `json:"index"`
				Message struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			}{},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client, _ := NewOpenAIClient(OpenAIConfig{
		APIKey:  "test-api-key",
		BaseURL: server.URL,
	})

	ctx := context.Background()
	_, err := client.Complete(ctx, "test")

	if err != ErrInvalidResponse {
		t.Errorf("Complete() error = %v, want ErrInvalidResponse", err)
	}
}

// =============================================================================
// Azure OpenAI 客户端测试
// =============================================================================

func TestNewAzureOpenAIClient(t *testing.T) {
	tests := []struct {
		name    string
		config  AzureOpenAIConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: AzureOpenAIConfig{
				APIKey:       "test-api-key",
				Endpoint:     "https://myresource.openai.azure.com",
				DeploymentID: "my-gpt-4",
			},
			wantErr: false,
		},
		{
			name: "missing API key",
			config: AzureOpenAIConfig{
				APIKey:       "",
				Endpoint:     "https://myresource.openai.azure.com",
				DeploymentID: "my-gpt-4",
			},
			wantErr: true,
		},
		{
			name: "missing endpoint",
			config: AzureOpenAIConfig{
				APIKey:       "test-api-key",
				Endpoint:     "",
				DeploymentID: "my-gpt-4",
			},
			wantErr: true,
		},
		{
			name: "missing deployment ID",
			config: AzureOpenAIConfig{
				APIKey:       "test-api-key",
				Endpoint:     "https://myresource.openai.azure.com",
				DeploymentID: "",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewAzureOpenAIClient(tt.config)

			if (err != nil) != tt.wantErr {
				t.Errorf("NewAzureOpenAIClient() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && client == nil {
				t.Error("expected client to be non-nil")
			}
		})
	}
}

// =============================================================================
// 消息转换测试
// =============================================================================

func TestConvertMessages(t *testing.T) {
	messages := []interfaces.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello!"},
		{Role: "assistant", Content: "Hi there!"},
	}

	converted := convertMessages(messages)

	if len(converted) != 3 {
		t.Errorf("len(converted) = %d, want 3", len(converted))
	}

	for i, msg := range converted {
		if msg.Role != messages[i].Role {
			t.Errorf("converted[%d].Role = %s, want %s", i, msg.Role, messages[i].Role)
		}
		if msg.Content != messages[i].Content {
			t.Errorf("converted[%d].Content = %s, want %s", i, msg.Content, messages[i].Content)
		}
	}
}

