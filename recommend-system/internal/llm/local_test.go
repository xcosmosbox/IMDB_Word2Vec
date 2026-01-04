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
// OllamaClient 创建测试
// =============================================================================

func TestNewOllamaClient(t *testing.T) {
	tests := []struct {
		name     string
		config   OllamaConfig
		expected OllamaConfig
	}{
		{
			name:   "default config",
			config: OllamaConfig{},
			expected: OllamaConfig{
				BaseURL:        "http://localhost:11434",
				Model:          "llama2",
				EmbeddingModel: "nomic-embed-text",
				Timeout:        120 * time.Second,
			},
		},
		{
			name: "custom config",
			config: OllamaConfig{
				BaseURL:        "http://ollama.local:11434",
				Model:          "qwen",
				EmbeddingModel: "bge-m3",
				Timeout:        60 * time.Second,
			},
			expected: OllamaConfig{
				BaseURL:        "http://ollama.local:11434",
				Model:          "qwen",
				EmbeddingModel: "bge-m3",
				Timeout:        60 * time.Second,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewOllamaClient(tt.config)

			if client == nil {
				t.Fatal("client should not be nil")
			}

			if client.baseURL != tt.expected.BaseURL {
				t.Errorf("baseURL = %s, want %s", client.baseURL, tt.expected.BaseURL)
			}
			if client.model != tt.expected.Model {
				t.Errorf("model = %s, want %s", client.model, tt.expected.Model)
			}
			if client.embeddingModel != tt.expected.EmbeddingModel {
				t.Errorf("embeddingModel = %s, want %s", client.embeddingModel, tt.expected.EmbeddingModel)
			}
		})
	}
}

func TestDefaultOllamaConfig(t *testing.T) {
	cfg := DefaultOllamaConfig()

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
// OllamaClient Complete 测试
// =============================================================================

func TestOllamaClient_Complete(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			w.WriteHeader(http.StatusNotFound)
			return
		}

		var req ollamaGenerateRequest
		json.NewDecoder(r.Body).Decode(&req)

		response := ollamaGenerateResponse{
			Model:    req.Model,
			Response: "Ollama response: " + req.Prompt,
			Done:     true,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewOllamaClient(OllamaConfig{
		BaseURL: server.URL,
		Model:   "llama2",
	})

	ctx := context.Background()
	response, err := client.Complete(ctx, "Hello")

	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}

	expected := "Ollama response: Hello"
	if response != expected {
		t.Errorf("Complete() = %s, want %s", response, expected)
	}
}

func TestOllamaClient_Complete_EmptyPrompt(t *testing.T) {
	client := NewOllamaClient(OllamaConfig{})

	ctx := context.Background()
	_, err := client.Complete(ctx, "")

	if err != ErrEmptyPrompt {
		t.Errorf("Complete() error = %v, want ErrEmptyPrompt", err)
	}
}

func TestOllamaClient_Complete_WithOptions(t *testing.T) {
	var receivedReq ollamaGenerateRequest

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedReq)

		response := ollamaGenerateResponse{
			Response: "response",
			Done:     true,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewOllamaClient(OllamaConfig{
		BaseURL: server.URL,
		Model:   "llama2",
	})

	ctx := context.Background()
	client.Complete(ctx, "test",
		interfaces.WithMaxTokens(512),
		interfaces.WithTemperature(0.8),
		interfaces.WithModel("qwen"),
	)

	if receivedReq.Model != "qwen" {
		t.Errorf("Model = %s, want qwen", receivedReq.Model)
	}
	if receivedReq.Options.NumPredict != 512 {
		t.Errorf("NumPredict = %d, want 512", receivedReq.Options.NumPredict)
	}
	if receivedReq.Options.Temperature != 0.8 {
		t.Errorf("Temperature = %f, want 0.8", receivedReq.Options.Temperature)
	}
}

// =============================================================================
// OllamaClient Chat 测试
// =============================================================================

func TestOllamaClient_Chat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			w.WriteHeader(http.StatusNotFound)
			return
		}

		response := ollamaChatResponse{
			Model: "llama2",
			Message: ollamaMessage{
				Role:    "assistant",
				Content: "Chat response from Ollama",
			},
			Done: true,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewOllamaClient(OllamaConfig{
		BaseURL: server.URL,
	})

	ctx := context.Background()
	messages := []interfaces.Message{
		{Role: "user", Content: "Hello"},
	}

	response, err := client.Chat(ctx, messages)

	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if response != "Chat response from Ollama" {
		t.Errorf("Chat() = %s, want 'Chat response from Ollama'", response)
	}
}

func TestOllamaClient_Chat_EmptyMessages(t *testing.T) {
	client := NewOllamaClient(OllamaConfig{})

	ctx := context.Background()
	_, err := client.Chat(ctx, []interfaces.Message{})

	if err != ErrEmptyMessages {
		t.Errorf("Chat() error = %v, want ErrEmptyMessages", err)
	}
}

// =============================================================================
// OllamaClient Embed 测试
// =============================================================================

func TestOllamaClient_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embeddings" {
			w.WriteHeader(http.StatusNotFound)
			return
		}

		response := ollamaEmbedResponse{
			Embedding: []float64{0.1, 0.2, 0.3, 0.4, 0.5},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewOllamaClient(OllamaConfig{
		BaseURL: server.URL,
	})

	ctx := context.Background()
	embedding, err := client.Embed(ctx, "test text")

	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}

	if len(embedding) != 5 {
		t.Errorf("len(embedding) = %d, want 5", len(embedding))
	}

	// 验证类型转换
	if embedding[0] != 0.1 {
		t.Errorf("embedding[0] = %f, want 0.1", embedding[0])
	}
}

func TestOllamaClient_Embed_EmptyText(t *testing.T) {
	client := NewOllamaClient(OllamaConfig{})

	ctx := context.Background()
	_, err := client.Embed(ctx, "")

	if err != ErrEmptyPrompt {
		t.Errorf("Embed() error = %v, want ErrEmptyPrompt", err)
	}
}

// =============================================================================
// HTTPInferenceClient 测试
// =============================================================================

func TestNewHTTPInferenceClient(t *testing.T) {
	tests := []struct {
		name    string
		config  HTTPInferenceConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: HTTPInferenceConfig{
				BaseURL: "http://localhost:8080",
			},
			wantErr: false,
		},
		{
			name: "missing base URL",
			config: HTTPInferenceConfig{
				BaseURL: "",
			},
			wantErr: true,
		},
		{
			name: "full config",
			config: HTTPInferenceConfig{
				BaseURL:       "http://localhost:8080",
				APIKey:        "test-key",
				ChatEndpoint:  "/api/chat",
				EmbedEndpoint: "/api/embed",
				Timeout:       30 * time.Second,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewHTTPInferenceClient(tt.config)

			if (err != nil) != tt.wantErr {
				t.Errorf("NewHTTPInferenceClient() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && client == nil {
				t.Error("expected client to be non-nil")
			}
		})
	}
}

func TestHTTPInferenceClient_Chat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 检查认证头
		if r.Header.Get("Authorization") != "Bearer test-key" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}

		response := map[string]interface{}{
			"response": "HTTP inference response",
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client, err := NewHTTPInferenceClient(HTTPInferenceConfig{
		BaseURL: server.URL,
		APIKey:  "test-key",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	ctx := context.Background()
	messages := []interfaces.Message{
		{Role: "user", Content: "Hello"},
	}

	response, err := client.Chat(ctx, messages)

	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if response != "HTTP inference response" {
		t.Errorf("Chat() = %s, want 'HTTP inference response'", response)
	}
}

func TestHTTPInferenceClient_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			"embedding": []float32{0.1, 0.2, 0.3},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client, _ := NewHTTPInferenceClient(HTTPInferenceConfig{
		BaseURL: server.URL,
	})

	ctx := context.Background()
	embedding, err := client.Embed(ctx, "test text")

	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}

	if len(embedding) != 3 {
		t.Errorf("len(embedding) = %d, want 3", len(embedding))
	}
}

// =============================================================================
// MockClient 测试
// =============================================================================

func TestMockClient(t *testing.T) {
	mock := NewMockClient()

	ctx := context.Background()

	// 测试默认响应
	response, err := mock.Complete(ctx, "test")
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if response == "" {
		t.Error("expected non-empty default response")
	}

	// 测试自定义响应
	mock.SetResponse("custom response")
	response, _ = mock.Complete(ctx, "test")
	if response != "custom response" {
		t.Errorf("Complete() = %s, want 'custom response'", response)
	}

	// 测试错误
	mock.SetError(ErrRequestTimeout)
	_, err = mock.Complete(ctx, "test")
	if err == nil {
		t.Error("expected error, got nil")
	}

	// 清除错误
	mock.ClearError()
	_, err = mock.Complete(ctx, "test")
	if err != nil {
		t.Errorf("Complete() error = %v after ClearError", err)
	}
}

func TestMockClient_CustomFunctions(t *testing.T) {
	mock := NewMockClient()

	callCount := 0
	mock.CompleteFunc = func(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
		callCount++
		return "custom: " + prompt, nil
	}

	ctx := context.Background()
	response, _ := mock.Complete(ctx, "hello")

	if response != "custom: hello" {
		t.Errorf("Complete() = %s, want 'custom: hello'", response)
	}
	if callCount != 1 {
		t.Errorf("callCount = %d, want 1", callCount)
	}
}

func TestMockClient_Embed(t *testing.T) {
	mock := NewMockClient()

	// 测试默认嵌入
	ctx := context.Background()
	embedding, err := mock.Embed(ctx, "test")
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(embedding) == 0 {
		t.Error("expected non-empty default embedding")
	}

	// 测试自定义嵌入
	customEmbedding := []float32{1.0, 2.0, 3.0}
	mock.SetEmbedding(customEmbedding)
	embedding, _ = mock.Embed(ctx, "test")
	if len(embedding) != 3 {
		t.Errorf("len(embedding) = %d, want 3", len(embedding))
	}
	if embedding[0] != 1.0 {
		t.Errorf("embedding[0] = %f, want 1.0", embedding[0])
	}
}

func TestMockClient_Chat(t *testing.T) {
	mock := NewMockClient()
	mock.SetResponse("chat response")

	ctx := context.Background()
	messages := []interfaces.Message{
		{Role: "user", Content: "hello"},
	}

	response, err := mock.Chat(ctx, messages)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if response != "chat response" {
		t.Errorf("Chat() = %s, want 'chat response'", response)
	}
}

// =============================================================================
// 消息转换测试
// =============================================================================

func TestConvertToOllamaMessages(t *testing.T) {
	messages := []interfaces.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello!"},
		{Role: "assistant", Content: "Hi there!"},
	}

	converted := convertToOllamaMessages(messages)

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

