# LLM 客户端模块

> 大语言模型客户端接口与实现

## 目录

- [概述](#概述)
- [架构设计](#架构设计)
- [快速开始](#快速开始)
- [API 参考](#api-参考)
- [客户端实现](#客户端实现)
- [高级功能](#高级功能)
- [测试](#测试)
- [最佳实践](#最佳实践)

---

## 概述

本模块提供统一的 LLM（大语言模型）客户端接口，支持多种后端实现：

- **OpenAI API**：标准 OpenAI 接口
- **Azure OpenAI**：Azure 托管的 OpenAI 服务
- **Ollama**：本地模型推理
- **自定义 HTTP 服务**：支持任意 HTTP 推理服务

### 核心特性

- 🔌 **统一接口**：所有客户端实现相同的 `LLMClient` 接口
- 🔄 **自动重试**：内置重试机制，支持指数退避
- 💾 **响应缓存**：减少重复调用，降低成本
- 🚦 **并发控制**：限制并发请求数量
- 🧪 **Mock 客户端**：方便单元测试

---

## 架构设计

```
┌──────────────────────────────────────────────────────────────┐
│                      interfaces.LLMClient                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
│  │  Complete   │ │    Chat     │ │    Embed    │             │
│  └─────────────┘ └─────────────┘ └─────────────┘             │
└──────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  OpenAIClient   │ │  OllamaClient   │ │ HTTPInference   │
│                 │ │                 │ │     Client      │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │
          ▼
┌─────────────────┐
│ AzureOpenAI     │
│    Client       │
└─────────────────┘
```

### 包装器模式

```
┌─────────────────────────────────────────────────────────────┐
│                      调用链示例                               │
│                                                             │
│  RateLimitedClient → RetryClient → CachedClient → 基础客户端  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 1. 创建 OpenAI 客户端

```go
import (
    "context"
    "recommend-system/internal/llm"
    "recommend-system/internal/interfaces"
)

// 创建客户端
client, err := llm.NewOpenAIClient(llm.OpenAIConfig{
    APIKey: "your-api-key",
    Model:  "gpt-4",
})
if err != nil {
    log.Fatal(err)
}

// 文本补全
response, err := client.Complete(ctx, "你好，请介绍一下自己")

// 对话
messages := []interfaces.Message{
    {Role: "system", Content: "你是一个推荐系统助手"},
    {Role: "user", Content: "推荐一些科技类产品"},
}
response, err := client.Chat(ctx, messages)

// 文本嵌入
embedding, err := client.Embed(ctx, "这是一段需要嵌入的文本")
```

### 2. 创建 Ollama 客户端（本地模型）

```go
client := llm.NewOllamaClient(llm.OllamaConfig{
    BaseURL: "http://localhost:11434",
    Model:   "qwen:7b",
})

response, err := client.Complete(ctx, "Hello, world!")
```

### 3. 使用选项控制参数

```go
response, err := client.Chat(ctx, messages,
    interfaces.WithMaxTokens(512),
    interfaces.WithTemperature(0.3),
    interfaces.WithModel("gpt-4-turbo"),
)
```

---

## API 参考

### LLMClient 接口

```go
type LLMClient interface {
    // Complete 文本补全
    Complete(ctx context.Context, prompt string, opts ...LLMOption) (string, error)
    
    // Embed 文本嵌入
    Embed(ctx context.Context, text string) ([]float32, error)
    
    // Chat 对话式交互
    Chat(ctx context.Context, messages []Message, opts ...LLMOption) (string, error)
}
```

### Message 结构

```go
type Message struct {
    Role    string `json:"role"`    // system, user, assistant
    Content string `json:"content"` // 消息内容
}
```

### LLMOption 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `WithMaxTokens(n)` | 最大 Token 数 | 256 |
| `WithTemperature(t)` | 温度参数 (0-2) | 0.7 |
| `WithModel(m)` | 模型名称 | gpt-3.5-turbo |

---

## 客户端实现

### OpenAI 客户端

```go
// 配置选项
type OpenAIConfig struct {
    APIKey         string        // API 密钥（必需）
    BaseURL        string        // API 地址，默认 https://api.openai.com/v1
    Model          string        // 默认模型，默认 gpt-3.5-turbo
    EmbeddingModel string        // 嵌入模型，默认 text-embedding-ada-002
    Timeout        time.Duration // 超时时间，默认 30s
}

// 创建客户端
client, err := llm.NewOpenAIClient(config)
```

### Azure OpenAI 客户端

```go
type AzureOpenAIConfig struct {
    APIKey       string        // Azure API 密钥
    Endpoint     string        // Azure 端点 URL
    DeploymentID string        // 模型部署 ID
    APIVersion   string        // API 版本，默认 2023-05-15
    Timeout      time.Duration // 超时时间
}

client, err := llm.NewAzureOpenAIClient(config)
```

### Ollama 客户端

```go
type OllamaConfig struct {
    BaseURL        string        // 服务地址，默认 http://localhost:11434
    Model          string        // 默认模型，默认 llama2
    EmbeddingModel string        // 嵌入模型，默认 nomic-embed-text
    Timeout        time.Duration // 超时时间，默认 120s
}

client := llm.NewOllamaClient(config)
```

### 自定义 HTTP 推理客户端

```go
type HTTPInferenceConfig struct {
    BaseURL       string        // 推理服务地址
    APIKey        string        // 可选的 API 密钥
    ChatEndpoint  string        // 对话端点，默认 /v1/chat
    EmbedEndpoint string        // 嵌入端点，默认 /v1/embed
    Timeout       time.Duration // 超时时间
}

client, err := llm.NewHTTPInferenceClient(config)
```

---

## 高级功能

### 1. 响应缓存

```go
// 创建带缓存的客户端
baseClient, _ := llm.NewOpenAIClient(config)
cachedClient := llm.NewCachedClient(baseClient, time.Hour)

// 相同的请求会从缓存返回
response1, _ := cachedClient.Complete(ctx, "Hello")
response2, _ := cachedClient.Complete(ctx, "Hello") // 从缓存返回
```

### 2. 自动重试

```go
// 创建带重试的客户端
baseClient, _ := llm.NewOpenAIClient(config)
retryClient := llm.NewRetryClient(baseClient, 3, time.Second)

// 失败时自动重试，最多 3 次
response, err := retryClient.Complete(ctx, "Hello")
```

### 3. 并发限制

```go
// 创建带并发限制的客户端
baseClient, _ := llm.NewOpenAIClient(config)
limitedClient := llm.NewRateLimitedClient(baseClient, 5) // 最大 5 并发

// 超过限制的请求会等待
response, err := limitedClient.Complete(ctx, "Hello")
```

### 4. 组合使用

```go
// 组合多个包装器
baseClient, _ := llm.NewOpenAIClient(config)

// 先缓存 -> 再重试 -> 再限流
client := llm.NewRateLimitedClient(
    llm.NewRetryClient(
        llm.NewCachedClient(baseClient, time.Hour),
        3,
        time.Second,
    ),
    10,
)
```

### 5. 消息构建辅助函数

```go
// 快速构建消息
systemMsg := llm.BuildSystemMessage("你是一个助手")
userMsg := llm.BuildUserMessage("你好")
assistantMsg := llm.BuildAssistantMessage("你好！")

// 快速构建消息列表
messages := llm.BuildMessages(
    "你是一个推荐系统助手",
    "推荐一些科技产品",
)
```

---

## 测试

### 使用 Mock 客户端

```go
// 创建 Mock 客户端
mock := llm.NewMockClient()

// 设置默认响应
mock.SetResponse("这是模拟响应")
mock.SetEmbedding([]float32{0.1, 0.2, 0.3})

// 设置自定义行为
mock.CompleteFunc = func(ctx context.Context, prompt string, opts ...interfaces.LLMOption) (string, error) {
    return "自定义响应: " + prompt, nil
}

// 模拟错误
mock.SetError(errors.New("模拟错误"))

// 清除错误
mock.ClearError()
```

### 运行单元测试

```bash
# 运行所有测试
go test -v ./internal/llm/...

# 运行特定测试
go test -v ./internal/llm/ -run TestOpenAIClient

# 测试覆盖率
go test -cover ./internal/llm/...
```

---

## 最佳实践

### 1. 错误处理

```go
response, err := client.Complete(ctx, prompt)
if err != nil {
    switch {
    case errors.Is(err, llm.ErrEmptyPrompt):
        // 处理空提示词
    case errors.Is(err, llm.ErrRateLimitExceeded):
        // 处理速率限制
    case errors.Is(err, llm.ErrContextCanceled):
        // 处理上下文取消
    default:
        // 其他错误
    }
}
```

### 2. 超时控制

```go
// 为 LLM 调用设置专用超时
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

response, err := client.Complete(ctx, prompt)
```

### 3. 成本控制

- 使用 `CachedClient` 缓存相同请求
- 合理设置 `MaxTokens` 限制输出长度
- 使用较小的模型（如 gpt-3.5-turbo）进行简单任务

### 4. 降级策略

```go
// 主客户端使用 OpenAI
primaryClient, _ := llm.NewOpenAIClient(openaiConfig)

// 备用客户端使用本地 Ollama
fallbackClient := llm.NewOllamaClient(ollamaConfig)

// 实现降级逻辑
response, err := primaryClient.Complete(ctx, prompt)
if err != nil {
    // 降级到本地模型
    response, err = fallbackClient.Complete(ctx, prompt)
}
```

---

## 文件结构

```
internal/llm/
├── client.go          # 接口定义、配置、包装器
├── client_test.go     # 客户端测试
├── openai.go          # OpenAI/Azure 实现
├── openai_test.go     # OpenAI 测试
├── local.go           # Ollama/HTTP 推理实现
├── local_test.go      # 本地客户端测试
└── README.md          # 本文档
```

---

## 错误码

| 错误 | 说明 |
|------|------|
| `ErrEmptyPrompt` | 提示词为空 |
| `ErrEmptyMessages` | 消息列表为空 |
| `ErrAPIKeyRequired` | 缺少 API 密钥 |
| `ErrRequestTimeout` | 请求超时 |
| `ErrRateLimitExceeded` | 超出速率限制 |
| `ErrModelNotAvailable` | 模型不可用 |
| `ErrInvalidResponse` | 无效的响应 |
| `ErrContextCanceled` | 上下文已取消 |

---

## 贡献指南

1. 新增客户端实现需要实现 `interfaces.LLMClient` 接口
2. 所有公开函数需要添加详细注释
3. 为新功能编写单元测试
4. 更新本文档

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.0 | 2026-01-04 | 初始版本 |

