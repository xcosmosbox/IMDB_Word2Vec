# Person D: Cold Start + LLM Service（冷启动与大模型服务）

## 你的角色
你是一名 Go 后端工程师，负责实现生成式推荐系统的 **冷启动服务** 和 **LLM 集成** 模块。

## 背景知识

冷启动是推荐系统的核心挑战，当用户或物品缺乏历史数据时，需要借助：
- 大语言模型（LLM）生成语义先验
- 跨域知识迁移
- 快速适应策略

## 你的任务

实现以下模块：

```
recommend-system/
├── internal/llm/
│   ├── client.go            # LLM 客户端接口
│   ├── openai.go            # OpenAI 实现
│   └── local.go             # 本地模型实现
└── internal/service/coldstart/
    └── service.go           # 冷启动服务
```

---

## 1. internal/llm/client.go

```go
package llm

import (
    "context"
)

// LLMClient LLM 客户端接口
type LLMClient interface {
    // Complete 文本补全
    Complete(ctx context.Context, prompt string, opts ...Option) (string, error)
    
    // Embed 文本嵌入
    Embed(ctx context.Context, text string) ([]float32, error)
    
    // Chat 对话式交互
    Chat(ctx context.Context, messages []Message, opts ...Option) (string, error)
}

// Message 消息
type Message struct {
    Role    string `json:"role"`    // system, user, assistant
    Content string `json:"content"`
}

// Options 选项
type Options struct {
    MaxTokens   int     `json:"max_tokens"`
    Temperature float64 `json:"temperature"`
    TopP        float64 `json:"top_p"`
    Model       string  `json:"model"`
}

type Option func(*Options)

func WithMaxTokens(n int) Option {
    return func(o *Options) { o.MaxTokens = n }
}

func WithTemperature(t float64) Option {
    return func(o *Options) { o.Temperature = t }
}

func WithModel(model string) Option {
    return func(o *Options) { o.Model = model }
}

// DefaultOptions 默认选项
func DefaultOptions() Options {
    return Options{
        MaxTokens:   256,
        Temperature: 0.7,
        TopP:        1.0,
        Model:       "gpt-3.5-turbo",
    }
}
```

---

## 2. internal/llm/openai.go

```go
package llm

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

// OpenAIClient OpenAI 客户端
type OpenAIClient struct {
    apiKey     string
    baseURL    string
    httpClient *http.Client
}

// OpenAIConfig 配置
type OpenAIConfig struct {
    APIKey  string
    BaseURL string
    Timeout time.Duration
}

// NewOpenAIClient 创建 OpenAI 客户端
func NewOpenAIClient(cfg OpenAIConfig) *OpenAIClient {
    baseURL := cfg.BaseURL
    if baseURL == "" {
        baseURL = "https://api.openai.com/v1"
    }
    
    timeout := cfg.Timeout
    if timeout == 0 {
        timeout = 30 * time.Second
    }
    
    return &OpenAIClient{
        apiKey:  cfg.APIKey,
        baseURL: baseURL,
        httpClient: &http.Client{
            Timeout: timeout,
        },
    }
}

// Chat 对话式交互
func (c *OpenAIClient) Chat(ctx context.Context, messages []Message, opts ...Option) (string, error) {
    options := DefaultOptions()
    for _, opt := range opts {
        opt(&options)
    }
    
    reqBody := map[string]interface{}{
        "model":       options.Model,
        "messages":    messages,
        "max_tokens":  options.MaxTokens,
        "temperature": options.Temperature,
    }
    
    body, err := json.Marshal(reqBody)
    if err != nil {
        return "", err
    }
    
    req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/chat/completions", bytes.NewReader(body))
    if err != nil {
        return "", err
    }
    
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Authorization", "Bearer "+c.apiKey)
    
    resp, err := c.httpClient.Do(req)
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        bodyBytes, _ := io.ReadAll(resp.Body)
        return "", fmt.Errorf("OpenAI API error: %s", string(bodyBytes))
    }
    
    var result struct {
        Choices []struct {
            Message struct {
                Content string `json:"content"`
            } `json:"message"`
        } `json:"choices"`
    }
    
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return "", err
    }
    
    if len(result.Choices) == 0 {
        return "", fmt.Errorf("no response from OpenAI")
    }
    
    return result.Choices[0].Message.Content, nil
}

// Complete 文本补全
func (c *OpenAIClient) Complete(ctx context.Context, prompt string, opts ...Option) (string, error) {
    messages := []Message{
        {Role: "user", Content: prompt},
    }
    return c.Chat(ctx, messages, opts...)
}

// Embed 文本嵌入
func (c *OpenAIClient) Embed(ctx context.Context, text string) ([]float32, error) {
    reqBody := map[string]interface{}{
        "model": "text-embedding-ada-002",
        "input": text,
    }
    
    body, err := json.Marshal(reqBody)
    if err != nil {
        return nil, err
    }
    
    req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/embeddings", bytes.NewReader(body))
    if err != nil {
        return nil, err
    }
    
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Authorization", "Bearer "+c.apiKey)
    
    resp, err := c.httpClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        bodyBytes, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("OpenAI API error: %s", string(bodyBytes))
    }
    
    var result struct {
        Data []struct {
            Embedding []float32 `json:"embedding"`
        } `json:"data"`
    }
    
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, err
    }
    
    if len(result.Data) == 0 {
        return nil, fmt.Errorf("no embedding returned")
    }
    
    return result.Data[0].Embedding, nil
}
```

---

## 3. internal/service/coldstart/service.go

```go
package coldstart

import (
    "context"
    "encoding/json"
    "fmt"
    "time"
    
    "recommend-system/internal/llm"
    "recommend-system/internal/model"
    "recommend-system/internal/repository"
    "recommend-system/internal/cache"
    "recommend-system/pkg/logger"
)

// ColdStartService 冷启动服务
type ColdStartService struct {
    llmClient  llm.LLMClient
    userRepo   repository.UserRepo
    itemRepo   repository.ItemRepo
    cache      cache.Cache
    logger     *logger.Logger
}

// NewColdStartService 创建冷启动服务
func NewColdStartService(
    llmClient llm.LLMClient,
    userRepo repository.UserRepo,
    itemRepo repository.ItemRepo,
    cache cache.Cache,
    logger *logger.Logger,
) *ColdStartService {
    return &ColdStartService{
        llmClient: llmClient,
        userRepo:  userRepo,
        itemRepo:  itemRepo,
        cache:     cache,
        logger:    logger,
    }
}

// HandleNewUser 处理新用户冷启动
func (s *ColdStartService) HandleNewUser(ctx context.Context, user *model.User) (*ColdStartResult, error) {
    // 1. 基于用户属性生成初始偏好
    prompt := s.buildUserPreferencePrompt(user)
    
    response, err := s.llmClient.Chat(ctx, []llm.Message{
        {Role: "system", Content: "你是一个推荐系统助手，负责分析用户偏好。请以 JSON 格式返回分析结果。"},
        {Role: "user", Content: prompt},
    }, llm.WithMaxTokens(512), llm.WithTemperature(0.3))
    
    if err != nil {
        s.logger.Warn("LLM cold start failed", "user_id", user.ID, "error", err)
        return s.fallbackUserColdStart(ctx, user)
    }
    
    // 2. 解析 LLM 响应
    var preferences UserPreferences
    if err := json.Unmarshal([]byte(response), &preferences); err != nil {
        s.logger.Warn("Failed to parse LLM response", "response", response)
        return s.fallbackUserColdStart(ctx, user)
    }
    
    // 3. 基于偏好查找初始推荐
    recommendations, err := s.getInitialRecommendations(ctx, &preferences)
    if err != nil {
        return nil, err
    }
    
    // 4. 缓存冷启动结果
    result := &ColdStartResult{
        UserID:          user.ID,
        Preferences:     preferences,
        Recommendations: recommendations,
        Strategy:        "llm_based",
        CreatedAt:       time.Now(),
    }
    
    _ = s.cache.Set(ctx, "coldstart:user:"+user.ID, result, 24*time.Hour)
    
    return result, nil
}

// HandleNewItem 处理新物品冷启动
func (s *ColdStartService) HandleNewItem(ctx context.Context, item *model.Item) (*ItemColdStartResult, error) {
    // 1. 使用 LLM 理解物品内容
    prompt := s.buildItemUnderstandingPrompt(item)
    
    response, err := s.llmClient.Chat(ctx, []llm.Message{
        {Role: "system", Content: "你是一个内容分析助手，负责分析物品特征。请以 JSON 格式返回分析结果。"},
        {Role: "user", Content: prompt},
    }, llm.WithMaxTokens(512), llm.WithTemperature(0.3))
    
    if err != nil {
        s.logger.Warn("LLM item analysis failed", "item_id", item.ID, "error", err)
        return s.fallbackItemColdStart(ctx, item)
    }
    
    // 2. 解析物品特征
    var features ItemFeatures
    if err := json.Unmarshal([]byte(response), &features); err != nil {
        s.logger.Warn("Failed to parse item features", "response", response)
        return s.fallbackItemColdStart(ctx, item)
    }
    
    // 3. 生成物品嵌入向量
    embedding, err := s.llmClient.Embed(ctx, item.Title+" "+item.Description)
    if err != nil {
        s.logger.Warn("Failed to generate item embedding", "item_id", item.ID, "error", err)
    }
    
    // 4. 查找相似物品（用于冷启动推荐）
    similarItems, err := s.findSimilarItems(ctx, embedding)
    if err != nil {
        s.logger.Warn("Failed to find similar items", "item_id", item.ID, "error", err)
    }
    
    result := &ItemColdStartResult{
        ItemID:       item.ID,
        Features:     features,
        Embedding:    embedding,
        SimilarItems: similarItems,
        Strategy:     "llm_based",
        CreatedAt:    time.Now(),
    }
    
    _ = s.cache.Set(ctx, "coldstart:item:"+item.ID, result, 24*time.Hour)
    
    return result, nil
}

// GetColdStartRecommendations 获取冷启动推荐
func (s *ColdStartService) GetColdStartRecommendations(ctx context.Context, userID string, limit int) ([]*model.Item, error) {
    // 1. 尝试从缓存获取
    var result ColdStartResult
    if err := s.cache.Get(ctx, "coldstart:user:"+userID, &result); err == nil {
        return s.expandRecommendations(ctx, result.Recommendations, limit)
    }
    
    // 2. 获取用户信息
    user, err := s.userRepo.GetByID(ctx, userID)
    if err != nil {
        return nil, err
    }
    
    // 3. 执行冷启动
    coldStartResult, err := s.HandleNewUser(ctx, user)
    if err != nil {
        return nil, err
    }
    
    return s.expandRecommendations(ctx, coldStartResult.Recommendations, limit)
}

// ExplainRecommendation 生成推荐解释
func (s *ColdStartService) ExplainRecommendation(ctx context.Context, userID, itemID string) (string, error) {
    // 获取用户和物品信息
    user, err := s.userRepo.GetByID(ctx, userID)
    if err != nil {
        return "", err
    }
    
    item, err := s.itemRepo.GetByID(ctx, itemID)
    if err != nil {
        return "", err
    }
    
    // 使用 LLM 生成解释
    prompt := fmt.Sprintf(`
用户信息：
- 年龄：%d
- 性别：%s

推荐物品：
- 标题：%s
- 类别：%s
- 描述：%s

请用一句话解释为什么向这位用户推荐这个物品。
`, user.Age, user.Gender, item.Title, item.Category, item.Description)
    
    explanation, err := s.llmClient.Complete(ctx, prompt, llm.WithMaxTokens(100))
    if err != nil {
        return "根据您的偏好为您推荐", nil
    }
    
    return explanation, nil
}

// 辅助函数

func (s *ColdStartService) buildUserPreferencePrompt(user *model.User) string {
    return fmt.Sprintf(`
分析以下用户的可能偏好：
- 年龄：%d
- 性别：%s

请返回 JSON 格式的偏好分析：
{
    "preferred_categories": ["类别1", "类别2"],
    "preferred_tags": ["标签1", "标签2"],
    "content_preference": "short/medium/long",
    "price_sensitivity": "low/medium/high"
}
`, user.Age, user.Gender)
}

func (s *ColdStartService) buildItemUnderstandingPrompt(item *model.Item) string {
    return fmt.Sprintf(`
分析以下物品：
- 标题：%s
- 类别：%s
- 描述：%s
- 标签：%v

请返回 JSON 格式的特征分析：
{
    "main_category": "主类别",
    "sub_categories": ["子类别1", "子类别2"],
    "target_audience": ["目标用户群1", "目标用户群2"],
    "content_type": "类型描述",
    "quality_score": 0.8
}
`, item.Title, item.Category, item.Description, item.Tags)
}

func (s *ColdStartService) fallbackUserColdStart(ctx context.Context, user *model.User) (*ColdStartResult, error) {
    // 降级策略：基于人口统计的默认推荐
    var categories []string
    
    if user.Age < 25 {
        categories = []string{"科技", "游戏", "动漫"}
    } else if user.Age < 40 {
        categories = []string{"商业", "新闻", "生活"}
    } else {
        categories = []string{"健康", "新闻", "文化"}
    }
    
    // 获取热门物品
    recommendations, err := s.getPopularItemsByCategories(ctx, categories, 20)
    if err != nil {
        return nil, err
    }
    
    return &ColdStartResult{
        UserID: user.ID,
        Preferences: UserPreferences{
            PreferredCategories: categories,
        },
        Recommendations: recommendations,
        Strategy:        "demographic_fallback",
        CreatedAt:       time.Now(),
    }, nil
}

func (s *ColdStartService) fallbackItemColdStart(ctx context.Context, item *model.Item) (*ItemColdStartResult, error) {
    return &ItemColdStartResult{
        ItemID: item.ID,
        Features: ItemFeatures{
            MainCategory: item.Category,
            SubCategories: item.Tags,
        },
        Strategy:  "fallback",
        CreatedAt: time.Now(),
    }, nil
}

func (s *ColdStartService) getInitialRecommendations(ctx context.Context, prefs *UserPreferences) ([]string, error) {
    return s.getPopularItemsByCategories(ctx, prefs.PreferredCategories, 50)
}

func (s *ColdStartService) getPopularItemsByCategories(ctx context.Context, categories []string, limit int) ([]string, error) {
    items, err := s.itemRepo.GetPopularByCategories(ctx, categories, limit)
    if err != nil {
        return nil, err
    }
    
    ids := make([]string, len(items))
    for i, item := range items {
        ids[i] = item.ID
    }
    return ids, nil
}

func (s *ColdStartService) findSimilarItems(ctx context.Context, embedding []float32) ([]string, error) {
    // 使用 Milvus 向量搜索
    // 实现细节...
    return nil, nil
}

func (s *ColdStartService) expandRecommendations(ctx context.Context, itemIDs []string, limit int) ([]*model.Item, error) {
    if len(itemIDs) > limit {
        itemIDs = itemIDs[:limit]
    }
    return s.itemRepo.GetByIDs(ctx, itemIDs)
}

// 数据结构

type UserPreferences struct {
    PreferredCategories []string `json:"preferred_categories"`
    PreferredTags       []string `json:"preferred_tags"`
    ContentPreference   string   `json:"content_preference"`
    PriceSensitivity    string   `json:"price_sensitivity"`
}

type ColdStartResult struct {
    UserID          string          `json:"user_id"`
    Preferences     UserPreferences `json:"preferences"`
    Recommendations []string        `json:"recommendations"`
    Strategy        string          `json:"strategy"`
    CreatedAt       time.Time       `json:"created_at"`
}

type ItemFeatures struct {
    MainCategory   string   `json:"main_category"`
    SubCategories  []string `json:"sub_categories"`
    TargetAudience []string `json:"target_audience"`
    ContentType    string   `json:"content_type"`
    QualityScore   float64  `json:"quality_score"`
}

type ItemColdStartResult struct {
    ItemID       string       `json:"item_id"`
    Features     ItemFeatures `json:"features"`
    Embedding    []float32    `json:"embedding,omitempty"`
    SimilarItems []string     `json:"similar_items"`
    Strategy     string       `json:"strategy"`
    CreatedAt    time.Time    `json:"created_at"`
}
```

---

## 注意事项

1. **LLM 调用**: 需要处理超时、重试、降级
2. **成本控制**: 缓存 LLM 结果，避免重复调用
3. **降级策略**: LLM 失败时使用基于规则的降级
4. **隐私保护**: 不要将敏感用户信息发送给 LLM

## 输出要求

请输出完整的可运行代码，包含：
1. 所有 Go 文件
2. 详细的中文注释
3. 单元测试

