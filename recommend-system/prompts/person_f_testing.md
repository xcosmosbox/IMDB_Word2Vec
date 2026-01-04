# Person F: Testing（测试）

## 你的角色
你是一名 Go 测试工程师，负责实现生成式推荐系统的 **完整测试套件**。

## 背景知识

高质量的测试是系统稳定性的保障，包括：
- **单元测试**: 测试单个函数/方法
- **集成测试**: 测试模块间交互
- **性能测试**: 测试系统性能和压力
- **端到端测试**: 测试完整业务流程

## 你的任务

实现以下测试模块：

```
recommend-system/
└── tests/
    ├── unit/
    │   ├── service_test.go       # 服务层测试
    │   ├── repository_test.go    # 数据访问层测试
    │   └── handler_test.go       # API 处理器测试
    ├── integration/
    │   ├── recommend_test.go     # 推荐服务集成测试
    │   ├── user_test.go          # 用户服务集成测试
    │   └── item_test.go          # 物品服务集成测试
    ├── mocks/
    │   ├── user_repo_mock.go     # Mock 用户仓库
    │   ├── item_repo_mock.go     # Mock 物品仓库
    │   └── cache_mock.go         # Mock 缓存
    ├── fixtures/
    │   └── test_data.go          # 测试数据
    └── benchmark/
        └── recommend_bench_test.go # 性能测试
```

---

## 1. tests/mocks/user_repo_mock.go

```go
package mocks

import (
    "context"
    "sync"
    
    "recommend-system/internal/model"
)

// MockUserRepo Mock 用户仓库
type MockUserRepo struct {
    mu        sync.RWMutex
    users     map[string]*model.User
    behaviors map[string][]*model.UserBehavior
}

// NewMockUserRepo 创建 Mock 用户仓库
func NewMockUserRepo() *MockUserRepo {
    return &MockUserRepo{
        users:     make(map[string]*model.User),
        behaviors: make(map[string][]*model.UserBehavior),
    }
}

// GetByID 获取用户
func (m *MockUserRepo) GetByID(ctx context.Context, userID string) (*model.User, error) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    user, ok := m.users[userID]
    if !ok {
        return nil, ErrNotFound
    }
    return user, nil
}

// Create 创建用户
func (m *MockUserRepo) Create(ctx context.Context, user *model.User) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    m.users[user.ID] = user
    return nil
}

// Update 更新用户
func (m *MockUserRepo) Update(ctx context.Context, user *model.User) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    if _, ok := m.users[user.ID]; !ok {
        return ErrNotFound
    }
    m.users[user.ID] = user
    return nil
}

// Delete 删除用户
func (m *MockUserRepo) Delete(ctx context.Context, userID string) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    delete(m.users, userID)
    return nil
}

// GetBehaviors 获取用户行为
func (m *MockUserRepo) GetBehaviors(ctx context.Context, userID string, limit int) ([]*model.UserBehavior, error) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    behaviors, ok := m.behaviors[userID]
    if !ok {
        return []*model.UserBehavior{}, nil
    }
    
    if len(behaviors) > limit {
        return behaviors[:limit], nil
    }
    return behaviors, nil
}

// AddBehavior 添加行为
func (m *MockUserRepo) AddBehavior(ctx context.Context, behavior *model.UserBehavior) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    m.behaviors[behavior.UserID] = append(m.behaviors[behavior.UserID], behavior)
    return nil
}

// SetUser 设置用户（测试用）
func (m *MockUserRepo) SetUser(user *model.User) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.users[user.ID] = user
}

// SetBehaviors 设置行为（测试用）
func (m *MockUserRepo) SetBehaviors(userID string, behaviors []*model.UserBehavior) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.behaviors[userID] = behaviors
}

// 错误定义
var ErrNotFound = errors.New("not found")
```

---

## 2. tests/mocks/cache_mock.go

```go
package mocks

import (
    "context"
    "encoding/json"
    "sync"
    "time"
)

// MockCache Mock 缓存
type MockCache struct {
    mu    sync.RWMutex
    data  map[string][]byte
    ttls  map[string]time.Time
}

// NewMockCache 创建 Mock 缓存
func NewMockCache() *MockCache {
    return &MockCache{
        data: make(map[string][]byte),
        ttls: make(map[string]time.Time),
    }
}

// Get 获取缓存
func (m *MockCache) Get(ctx context.Context, key string, value interface{}) error {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    // 检查是否过期
    if expiry, ok := m.ttls[key]; ok && time.Now().After(expiry) {
        return ErrCacheMiss
    }
    
    data, ok := m.data[key]
    if !ok {
        return ErrCacheMiss
    }
    
    return json.Unmarshal(data, value)
}

// Set 设置缓存
func (m *MockCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    data, err := json.Marshal(value)
    if err != nil {
        return err
    }
    
    m.data[key] = data
    m.ttls[key] = time.Now().Add(ttl)
    return nil
}

// Delete 删除缓存
func (m *MockCache) Delete(ctx context.Context, key string) error {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    delete(m.data, key)
    delete(m.ttls, key)
    return nil
}

// Clear 清空缓存（测试用）
func (m *MockCache) Clear() {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.data = make(map[string][]byte)
    m.ttls = make(map[string]time.Time)
}

// Exists 检查是否存在（测试用）
func (m *MockCache) Exists(key string) bool {
    m.mu.RLock()
    defer m.mu.RUnlock()
    _, ok := m.data[key]
    return ok
}

var ErrCacheMiss = errors.New("cache miss")
```

---

## 3. tests/fixtures/test_data.go

```go
package fixtures

import (
    "time"
    
    "recommend-system/internal/model"
)

// 测试用户
var TestUsers = []*model.User{
    {
        ID:        "user_001",
        Name:      "Alice",
        Email:     "alice@example.com",
        Age:       25,
        Gender:    "female",
        CreatedAt: time.Now().Add(-30 * 24 * time.Hour),
        UpdatedAt: time.Now(),
    },
    {
        ID:        "user_002",
        Name:      "Bob",
        Email:     "bob@example.com",
        Age:       35,
        Gender:    "male",
        CreatedAt: time.Now().Add(-60 * 24 * time.Hour),
        UpdatedAt: time.Now(),
    },
    {
        ID:        "user_003",
        Name:      "Charlie",
        Email:     "charlie@example.com",
        Age:       45,
        Gender:    "male",
        CreatedAt: time.Now().Add(-90 * 24 * time.Hour),
        UpdatedAt: time.Now(),
    },
}

// 测试物品
var TestItems = []*model.Item{
    {
        ID:          "item_001",
        Type:        "movie",
        Title:       "The Matrix",
        Description: "A computer hacker learns the truth about reality",
        Category:    "action",
        Tags:        []string{"sci-fi", "action", "classic"},
        Status:      "active",
        CreatedAt:   time.Now().Add(-365 * 24 * time.Hour),
        UpdatedAt:   time.Now(),
    },
    {
        ID:          "item_002",
        Type:        "movie",
        Title:       "Inception",
        Description: "A thief who steals corporate secrets through dreams",
        Category:    "thriller",
        Tags:        []string{"sci-fi", "thriller", "mind-bending"},
        Status:      "active",
        CreatedAt:   time.Now().Add(-200 * 24 * time.Hour),
        UpdatedAt:   time.Now(),
    },
    {
        ID:          "item_003",
        Type:        "product",
        Title:       "iPhone 15",
        Description: "Latest Apple smartphone",
        Category:    "electronics",
        Tags:        []string{"phone", "apple", "premium"},
        Status:      "active",
        CreatedAt:   time.Now().Add(-30 * 24 * time.Hour),
        UpdatedAt:   time.Now(),
    },
}

// 测试行为
var TestBehaviors = []*model.UserBehavior{
    {
        UserID:    "user_001",
        ItemID:    "item_001",
        Action:    "view",
        Timestamp: time.Now().Add(-7 * 24 * time.Hour),
    },
    {
        UserID:    "user_001",
        ItemID:    "item_001",
        Action:    "like",
        Timestamp: time.Now().Add(-6 * 24 * time.Hour),
    },
    {
        UserID:    "user_001",
        ItemID:    "item_002",
        Action:    "view",
        Timestamp: time.Now().Add(-5 * 24 * time.Hour),
    },
}

// GetTestUser 获取测试用户
func GetTestUser(id string) *model.User {
    for _, u := range TestUsers {
        if u.ID == id {
            return u
        }
    }
    return nil
}

// GetTestItem 获取测试物品
func GetTestItem(id string) *model.Item {
    for _, i := range TestItems {
        if i.ID == id {
            return i
        }
    }
    return nil
}
```

---

## 4. tests/unit/service_test.go

```go
package unit

import (
    "context"
    "testing"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    
    "recommend-system/internal/service/user"
    "recommend-system/tests/mocks"
    "recommend-system/tests/fixtures"
)

func TestUserService_GetUser(t *testing.T) {
    // 准备
    mockRepo := mocks.NewMockUserRepo()
    mockCache := mocks.NewMockCache()
    
    testUser := fixtures.TestUsers[0]
    mockRepo.SetUser(testUser)
    
    service := user.NewUserService(mockRepo, mockCache, nil)
    
    // 执行
    result, err := service.GetUser(context.Background(), testUser.ID)
    
    // 验证
    require.NoError(t, err)
    assert.Equal(t, testUser.ID, result.ID)
    assert.Equal(t, testUser.Name, result.Name)
    assert.Equal(t, testUser.Email, result.Email)
}

func TestUserService_GetUser_NotFound(t *testing.T) {
    // 准备
    mockRepo := mocks.NewMockUserRepo()
    mockCache := mocks.NewMockCache()
    
    service := user.NewUserService(mockRepo, mockCache, nil)
    
    // 执行
    result, err := service.GetUser(context.Background(), "non_existent_user")
    
    // 验证
    assert.Error(t, err)
    assert.Nil(t, result)
}

func TestUserService_GetUser_FromCache(t *testing.T) {
    // 准备
    mockRepo := mocks.NewMockUserRepo()
    mockCache := mocks.NewMockCache()
    
    testUser := fixtures.TestUsers[0]
    
    // 先设置缓存
    _ = mockCache.Set(context.Background(), "user:"+testUser.ID, testUser, time.Hour)
    
    service := user.NewUserService(mockRepo, mockCache, nil)
    
    // 执行（应该从缓存获取，不调用 repo）
    result, err := service.GetUser(context.Background(), testUser.ID)
    
    // 验证
    require.NoError(t, err)
    assert.Equal(t, testUser.ID, result.ID)
}

func TestUserService_CreateUser(t *testing.T) {
    // 准备
    mockRepo := mocks.NewMockUserRepo()
    mockCache := mocks.NewMockCache()
    
    service := user.NewUserService(mockRepo, mockCache, nil)
    
    req := &user.CreateUserRequest{
        Name:  "Test User",
        Email: "test@example.com",
        Age:   30,
    }
    
    // 执行
    result, err := service.CreateUser(context.Background(), req)
    
    // 验证
    require.NoError(t, err)
    assert.NotEmpty(t, result.ID)
    assert.Equal(t, req.Name, result.Name)
    assert.Equal(t, req.Email, result.Email)
}

func TestUserService_UpdateUser(t *testing.T) {
    // 准备
    mockRepo := mocks.NewMockUserRepo()
    mockCache := mocks.NewMockCache()
    
    testUser := fixtures.TestUsers[0]
    mockRepo.SetUser(testUser)
    
    service := user.NewUserService(mockRepo, mockCache, nil)
    
    req := &user.UpdateUserRequest{
        Name: "Updated Name",
    }
    
    // 执行
    result, err := service.UpdateUser(context.Background(), testUser.ID, req)
    
    // 验证
    require.NoError(t, err)
    assert.Equal(t, "Updated Name", result.Name)
}

func TestUserService_RecordBehavior(t *testing.T) {
    // 准备
    mockRepo := mocks.NewMockUserRepo()
    mockCache := mocks.NewMockCache()
    
    testUser := fixtures.TestUsers[0]
    mockRepo.SetUser(testUser)
    
    service := user.NewUserService(mockRepo, mockCache, nil)
    
    req := &user.RecordBehaviorRequest{
        UserID: testUser.ID,
        ItemID: "item_001",
        Action: "click",
    }
    
    // 执行
    err := service.RecordBehavior(context.Background(), req)
    
    // 验证
    require.NoError(t, err)
    
    // 检查行为是否记录
    behaviors, err := mockRepo.GetBehaviors(context.Background(), testUser.ID, 10)
    require.NoError(t, err)
    assert.Len(t, behaviors, 1)
    assert.Equal(t, "click", behaviors[0].Action)
}
```

---

## 5. tests/integration/recommend_test.go

```go
package integration

import (
    "context"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    
    "recommend-system/internal/service/recommend"
    "recommend-system/tests/mocks"
    "recommend-system/tests/fixtures"
)

// 集成测试需要更完整的环境设置
func setupTestEnv(t *testing.T) (*recommend.RecommendService, func()) {
    // 创建 Mock 依赖
    mockUserRepo := mocks.NewMockUserRepo()
    mockItemRepo := mocks.NewMockItemRepo()
    mockCache := mocks.NewMockCache()
    
    // 加载测试数据
    for _, user := range fixtures.TestUsers {
        mockUserRepo.SetUser(user)
    }
    for _, user := range fixtures.TestUsers {
        mockUserRepo.SetBehaviors(user.ID, fixtures.GetBehaviorsForUser(user.ID))
    }
    for _, item := range fixtures.TestItems {
        mockItemRepo.SetItem(item)
    }
    
    // 创建服务
    service := recommend.NewRecommendService(
        mockUserRepo,
        mockItemRepo,
        mockCache,
        nil, // inference client
        nil, // logger
    )
    
    cleanup := func() {
        mockCache.Clear()
    }
    
    return service, cleanup
}

func TestRecommendService_GetRecommendations(t *testing.T) {
    service, cleanup := setupTestEnv(t)
    defer cleanup()
    
    ctx := context.Background()
    
    // 执行
    req := &recommend.RecommendRequest{
        UserID: "user_001",
        Limit:  10,
    }
    
    result, err := service.GetRecommendations(ctx, req)
    
    // 验证
    require.NoError(t, err)
    assert.NotEmpty(t, result.Recommendations)
    assert.LessOrEqual(t, len(result.Recommendations), 10)
}

func TestRecommendService_GetRecommendations_NewUser(t *testing.T) {
    service, cleanup := setupTestEnv(t)
    defer cleanup()
    
    ctx := context.Background()
    
    // 测试新用户（冷启动场景）
    req := &recommend.RecommendRequest{
        UserID: "new_user_001",
        Limit:  10,
    }
    
    result, err := service.GetRecommendations(ctx, req)
    
    // 验证：应该返回热门推荐或冷启动推荐
    require.NoError(t, err)
    assert.NotEmpty(t, result.Recommendations)
}

func TestRecommendService_GetSimilarItems(t *testing.T) {
    service, cleanup := setupTestEnv(t)
    defer cleanup()
    
    ctx := context.Background()
    
    // 执行
    result, err := service.GetSimilarItems(ctx, "item_001", 5)
    
    // 验证
    require.NoError(t, err)
    assert.NotEmpty(t, result)
}

func TestRecommendService_SubmitFeedback(t *testing.T) {
    service, cleanup := setupTestEnv(t)
    defer cleanup()
    
    ctx := context.Background()
    
    // 执行
    feedback := &recommend.Feedback{
        UserID: "user_001",
        ItemID: "item_001",
        Action: "like",
    }
    
    err := service.SubmitFeedback(ctx, feedback)
    
    // 验证
    require.NoError(t, err)
}
```

---

## 6. tests/benchmark/recommend_bench_test.go

```go
package benchmark

import (
    "context"
    "testing"
    
    "recommend-system/internal/service/recommend"
    "recommend-system/tests/mocks"
    "recommend-system/tests/fixtures"
)

func BenchmarkGetRecommendations(b *testing.B) {
    // 设置
    mockUserRepo := mocks.NewMockUserRepo()
    mockItemRepo := mocks.NewMockItemRepo()
    mockCache := mocks.NewMockCache()
    
    for _, user := range fixtures.TestUsers {
        mockUserRepo.SetUser(user)
    }
    for _, item := range fixtures.TestItems {
        mockItemRepo.SetItem(item)
    }
    
    service := recommend.NewRecommendService(
        mockUserRepo,
        mockItemRepo,
        mockCache,
        nil,
        nil,
    )
    
    ctx := context.Background()
    req := &recommend.RecommendRequest{
        UserID: "user_001",
        Limit:  20,
    }
    
    // 预热
    for i := 0; i < 100; i++ {
        _, _ = service.GetRecommendations(ctx, req)
    }
    
    // 基准测试
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, _ = service.GetRecommendations(ctx, req)
    }
}

func BenchmarkGetRecommendations_Parallel(b *testing.B) {
    // 设置
    mockUserRepo := mocks.NewMockUserRepo()
    mockItemRepo := mocks.NewMockItemRepo()
    mockCache := mocks.NewMockCache()
    
    for _, user := range fixtures.TestUsers {
        mockUserRepo.SetUser(user)
    }
    for _, item := range fixtures.TestItems {
        mockItemRepo.SetItem(item)
    }
    
    service := recommend.NewRecommendService(
        mockUserRepo,
        mockItemRepo,
        mockCache,
        nil,
        nil,
    )
    
    ctx := context.Background()
    
    // 并行基准测试
    b.RunParallel(func(pb *testing.PB) {
        req := &recommend.RecommendRequest{
            UserID: "user_001",
            Limit:  20,
        }
        
        for pb.Next() {
            _, _ = service.GetRecommendations(ctx, req)
        }
    })
}

func BenchmarkCacheGet(b *testing.B) {
    cache := mocks.NewMockCache()
    ctx := context.Background()
    
    // 预设数据
    testData := map[string]string{"key": "value"}
    _ = cache.Set(ctx, "test_key", testData, time.Hour)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        var result map[string]string
        _ = cache.Get(ctx, "test_key", &result)
    }
}
```

---

## 7. Makefile 测试命令

```makefile
# 运行所有测试
test:
	go test ./tests/... -v

# 运行单元测试
test-unit:
	go test ./tests/unit/... -v

# 运行集成测试
test-integration:
	go test ./tests/integration/... -v

# 运行性能测试
test-benchmark:
	go test ./tests/benchmark/... -bench=. -benchmem

# 测试覆盖率
test-coverage:
	go test ./... -coverprofile=coverage.out
	go tool cover -html=coverage.out -o coverage.html

# 测试并生成报告
test-report:
	go test ./... -v -json > test-report.json
```

---

## 测试标准

| 指标 | 目标 |
|------|------|
| 单元测试覆盖率 | >= 80% |
| API 响应时间 | <= 200ms |
| 推荐接口 P99 | <= 50ms |
| 并发测试 | 1000 QPS |

## 输出要求

请输出完整的测试代码，包含：
1. 所有测试文件
2. Mock 实现
3. 测试数据
4. 性能测试
5. 测试脚本

