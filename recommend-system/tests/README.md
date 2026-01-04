# 生成式推荐系统测试套件

> Person F: Testing - 完整测试套件文档

本文档介绍生成式推荐系统的完整测试套件，包括测试架构、Mock 实现、测试数据、使用方法和最佳实践。

---

## 目录

1. [测试架构概览](#测试架构概览)
2. [目录结构](#目录结构)
3. [Mock 实现说明](#mock-实现说明)
4. [测试数据 (Fixtures)](#测试数据-fixtures)
5. [单元测试](#单元测试)
6. [集成测试](#集成测试)
7. [性能测试](#性能测试)
8. [运行测试](#运行测试)
9. [测试覆盖率](#测试覆盖率)
10. [最佳实践](#最佳实践)
11. [扩展指南](#扩展指南)

---

## 测试架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           测试套件架构                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         tests/ 根目录                                │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │   │
│  │  │   mocks/  │  │ fixtures/ │  │   unit/   │  │integration│        │   │
│  │  │  Mock实现  │  │  测试数据  │  │  单元测试  │  │  集成测试  │        │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │   │
│  │                                                  ┌───────────┐      │   │
│  │                                                  │ benchmark │      │   │
│  │                                                  │  性能测试  │      │   │
│  │                                                  └───────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  依赖关系:                                                                   │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │  unit/integration/benchmark  ──依赖──►  mocks/  ──依赖──►  fixtures/ │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 设计原则

1. **接口驱动**: 所有 Mock 实现 `interfaces` 包中定义的接口
2. **数据隔离**: 测试数据通过 fixtures 包统一管理，避免硬编码
3. **并发安全**: 所有 Mock 实现支持并发访问
4. **可观测性**: Mock 提供调用计数和错误注入能力

---

## 目录结构

```
recommend-system/tests/
├── README.md                    # 本文档
├── mocks/                       # Mock 实现
│   ├── errors.go               # 通用错误定义
│   ├── user_repo_mock.go       # 用户仓库 Mock
│   ├── item_repo_mock.go       # 物品仓库 Mock
│   ├── cache_mock.go           # 缓存 Mock
│   └── service_mocks.go        # 服务层 Mock（用户、物品、特征、冷启动、LLM、推理）
├── fixtures/                    # 测试数据
│   └── test_data.go            # 预定义测试数据
├── unit/                        # 单元测试
│   ├── service_test.go         # 服务层单元测试
│   ├── repository_test.go      # 数据访问层测试
│   └── handler_test.go         # API 处理器测试
├── integration/                 # 集成测试
│   ├── recommend_test.go       # 推荐服务集成测试
│   ├── user_test.go            # 用户服务集成测试
│   └── item_test.go            # 物品服务集成测试
└── benchmark/                   # 性能测试
    └── recommend_bench_test.go # 推荐服务性能测试
```

---

## Mock 实现说明

### 概述

Mock 实现位于 `tests/mocks/` 目录，实现了 `internal/interfaces/interfaces.go` 中定义的所有接口。

### 可用的 Mock 类型

| Mock 类型 | 实现接口 | 说明 |
|----------|---------|------|
| `MockUserRepository` | `interfaces.UserRepository` | 用户数据访问层 Mock |
| `MockItemRepository` | `interfaces.ItemRepository` | 物品数据访问层 Mock |
| `MockCache` | `interfaces.Cache` | 缓存服务 Mock |
| `MockUserService` | `interfaces.UserService` | 用户服务 Mock |
| `MockItemService` | `interfaces.ItemService` | 物品服务 Mock |
| `MockFeatureService` | `interfaces.FeatureService` | 特征服务 Mock |
| `MockColdStartService` | `interfaces.ColdStartService` | 冷启动服务 Mock |
| `MockLLMClient` | `interfaces.LLMClient` | LLM 客户端 Mock |
| `MockInferenceClient` | `interfaces.InferenceClient` | 推理客户端 Mock |
| `MockRecommendRepository` | `interfaces.RecommendRepository` | 推荐仓库 Mock |

### Mock 使用示例

```go
package example

import (
    "context"
    "testing"
    
    "recommend-system/tests/mocks"
    "recommend-system/tests/fixtures"
)

func TestExample(t *testing.T) {
    // 1. 创建 Mock 实例
    mockRepo := mocks.NewMockUserRepository()
    mockCache := mocks.NewMockCache()
    
    // 2. 设置测试数据
    testUser := fixtures.GetTestUser("user_001")
    mockRepo.SetUser(testUser)
    
    // 3. 执行测试
    ctx := context.Background()
    user, err := mockRepo.GetByID(ctx, "user_001")
    
    // 4. 验证结果
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    if user.ID != "user_001" {
        t.Errorf("expected user_001, got %s", user.ID)
    }
    
    // 5. 验证调用计数
    if mockRepo.GetByIDCalls != 1 {
        t.Errorf("expected 1 call, got %d", mockRepo.GetByIDCalls)
    }
}
```

### 错误注入

所有 Mock 支持错误注入，用于测试错误处理逻辑：

```go
// 设置获取用户时返回错误
mockRepo.GetByIDError = mocks.ErrNotFound

// 设置缓存服务不可用
mockCache.SetError = mocks.ErrServiceUnavailable

// 设置推理服务超时
mockInfer.InferError = mocks.ErrTimeout
```

### 调用计数

每个 Mock 方法都有对应的调用计数器：

```go
mockRepo := mocks.NewMockUserRepository()

// 执行操作
mockRepo.GetByID(ctx, "user_001")
mockRepo.GetByID(ctx, "user_002")
mockRepo.Create(ctx, user)

// 验证调用次数
assert.Equal(t, 2, mockRepo.GetByIDCalls)
assert.Equal(t, 1, mockRepo.CreateCalls)
```

---

## 测试数据 (Fixtures)

### 概述

`fixtures` 包提供预定义的测试数据，确保测试的一致性和可重复性。

### 可用数据

| 数据类型 | 变量/函数 | 说明 |
|---------|----------|------|
| 用户 | `TestUsers` | 5 个预定义测试用户 |
| 物品 | `TestItems` | 9 个预定义测试物品 |
| 物品统计 | `TestItemStats` | 5 个物品统计数据 |
| 用户行为 | `TestBehaviors` | 12 条用户行为记录 |
| 用户画像 | `TestUserProfiles` | 2 个用户画像 |
| 用户特征 | `TestUserFeatures` | 2 个用户特征 |
| 物品特征 | `TestItemFeatures` | 2 个物品特征 |

### 使用示例

```go
import "recommend-system/tests/fixtures"

// 获取单个测试用户
user := fixtures.GetTestUser("user_001")

// 获取所有测试用户
users := fixtures.GetAllTestUsers()

// 获取用户的行为列表
behaviors := fixtures.GetBehaviorsForUser("user_001")

// 获取测试物品
item := fixtures.GetTestItem("item_001")

// 按类型获取物品
movies := fixtures.GetTestItemsByType("movie")

// 创建自定义测试数据
customUser := fixtures.CreateTestUser("custom_id", "Custom User", "custom@test.com", 30, "male")
customItem := fixtures.CreateTestItem("custom_item", "movie", "Custom Movie", "action", []string{"test"})
customBehavior := fixtures.CreateTestBehavior("user_001", "item_001", "click")
```

### 测试用户说明

| 用户 ID | 名称 | 特点 |
|--------|------|------|
| user_001 | Alice | 活跃用户，有丰富的行为历史 |
| user_002 | Bob | 普通用户，行为记录适中 |
| user_003 | Charlie | 老用户，注册时间较长 |
| user_004 | Diana | 新注册用户 |
| user_new | NewUser | 冷启动测试用户，无历史行为 |

### 测试物品说明

| 物品 ID | 类型 | 类目 | 说明 |
|--------|------|------|------|
| item_001 | movie | action | 经典科幻电影 (The Matrix) |
| item_002 | movie | thriller | 科幻惊悚电影 (Inception) |
| item_003 | product | electronics | 手机商品 (iPhone 15) |
| item_004 | video | education | 编程教程视频 |
| item_005 | article | technology | 机器学习文章 |
| item_006 | movie | drama | 经典剧情电影 |
| item_007 | product | electronics | 笔记本商品 |
| item_008 | video | education | Kubernetes 教程 |
| item_inactive | movie | unknown | 已下线物品 |

---

## 单元测试

### 位置

`tests/unit/`

### 测试文件

#### service_test.go

测试服务层业务逻辑：

- `TestUserService_GetUser` - 获取用户
- `TestUserService_GetUser_NotFound` - 获取不存在的用户
- `TestUserService_GetUser_FromCache` - 从缓存获取用户
- `TestUserService_CreateUser` - 创建用户
- `TestUserService_UpdateUser` - 更新用户
- `TestUserService_DeleteUser` - 删除用户
- `TestUserService_RecordBehavior` - 记录用户行为
- `TestUserService_GetUserBehaviors` - 获取用户行为历史
- `TestUserService_GetUserProfile` - 获取用户画像
- `TestUserService_ConcurrentAccess` - 并发访问测试
- `TestUserService_ErrorPropagation` - 错误传播测试

#### repository_test.go

测试 Mock 仓库功能：

- 用户仓库 CRUD 操作测试
- 物品仓库 CRUD 操作测试
- 缓存操作测试
- 推荐仓库测试

#### handler_test.go

测试 API 处理器：

- HTTP 请求/响应测试
- 请求验证测试
- 中间件测试
- 错误响应测试

### 运行单元测试

```bash
# 运行所有单元测试
go test ./tests/unit/... -v

# 运行特定测试
go test ./tests/unit/... -v -run TestUserService_GetUser

# 显示覆盖率
go test ./tests/unit/... -v -cover
```

---

## 集成测试

### 位置

`tests/integration/`

### 测试文件

#### recommend_test.go

推荐服务集成测试：

- `TestRecommendIntegration_NormalUser` - 正常用户推荐流程
- `TestRecommendIntegration_ColdStartUser` - 冷启动用户推荐
- `TestRecommendIntegration_WithBehaviorRecording` - 推荐后行为记录
- `TestRecommendIntegration_ExposureFiltering` - 曝光过滤
- `TestRecommendIntegration_CacheInteraction` - 缓存交互
- `TestRecommendIntegration_EndToEnd` - 端到端推荐流程
- `TestRecommendIntegration_Concurrent` - 并发推荐请求
- `TestRecommendIntegration_InferenceFailure` - 推理失败恢复

#### user_test.go

用户服务集成测试：

- `TestUserIntegration_CRUD` - 用户 CRUD 完整流程
- `TestUserIntegration_BehaviorTracking` - 行为跟踪
- `TestUserIntegration_ProfileGeneration` - 画像生成
- `TestUserIntegration_CacheConsistency` - 缓存一致性
- `TestUserIntegration_ConcurrentOperations` - 并发操作
- `TestUserIntegration_DataIntegrity` - 数据完整性

#### item_test.go

物品服务集成测试：

- `TestItemIntegration_CRUD` - 物品 CRUD 完整流程
- `TestItemIntegration_BatchOperations` - 批量操作
- `TestItemIntegration_ListAndPagination` - 列表和分页
- `TestItemIntegration_Search` - 搜索功能
- `TestItemIntegration_Statistics` - 统计功能
- `TestItemIntegration_PopularItems` - 热门物品

### 运行集成测试

```bash
# 运行所有集成测试
go test ./tests/integration/... -v

# 运行推荐服务集成测试
go test ./tests/integration/... -v -run TestRecommendIntegration

# 运行用户服务集成测试
go test ./tests/integration/... -v -run TestUserIntegration
```

---

## 性能测试

### 位置

`tests/benchmark/`

### 基准测试

#### recommend_bench_test.go

| 测试名称 | 说明 |
|---------|------|
| `BenchmarkGetRecommendations` | 推荐接口性能 |
| `BenchmarkGetRecommendations_Parallel` | 并发推荐性能 |
| `BenchmarkCacheGet` | 缓存读取性能 |
| `BenchmarkCacheSet` | 缓存写入性能 |
| `BenchmarkCacheMGet` | 批量缓存读取 |
| `BenchmarkCacheMSet` | 批量缓存写入 |
| `BenchmarkUserRepoGetByID` | 用户获取性能 |
| `BenchmarkUserRepoGetBehaviors` | 行为获取性能 |
| `BenchmarkItemRepoGetByID` | 物品获取性能 |
| `BenchmarkItemRepoSearch` | 物品搜索性能 |
| `BenchmarkInference` | 推理性能 |
| `BenchmarkBatchInference` | 批量推理性能 |
| `BenchmarkEndToEndRecommendation` | 端到端推荐性能 |

### 运行性能测试

```bash
# 运行所有基准测试
go test ./tests/benchmark/... -bench=. -benchmem

# 运行特定基准测试
go test ./tests/benchmark/... -bench=BenchmarkGetRecommendations -benchmem

# 多次运行取平均值
go test ./tests/benchmark/... -bench=. -benchmem -count=5

# 输出 CPU profile
go test ./tests/benchmark/... -bench=. -cpuprofile=cpu.prof

# 输出内存 profile
go test ./tests/benchmark/... -bench=. -memprofile=mem.prof
```

### 性能指标解读

```
BenchmarkGetRecommendations-8    1000000    1234 ns/op    256 B/op    4 allocs/op
```

- `8`: 使用的 CPU 核心数
- `1000000`: 运行次数
- `1234 ns/op`: 每次操作耗时（纳秒）
- `256 B/op`: 每次操作分配的内存
- `4 allocs/op`: 每次操作的内存分配次数

---

## 运行测试

### Makefile 命令

在项目根目录的 Makefile 中添加以下命令：

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

### 快速命令

```bash
# 从项目根目录运行
cd recommend-system

# 运行所有测试
go test ./tests/... -v

# 快速验证（不显示详细输出）
go test ./tests/...

# 并行运行测试
go test ./tests/... -v -parallel 4

# 超时设置
go test ./tests/... -v -timeout 30s
```

---

## 测试覆盖率

### 生成覆盖率报告

```bash
# 生成覆盖率文件
go test ./... -coverprofile=coverage.out

# 查看覆盖率摘要
go tool cover -func=coverage.out

# 生成 HTML 报告
go tool cover -html=coverage.out -o coverage.html

# 在浏览器中打开
open coverage.html  # macOS
# 或
start coverage.html  # Windows
```

### 覆盖率目标

| 指标 | 目标 |
|------|------|
| 单元测试覆盖率 | >= 80% |
| 关键路径覆盖率 | >= 90% |
| 分支覆盖率 | >= 75% |

---

## 最佳实践

### 1. 测试命名

遵循 `Test<功能>_<场景>` 格式：

```go
// 好的命名
func TestUserService_GetUser(t *testing.T) {}
func TestUserService_GetUser_NotFound(t *testing.T) {}
func TestUserService_GetUser_FromCache(t *testing.T) {}

// 避免的命名
func TestGetUser(t *testing.T) {}
func Test1(t *testing.T) {}
```

### 2. 测试结构

使用 Arrange-Act-Assert (AAA) 模式：

```go
func TestExample(t *testing.T) {
    // Arrange - 准备
    mockRepo := mocks.NewMockUserRepository()
    testUser := fixtures.GetTestUser("user_001")
    mockRepo.SetUser(testUser)
    
    service := user.NewService(mockRepo, nil, nil)
    
    // Act - 执行
    result, err := service.GetUser(context.Background(), testUser.ID)
    
    // Assert - 验证
    require.NoError(t, err)
    assert.Equal(t, testUser.ID, result.ID)
}
```

### 3. 表驱动测试

对于多个相似场景，使用表驱动测试：

```go
func TestValidation(t *testing.T) {
    testCases := []struct {
        name    string
        input   *Request
        wantErr bool
    }{
        {"valid request", &Request{Name: "Test"}, false},
        {"empty name", &Request{Name: ""}, true},
        {"nil request", nil, true},
    }
    
    for _, tc := range testCases {
        t.Run(tc.name, func(t *testing.T) {
            err := Validate(tc.input)
            if tc.wantErr {
                assert.Error(t, err)
            } else {
                assert.NoError(t, err)
            }
        })
    }
}
```

### 4. 清理测试数据

使用 defer 确保测试后清理：

```go
func TestWithCleanup(t *testing.T) {
    env, cleanup := setupTestEnv(t)
    defer cleanup()
    
    // 测试代码...
}
```

### 5. 并发测试

测试并发安全性：

```go
func TestConcurrentSafety(t *testing.T) {
    const numGoroutines = 100
    var wg sync.WaitGroup
    
    for i := 0; i < numGoroutines; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            // 并发操作...
        }()
    }
    
    wg.Wait()
}
```

---

## 扩展指南

### 添加新的 Mock

1. 在 `mocks/` 目录创建新文件或在现有文件中添加
2. 实现对应的接口
3. 添加调用计数器和错误注入能力
4. 添加辅助方法（SetXxx, Reset 等）

```go
// mocks/new_service_mock.go
type MockNewService struct {
    mu sync.RWMutex
    
    // 调用计数器
    MethodCalls int
    
    // 可配置的错误
    MethodError error
    
    // 可配置的返回值
    MethodResult *SomeType
}

func NewMockNewService() *MockNewService {
    return &MockNewService{}
}

func (m *MockNewService) Method(ctx context.Context, param string) (*SomeType, error) {
    m.mu.Lock()
    m.MethodCalls++
    m.mu.Unlock()
    
    if m.MethodError != nil {
        return nil, m.MethodError
    }
    
    return m.MethodResult, nil
}

func (m *MockNewService) Reset() {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.MethodCalls = 0
    m.MethodError = nil
    m.MethodResult = nil
}
```

### 添加新的测试数据

在 `fixtures/test_data.go` 中添加：

```go
// 新增测试数据
var TestNewEntities = []*NewEntity{
    {ID: "new_001", ...},
    {ID: "new_002", ...},
}

// 获取函数
func GetTestNewEntity(id string) *NewEntity {
    for _, e := range TestNewEntities {
        if e.ID == id {
            copy := *e
            return &copy
        }
    }
    return nil
}
```

### 添加新的测试文件

1. 选择合适的目录（unit/integration/benchmark）
2. 创建测试文件，命名为 `xxx_test.go`
3. 导入必要的包
4. 编写测试函数

```go
// tests/unit/new_service_test.go
package unit

import (
    "context"
    "testing"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    
    "recommend-system/tests/fixtures"
    "recommend-system/tests/mocks"
)

func TestNewService_Method(t *testing.T) {
    // 测试代码...
}
```

---

## 常见问题

### Q: 测试超时怎么办？

A: 增加超时时间或优化测试代码：
```bash
go test ./... -timeout 60s
```

### Q: 测试失败但日志太多？

A: 使用 `-short` 跳过长时间测试，或使用 `-run` 指定测试：
```bash
go test ./... -short
go test ./... -run TestSpecificFunction
```

### Q: 如何调试失败的测试？

A: 使用 `-v` 显示详细输出，或使用 IDE 的调试功能：
```bash
go test ./... -v -run TestFailingTest
```

### Q: Mock 没有按预期工作？

A: 检查：
1. 是否正确设置了预期返回值
2. 是否设置了错误注入
3. 调用计数器是否正确

---

## 联系方式

如有问题，请联系项目维护者或在 Issue 中反馈。

---

*文档版本: 1.0.0*
*最后更新: 2025-01-04*

