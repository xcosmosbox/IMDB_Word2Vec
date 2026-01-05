# 前端基础设施文档

> Person F 开发 - 生成式推荐系统前端基础设施

## 目录

1. [项目概览](#项目概览)
2. [目录结构](#目录结构)
3. [API 模块](#api-模块)
4. [工具函数](#工具函数)
5. [状态管理](#状态管理)
6. [路由配置](#路由配置)
7. [通用组件](#通用组件)
8. [项目配置](#项目配置)
9. [单元测试](#单元测试)
10. [开发指南](#开发指南)

---

## 项目概览

本基础设施为生成式推荐系统前端提供核心功能支持，包括：

- **API 封装** - 统一的 HTTP 请求和响应处理
- **Provider 模式** - 可插拔的 API 服务实现（HTTP/Mock）
- **工具函数** - 格式化、存储、验证等通用工具
- **状态管理** - Pinia Store 配置和初始化
- **路由系统** - Vue Router 配置和导航守卫
- **通用组件** - 可复用的 UI 组件库

### 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Vue 3 | ^3.4 | 核心框架 |
| Vite | ^5.0 | 构建工具 |
| Vue Router | ^4.2 | 路由 |
| Pinia | ^2.1 | 状态管理 |
| Axios | ^1.6 | HTTP 客户端 |
| TypeScript | ^5.3 | 类型安全 |
| Vitest | ^1.2 | 单元测试 |

---

## 目录结构

```
frontend/
├── shared/                      # 共享代码
│   ├── api/                     # API 封装
│   │   ├── index.ts            # 统一导出
│   │   ├── interfaces.ts       # 接口定义（已提供）
│   │   ├── request.ts          # Axios 封装
│   │   ├── provider.ts         # API Provider（HTTP/Mock）
│   │   ├── auth.ts             # 认证 API
│   │   ├── user.ts             # 用户 API
│   │   ├── item.ts             # 物品 API
│   │   ├── recommend.ts        # 推荐 API
│   │   ├── analytics.ts        # 分析 API
│   │   └── admin.ts            # 管理员 API
│   ├── types/                   # 类型定义（已提供）
│   │   └── index.ts
│   ├── utils/                   # 工具函数
│   │   ├── index.ts            # 统一导出
│   │   ├── storage.ts          # 本地存储
│   │   └── format.ts           # 格式化函数
│   └── INFRASTRUCTURE.md       # 本文档
│
└── user-app/                    # 用户端应用
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    ├── vitest.config.ts
    ├── index.html
    └── src/
        ├── main.ts             # 应用入口
        ├── App.vue             # 根组件
        ├── router/
        │   └── index.ts        # 路由配置
        ├── stores/
        │   ├── index.ts        # Store 导出
        │   ├── user.ts         # 用户 Store
        │   ├── recommend.ts    # 推荐 Store
        │   └── item.ts         # 物品 Store
        ├── components/
        │   └── common/
        │       ├── index.ts
        │       ├── Loading.vue
        │       ├── Empty.vue
        │       ├── Error.vue
        │       ├── Modal.vue
        │       ├── Toast.vue
        │       ├── Button.vue
        │       ├── Input.vue
        │       └── Skeleton.vue
        ├── assets/
        │   └── styles/
        │       └── main.css
        └── __tests__/          # 单元测试
```

---

## API 模块

### 接口设计

所有 API 服务遵循 `interfaces.ts` 中定义的接口，确保可插拔设计。

```typescript
// 核心接口
interface IApiProvider {
  readonly auth: IAuthService
  readonly user: IUserService
  readonly item: IItemService
  readonly recommend: IRecommendService
  readonly analytics: IAnalyticsService
  readonly adminUser: IAdminUserService
  readonly adminItem: IAdminItemService
}
```

### 使用方式

#### 方式一：Vue 插件（推荐）

```typescript
// main.ts
import { createApp } from 'vue'
import { apiPlugin } from '@shared/api'

const app = createApp(App)
app.use(apiPlugin, { mock: import.meta.env.DEV })
```

```vue
<!-- 组件中使用 -->
<script setup lang="ts">
import { useApi } from '@shared/api'

const api = useApi()
const user = await api.user.getUser('user_001')
</script>
```

#### 方式二：直接导入

```typescript
import { userApi, recommendApi } from '@shared/api'

const user = await userApi.getUser('user_001')
const recs = await recommendApi.getRecommendations({ user_id: 'user_001' })
```

#### 方式三：Provider 实例

```typescript
import { HttpApiProvider, MockApiProvider } from '@shared/api'

// 生产环境
const api = new HttpApiProvider()

// 开发环境
const api = new MockApiProvider()

const items = await api.item.searchItems('推荐')
```

### Axios 封装

`request.ts` 提供了完整的 HTTP 请求封装：

- **自动 Token 注入** - 请求拦截器自动添加 Authorization 头
- **请求追踪** - 自动添加 X-Request-ID 用于日志追踪
- **统一错误处理** - 响应拦截器处理业务错误和 HTTP 错误
- **Token 过期处理** - 401 错误自动清除 token 并跳转登录

```typescript
import { request, getToken, setToken, removeToken } from '@shared/api'

// GET 请求
const user = await request.get<User>('/users/123')

// POST 请求
const result = await request.post<LoginResponse>('/auth/login', { email, password })

// Token 操作
setToken('new-token')
const token = getToken()
removeToken()
```

### Mock 数据

`MockApiProvider` 提供完整的模拟数据实现，用于开发和测试：

```typescript
// 自动延迟模拟网络请求
// 包含预置的用户、物品、推荐数据
// 支持基本的 CRUD 操作

const api = new MockApiProvider()
const users = await api.adminUser.listUsers({ page: 1, page_size: 10 })
```

---

## 工具函数

### 格式化函数

```typescript
import {
  formatDate,
  formatRelativeTime,
  formatNumber,
  formatLargeNumber,
  formatPercent,
  formatMoney,
  formatFileSize,
  truncate,
  maskPhone,
  maskEmail,
} from '@shared/utils'

// 日期格式化
formatDate(new Date(), 'YYYY-MM-DD HH:mm')  // '2024-01-15 10:30'
formatRelativeTime(date)  // '5 分钟前'

// 数字格式化
formatNumber(1234567)       // '1,234,567'
formatLargeNumber(12345)    // '1.2万'
formatPercent(0.125)        // '12.50%'
formatMoney(1234.5)         // '¥1,234.50'
formatFileSize(1048576)     // '1 MB'

// 文本处理
truncate('很长的文本...', 10)   // '很长的文本...'
maskPhone('13812345678')       // '138****5678'
maskEmail('test@example.com')  // 'te***@example.com'
```

### 存储工具

```typescript
import {
  storage,
  sessionStore,
  saveAuthToken,
  getAuthToken,
  clearAuth,
} from '@shared/utils'

// localStorage 操作
storage.set('key', { data: 'value' })
storage.set('temp', 'data', 3600000)  // 1小时过期
const data = storage.get('key')
storage.remove('key')

// sessionStorage 操作
sessionStore.set('session', 'data')
const session = sessionStore.get('session')

// Token 存储
saveAuthToken('token', 7 * 24 * 60 * 60 * 1000)  // 7天
const token = getAuthToken()
clearAuth()  // 清除所有认证数据
```

### 验证工具

```typescript
import {
  isValidEmail,
  isValidPhone,
  validatePassword,
  isValidUrl,
} from '@shared/utils'

isValidEmail('test@example.com')  // true
isValidPhone('13812345678')       // true
validatePassword('Abc12345')      // { valid: true, message: '...' }
isValidUrl('https://example.com') // true
```

### 防抖与节流

```typescript
import { debounce, throttle, sleep, retry } from '@shared/utils'

// 防抖
const debouncedSearch = debounce(search, 300)
input.addEventListener('input', debouncedSearch)

// 节流
const throttledScroll = throttle(handleScroll, 100)
window.addEventListener('scroll', throttledScroll)

// 延迟
await sleep(1000)

// 重试
const result = await retry(() => fetchData(), { maxAttempts: 3, delay: 1000 })
```

### 对象和数组工具

```typescript
import {
  deepClone,
  deepMerge,
  pick,
  omit,
  unique,
  groupBy,
  chunk,
  generateId,
  uuid,
} from '@shared/utils'

// 深拷贝
const cloned = deepClone(obj)

// 选取/排除属性
const picked = pick(obj, ['a', 'b'])
const omitted = omit(obj, ['c'])

// 数组操作
const uniqueItems = unique(items, 'id')
const grouped = groupBy(items, 'type')
const chunks = chunk(items, 10)

// ID 生成
const id = generateId(12)
const uid = uuid()
```

---

## 状态管理

### Store 结构

```
stores/
├── index.ts        # 统一导出和初始化
├── user.ts         # 用户状态（Person C 实现）
├── recommend.ts    # 推荐状态
└── item.ts         # 物品状态
```

### 初始化

```typescript
// main.ts
import { createPinia } from 'pinia'
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate'
import { initStores } from '@/stores'

const pinia = createPinia()
pinia.use(piniaPluginPersistedstate)
app.use(pinia)

// 初始化所有 Store
await initStores(api)
```

### 使用示例

```vue
<script setup lang="ts">
import { useUserStore, useRecommendStore, useItemStore } from '@/stores'

const userStore = useUserStore()
const recommendStore = useRecommendStore()
const itemStore = useItemStore()

// 登录
await userStore.login({ email, password })

// 获取推荐
await recommendStore.getRecommendations({ limit: 20 })

// 搜索物品
await itemStore.searchItems('推荐')
</script>
```

---

## 路由配置

### 路由定义

```typescript
// router/index.ts
const routes = [
  { path: '/', name: 'Home', component: () => import('@/views/Home.vue') },
  { path: '/search', name: 'Search', component: () => import('@/views/Search.vue') },
  { path: '/item/:id', name: 'ItemDetail', component: () => import('@/views/ItemDetail.vue') },
  { path: '/login', name: 'Login', component: () => import('@/views/Login.vue'), meta: { guest: true } },
  { path: '/profile', name: 'Profile', component: () => import('@/views/Profile.vue'), meta: { requiresAuth: true } },
  // ...
]
```

### 路由元信息

```typescript
declare module 'vue-router' {
  interface RouteMeta {
    title?: string          // 页面标题
    requiresAuth?: boolean  // 需要登录
    guest?: boolean         // 仅游客访问
    keepAlive?: boolean     // 缓存页面
    transition?: string     // 过渡动画
  }
}
```

### 导航守卫

- **登录检查** - 需要认证的页面自动跳转登录
- **游客限制** - 登录后不能访问登录/注册页
- **页面标题** - 自动设置 document.title
- **错误处理** - 动态导入失败自动提示刷新

### 路由辅助函数

```typescript
import { routerHelper } from '@/router'

routerHelper.goHome()
routerHelper.goLogin('/profile')  // 带 redirect
routerHelper.goItemDetail('item_001')
routerHelper.goSearch('推荐')
routerHelper.goBack()
```

---

## 通用组件

### Loading 组件

```vue
<template>
  <Loading tip="正在加载..." />
  <Loading size="large" fullscreen />
  <Loading size="small" color="#ff0000" />
</template>
```

### Empty 组件

```vue
<template>
  <Empty description="暂无数据">
    <Button @click="handleRefresh">刷新</Button>
  </Empty>
</template>
```

### Error 组件

```vue
<template>
  <Error
    type="network"
    title="网络错误"
    description="请检查网络连接"
    @retry="handleRetry"
    @back="handleBack"
  />
</template>
```

### Modal 组件

```vue
<template>
  <Modal
    v-model="visible"
    title="确认操作"
    @confirm="handleConfirm"
    @cancel="handleCancel"
  >
    <p>确定要执行此操作吗？</p>
  </Modal>
</template>
```

### Toast 组件

```vue
<template>
  <Toast
    v-model="showToast"
    type="success"
    message="操作成功"
    :duration="3000"
  />
</template>
```

### Button 组件

```vue
<template>
  <Button type="primary" @click="handleClick">主要按钮</Button>
  <Button type="secondary">次要按钮</Button>
  <Button type="outline" size="small">小按钮</Button>
  <Button type="danger" :loading="loading">删除</Button>
  <Button block>块级按钮</Button>
</template>
```

### Input 组件

```vue
<template>
  <Input
    v-model="value"
    placeholder="请输入"
    clearable
    :error="errorMessage"
  />
  <Input
    v-model="password"
    type="password"
    show-password
  />
  <Input
    v-model="content"
    :maxlength="100"
    show-count
  />
</template>
```

### Skeleton 组件

```vue
<template>
  <Skeleton type="text" :rows="3" />
  <Skeleton type="card" />
  <Skeleton type="list" :rows="5" />
  <Skeleton type="image" height="200px" />
</template>
```

---

## 项目配置

### Vite 配置

```typescript
// vite.config.ts
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@shared': resolve(__dirname, '../shared'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
})
```

### 环境变量

```bash
# .env - 开发环境
VITE_API_BASE_URL=/api/v1
VITE_USE_MOCK=true

# .env.production - 生产环境
VITE_API_BASE_URL=/api/v1
VITE_USE_MOCK=false
```

### TypeScript 配置

- 严格模式启用
- 路径别名配置（@, @shared）
- Vue 文件类型支持

---

## 单元测试

### 运行测试

```bash
# 运行所有测试
npm test

# 运行测试并显示 UI
npm run test:ui

# 生成覆盖率报告
npm run test:coverage
```

### 测试文件结构

```
src/__tests__/
├── setup.ts              # 测试设置
├── api/
│   └── provider.test.ts  # API Provider 测试
├── utils/
│   ├── format.test.ts    # 格式化函数测试
│   ├── storage.test.ts   # 存储函数测试
│   └── index.test.ts     # 通用工具测试
├── components/
│   ├── Loading.test.ts   # Loading 组件测试
│   └── Button.test.ts    # Button 组件测试
└── stores/
    └── recommend.test.ts # Recommend Store 测试
```

### 编写测试

```typescript
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import Loading from '@/components/common/Loading.vue'

describe('Loading 组件', () => {
  it('应该正确渲染', () => {
    const wrapper = mount(Loading)
    expect(wrapper.find('.loading').exists()).toBe(true)
  })

  it('应该显示提示文本', () => {
    const wrapper = mount(Loading, {
      props: { tip: '加载中...' },
    })
    expect(wrapper.find('.loading__tip').text()).toBe('加载中...')
  })
})
```

---

## 开发指南

### 快速开始

```bash
# 进入用户端目录
cd frontend/user-app

# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 构建生产版本
npm run build
```

### 开发规范

1. **类型安全** - 所有代码必须使用 TypeScript
2. **接口驱动** - API 调用必须通过接口进行
3. **组件隔离** - 使用 scoped 样式避免污染
4. **错误处理** - API 调用必须有 try-catch
5. **响应式** - 所有页面需支持移动端

### 添加新 API

1. 在 `interfaces.ts` 中定义接口（如果需要）
2. 在对应的服务文件中实现接口
3. 在 `provider.ts` 中添加 Mock 实现
4. 在 `index.ts` 中导出
5. 编写单元测试

### 添加新组件

1. 在 `components/common/` 下创建 Vue 文件
2. 使用 Composition API + TypeScript
3. 在 `components/common/index.ts` 中导出
4. 编写单元测试

### 添加新工具函数

1. 在对应的工具文件中添加函数
2. 在 `utils/index.ts` 中导出
3. 编写单元测试

---

## 依赖关系

```
Person F (基础设施) ──────────────────────────────────────────────┐
                                                                  │
Person A (首页) ←───────────── shared/api ←──────────────────────┤
Person B (搜索) ←───────────── shared/types ←────────────────────┤
Person C (用户中心) ←───────── stores/ ←─────────────────────────┤
Person D (数据管理) ←───────── router/ ←─────────────────────────┘
Person E (数据分析) ←──────────────────────────────────────────────
```

---

## 常见问题

### Q: 如何切换 Mock 和真实 API？

A: 修改环境变量 `VITE_USE_MOCK`，或在 `main.ts` 中手动设置：

```typescript
app.use(apiPlugin, { mock: true })  // 使用 Mock
app.use(apiPlugin, { mock: false }) // 使用 HTTP
```

### Q: 如何添加新的路由守卫？

A: 在 `router/index.ts` 中添加：

```typescript
router.beforeEach(async (to, from, next) => {
  // 自定义逻辑
  next()
})
```

### Q: 如何持久化 Store 状态？

A: 使用 `pinia-plugin-persistedstate`：

```typescript
export const useMyStore = defineStore('my', () => {
  // ...
}, {
  persist: true,  // 启用持久化
})
```

---

## 更新日志

### v1.0.0 (2024-01-15)

- ✅ 完成 API Provider 实现（HTTP/Mock）
- ✅ 完成工具函数库（格式化、存储、验证）
- ✅ 完成 Vue Router 配置
- ✅ 完成通用组件库
- ✅ 完成项目配置和入口文件
- ✅ 完成单元测试

---

## 联系方式

如有问题，请联系 Person F 或在项目仓库提交 Issue。

