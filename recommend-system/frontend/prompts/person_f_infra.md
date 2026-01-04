# Person F: 前端基础设施

## 你的角色
你是一名前端工程师，负责实现生成式推荐系统的 **前端基础设施**，包括项目初始化、API 封装、状态管理、路由配置、通用组件等。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读以下文件：**

1. **数据类型定义：**
```
frontend/shared/types/index.ts
```

2. **服务接口定义（核心）：**
```
frontend/shared/api/interfaces.ts
```

你需要实现的核心接口：

```typescript
// API Provider - 所有服务的统一入口
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

**你需要提供两种实现：**
1. `HttpApiProvider` - 生产环境，调用真实 API
2. `MockApiProvider` - 开发/测试环境，返回模拟数据

你需要实现：
1. **Axios 封装** - 基于类型定义的请求响应
2. **API 模块** - 所有接口的统一封装
3. **Pinia Store** - 状态管理
4. **Vue Router** - 路由配置与守卫
5. **通用组件** - 可复用组件

---

## 技术栈

- **框架**: Vue 3 + Composition API + TypeScript
- **构建**: Vite
- **路由**: Vue Router 4
- **状态管理**: Pinia + pinia-plugin-persistedstate
- **HTTP**: Axios
- **工具库**: VueUse

---

## 你的任务

```
frontend/
├── shared/
│   ├── api/
│   │   ├── index.ts              # API 统一导出
│   │   ├── request.ts            # Axios 封装
│   │   ├── user.ts               # 用户 API
│   │   ├── item.ts               # 物品 API
│   │   └── recommend.ts          # 推荐 API
│   ├── types/
│   │   └── index.ts              # 类型定义（已提供）
│   └── utils/
│       ├── index.ts              # 工具函数
│       ├── storage.ts            # 本地存储
│       └── format.ts             # 格式化函数
├── user-app/
│   ├── package.json              # 用户端依赖
│   ├── vite.config.ts            # Vite 配置
│   ├── tsconfig.json             # TS 配置
│   ├── index.html                # 入口 HTML
│   └── src/
│       ├── main.ts               # 应用入口
│       ├── App.vue               # 根组件
│       ├── router/
│       │   └── index.ts          # 路由配置
│       ├── stores/
│       │   ├── index.ts          # Store 导出
│       │   ├── user.ts           # 用户状态
│       │   ├── recommend.ts      # 推荐状态
│       │   └── item.ts           # 物品状态
│       └── components/
│           ├── common/
│           │   ├── Loading.vue   # 加载组件
│           │   ├── Empty.vue     # 空状态
│           │   └── Error.vue     # 错误组件
│           └── ...
└── admin/
    └── ... (类似结构)
```

---

## 1. Axios 封装 (shared/api/request.ts)

```typescript
import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios'
import type { ApiResponse, ApiError } from '../types'

// 配置
const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1'
const TIMEOUT = 30000

// 创建实例
const instance: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器
instance.interceptors.request.use(
  (config) => {
    // 添加 Token
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    
    // 添加请求 ID（用于追踪）
    config.headers['X-Request-ID'] = generateRequestId()
    
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
instance.interceptors.response.use(
  (response) => {
    const data = response.data as ApiResponse<any>
    
    // 业务错误处理
    if (data.code !== 0 && data.code !== 200) {
      const error: ApiError = {
        code: data.code,
        message: data.message || '请求失败',
      }
      return Promise.reject(error)
    }
    
    return data.data
  },
  (error: AxiosError<ApiError>) => {
    // HTTP 错误处理
    const status = error.response?.status
    const data = error.response?.data
    
    let message = '网络错误，请稍后重试'
    
    if (status === 401) {
      message = '登录已过期，请重新登录'
      // 清除 token 并跳转登录
      localStorage.removeItem('token')
      window.location.href = '/login'
    } else if (status === 403) {
      message = '没有权限访问'
    } else if (status === 404) {
      message = '请求的资源不存在'
    } else if (status === 500) {
      message = '服务器错误'
    } else if (data?.message) {
      message = data.message
    }
    
    const apiError: ApiError = {
      code: status || -1,
      message,
      details: data?.details,
    }
    
    return Promise.reject(apiError)
  }
)

// 生成请求 ID
function generateRequestId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

// 封装请求方法
export const request = {
  get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return instance.get(url, config)
  },
  
  post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    return instance.post(url, data, config)
  },
  
  put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    return instance.put(url, data, config)
  },
  
  delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return instance.delete(url, config)
  },
  
  patch<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    return instance.patch(url, data, config)
  },
}

export default instance
```

---

## 2. 用户 API (shared/api/user.ts)

```typescript
import { request } from './request'
import type {
  User,
  CreateUserRequest,
  UpdateUserRequest,
  LoginRequest,
  LoginResponse,
  RegisterRequest,
  UserProfile,
  UserBehavior,
  RecordBehaviorRequest,
} from '../types'

export const userApi = {
  // 登录
  login(data: LoginRequest): Promise<LoginResponse> {
    return request.post('/auth/login', data)
  },
  
  // 注册
  register(data: RegisterRequest): Promise<void> {
    return request.post('/auth/register', data)
  },
  
  // 获取当前用户
  getCurrentUser(): Promise<User> {
    return request.get('/user/me')
  },
  
  // 获取用户信息
  getUser(userId: string): Promise<User> {
    return request.get(`/users/${userId}`)
  },
  
  // 更新用户
  updateUser(userId: string, data: UpdateUserRequest): Promise<User> {
    return request.put(`/users/${userId}`, data)
  },
  
  // 获取用户画像
  getProfile(userId: string): Promise<UserProfile> {
    return request.get(`/users/${userId}/profile`)
  },
  
  // 获取用户行为历史
  getBehaviors(userId: string, limit = 50): Promise<UserBehavior[]> {
    return request.get(`/users/${userId}/behaviors`, { params: { limit } })
  },
  
  // 记录用户行为
  recordBehavior(data: RecordBehaviorRequest): Promise<void> {
    return request.post('/behaviors', data)
  },
}
```

---

## 3. 推荐 API (shared/api/recommend.ts)

```typescript
import { request } from './request'
import type {
  RecommendRequest,
  RecommendResponse,
  FeedbackRequest,
  Recommendation,
} from '../types'

export const recommendApi = {
  // 获取推荐
  getRecommendations(data: RecommendRequest): Promise<RecommendResponse> {
    return request.post('/recommend', data)
  },
  
  // 提交反馈
  submitFeedback(data: FeedbackRequest): Promise<void> {
    return request.post('/feedback', data)
  },
  
  // 获取相似推荐
  getSimilarItems(itemId: string, limit = 10): Promise<Recommendation[]> {
    return request.get(`/similar/${itemId}`, { params: { limit } })
  },
}
```

---

## 4. 物品 API (shared/api/item.ts)

```typescript
import { request } from './request'
import type {
  Item,
  CreateItemRequest,
  UpdateItemRequest,
  ListItemsRequest,
  ListItemsResponse,
  ItemStats,
  SimilarItem,
} from '../types'

export const itemApi = {
  // 获取物品
  getItem(itemId: string): Promise<Item> {
    return request.get(`/items/${itemId}`)
  },
  
  // 搜索物品
  searchItems(query: string, limit = 20): Promise<Item[]> {
    return request.get('/items/search', { params: { q: query, limit } })
  },
  
  // 列出物品
  listItems(params: ListItemsRequest): Promise<ListItemsResponse> {
    return request.get('/items', { params })
  },
  
  // 获取物品统计
  getItemStats(itemId: string): Promise<ItemStats> {
    return request.get(`/items/${itemId}/stats`)
  },
  
  // 获取相似物品
  getSimilarItems(itemId: string, limit = 10): Promise<SimilarItem[]> {
    return request.get(`/items/${itemId}/similar`, { params: { limit } })
  },
}
```

---

## 5. 推荐 Store (stores/recommend.ts)

```typescript
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  Recommendation,
  RecommendRequest,
  RecommendResponse,
  RecordBehaviorRequest,
} from '@shared/types'
import { recommendApi } from '@shared/api'
import { userApi } from '@shared/api'
import { useUserStore } from './user'

export const useRecommendStore = defineStore('recommend', () => {
  // 状态
  const recommendations = ref<Recommendation[]>([])
  const isLoading = ref(false)
  const lastRequestId = ref<string | null>(null)
  const error = ref<string | null>(null)
  
  // 计算属性
  const hasRecommendations = computed(() => recommendations.value.length > 0)
  
  // 获取推荐
  async function fetchRecommendations(request: RecommendRequest): Promise<void> {
    isLoading.value = true
    error.value = null
    
    try {
      const response: RecommendResponse = await recommendApi.getRecommendations(request)
      recommendations.value = response.recommendations
      lastRequestId.value = response.request_id
    } catch (e: any) {
      error.value = e.message || '获取推荐失败'
      throw e
    } finally {
      isLoading.value = false
    }
  }
  
  // 刷新推荐
  async function refreshRecommendations(): Promise<void> {
    const userStore = useUserStore()
    await fetchRecommendations({
      user_id: userStore.currentUser?.id || 'anonymous',
      limit: 50,
      scene: 'home',
      exclude_items: recommendations.value.map(r => r.item_id),
    })
  }
  
  // 记录行为
  async function recordBehavior(
    data: Omit<RecordBehaviorRequest, 'user_id'>
  ): Promise<void> {
    const userStore = useUserStore()
    
    try {
      await userApi.recordBehavior({
        ...data,
        user_id: userStore.currentUser?.id || 'anonymous',
      })
    } catch (e) {
      console.error('Failed to record behavior:', e)
    }
  }
  
  // 提交反馈
  async function submitFeedback(itemId: string, action: string): Promise<void> {
    const userStore = useUserStore()
    
    try {
      await recommendApi.submitFeedback({
        user_id: userStore.currentUser?.id || 'anonymous',
        item_id: itemId,
        action,
        request_id: lastRequestId.value || undefined,
      })
    } catch (e) {
      console.error('Failed to submit feedback:', e)
    }
  }
  
  // 清空推荐
  function clearRecommendations(): void {
    recommendations.value = []
    lastRequestId.value = null
  }
  
  return {
    // 状态
    recommendations,
    isLoading,
    lastRequestId,
    error,
    
    // 计算属性
    hasRecommendations,
    
    // 方法
    fetchRecommendations,
    refreshRecommendations,
    recordBehavior,
    submitFeedback,
    clearRecommendations,
  }
})
```

---

## 6. 路由配置 (router/index.ts)

```typescript
import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import { useUserStore } from '@/stores/user'

// 路由配置
const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/Home.vue'),
    meta: { title: '首页' },
  },
  {
    path: '/search',
    name: 'Search',
    component: () => import('@/views/Search.vue'),
    meta: { title: '搜索' },
  },
  {
    path: '/item/:id',
    name: 'ItemDetail',
    component: () => import('@/views/ItemDetail.vue'),
    meta: { title: '详情' },
  },
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue'),
    meta: { title: '登录', guest: true },
  },
  {
    path: '/register',
    name: 'Register',
    component: () => import('@/views/Register.vue'),
    meta: { title: '注册', guest: true },
  },
  {
    path: '/profile',
    name: 'Profile',
    component: () => import('@/views/Profile.vue'),
    meta: { title: '个人中心', requiresAuth: true },
  },
  {
    path: '/history',
    name: 'History',
    component: () => import('@/views/History.vue'),
    meta: { title: '历史记录', requiresAuth: true },
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('@/views/NotFound.vue'),
    meta: { title: '页面不存在' },
  },
]

// 创建路由
const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    }
    return { top: 0 }
  },
})

// 路由守卫
router.beforeEach(async (to, from, next) => {
  const userStore = useUserStore()
  
  // 设置页面标题
  document.title = `${to.meta.title || '推荐系统'} - AI 推荐`
  
  // 检查登录状态
  if (!userStore.currentUser && userStore.token) {
    try {
      await userStore.fetchCurrentUser()
    } catch {
      userStore.logout()
    }
  }
  
  // 需要登录的页面
  if (to.meta.requiresAuth && !userStore.isLoggedIn) {
    next({ name: 'Login', query: { redirect: to.fullPath } })
    return
  }
  
  // 登录后不能访问的页面
  if (to.meta.guest && userStore.isLoggedIn) {
    next({ name: 'Home' })
    return
  }
  
  next()
})

export default router
```

---

## 7. 通用组件 - Loading (components/common/Loading.vue)

```vue
<script setup lang="ts">
interface Props {
  size?: 'small' | 'default' | 'large'
  tip?: string
  fullscreen?: boolean
}

withDefaults(defineProps<Props>(), {
  size: 'default',
  fullscreen: false,
})

const sizeMap = {
  small: 24,
  default: 40,
  large: 56,
}
</script>

<template>
  <div class="loading" :class="{ fullscreen }">
    <div class="loading-spinner" :style="{ width: `${sizeMap[size]}px`, height: `${sizeMap[size]}px` }">
      <svg viewBox="0 0 50 50" class="circular">
        <circle cx="25" cy="25" r="20" fill="none" class="path" />
      </svg>
    </div>
    <p v-if="tip" class="loading-tip">{{ tip }}</p>
  </div>
</template>

<style scoped>
.loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.loading.fullscreen {
  position: fixed;
  inset: 0;
  background: rgba(26, 26, 46, 0.9);
  z-index: 9999;
}

.loading-spinner {
  animation: rotate 2s linear infinite;
}

.circular {
  width: 100%;
  height: 100%;
}

.path {
  stroke: #4facfe;
  stroke-width: 3;
  stroke-linecap: round;
  animation: dash 1.5s ease-in-out infinite;
}

.loading-tip {
  margin-top: 12px;
  color: #8892b0;
  font-size: 14px;
}

@keyframes rotate {
  100% {
    transform: rotate(360deg);
  }
}

@keyframes dash {
  0% {
    stroke-dasharray: 1, 150;
    stroke-dashoffset: 0;
  }
  50% {
    stroke-dasharray: 90, 150;
    stroke-dashoffset: -35;
  }
  100% {
    stroke-dasharray: 90, 150;
    stroke-dashoffset: -124;
  }
}
</style>
```

---

## 8. 应用入口 (main.ts)

```typescript
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate'
import App from './App.vue'
import router from './router'

// 样式
import './assets/styles/main.css'

// 创建应用
const app = createApp(App)

// Pinia
const pinia = createPinia()
pinia.use(piniaPluginPersistedstate)
app.use(pinia)

// 路由
app.use(router)

// 全局错误处理
app.config.errorHandler = (err, vm, info) => {
  console.error('Global error:', err, info)
}

// 挂载
app.mount('#app')
```

---

## 9. Vite 配置 (vite.config.ts)

```typescript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

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
  build: {
    target: 'es2020',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['vue', 'vue-router', 'pinia', 'axios'],
        },
      },
    },
  },
})
```

---

## 10. package.json

```json
{
  "name": "recommend-user-app",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vue-tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint src --ext .vue,.js,.ts,.jsx,.tsx --fix",
    "type-check": "vue-tsc --noEmit"
  },
  "dependencies": {
    "vue": "^3.4.0",
    "vue-router": "^4.2.0",
    "pinia": "^2.1.0",
    "pinia-plugin-persistedstate": "^3.2.0",
    "axios": "^1.6.0",
    "@vueuse/core": "^10.7.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.0.0",
    "vite": "^5.0.0",
    "typescript": "^5.3.0",
    "vue-tsc": "^1.8.0",
    "@types/node": "^20.10.0"
  }
}
```

---

## 注意事项

1. 所有 API 需要统一错误处理
2. Token 存储使用 localStorage
3. 路由守卫检查权限
4. Store 支持持久化
5. Vite 配置代理解决跨域

## 输出要求

请输出完整的可运行代码，包含：
1. 项目初始化文件
2. API 封装完整
3. Store 完整实现
4. 路由配置完整
5. 通用组件

