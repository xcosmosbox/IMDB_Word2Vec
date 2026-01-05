# 用户中心模块开发文档

> **Person C** - 用户端用户中心模块  
> 版本：1.0.0  
> 更新日期：2025-01-04

---

## 📋 目录

- [模块概述](#模块概述)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [核心功能](#核心功能)
- [组件说明](#组件说明)
- [状态管理](#状态管理)
- [表单验证](#表单验证)
- [页面路由](#页面路由)
- [样式规范](#样式规范)
- [测试说明](#测试说明)
- [开发指南](#开发指南)
- [常见问题](#常见问题)

---

## 模块概述

用户中心模块负责实现生成式推荐系统的用户认证和用户信息管理功能，包括：

- **用户认证**：登录、注册、登出
- **个人资料**：查看和编辑用户信息
- **用户画像**：展示用户偏好分析
- **历史记录**：浏览用户行为历史

### 设计原则

1. **接口驱动开发**：所有 API 调用通过依赖注入的接口进行
2. **类型安全**：全面使用 TypeScript 类型定义
3. **组件化**：可复用的 UI 组件设计
4. **暗色主题**：统一的暗色 UI 风格

---

## 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Vue 3 | ^3.4 | 核心框架 |
| TypeScript | ^5.3 | 类型安全 |
| Pinia | ^2.1 | 状态管理 |
| Vue Router | ^4.2 | 路由管理 |
| Vitest | ^1.0 | 单元测试 |
| @vue/test-utils | ^2.4 | 组件测试 |

---

## 目录结构

```
user-app/src/
├── views/                      # 页面组件
│   ├── Login.vue               # 登录页
│   ├── Register.vue            # 注册页
│   ├── Profile.vue             # 个人资料页
│   └── History.vue             # 历史记录页
│
├── components/                 # 通用组件
│   ├── AuthForm.vue            # 认证表单
│   ├── ProfileCard.vue         # 个人信息卡片
│   ├── PreferenceChart.vue     # 偏好图表
│   └── HistoryList.vue         # 历史记录列表
│
├── stores/                     # 状态管理
│   └── user.ts                 # 用户状态 Store
│
├── composables/                # 组合式函数
│   └── useFormValidation.ts    # 表单验证工具
│
├── __tests__/                  # 单元测试
│   ├── stores/
│   │   └── user.spec.ts
│   ├── composables/
│   │   └── useFormValidation.spec.ts
│   └── components/
│       ├── AuthForm.spec.ts
│       ├── ProfileCard.spec.ts
│       ├── PreferenceChart.spec.ts
│       └── HistoryList.spec.ts
│
└── docs/                       # 文档
    └── USER_CENTER_README.md   # 本文档
```

---

## 核心功能

### 1. 用户登录 (`Login.vue`)

用户使用邮箱和密码登录系统。

**功能特性：**
- 邮箱格式验证
- 密码最小长度验证
- 密码显示/隐藏切换
- 记住我选项
- 登录失败错误提示
- 登录成功后跳转

**使用示例：**
```vue
<template>
  <Login />
</template>
```

### 2. 用户注册 (`Register.vue`)

新用户创建账号。

**功能特性：**
- 昵称长度验证 (2-20 字符)
- 邮箱格式验证
- 密码强度验证 (包含字母和数字，至少 6 位)
- 确认密码一致性验证
- 可选信息 (年龄、性别)
- 用户协议确认
- 注册成功后自动登录

### 3. 个人资料 (`Profile.vue`)

查看和编辑用户信息。

**功能特性：**
- 显示用户基本信息
- 编辑模式切换
- 用户画像展示
- 偏好分析图表
- 活跃时段统计
- 操作入口 (历史记录、设置等)

### 4. 历史记录 (`History.vue`)

展示用户行为历史。

**功能特性：**
- 按日期分组显示
- 按操作类型过滤
- 支持 6 种操作类型 (浏览、点击、喜欢、不喜欢、购买、分享)
- 相对时间显示
- 点击跳转到物品详情

---

## 组件说明

### AuthForm

通用认证表单组件，用于登录和注册页面。

**Props:**

| 属性 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| type | `'login' \| 'register'` | 是 | - | 表单类型 |
| loading | `boolean` | 否 | `false` | 加载状态 |
| error | `string` | 否 | `''` | 错误信息 |
| submitText | `string` | 否 | 自动 | 提交按钮文字 |

**Events:**

| 事件名 | 参数 | 说明 |
|--------|------|------|
| submit | - | 表单提交 |

**Slots:**

| 插槽名 | 说明 |
|--------|------|
| fields | 表单字段 |
| extra | 额外内容 (如记住密码) |
| footer | 底部内容 (如其他登录方式) |

**使用示例：**
```vue
<AuthForm
  type="login"
  :loading="isLoading"
  :error="errorMessage"
  @submit="handleSubmit"
>
  <template #fields>
    <input v-model="email" type="email" />
    <input v-model="password" type="password" />
  </template>
</AuthForm>
```

### ProfileCard

个人信息卡片组件。

**Props:**

| 属性 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| user | `User` | 是 | - | 用户信息 |
| isEditing | `boolean` | 否 | `false` | 是否编辑模式 |
| loading | `boolean` | 否 | `false` | 保存加载状态 |

**Events:**

| 事件名 | 参数 | 说明 |
|--------|------|------|
| edit | - | 开始编辑 |
| save | `UpdateUserRequest` | 保存编辑 |
| cancel | - | 取消编辑 |

### PreferenceChart

偏好图表组件，纯 CSS 实现的条形图。

**Props:**

| 属性 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| data | `Record<string, number>` | 是 | - | 偏好数据 |
| title | `string` | 否 | `'内容偏好'` | 图表标题 |
| maxItems | `number` | 否 | `6` | 最大显示数量 |
| showPercentage | `boolean` | 否 | `true` | 是否显示百分比 |

**支持的类型映射：**
- `movie` → 电影
- `product` → 商品
- `article` → 文章
- `video` → 视频
- `music` → 音乐
- `book` → 图书
- `game` → 游戏
- `news` → 新闻

### HistoryList

历史记录列表组件。

**Props:**

| 属性 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| behaviors | `UserBehavior[]` | 是 | - | 行为数据 |
| groupByDate | `boolean` | 否 | `true` | 是否按日期分组 |
| showTimestamp | `boolean` | 否 | `true` | 是否显示时间戳 |

**Events:**

| 事件名 | 参数 | 说明 |
|--------|------|------|
| item-click | `itemId: string` | 点击物品 |

---

## 状态管理

### useUserStore

用户状态管理 Store，使用 Pinia Composition API 风格。

**State:**

```typescript
interface UserState {
  token: string | null           // JWT 令牌
  currentUser: User | null       // 当前用户
  profile: UserProfile | null    // 用户画像
  behaviors: UserBehavior[]      // 行为历史
  isLoading: boolean             // 加载状态
  error: string | null           // 错误信息
}
```

**Getters:**

```typescript
isLoggedIn: boolean      // 是否已登录
userId: string | null    // 用户 ID
displayName: string      // 显示名称
avatarInitial: string    // 头像首字母
```

**Actions:**

```typescript
// 认证操作
login(credentials: LoginRequest): Promise<void>
register(data: RegisterRequest): Promise<void>
logout(): Promise<void>
refreshToken(): Promise<string>

// 用户信息操作
fetchCurrentUser(): Promise<void>
fetchProfile(): Promise<void>
fetchBehaviors(limit?: number): Promise<void>
updateProfile(data: UpdateUserRequest): Promise<void>

// 初始化
init(): Promise<void>

// 工具方法
clearError(): void
```

**使用示例：**

```typescript
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()

// 登录
await userStore.login({
  email: 'user@example.com',
  password: 'password123'
})

// 检查登录状态
if (userStore.isLoggedIn) {
  console.log(`欢迎, ${userStore.displayName}`)
}

// 获取用户画像
await userStore.fetchProfile()
console.log(userStore.profile?.preferred_types)

// 登出
await userStore.logout()
```

**依赖注入：**

Store 通过 Vue 的 `inject` 获取 API Provider：

```typescript
// main.ts 中配置
import { createApp } from 'vue'
import { HttpApiProvider } from '@shared/api'

const app = createApp(App)
app.provide('api', new HttpApiProvider())
```

---

## 表单验证

### useFormValidation

通用表单验证组合式函数。

**预定义验证规则：**

```typescript
// 必填
required(message?: string): ValidationRule

// 邮箱格式
email(message?: string): ValidationRule

// 最小长度
minLength(min: number, message?: string): ValidationRule

// 最大长度
maxLength(max: number, message?: string): ValidationRule

// 数字范围
numberRange(min: number, max: number, message?: string): ValidationRule

// 密码强度
passwordStrength(message?: string): ValidationRule

// 确认密码
confirmPassword(getPassword: () => string, message?: string): ValidationRule

// 正则匹配
pattern(regex: RegExp, message: string): ValidationRule
```

**使用示例：**

```typescript
import {
  useFormValidation,
  required,
  email,
  minLength,
} from '@/composables/useFormValidation'

const { fields, validate, reset } = useFormValidation({
  email: {
    value: '',
    rules: [required('请输入邮箱'), email('邮箱格式不正确')]
  },
  password: {
    value: '',
    rules: [required('请输入密码'), minLength(6, '密码至少6位')]
  }
})

// 在模板中使用
// <input v-model="fields.email.value" @blur="fields.email.touched = true" />
// <span v-if="fields.email.touched && fields.email.error">{{ fields.email.error }}</span>

// 提交时验证
async function handleSubmit() {
  if (!validate()) {
    return
  }
  // 提交表单...
}
```

### useFieldValidation

单字段验证组合式函数（简化版）。

```typescript
import { useFieldValidation, required, email } from '@/composables/useFormValidation'

const emailField = useFieldValidation('', [required(), email()])

// 使用
emailField.value.value = 'test@example.com'
emailField.touch()  // 触发验证
console.log(emailField.isValid.value)  // true
```

---

## 页面路由

建议的路由配置：

```typescript
// router/index.ts
import { createRouter, createWebHistory } from 'vue-router'
import { useUserStore } from '@/stores/user'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue'),
    meta: { requiresGuest: true }
  },
  {
    path: '/register',
    name: 'Register',
    component: () => import('@/views/Register.vue'),
    meta: { requiresGuest: true }
  },
  {
    path: '/profile',
    name: 'Profile',
    component: () => import('@/views/Profile.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/history',
    name: 'History',
    component: () => import('@/views/History.vue'),
    meta: { requiresAuth: true }
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// 路由守卫
router.beforeEach(async (to, from, next) => {
  const userStore = useUserStore()
  
  // 初始化用户状态
  if (!userStore.currentUser && localStorage.getItem('token')) {
    await userStore.init()
  }
  
  // 需要登录的页面
  if (to.meta.requiresAuth && !userStore.isLoggedIn) {
    next({ path: '/login', query: { redirect: to.fullPath } })
    return
  }
  
  // 已登录用户不能访问的页面
  if (to.meta.requiresGuest && userStore.isLoggedIn) {
    next('/')
    return
  }
  
  next()
})

export default router
```

---

## 样式规范

### 颜色变量

```css
/* 主色调 */
--color-primary: #4facfe;
--color-primary-light: #00f2fe;
--color-secondary: #a78bfa;

/* 背景色 */
--bg-dark: #1a1a2e;
--bg-darker: #16213e;
--bg-card: rgba(255, 255, 255, 0.05);

/* 文字颜色 */
--text-primary: #ffffff;
--text-secondary: #ccd6f6;
--text-muted: #8892b0;

/* 状态颜色 */
--color-success: #34d399;
--color-error: #ff6b6b;
--color-warning: #fbbf24;
```

### 间距规范

```css
/* 基础间距 */
--spacing-xs: 0.25rem;   /* 4px */
--spacing-sm: 0.5rem;    /* 8px */
--spacing-md: 1rem;      /* 16px */
--spacing-lg: 1.5rem;    /* 24px */
--spacing-xl: 2rem;      /* 32px */
```

### 圆角规范

```css
--radius-sm: 0.5rem;     /* 8px */
--radius-md: 0.75rem;    /* 12px */
--radius-lg: 1rem;       /* 16px */
--radius-full: 9999px;   /* 圆形 */
```

---

## 测试说明

### 运行测试

```bash
# 运行所有测试
npm run test

# 运行并监听变化
npm run test:watch

# 生成覆盖率报告
npm run test:coverage
```

### 测试文件结构

```
__tests__/
├── stores/
│   └── user.spec.ts           # Store 测试
├── composables/
│   └── useFormValidation.spec.ts  # 组合式函数测试
└── components/
    ├── AuthForm.spec.ts       # 组件测试
    ├── ProfileCard.spec.ts
    ├── PreferenceChart.spec.ts
    └── HistoryList.spec.ts
```

### 测试覆盖内容

| 模块 | 测试内容 |
|------|----------|
| User Store | 登录、注册、登出、状态管理、计算属性 |
| useFormValidation | 验证规则、表单验证、字段操作、重置 |
| AuthForm | 渲染、错误显示、加载状态、插槽、事件 |
| ProfileCard | 查看模式、编辑模式、加载状态、边缘情况 |
| PreferenceChart | 渲染、数据处理、类型映射、进度条、空状态 |
| HistoryList | 渲染、时间显示、日期分组、空状态、事件 |

---

## 开发指南

### 1. 开始开发

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 运行测试
npm run test
```

### 2. 添加新组件

1. 在 `components/` 目录创建 `.vue` 文件
2. 使用 `<script setup lang="ts">` 和 Composition API
3. 添加完整的 Props 和 Events 类型定义
4. 使用 `scoped` 样式
5. 在 `__tests__/components/` 添加对应测试文件

### 3. 添加新页面

1. 在 `views/` 目录创建 `.vue` 文件
2. 在路由配置中添加路由
3. 设置适当的 `meta` 属性（如 `requiresAuth`）
4. 使用 `useUserStore` 访问用户状态

### 4. 扩展验证规则

```typescript
// 在 useFormValidation.ts 中添加
export const phone = (message = '手机号格式不正确'): ValidationRule => {
  return (value: string) => {
    if (!value) return true
    const phoneRegex = /^1[3-9]\d{9}$/
    return phoneRegex.test(value) || message
  }
}
```

### 5. 代码规范

- 使用 TypeScript 严格模式
- 组件使用 PascalCase 命名
- 文件使用 camelCase 或 PascalCase 命名
- 所有导出函数和组件添加 JSDoc 注释

---

## 常见问题

### Q: 如何处理 Token 过期？

A: Store 中的 `refreshToken` 方法会自动刷新令牌。如果刷新失败，会自动登出并跳转到登录页。

```typescript
// 在 axios 拦截器中处理
axios.interceptors.response.use(
  response => response,
  async error => {
    if (error.response?.status === 401) {
      try {
        await userStore.refreshToken()
        return axios(error.config)
      } catch {
        await userStore.logout()
        router.push('/login')
      }
    }
    return Promise.reject(error)
  }
)
```

### Q: 如何自定义组件样式？

A: 组件使用 `scoped` 样式，可以通过 CSS 变量或深度选择器覆盖：

```css
/* 使用 CSS 变量 */
.my-container {
  --color-primary: #ff6b6b;
}

/* 使用深度选择器 */
.my-container :deep(.form-input) {
  background: #fff;
}
```

### Q: 如何测试异步操作？

A: 使用 `vi.mock` 模拟 API，使用 `async/await` 处理异步：

```typescript
it('登录成功应该更新状态', async () => {
  mockAuthService.login.mockResolvedValue(mockLoginResponse)
  
  await userStore.login({ email: 'test@example.com', password: '123' })
  
  expect(userStore.isLoggedIn).toBe(true)
})
```

### Q: 如何处理 API Provider 未初始化？

A: Store 会检查 API Provider 是否存在，如果不存在会抛出错误：

```typescript
const api = getApi()
if (!api) {
  throw new Error('API Provider 未初始化')
}
```

确保在 `main.ts` 中正确配置：

```typescript
app.provide('api', new HttpApiProvider())
```

---

## 更新日志

### v1.0.0 (2025-01-04)

- ✅ 完成登录页面
- ✅ 完成注册页面
- ✅ 完成个人资料页面
- ✅ 完成历史记录页面
- ✅ 完成 AuthForm 组件
- ✅ 完成 ProfileCard 组件
- ✅ 完成 PreferenceChart 组件
- ✅ 完成 HistoryList 组件
- ✅ 完成 useFormValidation 组合式函数
- ✅ 完成单元测试
- ✅ 完成开发文档

---

## 联系方式

如有问题，请联系 Person C 或在项目仓库提交 Issue。

