# Person C: 用户端 - 用户中心

## 你的角色
你是一名前端工程师，负责实现生成式推荐系统的 **用户中心** 模块，包括登录、注册、个人资料、历史记录等。

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

你需要使用的核心接口：

```typescript
// 认证服务接口
interface IAuthService {
  login(credentials: LoginRequest): Promise<LoginResponse>
  register(data: RegisterRequest): Promise<void>
  logout(): Promise<void>
  getCurrentUser(): Promise<User>
}

// 用户服务接口
interface IUserService {
  getUser(userId: string): Promise<User>
  updateUser(userId: string, data: UpdateUserRequest): Promise<User>
  getProfile(userId: string): Promise<UserProfile>
  getBehaviors(userId: string, limit?: number): Promise<UserBehavior[]>
}
```

**⚠️ 不要直接导入具体实现！** 使用依赖注入：

```typescript
// ✅ 正确：通过 inject 获取接口
const api = inject<IApiProvider>('api')!
await api.auth.login({ email, password })
const profile = await api.user.getProfile(userId)

// ❌ 错误：直接导入具体实现
import { userApi } from '@shared/api'
```

---

## 技术栈

- **框架**: Vue 3 + Composition API + TypeScript
- **构建**: Vite
- **路由**: Vue Router
- **状态管理**: Pinia (持久化存储 token)
- **HTTP**: Axios
- **表单验证**: VeeValidate 或手动验证

---

## 你的任务

```
frontend/user-app/
├── src/
│   ├── views/
│   │   ├── Login.vue              # 登录页
│   │   ├── Register.vue           # 注册页
│   │   ├── Profile.vue            # 个人资料页
│   │   └── History.vue            # 历史记录页
│   ├── components/
│   │   ├── AuthForm.vue           # 认证表单
│   │   ├── ProfileCard.vue        # 个人信息卡片
│   │   ├── PreferenceChart.vue    # 偏好图表
│   │   └── HistoryList.vue        # 历史记录列表
│   ├── stores/
│   │   └── user.ts                # 用户状态管理
│   └── ...
```

---

## 1. 用户状态管理 (stores/user.ts)

```typescript
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { User, LoginRequest, RegisterRequest, UserProfile, UserBehavior } from '@shared/types'
import { userApi } from '@/api/user'

export const useUserStore = defineStore('user', () => {
  // 状态
  const token = ref<string | null>(localStorage.getItem('token'))
  const currentUser = ref<User | null>(null)
  const profile = ref<UserProfile | null>(null)
  const behaviors = ref<UserBehavior[]>([])
  
  // 计算属性
  const isLoggedIn = computed(() => !!token.value && !!currentUser.value)
  
  // 登录
  async function login(credentials: LoginRequest): Promise<void> {
    const response = await userApi.login(credentials)
    token.value = response.token
    currentUser.value = response.user
    localStorage.setItem('token', response.token)
  }
  
  // 注册
  async function register(data: RegisterRequest): Promise<void> {
    await userApi.register(data)
    // 注册后自动登录
    await login({ email: data.email, password: data.password })
  }
  
  // 登出
  function logout(): void {
    token.value = null
    currentUser.value = null
    profile.value = null
    behaviors.value = []
    localStorage.removeItem('token')
  }
  
  // 获取当前用户信息
  async function fetchCurrentUser(): Promise<void> {
    if (!token.value) return
    try {
      currentUser.value = await userApi.getCurrentUser()
    } catch {
      logout()
    }
  }
  
  // 获取用户画像
  async function fetchProfile(): Promise<void> {
    if (!currentUser.value) return
    profile.value = await userApi.getProfile(currentUser.value.id)
  }
  
  // 获取历史记录
  async function fetchBehaviors(limit = 50): Promise<void> {
    if (!currentUser.value) return
    behaviors.value = await userApi.getBehaviors(currentUser.value.id, limit)
  }
  
  // 更新用户信息
  async function updateProfile(data: Partial<User>): Promise<void> {
    if (!currentUser.value) return
    currentUser.value = await userApi.updateUser(currentUser.value.id, data)
  }
  
  // 初始化 - 检查本地 token
  async function init(): Promise<void> {
    if (token.value) {
      await fetchCurrentUser()
    }
  }
  
  return {
    // 状态
    token,
    currentUser,
    profile,
    behaviors,
    
    // 计算属性
    isLoggedIn,
    
    // 方法
    login,
    register,
    logout,
    fetchCurrentUser,
    fetchProfile,
    fetchBehaviors,
    updateProfile,
    init,
  }
})
```

---

## 2. 登录页 (Login.vue)

```vue
<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useUserStore } from '@/stores/user'
import AuthForm from '@/components/AuthForm.vue'

const router = useRouter()
const route = useRoute()
const userStore = useUserStore()

// 表单数据
const formData = reactive({
  email: '',
  password: '',
})

// 状态
const isLoading = ref(false)
const errorMessage = ref('')

// 验证规则
const rules = {
  email: [
    (v: string) => !!v || '请输入邮箱',
    (v: string) => /.+@.+\..+/.test(v) || '邮箱格式不正确',
  ],
  password: [
    (v: string) => !!v || '请输入密码',
    (v: string) => v.length >= 6 || '密码至少6位',
  ],
}

// 提交登录
async function handleSubmit() {
  errorMessage.value = ''
  isLoading.value = true
  
  try {
    await userStore.login({
      email: formData.email,
      password: formData.password,
    })
    
    // 跳转到目标页面或首页
    const redirect = (route.query.redirect as string) || '/'
    router.push(redirect)
  } catch (error: any) {
    errorMessage.value = error.message || '登录失败，请重试'
  } finally {
    isLoading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <div class="login-container">
      <!-- Logo -->
      <div class="logo-section">
        <div class="logo">✨</div>
        <h1 class="title">欢迎回来</h1>
        <p class="subtitle">登录以获取个性化推荐</p>
      </div>

      <!-- 登录表单 -->
      <AuthForm
        type="login"
        :loading="isLoading"
        :error="errorMessage"
        @submit="handleSubmit"
      >
        <template #fields>
          <div class="form-group">
            <label class="form-label">邮箱</label>
            <input
              v-model="formData.email"
              type="email"
              class="form-input"
              placeholder="your@email.com"
              autocomplete="email"
            />
          </div>
          
          <div class="form-group">
            <label class="form-label">密码</label>
            <input
              v-model="formData.password"
              type="password"
              class="form-input"
              placeholder="••••••••"
              autocomplete="current-password"
            />
          </div>
        </template>
      </AuthForm>

      <!-- 其他选项 -->
      <div class="auth-footer">
        <p>
          还没有账号？
          <router-link to="/register" class="link">立即注册</router-link>
        </p>
      </div>
    </div>

    <!-- 背景装饰 -->
    <div class="bg-decoration">
      <div class="circle circle-1"></div>
      <div class="circle circle-2"></div>
      <div class="circle circle-3"></div>
    </div>
  </div>
</template>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  position: relative;
  overflow: hidden;
}

.login-container {
  width: 100%;
  max-width: 400px;
  padding: 2rem;
  position: relative;
  z-index: 1;
}

.logo-section {
  text-align: center;
  margin-bottom: 2rem;
}

.logo {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.title {
  font-size: 2rem;
  font-weight: 700;
  color: #fff;
  margin-bottom: 0.5rem;
}

.subtitle {
  color: #8892b0;
  font-size: 1rem;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-label {
  display: block;
  color: #ccd6f6;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}

.form-input {
  width: 100%;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.05);
  border: 2px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.75rem;
  color: #fff;
  font-size: 1rem;
  transition: all 0.3s;
}

.form-input:focus {
  outline: none;
  border-color: #4facfe;
  box-shadow: 0 0 20px rgba(79, 172, 254, 0.2);
}

.form-input::placeholder {
  color: #8892b0;
}

.auth-footer {
  text-align: center;
  margin-top: 2rem;
  color: #8892b0;
}

.link {
  color: #4facfe;
  text-decoration: none;
  font-weight: 600;
}

.link:hover {
  text-decoration: underline;
}

/* 背景装饰 */
.bg-decoration {
  position: absolute;
  inset: 0;
  overflow: hidden;
  pointer-events: none;
}

.circle {
  position: absolute;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(79, 172, 254, 0.1) 0%, transparent 70%);
}

.circle-1 {
  width: 600px;
  height: 600px;
  top: -200px;
  right: -200px;
}

.circle-2 {
  width: 400px;
  height: 400px;
  bottom: -100px;
  left: -100px;
}

.circle-3 {
  width: 300px;
  height: 300px;
  bottom: 20%;
  right: 10%;
}
</style>
```

---

## 3. 个人资料页 (Profile.vue)

```vue
<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useUserStore } from '@/stores/user'
import ProfileCard from '@/components/ProfileCard.vue'
import PreferenceChart from '@/components/PreferenceChart.vue'

const userStore = useUserStore()

const isEditing = ref(false)
const editForm = ref({
  name: '',
  age: 0,
  gender: '',
})

// 加载用户画像
onMounted(async () => {
  await userStore.fetchProfile()
  if (userStore.currentUser) {
    editForm.value = {
      name: userStore.currentUser.name,
      age: userStore.currentUser.age || 0,
      gender: userStore.currentUser.gender || '',
    }
  }
})

// 开始编辑
function startEdit() {
  isEditing.value = true
}

// 保存编辑
async function saveEdit() {
  await userStore.updateProfile(editForm.value)
  isEditing.value = false
}

// 取消编辑
function cancelEdit() {
  if (userStore.currentUser) {
    editForm.value = {
      name: userStore.currentUser.name,
      age: userStore.currentUser.age || 0,
      gender: userStore.currentUser.gender || '',
    }
  }
  isEditing.value = false
}
</script>

<template>
  <div class="profile-page">
    <div class="page-header">
      <h1>个人中心</h1>
    </div>

    <div class="profile-content">
      <!-- 个人信息卡片 -->
      <section class="section">
        <ProfileCard
          :user="userStore.currentUser!"
          :is-editing="isEditing"
          v-model:form="editForm"
          @edit="startEdit"
          @save="saveEdit"
          @cancel="cancelEdit"
        />
      </section>

      <!-- 用户画像 -->
      <section v-if="userStore.profile" class="section">
        <h2 class="section-title">我的画像</h2>
        
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-value">{{ userStore.profile.total_actions }}</div>
            <div class="stat-label">总互动次数</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{{ Object.keys(userStore.profile.preferred_types).length }}</div>
            <div class="stat-label">喜好类型</div>
          </div>
        </div>

        <!-- 偏好图表 -->
        <PreferenceChart :data="userStore.profile.preferred_types" />
      </section>

      <!-- 账号操作 -->
      <section class="section">
        <h2 class="section-title">账号设置</h2>
        
        <div class="action-list">
          <router-link to="/history" class="action-item">
            <span class="action-icon">📜</span>
            <span class="action-text">浏览历史</span>
            <span class="action-arrow">→</span>
          </router-link>
          
          <button class="action-item logout" @click="userStore.logout()">
            <span class="action-icon">🚪</span>
            <span class="action-text">退出登录</span>
            <span class="action-arrow">→</span>
          </button>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.profile-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #eee;
  padding: 2rem;
}

.page-header {
  max-width: 800px;
  margin: 0 auto 2rem;
}

.page-header h1 {
  font-size: 2rem;
  font-weight: 700;
  background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.profile-content {
  max-width: 800px;
  margin: 0 auto;
}

.section {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 1rem;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.section-title {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 1rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.stat-card {
  background: rgba(79, 172, 254, 0.1);
  border-radius: 0.75rem;
  padding: 1.25rem;
  text-align: center;
}

.stat-value {
  font-size: 2rem;
  font-weight: 700;
  color: #4facfe;
}

.stat-label {
  font-size: 0.875rem;
  color: #8892b0;
  margin-top: 0.25rem;
}

.action-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.action-item {
  display: flex;
  align-items: center;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 0.75rem;
  color: #ccd6f6;
  text-decoration: none;
  border: none;
  cursor: pointer;
  transition: background 0.3s;
}

.action-item:hover {
  background: rgba(255, 255, 255, 0.1);
}

.action-item.logout {
  color: #ff6b6b;
}

.action-icon {
  font-size: 1.25rem;
  margin-right: 1rem;
}

.action-text {
  flex: 1;
  text-align: left;
}

.action-arrow {
  color: #8892b0;
}
</style>
```

---

## 4. 历史记录页 (History.vue)

```vue
<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useUserStore } from '@/stores/user'
import { useRouter } from 'vue-router'
import HistoryList from '@/components/HistoryList.vue'

const userStore = useUserStore()
const router = useRouter()

const isLoading = ref(true)
const filterAction = ref('all')

// 过滤后的历史
const filteredBehaviors = computed(() => {
  if (filterAction.value === 'all') {
    return userStore.behaviors
  }
  return userStore.behaviors.filter(b => b.action === filterAction.value)
})

// 加载历史
onMounted(async () => {
  try {
    await userStore.fetchBehaviors(100)
  } finally {
    isLoading.value = false
  }
})

// 点击历史项
function handleItemClick(itemId: string) {
  router.push(`/item/${itemId}`)
}
</script>

<template>
  <div class="history-page">
    <header class="page-header">
      <button class="back-btn" @click="router.back()">← 返回</button>
      <h1>浏览历史</h1>
    </header>

    <!-- 过滤器 -->
    <div class="filter-bar">
      <button
        v-for="action in ['all', 'view', 'click', 'like']"
        :key="action"
        class="filter-btn"
        :class="{ active: filterAction === action }"
        @click="filterAction = action"
      >
        {{ { all: '全部', view: '浏览', click: '点击', like: '喜欢' }[action] }}
      </button>
    </div>

    <!-- 历史列表 -->
    <main class="history-content">
      <div v-if="isLoading" class="loading">加载中...</div>
      
      <HistoryList
        v-else-if="filteredBehaviors.length > 0"
        :behaviors="filteredBehaviors"
        @item-click="handleItemClick"
      />
      
      <div v-else class="empty-state">
        <p>暂无历史记录</p>
      </div>
    </main>
  </div>
</template>

<style scoped>
.history-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #eee;
}

.page-header {
  padding: 2rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.back-btn {
  padding: 0.5rem 1rem;
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 0.5rem;
  color: #fff;
  cursor: pointer;
}

.page-header h1 {
  font-size: 1.5rem;
  font-weight: 600;
}

.filter-bar {
  display: flex;
  gap: 0.5rem;
  padding: 0 2rem 1rem;
}

.filter-btn {
  padding: 0.5rem 1rem;
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 2rem;
  color: #8892b0;
  cursor: pointer;
  transition: all 0.3s;
}

.filter-btn.active {
  background: #4facfe;
  color: #fff;
}

.history-content {
  padding: 0 2rem 2rem;
  max-width: 800px;
  margin: 0 auto;
}

.loading, .empty-state {
  text-align: center;
  padding: 4rem;
  color: #8892b0;
}
</style>
```

---

## 注意事项

1. Token 需要持久化到 `localStorage`
2. 路由守卫检查登录状态
3. 表单验证要完善
4. 错误提示用户友好

## 输出要求

请输出完整的可运行代码，包含：
1. Pinia Store 完整实现
2. 所有 Vue 组件
3. 表单验证逻辑
4. 完整的样式

