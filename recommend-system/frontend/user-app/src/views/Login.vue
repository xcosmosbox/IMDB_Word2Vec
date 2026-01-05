/**
 * Login - 登录页面
 * 
 * 用户登录页面，支持邮箱密码登录。
 * 登录成功后跳转至首页或来源页面。
 * 
 * @view
 * @author Person C
 */
<script setup lang="ts">
import { ref, inject, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useUserStore } from '@/stores/user'
import AuthForm from '@/components/AuthForm.vue'
import {
  useFormValidation,
  required,
  email,
  minLength,
} from '@/composables/useFormValidation'
import type { IApiProvider } from '@shared/api/interfaces'

// =============================================================================
// 依赖注入
// =============================================================================

const router = useRouter()
const route = useRoute()
const userStore = useUserStore()

// =============================================================================
// 表单验证
// =============================================================================

const { fields, validate, reset } = useFormValidation({
  email: {
    value: '',
    rules: [required('请输入邮箱'), email('邮箱格式不正确')],
  },
  password: {
    value: '',
    rules: [required('请输入密码'), minLength(6, '密码至少6位')],
  },
})

// =============================================================================
// 状态
// =============================================================================

const isLoading = ref(false)
const errorMessage = ref('')
const showPassword = ref(false)

// =============================================================================
// 事件处理
// =============================================================================

/**
 * 处理登录提交
 */
async function handleSubmit() {
  // 表单验证
  if (!validate()) {
    return
  }

  errorMessage.value = ''
  isLoading.value = true

  try {
    await userStore.login({
      email: fields.email.value,
      password: fields.password.value,
    })

    // 登录成功，跳转到目标页面
    const redirect = (route.query.redirect as string) || '/'
    router.push(redirect)
  } catch (error: any) {
    errorMessage.value = error?.message || '登录失败，请检查邮箱和密码'
  } finally {
    isLoading.value = false
  }
}

/**
 * 处理字段失焦
 */
function handleBlur(fieldName: 'email' | 'password') {
  fields[fieldName].touched = true
}

/**
 * 切换密码可见性
 */
function togglePasswordVisibility() {
  showPassword.value = !showPassword.value
}

// =============================================================================
// 生命周期
// =============================================================================

onMounted(() => {
  // 如果已登录，直接跳转
  if (userStore.isLoggedIn) {
    router.push('/')
  }
})
</script>

<template>
  <div class="login-page">
    <div class="login-container">
      <!-- Logo 区域 -->
      <div class="logo-section">
        <div class="logo">
          <span class="logo-icon">✨</span>
        </div>
        <h1 class="title">欢迎回来</h1>
        <p class="subtitle">登录以获取个性化推荐体验</p>
      </div>

      <!-- 登录表单 -->
      <AuthForm
        type="login"
        :loading="isLoading"
        :error="errorMessage"
        @submit="handleSubmit"
      >
        <template #fields>
          <!-- 邮箱输入 -->
          <div class="form-group">
            <label class="form-label" for="email">邮箱</label>
            <div class="input-wrapper">
              <input
                id="email"
                v-model="fields.email.value"
                type="email"
                class="form-input"
                :class="{ error: fields.email.touched && fields.email.error }"
                placeholder="your@email.com"
                autocomplete="email"
                @blur="handleBlur('email')"
              />
              <span class="input-icon">📧</span>
            </div>
            <Transition name="fade">
              <p v-if="fields.email.touched && fields.email.error" class="field-error">
                {{ fields.email.error }}
              </p>
            </Transition>
          </div>

          <!-- 密码输入 -->
          <div class="form-group">
            <label class="form-label" for="password">密码</label>
            <div class="input-wrapper">
              <input
                id="password"
                v-model="fields.password.value"
                :type="showPassword ? 'text' : 'password'"
                class="form-input"
                :class="{ error: fields.password.touched && fields.password.error }"
                placeholder="••••••••"
                autocomplete="current-password"
                @blur="handleBlur('password')"
              />
              <button
                type="button"
                class="password-toggle"
                @click="togglePasswordVisibility"
              >
                {{ showPassword ? '🙈' : '👁️' }}
              </button>
            </div>
            <Transition name="fade">
              <p v-if="fields.password.touched && fields.password.error" class="field-error">
                {{ fields.password.error }}
              </p>
            </Transition>
          </div>
        </template>

        <template #extra>
          <div class="form-options">
            <label class="remember-me">
              <input type="checkbox" />
              <span>记住我</span>
            </label>
            <a href="#" class="forgot-link">忘记密码？</a>
          </div>
        </template>
      </AuthForm>

      <!-- 注册提示 -->
      <div class="auth-footer">
        <p>
          还没有账号？
          <router-link to="/register" class="register-link">立即注册</router-link>
        </p>
      </div>

      <!-- 其他登录方式（可选） -->
      <div class="social-login">
        <div class="divider">
          <span>或</span>
        </div>
        <div class="social-buttons">
          <button type="button" class="social-btn" title="微信登录">
            <span>💬</span>
          </button>
          <button type="button" class="social-btn" title="QQ登录">
            <span>🐧</span>
          </button>
          <button type="button" class="social-btn" title="手机登录">
            <span>📱</span>
          </button>
        </div>
      </div>
    </div>

    <!-- 背景装饰 -->
    <div class="bg-decoration">
      <div class="circle circle-1"></div>
      <div class="circle circle-2"></div>
      <div class="circle circle-3"></div>
      <div class="grid-overlay"></div>
    </div>
  </div>
</template>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
  position: relative;
  overflow: hidden;
  padding: 2rem;
}

.login-container {
  width: 100%;
  max-width: 420px;
  position: relative;
  z-index: 1;
}

/* Logo 区域 */
.logo-section {
  text-align: center;
  margin-bottom: 2.5rem;
}

.logo {
  width: 80px;
  height: 80px;
  margin: 0 auto 1.5rem;
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  border-radius: 1.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 10px 40px rgba(79, 172, 254, 0.3);
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-10px);
  }
}

.logo-icon {
  font-size: 2.5rem;
}

.title {
  font-size: 2rem;
  font-weight: 700;
  color: #fff;
  margin: 0 0 0.5rem;
  letter-spacing: -0.02em;
}

.subtitle {
  font-size: 1rem;
  color: #8892b0;
  margin: 0;
}

/* 表单样式 */
.form-group {
  margin-bottom: 1.25rem;
}

.form-label {
  display: block;
  font-size: 0.9rem;
  color: #ccd6f6;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

.input-wrapper {
  position: relative;
}

.form-input {
  width: 100%;
  padding: 1rem 1rem 1rem 3rem;
  background: rgba(255, 255, 255, 0.05);
  border: 2px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.75rem;
  color: #fff;
  font-size: 1rem;
  transition: all 0.3s ease;
  box-sizing: border-box;
}

.form-input:focus {
  outline: none;
  border-color: #4facfe;
  box-shadow: 0 0 20px rgba(79, 172, 254, 0.2);
  background: rgba(255, 255, 255, 0.08);
}

.form-input.error {
  border-color: #ff6b6b;
}

.form-input::placeholder {
  color: #8892b0;
}

.input-icon {
  position: absolute;
  left: 1rem;
  top: 50%;
  transform: translateY(-50%);
  font-size: 1rem;
  pointer-events: none;
}

.password-toggle {
  position: absolute;
  right: 1rem;
  top: 50%;
  transform: translateY(-50%);
  background: none;
  border: none;
  font-size: 1rem;
  cursor: pointer;
  padding: 0.25rem;
  opacity: 0.7;
  transition: opacity 0.2s;
}

.password-toggle:hover {
  opacity: 1;
}

/* 字段错误 */
.field-error {
  font-size: 0.8rem;
  color: #ff6b6b;
  margin: 0.5rem 0 0;
}

/* 表单选项 */
.form-options {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.remember-me {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #8892b0;
  font-size: 0.9rem;
  cursor: pointer;
}

.remember-me input {
  accent-color: #4facfe;
}

.forgot-link {
  font-size: 0.9rem;
  color: #4facfe;
  text-decoration: none;
  transition: color 0.2s;
}

.forgot-link:hover {
  color: #00f2fe;
}

/* 注册提示 */
.auth-footer {
  text-align: center;
  margin-top: 2rem;
  color: #8892b0;
}

.register-link {
  color: #4facfe;
  text-decoration: none;
  font-weight: 600;
  transition: color 0.2s;
}

.register-link:hover {
  color: #00f2fe;
  text-decoration: underline;
}

/* 社交登录 */
.social-login {
  margin-top: 2rem;
}

.divider {
  display: flex;
  align-items: center;
  gap: 1rem;
  color: #8892b0;
  font-size: 0.85rem;
  margin-bottom: 1.5rem;
}

.divider::before,
.divider::after {
  content: '';
  flex: 1;
  height: 1px;
  background: rgba(255, 255, 255, 0.1);
}

.social-buttons {
  display: flex;
  justify-content: center;
  gap: 1rem;
}

.social-btn {
  width: 50px;
  height: 50px;
  border-radius: 0.75rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  font-size: 1.5rem;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.social-btn:hover {
  background: rgba(255, 255, 255, 0.1);
  border-color: rgba(255, 255, 255, 0.2);
  transform: translateY(-2px);
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
}

.circle-1 {
  width: 600px;
  height: 600px;
  top: -200px;
  right: -200px;
  background: radial-gradient(circle, rgba(79, 172, 254, 0.08) 0%, transparent 70%);
}

.circle-2 {
  width: 500px;
  height: 500px;
  bottom: -150px;
  left: -150px;
  background: radial-gradient(circle, rgba(0, 242, 254, 0.06) 0%, transparent 70%);
}

.circle-3 {
  width: 300px;
  height: 300px;
  top: 40%;
  right: 15%;
  background: radial-gradient(circle, rgba(167, 139, 250, 0.05) 0%, transparent 70%);
}

.grid-overlay {
  position: absolute;
  inset: 0;
  background-image: 
    linear-gradient(rgba(79, 172, 254, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(79, 172, 254, 0.03) 1px, transparent 1px);
  background-size: 50px 50px;
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: all 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(-5px);
}

/* 响应式 */
@media (max-width: 480px) {
  .login-page {
    padding: 1rem;
  }

  .logo {
    width: 64px;
    height: 64px;
  }

  .title {
    font-size: 1.75rem;
  }

  .form-options {
    flex-direction: column;
    gap: 0.75rem;
    align-items: flex-start;
  }
}
</style>

