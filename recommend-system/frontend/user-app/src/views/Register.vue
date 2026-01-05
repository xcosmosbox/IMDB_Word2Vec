/**
 * Register - 注册页面
 * 
 * 用户注册页面，支持创建新账号。
 * 注册成功后自动登录并跳转至首页。
 * 
 * @view
 * @author Person C
 */
<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useUserStore } from '@/stores/user'
import AuthForm from '@/components/AuthForm.vue'
import {
  useFormValidation,
  required,
  email,
  minLength,
  maxLength,
  passwordStrength,
  confirmPassword,
  numberRange,
} from '@/composables/useFormValidation'

// =============================================================================
// 依赖注入
// =============================================================================

const router = useRouter()
const userStore = useUserStore()

// =============================================================================
// 状态
// =============================================================================

const isLoading = ref(false)
const errorMessage = ref('')
const showPassword = ref(false)
const showConfirmPassword = ref(false)
const agreeTerms = ref(false)

// =============================================================================
// 表单验证
// =============================================================================

const { fields, validate } = useFormValidation({
  name: {
    value: '',
    rules: [
      required('请输入昵称'),
      minLength(2, '昵称至少2个字符'),
      maxLength(20, '昵称最多20个字符'),
    ],
  },
  email: {
    value: '',
    rules: [required('请输入邮箱'), email('邮箱格式不正确')],
  },
  password: {
    value: '',
    rules: [
      required('请输入密码'),
      passwordStrength('密码需包含字母和数字，长度至少6位'),
    ],
  },
  confirmPassword: {
    value: '',
    rules: [
      required('请确认密码'),
      confirmPassword(() => fields.password.value, '两次密码输入不一致'),
    ],
  },
  age: {
    value: null as number | null,
    rules: [numberRange(1, 150, '请输入有效年龄')],
  },
  gender: {
    value: '',
    rules: [],
  },
})

// =============================================================================
// 事件处理
// =============================================================================

/**
 * 处理注册提交
 */
async function handleSubmit() {
  // 表单验证
  if (!validate()) {
    return
  }

  // 检查协议
  if (!agreeTerms.value) {
    errorMessage.value = '请阅读并同意用户协议'
    return
  }

  errorMessage.value = ''
  isLoading.value = true

  try {
    await userStore.register({
      name: fields.name.value,
      email: fields.email.value,
      password: fields.password.value,
      age: fields.age.value || undefined,
      gender: fields.gender.value || undefined,
    })

    // 注册成功，跳转至首页
    router.push('/')
  } catch (error: any) {
    errorMessage.value = error?.message || '注册失败，请稍后重试'
  } finally {
    isLoading.value = false
  }
}

/**
 * 处理字段失焦
 */
function handleBlur(fieldName: keyof typeof fields) {
  fields[fieldName].touched = true
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
  <div class="register-page">
    <div class="register-container">
      <!-- Logo 区域 -->
      <div class="logo-section">
        <div class="logo">
          <span class="logo-icon">🚀</span>
        </div>
        <h1 class="title">创建账号</h1>
        <p class="subtitle">开启你的个性化推荐之旅</p>
      </div>

      <!-- 注册表单 -->
      <AuthForm
        type="register"
        :loading="isLoading"
        :error="errorMessage"
        submit-text="注册"
        @submit="handleSubmit"
      >
        <template #fields>
          <!-- 昵称输入 -->
          <div class="form-group">
            <label class="form-label" for="name">昵称 <span class="required">*</span></label>
            <div class="input-wrapper">
              <input
                id="name"
                v-model="fields.name.value"
                type="text"
                class="form-input"
                :class="{ error: fields.name.touched && fields.name.error }"
                placeholder="给自己取个名字"
                autocomplete="name"
                @blur="handleBlur('name')"
              />
              <span class="input-icon">👤</span>
            </div>
            <Transition name="fade">
              <p v-if="fields.name.touched && fields.name.error" class="field-error">
                {{ fields.name.error }}
              </p>
            </Transition>
          </div>

          <!-- 邮箱输入 -->
          <div class="form-group">
            <label class="form-label" for="email">邮箱 <span class="required">*</span></label>
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
          <div class="form-row">
            <div class="form-group">
              <label class="form-label" for="password">密码 <span class="required">*</span></label>
              <div class="input-wrapper">
                <input
                  id="password"
                  v-model="fields.password.value"
                  :type="showPassword ? 'text' : 'password'"
                  class="form-input"
                  :class="{ error: fields.password.touched && fields.password.error }"
                  placeholder="至少6位"
                  autocomplete="new-password"
                  @blur="handleBlur('password')"
                />
                <button
                  type="button"
                  class="password-toggle"
                  @click="showPassword = !showPassword"
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

            <div class="form-group">
              <label class="form-label" for="confirmPassword">确认密码 <span class="required">*</span></label>
              <div class="input-wrapper">
                <input
                  id="confirmPassword"
                  v-model="fields.confirmPassword.value"
                  :type="showConfirmPassword ? 'text' : 'password'"
                  class="form-input"
                  :class="{ error: fields.confirmPassword.touched && fields.confirmPassword.error }"
                  placeholder="再次输入"
                  autocomplete="new-password"
                  @blur="handleBlur('confirmPassword')"
                />
                <button
                  type="button"
                  class="password-toggle"
                  @click="showConfirmPassword = !showConfirmPassword"
                >
                  {{ showConfirmPassword ? '🙈' : '👁️' }}
                </button>
              </div>
              <Transition name="fade">
                <p v-if="fields.confirmPassword.touched && fields.confirmPassword.error" class="field-error">
                  {{ fields.confirmPassword.error }}
                </p>
              </Transition>
            </div>
          </div>

          <!-- 可选信息 -->
          <div class="optional-section">
            <div class="section-title">
              <span>可选信息</span>
              <span class="hint">帮助我们更好地推荐</span>
            </div>

            <div class="form-row">
              <div class="form-group">
                <label class="form-label" for="age">年龄</label>
                <input
                  id="age"
                  v-model.number="fields.age.value"
                  type="number"
                  class="form-input"
                  placeholder="年龄"
                  min="1"
                  max="150"
                />
              </div>

              <div class="form-group">
                <label class="form-label">性别</label>
                <div class="gender-options">
                  <label class="gender-option">
                    <input
                      v-model="fields.gender.value"
                      type="radio"
                      value="male"
                      name="gender"
                    />
                    <span class="option-label">男</span>
                  </label>
                  <label class="gender-option">
                    <input
                      v-model="fields.gender.value"
                      type="radio"
                      value="female"
                      name="gender"
                    />
                    <span class="option-label">女</span>
                  </label>
                  <label class="gender-option">
                    <input
                      v-model="fields.gender.value"
                      type="radio"
                      value="other"
                      name="gender"
                    />
                    <span class="option-label">其他</span>
                  </label>
                </div>
              </div>
            </div>
          </div>

          <!-- 用户协议 -->
          <div class="terms-section">
            <label class="terms-checkbox">
              <input v-model="agreeTerms" type="checkbox" />
              <span class="checkbox-text">
                我已阅读并同意
                <a href="#" class="terms-link">用户协议</a>
                和
                <a href="#" class="terms-link">隐私政策</a>
              </span>
            </label>
          </div>
        </template>
      </AuthForm>

      <!-- 登录提示 -->
      <div class="auth-footer">
        <p>
          已有账号？
          <router-link to="/login" class="login-link">立即登录</router-link>
        </p>
      </div>
    </div>

    <!-- 背景装饰 -->
    <div class="bg-decoration">
      <div class="circle circle-1"></div>
      <div class="circle circle-2"></div>
      <div class="floating-shapes">
        <div class="shape shape-1">⭐</div>
        <div class="shape shape-2">💫</div>
        <div class="shape shape-3">🎯</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.register-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
  position: relative;
  overflow: hidden;
  padding: 2rem;
}

.register-container {
  width: 100%;
  max-width: 480px;
  position: relative;
  z-index: 1;
}

/* Logo 区域 */
.logo-section {
  text-align: center;
  margin-bottom: 2rem;
}

.logo {
  width: 72px;
  height: 72px;
  margin: 0 auto 1.25rem;
  background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
  border-radius: 1.25rem;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 10px 40px rgba(167, 139, 250, 0.3);
}

.logo-icon {
  font-size: 2rem;
}

.title {
  font-size: 1.75rem;
  font-weight: 700;
  color: #fff;
  margin: 0 0 0.5rem;
}

.subtitle {
  font-size: 0.95rem;
  color: #8892b0;
  margin: 0;
}

/* 表单样式 */
.form-group {
  margin-bottom: 1rem;
  flex: 1;
}

.form-row {
  display: flex;
  gap: 1rem;
}

.form-label {
  display: block;
  font-size: 0.85rem;
  color: #ccd6f6;
  margin-bottom: 0.4rem;
  font-weight: 500;
}

.required {
  color: #f472b6;
}

.input-wrapper {
  position: relative;
}

.form-input {
  width: 100%;
  padding: 0.875rem 1rem 0.875rem 2.75rem;
  background: rgba(255, 255, 255, 0.05);
  border: 2px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.75rem;
  color: #fff;
  font-size: 0.95rem;
  transition: all 0.3s ease;
  box-sizing: border-box;
}

.form-input:focus {
  outline: none;
  border-color: #a78bfa;
  box-shadow: 0 0 15px rgba(167, 139, 250, 0.2);
}

.form-input.error {
  border-color: #ff6b6b;
}

.form-input::placeholder {
  color: #8892b0;
}

.input-icon {
  position: absolute;
  left: 0.875rem;
  top: 50%;
  transform: translateY(-50%);
  font-size: 0.9rem;
  pointer-events: none;
}

.password-toggle {
  position: absolute;
  right: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  background: none;
  border: none;
  font-size: 0.9rem;
  cursor: pointer;
  padding: 0.25rem;
  opacity: 0.7;
}

/* 字段错误 */
.field-error {
  font-size: 0.75rem;
  color: #ff6b6b;
  margin: 0.35rem 0 0;
}

/* 可选信息区域 */
.optional-section {
  margin-top: 1.5rem;
  padding-top: 1.5rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.section-title {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1rem;
  color: #ccd6f6;
  font-size: 0.9rem;
  font-weight: 500;
}

.hint {
  font-size: 0.8rem;
  color: #8892b0;
  font-weight: 400;
}

.optional-section .form-input {
  padding-left: 1rem;
}

/* 性别选项 */
.gender-options {
  display: flex;
  gap: 0.75rem;
  padding-top: 0.25rem;
}

.gender-option {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  cursor: pointer;
}

.gender-option input {
  accent-color: #a78bfa;
}

.option-label {
  color: #ccd6f6;
  font-size: 0.9rem;
}

/* 用户协议 */
.terms-section {
  margin-top: 1.25rem;
}

.terms-checkbox {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  cursor: pointer;
}

.terms-checkbox input {
  margin-top: 0.15rem;
  accent-color: #a78bfa;
}

.checkbox-text {
  font-size: 0.85rem;
  color: #8892b0;
  line-height: 1.4;
}

.terms-link {
  color: #a78bfa;
  text-decoration: none;
}

.terms-link:hover {
  text-decoration: underline;
}

/* 登录提示 */
.auth-footer {
  text-align: center;
  margin-top: 1.5rem;
  color: #8892b0;
}

.login-link {
  color: #a78bfa;
  text-decoration: none;
  font-weight: 600;
}

.login-link:hover {
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
}

.circle-1 {
  width: 500px;
  height: 500px;
  top: -150px;
  left: -150px;
  background: radial-gradient(circle, rgba(167, 139, 250, 0.08) 0%, transparent 70%);
}

.circle-2 {
  width: 400px;
  height: 400px;
  bottom: -100px;
  right: -100px;
  background: radial-gradient(circle, rgba(244, 114, 182, 0.06) 0%, transparent 70%);
}

.floating-shapes {
  position: absolute;
  inset: 0;
}

.shape {
  position: absolute;
  font-size: 1.5rem;
  opacity: 0.3;
  animation: float-shape 6s ease-in-out infinite;
}

.shape-1 {
  top: 20%;
  right: 15%;
  animation-delay: 0s;
}

.shape-2 {
  top: 60%;
  left: 10%;
  animation-delay: 2s;
}

.shape-3 {
  bottom: 25%;
  right: 20%;
  animation-delay: 4s;
}

@keyframes float-shape {
  0%, 100% {
    transform: translateY(0) rotate(0deg);
  }
  50% {
    transform: translateY(-20px) rotate(10deg);
  }
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: all 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* 响应式 */
@media (max-width: 540px) {
  .form-row {
    flex-direction: column;
    gap: 0;
  }

  .gender-options {
    flex-wrap: wrap;
  }
}
</style>

