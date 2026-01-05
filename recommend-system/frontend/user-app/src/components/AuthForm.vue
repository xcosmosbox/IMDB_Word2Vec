/**
 * AuthForm - 认证表单组件
 * 
 * 通用的认证表单组件，支持登录和注册模式。
 * 提供统一的表单布局、加载状态、错误提示。
 * 
 * @component
 * @author Person C
 */
<script setup lang="ts">
import { computed } from 'vue'

// =============================================================================
// Props 定义
// =============================================================================

interface Props {
  /** 表单类型：登录或注册 */
  type: 'login' | 'register'
  /** 加载状态 */
  loading?: boolean
  /** 错误信息 */
  error?: string
  /** 提交按钮文字 */
  submitText?: string
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
  error: '',
  submitText: '',
})

// =============================================================================
// Emits 定义
// =============================================================================

const emit = defineEmits<{
  /** 表单提交事件 */
  submit: []
}>()

// =============================================================================
// 计算属性
// =============================================================================

/** 按钮文字 */
const buttonText = computed(() => {
  if (props.submitText) return props.submitText
  return props.type === 'login' ? '登录' : '注册'
})

/** 表单标题 */
const formTitle = computed(() => {
  return props.type === 'login' ? '账号登录' : '创建账号'
})

// =============================================================================
// 事件处理
// =============================================================================

/**
 * 处理表单提交
 */
function handleSubmit(event: Event) {
  event.preventDefault()
  if (!props.loading) {
    emit('submit')
  }
}
</script>

<template>
  <form class="auth-form" @submit="handleSubmit">
    <!-- 表单头部 -->
    <div class="form-header">
      <h2 class="form-title">{{ formTitle }}</h2>
    </div>

    <!-- 错误提示 -->
    <Transition name="fade">
      <div v-if="error" class="error-alert" role="alert">
        <span class="error-icon">⚠️</span>
        <span class="error-text">{{ error }}</span>
      </div>
    </Transition>

    <!-- 表单字段插槽 -->
    <div class="form-fields">
      <slot name="fields"></slot>
    </div>

    <!-- 额外内容插槽（如记住密码、忘记密码等） -->
    <div v-if="$slots.extra" class="form-extra">
      <slot name="extra"></slot>
    </div>

    <!-- 提交按钮 -->
    <button
      type="submit"
      class="submit-btn"
      :class="{ loading: loading }"
      :disabled="loading"
    >
      <span v-if="loading" class="loading-spinner"></span>
      <span class="btn-text">{{ buttonText }}</span>
    </button>

    <!-- 底部插槽（如其他登录方式） -->
    <div v-if="$slots.footer" class="form-footer">
      <slot name="footer"></slot>
    </div>
  </form>
</template>

<style scoped>
.auth-form {
  width: 100%;
  max-width: 400px;
}

.form-header {
  margin-bottom: 2rem;
}

.form-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: #fff;
  text-align: center;
  margin: 0;
}

/* 错误提示 */
.error-alert {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background: rgba(255, 107, 107, 0.15);
  border: 1px solid rgba(255, 107, 107, 0.3);
  border-radius: 0.75rem;
  margin-bottom: 1.5rem;
}

.error-icon {
  font-size: 1.25rem;
  flex-shrink: 0;
}

.error-text {
  color: #ff6b6b;
  font-size: 0.9rem;
  line-height: 1.4;
}

/* 表单字段 */
.form-fields {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

/* 额外内容 */
.form-extra {
  margin-top: 1rem;
  margin-bottom: 1rem;
}

/* 提交按钮 */
.submit-btn {
  width: 100%;
  padding: 1rem;
  margin-top: 1.5rem;
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  border: none;
  border-radius: 0.75rem;
  color: #fff;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.submit-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(79, 172, 254, 0.4);
}

.submit-btn:active:not(:disabled) {
  transform: translateY(0);
}

.submit-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.submit-btn.loading {
  pointer-events: none;
}

/* 加载动画 */
.loading-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid transparent;
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* 表单底部 */
.form-footer {
  margin-top: 2rem;
  padding-top: 1.5rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: all 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>

