<!--
  Error 组件
  
  错误状态展示组件，用于显示各种错误信息
  
  @author Person F
-->
<script setup lang="ts">
import { computed } from 'vue'

/**
 * 错误类型
 */
type ErrorType = 'default' | 'network' | 'notfound' | 'permission' | 'server'

/**
 * 组件属性
 */
interface Props {
  /** 错误类型 */
  type?: ErrorType
  /** 错误标题 */
  title?: string
  /** 错误描述 */
  description?: string
  /** 是否显示重试按钮 */
  showRetry?: boolean
  /** 重试按钮文本 */
  retryText?: string
  /** 是否显示返回按钮 */
  showBack?: boolean
  /** 返回按钮文本 */
  backText?: string
}

const props = withDefaults(defineProps<Props>(), {
  type: 'default',
  showRetry: true,
  retryText: '重试',
  showBack: false,
  backText: '返回',
})

/**
 * 事件定义
 */
const emit = defineEmits<{
  (e: 'retry'): void
  (e: 'back'): void
}>()

/**
 * 错误类型配置
 */
const errorConfig: Record<ErrorType, { title: string; description: string; icon: string }> = {
  default: {
    title: '出错了',
    description: '抱歉，发生了一些错误',
    icon: 'error',
  },
  network: {
    title: '网络错误',
    description: '请检查您的网络连接',
    icon: 'wifi-off',
  },
  notfound: {
    title: '页面未找到',
    description: '抱歉，您访问的页面不存在',
    icon: '404',
  },
  permission: {
    title: '无权限访问',
    description: '您没有权限访问此内容',
    icon: 'lock',
  },
  server: {
    title: '服务器错误',
    description: '服务器出了一些问题，请稍后重试',
    icon: 'server',
  },
}

/**
 * 当前错误配置
 */
const currentConfig = computed(() => errorConfig[props.type] || errorConfig.default)

/**
 * 显示的标题
 */
const displayTitle = computed(() => props.title || currentConfig.value.title)

/**
 * 显示的描述
 */
const displayDescription = computed(() => props.description || currentConfig.value.description)

/**
 * 重试处理
 */
function handleRetry() {
  emit('retry')
}

/**
 * 返回处理
 */
function handleBack() {
  emit('back')
}
</script>

<template>
  <div class="error" role="alert">
    <!-- 错误图标 -->
    <div class="error__icon">
      <!-- 默认错误图标 -->
      <svg
        v-if="type === 'default' || type === 'server'"
        viewBox="0 0 64 64"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <circle cx="32" cy="32" r="28" stroke="currentColor" stroke-width="2" />
        <path d="M32 18V38" stroke="currentColor" stroke-width="3" stroke-linecap="round" />
        <circle cx="32" cy="46" r="2" fill="currentColor" />
      </svg>
      
      <!-- 网络错误图标 -->
      <svg
        v-else-if="type === 'network'"
        viewBox="0 0 64 64"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path d="M12 28C20 16 44 16 52 28" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
        <path d="M18 36C24 28 40 28 46 36" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
        <path d="M24 44C28 40 36 40 40 44" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
        <path d="M16 16L48 48" stroke="#ef4444" stroke-width="3" stroke-linecap="round" />
      </svg>
      
      <!-- 404 图标 -->
      <svg
        v-else-if="type === 'notfound'"
        viewBox="0 0 64 64"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <text x="8" y="42" font-size="24" font-weight="bold" fill="currentColor">404</text>
      </svg>
      
      <!-- 权限错误图标 -->
      <svg
        v-else-if="type === 'permission'"
        viewBox="0 0 64 64"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <rect x="16" y="28" width="32" height="24" rx="4" stroke="currentColor" stroke-width="2" />
        <path d="M24 28V20C24 15.5817 27.5817 12 32 12C36.4183 12 40 15.5817 40 20V28" stroke="currentColor" stroke-width="2" />
        <circle cx="32" cy="40" r="3" fill="currentColor" />
      </svg>
    </div>
    
    <!-- 错误标题 -->
    <h2 class="error__title">{{ displayTitle }}</h2>
    
    <!-- 错误描述 -->
    <p class="error__description">{{ displayDescription }}</p>
    
    <!-- 操作按钮 -->
    <div class="error__actions">
      <button
        v-if="showRetry"
        type="button"
        class="error__button error__button--primary"
        @click="handleRetry"
      >
        {{ retryText }}
      </button>
      <button
        v-if="showBack"
        type="button"
        class="error__button error__button--secondary"
        @click="handleBack"
      >
        {{ backText }}
      </button>
    </div>
    
    <!-- 额外内容插槽 -->
    <div v-if="$slots.default" class="error__extra">
      <slot></slot>
    </div>
  </div>
</template>

<style scoped>
.error {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 48px 24px;
  text-align: center;
  min-height: 300px;
}

.error__icon {
  width: 80px;
  height: 80px;
  margin-bottom: 24px;
  color: #ef4444;
}

.error__icon svg {
  width: 100%;
  height: 100%;
}

.error__title {
  margin: 0 0 12px;
  font-size: 24px;
  font-weight: 600;
  color: #e2e8f0;
}

.error__description {
  margin: 0 0 32px;
  font-size: 14px;
  color: #8892b0;
  max-width: 400px;
  line-height: 1.6;
}

.error__actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
}

.error__button {
  padding: 10px 24px;
  font-size: 14px;
  font-weight: 500;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  border: none;
  outline: none;
}

.error__button--primary {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  color: #0f0f23;
}

.error__button--primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(79, 172, 254, 0.4);
}

.error__button--primary:active {
  transform: translateY(0);
}

.error__button--secondary {
  background: rgba(255, 255, 255, 0.1);
  color: #e2e8f0;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.error__button--secondary:hover {
  background: rgba(255, 255, 255, 0.15);
}

.error__extra {
  margin-top: 24px;
}
</style>

