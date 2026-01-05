<!--
  Toast 组件
  
  轻提示组件，用于显示临时通知信息
  
  @author Person F
-->
<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'

/**
 * Toast 类型
 */
type ToastType = 'success' | 'error' | 'warning' | 'info'

/**
 * 组件属性
 */
interface Props {
  /** 是否显示 */
  modelValue: boolean
  /** Toast 类型 */
  type?: ToastType
  /** 消息内容 */
  message: string
  /** 显示时长（毫秒），0 表示不自动关闭 */
  duration?: number
  /** 位置 */
  position?: 'top' | 'bottom' | 'center'
}

const props = withDefaults(defineProps<Props>(), {
  type: 'info',
  duration: 3000,
  position: 'top',
})

/**
 * 事件定义
 */
const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'close'): void
}>()

/**
 * 内部可见状态
 */
const visible = ref(props.modelValue)
let timer: ReturnType<typeof setTimeout> | null = null

/**
 * 同步外部值
 */
watch(() => props.modelValue, (val) => {
  visible.value = val
  if (val) {
    startTimer()
  }
})

/**
 * 关闭 Toast
 */
function close() {
  visible.value = false
  emit('update:modelValue', false)
  emit('close')
}

/**
 * 开始计时
 */
function startTimer() {
  clearTimer()
  if (props.duration > 0) {
    timer = setTimeout(close, props.duration)
  }
}

/**
 * 清除计时
 */
function clearTimer() {
  if (timer) {
    clearTimeout(timer)
    timer = null
  }
}

/**
 * 类型图标映射
 */
const iconMap: Record<ToastType, string> = {
  success: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
  error: 'M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z',
  warning: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
  info: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
}

onMounted(() => {
  if (props.modelValue) {
    startTimer()
  }
})
</script>

<template>
  <Teleport to="body">
    <Transition name="toast">
      <div
        v-if="visible"
        class="toast"
        :class="[`toast--${type}`, `toast--${position}`]"
        role="alert"
        @mouseenter="clearTimer"
        @mouseleave="startTimer"
      >
        <!-- 图标 -->
        <svg class="toast__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path :d="iconMap[type]" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        
        <!-- 消息 -->
        <span class="toast__message">{{ message }}</span>
        
        <!-- 关闭按钮 -->
        <button
          type="button"
          class="toast__close"
          aria-label="关闭"
          @click="close"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M6 18L18 6M6 6l12 12" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </button>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.toast {
  position: fixed;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: #1a1a2e;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  z-index: 10000;
  max-width: 90vw;
}

.toast--top {
  top: 24px;
}

.toast--bottom {
  bottom: 24px;
}

.toast--center {
  top: 50%;
  transform: translate(-50%, -50%);
}

.toast__icon {
  flex-shrink: 0;
  width: 20px;
  height: 20px;
}

.toast--success .toast__icon {
  color: #10b981;
}

.toast--error .toast__icon {
  color: #ef4444;
}

.toast--warning .toast__icon {
  color: #f59e0b;
}

.toast--info .toast__icon {
  color: #4facfe;
}

.toast__message {
  flex: 1;
  font-size: 14px;
  color: #e2e8f0;
  line-height: 1.5;
}

.toast__close {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  padding: 0;
  border: none;
  background: none;
  color: #8892b0;
  cursor: pointer;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.toast__close:hover {
  color: #e2e8f0;
  background: rgba(255, 255, 255, 0.1);
}

.toast__close svg {
  width: 16px;
  height: 16px;
}

/* 过渡动画 */
.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}

.toast--top.toast-enter-from,
.toast--top.toast-leave-to {
  opacity: 0;
  transform: translate(-50%, -20px);
}

.toast--bottom.toast-enter-from,
.toast--bottom.toast-leave-to {
  opacity: 0;
  transform: translate(-50%, 20px);
}

.toast--center.toast-enter-from,
.toast--center.toast-leave-to {
  opacity: 0;
  transform: translate(-50%, -50%) scale(0.9);
}
</style>

