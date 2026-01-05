<!--
  Button 组件
  
  通用按钮组件，支持多种样式和状态
  
  @author Person F
-->
<script setup lang="ts">
import { computed } from 'vue'

/**
 * 按钮类型
 */
type ButtonType = 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger'

/**
 * 按钮尺寸
 */
type ButtonSize = 'small' | 'default' | 'large'

/**
 * 组件属性
 */
interface Props {
  /** 按钮类型 */
  type?: ButtonType
  /** 按钮尺寸 */
  size?: ButtonSize
  /** 是否禁用 */
  disabled?: boolean
  /** 是否加载中 */
  loading?: boolean
  /** 是否块级按钮 */
  block?: boolean
  /** 是否圆形按钮 */
  round?: boolean
  /** 原生按钮类型 */
  htmlType?: 'button' | 'submit' | 'reset'
  /** 图标位置 */
  iconPosition?: 'left' | 'right'
}

const props = withDefaults(defineProps<Props>(), {
  type: 'primary',
  size: 'default',
  disabled: false,
  loading: false,
  block: false,
  round: false,
  htmlType: 'button',
  iconPosition: 'left',
})

/**
 * 事件定义
 */
const emit = defineEmits<{
  (e: 'click', event: MouseEvent): void
}>()

/**
 * 计算类名
 */
const classes = computed(() => [
  'button',
  `button--${props.type}`,
  `button--${props.size}`,
  {
    'button--disabled': props.disabled,
    'button--loading': props.loading,
    'button--block': props.block,
    'button--round': props.round,
  },
])

/**
 * 点击处理
 */
function handleClick(event: MouseEvent) {
  if (props.disabled || props.loading) {
    event.preventDefault()
    return
  }
  emit('click', event)
}
</script>

<template>
  <button
    :class="classes"
    :type="htmlType"
    :disabled="disabled || loading"
    @click="handleClick"
  >
    <!-- 加载动画 -->
    <span v-if="loading" class="button__loading">
      <svg viewBox="0 0 24 24" class="button__spinner">
        <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="31.4 31.4" />
      </svg>
    </span>
    
    <!-- 左侧图标 -->
    <span v-if="$slots.icon && iconPosition === 'left' && !loading" class="button__icon">
      <slot name="icon"></slot>
    </span>
    
    <!-- 按钮内容 -->
    <span class="button__content">
      <slot></slot>
    </span>
    
    <!-- 右侧图标 -->
    <span v-if="$slots.icon && iconPosition === 'right' && !loading" class="button__icon">
      <slot name="icon"></slot>
    </span>
  </button>
</template>

<style scoped>
.button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  font-weight: 500;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  border: none;
  outline: none;
  white-space: nowrap;
  user-select: none;
}

/* 尺寸 */
.button--small {
  padding: 6px 12px;
  font-size: 12px;
  border-radius: 6px;
}

.button--default {
  padding: 10px 20px;
  font-size: 14px;
}

.button--large {
  padding: 14px 28px;
  font-size: 16px;
}

/* 类型样式 */
.button--primary {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  color: #0f0f23;
}

.button--primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(79, 172, 254, 0.4);
}

.button--secondary {
  background: rgba(255, 255, 255, 0.1);
  color: #e2e8f0;
}

.button--secondary:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.15);
}

.button--outline {
  background: transparent;
  color: #4facfe;
  border: 1px solid #4facfe;
}

.button--outline:hover:not(:disabled) {
  background: rgba(79, 172, 254, 0.1);
}

.button--ghost {
  background: transparent;
  color: #8892b0;
}

.button--ghost:hover:not(:disabled) {
  color: #e2e8f0;
  background: rgba(255, 255, 255, 0.05);
}

.button--danger {
  background: #ef4444;
  color: #fff;
}

.button--danger:hover:not(:disabled) {
  background: #dc2626;
}

/* 状态 */
.button--disabled,
.button--loading {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none !important;
  box-shadow: none !important;
}

.button--block {
  width: 100%;
}

.button--round {
  border-radius: 999px;
}

/* 加载动画 */
.button__loading {
  display: flex;
  align-items: center;
  justify-content: center;
}

.button__spinner {
  width: 16px;
  height: 16px;
  animation: spin 0.8s linear infinite;
}

.button--large .button__spinner {
  width: 20px;
  height: 20px;
}

.button--small .button__spinner {
  width: 14px;
  height: 14px;
}

/* 图标 */
.button__icon {
  display: flex;
  align-items: center;
  justify-content: center;
}

.button__icon svg {
  width: 18px;
  height: 18px;
}

.button--small .button__icon svg {
  width: 14px;
  height: 14px;
}

.button--large .button__icon svg {
  width: 20px;
  height: 20px;
}

/* 内容 */
.button__content {
  display: flex;
  align-items: center;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>

