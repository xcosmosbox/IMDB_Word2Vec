<!--
  Input 组件
  
  通用输入框组件，支持多种类型和状态
  
  @author Person F
-->
<script setup lang="ts">
import { ref, computed } from 'vue'

/**
 * 输入框尺寸
 */
type InputSize = 'small' | 'default' | 'large'

/**
 * 组件属性
 */
interface Props {
  /** 绑定值 */
  modelValue: string
  /** 输入框类型 */
  type?: 'text' | 'password' | 'email' | 'number' | 'tel' | 'url' | 'search'
  /** 占位符 */
  placeholder?: string
  /** 尺寸 */
  size?: InputSize
  /** 是否禁用 */
  disabled?: boolean
  /** 是否只读 */
  readonly?: boolean
  /** 是否可清除 */
  clearable?: boolean
  /** 是否显示密码切换 */
  showPassword?: boolean
  /** 最大长度 */
  maxlength?: number
  /** 是否显示字数统计 */
  showCount?: boolean
  /** 错误信息 */
  error?: string
  /** 是否自动聚焦 */
  autofocus?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  type: 'text',
  size: 'default',
  disabled: false,
  readonly: false,
  clearable: false,
  showPassword: false,
  showCount: false,
  autofocus: false,
})

/**
 * 事件定义
 */
const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
  (e: 'input', value: string): void
  (e: 'change', value: string): void
  (e: 'focus', event: FocusEvent): void
  (e: 'blur', event: FocusEvent): void
  (e: 'clear'): void
  (e: 'keydown', event: KeyboardEvent): void
  (e: 'keyup', event: KeyboardEvent): void
  (e: 'enter', event: KeyboardEvent): void
}>()

/**
 * 内部状态
 */
const inputRef = ref<HTMLInputElement | null>(null)
const isFocused = ref(false)
const passwordVisible = ref(false)

/**
 * 计算实际输入类型
 */
const inputType = computed(() => {
  if (props.type === 'password' && props.showPassword) {
    return passwordVisible.value ? 'text' : 'password'
  }
  return props.type
})

/**
 * 计算类名
 */
const classes = computed(() => [
  'input',
  `input--${props.size}`,
  {
    'input--focused': isFocused.value,
    'input--disabled': props.disabled,
    'input--error': !!props.error,
    'input--has-prefix': !!slots.prefix,
    'input--has-suffix': !!slots.suffix || props.clearable || props.showPassword,
  },
])

/**
 * 是否显示清除按钮
 */
const showClear = computed(() => {
  return props.clearable && !props.disabled && !props.readonly && props.modelValue
})

/**
 * 字数统计
 */
const count = computed(() => {
  const len = props.modelValue.length
  if (props.maxlength) {
    return `${len}/${props.maxlength}`
  }
  return String(len)
})

/**
 * 处理输入
 */
function handleInput(event: Event) {
  const target = event.target as HTMLInputElement
  const value = target.value
  emit('update:modelValue', value)
  emit('input', value)
}

/**
 * 处理变化
 */
function handleChange(event: Event) {
  const target = event.target as HTMLInputElement
  emit('change', target.value)
}

/**
 * 处理聚焦
 */
function handleFocus(event: FocusEvent) {
  isFocused.value = true
  emit('focus', event)
}

/**
 * 处理失焦
 */
function handleBlur(event: FocusEvent) {
  isFocused.value = false
  emit('blur', event)
}

/**
 * 处理清除
 */
function handleClear() {
  emit('update:modelValue', '')
  emit('clear')
  inputRef.value?.focus()
}

/**
 * 切换密码可见性
 */
function togglePassword() {
  passwordVisible.value = !passwordVisible.value
}

/**
 * 处理键盘事件
 */
function handleKeydown(event: KeyboardEvent) {
  emit('keydown', event)
  if (event.key === 'Enter') {
    emit('enter', event)
  }
}

function handleKeyup(event: KeyboardEvent) {
  emit('keyup', event)
}

/**
 * 获取插槽
 */
const slots = defineSlots<{
  prefix?: () => unknown
  suffix?: () => unknown
}>()

/**
 * 暴露方法
 */
defineExpose({
  focus: () => inputRef.value?.focus(),
  blur: () => inputRef.value?.blur(),
  select: () => inputRef.value?.select(),
})
</script>

<template>
  <div :class="classes">
    <!-- 前缀 -->
    <span v-if="$slots.prefix" class="input__prefix">
      <slot name="prefix"></slot>
    </span>
    
    <!-- 输入框 -->
    <input
      ref="inputRef"
      :type="inputType"
      :value="modelValue"
      :placeholder="placeholder"
      :disabled="disabled"
      :readonly="readonly"
      :maxlength="maxlength"
      :autofocus="autofocus"
      class="input__inner"
      @input="handleInput"
      @change="handleChange"
      @focus="handleFocus"
      @blur="handleBlur"
      @keydown="handleKeydown"
      @keyup="handleKeyup"
    />
    
    <!-- 后缀区域 -->
    <span v-if="showClear || showPassword || $slots.suffix || showCount" class="input__suffix">
      <!-- 字数统计 -->
      <span v-if="showCount" class="input__count">{{ count }}</span>
      
      <!-- 清除按钮 -->
      <button
        v-if="showClear"
        type="button"
        class="input__clear"
        tabindex="-1"
        @click="handleClear"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M15 9l-6 6M9 9l6 6" stroke-linecap="round" />
        </svg>
      </button>
      
      <!-- 密码切换 -->
      <button
        v-if="showPassword && type === 'password'"
        type="button"
        class="input__password-toggle"
        tabindex="-1"
        @click="togglePassword"
      >
        <svg v-if="passwordVisible" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
          <circle cx="12" cy="12" r="3" />
        </svg>
        <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
          <line x1="1" y1="1" x2="23" y2="23" />
        </svg>
      </button>
      
      <!-- 自定义后缀 -->
      <slot name="suffix"></slot>
    </span>
    
    <!-- 错误信息 -->
    <span v-if="error" class="input__error">{{ error }}</span>
  </div>
</template>

<style scoped>
.input {
  position: relative;
  display: flex;
  align-items: center;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  transition: all 0.2s ease;
}

/* 尺寸 */
.input--small {
  height: 32px;
  font-size: 13px;
}

.input--default {
  height: 40px;
  font-size: 14px;
}

.input--large {
  height: 48px;
  font-size: 16px;
}

/* 状态 */
.input--focused {
  border-color: #4facfe;
  box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
}

.input--disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.input--error {
  border-color: #ef4444;
}

.input--error.input--focused {
  box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.1);
}

/* 内部输入框 */
.input__inner {
  flex: 1;
  width: 100%;
  height: 100%;
  padding: 0 12px;
  border: none;
  background: transparent;
  color: #e2e8f0;
  font-size: inherit;
  outline: none;
}

.input__inner::placeholder {
  color: #64748b;
}

.input__inner:disabled {
  cursor: not-allowed;
}

/* 前缀 */
.input__prefix {
  display: flex;
  align-items: center;
  padding-left: 12px;
  color: #8892b0;
}

.input--has-prefix .input__inner {
  padding-left: 8px;
}

/* 后缀 */
.input__suffix {
  display: flex;
  align-items: center;
  gap: 4px;
  padding-right: 8px;
  color: #8892b0;
}

.input--has-suffix .input__inner {
  padding-right: 8px;
}

/* 清除按钮 */
.input__clear,
.input__password-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  padding: 0;
  border: none;
  background: none;
  color: #64748b;
  cursor: pointer;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.input__clear:hover,
.input__password-toggle:hover {
  color: #e2e8f0;
  background: rgba(255, 255, 255, 0.1);
}

.input__clear svg,
.input__password-toggle svg {
  width: 16px;
  height: 16px;
}

/* 字数统计 */
.input__count {
  font-size: 12px;
  color: #64748b;
  padding: 0 4px;
}

/* 错误信息 */
.input__error {
  position: absolute;
  left: 0;
  bottom: -20px;
  font-size: 12px;
  color: #ef4444;
}
</style>

