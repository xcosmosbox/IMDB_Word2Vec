<!--
  Modal 组件
  
  模态框组件，支持确认框、信息框等多种模式
  
  @author Person F
-->
<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'

/**
 * 组件属性
 */
interface Props {
  /** 是否显示 */
  modelValue: boolean
  /** 标题 */
  title?: string
  /** 宽度 */
  width?: string | number
  /** 是否显示关闭按钮 */
  showClose?: boolean
  /** 点击遮罩是否关闭 */
  closeOnClickOverlay?: boolean
  /** 按 ESC 是否关闭 */
  closeOnEsc?: boolean
  /** 是否显示底部按钮 */
  showFooter?: boolean
  /** 确认按钮文本 */
  confirmText?: string
  /** 取消按钮文本 */
  cancelText?: string
  /** 确认按钮是否加载中 */
  confirmLoading?: boolean
  /** 是否居中显示 */
  centered?: boolean
  /** z-index */
  zIndex?: number
}

const props = withDefaults(defineProps<Props>(), {
  width: '480px',
  showClose: true,
  closeOnClickOverlay: true,
  closeOnEsc: true,
  showFooter: true,
  confirmText: '确定',
  cancelText: '取消',
  confirmLoading: false,
  centered: true,
  zIndex: 1000,
})

/**
 * 事件定义
 */
const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'confirm'): void
  (e: 'cancel'): void
  (e: 'close'): void
}>()

/**
 * 内部可见状态
 */
const visible = ref(props.modelValue)

/**
 * 同步外部值
 */
watch(() => props.modelValue, (val) => {
  visible.value = val
})

/**
 * 同步内部值到外部
 */
watch(visible, (val) => {
  emit('update:modelValue', val)
})

/**
 * 关闭模态框
 */
function close() {
  visible.value = false
  emit('close')
}

/**
 * 点击遮罩
 */
function handleOverlayClick() {
  if (props.closeOnClickOverlay) {
    close()
  }
}

/**
 * 确认
 */
function handleConfirm() {
  emit('confirm')
}

/**
 * 取消
 */
function handleCancel() {
  emit('cancel')
  close()
}

/**
 * 键盘事件处理
 */
function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape' && props.closeOnEsc && visible.value) {
    close()
  }
}

/**
 * 计算宽度样式
 */
const widthStyle = typeof props.width === 'number' ? `${props.width}px` : props.width

/**
 * 生命周期
 */
onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <Teleport to="body">
    <Transition name="modal-fade">
      <div
        v-if="visible"
        class="modal"
        :class="{ 'modal--centered': centered }"
        :style="{ zIndex }"
        role="dialog"
        aria-modal="true"
        :aria-labelledby="title ? 'modal-title' : undefined"
      >
        <!-- 遮罩层 -->
        <div class="modal__overlay" @click="handleOverlayClick" />
        
        <!-- 模态框内容 -->
        <div
          class="modal__container"
          :style="{ width: widthStyle }"
        >
          <!-- 头部 -->
          <div v-if="title || showClose" class="modal__header">
            <h3 v-if="title" id="modal-title" class="modal__title">{{ title }}</h3>
            <button
              v-if="showClose"
              type="button"
              class="modal__close"
              aria-label="关闭"
              @click="close"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M18 6L6 18M6 6l12 12" stroke-linecap="round" stroke-linejoin="round" />
              </svg>
            </button>
          </div>
          
          <!-- 内容区域 -->
          <div class="modal__body">
            <slot></slot>
          </div>
          
          <!-- 底部按钮 -->
          <div v-if="showFooter" class="modal__footer">
            <slot name="footer">
              <button
                type="button"
                class="modal__button modal__button--cancel"
                @click="handleCancel"
              >
                {{ cancelText }}
              </button>
              <button
                type="button"
                class="modal__button modal__button--confirm"
                :disabled="confirmLoading"
                @click="handleConfirm"
              >
                <span v-if="confirmLoading" class="modal__loading"></span>
                {{ confirmText }}
              </button>
            </slot>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.modal {
  position: fixed;
  inset: 0;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 24px;
  overflow-y: auto;
}

.modal--centered {
  align-items: center;
}

.modal__overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(4px);
}

.modal__container {
  position: relative;
  background: #1a1a2e;
  border-radius: 12px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
  max-width: 100%;
  animation: modal-enter 0.3s ease;
}

.modal__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 24px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.modal__title {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #e2e8f0;
}

.modal__close {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  padding: 0;
  border: none;
  background: none;
  color: #8892b0;
  cursor: pointer;
  border-radius: 6px;
  transition: all 0.2s ease;
}

.modal__close:hover {
  color: #e2e8f0;
  background: rgba(255, 255, 255, 0.1);
}

.modal__close svg {
  width: 20px;
  height: 20px;
}

.modal__body {
  padding: 24px;
  color: #cbd5e0;
  font-size: 14px;
  line-height: 1.6;
}

.modal__footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 16px 24px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.modal__button {
  padding: 10px 20px;
  font-size: 14px;
  font-weight: 500;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  border: none;
  outline: none;
  display: flex;
  align-items: center;
  gap: 8px;
}

.modal__button--cancel {
  background: rgba(255, 255, 255, 0.1);
  color: #e2e8f0;
}

.modal__button--cancel:hover {
  background: rgba(255, 255, 255, 0.15);
}

.modal__button--confirm {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  color: #0f0f23;
}

.modal__button--confirm:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(79, 172, 254, 0.4);
}

.modal__button--confirm:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.modal__loading {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(0, 0, 0, 0.2);
  border-top-color: currentColor;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

/* 动画 */
@keyframes modal-enter {
  from {
    opacity: 0;
    transform: scale(0.95) translateY(-20px);
  }
  to {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* 过渡动画 */
.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.3s ease;
}

.modal-fade-enter-active .modal__container,
.modal-fade-leave-active .modal__container {
  transition: transform 0.3s ease, opacity 0.3s ease;
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}

.modal-fade-enter-from .modal__container,
.modal-fade-leave-to .modal__container {
  transform: scale(0.95) translateY(-20px);
  opacity: 0;
}
</style>

