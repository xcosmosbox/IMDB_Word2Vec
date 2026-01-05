<!--
  Loading 组件
  
  加载状态展示组件，支持多种尺寸和全屏模式
  
  @author Person F
-->
<script setup lang="ts">
/**
 * 组件属性
 */
interface Props {
  /** 加载器尺寸 */
  size?: 'small' | 'default' | 'large'
  /** 加载提示文本 */
  tip?: string
  /** 是否全屏显示 */
  fullscreen?: boolean
  /** 是否显示遮罩层 */
  overlay?: boolean
  /** 主题颜色 */
  color?: string
}

const props = withDefaults(defineProps<Props>(), {
  size: 'default',
  fullscreen: false,
  overlay: false,
  color: '#4facfe',
})

/**
 * 尺寸映射
 */
const sizeMap: Record<string, number> = {
  small: 24,
  default: 40,
  large: 56,
}

const spinnerSize = sizeMap[props.size] || sizeMap.default
</script>

<template>
  <div
    class="loading"
    :class="{
      'loading--fullscreen': fullscreen,
      'loading--overlay': overlay,
    }"
    role="status"
    aria-live="polite"
    aria-busy="true"
  >
    <!-- 加载动画 -->
    <div
      class="loading__spinner"
      :style="{
        width: `${spinnerSize}px`,
        height: `${spinnerSize}px`,
      }"
    >
      <svg viewBox="0 0 50 50" class="loading__circular">
        <circle
          cx="25"
          cy="25"
          r="20"
          fill="none"
          class="loading__path"
          :style="{ stroke: color }"
        />
      </svg>
    </div>
    
    <!-- 加载提示 -->
    <p v-if="tip" class="loading__tip">{{ tip }}</p>
    
    <!-- 屏幕阅读器文本 -->
    <span class="sr-only">正在加载...</span>
  </div>
</template>

<style scoped>
.loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.loading--fullscreen {
  position: fixed;
  inset: 0;
  background: rgba(26, 26, 46, 0.95);
  z-index: 9999;
  backdrop-filter: blur(4px);
}

.loading--overlay {
  position: absolute;
  inset: 0;
  background: rgba(26, 26, 46, 0.8);
  z-index: 10;
}

.loading__spinner {
  animation: loading-rotate 2s linear infinite;
}

.loading__circular {
  width: 100%;
  height: 100%;
}

.loading__path {
  stroke-width: 3;
  stroke-linecap: round;
  animation: loading-dash 1.5s ease-in-out infinite;
}

.loading__tip {
  margin-top: 16px;
  color: #8892b0;
  font-size: 14px;
  text-align: center;
}

/* 屏幕阅读器专用 */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

/* 动画 */
@keyframes loading-rotate {
  100% {
    transform: rotate(360deg);
  }
}

@keyframes loading-dash {
  0% {
    stroke-dasharray: 1, 150;
    stroke-dashoffset: 0;
  }
  50% {
    stroke-dasharray: 90, 150;
    stroke-dashoffset: -35;
  }
  100% {
    stroke-dasharray: 90, 150;
    stroke-dashoffset: -124;
  }
}
</style>

