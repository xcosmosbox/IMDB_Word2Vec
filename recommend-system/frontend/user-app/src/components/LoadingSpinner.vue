<script setup lang="ts">
/**
 * LoadingSpinner 加载动画组件
 * 
 * 提供多种尺寸和风格的加载指示器
 * 支持暗色主题，符合整体设计风格
 */

interface Props {
  /** 尺寸: small | medium | large */
  size?: 'small' | 'medium' | 'large'
  /** 颜色 (默认使用主题色) */
  color?: string
  /** 是否显示加载文字 */
  showText?: boolean
  /** 加载文字内容 */
  text?: string
  /** 是否全屏覆盖 */
  fullscreen?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  size: 'medium',
  color: '#4facfe',
  showText: false,
  text: '加载中...',
  fullscreen: false,
})

// 尺寸映射
const sizeMap = {
  small: '24px',
  medium: '40px',
  large: '64px',
}

const spinnerSize = sizeMap[props.size]
</script>

<template>
  <div 
    class="loading-spinner" 
    :class="{ 'loading-spinner--fullscreen': fullscreen }"
  >
    <div class="spinner-container">
      <!-- 外圈动画 -->
      <div 
        class="spinner-outer"
        :style="{ 
          width: spinnerSize, 
          height: spinnerSize,
          borderTopColor: color,
        }"
      ></div>
      
      <!-- 内圈动画 -->
      <div 
        class="spinner-inner"
        :style="{ 
          width: `calc(${spinnerSize} * 0.6)`, 
          height: `calc(${spinnerSize} * 0.6)`,
          borderBottomColor: color,
        }"
      ></div>
      
      <!-- 中心光点 -->
      <div 
        class="spinner-dot"
        :style="{ backgroundColor: color }"
      ></div>
    </div>
    
    <!-- 加载文字 -->
    <p 
      v-if="showText" 
      class="spinner-text"
      :style="{ color }"
    >
      {{ text }}
    </p>
  </div>
</template>

<style scoped>
.loading-spinner {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
}

.loading-spinner--fullscreen {
  position: fixed;
  inset: 0;
  background: rgba(26, 26, 46, 0.9);
  backdrop-filter: blur(8px);
  z-index: 9999;
}

.spinner-container {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}

.spinner-outer {
  position: absolute;
  border: 3px solid transparent;
  border-top-color: #4facfe;
  border-radius: 50%;
  animation: spin-clockwise 1s linear infinite;
}

.spinner-inner {
  position: absolute;
  border: 2px solid transparent;
  border-bottom-color: #00f2fe;
  border-radius: 50%;
  animation: spin-counter 0.8s linear infinite;
}

.spinner-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  animation: pulse 1.5s ease-in-out infinite;
}

.spinner-text {
  font-size: 0.9rem;
  font-weight: 500;
  color: #4facfe;
  margin: 0;
  animation: fade-pulse 1.5s ease-in-out infinite;
}

@keyframes spin-clockwise {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

@keyframes spin-counter {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(-360deg);
  }
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.5;
    transform: scale(0.8);
  }
}

@keyframes fade-pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.6;
  }
}
</style>

