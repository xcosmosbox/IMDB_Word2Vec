<!--
  Skeleton 组件
  
  骨架屏组件，用于内容加载时的占位显示
  
  @author Person F
-->
<script setup lang="ts">
import { computed } from 'vue'

/**
 * 骨架屏类型
 */
type SkeletonType = 'text' | 'circle' | 'rect' | 'card' | 'list' | 'image'

/**
 * 组件属性
 */
interface Props {
  /** 骨架类型 */
  type?: SkeletonType
  /** 宽度 */
  width?: string | number
  /** 高度 */
  height?: string | number
  /** 圆角 */
  radius?: string | number
  /** 是否显示动画 */
  animated?: boolean
  /** 行数（用于 text 和 list 类型） */
  rows?: number
  /** 段落模式（最后一行宽度较短） */
  paragraph?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  type: 'rect',
  animated: true,
  rows: 3,
  paragraph: true,
})

/**
 * 计算样式
 */
const style = computed(() => {
  const styles: Record<string, string> = {}
  
  if (props.width) {
    styles.width = typeof props.width === 'number' ? `${props.width}px` : props.width
  }
  
  if (props.height) {
    styles.height = typeof props.height === 'number' ? `${props.height}px` : props.height
  }
  
  if (props.radius) {
    styles.borderRadius = typeof props.radius === 'number' ? `${props.radius}px` : props.radius
  }
  
  return styles
})

/**
 * 生成行数组（用于 text 类型）
 */
const textRows = computed(() => {
  return Array.from({ length: props.rows }, (_, i) => {
    // 段落模式下，最后一行宽度为 60%
    if (props.paragraph && i === props.rows - 1) {
      return { width: '60%' }
    }
    return { width: '100%' }
  })
})
</script>

<template>
  <!-- 文本类型 -->
  <div v-if="type === 'text'" class="skeleton skeleton--text" :class="{ 'skeleton--animated': animated }">
    <div
      v-for="(row, index) in textRows"
      :key="index"
      class="skeleton__line"
      :style="{ width: row.width }"
    ></div>
  </div>
  
  <!-- 圆形类型 -->
  <div
    v-else-if="type === 'circle'"
    class="skeleton skeleton--circle"
    :class="{ 'skeleton--animated': animated }"
    :style="style"
  ></div>
  
  <!-- 图片类型 -->
  <div
    v-else-if="type === 'image'"
    class="skeleton skeleton--image"
    :class="{ 'skeleton--animated': animated }"
    :style="style"
  >
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <path d="M21 15l-5-5L5 21" />
    </svg>
  </div>
  
  <!-- 卡片类型 -->
  <div
    v-else-if="type === 'card'"
    class="skeleton skeleton--card"
    :class="{ 'skeleton--animated': animated }"
    :style="style"
  >
    <div class="skeleton__card-image"></div>
    <div class="skeleton__card-content">
      <div class="skeleton__card-title"></div>
      <div class="skeleton__card-desc"></div>
      <div class="skeleton__card-desc skeleton__card-desc--short"></div>
    </div>
  </div>
  
  <!-- 列表类型 -->
  <div
    v-else-if="type === 'list'"
    class="skeleton skeleton--list"
    :class="{ 'skeleton--animated': animated }"
    :style="style"
  >
    <div v-for="i in rows" :key="i" class="skeleton__list-item">
      <div class="skeleton__list-avatar"></div>
      <div class="skeleton__list-content">
        <div class="skeleton__list-title"></div>
        <div class="skeleton__list-desc"></div>
      </div>
    </div>
  </div>
  
  <!-- 矩形类型（默认） -->
  <div
    v-else
    class="skeleton skeleton--rect"
    :class="{ 'skeleton--animated': animated }"
    :style="style"
  ></div>
</template>

<style scoped>
.skeleton {
  background: rgba(255, 255, 255, 0.05);
  overflow: hidden;
}

.skeleton--animated {
  position: relative;
}

.skeleton--animated::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(255, 255, 255, 0.1),
    transparent
  );
  animation: shimmer 1.5s infinite;
}

@keyframes shimmer {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(100%);
  }
}

/* 矩形 */
.skeleton--rect {
  width: 100%;
  height: 20px;
  border-radius: 4px;
}

/* 圆形 */
.skeleton--circle {
  width: 48px;
  height: 48px;
  border-radius: 50%;
}

/* 图片 */
.skeleton--image {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 200px;
  border-radius: 8px;
  color: rgba(255, 255, 255, 0.2);
}

.skeleton--image svg {
  width: 48px;
  height: 48px;
}

/* 文本 */
.skeleton--text {
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: transparent;
}

.skeleton__line {
  height: 16px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
  position: relative;
  overflow: hidden;
}

.skeleton--animated .skeleton__line::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(255, 255, 255, 0.1),
    transparent
  );
  animation: shimmer 1.5s infinite;
}

/* 卡片 */
.skeleton--card {
  border-radius: 12px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.02);
}

.skeleton__card-image {
  width: 100%;
  height: 160px;
  background: rgba(255, 255, 255, 0.05);
}

.skeleton__card-content {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.skeleton__card-title {
  height: 20px;
  width: 70%;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
}

.skeleton__card-desc {
  height: 14px;
  width: 100%;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
}

.skeleton__card-desc--short {
  width: 50%;
}

/* 列表 */
.skeleton--list {
  display: flex;
  flex-direction: column;
  gap: 16px;
  background: transparent;
}

.skeleton__list-item {
  display: flex;
  gap: 12px;
}

.skeleton__list-avatar {
  flex-shrink: 0;
  width: 48px;
  height: 48px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 50%;
  position: relative;
  overflow: hidden;
}

.skeleton--animated .skeleton__list-avatar::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(255, 255, 255, 0.1),
    transparent
  );
  animation: shimmer 1.5s infinite;
}

.skeleton__list-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
  justify-content: center;
}

.skeleton__list-title {
  height: 16px;
  width: 40%;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
  position: relative;
  overflow: hidden;
}

.skeleton__list-desc {
  height: 14px;
  width: 80%;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
  position: relative;
  overflow: hidden;
}

.skeleton--animated .skeleton__list-title::after,
.skeleton--animated .skeleton__list-desc::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(255, 255, 255, 0.1),
    transparent
  );
  animation: shimmer 1.5s infinite;
}
</style>

