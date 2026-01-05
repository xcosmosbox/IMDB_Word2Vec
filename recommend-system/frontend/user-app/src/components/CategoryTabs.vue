<script setup lang="ts">
/**
 * CategoryTabs 分类标签页组件
 * 
 * 用于首页顶部的分类导航，支持：
 * - 响应式布局
 * - 滑动选中动画
 * - 键盘导航
 * - 粘性定位
 */

import { ref, computed, watch, nextTick } from 'vue'

interface Category {
  /** 分类唯一标识 */
  key: string
  /** 分类显示名称 */
  label: string
  /** 分类图标 (可选) */
  icon?: string
  /** 分类描述 (可选) */
  description?: string
}

interface Props {
  /** 分类列表 */
  categories: Category[]
  /** 当前激活的分类 */
  active: string
  /** 是否启用粘性定位 */
  sticky?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  sticky: true,
})

const emit = defineEmits<{
  /** 分类变更事件 */
  'update:active': [key: string]
}>()

// 滑块位置
const sliderStyle = ref({
  left: '0px',
  width: '0px',
})

// Tab 容器引用
const tabsContainerRef = ref<HTMLElement | null>(null)

// 当前激活的 tab 索引
const activeIndex = computed(() => 
  props.categories.findIndex(c => c.key === props.active)
)

/**
 * 更新滑块位置
 */
async function updateSliderPosition() {
  await nextTick()
  
  if (!tabsContainerRef.value) return
  
  const activeButton = tabsContainerRef.value.querySelector('.tab-btn.active') as HTMLElement
  if (activeButton) {
    sliderStyle.value = {
      left: `${activeButton.offsetLeft}px`,
      width: `${activeButton.offsetWidth}px`,
    }
  }
}

/**
 * 处理分类点击
 */
function handleTabClick(key: string) {
  emit('update:active', key)
}

/**
 * 处理键盘导航
 */
function handleKeydown(event: KeyboardEvent) {
  const { key } = event
  const currentIndex = activeIndex.value
  
  let newIndex = currentIndex
  
  switch (key) {
    case 'ArrowLeft':
      newIndex = Math.max(0, currentIndex - 1)
      break
    case 'ArrowRight':
      newIndex = Math.min(props.categories.length - 1, currentIndex + 1)
      break
    case 'Home':
      newIndex = 0
      break
    case 'End':
      newIndex = props.categories.length - 1
      break
    default:
      return
  }
  
  if (newIndex !== currentIndex) {
    event.preventDefault()
    emit('update:active', props.categories[newIndex].key)
  }
}

// 监听 active 变化，更新滑块位置
watch(() => props.active, () => {
  updateSliderPosition()
}, { immediate: true })

// 组件挂载后更新滑块位置
watch(tabsContainerRef, () => {
  updateSliderPosition()
})
</script>

<template>
  <nav 
    ref="tabsContainerRef"
    class="category-tabs"
    :class="{ 'category-tabs--sticky': sticky }"
    role="tablist"
    aria-label="内容分类"
    @keydown="handleKeydown"
  >
    <!-- 滑动指示器 -->
    <div 
      class="tab-slider" 
      :style="sliderStyle"
      aria-hidden="true"
    ></div>
    
    <!-- Tab 按钮列表 -->
    <button
      v-for="(cat, index) in categories"
      :key="cat.key"
      class="tab-btn"
      :class="{ active: active === cat.key }"
      role="tab"
      :aria-selected="active === cat.key"
      :aria-controls="`panel-${cat.key}`"
      :tabindex="active === cat.key ? 0 : -1"
      @click="handleTabClick(cat.key)"
    >
      <!-- 图标 -->
      <span v-if="cat.icon" class="tab-icon">{{ cat.icon }}</span>
      
      <!-- 标签文字 -->
      <span class="tab-label">{{ cat.label }}</span>
    </button>
  </nav>
</template>

<style scoped>
.category-tabs {
  position: relative;
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  padding: 1rem 2rem;
  background: rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  overflow-x: auto;
  scrollbar-width: none;
  -ms-overflow-style: none;
}

.category-tabs::-webkit-scrollbar {
  display: none;
}

.category-tabs--sticky {
  position: sticky;
  top: 0;
  z-index: 100;
}

/* 滑动指示器 */
.tab-slider {
  position: absolute;
  bottom: 0;
  height: 3px;
  background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
  border-radius: 1.5px 1.5px 0 0;
  transition: left 0.3s cubic-bezier(0.4, 0, 0.2, 1), 
              width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Tab 按钮 */
.tab-btn {
  position: relative;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  font-size: 0.95rem;
  font-weight: 500;
  color: #8892b0;
  background: transparent;
  border: none;
  border-radius: 2rem;
  cursor: pointer;
  transition: all 0.3s ease;
  white-space: nowrap;
  user-select: none;
}

.tab-btn:hover {
  color: #ccd6f6;
  background: rgba(255, 255, 255, 0.08);
}

.tab-btn:focus {
  outline: none;
}

.tab-btn:focus-visible {
  box-shadow: 0 0 0 2px rgba(79, 172, 254, 0.5);
}

.tab-btn.active {
  color: #fff;
  background: rgba(79, 172, 254, 0.15);
}

/* Tab 图标 */
.tab-icon {
  font-size: 1.1rem;
}

/* Tab 标签 */
.tab-label {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* 移动端适配 */
@media (max-width: 640px) {
  .category-tabs {
    justify-content: flex-start;
    padding: 0.75rem 1rem;
    gap: 0.25rem;
  }
  
  .tab-btn {
    padding: 0.6rem 1rem;
    font-size: 0.875rem;
  }
}

/* 平板适配 */
@media (min-width: 641px) and (max-width: 1024px) {
  .category-tabs {
    padding: 0.875rem 1.5rem;
  }
}
</style>

