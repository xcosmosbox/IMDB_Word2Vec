<script setup lang="ts">
/**
 * ItemCard 物品卡片组件
 * 
 * 展示推荐物品的卡片组件，支持：
 * - 多种物品类型 (电影/商品/文章/视频)
 * - 匹配度显示
 * - 推荐理由展示
 * - 悬停动效
 * - 骨架屏加载状态
 */

import { computed } from 'vue'
import type { Item } from '@shared/types'

interface Props {
  /** 物品数据 */
  item: Item
  /** 推荐分数 (0-1) */
  score?: number
  /** 推荐理由 */
  reason?: string
  /** 是否显示骨架屏 */
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
})

const emit = defineEmits<{
  /** 点击事件 */
  'click': []
  /** 喜欢事件 */
  'like': [itemId: string]
  /** 分享事件 */
  'share': [itemId: string]
}>()

// 类型颜色映射
const typeColorMap: Record<string, { primary: string; gradient: string }> = {
  movie: { 
    primary: '#e50914', 
    gradient: 'linear-gradient(135deg, #e50914 0%, #b8070f 100%)' 
  },
  product: { 
    primary: '#ff9900', 
    gradient: 'linear-gradient(135deg, #ff9900 0%, #cc7a00 100%)' 
  },
  article: { 
    primary: '#1da1f2', 
    gradient: 'linear-gradient(135deg, #1da1f2 0%, #0d8bd9 100%)' 
  },
  video: { 
    primary: '#ff0050', 
    gradient: 'linear-gradient(135deg, #ff0050 0%, #cc0040 100%)' 
  },
}

// 类型图标映射
const typeIconMap: Record<string, string> = {
  movie: '🎬',
  product: '🛒',
  article: '📄',
  video: '🎥',
}

// 类型标签映射
const typeLabelMap: Record<string, string> = {
  movie: '电影',
  product: '商品',
  article: '文章',
  video: '视频',
}

/** 获取类型颜色配置 */
const typeColors = computed(() => 
  typeColorMap[props.item.type] || { primary: '#4facfe', gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' }
)

/** 获取类型图标 */
const typeIcon = computed(() => typeIconMap[props.item.type] || '📦')

/** 获取类型标签 */
const typeLabel = computed(() => typeLabelMap[props.item.type] || props.item.type)

/** 格式化分数 */
const formattedScore = computed(() => {
  if (!props.score) return ''
  return `${Math.round(props.score * 100)}%`
})

/** 截断描述 */
const truncatedDescription = computed(() => {
  const desc = props.item.description || ''
  const maxLen = 100
  return desc.length > maxLen ? desc.slice(0, maxLen) + '...' : desc
})

/** 展示的标签 (最多3个) */
const displayTags = computed(() => 
  (props.item.tags || []).slice(0, 3)
)

/** 处理卡片点击 */
function handleClick() {
  emit('click')
}

/** 处理喜欢按钮点击 */
function handleLike(event: Event) {
  event.stopPropagation()
  emit('like', props.item.id)
}

/** 处理分享按钮点击 */
function handleShare(event: Event) {
  event.stopPropagation()
  emit('share', props.item.id)
}
</script>

<template>
  <!-- 骨架屏状态 -->
  <article v-if="loading" class="item-card item-card--skeleton">
    <div class="card-cover skeleton-cover">
      <div class="skeleton-shimmer"></div>
    </div>
    <div class="card-content">
      <div class="skeleton-title"></div>
      <div class="skeleton-desc"></div>
      <div class="skeleton-tags">
        <span class="skeleton-tag"></span>
        <span class="skeleton-tag"></span>
      </div>
    </div>
  </article>
  
  <!-- 正常状态 -->
  <article 
    v-else
    class="item-card"
    role="button"
    tabindex="0"
    @click="handleClick"
    @keydown.enter="handleClick"
    @keydown.space.prevent="handleClick"
  >
    <!-- 封面区域 -->
    <div class="card-cover">
      <!-- 封面占位背景 -->
      <div 
        class="cover-placeholder"
        :style="{ background: typeColors.gradient }"
      >
        <span class="type-icon">{{ typeIcon }}</span>
      </div>
      
      <!-- 类型标签 -->
      <span 
        class="type-tag" 
        :style="{ backgroundColor: typeColors.primary }"
      >
        {{ typeLabel }}
      </span>
      
      <!-- 匹配度徽章 -->
      <span v-if="score" class="match-score">
        <span class="score-icon">⚡</span>
        <span class="score-value">{{ formattedScore }}</span>
      </span>
      
      <!-- 悬停操作按钮 -->
      <div class="hover-actions">
        <button 
          class="action-btn action-btn--like" 
          title="喜欢"
          @click="handleLike"
        >
          ❤️
        </button>
        <button 
          class="action-btn action-btn--share" 
          title="分享"
          @click="handleShare"
        >
          🔗
        </button>
      </div>
    </div>

    <!-- 内容区域 -->
    <div class="card-content">
      <!-- 标题 -->
      <h3 class="card-title">{{ item.title }}</h3>
      
      <!-- 分类 -->
      <p v-if="item.category" class="card-category">
        {{ item.category }}
      </p>
      
      <!-- 描述 -->
      <p v-if="truncatedDescription" class="card-description">
        {{ truncatedDescription }}
      </p>
      
      <!-- 标签列表 -->
      <div v-if="displayTags.length > 0" class="card-tags">
        <span 
          v-for="tag in displayTags" 
          :key="tag"
          class="tag"
        >
          {{ tag }}
        </span>
      </div>
      
      <!-- 推荐理由 -->
      <p v-if="reason" class="card-reason">
        <span class="reason-icon">💡</span>
        {{ reason }}
      </p>
    </div>
  </article>
</template>

<style scoped>
/* 卡片容器 */
.item-card {
  position: relative;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 1rem;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  border: 1px solid rgba(255, 255, 255, 0.06);
}

.item-card:hover {
  transform: translateY(-8px) scale(1.02);
  box-shadow: 
    0 20px 40px rgba(0, 0, 0, 0.4),
    0 0 60px rgba(79, 172, 254, 0.1);
  border-color: rgba(79, 172, 254, 0.3);
}

.item-card:focus {
  outline: none;
}

.item-card:focus-visible {
  box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.5);
}

/* 封面区域 */
.card-cover {
  position: relative;
  aspect-ratio: 16/9;
  overflow: hidden;
}

.cover-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.4s ease;
}

.item-card:hover .cover-placeholder {
  transform: scale(1.05);
}

.type-icon {
  font-size: 3.5rem;
  opacity: 0.4;
  filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3));
}

/* 类型标签 */
.type-tag {
  position: absolute;
  top: 0.75rem;
  left: 0.75rem;
  padding: 0.25rem 0.75rem;
  font-size: 0.7rem;
  font-weight: 600;
  color: white;
  border-radius: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

/* 匹配度徽章 */
.match-score {
  position: absolute;
  bottom: 0.75rem;
  right: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.3rem 0.6rem;
  font-size: 0.75rem;
  font-weight: 600;
  background: rgba(0, 0, 0, 0.75);
  backdrop-filter: blur(8px);
  color: #4facfe;
  border-radius: 1rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.score-icon {
  font-size: 0.8rem;
}

/* 悬停操作按钮 */
.hover-actions {
  position: absolute;
  top: 0.75rem;
  right: 0.75rem;
  display: flex;
  gap: 0.5rem;
  opacity: 0;
  transform: translateY(-10px);
  transition: all 0.3s ease;
}

.item-card:hover .hover-actions {
  opacity: 1;
  transform: translateY(0);
}

.action-btn {
  width: 2rem;
  height: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(8px);
  border: none;
  border-radius: 50%;
  cursor: pointer;
  font-size: 0.875rem;
  transition: all 0.2s ease;
}

.action-btn:hover {
  transform: scale(1.15);
  background: rgba(0, 0, 0, 0.8);
}

/* 内容区域 */
.card-content {
  padding: 1.25rem;
}

.card-title {
  font-size: 1.05rem;
  font-weight: 600;
  color: #fff;
  margin: 0 0 0.4rem;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-category {
  font-size: 0.8rem;
  color: #64ffda;
  margin: 0 0 0.5rem;
  font-weight: 500;
}

.card-description {
  font-size: 0.85rem;
  color: #8892b0;
  line-height: 1.6;
  margin: 0 0 0.75rem;
}

/* 标签 */
.card-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-bottom: 0.75rem;
}

.tag {
  padding: 0.2rem 0.5rem;
  font-size: 0.7rem;
  background: rgba(79, 172, 254, 0.12);
  color: #4facfe;
  border-radius: 0.25rem;
  font-weight: 500;
}

/* 推荐理由 */
.card-reason {
  display: flex;
  align-items: flex-start;
  gap: 0.4rem;
  font-size: 0.8rem;
  color: #64ffda;
  margin: 0;
  font-style: italic;
  line-height: 1.5;
}

.reason-icon {
  flex-shrink: 0;
}

/* ============================================================
   骨架屏样式
   ============================================================ */

.item-card--skeleton {
  pointer-events: none;
}

.skeleton-cover {
  position: relative;
  background: rgba(255, 255, 255, 0.05);
  overflow: hidden;
}

.skeleton-shimmer {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(255, 255, 255, 0.08) 50%,
    transparent 100%
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

.skeleton-title {
  height: 1.2rem;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 0.25rem;
  margin-bottom: 0.75rem;
  width: 80%;
}

.skeleton-desc {
  height: 0.9rem;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 0.25rem;
  margin-bottom: 0.75rem;
  width: 100%;
}

.skeleton-tags {
  display: flex;
  gap: 0.5rem;
}

.skeleton-tag {
  width: 3rem;
  height: 1.2rem;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 0.25rem;
}
</style>

