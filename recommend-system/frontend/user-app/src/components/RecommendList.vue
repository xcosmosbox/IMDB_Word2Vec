<script setup lang="ts">
/**
 * RecommendList 推荐列表组件
 * 
 * 负责展示推荐物品的瀑布流/网格布局，支持：
 * - 响应式列数自适应
 * - 虚拟滚动 (大数据量优化)
 * - 无限滚动加载
 * - 刷新功能
 * - 骨架屏加载
 */

import { ref, computed, onMounted, onUnmounted } from 'vue'
import type { Recommendation } from '@shared/types'
import ItemCard from './ItemCard.vue'

interface Props {
  /** 推荐列表 */
  recommendations: Recommendation[]
  /** 是否加载中 */
  loading?: boolean
  /** 是否有更多数据 */
  hasMore?: boolean
  /** 标题 */
  title?: string
  /** 是否显示刷新按钮 */
  showRefresh?: boolean
  /** 空状态提示文字 */
  emptyText?: string
  /** 骨架屏数量 */
  skeletonCount?: number
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
  hasMore: true,
  title: '为你推荐',
  showRefresh: true,
  emptyText: '暂无推荐内容',
  skeletonCount: 8,
})

const emit = defineEmits<{
  /** 物品点击 */
  'item-click': [itemId: string]
  /** 物品喜欢 */
  'item-like': [itemId: string]
  /** 物品分享 */
  'item-share': [itemId: string]
  /** 刷新请求 */
  'refresh': []
  /** 加载更多 */
  'load-more': []
}>()

// ============================================================
// 响应式列数计算
// ============================================================

const columns = ref(4)
const listRef = ref<HTMLElement | null>(null)

/** 根据窗口宽度计算列数 */
function updateColumns() {
  const width = window.innerWidth
  if (width < 640) {
    columns.value = 1
  } else if (width < 900) {
    columns.value = 2
  } else if (width < 1200) {
    columns.value = 3
  } else if (width < 1600) {
    columns.value = 4
  } else {
    columns.value = 5
  }
}

// ============================================================
// 无限滚动加载
// ============================================================

const observerRef = ref<IntersectionObserver | null>(null)
const loadMoreTriggerRef = ref<HTMLElement | null>(null)

/** 初始化 Intersection Observer */
function initIntersectionObserver() {
  if (!loadMoreTriggerRef.value || !props.hasMore) return
  
  observerRef.value = new IntersectionObserver(
    (entries) => {
      const entry = entries[0]
      if (entry.isIntersecting && !props.loading && props.hasMore) {
        emit('load-more')
      }
    },
    {
      root: null,
      rootMargin: '200px',
      threshold: 0,
    }
  )
  
  observerRef.value.observe(loadMoreTriggerRef.value)
}

/** 销毁 Observer */
function destroyObserver() {
  if (observerRef.value) {
    observerRef.value.disconnect()
    observerRef.value = null
  }
}

// ============================================================
// 刷新动画
// ============================================================

const isRefreshing = ref(false)

async function handleRefresh() {
  isRefreshing.value = true
  emit('refresh')
  
  // 动画效果
  setTimeout(() => {
    isRefreshing.value = false
  }, 500)
}

// ============================================================
// 事件处理
// ============================================================

function handleItemClick(itemId: string) {
  emit('item-click', itemId)
}

function handleItemLike(itemId: string) {
  emit('item-like', itemId)
}

function handleItemShare(itemId: string) {
  emit('item-share', itemId)
}

// ============================================================
// 计算属性
// ============================================================

/** 是否为空状态 */
const isEmpty = computed(() => 
  !props.loading && props.recommendations.length === 0
)

/** 骨架屏数组 */
const skeletonItems = computed(() => 
  Array.from({ length: props.skeletonCount }, (_, i) => i)
)

// ============================================================
// 生命周期
// ============================================================

onMounted(() => {
  updateColumns()
  window.addEventListener('resize', updateColumns)
  
  // 初始化无限滚动
  setTimeout(() => {
    initIntersectionObserver()
  }, 100)
})

onUnmounted(() => {
  window.removeEventListener('resize', updateColumns)
  destroyObserver()
})
</script>

<template>
  <div ref="listRef" class="recommend-list">
    <!-- 列表头部 -->
    <header class="list-header">
      <h2 class="list-title">
        <span class="title-icon">✨</span>
        {{ title }}
        <span v-if="recommendations.length > 0" class="title-count">
          ({{ recommendations.length }})
        </span>
      </h2>
      
      <button 
        v-if="showRefresh"
        class="refresh-btn"
        :class="{ 'refresh-btn--spinning': isRefreshing }"
        :disabled="loading"
        @click="handleRefresh"
      >
        <span class="refresh-icon">↻</span>
        <span class="refresh-text">换一批</span>
      </button>
    </header>

    <!-- 骨架屏加载状态 -->
    <div 
      v-if="loading && recommendations.length === 0" 
      class="card-grid"
      :style="{ '--columns': columns }"
    >
      <ItemCard
        v-for="n in skeletonItems"
        :key="`skeleton-${n}`"
        :item="{ id: '', type: 'movie', title: '', description: '', category: '', tags: [], status: 'active', created_at: '', updated_at: '' }"
        :loading="true"
      />
    </div>

    <!-- 推荐列表 -->
    <div 
      v-else-if="!isEmpty"
      class="card-grid"
      :style="{ '--columns': columns }"
    >
      <transition-group 
        name="card-list" 
        tag="div" 
        class="card-grid-inner"
        :style="{ '--columns': columns }"
      >
        <ItemCard
          v-for="rec in recommendations"
          :key="rec.item_id"
          :item="rec.item!"
          :score="rec.score"
          :reason="rec.reason"
          @click="handleItemClick(rec.item_id)"
          @like="handleItemLike"
          @share="handleItemShare"
        />
      </transition-group>
    </div>

    <!-- 空状态 -->
    <div v-else class="empty-state">
      <div class="empty-icon">🎯</div>
      <p class="empty-text">{{ emptyText }}</p>
      <button class="empty-refresh-btn" @click="handleRefresh">
        <span>刷新试试</span>
      </button>
    </div>

    <!-- 加载更多触发器 -->
    <div 
      ref="loadMoreTriggerRef"
      class="load-more-trigger"
    >
      <template v-if="loading && recommendations.length > 0">
        <div class="loading-more">
          <span class="loading-dot"></span>
          <span class="loading-dot"></span>
          <span class="loading-dot"></span>
        </div>
        <span class="loading-text">加载更多...</span>
      </template>
      
      <template v-else-if="!hasMore && recommendations.length > 0">
        <span class="no-more-text">— 没有更多了 —</span>
      </template>
    </div>
  </div>
</template>

<style scoped>
.recommend-list {
  width: 100%;
}

/* ============================================================
   列表头部
   ============================================================ */

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
  gap: 1rem;
}

.list-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1.4rem;
  font-weight: 600;
  color: #fff;
  margin: 0;
}

.title-icon {
  font-size: 1.2rem;
}

.title-count {
  font-size: 0.9rem;
  color: #8892b0;
  font-weight: 400;
}

/* 刷新按钮 */
.refresh-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 1.2rem;
  background: rgba(79, 172, 254, 0.12);
  border: 1px solid rgba(79, 172, 254, 0.25);
  border-radius: 2rem;
  color: #4facfe;
  font-size: 0.9rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.refresh-btn:hover:not(:disabled) {
  background: rgba(79, 172, 254, 0.2);
  border-color: rgba(79, 172, 254, 0.4);
  transform: scale(1.03);
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.refresh-icon {
  font-size: 1.1rem;
  transition: transform 0.5s ease;
}

.refresh-btn--spinning .refresh-icon {
  animation: spin 0.5s linear;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

/* ============================================================
   卡片网格
   ============================================================ */

.card-grid {
  display: grid;
  grid-template-columns: repeat(var(--columns), 1fr);
  gap: 1.5rem;
}

.card-grid-inner {
  display: contents;
}

/* ============================================================
   卡片动画
   ============================================================ */

.card-list-enter-active,
.card-list-leave-active {
  transition: all 0.4s ease;
}

.card-list-enter-from {
  opacity: 0;
  transform: translateY(20px) scale(0.95);
}

.card-list-leave-to {
  opacity: 0;
  transform: translateY(-20px) scale(0.95);
}

.card-list-move {
  transition: transform 0.4s ease;
}

/* ============================================================
   空状态
   ============================================================ */

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem 2rem;
  text-align: center;
}

.empty-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
  opacity: 0.6;
}

.empty-text {
  font-size: 1.1rem;
  color: #8892b0;
  margin: 0 0 1.5rem;
}

.empty-refresh-btn {
  padding: 0.75rem 2rem;
  background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
  border: none;
  border-radius: 2rem;
  color: white;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.empty-refresh-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 8px 24px rgba(79, 172, 254, 0.4);
}

/* ============================================================
   加载更多
   ============================================================ */

.load-more-trigger {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  min-height: 80px;
}

.loading-more {
  display: flex;
  gap: 0.4rem;
  margin-bottom: 0.75rem;
}

.loading-dot {
  width: 8px;
  height: 8px;
  background: #4facfe;
  border-radius: 50%;
  animation: bounce 1.4s ease-in-out infinite;
}

.loading-dot:nth-child(1) {
  animation-delay: 0s;
}

.loading-dot:nth-child(2) {
  animation-delay: 0.2s;
}

.loading-dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes bounce {
  0%, 80%, 100% {
    transform: scale(0.6);
    opacity: 0.5;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}

.loading-text {
  font-size: 0.85rem;
  color: #8892b0;
}

.no-more-text {
  font-size: 0.85rem;
  color: #5a6378;
}

/* ============================================================
   响应式适配
   ============================================================ */

@media (max-width: 640px) {
  .list-header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .list-title {
    font-size: 1.2rem;
  }
  
  .refresh-btn {
    width: 100%;
    justify-content: center;
  }
  
  .card-grid {
    gap: 1rem;
  }
}
</style>

