<script setup lang="ts">
/**
 * SimilarItems.vue - 相似物品推荐组件
 * 
 * 功能：
 * - 横向滚动展示相似物品
 * - 显示相似度分数
 * - 响应式布局
 * 
 * Person B 开发
 */
import { computed } from 'vue'
import type { SimilarItem } from '@shared/types'

interface Props {
  /** 相似物品列表 */
  items: SimilarItem[]
}

const props = defineProps<Props>()

const emit = defineEmits<{
  /** 点击物品 */
  'item-click': [itemId: string]
}>()

// =========================================================================
// 计算属性
// =========================================================================

/** 是否有相似物品 */
const hasItems = computed(() => props.items.length > 0)

/** 类型渐变色映射 */
const typeGradients: Record<string, string> = {
  movie: 'linear-gradient(135deg, #dc2626 0%, #9333ea 100%)',
  product: 'linear-gradient(135deg, #059669 0%, #0891b2 100%)',
  article: 'linear-gradient(135deg, #ea580c 0%, #ca8a04 100%)',
  video: 'linear-gradient(135deg, #db2777 0%, #9333ea 100%)',
}

/** 类型显示名映射 */
const typeNames: Record<string, string> = {
  movie: '电影',
  product: '商品',
  article: '文章',
  video: '视频',
}

// =========================================================================
// 方法
// =========================================================================

/**
 * 获取类型渐变色
 * @param type 物品类型
 */
function getTypeGradient(type: string): string {
  return typeGradients[type] || 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)'
}

/**
 * 获取类型显示名
 * @param type 物品类型
 */
function getTypeName(type: string): string {
  return typeNames[type] || type
}

/**
 * 格式化相似度分数
 * @param score 相似度分数 (0-1)
 */
function formatScore(score: number): string {
  return Math.round(score * 100) + '%'
}

/**
 * 获取相似度等级样式类
 * @param score 相似度分数
 */
function getScoreClass(score: number): string {
  if (score >= 0.9) return 'score-high'
  if (score >= 0.7) return 'score-medium'
  return 'score-low'
}

/**
 * 处理物品点击
 * @param itemId 物品ID
 */
function handleItemClick(itemId: string) {
  emit('item-click', itemId)
}
</script>

<template>
  <div class="similar-items">
    <!-- 有相似物品 -->
    <div v-if="hasItems" class="items-scroll" data-testid="items-scroll">
      <article
        v-for="{ item, score } in items"
        :key="item.id"
        class="similar-card"
        @click="handleItemClick(item.id)"
        data-testid="similar-card"
      >
        <!-- 封面 -->
        <div 
          class="card-cover"
          :style="{ background: getTypeGradient(item.type) }"
        >
          <!-- 相似度标签 -->
          <span :class="['similarity-badge', getScoreClass(score)]">
            {{ formatScore(score) }} 匹配
          </span>
          
          <!-- 类型标签 -->
          <span class="type-tag">{{ getTypeName(item.type) }}</span>
        </div>
        
        <!-- 信息 -->
        <div class="card-info">
          <h4 class="card-title">{{ item.title }}</h4>
          
          <p v-if="item.description" class="card-description">
            {{ item.description }}
          </p>
          
          <div class="card-meta">
            <span v-if="item.category" class="card-category">
              {{ item.category }}
            </span>
          </div>
        </div>
      </article>
    </div>

    <!-- 无相似物品 -->
    <div v-else class="empty-state" data-testid="empty-state">
      <p>暂无相似推荐</p>
    </div>
  </div>
</template>

<style scoped>
.similar-items {
  overflow: hidden;
}

/* 横向滚动容器 */
.items-scroll {
  display: flex;
  gap: 1rem;
  overflow-x: auto;
  padding: 0.5rem 0 1rem;
  scroll-snap-type: x mandatory;
  scrollbar-width: thin;
  scrollbar-color: rgba(99, 102, 241, 0.5) transparent;
}

.items-scroll::-webkit-scrollbar {
  height: 6px;
}

.items-scroll::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 3px;
}

.items-scroll::-webkit-scrollbar-thumb {
  background: linear-gradient(90deg, #6366f1, #8b5cf6);
  border-radius: 3px;
}

/* 相似物品卡片 */
.similar-card {
  flex: 0 0 220px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 0.75rem;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  scroll-snap-align: start;
}

.similar-card:hover {
  transform: translateY(-6px);
  border-color: rgba(99, 102, 241, 0.3);
  box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4);
}

/* 封面 */
.card-cover {
  aspect-ratio: 16/10;
  position: relative;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 0.5rem;
}

.similarity-badge {
  padding: 0.25rem 0.5rem;
  border-radius: 0.25rem;
  font-size: 0.7rem;
  font-weight: 700;
  backdrop-filter: blur(8px);
}

.score-high {
  background: rgba(34, 197, 94, 0.8);
  color: #fff;
}

.score-medium {
  background: rgba(234, 179, 8, 0.8);
  color: #fff;
}

.score-low {
  background: rgba(0, 0, 0, 0.6);
  color: #fff;
}

.type-tag {
  padding: 0.2rem 0.4rem;
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(4px);
  border-radius: 0.25rem;
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* 信息区域 */
.card-info {
  padding: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.card-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: #f1f5f9;
  margin: 0;
  line-height: 1.3;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-description {
  font-size: 0.75rem;
  color: #94a3b8;
  margin: 0;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-meta {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.25rem;
}

.card-category {
  font-size: 0.7rem;
  color: #6366f1;
  background: rgba(99, 102, 241, 0.1);
  padding: 0.15rem 0.4rem;
  border-radius: 0.25rem;
}

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 2rem;
  color: #64748b;
}

/* 响应式 */
@media (max-width: 640px) {
  .similar-card {
    flex: 0 0 180px;
  }
  
  .card-title {
    font-size: 0.85rem;
  }
}
</style>

