<script setup lang="ts">
/**
 * ItemInfo.vue - 物品信息展示组件
 * 
 * 功能：
 * - 展示物品详细信息（标题、描述、分类、标签等）
 * - 展示物品统计数据（浏览数、点赞数、评分等）
 * - 响应式布局
 * 
 * Person B 开发
 */
import { computed } from 'vue'
import type { Item, ItemStats } from '@shared/types'

interface Props {
  /** 物品信息 */
  item: Item
  /** 物品统计（可选） */
  stats?: ItemStats | null
}

const props = defineProps<Props>()

// =========================================================================
// 计算属性
// =========================================================================

/** 类型显示名 */
const typeDisplayName = computed(() => {
  const names: Record<string, string> = {
    movie: '电影',
    product: '商品',
    article: '文章',
    video: '视频',
  }
  return names[props.item.type] || props.item.type
})

/** 状态显示名 */
const statusDisplayName = computed(() => {
  return props.item.status === 'active' ? '已发布' : '已下架'
})

/** 状态样式类 */
const statusClass = computed(() => {
  return props.item.status === 'active' ? 'status-active' : 'status-inactive'
})

/** 格式化日期 */
function formatDate(dateString: string): string {
  const date = new Date(dateString)
  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

/** 格式化数字（简化显示） */
function formatNumber(num: number): string {
  if (num >= 10000) {
    return (num / 10000).toFixed(1) + '万'
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'k'
  }
  return num.toString()
}

/** 格式化评分 */
function formatRating(rating: number): string {
  return rating.toFixed(1)
}

/** 元数据列表 */
const metadataItems = computed(() => {
  if (!props.item.metadata) return []
  
  const items: { label: string; value: string }[] = []
  const metadata = props.item.metadata
  
  // 根据物品类型显示不同的元数据
  if (props.item.type === 'movie') {
    if (metadata.director) items.push({ label: '导演', value: metadata.director })
    if (metadata.cast) items.push({ label: '主演', value: metadata.cast })
    if (metadata.year) items.push({ label: '年份', value: metadata.year })
    if (metadata.duration) items.push({ label: '时长', value: metadata.duration })
    if (metadata.region) items.push({ label: '地区', value: metadata.region })
  } else if (props.item.type === 'product') {
    if (metadata.brand) items.push({ label: '品牌', value: metadata.brand })
    if (metadata.price) items.push({ label: '价格', value: '¥' + metadata.price })
    if (metadata.stock) items.push({ label: '库存', value: metadata.stock })
  } else if (props.item.type === 'article') {
    if (metadata.author) items.push({ label: '作者', value: metadata.author })
    if (metadata.wordCount) items.push({ label: '字数', value: formatNumber(metadata.wordCount) })
    if (metadata.source) items.push({ label: '来源', value: metadata.source })
  } else if (props.item.type === 'video') {
    if (metadata.creator) items.push({ label: '创作者', value: metadata.creator })
    if (metadata.duration) items.push({ label: '时长', value: metadata.duration })
    if (metadata.quality) items.push({ label: '画质', value: metadata.quality })
  }
  
  return items
})
</script>

<template>
  <div class="item-info">
    <!-- 标题区域 -->
    <header class="info-header">
      <h1 class="item-title" data-testid="item-title">{{ item.title }}</h1>
      
      <div class="item-badges">
        <span class="type-badge" data-testid="item-type">{{ typeDisplayName }}</span>
        <span :class="['status-badge', statusClass]" data-testid="item-status">
          {{ statusDisplayName }}
        </span>
      </div>
    </header>

    <!-- 统计数据 -->
    <div v-if="stats" class="stats-bar" data-testid="stats-bar">
      <div class="stat-item">
        <span class="stat-icon">👁️</span>
        <span class="stat-value">{{ formatNumber(stats.view_count) }}</span>
        <span class="stat-label">浏览</span>
      </div>
      
      <div class="stat-item">
        <span class="stat-icon">❤️</span>
        <span class="stat-value">{{ formatNumber(stats.like_count) }}</span>
        <span class="stat-label">喜欢</span>
      </div>
      
      <div class="stat-item">
        <span class="stat-icon">🔗</span>
        <span class="stat-value">{{ formatNumber(stats.share_count) }}</span>
        <span class="stat-label">分享</span>
      </div>
      
      <div v-if="stats.avg_rating > 0" class="stat-item rating">
        <span class="stat-icon">⭐</span>
        <span class="stat-value">{{ formatRating(stats.avg_rating) }}</span>
        <span class="stat-label">评分</span>
      </div>
    </div>

    <!-- 分类和标签 -->
    <div class="category-tags">
      <span v-if="item.category" class="category" data-testid="item-category">
        <span class="category-icon">📁</span>
        {{ item.category }}
      </span>
      
      <div v-if="item.tags?.length" class="tags" data-testid="item-tags">
        <span 
          v-for="tag in item.tags" 
          :key="tag"
          class="tag"
        >
          {{ tag }}
        </span>
      </div>
    </div>

    <!-- 描述 -->
    <div v-if="item.description" class="description" data-testid="item-description">
      <h3 class="section-title">简介</h3>
      <p>{{ item.description }}</p>
    </div>

    <!-- 元数据 -->
    <div v-if="metadataItems.length > 0" class="metadata" data-testid="item-metadata">
      <h3 class="section-title">详细信息</h3>
      <dl class="metadata-list">
        <div 
          v-for="meta in metadataItems" 
          :key="meta.label"
          class="metadata-item"
        >
          <dt>{{ meta.label }}</dt>
          <dd>{{ meta.value }}</dd>
        </div>
      </dl>
    </div>

    <!-- 时间信息 -->
    <div class="timestamps">
      <span class="timestamp">
        创建于 {{ formatDate(item.created_at) }}
      </span>
      <span v-if="item.updated_at !== item.created_at" class="timestamp">
        更新于 {{ formatDate(item.updated_at) }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.item-info {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

/* 标题区域 */
.info-header {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.item-title {
  font-size: 2rem;
  font-weight: 700;
  color: #f1f5f9;
  line-height: 1.3;
  margin: 0;
  font-family: 'Playfair Display', 'Noto Serif SC', serif;
}

.item-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.type-badge {
  padding: 0.35rem 0.75rem;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2));
  border: 1px solid rgba(99, 102, 241, 0.3);
  border-radius: 0.5rem;
  font-size: 0.85rem;
  font-weight: 600;
  color: #a5b4fc;
}

.status-badge {
  padding: 0.35rem 0.75rem;
  border-radius: 0.5rem;
  font-size: 0.85rem;
  font-weight: 600;
}

.status-active {
  background: rgba(34, 197, 94, 0.15);
  border: 1px solid rgba(34, 197, 94, 0.3);
  color: #86efac;
}

.status-inactive {
  background: rgba(239, 68, 68, 0.15);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: #fca5a5;
}

/* 统计数据 */
.stats-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 1.5rem;
  padding: 1rem 1.25rem;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 0.75rem;
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.stat-icon {
  font-size: 1.1rem;
}

.stat-value {
  font-size: 1.1rem;
  font-weight: 700;
  color: #f1f5f9;
}

.stat-label {
  font-size: 0.85rem;
  color: #94a3b8;
}

.stat-item.rating .stat-value {
  color: #fbbf24;
}

/* 分类和标签 */
.category-tags {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem;
}

.category {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.4rem 0.75rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 0.5rem;
  font-size: 0.9rem;
  color: #cbd5e1;
}

.category-icon {
  font-size: 0.9rem;
}

.tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.tag {
  padding: 0.3rem 0.6rem;
  background: rgba(99, 102, 241, 0.1);
  border: 1px solid rgba(99, 102, 241, 0.2);
  border-radius: 0.4rem;
  font-size: 0.8rem;
  color: #a5b4fc;
}

/* 描述 */
.description {
  padding-top: 0.5rem;
}

.section-title {
  font-size: 1rem;
  font-weight: 600;
  color: #94a3b8;
  margin-bottom: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.description p {
  font-size: 1rem;
  line-height: 1.7;
  color: #cbd5e1;
  margin: 0;
}

/* 元数据 */
.metadata {
  padding-top: 0.5rem;
}

.metadata-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 0.75rem;
  margin: 0;
}

.metadata-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  padding: 0.75rem;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 0.5rem;
}

.metadata-item dt {
  font-size: 0.8rem;
  color: #64748b;
}

.metadata-item dd {
  font-size: 0.95rem;
  color: #e2e8f0;
  margin: 0;
}

/* 时间信息 */
.timestamps {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  padding-top: 0.5rem;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
}

.timestamp {
  font-size: 0.8rem;
  color: #64748b;
}

/* 响应式 */
@media (max-width: 640px) {
  .item-title {
    font-size: 1.5rem;
  }
  
  .stats-bar {
    gap: 1rem;
  }
  
  .metadata-list {
    grid-template-columns: 1fr 1fr;
  }
}
</style>

