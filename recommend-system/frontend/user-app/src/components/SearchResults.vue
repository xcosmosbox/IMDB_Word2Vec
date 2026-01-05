<script setup lang="ts">
/**
 * SearchResults.vue - 搜索结果展示组件
 * 
 * 功能：
 * - 网格/列表布局展示搜索结果
 * - 空结果状态展示
 * - 结果高亮关键词
 * - 响应式布局
 * 
 * Person B 开发
 */
import { computed } from 'vue'
import type { Item } from '@shared/types'

interface Props {
  /** 搜索结果列表 */
  items: Item[]
  /** 搜索关键词（用于高亮） */
  query?: string
}

const props = withDefaults(defineProps<Props>(), {
  query: '',
})

const emit = defineEmits<{
  /** 点击结果项 */
  'item-click': [itemId: string]
}>()

// =========================================================================
// 计算属性
// =========================================================================

/** 是否有结果 */
const hasResults = computed(() => props.items.length > 0)

/** 类型显示名映射 */
const typeNames: Record<string, string> = {
  movie: '电影',
  product: '商品',
  article: '文章',
  video: '视频',
}

/** 类型图标映射 */
const typeIcons: Record<string, string> = {
  movie: '🎬',
  product: '🛍️',
  article: '📄',
  video: '🎥',
}

/** 类型渐变色映射 */
const typeGradients: Record<string, string> = {
  movie: 'linear-gradient(135deg, #dc2626 0%, #9333ea 100%)',
  product: 'linear-gradient(135deg, #059669 0%, #0891b2 100%)',
  article: 'linear-gradient(135deg, #ea580c 0%, #ca8a04 100%)',
  video: 'linear-gradient(135deg, #db2777 0%, #9333ea 100%)',
}

// =========================================================================
// 方法
// =========================================================================

/**
 * 获取类型显示名
 * @param type 物品类型
 */
function getTypeName(type: string): string {
  return typeNames[type] || type
}

/**
 * 获取类型图标
 * @param type 物品类型
 */
function getTypeIcon(type: string): string {
  return typeIcons[type] || '📦'
}

/**
 * 获取类型渐变色
 * @param type 物品类型
 */
function getTypeGradient(type: string): string {
  return typeGradients[type] || 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)'
}

/**
 * 高亮搜索关键词
 * @param text 原始文本
 */
function highlightText(text: string): string {
  if (!props.query || !text) return text
  
  const query = props.query.trim()
  if (!query) return text
  
  // 转义正则特殊字符
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const regex = new RegExp(`(${escaped})`, 'gi')
  
  return text.replace(regex, '<mark class="highlight">$1</mark>')
}

/**
 * 截断文本
 * @param text 原始文本
 * @param maxLength 最大长度
 */
function truncateText(text: string, maxLength: number): string {
  if (!text) return ''
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength) + '...'
}

/**
 * 处理结果项点击
 * @param itemId 物品ID
 */
function handleItemClick(itemId: string) {
  emit('item-click', itemId)
}
</script>

<template>
  <div class="search-results">
    <!-- 有结果 -->
    <div v-if="hasResults" class="results-grid" data-testid="results-grid">
      <article
        v-for="item in items"
        :key="item.id"
        class="result-card"
        @click="handleItemClick(item.id)"
        data-testid="result-card"
      >
        <!-- 封面 -->
        <div 
          class="card-cover"
          :style="{ background: getTypeGradient(item.type) }"
        >
          <span class="card-type-icon">{{ getTypeIcon(item.type) }}</span>
          <span class="card-type-badge">{{ getTypeName(item.type) }}</span>
        </div>
        
        <!-- 内容 -->
        <div class="card-content">
          <h3 
            class="card-title"
            v-html="highlightText(item.title)"
          ></h3>
          
          <p 
            v-if="item.description"
            class="card-description"
            v-html="highlightText(truncateText(item.description, 80))"
          ></p>
          
          <div class="card-meta">
            <span v-if="item.category" class="card-category">
              {{ item.category }}
            </span>
            <div v-if="item.tags?.length" class="card-tags">
              <span 
                v-for="tag in item.tags.slice(0, 2)" 
                :key="tag"
                class="card-tag"
              >
                {{ tag }}
              </span>
            </div>
          </div>
        </div>
        
        <!-- 悬浮指示 -->
        <div class="card-hover-indicator">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="m9 18 6-6-6-6"/>
          </svg>
        </div>
      </article>
    </div>

    <!-- 无结果 -->
    <div v-else class="empty-results" data-testid="empty-results">
      <div class="empty-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="11" cy="11" r="8"></circle>
          <path d="m21 21-4.3-4.3"></path>
          <path d="M8 8l6 6"></path>
          <path d="M14 8l-6 6"></path>
        </svg>
      </div>
      <h3>未找到相关结果</h3>
      <p v-if="query">
        未找到与 "<strong>{{ query }}</strong>" 相关的内容
      </p>
      <p v-else>暂无搜索结果</p>
      <div class="empty-tips">
        <p>建议：</p>
        <ul>
          <li>检查输入是否有误</li>
          <li>尝试使用其他关键词</li>
          <li>使用更简短的搜索词</li>
        </ul>
      </div>
    </div>
  </div>
</template>

<style scoped>
.search-results {
  width: 100%;
}

/* 结果网格 */
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5rem;
}

@media (max-width: 640px) {
  .results-grid {
    grid-template-columns: 1fr;
  }
}

/* 结果卡片 */
.result-card {
  display: flex;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 1rem;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
}

.result-card:hover {
  transform: translateY(-4px);
  background: rgba(255, 255, 255, 0.06);
  border-color: rgba(99, 102, 241, 0.3);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}

/* 封面 */
.card-cover {
  flex: 0 0 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  position: relative;
}

.card-type-icon {
  font-size: 2.5rem;
}

.card-type-badge {
  position: absolute;
  bottom: 0.5rem;
  left: 50%;
  transform: translateX(-50%);
  padding: 0.2rem 0.6rem;
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(4px);
  border-radius: 0.25rem;
  font-size: 0.7rem;
  font-weight: 600;
  white-space: nowrap;
}

/* 内容 */
.card-content {
  flex: 1;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  min-width: 0;
}

.card-title {
  font-size: 1rem;
  font-weight: 600;
  color: #f1f5f9;
  margin: 0;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-description {
  font-size: 0.85rem;
  color: #94a3b8;
  margin: 0;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin-top: auto;
}

.card-category {
  font-size: 0.75rem;
  color: #6366f1;
  background: rgba(99, 102, 241, 0.15);
  padding: 0.2rem 0.5rem;
  border-radius: 0.25rem;
}

.card-tags {
  display: flex;
  gap: 0.25rem;
}

.card-tag {
  font-size: 0.7rem;
  color: #94a3b8;
  background: rgba(255, 255, 255, 0.05);
  padding: 0.15rem 0.4rem;
  border-radius: 0.25rem;
}

/* 悬浮指示器 */
.card-hover-indicator {
  position: absolute;
  right: 0.75rem;
  top: 50%;
  transform: translateY(-50%) translateX(10px);
  opacity: 0;
  color: #6366f1;
  transition: all 0.3s;
}

.result-card:hover .card-hover-indicator {
  opacity: 1;
  transform: translateY(-50%) translateX(0);
}

/* 高亮样式 */
:deep(.highlight) {
  background: rgba(99, 102, 241, 0.3);
  color: #a5b4fc;
  padding: 0 0.15rem;
  border-radius: 0.15rem;
}

/* 空结果 */
.empty-results {
  text-align: center;
  padding: 4rem 2rem;
}

.empty-icon {
  color: #475569;
  margin-bottom: 1.5rem;
}

.empty-results h3 {
  font-size: 1.5rem;
  color: #e2e8f0;
  margin-bottom: 0.5rem;
}

.empty-results > p {
  color: #94a3b8;
  margin-bottom: 2rem;
}

.empty-results strong {
  color: #6366f1;
}

.empty-tips {
  text-align: left;
  display: inline-block;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 0.75rem;
  padding: 1rem 1.5rem;
}

.empty-tips p {
  color: #94a3b8;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}

.empty-tips ul {
  margin: 0;
  padding-left: 1.25rem;
}

.empty-tips li {
  color: #64748b;
  font-size: 0.85rem;
  line-height: 1.6;
}
</style>

