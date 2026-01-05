<script setup lang="ts">
/**
 * Search.vue - 搜索页面
 * 
 * 功能：
 * - 搜索输入和触发
 * - 搜索结果展示
 * - 结果过滤（按类型、分类）
 * - 热门搜索推荐
 * - URL 参数同步
 * 
 * Person B 开发
 */
import { ref, watch, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useItemStore } from '@/stores/item'
import SearchBar from '@/components/SearchBar.vue'
import SearchResults from '@/components/SearchResults.vue'
import type { Item } from '@shared/types'

const route = useRoute()
const router = useRouter()
const itemStore = useItemStore()

// =========================================================================
// 状态
// =========================================================================

/** 搜索关键词 */
const query = ref((route.query.q as string) || '')

/** 是否正在搜索 */
const isSearching = ref(false)

/** 搜索结果列表 */
const searchResults = ref<Item[]>([])

/** 是否已执行过搜索 */
const hasSearched = ref(false)

/** 搜索错误信息 */
const searchError = ref<string | null>(null)

// 过滤条件
const filters = ref({
  type: '' as string,
  category: '' as string,
})

// 热门搜索词
const hotSearches = ref([
  '科幻电影',
  '动作片',
  '喜剧电影',
  '编程教程',
  '美食视频',
  '数码产品',
  '经典老片',
  '纪录片',
])

// =========================================================================
// 计算属性
// =========================================================================

/** 过滤后的结果 */
const filteredResults = computed(() => {
  let results = searchResults.value
  
  if (filters.value.type) {
    results = results.filter(item => item.type === filters.value.type)
  }
  if (filters.value.category) {
    results = results.filter(item => item.category === filters.value.category)
  }
  
  return results
})

/** 可用的类型过滤选项 */
const availableTypes = computed(() => {
  const types = new Set(searchResults.value.map(item => item.type))
  return Array.from(types)
})

/** 可用的分类过滤选项 */
const availableCategories = computed(() => {
  const categories = new Set(searchResults.value.map(item => item.category).filter(Boolean))
  return Array.from(categories)
})

/** 结果数量文本 */
const resultCountText = computed(() => {
  const total = searchResults.value.length
  const filtered = filteredResults.value.length
  
  if (total === 0) return ''
  if (filtered === total) return `共 ${total} 个结果`
  return `显示 ${filtered} / ${total} 个结果`
})

// =========================================================================
// 方法
// =========================================================================

/**
 * 执行搜索
 * @param searchQuery 搜索关键词
 */
async function handleSearch(searchQuery: string) {
  if (!searchQuery.trim()) return
  
  query.value = searchQuery
  isSearching.value = true
  hasSearched.value = true
  searchError.value = null
  
  // 更新 URL 参数
  router.replace({ query: { q: searchQuery } })
  
  // 重置过滤条件
  filters.value = { type: '', category: '' }
  
  try {
    searchResults.value = await itemStore.searchItems(searchQuery, 50)
  } catch (error) {
    searchError.value = error instanceof Error ? error.message : '搜索失败，请稍后重试'
    searchResults.value = []
  } finally {
    isSearching.value = false
  }
}

/**
 * 清空搜索
 */
function handleClear() {
  query.value = ''
  searchResults.value = []
  hasSearched.value = false
  searchError.value = null
  filters.value = { type: '', category: '' }
  router.replace({ query: {} })
}

/**
 * 点击搜索结果项
 * @param itemId 物品ID
 */
function handleItemClick(itemId: string) {
  router.push(`/item/${itemId}`)
}

/**
 * 点击热门搜索词
 * @param tag 搜索词
 */
function handleHotTagClick(tag: string) {
  handleSearch(tag)
}

/**
 * 重置过滤条件
 */
function resetFilters() {
  filters.value = { type: '', category: '' }
}

// =========================================================================
// 监听器
// =========================================================================

// 监听路由参数变化
watch(() => route.query.q, (newQuery) => {
  if (newQuery && newQuery !== query.value) {
    handleSearch(newQuery as string)
  }
}, { immediate: true })

// =========================================================================
// 生命周期
// =========================================================================

onMounted(() => {
  // 如果 URL 中有搜索参数，执行搜索
  if (route.query.q) {
    handleSearch(route.query.q as string)
  }
})
</script>

<template>
  <div class="search-page">
    <!-- 搜索头部 -->
    <header class="search-header">
      <div class="header-content">
        <h1 class="page-title">探索发现</h1>
        <SearchBar
          v-model="query"
          @search="handleSearch"
          @clear="handleClear"
          placeholder="搜索电影、商品、文章..."
          :loading="isSearching"
          :autofocus="!hasSearched"
          data-testid="main-search-bar"
        />
      </div>
    </header>

    <!-- 主内容区 -->
    <main class="search-content">
      <!-- 过滤器栏 -->
      <div v-if="hasSearched && searchResults.length > 0" class="filter-bar">
        <div class="filter-group">
          <label class="filter-label">类型</label>
          <select 
            v-model="filters.type" 
            class="filter-select"
            data-testid="type-filter"
          >
            <option value="">全部类型</option>
            <option v-for="type in availableTypes" :key="type" :value="type">
              {{ type === 'movie' ? '电影' : type === 'product' ? '商品' : type === 'article' ? '文章' : '视频' }}
            </option>
          </select>
        </div>
        
        <div v-if="availableCategories.length > 0" class="filter-group">
          <label class="filter-label">分类</label>
          <select 
            v-model="filters.category" 
            class="filter-select"
            data-testid="category-filter"
          >
            <option value="">全部分类</option>
            <option v-for="category in availableCategories" :key="category" :value="category">
              {{ category }}
            </option>
          </select>
        </div>
        
        <button 
          v-if="filters.type || filters.category"
          class="reset-filters-btn"
          @click="resetFilters"
        >
          重置筛选
        </button>
        
        <span class="result-count">{{ resultCountText }}</span>
      </div>

      <!-- 加载状态 -->
      <div v-if="isSearching" class="loading-state" data-testid="loading-state">
        <div class="loading-animation">
          <div class="loading-dot"></div>
          <div class="loading-dot"></div>
          <div class="loading-dot"></div>
        </div>
        <p class="loading-text">正在搜索...</p>
      </div>

      <!-- 错误状态 -->
      <div v-else-if="searchError" class="error-state" data-testid="error-state">
        <div class="error-icon">⚠️</div>
        <h2>搜索出错</h2>
        <p>{{ searchError }}</p>
        <button class="retry-btn" @click="handleSearch(query)">重新搜索</button>
      </div>

      <!-- 搜索结果 -->
      <SearchResults
        v-else-if="hasSearched"
        :items="filteredResults"
        :query="query"
        @item-click="handleItemClick"
        data-testid="search-results"
      />

      <!-- 初始状态 - 热门搜索 -->
      <div v-else class="initial-state" data-testid="initial-state">
        <div class="welcome-section">
          <div class="search-illustration">
            <svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="11" cy="11" r="8"></circle>
              <path d="m21 21-4.3-4.3"></path>
            </svg>
          </div>
          <h2>发现精彩内容</h2>
          <p>输入关键词，探索你感兴趣的电影、商品和更多</p>
        </div>
        
        <!-- 热门搜索 -->
        <div class="hot-searches">
          <h3>
            <span class="hot-icon">🔥</span>
            热门搜索
          </h3>
          <div class="hot-tags">
            <button 
              v-for="tag in hotSearches"
              :key="tag"
              @click="handleHotTagClick(tag)"
              class="hot-tag"
              data-testid="hot-tag"
            >
              {{ tag }}
            </button>
          </div>
        </div>
        
        <!-- 最近浏览 -->
        <div v-if="itemStore.recentlyViewed.length > 0" class="recent-section">
          <h3>
            <span class="recent-icon">🕐</span>
            最近浏览
          </h3>
          <div class="recent-items">
            <div 
              v-for="item in itemStore.recentlyViewed.slice(0, 6)"
              :key="item.id"
              class="recent-item"
              @click="handleItemClick(item.id)"
            >
              <div class="recent-item-cover">
                <span class="recent-item-type">{{ item.type }}</span>
              </div>
              <span class="recent-item-title">{{ item.title }}</span>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<style scoped>
.search-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1b2a 100%);
  color: #e2e8f0;
}

/* 搜索头部 */
.search-header {
  padding: 2rem 1.5rem;
  background: rgba(0, 0, 0, 0.4);
  position: sticky;
  top: 0;
  z-index: 100;
  backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.header-content {
  max-width: 900px;
  margin: 0 auto;
}

.page-title {
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
  text-align: center;
  background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-family: 'Playfair Display', 'Noto Serif SC', serif;
}

/* 过滤器栏 */
.filter-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 1rem;
  padding: 1rem 1.5rem;
  background: rgba(255, 255, 255, 0.03);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.filter-group {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.filter-label {
  font-size: 0.85rem;
  color: #94a3b8;
}

.filter-select {
  padding: 0.5rem 1rem;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 0.5rem;
  color: #f1f5f9;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s;
  font-family: 'Nunito', 'PingFang SC', sans-serif;
}

.filter-select:hover {
  border-color: rgba(99, 102, 241, 0.5);
}

.filter-select:focus {
  outline: none;
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
}

.reset-filters-btn {
  padding: 0.5rem 1rem;
  background: rgba(239, 68, 68, 0.15);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 0.5rem;
  color: #fca5a5;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s;
}

.reset-filters-btn:hover {
  background: rgba(239, 68, 68, 0.25);
}

.result-count {
  margin-left: auto;
  font-size: 0.9rem;
  color: #94a3b8;
}

/* 主内容区 */
.search-content {
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
}

/* 加载状态 */
.loading-state {
  text-align: center;
  padding: 6rem 2rem;
}

.loading-animation {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
}

.loading-dot {
  width: 12px;
  height: 12px;
  background: linear-gradient(135deg, #6366f1, #a855f7);
  border-radius: 50%;
  animation: loading-bounce 1.4s ease-in-out infinite;
}

.loading-dot:nth-child(1) { animation-delay: 0s; }
.loading-dot:nth-child(2) { animation-delay: 0.2s; }
.loading-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes loading-bounce {
  0%, 80%, 100% {
    transform: scale(0.8);
    opacity: 0.5;
  }
  40% {
    transform: scale(1.2);
    opacity: 1;
  }
}

.loading-text {
  color: #94a3b8;
  font-size: 1rem;
}

/* 错误状态 */
.error-state {
  text-align: center;
  padding: 4rem 2rem;
}

.error-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.error-state h2 {
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
  color: #fca5a5;
}

.error-state p {
  color: #94a3b8;
  margin-bottom: 1.5rem;
}

.retry-btn {
  padding: 0.75rem 2rem;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  border: none;
  border-radius: 0.5rem;
  color: #fff;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s;
}

.retry-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
}

/* 初始状态 */
.initial-state {
  padding: 2rem 0;
}

.welcome-section {
  text-align: center;
  margin-bottom: 3rem;
}

.search-illustration {
  color: #6366f1;
  opacity: 0.6;
  margin-bottom: 1.5rem;
}

.welcome-section h2 {
  font-size: 2rem;
  font-weight: 600;
  margin-bottom: 0.75rem;
  color: #f1f5f9;
}

.welcome-section p {
  color: #94a3b8;
  font-size: 1.1rem;
}

/* 热门搜索 */
.hot-searches {
  margin-bottom: 3rem;
}

.hot-searches h3,
.recent-section h3 {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #e2e8f0;
}

.hot-icon, .recent-icon {
  font-size: 1.2rem;
}

.hot-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.hot-tag {
  padding: 0.6rem 1.25rem;
  background: rgba(99, 102, 241, 0.1);
  border: 1px solid rgba(99, 102, 241, 0.25);
  border-radius: 2rem;
  color: #a5b4fc;
  cursor: pointer;
  transition: all 0.3s;
  font-size: 0.95rem;
  font-family: 'Nunito', 'PingFang SC', sans-serif;
}

.hot-tag:hover {
  background: rgba(99, 102, 241, 0.2);
  border-color: rgba(99, 102, 241, 0.4);
  transform: translateY(-2px);
  color: #c7d2fe;
}

/* 最近浏览 */
.recent-section {
  margin-top: 3rem;
}

.recent-items {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 1rem;
}

.recent-item {
  cursor: pointer;
  transition: all 0.3s;
}

.recent-item:hover {
  transform: translateY(-4px);
}

.recent-item-cover {
  aspect-ratio: 16/10;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(168, 85, 247, 0.3));
  border-radius: 0.75rem;
  display: flex;
  align-items: flex-end;
  justify-content: flex-start;
  padding: 0.5rem;
  margin-bottom: 0.5rem;
}

.recent-item-type {
  padding: 0.25rem 0.5rem;
  background: rgba(0, 0, 0, 0.6);
  border-radius: 0.25rem;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.recent-item-title {
  font-size: 0.9rem;
  color: #cbd5e1;
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 响应式 */
@media (max-width: 768px) {
  .search-header {
    padding: 1.5rem 1rem;
  }
  
  .page-title {
    font-size: 1.5rem;
    margin-bottom: 1rem;
  }
  
  .filter-bar {
    padding: 0.75rem 1rem;
  }
  
  .search-content {
    padding: 1.5rem 1rem;
  }
  
  .welcome-section h2 {
    font-size: 1.5rem;
  }
  
  .recent-items {
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  }
}
</style>

