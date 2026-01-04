# Person B: 用户端 - 搜索与详情页

## 你的角色
你是一名前端工程师，负责实现生成式推荐系统的 **搜索功能和物品详情页** 模块。

---

## ⚠️ 重要：类型驱动开发

**开始编码前，必须先阅读类型定义文件：**

```
frontend/shared/types/index.ts
```

你需要使用的核心类型：

```typescript
interface Item { id, type, title, description, category, tags, ... }
interface SimilarItem { item, score }
interface ItemStats { view_count, click_count, like_count, ... }
```

---

## 技术栈

- **框架**: Vue 3 + Composition API + TypeScript
- **构建**: Vite
- **路由**: Vue Router
- **状态管理**: Pinia
- **HTTP**: Axios

---

## 你的任务

```
frontend/user-app/
├── src/
│   ├── views/
│   │   ├── Search.vue            # 搜索页面
│   │   └── ItemDetail.vue        # 物品详情页
│   ├── components/
│   │   ├── SearchBar.vue         # 搜索栏
│   │   ├── SearchResults.vue     # 搜索结果
│   │   ├── ItemInfo.vue          # 物品信息展示
│   │   ├── SimilarItems.vue      # 相似推荐
│   │   └── ActionButtons.vue     # 操作按钮组
│   └── ...
```

---

## 1. 搜索页面 (Search.vue)

```vue
<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useItemStore } from '@/stores/item'
import SearchBar from '@/components/SearchBar.vue'
import SearchResults from '@/components/SearchResults.vue'
import type { Item } from '@shared/types'

const route = useRoute()
const router = useRouter()
const itemStore = useItemStore()

// 搜索状态
const query = ref((route.query.q as string) || '')
const isSearching = ref(false)
const searchResults = ref<Item[]>([])
const hasSearched = ref(false)

// 过滤条件
const filters = ref({
  type: '',
  category: '',
})

// 过滤后的结果
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

// 搜索函数
async function handleSearch(searchQuery: string) {
  if (!searchQuery.trim()) return
  
  query.value = searchQuery
  isSearching.value = true
  hasSearched.value = true
  
  // 更新 URL
  router.replace({ query: { q: searchQuery } })
  
  try {
    searchResults.value = await itemStore.searchItems(searchQuery, 50)
  } finally {
    isSearching.value = false
  }
}

// 点击结果项
function handleItemClick(itemId: string) {
  router.push(`/item/${itemId}`)
}

// 监听路由参数变化
watch(() => route.query.q, (newQuery) => {
  if (newQuery && newQuery !== query.value) {
    handleSearch(newQuery as string)
  }
}, { immediate: true })
</script>

<template>
  <div class="search-page">
    <!-- 搜索头部 -->
    <header class="search-header">
      <SearchBar
        v-model="query"
        @search="handleSearch"
        placeholder="搜索电影、商品、文章..."
        :loading="isSearching"
      />
    </header>

    <!-- 过滤器 -->
    <div v-if="hasSearched" class="filter-bar">
      <select v-model="filters.type" class="filter-select">
        <option value="">全部类型</option>
        <option value="movie">电影</option>
        <option value="product">商品</option>
        <option value="article">文章</option>
        <option value="video">视频</option>
      </select>
    </div>

    <!-- 搜索结果 -->
    <main class="search-content">
      <!-- 加载状态 -->
      <div v-if="isSearching" class="loading-state">
        <div class="spinner"></div>
        <p>搜索中...</p>
      </div>

      <!-- 结果列表 -->
      <SearchResults
        v-else-if="hasSearched"
        :items="filteredResults"
        :query="query"
        @item-click="handleItemClick"
      />

      <!-- 初始状态 -->
      <div v-else class="initial-state">
        <div class="search-icon">🔍</div>
        <h2>搜索你感兴趣的内容</h2>
        <p>输入关键词开始搜索</p>
        
        <!-- 热门搜索 -->
        <div class="hot-searches">
          <h3>热门搜索</h3>
          <div class="hot-tags">
            <button 
              v-for="tag in ['科幻电影', 'iPhone', '编程教程', '美食视频']"
              :key="tag"
              @click="handleSearch(tag)"
              class="hot-tag"
            >
              {{ tag }}
            </button>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<style scoped>
.search-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #eee;
}

.search-header {
  padding: 2rem;
  background: rgba(0, 0, 0, 0.3);
  position: sticky;
  top: 0;
  z-index: 100;
  backdrop-filter: blur(10px);
}

.filter-bar {
  display: flex;
  gap: 1rem;
  padding: 1rem 2rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.filter-select {
  padding: 0.5rem 1rem;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 0.5rem;
  color: #fff;
  font-size: 0.9rem;
}

.search-content {
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.loading-state, .initial-state {
  text-align: center;
  padding: 4rem;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(79, 172, 254, 0.3);
  border-top-color: #4facfe;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.search-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.initial-state h2 {
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
}

.initial-state p {
  color: #8892b0;
}

.hot-searches {
  margin-top: 2rem;
}

.hot-searches h3 {
  font-size: 1rem;
  color: #8892b0;
  margin-bottom: 1rem;
}

.hot-tags {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.75rem;
}

.hot-tag {
  padding: 0.5rem 1rem;
  background: rgba(79, 172, 254, 0.1);
  border: 1px solid rgba(79, 172, 254, 0.3);
  border-radius: 2rem;
  color: #4facfe;
  cursor: pointer;
  transition: all 0.3s;
}

.hot-tag:hover {
  background: rgba(79, 172, 254, 0.2);
  transform: scale(1.05);
}
</style>
```

---

## 2. 物品详情页 (ItemDetail.vue)

```vue
<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useItemStore } from '@/stores/item'
import { useRecommendStore } from '@/stores/recommend'
import ItemInfo from '@/components/ItemInfo.vue'
import SimilarItems from '@/components/SimilarItems.vue'
import ActionButtons from '@/components/ActionButtons.vue'
import type { Item, SimilarItem, ItemStats } from '@shared/types'

const route = useRoute()
const router = useRouter()
const itemStore = useItemStore()
const recommendStore = useRecommendStore()

// 状态
const item = ref<Item | null>(null)
const stats = ref<ItemStats | null>(null)
const similarItems = ref<SimilarItem[]>([])
const isLoading = ref(true)
const isLiked = ref(false)

// 物品 ID
const itemId = computed(() => route.params.id as string)

// 加载数据
async function loadItemData() {
  isLoading.value = true
  
  try {
    // 并行加载
    const [itemData, statsData, similar] = await Promise.all([
      itemStore.getItem(itemId.value),
      itemStore.getItemStats(itemId.value),
      itemStore.getSimilarItems(itemId.value, 12),
    ])
    
    item.value = itemData
    stats.value = statsData
    similarItems.value = similar
    
    // 记录浏览行为
    recommendStore.recordBehavior({
      item_id: itemId.value,
      action: 'view',
    })
  } catch (error) {
    console.error('Failed to load item:', error)
  } finally {
    isLoading.value = false
  }
}

// 操作处理
async function handleLike() {
  isLiked.value = !isLiked.value
  await recommendStore.recordBehavior({
    item_id: itemId.value,
    action: isLiked.value ? 'like' : 'dislike',
  })
}

async function handleShare() {
  await recommendStore.recordBehavior({
    item_id: itemId.value,
    action: 'share',
  })
  // 打开分享面板
}

function handleSimilarClick(id: string) {
  router.push(`/item/${id}`)
}

onMounted(() => {
  loadItemData()
})
</script>

<template>
  <div class="detail-page">
    <!-- 返回按钮 -->
    <button class="back-btn" @click="router.back()">
      ← 返回
    </button>

    <!-- 加载状态 -->
    <div v-if="isLoading" class="loading-container">
      <div class="skeleton-header"></div>
      <div class="skeleton-content"></div>
    </div>

    <!-- 详情内容 -->
    <template v-else-if="item">
      <article class="detail-content">
        <!-- 封面区域 -->
        <div class="cover-section">
          <div 
            class="cover-image"
            :style="{ 
              background: `linear-gradient(135deg, 
                ${item.type === 'movie' ? '#e50914' : '#4facfe'} 0%, 
                ${item.type === 'movie' ? '#b20710' : '#00f2fe'} 100%)` 
            }"
          >
            <span class="type-badge">{{ item.type }}</span>
          </div>
        </div>

        <!-- 信息区域 -->
        <div class="info-section">
          <ItemInfo :item="item" :stats="stats" />
          
          <!-- 操作按钮 -->
          <ActionButtons
            :is-liked="isLiked"
            @like="handleLike"
            @share="handleShare"
          />
        </div>
      </article>

      <!-- 相似推荐 -->
      <section class="similar-section">
        <h2 class="section-title">相似推荐</h2>
        <SimilarItems
          :items="similarItems"
          @item-click="handleSimilarClick"
        />
      </section>
    </template>

    <!-- 404 状态 -->
    <div v-else class="not-found">
      <h2>物品不存在</h2>
      <button @click="router.push('/')">返回首页</button>
    </div>
  </div>
</template>

<style scoped>
.detail-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #eee;
  padding: 2rem;
}

.back-btn {
  display: inline-flex;
  align-items: center;
  padding: 0.5rem 1rem;
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 0.5rem;
  color: #fff;
  cursor: pointer;
  margin-bottom: 2rem;
  transition: background 0.3s;
}

.back-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.detail-content {
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 3rem;
  max-width: 1200px;
  margin: 0 auto;
}

@media (max-width: 900px) {
  .detail-content {
    grid-template-columns: 1fr;
  }
}

.cover-section {
  position: relative;
}

.cover-image {
  aspect-ratio: 2/3;
  border-radius: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

.type-badge {
  position: absolute;
  top: 1rem;
  left: 1rem;
  padding: 0.5rem 1rem;
  background: rgba(0, 0, 0, 0.5);
  border-radius: 0.5rem;
  font-weight: 600;
  text-transform: uppercase;
}

.info-section {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.similar-section {
  margin-top: 4rem;
  max-width: 1200px;
  margin-left: auto;
  margin-right: auto;
}

.section-title {
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
  padding-left: 1rem;
  border-left: 4px solid #4facfe;
}

.loading-container {
  max-width: 1200px;
  margin: 0 auto;
}

.skeleton-header {
  height: 400px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 1rem;
  animation: pulse 1.5s infinite;
}

.skeleton-content {
  height: 200px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 1rem;
  margin-top: 2rem;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.not-found {
  text-align: center;
  padding: 4rem;
}
</style>
```

---

## 3. 搜索栏 (SearchBar.vue)

```vue
<script setup lang="ts">
import { ref, watch } from 'vue'

interface Props {
  modelValue: string
  placeholder?: string
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  placeholder: '搜索...',
  loading: false,
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
  'search': [query: string]
}>()

const inputValue = ref(props.modelValue)

// 同步外部值
watch(() => props.modelValue, (val) => {
  inputValue.value = val
})

// 处理输入
function handleInput(e: Event) {
  const value = (e.target as HTMLInputElement).value
  inputValue.value = value
  emit('update:modelValue', value)
}

// 处理搜索
function handleSearch() {
  emit('search', inputValue.value)
}

// 处理键盘事件
function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter') {
    handleSearch()
  }
}

// 清空输入
function handleClear() {
  inputValue.value = ''
  emit('update:modelValue', '')
}
</script>

<template>
  <div class="search-bar">
    <div class="search-input-wrapper">
      <span class="search-icon">🔍</span>
      
      <input
        type="text"
        class="search-input"
        :value="inputValue"
        :placeholder="placeholder"
        @input="handleInput"
        @keydown="handleKeydown"
      />
      
      <button
        v-if="inputValue"
        class="clear-btn"
        @click="handleClear"
      >
        ✕
      </button>
    </div>
    
    <button
      class="search-btn"
      :disabled="loading"
      @click="handleSearch"
    >
      <span v-if="loading" class="loading-spinner"></span>
      <span v-else>搜索</span>
    </button>
  </div>
</template>

<style scoped>
.search-bar {
  display: flex;
  gap: 1rem;
  max-width: 800px;
  margin: 0 auto;
}

.search-input-wrapper {
  flex: 1;
  display: flex;
  align-items: center;
  background: rgba(255, 255, 255, 0.1);
  border: 2px solid rgba(255, 255, 255, 0.2);
  border-radius: 3rem;
  padding: 0 1.5rem;
  transition: all 0.3s;
}

.search-input-wrapper:focus-within {
  border-color: #4facfe;
  box-shadow: 0 0 20px rgba(79, 172, 254, 0.3);
}

.search-icon {
  font-size: 1.25rem;
  margin-right: 0.75rem;
}

.search-input {
  flex: 1;
  padding: 1rem 0;
  background: transparent;
  border: none;
  outline: none;
  color: #fff;
  font-size: 1.1rem;
}

.search-input::placeholder {
  color: #8892b0;
}

.clear-btn {
  padding: 0.5rem;
  background: transparent;
  border: none;
  color: #8892b0;
  cursor: pointer;
  transition: color 0.3s;
}

.clear-btn:hover {
  color: #fff;
}

.search-btn {
  padding: 1rem 2rem;
  background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
  border: none;
  border-radius: 3rem;
  color: #fff;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
  min-width: 100px;
}

.search-btn:hover:not(:disabled) {
  transform: scale(1.05);
  box-shadow: 0 10px 30px rgba(79, 172, 254, 0.4);
}

.search-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.loading-spinner {
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
```

---

## 4. 相似推荐 (SimilarItems.vue)

```vue
<script setup lang="ts">
import type { SimilarItem } from '@shared/types'

interface Props {
  items: SimilarItem[]
}

defineProps<Props>()

const emit = defineEmits<{
  'item-click': [itemId: string]
}>()
</script>

<template>
  <div class="similar-items">
    <div class="items-scroll">
      <div
        v-for="{ item, score } in items"
        :key="item.id"
        class="similar-card"
        @click="emit('item-click', item.id)"
      >
        <div class="card-cover">
          <span class="similarity">{{ Math.round(score * 100) }}%</span>
        </div>
        <div class="card-info">
          <h4 class="card-title">{{ item.title }}</h4>
          <span class="card-type">{{ item.type }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.similar-items {
  overflow: hidden;
}

.items-scroll {
  display: flex;
  gap: 1rem;
  overflow-x: auto;
  padding: 1rem 0;
  scroll-snap-type: x mandatory;
}

.items-scroll::-webkit-scrollbar {
  height: 6px;
}

.items-scroll::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
}

.items-scroll::-webkit-scrollbar-thumb {
  background: #4facfe;
  border-radius: 3px;
}

.similar-card {
  flex: 0 0 200px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 0.75rem;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.3s;
  scroll-snap-align: start;
}

.similar-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.card-cover {
  aspect-ratio: 16/10;
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  display: flex;
  align-items: flex-end;
  justify-content: flex-end;
  padding: 0.5rem;
}

.similarity {
  padding: 0.25rem 0.5rem;
  background: rgba(0, 0, 0, 0.7);
  border-radius: 0.25rem;
  font-size: 0.75rem;
  font-weight: 600;
}

.card-info {
  padding: 0.75rem;
}

.card-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff;
  margin-bottom: 0.25rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.card-type {
  font-size: 0.75rem;
  color: #8892b0;
  text-transform: uppercase;
}
</style>
```

---

## 注意事项

1. 所有类型从 `@shared/types` 导入
2. 使用 Vue Router 进行页面导航
3. 搜索支持 URL 参数 (`?q=xxx`)
4. 记录用户行为（view, click）

## 输出要求

请输出完整的可运行代码，包含：
1. 所有 Vue 组件
2. TypeScript 类型正确
3. 完整的样式
4. 骨架屏/加载状态

