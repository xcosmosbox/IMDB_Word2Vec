# Person A: 用户端 - 首页与推荐展示

## 你的角色
你是一名前端工程师，负责实现生成式推荐系统的 **用户端首页和推荐展示** 模块。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读以下文件：**

1. **数据类型定义：**
```
frontend/shared/types/index.ts
```

2. **服务接口定义（核心）：**
```
frontend/shared/api/interfaces.ts
```

你需要使用的核心接口：

```typescript
// 推荐服务接口
interface IRecommendService {
  getRecommendations(request: RecommendRequest): Promise<RecommendResponse>
  submitFeedback(feedback: FeedbackRequest): Promise<void>
}

// 用户服务接口
interface IUserService {
  recordBehavior(data: RecordBehaviorRequest): Promise<void>
}
```

**⚠️ 不要直接导入具体实现！** 使用依赖注入：

```typescript
// ✅ 正确：通过 inject 获取接口
const api = inject<IApiProvider>('api')!
await api.recommend.getRecommendations(request)

// ❌ 错误：直接导入具体实现
import { recommendApi } from '@shared/api'
```

---

## 技术栈

- **框架**: Vue 3 + Composition API + TypeScript
- **构建**: Vite
- **UI**: 自定义组件 (不使用 UI 库，追求独特设计)
- **状态管理**: Pinia
- **HTTP**: Axios

---

## 你的任务

```
frontend/user-app/
├── src/
│   ├── views/
│   │   └── Home.vue              # 首页
│   ├── components/
│   │   ├── RecommendList.vue     # 推荐列表
│   │   ├── ItemCard.vue          # 物品卡片
│   │   ├── CategoryTabs.vue      # 分类标签页
│   │   └── LoadingSpinner.vue    # 加载动画
│   └── ...
```

---

## 1. 首页 (Home.vue)

```vue
<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRecommendStore } from '@/stores/recommend'
import { useUserStore } from '@/stores/user'
import RecommendList from '@/components/RecommendList.vue'
import CategoryTabs from '@/components/CategoryTabs.vue'

const recommendStore = useRecommendStore()
const userStore = useUserStore()

const activeCategory = ref('all')
const isLoading = ref(true)

// 分类列表
const categories = [
  { key: 'all', label: '为你推荐' },
  { key: 'movie', label: '电影' },
  { key: 'product', label: '商品' },
  { key: 'article', label: '文章' },
  { key: 'video', label: '视频' },
]

// 根据分类过滤推荐
const filteredRecommendations = computed(() => {
  if (activeCategory.value === 'all') {
    return recommendStore.recommendations
  }
  return recommendStore.recommendations.filter(
    r => r.item?.type === activeCategory.value
  )
})

// 加载推荐
async function loadRecommendations() {
  isLoading.value = true
  try {
    await recommendStore.fetchRecommendations({
      user_id: userStore.currentUser?.id || 'anonymous',
      limit: 50,
      scene: 'home',
    })
  } finally {
    isLoading.value = false
  }
}

// 刷新推荐
async function refreshRecommendations() {
  await loadRecommendations()
}

// 处理物品点击
function handleItemClick(itemId: string) {
  // 记录点击行为
  recommendStore.recordBehavior({
    item_id: itemId,
    action: 'click',
  })
  // 导航到详情页
  // router.push(`/item/${itemId}`)
}

onMounted(() => {
  loadRecommendations()
})
</script>

<template>
  <div class="home-page">
    <!-- 顶部横幅 -->
    <header class="hero-section">
      <h1 class="hero-title">发现你的下一个最爱</h1>
      <p class="hero-subtitle">基于 AI 的个性化推荐</p>
    </header>

    <!-- 分类标签 -->
    <CategoryTabs
      :categories="categories"
      v-model:active="activeCategory"
    />

    <!-- 推荐列表 -->
    <main class="content-section">
      <div v-if="isLoading" class="loading-container">
        <LoadingSpinner />
      </div>
      
      <template v-else>
        <RecommendList
          :recommendations="filteredRecommendations"
          @item-click="handleItemClick"
          @refresh="refreshRecommendations"
        />
      </template>
    </main>
  </div>
</template>

<style scoped>
.home-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #eee;
}

.hero-section {
  padding: 4rem 2rem;
  text-align: center;
  background: linear-gradient(180deg, rgba(79, 172, 254, 0.1) 0%, transparent 100%);
}

.hero-title {
  font-size: 2.5rem;
  font-weight: 700;
  background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 1rem;
}

.hero-subtitle {
  font-size: 1.2rem;
  color: #8892b0;
}

.content-section {
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
}

.loading-container {
  display: flex;
  justify-content: center;
  padding: 4rem;
}
</style>
```

---

## 2. 推荐列表 (RecommendList.vue)

```vue
<script setup lang="ts">
import { ref } from 'vue'
import type { Recommendation } from '@shared/types'
import ItemCard from './ItemCard.vue'

interface Props {
  recommendations: Recommendation[]
}

const props = defineProps<Props>()

const emit = defineEmits<{
  'item-click': [itemId: string]
  'refresh': []
}>()

// 瀑布流列数
const columns = ref(4)

// 处理响应式布局
function updateColumns() {
  const width = window.innerWidth
  if (width < 640) columns.value = 1
  else if (width < 1024) columns.value = 2
  else if (width < 1440) columns.value = 3
  else columns.value = 4
}

// 监听窗口大小变化
if (typeof window !== 'undefined') {
  window.addEventListener('resize', updateColumns)
  updateColumns()
}
</script>

<template>
  <div class="recommend-list">
    <!-- 刷新按钮 -->
    <div class="list-header">
      <h2 class="list-title">为你推荐</h2>
      <button class="refresh-btn" @click="emit('refresh')">
        <span class="refresh-icon">↻</span>
        换一批
      </button>
    </div>

    <!-- 卡片网格 -->
    <div class="card-grid" :style="{ '--columns': columns }">
      <ItemCard
        v-for="rec in recommendations"
        :key="rec.item_id"
        :item="rec.item!"
        :score="rec.score"
        :reason="rec.reason"
        @click="emit('item-click', rec.item_id)"
      />
    </div>

    <!-- 空状态 -->
    <div v-if="recommendations.length === 0" class="empty-state">
      <p>暂无推荐内容</p>
      <button @click="emit('refresh')">刷新试试</button>
    </div>
  </div>
</template>

<style scoped>
.recommend-list {
  width: 100%;
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.list-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: #fff;
}

.refresh-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: rgba(79, 172, 254, 0.2);
  border: 1px solid rgba(79, 172, 254, 0.3);
  border-radius: 2rem;
  color: #4facfe;
  cursor: pointer;
  transition: all 0.3s ease;
}

.refresh-btn:hover {
  background: rgba(79, 172, 254, 0.3);
  transform: scale(1.05);
}

.refresh-icon {
  font-size: 1.2rem;
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(var(--columns), 1fr);
  gap: 1.5rem;
}

.empty-state {
  text-align: center;
  padding: 4rem;
  color: #8892b0;
}
</style>
```

---

## 3. 物品卡片 (ItemCard.vue)

```vue
<script setup lang="ts">
import { computed } from 'vue'
import type { Item } from '@shared/types'

interface Props {
  item: Item
  score?: number
  reason?: string
}

const props = defineProps<Props>()

// 类型颜色映射
const typeColors: Record<string, string> = {
  movie: '#e50914',
  product: '#ff9900',
  article: '#1da1f2',
  video: '#ff0050',
}

const typeColor = computed(() => typeColors[props.item.type] || '#4facfe')

// 格式化分数
const formattedScore = computed(() => {
  if (!props.score) return ''
  return `${Math.round(props.score * 100)}% 匹配`
})
</script>

<template>
  <article class="item-card" @click="$emit('click')">
    <!-- 封面图 -->
    <div class="card-cover">
      <div 
        class="cover-placeholder"
        :style="{ background: `linear-gradient(135deg, ${typeColor} 0%, ${typeColor}80 100%)` }"
      >
        <span class="type-icon">{{ item.type[0].toUpperCase() }}</span>
      </div>
      
      <!-- 类型标签 -->
      <span class="type-tag" :style="{ backgroundColor: typeColor }">
        {{ item.type }}
      </span>
      
      <!-- 匹配度 -->
      <span v-if="score" class="match-score">
        {{ formattedScore }}
      </span>
    </div>

    <!-- 卡片内容 -->
    <div class="card-content">
      <h3 class="card-title">{{ item.title }}</h3>
      
      <p class="card-description">
        {{ item.description?.slice(0, 100) }}{{ item.description?.length > 100 ? '...' : '' }}
      </p>
      
      <!-- 标签 -->
      <div class="card-tags">
        <span 
          v-for="tag in item.tags?.slice(0, 3)" 
          :key="tag"
          class="tag"
        >
          {{ tag }}
        </span>
      </div>
      
      <!-- 推荐理由 -->
      <p v-if="reason" class="card-reason">
        💡 {{ reason }}
      </p>
    </div>
  </article>
</template>

<style scoped>
.item-card {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 1rem;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.3s ease;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.item-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
  border-color: rgba(79, 172, 254, 0.3);
}

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
}

.type-icon {
  font-size: 3rem;
  font-weight: 700;
  color: rgba(255, 255, 255, 0.3);
}

.type-tag {
  position: absolute;
  top: 0.75rem;
  left: 0.75rem;
  padding: 0.25rem 0.75rem;
  font-size: 0.75rem;
  font-weight: 600;
  color: white;
  border-radius: 1rem;
  text-transform: uppercase;
}

.match-score {
  position: absolute;
  bottom: 0.75rem;
  right: 0.75rem;
  padding: 0.25rem 0.75rem;
  font-size: 0.75rem;
  font-weight: 600;
  background: rgba(0, 0, 0, 0.7);
  color: #4facfe;
  border-radius: 1rem;
}

.card-content {
  padding: 1.25rem;
}

.card-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: #fff;
  margin-bottom: 0.5rem;
  line-height: 1.4;
}

.card-description {
  font-size: 0.875rem;
  color: #8892b0;
  line-height: 1.6;
  margin-bottom: 0.75rem;
}

.card-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.tag {
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
  background: rgba(79, 172, 254, 0.1);
  color: #4facfe;
  border-radius: 0.25rem;
}

.card-reason {
  font-size: 0.8rem;
  color: #64ffda;
  font-style: italic;
}
</style>
```

---

## 4. 分类标签页 (CategoryTabs.vue)

```vue
<script setup lang="ts">
interface Category {
  key: string
  label: string
}

interface Props {
  categories: Category[]
  active: string
}

const props = defineProps<Props>()
const emit = defineEmits<{
  'update:active': [key: string]
}>()
</script>

<template>
  <nav class="category-tabs">
    <button
      v-for="cat in categories"
      :key="cat.key"
      class="tab-btn"
      :class="{ active: active === cat.key }"
      @click="emit('update:active', cat.key)"
    >
      {{ cat.label }}
    </button>
  </nav>
</template>

<style scoped>
.category-tabs {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  padding: 1rem 2rem;
  background: rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(10px);
  position: sticky;
  top: 0;
  z-index: 100;
}

.tab-btn {
  padding: 0.75rem 1.5rem;
  font-size: 0.95rem;
  font-weight: 500;
  color: #8892b0;
  background: transparent;
  border: none;
  border-radius: 2rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.tab-btn:hover {
  color: #fff;
  background: rgba(255, 255, 255, 0.1);
}

.tab-btn.active {
  color: #fff;
  background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
}
</style>
```

---

## 设计要求

### 🎨 视觉风格

1. **暗色主题** - 深蓝色渐变背景 (#1a1a2e → #16213e)
2. **霓虹色彩** - 主色调青蓝渐变 (#4facfe → #00f2fe)
3. **玻璃拟态** - 半透明卡片 + 模糊效果
4. **流畅动画** - hover 效果、过渡动画

### 📱 响应式

- 桌面: 4 列卡片
- 平板: 2-3 列
- 手机: 1 列

### ⚡ 性能

- 虚拟滚动（大量数据）
- 图片懒加载
- 骨架屏加载

---

## 注意事项

1. 所有类型从 `@shared/types` 导入
2. 使用 Composition API + `<script setup>`
3. 遵循 Vue 3 最佳实践
4. 组件需支持暗色主题

## 输出要求

请输出完整的可运行代码，包含：
1. 所有 Vue 组件
2. TypeScript 类型正确
3. 完整的样式
4. 响应式布局

