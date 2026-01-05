<script setup lang="ts">
/**
 * Home.vue - 首页视图
 * 
 * 生成式推荐系统的用户端首页，功能包括：
 * 1. 个性化推荐展示
 * 2. 分类筛选
 * 3. 刷新/换一批
 * 4. 无限滚动加载
 * 5. 用户行为记录
 * 
 * 遵循接口驱动开发原则，通过依赖注入使用 API 服务
 */

import { ref, computed, onMounted, inject, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useRecommendStore } from '@/stores/recommend'
import type { IApiProvider } from '@shared/api/interfaces'
import RecommendList from '@/components/RecommendList.vue'
import CategoryTabs from '@/components/CategoryTabs.vue'
import LoadingSpinner from '@/components/LoadingSpinner.vue'

// ============================================================
// 依赖注入
// ============================================================

/** 注入 API Provider */
const apiProvider = inject<IApiProvider>('api')

/** Vue Router */
const router = useRouter()

/** 推荐 Store */
const recommendStore = useRecommendStore()

// 初始化 API Provider
if (apiProvider) {
  recommendStore.setApiProvider(apiProvider)
}

// ============================================================
// 响应式状态
// ============================================================

/** 当前激活的分类 */
const activeCategory = ref('all')

/** 初始加载状态 */
const isInitialLoading = ref(true)

/** 是否有更多数据 */
const hasMore = ref(true)

/** 当前用户 ID (实际应从用户 Store 获取) */
const currentUserId = ref('user_demo')

// ============================================================
// 分类配置
// ============================================================

/** 分类列表 */
const categories = [
  { key: 'all', label: '为你推荐', icon: '✨' },
  { key: 'movie', label: '电影', icon: '🎬' },
  { key: 'product', label: '商品', icon: '🛒' },
  { key: 'article', label: '文章', icon: '📄' },
  { key: 'video', label: '视频', icon: '🎥' },
]

// ============================================================
// 计算属性
// ============================================================

/** 根据分类过滤推荐 */
const filteredRecommendations = computed(() => {
  if (activeCategory.value === 'all') {
    return recommendStore.recommendations
  }
  return recommendStore.recommendations.filter(
    rec => rec.item?.type === activeCategory.value
  )
})

/** 当前分类的名称 */
const currentCategoryLabel = computed(() => {
  const cat = categories.find(c => c.key === activeCategory.value)
  return cat?.label || '推荐'
})

// ============================================================
// 方法
// ============================================================

/**
 * 加载推荐数据
 */
async function loadRecommendations() {
  try {
    await recommendStore.fetchRecommendations({
      user_id: currentUserId.value,
      limit: 50,
      scene: 'home',
    })
    hasMore.value = true
  } catch (error) {
    console.error('加载推荐失败:', error)
    hasMore.value = false
  }
}

/**
 * 刷新推荐 (换一批)
 */
async function handleRefresh() {
  try {
    await recommendStore.refreshRecommendations(currentUserId.value, 50)
  } catch (error) {
    console.error('刷新推荐失败:', error)
  }
}

/**
 * 加载更多
 */
async function handleLoadMore() {
  if (recommendStore.isLoading || !hasMore.value) return
  
  try {
    const beforeCount = recommendStore.recommendations.length
    await recommendStore.loadMoreRecommendations(currentUserId.value, 20)
    const afterCount = recommendStore.recommendations.length
    
    // 如果没有新数据，说明没有更多了
    if (afterCount === beforeCount) {
      hasMore.value = false
    }
  } catch (error) {
    console.error('加载更多失败:', error)
    hasMore.value = false
  }
}

/**
 * 处理物品点击
 */
function handleItemClick(itemId: string) {
  // 记录点击行为
  recommendStore.recordBehavior({
    user_id: currentUserId.value,
    item_id: itemId,
    action: 'click',
    context: {
      scene: 'home',
      category: activeCategory.value,
    },
  })
  
  // 导航到详情页
  router.push(`/item/${itemId}`)
}

/**
 * 处理物品喜欢
 */
function handleItemLike(itemId: string) {
  recommendStore.recordBehavior({
    user_id: currentUserId.value,
    item_id: itemId,
    action: 'like',
    context: {
      scene: 'home',
    },
  })
  
  // 可以添加 UI 反馈
  console.log('已添加到喜欢:', itemId)
}

/**
 * 处理物品分享
 */
function handleItemShare(itemId: string) {
  recommendStore.recordBehavior({
    user_id: currentUserId.value,
    item_id: itemId,
    action: 'share',
    context: {
      scene: 'home',
    },
  })
  
  // 实现分享逻辑
  console.log('分享:', itemId)
}

// ============================================================
// 分类切换处理
// ============================================================

watch(activeCategory, (newCategory) => {
  // 记录分类切换行为 (用于分析)
  console.log('切换分类:', newCategory)
})

// ============================================================
// 生命周期
// ============================================================

onMounted(async () => {
  isInitialLoading.value = true
  
  try {
    await loadRecommendations()
  } finally {
    isInitialLoading.value = false
  }
})
</script>

<template>
  <div class="home-page">
    <!-- 顶部英雄区域 -->
    <header class="hero-section">
      <div class="hero-content">
        <h1 class="hero-title">
          <span class="hero-title-text">发现你的下一个最爱</span>
          <span class="hero-title-highlight"></span>
        </h1>
        <p class="hero-subtitle">
          基于 <span class="highlight-text">AI 生成式推荐</span> 的个性化内容发现
        </p>
        
        <!-- 推荐策略标签 -->
        <div v-if="recommendStore.strategy" class="strategy-badge">
          <span class="strategy-icon">🧠</span>
          <span class="strategy-text">{{ recommendStore.strategy }}</span>
        </div>
      </div>
      
      <!-- 装饰性背景元素 -->
      <div class="hero-decoration">
        <div class="decoration-orb decoration-orb--1"></div>
        <div class="decoration-orb decoration-orb--2"></div>
        <div class="decoration-orb decoration-orb--3"></div>
      </div>
    </header>

    <!-- 分类标签导航 -->
    <CategoryTabs
      :categories="categories"
      v-model:active="activeCategory"
      :sticky="true"
    />

    <!-- 主内容区域 -->
    <main class="content-section">
      <!-- 初始加载状态 -->
      <div v-if="isInitialLoading" class="initial-loading">
        <LoadingSpinner 
          size="large" 
          :show-text="true" 
          text="正在为你加载个性化推荐..."
        />
      </div>
      
      <!-- 错误状态 -->
      <div v-else-if="recommendStore.error" class="error-state">
        <div class="error-icon">😵</div>
        <h3 class="error-title">加载失败</h3>
        <p class="error-message">{{ recommendStore.error }}</p>
        <button class="retry-btn" @click="loadRecommendations">
          重新加载
        </button>
      </div>
      
      <!-- 推荐列表 -->
      <RecommendList
        v-else
        :recommendations="filteredRecommendations"
        :loading="recommendStore.isLoading"
        :has-more="hasMore"
        :title="currentCategoryLabel"
        :show-refresh="true"
        :empty-text="`暂无${currentCategoryLabel}内容`"
        @item-click="handleItemClick"
        @item-like="handleItemLike"
        @item-share="handleItemShare"
        @refresh="handleRefresh"
        @load-more="handleLoadMore"
      />
    </main>

    <!-- 底部信息 -->
    <footer class="page-footer">
      <p class="footer-text">
        由 <span class="footer-highlight">生成式推荐系统</span> 驱动
      </p>
      <p class="footer-stats" v-if="recommendStore.totalCount > 0">
        已为你推荐 {{ recommendStore.totalCount }} 个内容
      </p>
    </footer>
  </div>
</template>

<style scoped>
/* ============================================================
   页面容器
   ============================================================ */

.home-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
  color: #e6e6e6;
  font-family: 'Inter', 'Noto Sans SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ============================================================
   英雄区域
   ============================================================ */

.hero-section {
  position: relative;
  padding: 5rem 2rem 3rem;
  text-align: center;
  overflow: hidden;
  background: 
    radial-gradient(ellipse at top, rgba(79, 172, 254, 0.08) 0%, transparent 50%),
    radial-gradient(ellipse at bottom right, rgba(0, 242, 254, 0.05) 0%, transparent 50%);
}

.hero-content {
  position: relative;
  z-index: 2;
  max-width: 800px;
  margin: 0 auto;
}

.hero-title {
  position: relative;
  display: inline-block;
  margin: 0 0 1rem;
}

.hero-title-text {
  font-size: clamp(2rem, 5vw, 3rem);
  font-weight: 800;
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 50%, #4facfe 100%);
  background-size: 200% 200%;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: gradient-shift 5s ease-in-out infinite;
  letter-spacing: -0.02em;
}

@keyframes gradient-shift {
  0%, 100% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
}

.hero-subtitle {
  font-size: clamp(1rem, 2.5vw, 1.3rem);
  color: #8892b0;
  margin: 0;
  font-weight: 400;
  line-height: 1.6;
}

.highlight-text {
  color: #64ffda;
  font-weight: 500;
}

/* 推荐策略标签 */
.strategy-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 1.5rem;
  padding: 0.5rem 1rem;
  background: rgba(100, 255, 218, 0.1);
  border: 1px solid rgba(100, 255, 218, 0.2);
  border-radius: 2rem;
  font-size: 0.85rem;
  color: #64ffda;
}

.strategy-icon {
  font-size: 1rem;
}

/* 装饰性背景元素 */
.hero-decoration {
  position: absolute;
  inset: 0;
  overflow: hidden;
  pointer-events: none;
}

.decoration-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(60px);
  opacity: 0.4;
}

.decoration-orb--1 {
  width: 300px;
  height: 300px;
  background: radial-gradient(circle, rgba(79, 172, 254, 0.4) 0%, transparent 70%);
  top: -100px;
  left: -50px;
  animation: float-1 8s ease-in-out infinite;
}

.decoration-orb--2 {
  width: 200px;
  height: 200px;
  background: radial-gradient(circle, rgba(0, 242, 254, 0.3) 0%, transparent 70%);
  top: 50%;
  right: -50px;
  animation: float-2 10s ease-in-out infinite;
}

.decoration-orb--3 {
  width: 150px;
  height: 150px;
  background: radial-gradient(circle, rgba(100, 255, 218, 0.3) 0%, transparent 70%);
  bottom: -50px;
  left: 30%;
  animation: float-3 12s ease-in-out infinite;
}

@keyframes float-1 {
  0%, 100% {
    transform: translate(0, 0) scale(1);
  }
  50% {
    transform: translate(20px, 30px) scale(1.1);
  }
}

@keyframes float-2 {
  0%, 100% {
    transform: translate(0, 0) scale(1);
  }
  50% {
    transform: translate(-30px, -20px) scale(0.9);
  }
}

@keyframes float-3 {
  0%, 100% {
    transform: translate(0, 0) scale(1);
  }
  50% {
    transform: translate(40px, -10px) scale(1.05);
  }
}

/* ============================================================
   主内容区域
   ============================================================ */

.content-section {
  padding: 2rem;
  max-width: 1600px;
  margin: 0 auto;
  min-height: 60vh;
}

/* 初始加载状态 */
.initial-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 400px;
}

/* 错误状态 */
.error-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem 2rem;
  text-align: center;
}

.error-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.error-title {
  font-size: 1.5rem;
  color: #ff6b6b;
  margin: 0 0 0.5rem;
}

.error-message {
  color: #8892b0;
  margin: 0 0 1.5rem;
}

.retry-btn {
  padding: 0.75rem 2rem;
  background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
  border: none;
  border-radius: 2rem;
  color: white;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.retry-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 8px 24px rgba(79, 172, 254, 0.4);
}

/* ============================================================
   页脚
   ============================================================ */

.page-footer {
  padding: 3rem 2rem;
  text-align: center;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
  background: rgba(0, 0, 0, 0.2);
}

.footer-text {
  font-size: 0.9rem;
  color: #5a6378;
  margin: 0 0 0.5rem;
}

.footer-highlight {
  color: #4facfe;
}

.footer-stats {
  font-size: 0.85rem;
  color: #8892b0;
  margin: 0;
}

/* ============================================================
   响应式适配
   ============================================================ */

@media (max-width: 640px) {
  .hero-section {
    padding: 3rem 1.5rem 2rem;
  }
  
  .content-section {
    padding: 1.5rem 1rem;
  }
  
  .hero-title-text {
    line-height: 1.2;
  }
}

@media (min-width: 641px) and (max-width: 1024px) {
  .hero-section {
    padding: 4rem 2rem 2.5rem;
  }
}

/* ============================================================
   暗色主题滚动条
   ============================================================ */

::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.2);
}

::-webkit-scrollbar-thumb {
  background: rgba(79, 172, 254, 0.3);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(79, 172, 254, 0.5);
}
</style>

