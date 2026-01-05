<script setup lang="ts">
/**
 * ItemDetail.vue - 物品详情页
 * 
 * 功能：
 * - 物品详细信息展示
 * - 物品统计数据展示
 * - 相似物品推荐
 * - 用户交互（喜欢、分享等）
 * - 浏览行为记录
 * 
 * Person B 开发
 */
import { ref, onMounted, computed, watch } from 'vue'
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

// =========================================================================
// 状态
// =========================================================================

/** 物品详情 */
const item = ref<Item | null>(null)

/** 物品统计数据 */
const stats = ref<ItemStats | null>(null)

/** 相似物品列表 */
const similarItems = ref<SimilarItem[]>([])

/** 是否正在加载 */
const isLoading = ref(true)

/** 是否已点赞 */
const isLiked = ref(false)

/** 加载错误 */
const loadError = ref<string | null>(null)

/** 是否显示分享面板 */
const showSharePanel = ref(false)

// =========================================================================
// 计算属性
// =========================================================================

/** 当前物品 ID */
const itemId = computed(() => route.params.id as string)

/** 封面渐变色 - 根据物品类型生成 */
const coverGradient = computed(() => {
  if (!item.value) return 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)'
  
  const gradients: Record<string, string> = {
    movie: 'linear-gradient(135deg, #dc2626 0%, #9333ea 100%)',
    product: 'linear-gradient(135deg, #059669 0%, #0891b2 100%)',
    article: 'linear-gradient(135deg, #ea580c 0%, #ca8a04 100%)',
    video: 'linear-gradient(135deg, #db2777 0%, #9333ea 100%)',
  }
  
  return gradients[item.value.type] || gradients.movie
})

/** 类型显示名 */
const typeDisplayName = computed(() => {
  if (!item.value) return ''
  
  const names: Record<string, string> = {
    movie: '电影',
    product: '商品',
    article: '文章',
    video: '视频',
  }
  
  return names[item.value.type] || item.value.type
})

// =========================================================================
// 方法
// =========================================================================

/**
 * 加载物品数据
 */
async function loadItemData() {
  isLoading.value = true
  loadError.value = null
  
  try {
    // 并行加载所有数据
    const [itemData, statsData, similar] = await Promise.all([
      itemStore.getItem(itemId.value),
      itemStore.getItemStats(itemId.value).catch(() => null),
      itemStore.getSimilarItems(itemId.value, 12).catch(() => []),
    ])
    
    item.value = itemData
    stats.value = statsData
    similarItems.value = similar
    
    // 检查是否已喜欢
    isLiked.value = recommendStore.isItemLiked(itemId.value)
    
    // 记录浏览行为
    await recommendStore.recordBehavior({
      item_id: itemId.value,
      action: 'view',
    })
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : '加载物品详情失败'
    console.error('Failed to load item:', error)
  } finally {
    isLoading.value = false
  }
}

/**
 * 处理点赞/取消点赞
 */
async function handleLike() {
  const newState = await recommendStore.toggleLike(itemId.value)
  isLiked.value = newState
}

/**
 * 处理分享
 */
async function handleShare() {
  showSharePanel.value = true
  
  await recommendStore.recordBehavior({
    item_id: itemId.value,
    action: 'share',
  })
}

/**
 * 关闭分享面板
 */
function closeSharePanel() {
  showSharePanel.value = false
}

/**
 * 复制分享链接
 */
async function copyShareLink() {
  const link = window.location.href
  
  try {
    await navigator.clipboard.writeText(link)
    // 可以添加提示
  } catch (error) {
    console.error('Failed to copy link:', error)
  }
  
  closeSharePanel()
}

/**
 * 点击相似物品
 * @param id 物品ID
 */
function handleSimilarClick(id: string) {
  router.push(`/item/${id}`)
}

/**
 * 返回上一页
 */
function goBack() {
  if (window.history.length > 1) {
    router.back()
  } else {
    router.push('/')
  }
}

/**
 * 返回首页
 */
function goHome() {
  router.push('/')
}

// =========================================================================
// 监听器
// =========================================================================

// 监听路由参数变化，重新加载数据
watch(() => route.params.id, (newId) => {
  if (newId) {
    loadItemData()
  }
})

// =========================================================================
// 生命周期
// =========================================================================

onMounted(() => {
  loadItemData()
})
</script>

<template>
  <div class="detail-page">
    <!-- 顶部导航栏 -->
    <nav class="top-nav">
      <button class="nav-btn back-btn" @click="goBack" data-testid="back-button">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="m15 18-6-6 6-6"/>
        </svg>
        <span>返回</span>
      </button>
      
      <button class="nav-btn home-btn" @click="goHome">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
          <polyline points="9 22 9 12 15 12 15 22"/>
        </svg>
      </button>
    </nav>

    <!-- 加载状态 - 骨架屏 -->
    <div v-if="isLoading" class="loading-container" data-testid="loading-skeleton">
      <div class="skeleton-hero">
        <div class="skeleton-cover pulse"></div>
        <div class="skeleton-info">
          <div class="skeleton-title pulse"></div>
          <div class="skeleton-meta pulse"></div>
          <div class="skeleton-desc pulse"></div>
          <div class="skeleton-desc pulse" style="width: 60%"></div>
        </div>
      </div>
      <div class="skeleton-section">
        <div class="skeleton-section-title pulse"></div>
        <div class="skeleton-cards">
          <div v-for="i in 4" :key="i" class="skeleton-card pulse"></div>
        </div>
      </div>
    </div>

    <!-- 错误状态 -->
    <div v-else-if="loadError" class="error-container" data-testid="error-state">
      <div class="error-content">
        <div class="error-icon">😞</div>
        <h2>加载失败</h2>
        <p>{{ loadError }}</p>
        <div class="error-actions">
          <button class="retry-btn" @click="loadItemData">重新加载</button>
          <button class="home-link-btn" @click="goHome">返回首页</button>
        </div>
      </div>
    </div>

    <!-- 详情内容 -->
    <template v-else-if="item">
      <!-- Hero 区域 -->
      <section class="hero-section">
        <!-- 背景模糊层 -->
        <div class="hero-backdrop" :style="{ background: coverGradient }"></div>
        
        <div class="hero-content">
          <!-- 封面 -->
          <div class="cover-container">
            <div 
              class="cover-image"
              :style="{ background: coverGradient }"
              data-testid="item-cover"
            >
              <span class="type-badge">{{ typeDisplayName }}</span>
              <div class="cover-overlay">
                <span class="cover-title">{{ item.title }}</span>
              </div>
            </div>
          </div>

          <!-- 信息区 -->
          <div class="info-container">
            <ItemInfo 
              :item="item" 
              :stats="stats"
              data-testid="item-info"
            />
            
            <!-- 操作按钮 -->
            <ActionButtons
              :is-liked="isLiked"
              @like="handleLike"
              @share="handleShare"
              data-testid="action-buttons"
            />
          </div>
        </div>
      </section>

      <!-- 相似推荐区域 -->
      <section v-if="similarItems.length > 0" class="similar-section">
        <h2 class="section-title">
          <span class="title-icon">✨</span>
          相似推荐
        </h2>
        <SimilarItems
          :items="similarItems"
          @item-click="handleSimilarClick"
          data-testid="similar-items"
        />
      </section>

      <!-- 分享面板 -->
      <Teleport to="body">
        <Transition name="fade">
          <div v-if="showSharePanel" class="share-overlay" @click.self="closeSharePanel">
            <div class="share-panel">
              <h3>分享</h3>
              <p class="share-title">{{ item.title }}</p>
              
              <div class="share-options">
                <button class="share-option" @click="copyShareLink">
                  <span class="share-option-icon">🔗</span>
                  <span>复制链接</span>
                </button>
              </div>
              
              <button class="close-share-btn" @click="closeSharePanel">取消</button>
            </div>
          </div>
        </Transition>
      </Teleport>
    </template>

    <!-- 404 状态 -->
    <div v-else class="not-found-container" data-testid="not-found">
      <div class="not-found-content">
        <div class="not-found-icon">🔍</div>
        <h2>物品不存在</h2>
        <p>您访问的物品可能已被删除或不存在</p>
        <button class="home-link-btn" @click="goHome">返回首页</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.detail-page {
  min-height: 100vh;
  background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
  color: #e2e8f0;
}

/* 顶部导航 */
.top-nav {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  display: flex;
  justify-content: space-between;
  padding: 1rem 1.5rem;
  z-index: 100;
  background: linear-gradient(180deg, rgba(15, 15, 35, 0.9) 0%, transparent 100%);
}

.nav-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 1rem;
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 0.75rem;
  color: #f1f5f9;
  cursor: pointer;
  transition: all 0.3s;
  backdrop-filter: blur(10px);
  font-family: 'Nunito', 'PingFang SC', sans-serif;
}

.nav-btn:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-1px);
}

/* 骨架屏加载 */
.loading-container {
  padding: 5rem 2rem 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.skeleton-hero {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 2rem;
  margin-bottom: 3rem;
}

@media (max-width: 900px) {
  .skeleton-hero {
    grid-template-columns: 1fr;
  }
}

.skeleton-cover {
  aspect-ratio: 2/3;
  border-radius: 1rem;
  background: rgba(255, 255, 255, 0.1);
}

.skeleton-info {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding-top: 1rem;
}

.skeleton-title {
  height: 2.5rem;
  width: 70%;
  border-radius: 0.5rem;
  background: rgba(255, 255, 255, 0.1);
}

.skeleton-meta {
  height: 1.5rem;
  width: 50%;
  border-radius: 0.5rem;
  background: rgba(255, 255, 255, 0.1);
}

.skeleton-desc {
  height: 1rem;
  width: 100%;
  border-radius: 0.5rem;
  background: rgba(255, 255, 255, 0.1);
}

.skeleton-section {
  margin-top: 2rem;
}

.skeleton-section-title {
  height: 1.5rem;
  width: 150px;
  border-radius: 0.5rem;
  background: rgba(255, 255, 255, 0.1);
  margin-bottom: 1rem;
}

.skeleton-cards {
  display: flex;
  gap: 1rem;
  overflow: hidden;
}

.skeleton-card {
  flex: 0 0 180px;
  aspect-ratio: 16/10;
  border-radius: 0.75rem;
  background: rgba(255, 255, 255, 0.1);
}

.pulse {
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* 错误状态 */
.error-container,
.not-found-container {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
}

.error-content,
.not-found-content {
  text-align: center;
  max-width: 400px;
}

.error-icon,
.not-found-icon {
  font-size: 5rem;
  margin-bottom: 1.5rem;
}

.error-content h2,
.not-found-content h2 {
  font-size: 1.75rem;
  margin-bottom: 0.75rem;
  color: #f1f5f9;
}

.error-content p,
.not-found-content p {
  color: #94a3b8;
  margin-bottom: 2rem;
}

.error-actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
}

.retry-btn {
  padding: 0.75rem 1.5rem;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  border: none;
  border-radius: 0.75rem;
  color: #fff;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
}

.retry-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
}

.home-link-btn {
  padding: 0.75rem 1.5rem;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 0.75rem;
  color: #f1f5f9;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
}

.home-link-btn:hover {
  background: rgba(255, 255, 255, 0.15);
}

/* Hero 区域 */
.hero-section {
  position: relative;
  padding: 6rem 2rem 3rem;
  overflow: hidden;
}

.hero-backdrop {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 60%;
  opacity: 0.15;
  filter: blur(100px);
}

.hero-content {
  position: relative;
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 3rem;
  max-width: 1200px;
  margin: 0 auto;
}

@media (max-width: 900px) {
  .hero-content {
    grid-template-columns: 1fr;
    gap: 2rem;
  }
  
  .cover-container {
    max-width: 280px;
    margin: 0 auto;
  }
}

/* 封面 */
.cover-container {
  position: relative;
}

.cover-image {
  aspect-ratio: 2/3;
  border-radius: 1rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
}

.type-badge {
  position: absolute;
  top: 1rem;
  left: 1rem;
  padding: 0.4rem 0.8rem;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(8px);
  border-radius: 0.5rem;
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.05em;
}

.cover-overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 3rem 1rem 1rem;
  background: linear-gradient(transparent, rgba(0, 0, 0, 0.8));
}

.cover-title {
  font-size: 1.1rem;
  font-weight: 600;
  line-height: 1.4;
}

/* 信息区域 */
.info-container {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

/* 相似推荐区域 */
.similar-section {
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
  padding-left: 0.5rem;
  border-left: 4px solid #6366f1;
}

.title-icon {
  font-size: 1.25rem;
}

/* 分享面板 */
.share-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(8px);
  display: flex;
  align-items: flex-end;
  justify-content: center;
  z-index: 1000;
  padding: 1rem;
}

.share-panel {
  background: #1e1e3f;
  border-radius: 1.5rem 1.5rem 0 0;
  padding: 1.5rem;
  width: 100%;
  max-width: 400px;
  max-height: 80vh;
}

.share-panel h3 {
  font-size: 1.25rem;
  text-align: center;
  margin-bottom: 0.5rem;
}

.share-title {
  text-align: center;
  color: #94a3b8;
  font-size: 0.9rem;
  margin-bottom: 1.5rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.share-options {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  justify-content: center;
  margin-bottom: 1.5rem;
}

.share-option {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.05);
  border: none;
  border-radius: 0.75rem;
  color: #e2e8f0;
  cursor: pointer;
  transition: all 0.2s;
  min-width: 80px;
}

.share-option:hover {
  background: rgba(255, 255, 255, 0.1);
}

.share-option-icon {
  font-size: 1.5rem;
}

.close-share-btn {
  width: 100%;
  padding: 0.75rem;
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 0.75rem;
  color: #94a3b8;
  cursor: pointer;
  transition: all 0.2s;
}

.close-share-btn:hover {
  background: rgba(255, 255, 255, 0.15);
  color: #f1f5f9;
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.fade-enter-active .share-panel,
.fade-leave-active .share-panel {
  transition: transform 0.3s ease;
}

.fade-enter-from .share-panel,
.fade-leave-to .share-panel {
  transform: translateY(100%);
}
</style>

