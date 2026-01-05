/**
 * History - 历史记录页面
 * 
 * 展示用户的浏览历史记录，支持按类型过滤。
 * 
 * @view
 * @author Person C
 */
<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useUserStore } from '@/stores/user'
import HistoryList from '@/components/HistoryList.vue'

// =============================================================================
// 依赖注入
// =============================================================================

const router = useRouter()
const userStore = useUserStore()

// =============================================================================
// 状态
// =============================================================================

const isLoading = ref(true)
const filterAction = ref('all')

// 过滤选项
const filterOptions = [
  { value: 'all', label: '全部', icon: '📋' },
  { value: 'view', label: '浏览', icon: '👁️' },
  { value: 'click', label: '点击', icon: '👆' },
  { value: 'like', label: '喜欢', icon: '❤️' },
  { value: 'buy', label: '购买', icon: '🛒' },
  { value: 'share', label: '分享', icon: '🔗' },
]

// =============================================================================
// 计算属性
// =============================================================================

/** 过滤后的行为数据 */
const filteredBehaviors = computed(() => {
  if (filterAction.value === 'all') {
    return userStore.behaviors
  }
  return userStore.behaviors.filter((b) => b.action === filterAction.value)
})

/** 各类型统计 */
const actionStats = computed(() => {
  const stats: Record<string, number> = { all: userStore.behaviors.length }
  
  userStore.behaviors.forEach((b) => {
    stats[b.action] = (stats[b.action] || 0) + 1
  })
  
  return stats
})

/** 是否有数据 */
const hasData = computed(() => userStore.behaviors.length > 0)

// =============================================================================
// 事件处理
// =============================================================================

/**
 * 处理过滤切换
 */
function handleFilterChange(action: string) {
  filterAction.value = action
}

/**
 * 处理物品点击
 */
function handleItemClick(itemId: string) {
  router.push(`/item/${itemId}`)
}

/**
 * 刷新数据
 */
async function refreshData() {
  isLoading.value = true
  try {
    await userStore.fetchBehaviors(100)
  } finally {
    isLoading.value = false
  }
}

/**
 * 返回上一页
 */
function goBack() {
  router.back()
}

// =============================================================================
// 生命周期
// =============================================================================

onMounted(async () => {
  // 检查登录状态
  if (!userStore.isLoggedIn) {
    router.push('/login?redirect=/history')
    return
  }
  
  try {
    await userStore.fetchBehaviors(100)
  } finally {
    isLoading.value = false
  }
})
</script>

<template>
  <div class="history-page">
    <!-- 页面头部 -->
    <header class="page-header">
      <button class="back-btn" @click="goBack">
        <span class="back-icon">←</span>
      </button>
      <h1 class="page-title">浏览历史</h1>
      <button class="refresh-btn" :disabled="isLoading" @click="refreshData">
        <span class="refresh-icon" :class="{ spinning: isLoading }">🔄</span>
      </button>
    </header>

    <!-- 过滤器 -->
    <div class="filter-section">
      <div class="filter-bar">
        <button
          v-for="option in filterOptions"
          :key="option.value"
          class="filter-btn"
          :class="{ active: filterAction === option.value }"
          @click="handleFilterChange(option.value)"
        >
          <span class="filter-icon">{{ option.icon }}</span>
          <span class="filter-label">{{ option.label }}</span>
          <span v-if="actionStats[option.value]" class="filter-count">
            {{ actionStats[option.value] }}
          </span>
        </button>
      </div>
    </div>

    <!-- 主要内容 -->
    <main class="history-content">
      <!-- 加载状态 -->
      <div v-if="isLoading" class="loading-state">
        <div class="loading-spinner"></div>
        <p>加载历史记录...</p>
      </div>

      <!-- 历史列表 -->
      <template v-else>
        <div v-if="hasData" class="list-header">
          <span class="result-count">
            共 {{ filteredBehaviors.length }} 条记录
          </span>
        </div>

        <HistoryList
          :behaviors="filteredBehaviors"
          :group-by-date="true"
          :show-timestamp="true"
          @item-click="handleItemClick"
        />
      </template>
    </main>

    <!-- 底部空间 -->
    <div class="bottom-spacer"></div>
  </div>
</template>

<style scoped>
.history-page {
  min-height: 100vh;
  background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
  color: #eee;
}

/* 页面头部 */
.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.5rem;
  background: rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(10px);
  position: sticky;
  top: 0;
  z-index: 100;
}

.back-btn,
.refresh-btn {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 0.75rem;
  color: #fff;
  font-size: 1.25rem;
  cursor: pointer;
  transition: all 0.2s;
}

.back-btn:hover,
.refresh-btn:hover {
  background: rgba(255, 255, 255, 0.15);
}

.refresh-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.refresh-icon {
  display: inline-block;
  transition: transform 0.3s;
}

.refresh-icon.spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.page-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #fff;
  margin: 0;
}

/* 过滤器 */
.filter-section {
  padding: 1rem 1.5rem;
  background: rgba(0, 0, 0, 0.1);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.filter-bar {
  display: flex;
  gap: 0.5rem;
  overflow-x: auto;
  padding-bottom: 0.25rem;
  scrollbar-width: none;
  -ms-overflow-style: none;
}

.filter-bar::-webkit-scrollbar {
  display: none;
}

.filter-btn {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.5rem 0.875rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 2rem;
  color: #8892b0;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
  flex-shrink: 0;
}

.filter-btn:hover {
  background: rgba(255, 255, 255, 0.1);
  color: #ccd6f6;
}

.filter-btn.active {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  border-color: transparent;
  color: #fff;
}

.filter-icon {
  font-size: 0.9rem;
}

.filter-label {
  font-weight: 500;
}

.filter-count {
  font-size: 0.75rem;
  padding: 0.1rem 0.4rem;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 1rem;
}

.filter-btn.active .filter-count {
  background: rgba(255, 255, 255, 0.2);
}

/* 主要内容 */
.history-content {
  max-width: 800px;
  margin: 0 auto;
  padding: 1.5rem;
}

/* 加载状态 */
.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem;
  color: #8892b0;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(255, 255, 255, 0.1);
  border-top-color: #4facfe;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-bottom: 1rem;
}

/* 列表头部 */
.list-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.result-count {
  font-size: 0.9rem;
  color: #8892b0;
}

/* 底部空间 */
.bottom-spacer {
  height: 4rem;
}

/* 响应式 */
@media (max-width: 480px) {
  .page-header {
    padding: 1rem;
  }

  .filter-section {
    padding: 0.75rem 1rem;
  }

  .history-content {
    padding: 1rem;
  }

  .filter-btn {
    padding: 0.4rem 0.75rem;
    font-size: 0.8rem;
  }
}
</style>

