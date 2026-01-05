/**
 * Profile - 个人资料页面
 * 
 * 展示和编辑用户个人信息、用户画像等。
 * 
 * @view
 * @author Person C
 */
<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useUserStore } from '@/stores/user'
import ProfileCard from '@/components/ProfileCard.vue'
import PreferenceChart from '@/components/PreferenceChart.vue'
import type { UpdateUserRequest } from '@shared/types'

// =============================================================================
// 依赖注入
// =============================================================================

const router = useRouter()
const userStore = useUserStore()

// =============================================================================
// 状态
// =============================================================================

const isEditing = ref(false)
const isSaving = ref(false)

// =============================================================================
// 计算属性
// =============================================================================

/** 活跃时间数据 */
const activeHoursData = computed(() => {
  if (!userStore.profile?.active_hours) return []
  
  const hourLabels = ['0-4', '4-8', '8-12', '12-16', '16-20', '20-24']
  const data = userStore.profile.active_hours
  
  return hourLabels.map((label, index) => {
    const startHour = index * 4
    let count = 0
    for (let h = startHour; h < startHour + 4; h++) {
      count += data[h] || 0
    }
    return { label, count }
  })
})

/** 最后活跃时间格式化 */
const lastActiveFormatted = computed(() => {
  if (!userStore.profile?.last_active) return '未知'
  
  const date = new Date(userStore.profile.last_active)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  
  if (diffMins < 1) return '刚刚'
  if (diffMins < 60) return `${diffMins} 分钟前`
  if (diffMins < 1440) return `${Math.floor(diffMins / 60)} 小时前`
  return date.toLocaleDateString('zh-CN')
})

// =============================================================================
// 事件处理
// =============================================================================

/**
 * 开始编辑
 */
function handleEdit() {
  isEditing.value = true
}

/**
 * 保存编辑
 */
async function handleSave(data: UpdateUserRequest) {
  isSaving.value = true
  
  try {
    await userStore.updateProfile(data)
    isEditing.value = false
  } catch (error: any) {
    // 错误由 store 处理
    console.error('保存失败:', error)
  } finally {
    isSaving.value = false
  }
}

/**
 * 取消编辑
 */
function handleCancel() {
  isEditing.value = false
}

/**
 * 跳转到历史记录
 */
function goToHistory() {
  router.push('/history')
}

/**
 * 退出登录
 */
async function handleLogout() {
  await userStore.logout()
  router.push('/login')
}

// =============================================================================
// 生命周期
// =============================================================================

onMounted(async () => {
  // 检查登录状态
  if (!userStore.isLoggedIn) {
    router.push('/login')
    return
  }
  
  // 获取用户画像
  await userStore.fetchProfile()
})
</script>

<template>
  <div class="profile-page">
    <!-- 页面头部 -->
    <header class="page-header">
      <button class="back-btn" @click="router.back()">
        <span class="back-icon">←</span>
        <span>返回</span>
      </button>
      <h1 class="page-title">个人中心</h1>
      <div class="header-spacer"></div>
    </header>

    <!-- 主要内容 -->
    <main class="profile-content">
      <!-- 加载状态 -->
      <div v-if="userStore.isLoading && !userStore.currentUser" class="loading-state">
        <div class="loading-spinner"></div>
        <p>加载中...</p>
      </div>

      <!-- 用户未登录 -->
      <div v-else-if="!userStore.currentUser" class="login-prompt">
        <p>请先登录</p>
        <router-link to="/login" class="login-btn">去登录</router-link>
      </div>

      <!-- 正常内容 -->
      <template v-else>
        <!-- 错误提示 -->
        <Transition name="fade">
          <div v-if="userStore.error" class="error-alert">
            <span>⚠️</span>
            <span>{{ userStore.error }}</span>
            <button @click="userStore.clearError()">×</button>
          </div>
        </Transition>

        <!-- 个人信息卡片 -->
        <section class="section">
          <ProfileCard
            :user="userStore.currentUser"
            :is-editing="isEditing"
            :loading="isSaving"
            @edit="handleEdit"
            @save="handleSave"
            @cancel="handleCancel"
          />
        </section>

        <!-- 用户画像 -->
        <section v-if="userStore.profile" class="section">
          <h2 class="section-title">我的画像</h2>
          
          <!-- 统计数据 -->
          <div class="stats-grid">
            <div class="stat-card">
              <div class="stat-icon">📊</div>
              <div class="stat-info">
                <div class="stat-value">{{ userStore.profile.total_actions }}</div>
                <div class="stat-label">总互动次数</div>
              </div>
            </div>
            <div class="stat-card">
              <div class="stat-icon">🎯</div>
              <div class="stat-info">
                <div class="stat-value">{{ Object.keys(userStore.profile.preferred_types).length }}</div>
                <div class="stat-label">喜好类型</div>
              </div>
            </div>
            <div class="stat-card">
              <div class="stat-icon">⏰</div>
              <div class="stat-info">
                <div class="stat-value">{{ lastActiveFormatted }}</div>
                <div class="stat-label">最近活跃</div>
              </div>
            </div>
          </div>

          <!-- 偏好图表 -->
          <div class="chart-section">
            <PreferenceChart
              :data="userStore.profile.preferred_types"
              title="内容偏好分布"
            />
          </div>

          <!-- 活跃时段 -->
          <div v-if="activeHoursData.length > 0" class="active-hours">
            <h3 class="subsection-title">活跃时段</h3>
            <div class="hours-chart">
              <div
                v-for="item in activeHoursData"
                :key="item.label"
                class="hour-bar"
              >
                <div
                  class="bar-fill"
                  :style="{
                    height: `${Math.min((item.count / 100) * 100, 100)}%`,
                  }"
                ></div>
                <span class="hour-label">{{ item.label }}</span>
              </div>
            </div>
          </div>
        </section>

        <!-- 账号操作 -->
        <section class="section">
          <h2 class="section-title">更多操作</h2>
          
          <div class="action-list">
            <button class="action-item" @click="goToHistory">
              <span class="action-icon">📜</span>
              <span class="action-text">浏览历史</span>
              <span class="action-arrow">→</span>
            </button>
            
            <button class="action-item">
              <span class="action-icon">⚙️</span>
              <span class="action-text">偏好设置</span>
              <span class="action-arrow">→</span>
            </button>
            
            <button class="action-item">
              <span class="action-icon">🔔</span>
              <span class="action-text">通知设置</span>
              <span class="action-arrow">→</span>
            </button>
            
            <button class="action-item">
              <span class="action-icon">❓</span>
              <span class="action-text">帮助与反馈</span>
              <span class="action-arrow">→</span>
            </button>
            
            <button class="action-item logout" @click="handleLogout">
              <span class="action-icon">🚪</span>
              <span class="action-text">退出登录</span>
              <span class="action-arrow">→</span>
            </button>
          </div>
        </section>

        <!-- 版本信息 -->
        <div class="version-info">
          <p>版本 1.0.0</p>
        </div>
      </template>
    </main>
  </div>
</template>

<style scoped>
.profile-page {
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

.back-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 0.5rem;
  color: #fff;
  font-size: 0.9rem;
  cursor: pointer;
  transition: background 0.2s;
}

.back-btn:hover {
  background: rgba(255, 255, 255, 0.15);
}

.back-icon {
  font-size: 1rem;
}

.page-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #fff;
  margin: 0;
}

.header-spacer {
  width: 80px;
}

/* 主要内容 */
.profile-content {
  max-width: 600px;
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

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* 登录提示 */
.login-prompt {
  text-align: center;
  padding: 4rem;
}

.login-btn {
  display: inline-block;
  margin-top: 1rem;
  padding: 0.75rem 2rem;
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  border-radius: 0.75rem;
  color: #fff;
  text-decoration: none;
  font-weight: 600;
}

/* 错误提示 */
.error-alert {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background: rgba(255, 107, 107, 0.15);
  border: 1px solid rgba(255, 107, 107, 0.3);
  border-radius: 0.75rem;
  margin-bottom: 1.5rem;
  color: #ff6b6b;
}

.error-alert button {
  margin-left: auto;
  background: none;
  border: none;
  color: #ff6b6b;
  font-size: 1.25rem;
  cursor: pointer;
}

/* 区块 */
.section {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 1rem;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.section-title {
  font-size: 1.125rem;
  font-weight: 600;
  color: #fff;
  margin: 0 0 1.25rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.subsection-title {
  font-size: 0.95rem;
  font-weight: 600;
  color: #ccd6f6;
  margin: 1.5rem 0 1rem;
}

/* 统计网格 */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.stat-card {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background: rgba(79, 172, 254, 0.08);
  border-radius: 0.75rem;
}

.stat-icon {
  font-size: 1.5rem;
}

.stat-info {
  flex: 1;
  min-width: 0;
}

.stat-value {
  font-size: 1.25rem;
  font-weight: 700;
  color: #4facfe;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.stat-label {
  font-size: 0.75rem;
  color: #8892b0;
}

/* 图表区域 */
.chart-section {
  margin-top: 1rem;
}

/* 活跃时段 */
.active-hours {
  margin-top: 1.5rem;
}

.hours-chart {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  height: 100px;
  padding: 0.5rem 0;
}

.hour-bar {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  height: 100%;
}

.bar-fill {
  width: 70%;
  max-width: 40px;
  background: linear-gradient(180deg, #4facfe 0%, #00f2fe 100%);
  border-radius: 4px 4px 0 0;
  min-height: 4px;
  transition: height 0.3s ease;
}

.hour-label {
  font-size: 0.7rem;
  color: #8892b0;
}

/* 操作列表 */
.action-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.action-item {
  display: flex;
  align-items: center;
  width: 100%;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.03);
  border: none;
  border-radius: 0.75rem;
  color: #ccd6f6;
  font-size: 0.95rem;
  cursor: pointer;
  transition: all 0.2s ease;
  text-align: left;
}

.action-item:hover {
  background: rgba(255, 255, 255, 0.08);
  transform: translateX(4px);
}

.action-item.logout {
  color: #ff6b6b;
}

.action-icon {
  font-size: 1.25rem;
  margin-right: 1rem;
}

.action-text {
  flex: 1;
}

.action-arrow {
  color: #8892b0;
  transition: color 0.2s;
}

.action-item:hover .action-arrow {
  color: #4facfe;
}

/* 版本信息 */
.version-info {
  text-align: center;
  padding: 2rem;
  color: #8892b0;
  font-size: 0.85rem;
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: all 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

/* 响应式 */
@media (max-width: 480px) {
  .profile-content {
    padding: 1rem;
  }

  .stats-grid {
    grid-template-columns: 1fr;
  }

  .stat-card {
    padding: 0.875rem;
  }

  .page-header {
    padding: 1rem;
  }

  .header-spacer {
    display: none;
  }
}
</style>

