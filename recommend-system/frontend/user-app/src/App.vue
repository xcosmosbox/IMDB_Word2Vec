<!--
  根组件
  
  应用的根组件，负责整体布局和全局状态
  
  @author Person F
-->
<script setup lang="ts">
import { onMounted, provide, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useUserStore, useRecommendStore } from '@/stores'
import { useApi } from '@shared/api'

// =============================================================================
// 依赖获取
// =============================================================================

const router = useRouter()
const route = useRoute()
const userStore = useUserStore()
const recommendStore = useRecommendStore()
const api = useApi()

// =============================================================================
// 初始化
// =============================================================================

onMounted(async () => {
  // 初始化用户状态
  await userStore.init()
  
  // 如果用户已登录，设置推荐 store 的用户 ID
  if (userStore.userId) {
    recommendStore.setCurrentUserId(userStore.userId)
  }
})

// =============================================================================
// 监听用户状态变化
// =============================================================================

watch(
  () => userStore.userId,
  (newUserId) => {
    if (newUserId) {
      recommendStore.setCurrentUserId(newUserId)
    }
  }
)

// =============================================================================
// 页面过渡动画
// =============================================================================

/**
 * 获取过渡动画名称
 */
function getTransitionName(): string {
  // 可以根据路由配置或导航方向返回不同的动画
  return route.meta.transition as string || 'fade'
}
</script>

<template>
  <div id="app-container" class="app">
    <!-- 路由视图 -->
    <router-view v-slot="{ Component, route: currentRoute }">
      <Transition :name="getTransitionName()" mode="out-in">
        <KeepAlive :include="['Home', 'Search', 'Category']">
          <component :is="Component" :key="currentRoute.path" />
        </KeepAlive>
      </Transition>
    </router-view>
  </div>
</template>

<style>
/* 全局重置和基础样式 */
*,
*::before,
*::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html {
  font-size: 16px;
  -webkit-text-size-adjust: 100%;
}

body {
  min-height: 100vh;
  font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  color: #e2e8f0;
  background: #0f0f23;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* 链接样式 */
a {
  color: #4facfe;
  text-decoration: none;
  transition: color 0.2s ease;
}

a:hover {
  color: #00f2fe;
}

/* 按钮重置 */
button {
  font-family: inherit;
  cursor: pointer;
}

/* 输入框重置 */
input,
textarea,
select {
  font-family: inherit;
  font-size: inherit;
}

/* 图片响应式 */
img {
  max-width: 100%;
  height: auto;
  vertical-align: middle;
}

/* 列表重置 */
ul,
ol {
  list-style: none;
}

/* 滚动条样式 */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.05);
}

::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.15);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.25);
}

/* 选中文本颜色 */
::selection {
  background: rgba(79, 172, 254, 0.3);
  color: #fff;
}
</style>

<style scoped>
/* 应用容器 */
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* 过渡动画 - 淡入淡出 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* 过渡动画 - 滑动 */
.slide-enter-active,
.slide-leave-active {
  transition: all 0.3s ease;
}

.slide-enter-from {
  opacity: 0;
  transform: translateX(30px);
}

.slide-leave-to {
  opacity: 0;
  transform: translateX(-30px);
}

/* 过渡动画 - 缩放 */
.scale-enter-active,
.scale-leave-active {
  transition: all 0.3s ease;
}

.scale-enter-from,
.scale-leave-to {
  opacity: 0;
  transform: scale(0.95);
}
</style>

