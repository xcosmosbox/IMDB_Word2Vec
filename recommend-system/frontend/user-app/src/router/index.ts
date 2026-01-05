/**
 * Vue Router 配置 - 用户端
 * 
 * 负责用户端应用的路由配置，包括：
 * - 路由定义
 * - 导航守卫
 * - 页面标题设置
 * 
 * @module user-app/router
 * @author Person F
 */

import { createRouter, createWebHistory, type RouteRecordRaw, type RouteLocationNormalized } from 'vue-router'

// =============================================================================
// 路由元信息类型扩展
// =============================================================================

declare module 'vue-router' {
  interface RouteMeta {
    /** 页面标题 */
    title?: string
    /** 是否需要认证 */
    requiresAuth?: boolean
    /** 是否只允许游客访问 */
    guest?: boolean
    /** 是否缓存页面 */
    keepAlive?: boolean
    /** 页面过渡动画 */
    transition?: string
  }
}

// =============================================================================
// 路由定义
// =============================================================================

const routes: RouteRecordRaw[] = [
  // 首页
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/Home.vue'),
    meta: {
      title: '首页',
      keepAlive: true,
    },
  },
  
  // 搜索页
  {
    path: '/search',
    name: 'Search',
    component: () => import('@/views/Search.vue'),
    meta: {
      title: '搜索',
      keepAlive: true,
    },
  },
  
  // 物品详情页
  {
    path: '/item/:id',
    name: 'ItemDetail',
    component: () => import('@/views/ItemDetail.vue'),
    meta: {
      title: '详情',
    },
    props: true,
  },
  
  // 分类页
  {
    path: '/category/:type',
    name: 'Category',
    component: () => import('@/views/Category.vue'),
    meta: {
      title: '分类',
      keepAlive: true,
    },
    props: true,
  },
  
  // 登录页
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue'),
    meta: {
      title: '登录',
      guest: true,
    },
  },
  
  // 注册页
  {
    path: '/register',
    name: 'Register',
    component: () => import('@/views/Register.vue'),
    meta: {
      title: '注册',
      guest: true,
    },
  },
  
  // 个人中心
  {
    path: '/profile',
    name: 'Profile',
    component: () => import('@/views/Profile.vue'),
    meta: {
      title: '个人中心',
      requiresAuth: true,
    },
  },
  
  // 历史记录
  {
    path: '/history',
    name: 'History',
    component: () => import('@/views/History.vue'),
    meta: {
      title: '浏览历史',
      requiresAuth: true,
    },
  },
  
  // 收藏列表
  {
    path: '/favorites',
    name: 'Favorites',
    component: () => import('@/views/Favorites.vue'),
    meta: {
      title: '我的收藏',
      requiresAuth: true,
    },
  },
  
  // 设置页
  {
    path: '/settings',
    name: 'Settings',
    component: () => import('@/views/Settings.vue'),
    meta: {
      title: '设置',
      requiresAuth: true,
    },
  },
  
  // 404 页面
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('@/views/NotFound.vue'),
    meta: {
      title: '页面未找到',
    },
  },
]

// =============================================================================
// 创建路由实例
// =============================================================================

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
  scrollBehavior(to, from, savedPosition) {
    // 如果有保存的位置（后退/前进），恢复到保存的位置
    if (savedPosition) {
      return savedPosition
    }
    
    // 如果是锚点，滚动到锚点
    if (to.hash) {
      return {
        el: to.hash,
        behavior: 'smooth',
      }
    }
    
    // 默认滚动到顶部
    return { top: 0, behavior: 'smooth' }
  },
})

// =============================================================================
// 导航守卫
// =============================================================================

/**
 * 获取用户登录状态
 * 
 * 注意：这里简化处理，实际应该从 store 或 token 判断
 */
function isAuthenticated(): boolean {
  const token = localStorage.getItem('recommend_auth_token') || localStorage.getItem('token')
  return !!token
}

/**
 * 全局前置守卫
 */
router.beforeEach(async (to: RouteLocationNormalized, from: RouteLocationNormalized, next) => {
  // 开始加载进度条（如果使用 NProgress）
  // NProgress.start()
  
  const authenticated = isAuthenticated()
  
  // 需要登录的页面
  if (to.meta.requiresAuth && !authenticated) {
    // 保存目标路由，登录后跳转
    next({
      name: 'Login',
      query: { redirect: to.fullPath },
    })
    return
  }
  
  // 已登录用户不能访问的页面（如登录、注册）
  if (to.meta.guest && authenticated) {
    next({ name: 'Home' })
    return
  }
  
  next()
})

/**
 * 全局后置守卫
 */
router.afterEach((to: RouteLocationNormalized) => {
  // 设置页面标题
  const appName = 'AI 推荐'
  document.title = to.meta.title ? `${to.meta.title} - ${appName}` : appName
  
  // 结束加载进度条
  // NProgress.done()
})

/**
 * 全局错误处理
 */
router.onError((error) => {
  console.error('Router error:', error)
  
  // 处理动态导入失败
  if (error.message.includes('Failed to fetch dynamically imported module')) {
    // 提示用户刷新页面
    if (window.confirm('页面加载失败，是否刷新重试？')) {
      window.location.reload()
    }
  }
})

// =============================================================================
// 导出
// =============================================================================

export default router

// 导出路由工具函数
export { routes }

/**
 * 路由跳转辅助函数
 */
export const routerHelper = {
  /**
   * 跳转到首页
   */
  goHome() {
    router.push({ name: 'Home' })
  },
  
  /**
   * 跳转到登录页
   */
  goLogin(redirect?: string) {
    router.push({
      name: 'Login',
      query: redirect ? { redirect } : undefined,
    })
  },
  
  /**
   * 跳转到物品详情页
   */
  goItemDetail(itemId: string) {
    router.push({ name: 'ItemDetail', params: { id: itemId } })
  },
  
  /**
   * 跳转到搜索页
   */
  goSearch(query?: string) {
    router.push({
      name: 'Search',
      query: query ? { q: query } : undefined,
    })
  },
  
  /**
   * 跳转到分类页
   */
  goCategory(type: string) {
    router.push({ name: 'Category', params: { type } })
  },
  
  /**
   * 返回上一页
   */
  goBack() {
    if (window.history.length > 1) {
      router.back()
    } else {
      router.push({ name: 'Home' })
    }
  },
}

