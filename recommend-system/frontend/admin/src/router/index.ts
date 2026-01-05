/**
 * Admin Router - 管理后台路由配置
 * 
 * 功能：
 * - 路由定义
 * - 路由守卫
 * - 权限控制
 */
import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import { useAdminStore } from '@/stores/admin'

// 布局组件
import AdminLayout from '@/layouts/AdminLayout.vue'

// 路由配置
const routes: RouteRecordRaw[] = [
  // 登录页
  {
    path: '/admin/login',
    name: 'AdminLogin',
    component: () => import('@/views/auth/Login.vue'),
    meta: {
      title: '管理员登录',
      requiresAuth: false,
    },
  },

  // 管理后台主体
  {
    path: '/admin',
    component: AdminLayout,
    redirect: '/admin/dashboard',
    meta: {
      requiresAuth: true,
    },
    children: [
      // 仪表盘
      {
        path: 'dashboard',
        name: 'AdminDashboard',
        component: () => import('@/views/dashboard/Dashboard.vue'),
        meta: {
          title: '仪表盘',
        },
      },

      // 用户管理
      {
        path: 'users',
        name: 'UserList',
        component: () => import('@/views/users/UserList.vue'),
        meta: {
          title: '用户列表',
          permission: 'user:read',
        },
      },
      {
        path: 'users/create',
        name: 'UserCreate',
        component: () => import('@/views/users/UserForm.vue'),
        meta: {
          title: '新增用户',
          permission: 'user:write',
        },
      },
      {
        path: 'users/:id',
        name: 'UserDetail',
        component: () => import('@/views/users/UserDetail.vue'),
        meta: {
          title: '用户详情',
          permission: 'user:read',
        },
      },
      {
        path: 'users/:id/edit',
        name: 'UserEdit',
        component: () => import('@/views/users/UserForm.vue'),
        meta: {
          title: '编辑用户',
          permission: 'user:write',
        },
      },

      // 物品管理
      {
        path: 'items',
        name: 'ItemList',
        component: () => import('@/views/items/ItemList.vue'),
        meta: {
          title: '物品列表',
          permission: 'item:read',
        },
      },
      {
        path: 'items/create',
        name: 'ItemCreate',
        component: () => import('@/views/items/ItemForm.vue'),
        meta: {
          title: '新增物品',
          permission: 'item:write',
        },
      },
      {
        path: 'items/:id',
        name: 'ItemDetail',
        component: () => import('@/views/items/ItemDetail.vue'),
        meta: {
          title: '物品详情',
          permission: 'item:read',
        },
      },
      {
        path: 'items/:id/edit',
        name: 'ItemEdit',
        component: () => import('@/views/items/ItemForm.vue'),
        meta: {
          title: '编辑物品',
          permission: 'item:write',
        },
      },

      // 数据分析（由 Person E 实现）
      {
        path: 'analytics',
        name: 'Analytics',
        component: () => import('@/views/analytics/Analytics.vue'),
        meta: {
          title: '数据分析',
          permission: 'analytics:read',
        },
      },

      // 系统设置
      {
        path: 'settings',
        name: 'Settings',
        component: () => import('@/views/settings/Settings.vue'),
        meta: {
          title: '系统设置',
          permission: 'settings:read',
        },
      },

      // 个人信息
      {
        path: 'profile',
        name: 'AdminProfile',
        component: () => import('@/views/profile/Profile.vue'),
        meta: {
          title: '个人信息',
        },
      },
    ],
  },

  // 404 页面
  {
    path: '/admin/:pathMatch(.*)*',
    name: 'AdminNotFound',
    component: () => import('@/views/error/NotFound.vue'),
    meta: {
      title: '页面不存在',
    },
  },
]

// 创建路由实例
const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    }
    return { top: 0 }
  },
})

// 路由守卫
router.beforeEach(async (to, from, next) => {
  // 设置页面标题
  const title = to.meta.title as string
  if (title) {
    document.title = `${title} - 推荐系统管理后台`
  }

  // 不需要认证的页面
  if (to.meta.requiresAuth === false) {
    next()
    return
  }

  // 需要认证
  const adminStore = useAdminStore()

  // 初始化 store（从本地存储恢复状态）
  if (!adminStore.isLoggedIn) {
    await adminStore.initialize()
  }

  // 检查登录状态
  if (!adminStore.isLoggedIn) {
    next({
      path: '/admin/login',
      query: { redirect: to.fullPath },
    })
    return
  }

  // 检查 token 是否过期
  if (adminStore.isTokenExpired) {
    try {
      await adminStore.refreshToken()
    } catch {
      await adminStore.logout()
      next({
        path: '/admin/login',
        query: { redirect: to.fullPath },
      })
      return
    }
  }

  // 检查权限
  const requiredPermission = to.meta.permission as string | undefined
  if (requiredPermission && !adminStore.hasPermission(requiredPermission)) {
    // 无权限，跳转到首页或显示无权限页面
    next({ path: '/admin/dashboard' })
    return
  }

  next()
})

export default router

