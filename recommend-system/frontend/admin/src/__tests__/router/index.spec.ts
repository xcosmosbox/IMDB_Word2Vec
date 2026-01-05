/**
 * Router 单元测试
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

// 模拟路由配置
const mockRoutes: RouteRecordRaw[] = [
  {
    path: '/admin/login',
    name: 'AdminLogin',
    component: { template: '<div>Login</div>' },
    meta: {
      title: '管理员登录',
      requiresAuth: false,
    },
  },
  {
    path: '/admin',
    component: { template: '<div><router-view /></div>' },
    redirect: '/admin/dashboard',
    meta: {
      requiresAuth: true,
    },
    children: [
      {
        path: 'dashboard',
        name: 'AdminDashboard',
        component: { template: '<div>Dashboard</div>' },
        meta: {
          title: '仪表盘',
        },
      },
      {
        path: 'users',
        name: 'UserList',
        component: { template: '<div>UserList</div>' },
        meta: {
          title: '用户列表',
          permission: 'user:read',
        },
      },
      {
        path: 'items',
        name: 'ItemList',
        component: { template: '<div>ItemList</div>' },
        meta: {
          title: '物品列表',
          permission: 'item:read',
        },
      },
    ],
  },
]

describe('Admin Router', () => {
  let router: ReturnType<typeof createRouter>

  beforeEach(() => {
    router = createRouter({
      history: createWebHistory(),
      routes: mockRoutes,
    })
  })

  describe('路由配置', () => {
    it('应该包含登录路由', () => {
      const loginRoute = router.getRoutes().find(r => r.name === 'AdminLogin')
      expect(loginRoute).toBeTruthy()
      expect(loginRoute?.meta?.requiresAuth).toBe(false)
    })

    it('应该包含仪表盘路由', () => {
      const dashboardRoute = router.getRoutes().find(r => r.name === 'AdminDashboard')
      expect(dashboardRoute).toBeTruthy()
      expect(dashboardRoute?.meta?.title).toBe('仪表盘')
    })

    it('应该包含用户管理路由', () => {
      const userRoute = router.getRoutes().find(r => r.name === 'UserList')
      expect(userRoute).toBeTruthy()
      expect(userRoute?.meta?.permission).toBe('user:read')
    })

    it('应该包含物品管理路由', () => {
      const itemRoute = router.getRoutes().find(r => r.name === 'ItemList')
      expect(itemRoute).toBeTruthy()
      expect(itemRoute?.meta?.permission).toBe('item:read')
    })
  })

  describe('路由导航', () => {
    it('应该能导航到登录页', async () => {
      await router.push('/admin/login')
      expect(router.currentRoute.value.name).toBe('AdminLogin')
    })

    it('应该能导航到仪表盘', async () => {
      await router.push('/admin/dashboard')
      expect(router.currentRoute.value.name).toBe('AdminDashboard')
    })

    it('/admin 应该重定向到 /admin/dashboard', async () => {
      await router.push('/admin')
      expect(router.currentRoute.value.path).toBe('/admin/dashboard')
    })
  })

  describe('路由元信息', () => {
    it('登录页不需要认证', async () => {
      await router.push('/admin/login')
      expect(router.currentRoute.value.meta.requiresAuth).toBe(false)
    })

    it('用户列表需要 user:read 权限', async () => {
      await router.push('/admin/users')
      expect(router.currentRoute.value.meta.permission).toBe('user:read')
    })

    it('物品列表需要 item:read 权限', async () => {
      await router.push('/admin/items')
      expect(router.currentRoute.value.meta.permission).toBe('item:read')
    })
  })
})

