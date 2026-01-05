/**
 * Admin Store 单元测试
 */
import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useAdminStore } from '@/stores/admin'

// Mock localStorage
const localStorageMock = {
  store: {} as Record<string, string>,
  getItem: vi.fn((key: string) => localStorageMock.store[key] || null),
  setItem: vi.fn((key: string, value: string) => {
    localStorageMock.store[key] = value
  }),
  removeItem: vi.fn((key: string) => {
    delete localStorageMock.store[key]
  }),
  clear: vi.fn(() => {
    localStorageMock.store = {}
  }),
}

Object.defineProperty(window, 'localStorage', { value: localStorageMock })

describe('useAdminStore', () => {
  beforeEach(() => {
    // 每个测试前重置 pinia
    setActivePinia(createPinia())
    localStorageMock.clear()
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('初始状态', () => {
    it('应该正确初始化默认状态', () => {
      const store = useAdminStore()

      expect(store.token).toBeNull()
      expect(store.currentAdmin).toBeNull()
      expect(store.isLoggedIn).toBe(false)
      expect(store.isTokenExpired).toBe(true)
    })

    it('应该从 localStorage 恢复 token', () => {
      localStorageMock.store['admin_token'] = 'saved_token'
      
      const store = useAdminStore()
      
      expect(store.token).toBe('saved_token')
    })
  })

  describe('login', () => {
    it('应该成功登录并设置状态', async () => {
      const store = useAdminStore()

      await store.login({
        email: 'admin@example.com',
        password: 'password123',
      })

      expect(store.isLoggedIn).toBe(true)
      expect(store.token).toBeTruthy()
      expect(store.currentAdmin).toBeTruthy()
      expect(store.currentAdmin?.email).toBe('admin@example.com')
      expect(localStorageMock.setItem).toHaveBeenCalled()
    })

    it('应该将 token 存储到 localStorage', async () => {
      const store = useAdminStore()

      await store.login({
        email: 'admin@example.com',
        password: 'password123',
      })

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        'admin_token',
        expect.any(String)
      )
    })
  })

  describe('logout', () => {
    it('应该清除所有状态', async () => {
      const store = useAdminStore()

      // 先登录
      await store.login({
        email: 'admin@example.com',
        password: 'password123',
      })

      expect(store.isLoggedIn).toBe(true)

      // 然后登出
      await store.logout()

      expect(store.token).toBeNull()
      expect(store.currentAdmin).toBeNull()
      expect(store.isLoggedIn).toBe(false)
      expect(localStorageMock.removeItem).toHaveBeenCalledWith('admin_token')
    })
  })

  describe('hasPermission', () => {
    it('超级管理员应该拥有所有权限', async () => {
      const store = useAdminStore()

      await store.login({
        email: 'admin@example.com',
        password: 'password123',
      })

      // 登录后默认是 super_admin
      expect(store.hasPermission('any_permission')).toBe(true)
      expect(store.hasPermission('user:read')).toBe(true)
      expect(store.hasPermission('user:write')).toBe(true)
    })

    it('未登录时应该没有任何权限', () => {
      const store = useAdminStore()

      expect(store.hasPermission('user:read')).toBe(false)
    })
  })

  describe('hasAnyPermission', () => {
    it('应该正确检查多个权限（任意一个）', async () => {
      const store = useAdminStore()

      await store.login({
        email: 'admin@example.com',
        password: 'password123',
      })

      expect(store.hasAnyPermission(['user:read', 'unknown:permission'])).toBe(true)
    })
  })

  describe('hasAllPermissions', () => {
    it('应该正确检查多个权限（全部）', async () => {
      const store = useAdminStore()

      await store.login({
        email: 'admin@example.com',
        password: 'password123',
      })

      expect(store.hasAllPermissions(['user:read', 'user:write'])).toBe(true)
    })
  })

  describe('updateAdminInfo', () => {
    it('应该更新管理员信息', async () => {
      const store = useAdminStore()

      await store.login({
        email: 'admin@example.com',
        password: 'password123',
      })

      store.updateAdminInfo({ name: '新名称' })

      expect(store.currentAdmin?.name).toBe('新名称')
    })

    it('未登录时更新应该无效', () => {
      const store = useAdminStore()

      store.updateAdminInfo({ name: '新名称' })

      expect(store.currentAdmin).toBeNull()
    })
  })

  describe('计算属性', () => {
    it('adminName 应该返回正确的名称', async () => {
      const store = useAdminStore()

      expect(store.adminName).toBe('管理员')

      await store.login({
        email: 'admin@example.com',
        password: 'password123',
      })

      expect(store.adminName).toBe('系统管理员')
    })

    it('isSuperAdmin 应该正确判断超级管理员', async () => {
      const store = useAdminStore()

      expect(store.isSuperAdmin).toBe(false)

      await store.login({
        email: 'admin@example.com',
        password: 'password123',
      })

      expect(store.isSuperAdmin).toBe(true)
    })
  })
})

