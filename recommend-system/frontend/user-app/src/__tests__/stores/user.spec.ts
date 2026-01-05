/**
 * User Store 单元测试
 * 
 * 测试用户状态管理的核心功能：登录、注册、登出、获取用户信息等。
 * 
 * @author Person C
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useUserStore } from '@/stores/user'
import type { User, LoginResponse, UserProfile, UserBehavior } from '@shared/types'

// =============================================================================
// Mock 数据
// =============================================================================

const mockUser: User = {
  id: 'user-123',
  name: '测试用户',
  email: 'test@example.com',
  age: 25,
  gender: 'male',
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-04T00:00:00Z',
}

const mockLoginResponse: LoginResponse = {
  token: 'mock-jwt-token',
  user: mockUser,
  expires_at: '2025-01-05T00:00:00Z',
}

const mockProfile: UserProfile = {
  user: mockUser,
  total_actions: 150,
  preferred_types: { movie: 50, product: 30, article: 70 },
  active_hours: { 9: 10, 10: 15, 20: 25, 21: 20 },
  last_active: '2025-01-04T10:00:00Z',
}

const mockBehaviors: UserBehavior[] = [
  {
    user_id: 'user-123',
    item_id: 'item-1',
    action: 'view',
    timestamp: '2025-01-04T10:00:00Z',
  },
  {
    user_id: 'user-123',
    item_id: 'item-2',
    action: 'like',
    timestamp: '2025-01-04T09:00:00Z',
  },
]

// =============================================================================
// Mock API Provider
// =============================================================================

const mockAuthService = {
  login: vi.fn(),
  register: vi.fn(),
  logout: vi.fn(),
  refreshToken: vi.fn(),
  getCurrentUser: vi.fn(),
}

const mockUserService = {
  getUser: vi.fn(),
  updateUser: vi.fn(),
  getProfile: vi.fn(),
  getBehaviors: vi.fn(),
  recordBehavior: vi.fn(),
}

const mockApiProvider = {
  auth: mockAuthService,
  user: mockUserService,
  item: {} as any,
  recommend: {} as any,
  analytics: {} as any,
  adminUser: {} as any,
  adminItem: {} as any,
}

// Mock Vue inject
vi.mock('vue', async () => {
  const actual = await vi.importActual('vue')
  return {
    ...actual,
    inject: () => mockApiProvider,
  }
})

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value
    },
    removeItem: (key: string) => {
      delete store[key]
    },
    clear: () => {
      store = {}
    },
  }
})()

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
})

// =============================================================================
// 测试套件
// =============================================================================

describe('useUserStore', () => {
  beforeEach(() => {
    // 创建新的 Pinia 实例
    setActivePinia(createPinia())
    // 清理 mock
    vi.clearAllMocks()
    localStorageMock.clear()
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  // ===========================================================================
  // 初始状态测试
  // ===========================================================================

  describe('初始状态', () => {
    it('应该有正确的初始状态', () => {
      const store = useUserStore()

      expect(store.token).toBeNull()
      expect(store.currentUser).toBeNull()
      expect(store.profile).toBeNull()
      expect(store.behaviors).toEqual([])
      expect(store.isLoading).toBe(false)
      expect(store.error).toBeNull()
    })

    it('isLoggedIn 应该在未登录时返回 false', () => {
      const store = useUserStore()

      expect(store.isLoggedIn).toBe(false)
    })

    it('displayName 应该在未登录时返回"游客"', () => {
      const store = useUserStore()

      expect(store.displayName).toBe('游客')
    })
  })

  // ===========================================================================
  // 登录测试
  // ===========================================================================

  describe('login', () => {
    it('登录成功时应该更新状态', async () => {
      mockAuthService.login.mockResolvedValue(mockLoginResponse)
      const store = useUserStore()

      await store.login({ email: 'test@example.com', password: 'password123' })

      expect(store.token).toBe('mock-jwt-token')
      expect(store.currentUser).toEqual(mockUser)
      expect(store.isLoggedIn).toBe(true)
      expect(store.error).toBeNull()
    })

    it('登录成功时应该保存 token 到 localStorage', async () => {
      mockAuthService.login.mockResolvedValue(mockLoginResponse)
      const store = useUserStore()

      await store.login({ email: 'test@example.com', password: 'password123' })

      expect(localStorageMock.getItem('token')).toBe('mock-jwt-token')
    })

    it('登录失败时应该设置错误信息', async () => {
      mockAuthService.login.mockRejectedValue(new Error('密码错误'))
      const store = useUserStore()

      await expect(
        store.login({ email: 'test@example.com', password: 'wrongpassword' })
      ).rejects.toThrow('密码错误')

      expect(store.token).toBeNull()
      expect(store.currentUser).toBeNull()
      expect(store.error).toBe('密码错误')
    })

    it('登录时应该显示加载状态', async () => {
      mockAuthService.login.mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve(mockLoginResponse), 100))
      )
      const store = useUserStore()

      const loginPromise = store.login({ email: 'test@example.com', password: 'password123' })
      expect(store.isLoading).toBe(true)

      await loginPromise
      expect(store.isLoading).toBe(false)
    })
  })

  // ===========================================================================
  // 注册测试
  // ===========================================================================

  describe('register', () => {
    it('注册成功后应该自动登录', async () => {
      mockAuthService.register.mockResolvedValue(undefined)
      mockAuthService.login.mockResolvedValue(mockLoginResponse)
      const store = useUserStore()

      await store.register({
        name: '新用户',
        email: 'new@example.com',
        password: 'password123',
      })

      expect(mockAuthService.register).toHaveBeenCalled()
      expect(mockAuthService.login).toHaveBeenCalled()
      expect(store.isLoggedIn).toBe(true)
    })

    it('注册失败时应该设置错误信息', async () => {
      mockAuthService.register.mockRejectedValue(new Error('邮箱已存在'))
      const store = useUserStore()

      await expect(
        store.register({
          name: '新用户',
          email: 'existing@example.com',
          password: 'password123',
        })
      ).rejects.toThrow('邮箱已存在')

      expect(store.error).toBe('邮箱已存在')
    })
  })

  // ===========================================================================
  // 登出测试
  // ===========================================================================

  describe('logout', () => {
    it('登出时应该清除所有状态', async () => {
      // 先登录
      mockAuthService.login.mockResolvedValue(mockLoginResponse)
      const store = useUserStore()
      await store.login({ email: 'test@example.com', password: 'password123' })

      // 然后登出
      mockAuthService.logout.mockResolvedValue(undefined)
      await store.logout()

      expect(store.token).toBeNull()
      expect(store.currentUser).toBeNull()
      expect(store.profile).toBeNull()
      expect(store.behaviors).toEqual([])
      expect(store.isLoggedIn).toBe(false)
    })

    it('登出时应该清除 localStorage 中的 token', async () => {
      localStorageMock.setItem('token', 'some-token')
      mockAuthService.logout.mockResolvedValue(undefined)
      const store = useUserStore()

      await store.logout()

      expect(localStorageMock.getItem('token')).toBeNull()
    })

    it('即使 API 调用失败也应该清除本地状态', async () => {
      mockAuthService.login.mockResolvedValue(mockLoginResponse)
      mockAuthService.logout.mockRejectedValue(new Error('网络错误'))
      const store = useUserStore()

      await store.login({ email: 'test@example.com', password: 'password123' })
      await store.logout()

      expect(store.token).toBeNull()
      expect(store.currentUser).toBeNull()
    })
  })

  // ===========================================================================
  // 获取用户画像测试
  // ===========================================================================

  describe('fetchProfile', () => {
    it('应该获取并存储用户画像', async () => {
      mockAuthService.login.mockResolvedValue(mockLoginResponse)
      mockUserService.getProfile.mockResolvedValue(mockProfile)
      const store = useUserStore()

      await store.login({ email: 'test@example.com', password: 'password123' })
      await store.fetchProfile()

      expect(store.profile).toEqual(mockProfile)
      expect(mockUserService.getProfile).toHaveBeenCalledWith('user-123')
    })

    it('未登录时不应该发起请求', async () => {
      const store = useUserStore()

      await store.fetchProfile()

      expect(mockUserService.getProfile).not.toHaveBeenCalled()
    })
  })

  // ===========================================================================
  // 获取行为历史测试
  // ===========================================================================

  describe('fetchBehaviors', () => {
    it('应该获取并存储行为历史', async () => {
      mockAuthService.login.mockResolvedValue(mockLoginResponse)
      mockUserService.getBehaviors.mockResolvedValue(mockBehaviors)
      const store = useUserStore()

      await store.login({ email: 'test@example.com', password: 'password123' })
      await store.fetchBehaviors(50)

      expect(store.behaviors).toEqual(mockBehaviors)
      expect(mockUserService.getBehaviors).toHaveBeenCalledWith('user-123', 50)
    })

    it('应该使用默认限制 50', async () => {
      mockAuthService.login.mockResolvedValue(mockLoginResponse)
      mockUserService.getBehaviors.mockResolvedValue(mockBehaviors)
      const store = useUserStore()

      await store.login({ email: 'test@example.com', password: 'password123' })
      await store.fetchBehaviors()

      expect(mockUserService.getBehaviors).toHaveBeenCalledWith('user-123', 50)
    })
  })

  // ===========================================================================
  // 更新用户信息测试
  // ===========================================================================

  describe('updateProfile', () => {
    it('应该更新用户信息', async () => {
      const updatedUser = { ...mockUser, name: '新名字' }
      mockAuthService.login.mockResolvedValue(mockLoginResponse)
      mockUserService.updateUser.mockResolvedValue(updatedUser)
      const store = useUserStore()

      await store.login({ email: 'test@example.com', password: 'password123' })
      await store.updateProfile({ name: '新名字' })

      expect(store.currentUser?.name).toBe('新名字')
      expect(mockUserService.updateUser).toHaveBeenCalledWith('user-123', { name: '新名字' })
    })

    it('更新失败时应该设置错误', async () => {
      mockAuthService.login.mockResolvedValue(mockLoginResponse)
      mockUserService.updateUser.mockRejectedValue(new Error('更新失败'))
      const store = useUserStore()

      await store.login({ email: 'test@example.com', password: 'password123' })

      await expect(store.updateProfile({ name: '新名字' })).rejects.toThrow('更新失败')

      expect(store.error).toBe('更新失败')
    })
  })

  // ===========================================================================
  // 计算属性测试
  // ===========================================================================

  describe('计算属性', () => {
    it('userId 应该返回当前用户 ID', async () => {
      mockAuthService.login.mockResolvedValue(mockLoginResponse)
      const store = useUserStore()

      await store.login({ email: 'test@example.com', password: 'password123' })

      expect(store.userId).toBe('user-123')
    })

    it('displayName 应该返回用户名称', async () => {
      mockAuthService.login.mockResolvedValue(mockLoginResponse)
      const store = useUserStore()

      await store.login({ email: 'test@example.com', password: 'password123' })

      expect(store.displayName).toBe('测试用户')
    })

    it('avatarInitial 应该返回用户名首字母', async () => {
      mockAuthService.login.mockResolvedValue(mockLoginResponse)
      const store = useUserStore()

      await store.login({ email: 'test@example.com', password: 'password123' })

      expect(store.avatarInitial).toBe('测')
    })
  })

  // ===========================================================================
  // 错误处理测试
  // ===========================================================================

  describe('错误处理', () => {
    it('clearError 应该清除错误信息', async () => {
      mockAuthService.login.mockRejectedValue(new Error('测试错误'))
      const store = useUserStore()

      try {
        await store.login({ email: 'test@example.com', password: 'wrong' })
      } catch {
        // 忽略错误
      }

      expect(store.error).toBe('测试错误')

      store.clearError()

      expect(store.error).toBeNull()
    })
  })

  // ===========================================================================
  // 初始化测试
  // ===========================================================================

  describe('init', () => {
    it('有 token 时应该获取当前用户', async () => {
      localStorageMock.setItem('token', 'existing-token')
      mockAuthService.getCurrentUser.mockResolvedValue(mockUser)

      // 需要重新创建 store 以读取 localStorage
      setActivePinia(createPinia())
      const store = useUserStore()
      
      // 手动设置 token（因为 store 在创建时会读取 localStorage）
      store.token = 'existing-token'

      await store.init()

      expect(mockAuthService.getCurrentUser).toHaveBeenCalled()
    })

    it('没有 token 时不应该发起请求', async () => {
      const store = useUserStore()

      await store.init()

      expect(mockAuthService.getCurrentUser).not.toHaveBeenCalled()
    })
  })
})

