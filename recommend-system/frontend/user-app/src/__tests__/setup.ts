/**
 * Vitest 测试设置文件
 * 
 * @author Person F
 */

import { vi } from 'vitest'
import { config } from '@vue/test-utils'

// =============================================================================
// 全局 Mock
// =============================================================================

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
  length: 0,
  key: vi.fn(),
}

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
})

// Mock sessionStorage
Object.defineProperty(window, 'sessionStorage', {
  value: localStorageMock,
})

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// Mock IntersectionObserver
class IntersectionObserverMock {
  observe = vi.fn()
  unobserve = vi.fn()
  disconnect = vi.fn()
}

Object.defineProperty(window, 'IntersectionObserver', {
  value: IntersectionObserverMock,
})

// Mock ResizeObserver
class ResizeObserverMock {
  observe = vi.fn()
  unobserve = vi.fn()
  disconnect = vi.fn()
}

Object.defineProperty(window, 'ResizeObserver', {
  value: ResizeObserverMock,
})

// =============================================================================
// Vue Test Utils 配置
// =============================================================================

// 全局组件 stubs
config.global.stubs = {
  Teleport: true,
  Transition: false,
  TransitionGroup: false,
}

// 全局插件
config.global.plugins = []

// =============================================================================
// 测试工具函数
// =============================================================================

/**
 * 等待 DOM 更新
 */
export function flushPromises(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0))
}

/**
 * 创建模拟的 API Provider
 */
export function createMockApiProvider() {
  return {
    auth: {
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
      refreshToken: vi.fn(),
      getCurrentUser: vi.fn(),
    },
    user: {
      getUser: vi.fn(),
      updateUser: vi.fn(),
      getProfile: vi.fn(),
      getBehaviors: vi.fn(),
      recordBehavior: vi.fn(),
    },
    item: {
      getItem: vi.fn(),
      searchItems: vi.fn(),
      listItems: vi.fn(),
      getItemStats: vi.fn(),
      getSimilarItems: vi.fn(),
    },
    recommend: {
      getRecommendations: vi.fn(),
      submitFeedback: vi.fn(),
      getSimilarRecommendations: vi.fn(),
    },
    analytics: {
      getDashboardStats: vi.fn(),
      getUserTrend: vi.fn(),
      getItemTypeStats: vi.fn(),
      getRecommendationTrend: vi.fn(),
      getTopCategories: vi.fn(),
      getCTRTrend: vi.fn(),
    },
    adminUser: {
      listUsers: vi.fn(),
      getUser: vi.fn(),
      createUser: vi.fn(),
      updateUser: vi.fn(),
      deleteUser: vi.fn(),
    },
    adminItem: {
      listItems: vi.fn(),
      getItem: vi.fn(),
      createItem: vi.fn(),
      updateItem: vi.fn(),
      deleteItem: vi.fn(),
    },
  }
}
