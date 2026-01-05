/**
 * Recommend Store 单元测试
 * 
 * @author Person F
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useRecommendStore } from '@/stores/recommend'
import type { IApiProvider } from '@shared/api/interfaces'
import { createMockApiProvider } from '../setup'

// Mock inject
vi.mock('vue', async () => {
  const actual = await vi.importActual('vue')
  return {
    ...actual,
    inject: vi.fn(() => mockApi),
  }
})

let mockApi: IApiProvider

describe('Recommend Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    mockApi = createMockApiProvider() as unknown as IApiProvider
    localStorage.clear()
  })

  describe('初始状态', () => {
    it('应该有正确的初始状态', () => {
      const store = useRecommendStore()

      expect(store.recommendations).toEqual([])
      expect(store.currentRequestId).toBeNull()
      expect(store.currentStrategy).toBe('')
      expect(store.isLoading).toBe(false)
      expect(store.error).toBeNull()
    })
  })

  describe('setCurrentUserId', () => {
    it('应该设置用户 ID', () => {
      const store = useRecommendStore()

      store.setCurrentUserId('user_001')

      expect(store.currentUserId).toBe('user_001')
    })
  })

  describe('getRecommendations', () => {
    it('应该获取推荐列表', async () => {
      const store = useRecommendStore()
      store.setCurrentUserId('user_001')

      const mockResponse = {
        recommendations: [
          { item_id: 'item_001', score: 0.9 },
          { item_id: 'item_002', score: 0.8 },
        ],
        request_id: 'req_001',
        strategy: 'collaborative',
      }

      vi.mocked(mockApi.recommend.getRecommendations).mockResolvedValue(mockResponse)

      await store.getRecommendations({ limit: 10 })

      expect(store.recommendations).toEqual(mockResponse.recommendations)
      expect(store.currentRequestId).toBe('req_001')
      expect(store.currentStrategy).toBe('collaborative')
    })

    it('未设置用户 ID 时应该抛出错误', async () => {
      const store = useRecommendStore()

      await expect(store.getRecommendations()).rejects.toThrow('User ID not set')
    })
  })

  describe('喜欢/不喜欢功能', () => {
    it('应该正确切换喜欢状态', async () => {
      const store = useRecommendStore()
      store.setCurrentUserId('user_001')

      // 初始状态
      expect(store.isItemLiked('item_001')).toBe(false)

      // 喜欢
      await store.toggleLike('item_001')
      expect(store.isItemLiked('item_001')).toBe(true)

      // 取消喜欢
      await store.toggleLike('item_001')
      expect(store.isItemLiked('item_001')).toBe(false)
    })

    it('喜欢应该清除不喜欢状态', async () => {
      const store = useRecommendStore()
      store.setCurrentUserId('user_001')

      // 先设置不喜欢
      store.dislikedItems.add('item_001')
      expect(store.isItemDisliked('item_001')).toBe(true)

      // 喜欢后应该清除不喜欢
      await store.recordBehavior({ item_id: 'item_001', action: 'like' })
      expect(store.isItemLiked('item_001')).toBe(true)
      expect(store.isItemDisliked('item_001')).toBe(false)
    })
  })

  describe('计算属性', () => {
    it('hasRecommendations 应该正确计算', () => {
      const store = useRecommendStore()

      expect(store.hasRecommendations).toBe(false)

      store.recommendations = [{ item_id: 'item_001', score: 0.9 }]

      expect(store.hasRecommendations).toBe(true)
    })

    it('recommendationCount 应该正确计算', () => {
      const store = useRecommendStore()

      expect(store.recommendationCount).toBe(0)

      store.recommendations = [
        { item_id: 'item_001', score: 0.9 },
        { item_id: 'item_002', score: 0.8 },
      ]

      expect(store.recommendationCount).toBe(2)
    })

    it('likedCount 应该正确计算', () => {
      const store = useRecommendStore()

      expect(store.likedCount).toBe(0)

      store.likedItems.add('item_001')
      store.likedItems.add('item_002')

      expect(store.likedCount).toBe(2)
    })
  })

  describe('clearRecommendations', () => {
    it('应该清除推荐数据', () => {
      const store = useRecommendStore()

      store.recommendations = [{ item_id: 'item_001', score: 0.9 }]
      store.currentRequestId = 'req_001'
      store.currentStrategy = 'collaborative'

      store.clearRecommendations()

      expect(store.recommendations).toEqual([])
      expect(store.currentRequestId).toBeNull()
      expect(store.currentStrategy).toBe('')
    })
  })

  describe('$reset', () => {
    it('应该重置所有状态', () => {
      const store = useRecommendStore()

      store.recommendations = [{ item_id: 'item_001', score: 0.9 }]
      store.currentRequestId = 'req_001'
      store.isLoading = true
      store.error = '错误'

      store.$reset()

      expect(store.recommendations).toEqual([])
      expect(store.currentRequestId).toBeNull()
      expect(store.isLoading).toBe(false)
      expect(store.error).toBeNull()
    })
  })
})

