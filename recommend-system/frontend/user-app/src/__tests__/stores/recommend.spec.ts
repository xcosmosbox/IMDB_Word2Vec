/**
 * Recommend Store 单元测试
 * 
 * 测试推荐状态管理的核心功能
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useRecommendStore } from '@/stores/recommend'
import type { IApiProvider } from '@shared/api/interfaces'
import type { RecommendResponse, Recommendation } from '@shared/types'

// Mock API Provider
const createMockApiProvider = (): IApiProvider => ({
  auth: {} as any,
  user: {
    getUser: vi.fn(),
    updateUser: vi.fn(),
    getProfile: vi.fn(),
    getBehaviors: vi.fn(),
    recordBehavior: vi.fn().mockResolvedValue(undefined),
  },
  item: {} as any,
  recommend: {
    getRecommendations: vi.fn(),
    submitFeedback: vi.fn().mockResolvedValue(undefined),
    getSimilarRecommendations: vi.fn(),
  },
  analytics: {} as any,
  adminUser: {} as any,
  adminItem: {} as any,
})

// Mock 推荐数据
const mockRecommendations: Recommendation[] = [
  {
    item_id: 'item_1',
    score: 0.95,
    reason: '根据你的浏览历史推荐',
    item: {
      id: 'item_1',
      type: 'movie',
      title: '肖申克的救赎',
      description: '一部经典的电影',
      category: '剧情',
      tags: ['经典', '励志'],
      status: 'active',
      created_at: '2024-01-01',
      updated_at: '2024-01-01',
    },
  },
  {
    item_id: 'item_2',
    score: 0.88,
    reason: '相似用户喜欢',
    item: {
      id: 'item_2',
      type: 'product',
      title: '无线蓝牙耳机',
      description: '高品质音频体验',
      category: '电子产品',
      tags: ['数码', '音频'],
      status: 'active',
      created_at: '2024-01-01',
      updated_at: '2024-01-01',
    },
  },
  {
    item_id: 'item_3',
    score: 0.75,
    reason: '热门推荐',
    item: {
      id: 'item_3',
      type: 'article',
      title: 'Vue 3 最佳实践',
      description: '深入了解 Vue 3 的开发技巧',
      category: '技术',
      tags: ['前端', 'Vue'],
      status: 'active',
      created_at: '2024-01-01',
      updated_at: '2024-01-01',
    },
  },
]

const mockRecommendResponse: RecommendResponse = {
  recommendations: mockRecommendations,
  request_id: 'req_123456',
  strategy: 'generative_hybrid',
}

describe('Recommend Store', () => {
  let store: ReturnType<typeof useRecommendStore>
  let mockApiProvider: IApiProvider

  beforeEach(() => {
    // 创建新的 Pinia 实例
    setActivePinia(createPinia())
    store = useRecommendStore()
    
    // 创建 Mock API Provider
    mockApiProvider = createMockApiProvider()
    store.setApiProvider(mockApiProvider)
  })

  describe('初始状态', () => {
    it('应该有正确的初始状态', () => {
      // 重新创建 store 以测试初始状态
      setActivePinia(createPinia())
      const freshStore = useRecommendStore()
      
      expect(freshStore.recommendations).toEqual([])
      expect(freshStore.isLoading).toBe(false)
      expect(freshStore.error).toBeNull()
      expect(freshStore.currentRequestId).toBe('')
      expect(freshStore.strategy).toBe('')
    })

    it('初始时 totalCount 应该为 0', () => {
      setActivePinia(createPinia())
      const freshStore = useRecommendStore()
      
      expect(freshStore.totalCount).toBe(0)
    })

    it('初始时 hasRecommendations 应该为 false', () => {
      setActivePinia(createPinia())
      const freshStore = useRecommendStore()
      
      expect(freshStore.hasRecommendations).toBe(false)
    })
  })

  describe('fetchRecommendations', () => {
    it('应该成功获取推荐列表', async () => {
      vi.mocked(mockApiProvider.recommend.getRecommendations).mockResolvedValue(mockRecommendResponse)

      await store.fetchRecommendations({
        user_id: 'user_1',
        limit: 10,
        scene: 'home',
      })

      expect(store.recommendations).toHaveLength(3)
      expect(store.currentRequestId).toBe('req_123456')
      expect(store.strategy).toBe('generative_hybrid')
      expect(store.isLoading).toBe(false)
      expect(store.error).toBeNull()
    })

    it('获取推荐时应该设置 isLoading 为 true', async () => {
      let loadingDuringRequest = false
      
      vi.mocked(mockApiProvider.recommend.getRecommendations).mockImplementation(async () => {
        loadingDuringRequest = store.isLoading
        return mockRecommendResponse
      })

      await store.fetchRecommendations({
        user_id: 'user_1',
        limit: 10,
        scene: 'home',
      })

      expect(loadingDuringRequest).toBe(true)
      expect(store.isLoading).toBe(false)
    })

    it('获取推荐失败时应该设置错误信息', async () => {
      const errorMessage = '网络错误'
      vi.mocked(mockApiProvider.recommend.getRecommendations).mockRejectedValue(new Error(errorMessage))

      await expect(store.fetchRecommendations({
        user_id: 'user_1',
        limit: 10,
        scene: 'home',
      })).rejects.toThrow(errorMessage)

      expect(store.error).toBe(errorMessage)
      expect(store.isLoading).toBe(false)
    })

    it('没有 API Provider 时应该抛出错误', async () => {
      setActivePinia(createPinia())
      const storeWithoutApi = useRecommendStore()

      await expect(storeWithoutApi.fetchRecommendations({
        user_id: 'user_1',
        limit: 10,
        scene: 'home',
      })).rejects.toThrow('API Provider 未注入')
    })
  })

  describe('Getters', () => {
    beforeEach(async () => {
      vi.mocked(mockApiProvider.recommend.getRecommendations).mockResolvedValue(mockRecommendResponse)
      await store.fetchRecommendations({
        user_id: 'user_1',
        limit: 10,
        scene: 'home',
      })
    })

    it('totalCount 应该返回正确的推荐数量', () => {
      expect(store.totalCount).toBe(3)
    })

    it('hasRecommendations 应该返回 true', () => {
      expect(store.hasRecommendations).toBe(true)
    })

    it('topRecommendations 应该只返回高分推荐', () => {
      const topRecs = store.topRecommendations
      
      expect(topRecs).toHaveLength(1)
      expect(topRecs[0].score).toBeGreaterThanOrEqual(0.8)
    })

    it('recommendationsByType 应该正确分组', () => {
      const byType = store.recommendationsByType
      
      expect(byType.movie).toHaveLength(1)
      expect(byType.product).toHaveLength(1)
      expect(byType.article).toHaveLength(1)
    })
  })

  describe('refreshRecommendations', () => {
    it('应该将当前推荐添加到已浏览列表', async () => {
      vi.mocked(mockApiProvider.recommend.getRecommendations).mockResolvedValue(mockRecommendResponse)
      
      await store.fetchRecommendations({
        user_id: 'user_1',
        limit: 10,
        scene: 'home',
      })

      const newResponse: RecommendResponse = {
        recommendations: [
          {
            item_id: 'item_4',
            score: 0.9,
            item: {
              id: 'item_4',
              type: 'video',
              title: '新视频',
              description: '新内容',
              category: '娱乐',
              tags: ['热门'],
              status: 'active',
              created_at: '2024-01-01',
              updated_at: '2024-01-01',
            },
          },
        ],
        request_id: 'req_789',
        strategy: 'refresh',
      }
      vi.mocked(mockApiProvider.recommend.getRecommendations).mockResolvedValue(newResponse)

      await store.refreshRecommendations('user_1', 10)

      // 验证已浏览列表包含之前的推荐
      expect(store.viewedItemIds.has('item_1')).toBe(true)
      expect(store.viewedItemIds.has('item_2')).toBe(true)
      expect(store.viewedItemIds.has('item_3')).toBe(true)
    })
  })

  describe('recordBehavior', () => {
    it('应该成功记录用户行为', async () => {
      vi.mocked(mockApiProvider.recommend.getRecommendations).mockResolvedValue(mockRecommendResponse)
      await store.fetchRecommendations({
        user_id: 'user_1',
        limit: 10,
        scene: 'home',
      })

      await store.recordBehavior({
        user_id: 'user_1',
        item_id: 'item_1',
        action: 'click',
      })

      expect(mockApiProvider.user.recordBehavior).toHaveBeenCalledWith({
        user_id: 'user_1',
        item_id: 'item_1',
        action: 'click',
        context: {
          request_id: 'req_123456',
        },
      })
    })

    it('点击行为应该更新已浏览列表', async () => {
      await store.recordBehavior({
        user_id: 'user_1',
        item_id: 'item_test',
        action: 'click',
      })

      expect(store.viewedItemIds.has('item_test')).toBe(true)
    })

    it('行为记录失败不应该抛出错误', async () => {
      vi.mocked(mockApiProvider.user.recordBehavior).mockRejectedValue(new Error('记录失败'))

      // 不应该抛出错误
      await expect(store.recordBehavior({
        user_id: 'user_1',
        item_id: 'item_1',
        action: 'view',
      })).resolves.toBeUndefined()
    })
  })

  describe('submitFeedback', () => {
    it('应该成功提交反馈', async () => {
      vi.mocked(mockApiProvider.recommend.getRecommendations).mockResolvedValue(mockRecommendResponse)
      await store.fetchRecommendations({
        user_id: 'user_1',
        limit: 10,
        scene: 'home',
      })

      await store.submitFeedback('item_1', 'like')

      expect(mockApiProvider.recommend.submitFeedback).toHaveBeenCalledWith({
        user_id: 'current_user',
        item_id: 'item_1',
        action: 'like',
        request_id: 'req_123456',
      })
    })
  })

  describe('clearRecommendations', () => {
    it('应该清空推荐列表和相关状态', async () => {
      vi.mocked(mockApiProvider.recommend.getRecommendations).mockResolvedValue(mockRecommendResponse)
      await store.fetchRecommendations({
        user_id: 'user_1',
        limit: 10,
        scene: 'home',
      })

      store.clearRecommendations()

      expect(store.recommendations).toEqual([])
      expect(store.currentRequestId).toBe('')
      expect(store.strategy).toBe('')
      expect(store.error).toBeNull()
    })
  })

  describe('$reset', () => {
    it('应该重置所有状态到初始值', async () => {
      vi.mocked(mockApiProvider.recommend.getRecommendations).mockResolvedValue(mockRecommendResponse)
      await store.fetchRecommendations({
        user_id: 'user_1',
        limit: 10,
        scene: 'home',
      })
      store.viewedItemIds.add('item_test')

      store.$reset()

      expect(store.recommendations).toEqual([])
      expect(store.isLoading).toBe(false)
      expect(store.error).toBeNull()
      expect(store.currentRequestId).toBe('')
      expect(store.strategy).toBe('')
      expect(store.viewedItemIds.size).toBe(0)
    })
  })
})

