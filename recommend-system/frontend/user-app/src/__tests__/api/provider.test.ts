/**
 * API Provider 单元测试
 * 
 * @author Person F
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import {
  HttpApiProvider,
  MockApiProvider,
  getApiProvider,
  setApiProvider,
  resetApiProvider,
} from '@shared/api'

describe('API Provider', () => {
  beforeEach(() => {
    // 重置单例
    resetApiProvider()
  })

  describe('HttpApiProvider', () => {
    it('应该创建所有服务实例', () => {
      const provider = new HttpApiProvider()

      expect(provider.auth).toBeDefined()
      expect(provider.user).toBeDefined()
      expect(provider.item).toBeDefined()
      expect(provider.recommend).toBeDefined()
      expect(provider.analytics).toBeDefined()
      expect(provider.adminUser).toBeDefined()
      expect(provider.adminItem).toBeDefined()
    })
  })

  describe('MockApiProvider', () => {
    it('应该创建所有服务实例', () => {
      const provider = new MockApiProvider()

      expect(provider.auth).toBeDefined()
      expect(provider.user).toBeDefined()
      expect(provider.item).toBeDefined()
      expect(provider.recommend).toBeDefined()
      expect(provider.analytics).toBeDefined()
      expect(provider.adminUser).toBeDefined()
      expect(provider.adminItem).toBeDefined()
    })

    describe('auth 服务', () => {
      it('应该能够登录', async () => {
        const provider = new MockApiProvider()

        const result = await provider.auth.login({
          email: 'zhangsan@example.com',
          password: '123456',
        })

        expect(result.token).toBeDefined()
        expect(result.user).toBeDefined()
        expect(result.user.email).toBe('zhangsan@example.com')
      })

      it('登录不存在的用户应该抛出错误', async () => {
        const provider = new MockApiProvider()

        await expect(
          provider.auth.login({
            email: 'notexist@example.com',
            password: '123456',
          })
        ).rejects.toThrow('用户不存在')
      })

      it('应该能够注册新用户', async () => {
        const provider = new MockApiProvider()

        await expect(
          provider.auth.register({
            name: '测试用户',
            email: 'test@example.com',
            password: '123456',
          })
        ).resolves.toBeUndefined()
      })

      it('注册已存在的邮箱应该抛出错误', async () => {
        const provider = new MockApiProvider()

        await expect(
          provider.auth.register({
            name: '张三',
            email: 'zhangsan@example.com',
            password: '123456',
          })
        ).rejects.toThrow('邮箱已被注册')
      })
    })

    describe('user 服务', () => {
      it('应该能够获取用户信息', async () => {
        const provider = new MockApiProvider()

        const user = await provider.user.getUser('user_001')

        expect(user.id).toBe('user_001')
        expect(user.name).toBe('张三')
      })

      it('获取不存在的用户应该抛出错误', async () => {
        const provider = new MockApiProvider()

        await expect(provider.user.getUser('not_exist')).rejects.toThrow('用户不存在')
      })

      it('应该能够获取用户画像', async () => {
        const provider = new MockApiProvider()

        const profile = await provider.user.getProfile('user_001')

        expect(profile.user.id).toBe('user_001')
        expect(profile.preferred_types).toBeDefined()
        expect(profile.active_hours).toBeDefined()
      })

      it('应该能够获取用户行为历史', async () => {
        const provider = new MockApiProvider()

        const behaviors = await provider.user.getBehaviors('user_001', 10)

        expect(Array.isArray(behaviors)).toBe(true)
        expect(behaviors.length).toBeLessThanOrEqual(10)
      })
    })

    describe('item 服务', () => {
      it('应该能够获取物品详情', async () => {
        const provider = new MockApiProvider()

        const item = await provider.item.getItem('item_001')

        expect(item.id).toBe('item_001')
        expect(item.title).toBe('肖申克的救赎')
      })

      it('应该能够搜索物品', async () => {
        const provider = new MockApiProvider()

        const items = await provider.item.searchItems('肖申克')

        expect(items.length).toBeGreaterThan(0)
        expect(items[0].title).toContain('肖申克')
      })

      it('应该能够列出物品', async () => {
        const provider = new MockApiProvider()

        const response = await provider.item.listItems({ page: 1, page_size: 5 })

        expect(response.items.length).toBeLessThanOrEqual(5)
        expect(response.total).toBeGreaterThan(0)
        expect(response.page).toBe(1)
      })

      it('应该能够按类型筛选物品', async () => {
        const provider = new MockApiProvider()

        const response = await provider.item.listItems({ type: 'movie', page: 1, page_size: 10 })

        response.items.forEach(item => {
          expect(item.type).toBe('movie')
        })
      })
    })

    describe('recommend 服务', () => {
      it('应该能够获取推荐列表', async () => {
        const provider = new MockApiProvider()

        const response = await provider.recommend.getRecommendations({
          user_id: 'user_001',
          limit: 5,
        })

        expect(response.recommendations.length).toBeLessThanOrEqual(5)
        expect(response.request_id).toBeDefined()
        expect(response.strategy).toBeDefined()
      })

      it('应该能够排除指定物品', async () => {
        const provider = new MockApiProvider()

        const response = await provider.recommend.getRecommendations({
          user_id: 'user_001',
          limit: 10,
          exclude_items: ['item_001'],
        })

        const itemIds = response.recommendations.map(r => r.item_id)
        expect(itemIds).not.toContain('item_001')
      })

      it('应该能够获取相似推荐', async () => {
        const provider = new MockApiProvider()

        const recommendations = await provider.recommend.getSimilarRecommendations('item_001', 5)

        expect(recommendations.length).toBeLessThanOrEqual(5)
        const itemIds = recommendations.map(r => r.item_id)
        expect(itemIds).not.toContain('item_001')
      })
    })

    describe('analytics 服务', () => {
      it('应该能够获取仪表盘统计', async () => {
        const provider = new MockApiProvider()

        const stats = await provider.analytics.getDashboardStats()

        expect(stats.total_users).toBeGreaterThan(0)
        expect(stats.total_items).toBeGreaterThan(0)
        expect(stats.total_recommendations).toBeGreaterThan(0)
      })

      it('应该能够获取用户增长趋势', async () => {
        const provider = new MockApiProvider()

        const trend = await provider.analytics.getUserTrend(7)

        expect(trend.length).toBe(7)
        trend.forEach(point => {
          expect(point.timestamp).toBeDefined()
          expect(point.value).toBeGreaterThanOrEqual(0)
        })
      })
    })
  })

  describe('getApiProvider', () => {
    it('应该返回单例', () => {
      const provider1 = getApiProvider()
      const provider2 = getApiProvider()

      expect(provider1).toBe(provider2)
    })

    it('传入 true 应该返回 MockApiProvider', () => {
      const provider = getApiProvider(true)

      expect(provider).toBeInstanceOf(MockApiProvider)
    })

    it('传入 false 应该返回 HttpApiProvider', () => {
      const provider = getApiProvider(false)

      expect(provider).toBeInstanceOf(HttpApiProvider)
    })
  })

  describe('setApiProvider', () => {
    it('应该能够设置自定义 Provider', () => {
      const customProvider = new MockApiProvider()
      setApiProvider(customProvider)

      const provider = getApiProvider()

      expect(provider).toBe(customProvider)
    })
  })

  describe('resetApiProvider', () => {
    it('应该重置单例', () => {
      const provider1 = getApiProvider()
      resetApiProvider()
      const provider2 = getApiProvider()

      expect(provider1).not.toBe(provider2)
    })
  })
})

