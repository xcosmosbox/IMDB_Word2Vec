/**
 * Analytics API 单元测试
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mockAnalyticsApi } from '../mock/analytics'
import type {
  DashboardStats,
  TimeSeriesPoint,
  CategoryStats,
} from '@shared/types'

describe('Mock Analytics API', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2025-01-05'))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('getDashboardStats', () => {
    it('应返回仪表盘统计数据', async () => {
      const stats = await mockAnalyticsApi.getDashboardStats()

      expect(stats).toBeDefined()
      expect(stats.total_users).toBeGreaterThan(0)
      expect(stats.total_items).toBeGreaterThan(0)
      expect(stats.total_recommendations).toBeGreaterThan(0)
      expect(stats.daily_active_users).toBeGreaterThan(0)
      expect(stats.recommendation_ctr).toBeGreaterThan(0)
      expect(stats.recommendation_ctr).toBeLessThan(1)
      expect(stats.avg_response_time).toBeGreaterThan(0)
    })

    it('应返回正确的数据类型', async () => {
      const stats = await mockAnalyticsApi.getDashboardStats()

      expect(typeof stats.total_users).toBe('number')
      expect(typeof stats.total_items).toBe('number')
      expect(typeof stats.total_recommendations).toBe('number')
      expect(typeof stats.daily_active_users).toBe('number')
      expect(typeof stats.recommendation_ctr).toBe('number')
      expect(typeof stats.avg_response_time).toBe('number')
    })
  })

  describe('getUserTrend', () => {
    it('应返回指定天数的用户趋势数据', async () => {
      const days = 30
      const trend = await mockAnalyticsApi.getUserTrend(days)

      expect(trend).toHaveLength(days)
    })

    it('应返回正确格式的时间序列数据', async () => {
      const trend = await mockAnalyticsApi.getUserTrend(7)

      trend.forEach((point: TimeSeriesPoint) => {
        expect(point.timestamp).toBeDefined()
        expect(point.value).toBeDefined()
        expect(typeof point.timestamp).toBe('string')
        expect(typeof point.value).toBe('number')
        expect(point.value).toBeGreaterThanOrEqual(0)
      })
    })

    it('应返回按时间排序的数据', async () => {
      const trend = await mockAnalyticsApi.getUserTrend(7)

      for (let i = 1; i < trend.length; i++) {
        const prevDate = new Date(trend[i - 1].timestamp)
        const currDate = new Date(trend[i].timestamp)
        expect(currDate.getTime()).toBeGreaterThan(prevDate.getTime())
      }
    })
  })

  describe('getItemTypeStats', () => {
    it('应返回物品类型统计', async () => {
      const stats = await mockAnalyticsApi.getItemTypeStats()

      expect(stats).toBeDefined()
      expect(Array.isArray(stats)).toBe(true)
      expect(stats.length).toBeGreaterThan(0)
    })

    it('应包含所有物品类型', async () => {
      const stats = await mockAnalyticsApi.getItemTypeStats()
      const categories = stats.map((s: CategoryStats) => s.category)

      expect(categories).toContain('movie')
      expect(categories).toContain('product')
      expect(categories).toContain('article')
      expect(categories).toContain('video')
    })

    it('百分比总和应约等于100', async () => {
      const stats = await mockAnalyticsApi.getItemTypeStats()
      const totalPercentage = stats.reduce(
        (sum: number, s: CategoryStats) => sum + s.percentage, 
        0
      )

      expect(Math.abs(totalPercentage - 100)).toBeLessThan(1)
    })
  })

  describe('getRecommendationTrend', () => {
    it('应返回推荐趋势数据', async () => {
      const trend = await mockAnalyticsApi.getRecommendationTrend(30)

      expect(trend).toHaveLength(30)
      trend.forEach((point: TimeSeriesPoint) => {
        expect(point.timestamp).toBeDefined()
        expect(point.value).toBeGreaterThan(0)
      })
    })
  })

  describe('getTopCategories', () => {
    it('应返回指定数量的热门分类', async () => {
      const limit = 5
      const categories = await mockAnalyticsApi.getTopCategories(limit)

      expect(categories).toHaveLength(limit)
    })

    it('应按数量降序排列', async () => {
      const categories = await mockAnalyticsApi.getTopCategories(10)

      for (let i = 1; i < categories.length; i++) {
        expect(categories[i - 1].count).toBeGreaterThanOrEqual(categories[i].count)
      }
    })
  })

  describe('getCTRTrend', () => {
    it('应返回指定日期范围的CTR趋势', async () => {
      const startDate = '2025-01-01'
      const endDate = '2025-01-05'
      const trend = await mockAnalyticsApi.getCTRTrend(startDate, endDate)

      expect(trend).toHaveLength(5)
    })

    it('CTR值应在合理范围内', async () => {
      const trend = await mockAnalyticsApi.getCTRTrend('2025-01-01', '2025-01-05')

      trend.forEach((point: TimeSeriesPoint) => {
        expect(point.value).toBeGreaterThan(0)
        expect(point.value).toBeLessThan(1)
      })
    })
  })

  describe('getRecommendationStats', () => {
    it('应返回推荐统计数据', async () => {
      const stats = await mockAnalyticsApi.getRecommendationStats('2025-01-01', '2025-01-05')

      expect(stats.totalRecommendations).toBeGreaterThan(0)
      expect(stats.avgCTR).toBeGreaterThan(0)
      expect(stats.avgCTR).toBeLessThan(1)
      expect(stats.avgResponseTime).toBeGreaterThan(0)
      expect(stats.successRate).toBeGreaterThan(0.9)
    })
  })

  describe('getLatencyTrend', () => {
    it('应返回响应延迟趋势', async () => {
      const trend = await mockAnalyticsApi.getLatencyTrend('2025-01-01', '2025-01-05')

      expect(trend).toHaveLength(5)
      trend.forEach((point: TimeSeriesPoint) => {
        expect(point.value).toBeGreaterThan(0)
        expect(point.value).toBeLessThan(100) // 合理的响应时间
      })
    })
  })

  describe('getTopRecommendedItems', () => {
    it('应返回指定数量的热门推荐物品', async () => {
      const limit = 10
      const items = await mockAnalyticsApi.getTopRecommendedItems(limit)

      expect(items).toHaveLength(limit)
    })

    it('应包含必要的字段', async () => {
      const items = await mockAnalyticsApi.getTopRecommendedItems(5)

      items.forEach(item => {
        expect(item.item_id).toBeDefined()
        expect(item.title).toBeDefined()
        expect(item.count).toBeGreaterThan(0)
        expect(item.ctr).toBeGreaterThan(0)
        expect(item.ctr).toBeLessThan(1)
      })
    })
  })

  describe('getUserActivityDistribution', () => {
    it('应返回24小时的活跃度分布', async () => {
      const distribution = await mockAnalyticsApi.getUserActivityDistribution()

      expect(distribution).toHaveLength(24)
    })

    it('应包含正确的小时数据', async () => {
      const distribution = await mockAnalyticsApi.getUserActivityDistribution()

      distribution.forEach((item, index) => {
        expect(item.hour).toBe(index)
        expect(item.count).toBeGreaterThanOrEqual(0)
      })
    })
  })

  describe('getUserGenderStats', () => {
    it('应返回性别统计数据', async () => {
      const stats = await mockAnalyticsApi.getUserGenderStats()

      expect(stats.length).toBeGreaterThan(0)
      const categories = stats.map(s => s.category)
      expect(categories).toContain('male')
      expect(categories).toContain('female')
    })
  })

  describe('getUserAgeDistribution', () => {
    it('应返回年龄分布数据', async () => {
      const distribution = await mockAnalyticsApi.getUserAgeDistribution()

      expect(distribution.length).toBeGreaterThan(0)
      distribution.forEach(item => {
        expect(item.category).toBeDefined()
        expect(item.count).toBeGreaterThan(0)
        expect(item.percentage).toBeGreaterThan(0)
      })
    })
  })

  describe('getItemGrowthTrend', () => {
    it('应返回物品增长趋势', async () => {
      const trend = await mockAnalyticsApi.getItemGrowthTrend(7)

      expect(trend).toHaveLength(7)
      trend.forEach(item => {
        expect(item.date).toBeDefined()
        expect(item.movie).toBeGreaterThanOrEqual(0)
        expect(item.product).toBeGreaterThanOrEqual(0)
        expect(item.article).toBeGreaterThanOrEqual(0)
        expect(item.video).toBeGreaterThanOrEqual(0)
      })
    })
  })

  describe('getItemStatusStats', () => {
    it('应返回物品状态统计', async () => {
      const stats = await mockAnalyticsApi.getItemStatusStats()

      expect(stats.length).toBe(2)
      const categories = stats.map(s => s.category)
      expect(categories).toContain('active')
      expect(categories).toContain('inactive')
    })
  })

  describe('getItemCategoryStats', () => {
    it('应返回物品分类统计', async () => {
      const stats = await mockAnalyticsApi.getItemCategoryStats()

      expect(stats.length).toBeGreaterThan(0)
      stats.forEach(item => {
        expect(item.category).toBeDefined()
        expect(item.count).toBeGreaterThan(0)
        expect(item.percentage).toBeGreaterThan(0)
      })
    })
  })
})

