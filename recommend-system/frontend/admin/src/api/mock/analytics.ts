/**
 * 数据分析 Mock API 实现
 * 
 * 提供开发和测试环境使用的模拟数据
 */

import type { IAnalyticsService } from '@shared/api/interfaces'
import type {
  DashboardStats,
  TimeSeriesPoint,
  CategoryStats,
} from '@shared/types'
import type {
  RecommendationStats,
  TopRecommendedItem,
  UserActivityDistribution,
  ItemGrowthTrend,
} from '../analytics'

/**
 * 生成时间序列数据
 * @param days 天数
 * @param baseValue 基准值
 * @param variance 波动范围
 */
function generateTimeSeries(days: number, baseValue: number, variance: number): TimeSeriesPoint[] {
  const data: TimeSeriesPoint[] = []
  const now = new Date()
  
  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(now)
    date.setDate(date.getDate() - i)
    
    const randomVariance = (Math.random() - 0.5) * 2 * variance
    const trendFactor = 1 + (days - i) * 0.01 // 轻微上升趋势
    const value = Math.round((baseValue + randomVariance) * trendFactor)
    
    data.push({
      timestamp: date.toISOString().split('T')[0],
      value: Math.max(0, value),
    })
  }
  
  return data
}

/**
 * 模拟延迟
 */
async function delay(ms: number = 300): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * Mock 分析服务实现
 */
export const mockAnalyticsApi: IAnalyticsService & {
  getRecommendationStats: (startDate: string, endDate: string) => Promise<RecommendationStats>
  getLatencyTrend: (startDate: string, endDate: string) => Promise<TimeSeriesPoint[]>
  getTopRecommendedItems: (limit: number) => Promise<TopRecommendedItem[]>
  getUserActivityDistribution: () => Promise<UserActivityDistribution[]>
  getUserGenderStats: () => Promise<CategoryStats[]>
  getUserAgeDistribution: () => Promise<CategoryStats[]>
  getItemGrowthTrend: (days: number) => Promise<ItemGrowthTrend[]>
  getItemStatusStats: () => Promise<CategoryStats[]>
  getItemCategoryStats: () => Promise<CategoryStats[]>
} = {
  async getDashboardStats(): Promise<DashboardStats> {
    await delay()
    return {
      total_users: 125680,
      total_items: 45230,
      total_recommendations: 1520340,
      daily_active_users: 28450,
      recommendation_ctr: 0.0856,
      avg_response_time: 23.5,
    }
  },

  async getUserTrend(days: number): Promise<TimeSeriesPoint[]> {
    await delay()
    return generateTimeSeries(days, 1200, 300)
  },

  async getItemTypeStats(): Promise<CategoryStats[]> {
    await delay()
    return [
      { category: 'movie', count: 15420, percentage: 34.1 },
      { category: 'product', count: 12350, percentage: 27.3 },
      { category: 'article', count: 10280, percentage: 22.7 },
      { category: 'video', count: 7180, percentage: 15.9 },
    ]
  },

  async getRecommendationTrend(days: number): Promise<TimeSeriesPoint[]> {
    await delay()
    return generateTimeSeries(days, 50000, 10000)
  },

  async getTopCategories(limit: number): Promise<CategoryStats[]> {
    await delay()
    const categories = [
      { category: '科幻', count: 8520, percentage: 18.8 },
      { category: '动作', count: 7230, percentage: 16.0 },
      { category: '喜剧', count: 6180, percentage: 13.7 },
      { category: '爱情', count: 5420, percentage: 12.0 },
      { category: '悬疑', count: 4850, percentage: 10.7 },
      { category: '恐怖', count: 3920, percentage: 8.7 },
      { category: '动画', count: 3450, percentage: 7.6 },
      { category: '纪录片', count: 2680, percentage: 5.9 },
      { category: '家庭', count: 1820, percentage: 4.0 },
      { category: '音乐', count: 1160, percentage: 2.6 },
    ]
    return categories.slice(0, limit)
  },

  async getCTRTrend(startDate: string, endDate: string): Promise<TimeSeriesPoint[]> {
    await delay()
    const start = new Date(startDate)
    const end = new Date(endDate)
    const days = Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24)) + 1
    
    const data: TimeSeriesPoint[] = []
    for (let i = 0; i < days; i++) {
      const date = new Date(start)
      date.setDate(date.getDate() + i)
      
      const baseCTR = 0.08
      const variance = (Math.random() - 0.5) * 0.02
      const value = Math.max(0, baseCTR + variance)
      
      data.push({
        timestamp: date.toISOString().split('T')[0],
        value: Number(value.toFixed(4)),
      })
    }
    
    return data
  },

  async getRecommendationStats(_startDate: string, _endDate: string): Promise<RecommendationStats> {
    await delay()
    return {
      totalRecommendations: 1520340,
      avgCTR: 0.0856,
      avgResponseTime: 23.5,
      successRate: 0.9945,
    }
  },

  async getLatencyTrend(startDate: string, endDate: string): Promise<TimeSeriesPoint[]> {
    await delay()
    const start = new Date(startDate)
    const end = new Date(endDate)
    const days = Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24)) + 1
    
    const data: TimeSeriesPoint[] = []
    for (let i = 0; i < days; i++) {
      const date = new Date(start)
      date.setDate(date.getDate() + i)
      
      const baseLatency = 25
      const variance = (Math.random() - 0.5) * 10
      const value = Math.max(10, baseLatency + variance)
      
      data.push({
        timestamp: date.toISOString().split('T')[0],
        value: Number(value.toFixed(1)),
      })
    }
    
    return data
  },

  async getTopRecommendedItems(limit: number): Promise<TopRecommendedItem[]> {
    await delay()
    const items: TopRecommendedItem[] = []
    
    const titles = [
      '肖申克的救赎', '教父', '黑暗骑士', '盗梦空间', '阿甘正传',
      '星际穿越', '泰坦尼克号', '霸王别姬', '千与千寻', '这个杀手不太冷',
      '辛德勒的名单', '美丽人生', '机器人总动员', '飞屋环游记', '疯狂动物城',
      '寻梦环游记', '你的名字', '哈利波特', '复仇者联盟', '蜘蛛侠',
    ]
    
    for (let i = 0; i < Math.min(limit, titles.length); i++) {
      const baseCount = 50000 - i * 2000
      const baseCTR = 0.12 - i * 0.003
      
      items.push({
        item_id: `item_${i + 1}`,
        title: titles[i],
        count: baseCount + Math.floor(Math.random() * 1000),
        ctr: Number((baseCTR + (Math.random() - 0.5) * 0.01).toFixed(4)),
      })
    }
    
    return items
  },

  async getUserActivityDistribution(): Promise<UserActivityDistribution[]> {
    await delay()
    const distribution: UserActivityDistribution[] = []
    
    // 模拟每小时的活跃用户数，有明显的高峰期
    const hourlyPattern = [
      0.2, 0.1, 0.08, 0.05, 0.03, 0.05,   // 0-5点
      0.1, 0.25, 0.4, 0.5, 0.55, 0.6,     // 6-11点
      0.7, 0.65, 0.55, 0.5, 0.55, 0.65,   // 12-17点
      0.75, 0.9, 1.0, 0.95, 0.7, 0.4,     // 18-23点
    ]
    
    const baseCount = 5000
    for (let hour = 0; hour < 24; hour++) {
      distribution.push({
        hour,
        count: Math.round(baseCount * hourlyPattern[hour] + Math.random() * 200),
      })
    }
    
    return distribution
  },

  async getUserGenderStats(): Promise<CategoryStats[]> {
    await delay()
    return [
      { category: 'male', count: 68520, percentage: 54.5 },
      { category: 'female', count: 52160, percentage: 41.5 },
      { category: 'unknown', count: 5000, percentage: 4.0 },
    ]
  },

  async getUserAgeDistribution(): Promise<CategoryStats[]> {
    await delay()
    return [
      { category: '18岁以下', count: 8520, percentage: 6.8 },
      { category: '18-24岁', count: 32450, percentage: 25.8 },
      { category: '25-34岁', count: 45680, percentage: 36.3 },
      { category: '35-44岁', count: 22340, percentage: 17.8 },
      { category: '45-54岁', count: 11280, percentage: 9.0 },
      { category: '55岁以上', count: 5410, percentage: 4.3 },
    ]
  },

  async getItemGrowthTrend(days: number): Promise<ItemGrowthTrend[]> {
    await delay()
    const data: ItemGrowthTrend[] = []
    const now = new Date()
    
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date(now)
      date.setDate(date.getDate() - i)
      
      data.push({
        date: date.toISOString().split('T')[0],
        movie: 20 + Math.floor(Math.random() * 15),
        product: 35 + Math.floor(Math.random() * 20),
        article: 45 + Math.floor(Math.random() * 25),
        video: 15 + Math.floor(Math.random() * 10),
      })
    }
    
    return data
  },

  async getItemStatusStats(): Promise<CategoryStats[]> {
    await delay()
    return [
      { category: 'active', count: 41250, percentage: 91.2 },
      { category: 'inactive', count: 3980, percentage: 8.8 },
    ]
  },

  async getItemCategoryStats(): Promise<CategoryStats[]> {
    await delay()
    return [
      { category: '电子产品', count: 8520, percentage: 18.8 },
      { category: '服装配饰', count: 7230, percentage: 16.0 },
      { category: '图书音像', count: 6180, percentage: 13.7 },
      { category: '家居生活', count: 5420, percentage: 12.0 },
      { category: '食品饮料', count: 4850, percentage: 10.7 },
      { category: '运动户外', count: 3920, percentage: 8.7 },
      { category: '母婴用品', count: 3450, percentage: 7.6 },
      { category: '美妆个护', count: 2680, percentage: 5.9 },
      { category: '数码配件', count: 1820, percentage: 4.0 },
      { category: '其他', count: 1160, percentage: 2.6 },
    ]
  },
}

export default mockAnalyticsApi

