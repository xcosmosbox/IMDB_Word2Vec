/**
 * 数据分析 API 接口封装
 * 
 * 实现 IAnalyticsService 接口，提供仪表盘统计、趋势分析等功能
 */

import type { IAnalyticsService } from '@shared/api/interfaces'
import type {
  DashboardStats,
  TimeSeriesPoint,
  CategoryStats,
} from '@shared/types'
import { http } from './http'

/**
 * 推荐统计数据
 */
export interface RecommendationStats {
  totalRecommendations: number
  avgCTR: number
  avgResponseTime: number
  successRate: number
}

/**
 * 热门推荐物品
 */
export interface TopRecommendedItem {
  item_id: string
  title: string
  count: number
  ctr: number
}

/**
 * 用户活跃度分布
 */
export interface UserActivityDistribution {
  hour: number
  count: number
}

/**
 * 物品增长趋势
 */
export interface ItemGrowthTrend {
  date: string
  movie: number
  product: number
  article: number
  video: number
}

/**
 * 分析服务 HTTP 实现
 */
export const analyticsApi: IAnalyticsService & {
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
  /**
   * 获取仪表盘统计数据
   */
  async getDashboardStats(): Promise<DashboardStats> {
    const response = await http.get<DashboardStats>('/api/v1/admin/analytics/dashboard')
    return response.data
  },

  /**
   * 获取用户增长趋势
   * @param days 天数
   */
  async getUserTrend(days: number): Promise<TimeSeriesPoint[]> {
    const response = await http.get<TimeSeriesPoint[]>('/api/v1/admin/analytics/users/trend', {
      params: { days },
    })
    return response.data
  },

  /**
   * 获取物品类型统计
   */
  async getItemTypeStats(): Promise<CategoryStats[]> {
    const response = await http.get<CategoryStats[]>('/api/v1/admin/analytics/items/type-stats')
    return response.data
  },

  /**
   * 获取推荐量趋势
   * @param days 天数
   */
  async getRecommendationTrend(days: number): Promise<TimeSeriesPoint[]> {
    const response = await http.get<TimeSeriesPoint[]>('/api/v1/admin/analytics/recommendations/trend', {
      params: { days },
    })
    return response.data
  },

  /**
   * 获取热门分类
   * @param limit 数量限制
   */
  async getTopCategories(limit: number): Promise<CategoryStats[]> {
    const response = await http.get<CategoryStats[]>('/api/v1/admin/analytics/categories/top', {
      params: { limit },
    })
    return response.data
  },

  /**
   * 获取点击率趋势
   * @param startDate 开始日期
   * @param endDate 结束日期
   */
  async getCTRTrend(startDate: string, endDate: string): Promise<TimeSeriesPoint[]> {
    const response = await http.get<TimeSeriesPoint[]>('/api/v1/admin/analytics/ctr/trend', {
      params: { start_date: startDate, end_date: endDate },
    })
    return response.data
  },

  /**
   * 获取推荐统计数据
   * @param startDate 开始日期
   * @param endDate 结束日期
   */
  async getRecommendationStats(startDate: string, endDate: string): Promise<RecommendationStats> {
    const response = await http.get<RecommendationStats>('/api/v1/admin/analytics/recommendations/stats', {
      params: { start_date: startDate, end_date: endDate },
    })
    return response.data
  },

  /**
   * 获取响应延迟趋势
   * @param startDate 开始日期
   * @param endDate 结束日期
   */
  async getLatencyTrend(startDate: string, endDate: string): Promise<TimeSeriesPoint[]> {
    const response = await http.get<TimeSeriesPoint[]>('/api/v1/admin/analytics/latency/trend', {
      params: { start_date: startDate, end_date: endDate },
    })
    return response.data
  },

  /**
   * 获取热门推荐物品
   * @param limit 数量限制
   */
  async getTopRecommendedItems(limit: number): Promise<TopRecommendedItem[]> {
    const response = await http.get<TopRecommendedItem[]>('/api/v1/admin/analytics/recommendations/top-items', {
      params: { limit },
    })
    return response.data
  },

  /**
   * 获取用户活跃度分布（按小时）
   */
  async getUserActivityDistribution(): Promise<UserActivityDistribution[]> {
    const response = await http.get<UserActivityDistribution[]>('/api/v1/admin/analytics/users/activity-distribution')
    return response.data
  },

  /**
   * 获取用户性别统计
   */
  async getUserGenderStats(): Promise<CategoryStats[]> {
    const response = await http.get<CategoryStats[]>('/api/v1/admin/analytics/users/gender-stats')
    return response.data
  },

  /**
   * 获取用户年龄分布
   */
  async getUserAgeDistribution(): Promise<CategoryStats[]> {
    const response = await http.get<CategoryStats[]>('/api/v1/admin/analytics/users/age-distribution')
    return response.data
  },

  /**
   * 获取物品增长趋势（按类型）
   * @param days 天数
   */
  async getItemGrowthTrend(days: number): Promise<ItemGrowthTrend[]> {
    const response = await http.get<ItemGrowthTrend[]>('/api/v1/admin/analytics/items/growth-trend', {
      params: { days },
    })
    return response.data
  },

  /**
   * 获取物品状态统计
   */
  async getItemStatusStats(): Promise<CategoryStats[]> {
    const response = await http.get<CategoryStats[]>('/api/v1/admin/analytics/items/status-stats')
    return response.data
  },

  /**
   * 获取物品分类统计
   */
  async getItemCategoryStats(): Promise<CategoryStats[]> {
    const response = await http.get<CategoryStats[]>('/api/v1/admin/analytics/items/category-stats')
    return response.data
  },
}

export default analyticsApi

