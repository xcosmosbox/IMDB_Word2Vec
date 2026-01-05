/**
 * 分析 API 模块（管理后台使用）
 */

import { request } from './request'
import type {
  DashboardStats,
  TimeSeriesPoint,
  CategoryStats,
} from '../types'
import type { IAnalyticsService } from './interfaces'

/**
 * 分析服务实现
 */
export class AnalyticsService implements IAnalyticsService {
  /**
   * 获取仪表盘统计
   */
  async getDashboardStats(): Promise<DashboardStats> {
    return request.get('/admin/analytics/dashboard')
  }
  
  /**
   * 获取用户增长趋势
   */
  async getUserTrend(days: number): Promise<TimeSeriesPoint[]> {
    return request.get('/admin/analytics/user-trend', { params: { days } })
  }
  
  /**
   * 获取物品类型统计
   */
  async getItemTypeStats(): Promise<CategoryStats[]> {
    return request.get('/admin/analytics/item-types')
  }
  
  /**
   * 获取推荐量趋势
   */
  async getRecommendationTrend(days: number): Promise<TimeSeriesPoint[]> {
    return request.get('/admin/analytics/recommendation-trend', { params: { days } })
  }
  
  /**
   * 获取热门分类
   */
  async getTopCategories(limit: number): Promise<CategoryStats[]> {
    return request.get('/admin/analytics/top-categories', { params: { limit } })
  }
  
  /**
   * 获取点击率趋势
   */
  async getCTRTrend(startDate: string, endDate: string): Promise<TimeSeriesPoint[]> {
    return request.get('/admin/analytics/ctr-trend', { params: { start: startDate, end: endDate } })
  }
}

// 导出单例
export const analyticsService = new AnalyticsService()

// 导出便捷 API
export const analyticsApi = {
  getDashboardStats: () => analyticsService.getDashboardStats(),
  getUserTrend: (days: number) => analyticsService.getUserTrend(days),
  getItemTypeStats: () => analyticsService.getItemTypeStats(),
  getRecommendationTrend: (days: number) => analyticsService.getRecommendationTrend(days),
  getTopCategories: (limit: number) => analyticsService.getTopCategories(limit),
  getCTRTrend: (startDate: string, endDate: string) => analyticsService.getCTRTrend(startDate, endDate),
}

