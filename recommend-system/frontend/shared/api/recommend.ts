/**
 * 推荐 API 模块
 */

import { request } from './request'
import type {
  RecommendRequest,
  RecommendResponse,
  FeedbackRequest,
  Recommendation,
} from '../types'
import type { IRecommendService } from './interfaces'

/**
 * 推荐服务实现
 */
export class RecommendService implements IRecommendService {
  /**
   * 获取推荐列表
   */
  async getRecommendations(req: RecommendRequest): Promise<RecommendResponse> {
    return request.post('/recommend', req)
  }
  
  /**
   * 提交反馈
   */
  async submitFeedback(feedback: FeedbackRequest): Promise<void> {
    return request.post('/feedback', feedback)
  }
  
  /**
   * 获取相似推荐
   */
  async getSimilarRecommendations(itemId: string, limit = 10): Promise<Recommendation[]> {
    return request.get(`/recommend/similar/${itemId}`, { params: { limit } })
  }
}

// 导出单例
export const recommendService = new RecommendService()

// 导出便捷 API
export const recommendApi = {
  getRecommendations: (req: RecommendRequest) => recommendService.getRecommendations(req),
  submitFeedback: (feedback: FeedbackRequest) => recommendService.submitFeedback(feedback),
  getSimilarRecommendations: (itemId: string, limit?: number) => 
    recommendService.getSimilarRecommendations(itemId, limit),
}

