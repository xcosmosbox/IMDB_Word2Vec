/**
 * 用户 API 模块
 */

import { request } from './request'
import type {
  User,
  UpdateUserRequest,
  UserProfile,
  UserBehavior,
  RecordBehaviorRequest,
} from '../types'
import type { IUserService } from './interfaces'

/**
 * 用户服务实现
 */
export class UserService implements IUserService {
  /**
   * 获取用户信息
   */
  async getUser(userId: string): Promise<User> {
    return request.get(`/users/${userId}`)
  }
  
  /**
   * 更新用户信息
   */
  async updateUser(userId: string, data: UpdateUserRequest): Promise<User> {
    return request.put(`/users/${userId}`, data)
  }
  
  /**
   * 获取用户画像
   */
  async getProfile(userId: string): Promise<UserProfile> {
    return request.get(`/users/${userId}/profile`)
  }
  
  /**
   * 获取用户行为历史
   */
  async getBehaviors(userId: string, limit = 50): Promise<UserBehavior[]> {
    return request.get(`/users/${userId}/behaviors`, { params: { limit } })
  }
  
  /**
   * 记录用户行为
   */
  async recordBehavior(data: RecordBehaviorRequest): Promise<void> {
    return request.post('/behaviors', data)
  }
}

// 导出单例
export const userService = new UserService()

// 导出便捷 API
export const userApi = {
  getUser: (userId: string) => userService.getUser(userId),
  updateUser: (userId: string, data: UpdateUserRequest) => userService.updateUser(userId, data),
  getProfile: (userId: string) => userService.getProfile(userId),
  getBehaviors: (userId: string, limit?: number) => userService.getBehaviors(userId, limit),
  recordBehavior: (data: RecordBehaviorRequest) => userService.recordBehavior(data),
}

