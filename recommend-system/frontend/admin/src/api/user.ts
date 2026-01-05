/**
 * 管理员用户服务 API 封装
 * 
 * 实现 IAdminUserService 接口
 */

import type { IAdminUserService } from '@shared/api/interfaces'
import type { User, CreateUserRequest, UpdateUserRequest } from '@shared/types'
import { http } from './http'

/**
 * 用户列表查询参数
 */
export interface ListUsersParams {
  page: number
  page_size: number
  keyword?: string
  gender?: string
}

/**
 * 用户列表响应
 */
export interface ListUsersResponse {
  items: User[]
  total: number
}

/**
 * 管理员用户服务实现
 */
class AdminUserService implements IAdminUserService {
  /**
   * 列出用户
   */
  async listUsers(params: ListUsersParams): Promise<ListUsersResponse> {
    const response = await http.get<ListUsersResponse>('/admin/users', { params })
    return response.data
  }

  /**
   * 获取用户详情
   */
  async getUser(userId: string): Promise<User> {
    const response = await http.get<User>(`/admin/users/${userId}`)
    return response.data
  }

  /**
   * 创建用户
   */
  async createUser(data: CreateUserRequest): Promise<User> {
    const response = await http.post<User>('/admin/users', data)
    return response.data
  }

  /**
   * 更新用户
   */
  async updateUser(userId: string, data: UpdateUserRequest): Promise<User> {
    const response = await http.put<User>(`/admin/users/${userId}`, data)
    return response.data
  }

  /**
   * 删除用户
   */
  async deleteUser(userId: string): Promise<void> {
    await http.delete(`/admin/users/${userId}`)
  }
}

// 导出单例实例
export const adminUserApi = new AdminUserService()

