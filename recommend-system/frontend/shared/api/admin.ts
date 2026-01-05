/**
 * 管理员 API 模块
 * 
 * 提供管理后台的用户和物品 CRUD 操作
 * 
 * @module shared/api/admin
 * @author Person F
 */

import { request } from './request'
import type {
  User,
  CreateUserRequest,
  UpdateUserRequest,
  Item,
  CreateItemRequest,
  UpdateItemRequest,
} from '../types'
import type { IAdminUserService, IAdminItemService } from './interfaces'

// =============================================================================
// 管理员用户服务
// =============================================================================

/**
 * 管理员用户服务实现
 */
export class AdminUserService implements IAdminUserService {
  /**
   * 列出用户
   */
  async listUsers(params: {
    page: number
    page_size: number
    keyword?: string
    gender?: string
  }): Promise<{ items: User[]; total: number }> {
    return request.get('/admin/users', { params })
  }

  /**
   * 获取用户
   */
  async getUser(userId: string): Promise<User> {
    return request.get(`/admin/users/${userId}`)
  }

  /**
   * 创建用户
   */
  async createUser(data: CreateUserRequest): Promise<User> {
    return request.post('/admin/users', data)
  }

  /**
   * 更新用户
   */
  async updateUser(userId: string, data: UpdateUserRequest): Promise<User> {
    return request.put(`/admin/users/${userId}`, data)
  }

  /**
   * 删除用户
   */
  async deleteUser(userId: string): Promise<void> {
    return request.delete(`/admin/users/${userId}`)
  }
}

// =============================================================================
// 管理员物品服务
// =============================================================================

/**
 * 管理员物品服务实现
 */
export class AdminItemService implements IAdminItemService {
  /**
   * 列出物品
   */
  async listItems(params: {
    page: number
    page_size: number
    type?: string
    keyword?: string
  }): Promise<{ items: Item[]; total: number }> {
    return request.get('/admin/items', { params })
  }

  /**
   * 获取物品
   */
  async getItem(itemId: string): Promise<Item> {
    return request.get(`/admin/items/${itemId}`)
  }

  /**
   * 创建物品
   */
  async createItem(data: CreateItemRequest): Promise<Item> {
    return request.post('/admin/items', data)
  }

  /**
   * 更新物品
   */
  async updateItem(itemId: string, data: UpdateItemRequest): Promise<Item> {
    return request.put(`/admin/items/${itemId}`, data)
  }

  /**
   * 删除物品
   */
  async deleteItem(itemId: string): Promise<void> {
    return request.delete(`/admin/items/${itemId}`)
  }
}

// =============================================================================
// 导出单例
// =============================================================================

export const adminUserService = new AdminUserService()
export const adminItemService = new AdminItemService()

// =============================================================================
// 导出便捷 API
// =============================================================================

export const adminUserApi = {
  listUsers: (params: Parameters<AdminUserService['listUsers']>[0]) => 
    adminUserService.listUsers(params),
  getUser: (userId: string) => adminUserService.getUser(userId),
  createUser: (data: CreateUserRequest) => adminUserService.createUser(data),
  updateUser: (userId: string, data: UpdateUserRequest) => 
    adminUserService.updateUser(userId, data),
  deleteUser: (userId: string) => adminUserService.deleteUser(userId),
}

export const adminItemApi = {
  listItems: (params: Parameters<AdminItemService['listItems']>[0]) => 
    adminItemService.listItems(params),
  getItem: (itemId: string) => adminItemService.getItem(itemId),
  createItem: (data: CreateItemRequest) => adminItemService.createItem(data),
  updateItem: (itemId: string, data: UpdateItemRequest) => 
    adminItemService.updateItem(itemId, data),
  deleteItem: (itemId: string) => adminItemService.deleteItem(itemId),
}

