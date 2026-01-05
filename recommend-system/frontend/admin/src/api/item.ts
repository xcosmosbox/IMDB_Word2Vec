/**
 * 管理员物品服务 API 封装
 * 
 * 实现 IAdminItemService 接口
 */

import type { IAdminItemService } from '@shared/api/interfaces'
import type { Item, CreateItemRequest, UpdateItemRequest } from '@shared/types'
import { http } from './http'

/**
 * 物品列表查询参数
 */
export interface ListItemsParams {
  page: number
  page_size: number
  type?: string
  keyword?: string
  status?: string
}

/**
 * 物品列表响应
 */
export interface ListItemsResponse {
  items: Item[]
  total: number
}

/**
 * 管理员物品服务实现
 */
class AdminItemService implements IAdminItemService {
  /**
   * 列出物品
   */
  async listItems(params: ListItemsParams): Promise<ListItemsResponse> {
    const response = await http.get<ListItemsResponse>('/admin/items', { params })
    return response.data
  }

  /**
   * 获取物品详情
   */
  async getItem(itemId: string): Promise<Item> {
    const response = await http.get<Item>(`/admin/items/${itemId}`)
    return response.data
  }

  /**
   * 创建物品
   */
  async createItem(data: CreateItemRequest): Promise<Item> {
    const response = await http.post<Item>('/admin/items', data)
    return response.data
  }

  /**
   * 更新物品
   */
  async updateItem(itemId: string, data: UpdateItemRequest): Promise<Item> {
    const response = await http.put<Item>(`/admin/items/${itemId}`, data)
    return response.data
  }

  /**
   * 删除物品
   */
  async deleteItem(itemId: string): Promise<void> {
    await http.delete(`/admin/items/${itemId}`)
  }
}

// 导出单例实例
export const adminItemApi = new AdminItemService()

