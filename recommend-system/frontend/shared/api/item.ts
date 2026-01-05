/**
 * 物品 API 模块
 */

import { request } from './request'
import type {
  Item,
  ListItemsRequest,
  ListItemsResponse,
  ItemStats,
  SimilarItem,
} from '../types'
import type { IItemService } from './interfaces'

/**
 * 物品服务实现
 */
export class ItemService implements IItemService {
  /**
   * 获取物品详情
   */
  async getItem(itemId: string): Promise<Item> {
    return request.get(`/items/${itemId}`)
  }
  
  /**
   * 搜索物品
   */
  async searchItems(query: string, limit = 20): Promise<Item[]> {
    return request.get('/items/search', { params: { q: query, limit } })
  }
  
  /**
   * 列出物品
   */
  async listItems(params: ListItemsRequest): Promise<ListItemsResponse> {
    return request.get('/items', { params })
  }
  
  /**
   * 获取物品统计
   */
  async getItemStats(itemId: string): Promise<ItemStats> {
    return request.get(`/items/${itemId}/stats`)
  }
  
  /**
   * 获取相似物品
   */
  async getSimilarItems(itemId: string, limit = 10): Promise<SimilarItem[]> {
    return request.get(`/items/${itemId}/similar`, { params: { limit } })
  }
}

// 导出单例
export const itemService = new ItemService()

// 导出便捷 API
export const itemApi = {
  getItem: (itemId: string) => itemService.getItem(itemId),
  searchItems: (query: string, limit?: number) => itemService.searchItems(query, limit),
  listItems: (params: ListItemsRequest) => itemService.listItems(params),
  getItemStats: (itemId: string) => itemService.getItemStats(itemId),
  getSimilarItems: (itemId: string, limit?: number) => itemService.getSimilarItems(itemId, limit),
}

