/**
 * API Provider - 依赖注入容器
 * 
 * 实现 IApiProvider 接口，提供所有服务的统一入口
 * 支持 Mock 模式和 HTTP 模式
 */

import type {
  IApiProvider,
  IAuthService,
  IUserService,
  IItemService,
  IRecommendService,
  IAnalyticsService,
  IAdminUserService,
  IAdminItemService,
} from '@shared/api/interfaces'
import { adminUserApi } from './user'
import { adminItemApi } from './item'
import { analyticsApi } from './analytics'
import { mockAnalyticsApi } from './mock/analytics'

/**
 * HTTP API Provider 实现
 * 
 * 生产环境使用，连接真实后端 API
 */
class HttpApiProvider implements Partial<IApiProvider> {
  readonly adminUser: IAdminUserService = adminUserApi
  readonly adminItem: IAdminItemService = adminItemApi
  readonly analytics: IAnalyticsService = analyticsApi
  
  // 其他服务由其他模块实现
  // readonly auth: IAuthService
  // readonly user: IUserService
  // readonly item: IItemService
  // readonly recommend: IRecommendService
}

/**
 * Mock API Provider 实现
 * 
 * 开发/测试环境使用，返回模拟数据
 */
export class MockApiProvider implements Partial<IApiProvider> {
  readonly analytics: IAnalyticsService = mockAnalyticsApi
  
  readonly adminUser: IAdminUserService = {
    async listUsers(params) {
      // 模拟延迟
      await new Promise(resolve => setTimeout(resolve, 300))
      
      const mockUsers = Array.from({ length: 10 }, (_, i) => ({
        id: `user_${i + 1}`,
        name: `用户${i + 1}`,
        email: `user${i + 1}@example.com`,
        age: 20 + i,
        gender: i % 2 === 0 ? 'male' : 'female',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }))
      
      return {
        items: mockUsers.slice((params.page - 1) * params.page_size, params.page * params.page_size),
        total: mockUsers.length,
      }
    },
    async getUser(userId) {
      await new Promise(resolve => setTimeout(resolve, 200))
      return {
        id: userId,
        name: '测试用户',
        email: 'test@example.com',
        age: 25,
        gender: 'male',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }
    },
    async createUser(data) {
      await new Promise(resolve => setTimeout(resolve, 300))
      return {
        id: `user_${Date.now()}`,
        ...data,
        age: data.age || 0,
        gender: data.gender || 'other',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }
    },
    async updateUser(userId, data) {
      await new Promise(resolve => setTimeout(resolve, 300))
      return {
        id: userId,
        name: data.name || '测试用户',
        email: data.email || 'test@example.com',
        age: data.age || 25,
        gender: data.gender || 'male',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }
    },
    async deleteUser() {
      await new Promise(resolve => setTimeout(resolve, 200))
    },
  }

  readonly adminItem: IAdminItemService = {
    async listItems(params) {
      await new Promise(resolve => setTimeout(resolve, 300))
      
      const mockItems = Array.from({ length: 20 }, (_, i) => ({
        id: `item_${i + 1}`,
        type: (['movie', 'product', 'article', 'video'] as const)[i % 4],
        title: `物品标题 ${i + 1}`,
        description: `这是物品 ${i + 1} 的描述信息`,
        category: `分类${(i % 5) + 1}`,
        tags: [`标签${i % 3 + 1}`, `标签${i % 4 + 1}`],
        status: (i % 3 === 0 ? 'inactive' : 'active') as 'active' | 'inactive',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }))
      
      return {
        items: mockItems.slice((params.page - 1) * params.page_size, params.page * params.page_size),
        total: mockItems.length,
      }
    },
    async getItem(itemId) {
      await new Promise(resolve => setTimeout(resolve, 200))
      return {
        id: itemId,
        type: 'movie',
        title: '测试物品',
        description: '这是测试物品的描述',
        category: '测试分类',
        tags: ['标签1', '标签2'],
        status: 'active',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }
    },
    async createItem(data) {
      await new Promise(resolve => setTimeout(resolve, 300))
      return {
        id: `item_${Date.now()}`,
        ...data,
        description: data.description || '',
        category: data.category || '',
        tags: data.tags || [],
        status: 'active',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }
    },
    async updateItem(itemId, data) {
      await new Promise(resolve => setTimeout(resolve, 300))
      return {
        id: itemId,
        type: 'movie',
        title: data.title || '测试物品',
        description: data.description || '',
        category: data.category || '',
        tags: data.tags || [],
        status: 'active',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }
    },
    async deleteItem() {
      await new Promise(resolve => setTimeout(resolve, 200))
    },
  }
}

// 默认导出 HTTP Provider
export const apiProvider = new HttpApiProvider()

// 导出 Mock Provider 用于测试
export const mockApiProvider = new MockApiProvider()

/**
 * 创建 API Provider
 * 
 * @param useMock - 是否使用 Mock 模式
 */
export function createApiProvider(useMock = false): Partial<IApiProvider> {
  return useMock ? new MockApiProvider() : new HttpApiProvider()
}

