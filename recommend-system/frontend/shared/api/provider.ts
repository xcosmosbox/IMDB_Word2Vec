/**
 * API Provider 实现
 * 
 * 提供两种实现：
 * 1. HttpApiProvider - 生产环境，调用真实 API
 * 2. MockApiProvider - 开发/测试环境，返回模拟数据
 * 
 * @module shared/api/provider
 * @author Person F
 */

import type { IApiProvider } from './interfaces'
import { AuthService } from './auth'
import { UserService } from './user'
import { ItemService } from './item'
import { RecommendService } from './recommend'
import { AnalyticsService } from './analytics'
import { AdminUserService, AdminItemService } from './admin'

// =============================================================================
// HTTP API Provider - 生产环境实现
// =============================================================================

/**
 * HTTP API Provider
 * 
 * 生产环境使用，调用真实后端 API
 * 
 * @example
 * ```typescript
 * const api = new HttpApiProvider()
 * const user = await api.user.getUser('123')
 * ```
 */
export class HttpApiProvider implements IApiProvider {
  private _auth: AuthService
  private _user: UserService
  private _item: ItemService
  private _recommend: RecommendService
  private _analytics: AnalyticsService
  private _adminUser: AdminUserService
  private _adminItem: AdminItemService

  constructor() {
    this._auth = new AuthService()
    this._user = new UserService()
    this._item = new ItemService()
    this._recommend = new RecommendService()
    this._analytics = new AnalyticsService()
    this._adminUser = new AdminUserService()
    this._adminItem = new AdminItemService()
  }

  get auth() {
    return this._auth
  }

  get user() {
    return this._user
  }

  get item() {
    return this._item
  }

  get recommend() {
    return this._recommend
  }

  get analytics() {
    return this._analytics
  }

  get adminUser() {
    return this._adminUser
  }

  get adminItem() {
    return this._adminItem
  }
}

// =============================================================================
// 单例实例导出
// =============================================================================

let apiProviderInstance: IApiProvider | null = null

/**
 * 获取 API Provider 实例
 * 
 * @param useMock - 是否使用 Mock 实现
 * @returns API Provider 实例
 */
export function getApiProvider(useMock = false): IApiProvider {
  if (!apiProviderInstance) {
    apiProviderInstance = useMock ? new MockApiProvider() : new HttpApiProvider()
  }
  return apiProviderInstance
}

/**
 * 设置 API Provider 实例（用于测试）
 * 
 * @param provider - 自定义 Provider 实例
 */
export function setApiProvider(provider: IApiProvider): void {
  apiProviderInstance = provider
}

/**
 * 重置 API Provider 实例（用于测试）
 */
export function resetApiProvider(): void {
  apiProviderInstance = null
}

// =============================================================================
// Mock API Provider - 开发/测试环境实现
// =============================================================================

import type {
  User,
  CreateUserRequest,
  UpdateUserRequest,
  LoginRequest,
  LoginResponse,
  RegisterRequest,
  UserProfile,
  UserBehavior,
  RecordBehaviorRequest,
  Item,
  CreateItemRequest,
  UpdateItemRequest,
  ListItemsRequest,
  ListItemsResponse,
  ItemStats,
  SimilarItem,
  RecommendRequest,
  RecommendResponse,
  FeedbackRequest,
  Recommendation,
  DashboardStats,
  TimeSeriesPoint,
  CategoryStats,
} from '../types'

import type {
  IAuthService,
  IUserService,
  IItemService,
  IRecommendService,
  IAnalyticsService,
  IAdminUserService,
  IAdminItemService,
} from './interfaces'

// =============================================================================
// Mock 数据生成器
// =============================================================================

/**
 * 生成随机 ID
 */
function generateId(): string {
  return Math.random().toString(36).substring(2, 15)
}

/**
 * 模拟延迟
 */
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * 生成随机日期字符串
 */
function randomDate(start: Date = new Date(2023, 0, 1), end: Date = new Date()): string {
  const date = new Date(start.getTime() + Math.random() * (end.getTime() - start.getTime()))
  return date.toISOString()
}

// =============================================================================
// Mock 用户数据
// =============================================================================

const mockUsers: User[] = [
  {
    id: 'user_001',
    name: '张三',
    email: 'zhangsan@example.com',
    age: 28,
    gender: 'male',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  {
    id: 'user_002',
    name: '李四',
    email: 'lisi@example.com',
    age: 32,
    gender: 'female',
    created_at: '2024-01-02T00:00:00Z',
    updated_at: '2024-01-02T00:00:00Z',
  },
  {
    id: 'user_003',
    name: '王五',
    email: 'wangwu@example.com',
    age: 25,
    gender: 'male',
    created_at: '2024-01-03T00:00:00Z',
    updated_at: '2024-01-03T00:00:00Z',
  },
]

// =============================================================================
// Mock 物品数据
// =============================================================================

const mockItems: Item[] = [
  {
    id: 'item_001',
    type: 'movie',
    title: '肖申克的救赎',
    description: '一部关于希望和自由的经典电影',
    category: '剧情',
    tags: ['经典', '监狱', '励志'],
    status: 'active',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    metadata: {
      year: 1994,
      director: '弗兰克·德拉邦特',
      rating: 9.7,
      poster: 'https://via.placeholder.com/300x450?text=Shawshank',
    },
  },
  {
    id: 'item_002',
    type: 'movie',
    title: '霸王别姬',
    description: '关于京剧演员的传奇人生',
    category: '剧情',
    tags: ['经典', '京剧', '历史'],
    status: 'active',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    metadata: {
      year: 1993,
      director: '陈凯歌',
      rating: 9.6,
      poster: 'https://via.placeholder.com/300x450?text=Farewell',
    },
  },
  {
    id: 'item_003',
    type: 'movie',
    title: '阿甘正传',
    description: '一个智商不高但心地善良的人的传奇人生',
    category: '剧情',
    tags: ['励志', '美国', '历史'],
    status: 'active',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    metadata: {
      year: 1994,
      director: '罗伯特·泽米吉斯',
      rating: 9.5,
      poster: 'https://via.placeholder.com/300x450?text=ForrestGump',
    },
  },
  {
    id: 'item_004',
    type: 'movie',
    title: '泰坦尼克号',
    description: '一段发生在豪华邮轮上的爱情故事',
    category: '爱情',
    tags: ['爱情', '灾难', '经典'],
    status: 'active',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    metadata: {
      year: 1997,
      director: '詹姆斯·卡梅隆',
      rating: 9.4,
      poster: 'https://via.placeholder.com/300x450?text=Titanic',
    },
  },
  {
    id: 'item_005',
    type: 'movie',
    title: '千与千寻',
    description: '宫崎骏的经典动画作品',
    category: '动画',
    tags: ['动画', '奇幻', '日本'],
    status: 'active',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    metadata: {
      year: 2001,
      director: '宫崎骏',
      rating: 9.4,
      poster: 'https://via.placeholder.com/300x450?text=SpiritedAway',
    },
  },
  {
    id: 'item_006',
    type: 'product',
    title: 'iPhone 15 Pro',
    description: '苹果最新旗舰手机',
    category: '手机',
    tags: ['苹果', '5G', '旗舰'],
    status: 'active',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    metadata: {
      price: 8999,
      brand: 'Apple',
      image: 'https://via.placeholder.com/300x300?text=iPhone15',
    },
  },
  {
    id: 'item_007',
    type: 'article',
    title: 'Vue 3 Composition API 最佳实践',
    description: '深入探讨 Vue 3 组合式 API 的使用技巧',
    category: '技术',
    tags: ['Vue', '前端', '教程'],
    status: 'active',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    metadata: {
      author: '技术博主',
      readTime: 10,
    },
  },
  {
    id: 'item_008',
    type: 'video',
    title: 'TypeScript 入门到精通',
    description: '完整的 TypeScript 学习教程',
    category: '教育',
    tags: ['TypeScript', '编程', '教程'],
    status: 'active',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    metadata: {
      duration: 3600,
      thumbnail: 'https://via.placeholder.com/640x360?text=TypeScript',
    },
  },
]

// =============================================================================
// Mock 服务实现
// =============================================================================

/**
 * Mock 认证服务
 */
class MockAuthService implements IAuthService {
  private currentUser: User | null = null
  private token: string | null = null

  async login(credentials: LoginRequest): Promise<LoginResponse> {
    await delay(500)
    
    const user = mockUsers.find(u => u.email === credentials.email)
    if (!user) {
      throw new Error('用户不存在')
    }
    
    // Mock: 任何密码都可以登录
    this.currentUser = user
    this.token = `mock_token_${generateId()}`
    
    return {
      token: this.token,
      user: user,
      expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
    }
  }

  async register(data: RegisterRequest): Promise<void> {
    await delay(500)
    
    const exists = mockUsers.find(u => u.email === data.email)
    if (exists) {
      throw new Error('邮箱已被注册')
    }
    
    const newUser: User = {
      id: `user_${generateId()}`,
      name: data.name,
      email: data.email,
      age: data.age || 0,
      gender: data.gender || 'unknown',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }
    
    mockUsers.push(newUser)
  }

  async logout(): Promise<void> {
    await delay(200)
    this.currentUser = null
    this.token = null
  }

  async refreshToken(): Promise<string> {
    await delay(200)
    this.token = `mock_token_${generateId()}`
    return this.token
  }

  async getCurrentUser(): Promise<User> {
    await delay(300)
    if (!this.currentUser) {
      // 返回默认用户用于测试
      return mockUsers[0]
    }
    return this.currentUser
  }
}

/**
 * Mock 用户服务
 */
class MockUserService implements IUserService {
  async getUser(userId: string): Promise<User> {
    await delay(300)
    const user = mockUsers.find(u => u.id === userId)
    if (!user) {
      throw new Error('用户不存在')
    }
    return user
  }

  async updateUser(userId: string, data: UpdateUserRequest): Promise<User> {
    await delay(400)
    const userIndex = mockUsers.findIndex(u => u.id === userId)
    if (userIndex === -1) {
      throw new Error('用户不存在')
    }
    
    mockUsers[userIndex] = {
      ...mockUsers[userIndex],
      ...data,
      updated_at: new Date().toISOString(),
    }
    
    return mockUsers[userIndex]
  }

  async getProfile(userId: string): Promise<UserProfile> {
    await delay(400)
    const user = mockUsers.find(u => u.id === userId)
    if (!user) {
      throw new Error('用户不存在')
    }
    
    return {
      user,
      total_actions: Math.floor(Math.random() * 1000),
      preferred_types: {
        movie: 0.6,
        product: 0.2,
        article: 0.15,
        video: 0.05,
      },
      active_hours: {
        8: 10,
        12: 25,
        18: 35,
        21: 45,
        23: 20,
      },
      last_active: new Date().toISOString(),
    }
  }

  async getBehaviors(userId: string, limit = 50): Promise<UserBehavior[]> {
    await delay(400)
    
    const behaviors: UserBehavior[] = []
    const actions = ['view', 'click', 'like', 'dislike', 'buy', 'share']
    
    for (let i = 0; i < Math.min(limit, 20); i++) {
      behaviors.push({
        user_id: userId,
        item_id: mockItems[Math.floor(Math.random() * mockItems.length)].id,
        action: actions[Math.floor(Math.random() * actions.length)],
        timestamp: randomDate(),
      })
    }
    
    return behaviors.sort((a, b) => 
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    )
  }

  async recordBehavior(data: RecordBehaviorRequest): Promise<void> {
    await delay(200)
    console.log('Mock: Recorded behavior', data)
  }
}

/**
 * Mock 物品服务
 */
class MockItemService implements IItemService {
  async getItem(itemId: string): Promise<Item> {
    await delay(300)
    const item = mockItems.find(i => i.id === itemId)
    if (!item) {
      throw new Error('物品不存在')
    }
    return item
  }

  async searchItems(query: string, limit = 20): Promise<Item[]> {
    await delay(400)
    const lowerQuery = query.toLowerCase()
    return mockItems
      .filter(item => 
        item.title.toLowerCase().includes(lowerQuery) ||
        item.description.toLowerCase().includes(lowerQuery) ||
        item.tags.some(tag => tag.toLowerCase().includes(lowerQuery))
      )
      .slice(0, limit)
  }

  async listItems(params: ListItemsRequest): Promise<ListItemsResponse> {
    await delay(400)
    
    let filtered = [...mockItems]
    
    if (params.type) {
      filtered = filtered.filter(i => i.type === params.type)
    }
    
    if (params.category) {
      filtered = filtered.filter(i => i.category === params.category)
    }
    
    const page = params.page || 1
    const pageSize = params.page_size || 20
    const start = (page - 1) * pageSize
    const items = filtered.slice(start, start + pageSize)
    
    return {
      items,
      total: filtered.length,
      page,
    }
  }

  async getItemStats(itemId: string): Promise<ItemStats> {
    await delay(300)
    return {
      item_id: itemId,
      view_count: Math.floor(Math.random() * 10000),
      click_count: Math.floor(Math.random() * 5000),
      like_count: Math.floor(Math.random() * 1000),
      share_count: Math.floor(Math.random() * 500),
      avg_rating: Number((Math.random() * 2 + 3).toFixed(1)),
    }
  }

  async getSimilarItems(itemId: string, limit = 10): Promise<SimilarItem[]> {
    await delay(400)
    const currentItem = mockItems.find(i => i.id === itemId)
    
    return mockItems
      .filter(i => i.id !== itemId)
      .map(item => ({
        item,
        score: Math.random(),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
  }
}

/**
 * Mock 推荐服务
 */
class MockRecommendService implements IRecommendService {
  async getRecommendations(request: RecommendRequest): Promise<RecommendResponse> {
    await delay(500)
    
    const reasons = [
      '根据您的浏览历史推荐',
      '与您喜欢的内容相似',
      '热门推荐',
      '为您精选',
      '根据您的兴趣推荐',
    ]
    
    let filtered = [...mockItems]
    
    if (request.exclude_items?.length) {
      filtered = filtered.filter(i => !request.exclude_items!.includes(i.id))
    }
    
    const recommendations: Recommendation[] = filtered
      .slice(0, request.limit || 20)
      .map(item => ({
        item_id: item.id,
        score: Math.random(),
        reason: reasons[Math.floor(Math.random() * reasons.length)],
        item,
      }))
      .sort((a, b) => b.score - a.score)
    
    return {
      recommendations,
      request_id: `req_${generateId()}`,
      strategy: 'collaborative_filtering',
    }
  }

  async submitFeedback(feedback: FeedbackRequest): Promise<void> {
    await delay(200)
    console.log('Mock: Submitted feedback', feedback)
  }

  async getSimilarRecommendations(itemId: string, limit = 10): Promise<Recommendation[]> {
    await delay(400)
    
    return mockItems
      .filter(i => i.id !== itemId)
      .slice(0, limit)
      .map(item => ({
        item_id: item.id,
        score: Math.random(),
        reason: '相似物品推荐',
        item,
      }))
  }
}

/**
 * Mock 分析服务
 */
class MockAnalyticsService implements IAnalyticsService {
  async getDashboardStats(): Promise<DashboardStats> {
    await delay(400)
    return {
      total_users: 15823,
      total_items: 8456,
      total_recommendations: 2456789,
      daily_active_users: 3256,
      recommendation_ctr: 0.125,
      avg_response_time: 45.6,
    }
  }

  async getUserTrend(days: number): Promise<TimeSeriesPoint[]> {
    await delay(400)
    const points: TimeSeriesPoint[] = []
    const now = new Date()
    
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date(now)
      date.setDate(date.getDate() - i)
      points.push({
        timestamp: date.toISOString().split('T')[0],
        value: Math.floor(Math.random() * 500 + 200),
      })
    }
    
    return points
  }

  async getItemTypeStats(): Promise<CategoryStats[]> {
    await delay(300)
    return [
      { category: 'movie', count: 3500, percentage: 0.41 },
      { category: 'product', count: 2800, percentage: 0.33 },
      { category: 'article', count: 1500, percentage: 0.18 },
      { category: 'video', count: 656, percentage: 0.08 },
    ]
  }

  async getRecommendationTrend(days: number): Promise<TimeSeriesPoint[]> {
    await delay(400)
    const points: TimeSeriesPoint[] = []
    const now = new Date()
    
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date(now)
      date.setDate(date.getDate() - i)
      points.push({
        timestamp: date.toISOString().split('T')[0],
        value: Math.floor(Math.random() * 50000 + 80000),
      })
    }
    
    return points
  }

  async getTopCategories(limit: number): Promise<CategoryStats[]> {
    await delay(300)
    return [
      { category: '剧情', count: 1200, percentage: 0.25 },
      { category: '动作', count: 950, percentage: 0.20 },
      { category: '喜剧', count: 800, percentage: 0.17 },
      { category: '科幻', count: 650, percentage: 0.14 },
      { category: '爱情', count: 550, percentage: 0.12 },
    ].slice(0, limit)
  }

  async getCTRTrend(startDate: string, endDate: string): Promise<TimeSeriesPoint[]> {
    await delay(400)
    const points: TimeSeriesPoint[] = []
    const start = new Date(startDate)
    const end = new Date(endDate)
    
    const current = new Date(start)
    while (current <= end) {
      points.push({
        timestamp: current.toISOString().split('T')[0],
        value: Number((Math.random() * 0.05 + 0.10).toFixed(4)),
      })
      current.setDate(current.getDate() + 1)
    }
    
    return points
  }
}

/**
 * Mock 管理员用户服务
 */
class MockAdminUserService implements IAdminUserService {
  async listUsers(params: { page: number; page_size: number; keyword?: string; gender?: string }): Promise<{ items: User[]; total: number }> {
    await delay(400)
    
    let filtered = [...mockUsers]
    
    if (params.keyword) {
      const keyword = params.keyword.toLowerCase()
      filtered = filtered.filter(u => 
        u.name.toLowerCase().includes(keyword) ||
        u.email.toLowerCase().includes(keyword)
      )
    }
    
    if (params.gender) {
      filtered = filtered.filter(u => u.gender === params.gender)
    }
    
    const start = (params.page - 1) * params.page_size
    const items = filtered.slice(start, start + params.page_size)
    
    return {
      items,
      total: filtered.length,
    }
  }

  async getUser(userId: string): Promise<User> {
    await delay(300)
    const user = mockUsers.find(u => u.id === userId)
    if (!user) {
      throw new Error('用户不存在')
    }
    return user
  }

  async createUser(data: CreateUserRequest): Promise<User> {
    await delay(400)
    
    const newUser: User = {
      id: `user_${generateId()}`,
      name: data.name,
      email: data.email,
      age: data.age || 0,
      gender: data.gender || 'unknown',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }
    
    mockUsers.push(newUser)
    return newUser
  }

  async updateUser(userId: string, data: UpdateUserRequest): Promise<User> {
    await delay(400)
    const userIndex = mockUsers.findIndex(u => u.id === userId)
    if (userIndex === -1) {
      throw new Error('用户不存在')
    }
    
    mockUsers[userIndex] = {
      ...mockUsers[userIndex],
      ...data,
      updated_at: new Date().toISOString(),
    }
    
    return mockUsers[userIndex]
  }

  async deleteUser(userId: string): Promise<void> {
    await delay(300)
    const userIndex = mockUsers.findIndex(u => u.id === userId)
    if (userIndex === -1) {
      throw new Error('用户不存在')
    }
    mockUsers.splice(userIndex, 1)
  }
}

/**
 * Mock 管理员物品服务
 */
class MockAdminItemService implements IAdminItemService {
  async listItems(params: { page: number; page_size: number; type?: string; keyword?: string }): Promise<{ items: Item[]; total: number }> {
    await delay(400)
    
    let filtered = [...mockItems]
    
    if (params.type) {
      filtered = filtered.filter(i => i.type === params.type)
    }
    
    if (params.keyword) {
      const keyword = params.keyword.toLowerCase()
      filtered = filtered.filter(i => 
        i.title.toLowerCase().includes(keyword) ||
        i.description.toLowerCase().includes(keyword)
      )
    }
    
    const start = (params.page - 1) * params.page_size
    const items = filtered.slice(start, start + params.page_size)
    
    return {
      items,
      total: filtered.length,
    }
  }

  async getItem(itemId: string): Promise<Item> {
    await delay(300)
    const item = mockItems.find(i => i.id === itemId)
    if (!item) {
      throw new Error('物品不存在')
    }
    return item
  }

  async createItem(data: CreateItemRequest): Promise<Item> {
    await delay(400)
    
    const newItem: Item = {
      id: `item_${generateId()}`,
      type: data.type,
      title: data.title,
      description: data.description || '',
      category: data.category || '',
      tags: data.tags || [],
      metadata: data.metadata,
      status: 'active',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }
    
    mockItems.push(newItem)
    return newItem
  }

  async updateItem(itemId: string, data: UpdateItemRequest): Promise<Item> {
    await delay(400)
    const itemIndex = mockItems.findIndex(i => i.id === itemId)
    if (itemIndex === -1) {
      throw new Error('物品不存在')
    }
    
    mockItems[itemIndex] = {
      ...mockItems[itemIndex],
      ...data,
      updated_at: new Date().toISOString(),
    }
    
    return mockItems[itemIndex]
  }

  async deleteItem(itemId: string): Promise<void> {
    await delay(300)
    const itemIndex = mockItems.findIndex(i => i.id === itemId)
    if (itemIndex === -1) {
      throw new Error('物品不存在')
    }
    mockItems.splice(itemIndex, 1)
  }
}

// =============================================================================
// Mock API Provider
// =============================================================================

/**
 * Mock API Provider
 * 
 * 开发/测试环境使用，返回模拟数据
 * 
 * @example
 * ```typescript
 * const api = new MockApiProvider()
 * const user = await api.user.getUser('user_001')
 * ```
 */
export class MockApiProvider implements IApiProvider {
  private _auth: MockAuthService
  private _user: MockUserService
  private _item: MockItemService
  private _recommend: MockRecommendService
  private _analytics: MockAnalyticsService
  private _adminUser: MockAdminUserService
  private _adminItem: MockAdminItemService

  constructor() {
    this._auth = new MockAuthService()
    this._user = new MockUserService()
    this._item = new MockItemService()
    this._recommend = new MockRecommendService()
    this._analytics = new MockAnalyticsService()
    this._adminUser = new MockAdminUserService()
    this._adminItem = new MockAdminItemService()
  }

  get auth() {
    return this._auth
  }

  get user() {
    return this._user
  }

  get item() {
    return this._item
  }

  get recommend() {
    return this._recommend
  }

  get analytics() {
    return this._analytics
  }

  get adminUser() {
    return this._adminUser
  }

  get adminItem() {
    return this._adminItem
  }
}

