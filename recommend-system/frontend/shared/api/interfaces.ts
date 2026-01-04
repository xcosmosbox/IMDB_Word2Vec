/**
 * 前端 API 服务接口定义
 * 
 * 这是前端的"接口驱动开发"核心文件。
 * 所有 API 实现必须遵循这些接口，确保可插拔设计。
 * 
 * 使用场景：
 * 1. 开发时使用 MockApiProvider
 * 2. 生产时使用 HttpApiProvider
 * 3. 测试时注入任意实现
 */

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

// =============================================================================
// 认证服务接口
// =============================================================================

export interface IAuthService {
  /** 登录 */
  login(credentials: LoginRequest): Promise<LoginResponse>
  
  /** 注册 */
  register(data: RegisterRequest): Promise<void>
  
  /** 登出 */
  logout(): Promise<void>
  
  /** 刷新 Token */
  refreshToken(): Promise<string>
  
  /** 获取当前用户 */
  getCurrentUser(): Promise<User>
}

// =============================================================================
// 用户服务接口
// =============================================================================

export interface IUserService {
  /** 获取用户信息 */
  getUser(userId: string): Promise<User>
  
  /** 更新用户信息 */
  updateUser(userId: string, data: UpdateUserRequest): Promise<User>
  
  /** 获取用户画像 */
  getProfile(userId: string): Promise<UserProfile>
  
  /** 获取用户行为历史 */
  getBehaviors(userId: string, limit?: number): Promise<UserBehavior[]>
  
  /** 记录用户行为 */
  recordBehavior(data: RecordBehaviorRequest): Promise<void>
}

// =============================================================================
// 物品服务接口
// =============================================================================

export interface IItemService {
  /** 获取物品详情 */
  getItem(itemId: string): Promise<Item>
  
  /** 搜索物品 */
  searchItems(query: string, limit?: number): Promise<Item[]>
  
  /** 列出物品 */
  listItems(params: ListItemsRequest): Promise<ListItemsResponse>
  
  /** 获取物品统计 */
  getItemStats(itemId: string): Promise<ItemStats>
  
  /** 获取相似物品 */
  getSimilarItems(itemId: string, limit?: number): Promise<SimilarItem[]>
}

// =============================================================================
// 推荐服务接口
// =============================================================================

export interface IRecommendService {
  /** 获取推荐列表 */
  getRecommendations(request: RecommendRequest): Promise<RecommendResponse>
  
  /** 提交反馈 */
  submitFeedback(feedback: FeedbackRequest): Promise<void>
  
  /** 获取相似推荐 */
  getSimilarRecommendations(itemId: string, limit?: number): Promise<Recommendation[]>
}

// =============================================================================
// 分析服务接口 (管理后台)
// =============================================================================

export interface IAnalyticsService {
  /** 获取仪表盘统计 */
  getDashboardStats(): Promise<DashboardStats>
  
  /** 获取用户增长趋势 */
  getUserTrend(days: number): Promise<TimeSeriesPoint[]>
  
  /** 获取物品类型统计 */
  getItemTypeStats(): Promise<CategoryStats[]>
  
  /** 获取推荐量趋势 */
  getRecommendationTrend(days: number): Promise<TimeSeriesPoint[]>
  
  /** 获取热门分类 */
  getTopCategories(limit: number): Promise<CategoryStats[]>
  
  /** 获取点击率趋势 */
  getCTRTrend(startDate: string, endDate: string): Promise<TimeSeriesPoint[]>
}

// =============================================================================
// 管理员 CRUD 服务接口
// =============================================================================

export interface IAdminUserService {
  /** 列出用户 */
  listUsers(params: { page: number; page_size: number; keyword?: string; gender?: string }): Promise<{ items: User[]; total: number }>
  
  /** 获取用户 */
  getUser(userId: string): Promise<User>
  
  /** 创建用户 */
  createUser(data: CreateUserRequest): Promise<User>
  
  /** 更新用户 */
  updateUser(userId: string, data: UpdateUserRequest): Promise<User>
  
  /** 删除用户 */
  deleteUser(userId: string): Promise<void>
}

export interface IAdminItemService {
  /** 列出物品 */
  listItems(params: { page: number; page_size: number; type?: string; keyword?: string }): Promise<{ items: Item[]; total: number }>
  
  /** 获取物品 */
  getItem(itemId: string): Promise<Item>
  
  /** 创建物品 */
  createItem(data: CreateItemRequest): Promise<Item>
  
  /** 更新物品 */
  updateItem(itemId: string, data: UpdateItemRequest): Promise<Item>
  
  /** 删除物品 */
  deleteItem(itemId: string): Promise<void>
}

// =============================================================================
// API Provider 接口 (依赖注入容器)
// =============================================================================

/**
 * API Provider - 所有服务的统一入口
 * 
 * 使用方式：
 * ```typescript
 * // 生产环境
 * const api = new HttpApiProvider()
 * 
 * // 开发/测试环境
 * const api = new MockApiProvider()
 * 
 * // 使用
 * const user = await api.user.getUser('123')
 * const recs = await api.recommend.getRecommendations(req)
 * ```
 */
export interface IApiProvider {
  readonly auth: IAuthService
  readonly user: IUserService
  readonly item: IItemService
  readonly recommend: IRecommendService
  readonly analytics: IAnalyticsService
  readonly adminUser: IAdminUserService
  readonly adminItem: IAdminItemService
}

// =============================================================================
// 存储接口 (可选：用于持久化抽象)
// =============================================================================

export interface IStorageService {
  get<T>(key: string): T | null
  set<T>(key: string, value: T): void
  remove(key: string): void
  clear(): void
}

// =============================================================================
// 缓存接口 (可选：用于性能优化)
// =============================================================================

export interface ICacheService {
  get<T>(key: string): Promise<T | null>
  set<T>(key: string, value: T, ttlSeconds?: number): Promise<void>
  delete(key: string): Promise<void>
  has(key: string): Promise<boolean>
}

