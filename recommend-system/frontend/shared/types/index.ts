/**
 * 前端共享类型定义
 * 
 * 所有前端开发者必须使用这些类型定义，确保与后端 API 对接一致。
 * 类型定义与后端 internal/interfaces/interfaces.go 保持同步。
 */

// =============================================================================
// 用户相关类型
// =============================================================================

/** 用户 */
export interface User {
  id: string;
  name: string;
  email: string;
  age: number;
  gender: string;
  metadata?: Record<string, string>;
  created_at: string;
  updated_at: string;
}

/** 创建用户请求 */
export interface CreateUserRequest {
  name: string;
  email: string;
  age?: number;
  gender?: string;
}

/** 更新用户请求 */
export interface UpdateUserRequest {
  name?: string;
  email?: string;
  age?: number;
  gender?: string;
}

/** 用户行为 */
export interface UserBehavior {
  user_id: string;
  item_id: string;
  action: string;
  timestamp: string;
  context?: Record<string, string>;
}

/** 记录行为请求 */
export interface RecordBehaviorRequest {
  user_id: string;
  item_id: string;
  action: 'view' | 'click' | 'like' | 'dislike' | 'buy' | 'share';
  context?: Record<string, string>;
}

/** 用户画像 */
export interface UserProfile {
  user: User;
  total_actions: number;
  preferred_types: Record<string, number>;
  active_hours: Record<number, number>;
  last_active: string;
}

// =============================================================================
// 物品相关类型
// =============================================================================

/** 物品 */
export interface Item {
  id: string;
  type: 'movie' | 'product' | 'article' | 'video';
  title: string;
  description: string;
  category: string;
  tags: string[];
  metadata?: Record<string, any>;
  status: 'active' | 'inactive';
  created_at: string;
  updated_at: string;
}

/** 创建物品请求 */
export interface CreateItemRequest {
  type: 'movie' | 'product' | 'article' | 'video';
  title: string;
  description?: string;
  category?: string;
  tags?: string[];
  metadata?: Record<string, any>;
}

/** 更新物品请求 */
export interface UpdateItemRequest {
  title?: string;
  description?: string;
  category?: string;
  tags?: string[];
  metadata?: Record<string, any>;
}

/** 物品列表请求 */
export interface ListItemsRequest {
  type?: string;
  category?: string;
  page?: number;
  page_size?: number;
}

/** 物品列表响应 */
export interface ListItemsResponse {
  items: Item[];
  total: number;
  page: number;
}

/** 物品统计 */
export interface ItemStats {
  item_id: string;
  view_count: number;
  click_count: number;
  like_count: number;
  share_count: number;
  avg_rating: number;
}

/** 相似物品 */
export interface SimilarItem {
  item: Item;
  score: number;
}

// =============================================================================
// 推荐相关类型
// =============================================================================

/** 推荐项 */
export interface Recommendation {
  item_id: string;
  score: number;
  reason?: string;
  item?: Item;  // 展开后的物品详情
}

/** 推荐请求 */
export interface RecommendRequest {
  user_id: string;
  limit?: number;
  exclude_items?: string[];
  scene?: 'home' | 'search' | 'detail';
  context?: Record<string, string>;
}

/** 推荐响应 */
export interface RecommendResponse {
  recommendations: Recommendation[];
  request_id: string;
  strategy: string;
}

/** 反馈请求 */
export interface FeedbackRequest {
  user_id: string;
  item_id: string;
  action: string;
  request_id?: string;
}

// =============================================================================
// API 响应通用类型
// =============================================================================

/** API 响应包装 */
export interface ApiResponse<T> {
  code: number;
  data: T;
  message?: string;
}

/** 分页响应 */
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

/** API 错误 */
export interface ApiError {
  code: number;
  message: string;
  details?: Record<string, any>;
}

// =============================================================================
// 统计和分析类型
// =============================================================================

/** 仪表盘统计 */
export interface DashboardStats {
  total_users: number;
  total_items: number;
  total_recommendations: number;
  daily_active_users: number;
  recommendation_ctr: number;
  avg_response_time: number;
}

/** 时间序列数据点 */
export interface TimeSeriesPoint {
  timestamp: string;
  value: number;
}

/** 分类统计 */
export interface CategoryStats {
  category: string;
  count: number;
  percentage: number;
}

// =============================================================================
// 认证相关类型
// =============================================================================

/** 登录请求 */
export interface LoginRequest {
  email: string;
  password: string;
}

/** 登录响应 */
export interface LoginResponse {
  token: string;
  user: User;
  expires_at: string;
}

/** 注册请求 */
export interface RegisterRequest {
  name: string;
  email: string;
  password: string;
  age?: number;
  gender?: string;
}

