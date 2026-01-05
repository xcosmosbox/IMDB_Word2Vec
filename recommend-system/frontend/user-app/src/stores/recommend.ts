/**
 * 推荐状态管理 Store
 * 
 * 负责推荐相关的状态管理和用户行为记录
 * 使用依赖注入获取 API Provider
 */
import { defineStore } from 'pinia'
import { ref, computed, inject } from 'vue'
import type { 
  Recommendation, 
  RecommendRequest, 
  RecommendResponse,
  FeedbackRequest,
  RecordBehaviorRequest,
} from '@shared/types'
import type { IApiProvider } from '@shared/api/interfaces'

/** 用户行为类型 */
export type BehaviorAction = 'view' | 'click' | 'like' | 'dislike' | 'buy' | 'share'

/** 行为记录参数 */
export interface BehaviorParams {
  item_id: string
  action: BehaviorAction
  context?: Record<string, string>
}

export const useRecommendStore = defineStore('recommend', () => {
  // API Provider 注入
  const api = inject<IApiProvider>('api')
  
  // =========================================================================
  // 状态
  // =========================================================================
  
  /** 当前用户ID (由认证模块设置) */
  const currentUserId = ref<string | null>(null)
  
  /** 推荐列表 */
  const recommendations = ref<Recommendation[]>([])
  
  /** 当前请求ID (用于反馈追踪) */
  const currentRequestId = ref<string | null>(null)
  
  /** 推荐策略 */
  const currentStrategy = ref<string>('')
  
  /** 是否正在加载推荐 */
  const isLoading = ref(false)
  
  /** 用户喜欢的物品ID集合 */
  const likedItems = ref<Set<string>>(new Set())
  
  /** 用户不喜欢的物品ID集合 */
  const dislikedItems = ref<Set<string>>(new Set())
  
  /** 待发送的行为队列 */
  const behaviorQueue = ref<RecordBehaviorRequest[]>([])
  
  /** 错误信息 */
  const error = ref<string | null>(null)
  
  // =========================================================================
  // 计算属性
  // =========================================================================
  
  /** 是否有推荐结果 */
  const hasRecommendations = computed(() => recommendations.value.length > 0)
  
  /** 推荐数量 */
  const recommendationCount = computed(() => recommendations.value.length)
  
  /** 喜欢的物品数量 */
  const likedCount = computed(() => likedItems.value.size)
  
  // =========================================================================
  // 方法
  // =========================================================================
  
  /**
   * 设置当前用户ID
   * @param userId 用户ID
   */
  function setCurrentUserId(userId: string) {
    currentUserId.value = userId
  }
  
  /**
   * 获取推荐列表
   * @param options 推荐请求选项
   * @returns 推荐响应
   */
  async function getRecommendations(options: Partial<RecommendRequest> = {}): Promise<RecommendResponse> {
    if (!api) {
      throw new Error('API Provider not injected')
    }
    
    if (!currentUserId.value) {
      throw new Error('User ID not set')
    }
    
    isLoading.value = true
    error.value = null
    
    try {
      const request: RecommendRequest = {
        user_id: currentUserId.value,
        limit: options.limit ?? 20,
        exclude_items: options.exclude_items ?? [],
        scene: options.scene ?? 'home',
        context: options.context ?? {},
      }
      
      const response = await api.recommend.getRecommendations(request)
      
      recommendations.value = response.recommendations
      currentRequestId.value = response.request_id
      currentStrategy.value = response.strategy
      
      return response
    } catch (e) {
      error.value = e instanceof Error ? e.message : '获取推荐失败'
      throw e
    } finally {
      isLoading.value = false
    }
  }
  
  /**
   * 提交反馈
   * @param feedback 反馈请求
   */
  async function submitFeedback(feedback: FeedbackRequest): Promise<void> {
    if (!api) {
      throw new Error('API Provider not injected')
    }
    
    try {
      // 添加请求ID用于追踪
      const feedbackWithRequestId: FeedbackRequest = {
        ...feedback,
        request_id: feedback.request_id ?? currentRequestId.value ?? undefined,
      }
      
      await api.recommend.submitFeedback(feedbackWithRequestId)
    } catch (e) {
      error.value = e instanceof Error ? e.message : '提交反馈失败'
      throw e
    }
  }
  
  /**
   * 记录用户行为
   * @param params 行为参数
   */
  async function recordBehavior(params: BehaviorParams): Promise<void> {
    if (!currentUserId.value) {
      console.warn('User ID not set, behavior not recorded')
      return
    }
    
    const behaviorRequest: RecordBehaviorRequest = {
      user_id: currentUserId.value,
      item_id: params.item_id,
      action: params.action,
      context: params.context,
    }
    
    // 更新本地状态
    if (params.action === 'like') {
      likedItems.value.add(params.item_id)
      dislikedItems.value.delete(params.item_id)
    } else if (params.action === 'dislike') {
      dislikedItems.value.add(params.item_id)
      likedItems.value.delete(params.item_id)
    }
    
    // 添加到队列
    behaviorQueue.value.push(behaviorRequest)
    
    // 尝试批量发送
    await flushBehaviorQueue()
  }
  
  /**
   * 批量发送行为队列
   */
  async function flushBehaviorQueue(): Promise<void> {
    if (!api || behaviorQueue.value.length === 0) {
      return
    }
    
    // 取出队列中的行为
    const behaviors = [...behaviorQueue.value]
    behaviorQueue.value = []
    
    try {
      // 逐个发送 (可以优化为批量接口)
      await Promise.all(
        behaviors.map(behavior => 
          submitFeedback({
            user_id: behavior.user_id,
            item_id: behavior.item_id,
            action: behavior.action,
          })
        )
      )
    } catch (e) {
      // 发送失败，放回队列
      behaviorQueue.value = [...behaviors, ...behaviorQueue.value]
      console.error('Failed to flush behavior queue:', e)
    }
  }
  
  /**
   * 获取相似推荐
   * @param itemId 物品ID
   * @param limit 返回数量
   * @returns 推荐列表
   */
  async function getSimilarRecommendations(itemId: string, limit: number = 10): Promise<Recommendation[]> {
    if (!api) {
      throw new Error('API Provider not injected')
    }
    
    try {
      const similar = await api.recommend.getSimilarRecommendations(itemId, limit)
      return similar
    } catch (e) {
      error.value = e instanceof Error ? e.message : '获取相似推荐失败'
      throw e
    }
  }
  
  /**
   * 检查物品是否被喜欢
   * @param itemId 物品ID
   * @returns 是否被喜欢
   */
  function isItemLiked(itemId: string): boolean {
    return likedItems.value.has(itemId)
  }
  
  /**
   * 检查物品是否被不喜欢
   * @param itemId 物品ID
   * @returns 是否被不喜欢
   */
  function isItemDisliked(itemId: string): boolean {
    return dislikedItems.value.has(itemId)
  }
  
  /**
   * 切换物品喜欢状态
   * @param itemId 物品ID
   * @returns 新的喜欢状态
   */
  async function toggleLike(itemId: string): Promise<boolean> {
    const isLiked = likedItems.value.has(itemId)
    
    if (isLiked) {
      likedItems.value.delete(itemId)
      await recordBehavior({ item_id: itemId, action: 'dislike' })
    } else {
      likedItems.value.add(itemId)
      dislikedItems.value.delete(itemId)
      await recordBehavior({ item_id: itemId, action: 'like' })
    }
    
    return !isLiked
  }
  
  /**
   * 加载用户喜好数据
   */
  function loadUserPreferences() {
    try {
      const storedLiked = localStorage.getItem('likedItems')
      if (storedLiked) {
        likedItems.value = new Set(JSON.parse(storedLiked))
      }
      
      const storedDisliked = localStorage.getItem('dislikedItems')
      if (storedDisliked) {
        dislikedItems.value = new Set(JSON.parse(storedDisliked))
      }
    } catch (e) {
      console.warn('Failed to load user preferences')
    }
  }
  
  /**
   * 保存用户喜好数据
   */
  function saveUserPreferences() {
    try {
      localStorage.setItem('likedItems', JSON.stringify([...likedItems.value]))
      localStorage.setItem('dislikedItems', JSON.stringify([...dislikedItems.value]))
    } catch (e) {
      console.warn('Failed to save user preferences')
    }
  }
  
  /**
   * 清除推荐
   */
  function clearRecommendations() {
    recommendations.value = []
    currentRequestId.value = null
    currentStrategy.value = ''
  }
  
  /**
   * 重置状态
   */
  function $reset() {
    recommendations.value = []
    currentRequestId.value = null
    currentStrategy.value = ''
    isLoading.value = false
    error.value = null
    behaviorQueue.value = []
    // 保留用户喜好数据
  }
  
  // 初始化时加载用户喜好
  loadUserPreferences()
  
  // 页面卸载时保存喜好数据
  if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', saveUserPreferences)
  }
  
  return {
    // 状态
    currentUserId,
    recommendations,
    currentRequestId,
    currentStrategy,
    isLoading,
    likedItems,
    dislikedItems,
    behaviorQueue,
    error,
    
    // 计算属性
    hasRecommendations,
    recommendationCount,
    likedCount,
    
    // 方法
    setCurrentUserId,
    getRecommendations,
    submitFeedback,
    recordBehavior,
    flushBehaviorQueue,
    getSimilarRecommendations,
    isItemLiked,
    isItemDisliked,
    toggleLike,
    loadUserPreferences,
    saveUserPreferences,
    clearRecommendations,
    $reset,
  }
})
