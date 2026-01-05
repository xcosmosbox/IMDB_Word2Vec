/**
 * Pinia Store 统一导出
 * 
 * @module user-app/stores
 * @author Person F
 */

// =============================================================================
// Store 导出
// =============================================================================

export { useUserStore } from './user'
export type { UserStore } from './user'

export { useRecommendStore } from './recommend'
export type { BehaviorAction, BehaviorParams } from './recommend'

export { useItemStore } from './item'

// =============================================================================
// Store 初始化函数
// =============================================================================

import type { IApiProvider } from '@shared/api/interfaces'
import { useUserStore } from './user'
import { useRecommendStore } from './recommend'
import { useItemStore } from './item'

/**
 * 初始化所有 Store
 * 
 * 在应用启动时调用，用于恢复持久化状态
 * 
 * @param api - API Provider 实例
 * 
 * @example
 * ```typescript
 * // main.ts
 * import { initStores } from '@/stores'
 * 
 * const api = createApiProvider()
 * await initStores(api)
 * ```
 */
export async function initStores(_api: IApiProvider): Promise<void> {
  const userStore = useUserStore()
  const recommendStore = useRecommendStore()
  const itemStore = useItemStore()
  
  // 初始化用户状态（如果有保存的 token，尝试恢复用户信息）
  await userStore.init()
  
  // 如果用户已登录，设置推荐 store 的用户 ID
  if (userStore.userId) {
    recommendStore.setCurrentUserId(userStore.userId)
  }
  
  // 加载本地持久化的数据
  recommendStore.loadUserPreferences()
  itemStore.loadRecentlyViewed()
}

/**
 * 重置所有 Store
 * 
 * 用于登出时清理状态
 */
export function resetStores(): void {
  const userStore = useUserStore()
  const recommendStore = useRecommendStore()
  const itemStore = useItemStore()
  
  recommendStore.$reset()
  itemStore.$reset()
  // userStore 的重置在 logout 中处理
}

