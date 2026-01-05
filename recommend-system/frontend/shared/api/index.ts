/**
 * API 模块统一导出
 * 
 * 此文件是所有 API 相关功能的统一入口
 * 
 * @module shared/api
 * @author Person F
 */

// =============================================================================
// 类型导出
// =============================================================================

export type {
  IApiProvider,
  IAuthService,
  IUserService,
  IItemService,
  IRecommendService,
  IAnalyticsService,
  IAdminUserService,
  IAdminItemService,
  IStorageService,
  ICacheService,
} from './interfaces'

// =============================================================================
// 请求工具导出
// =============================================================================

export {
  request,
  getToken,
  setToken,
  removeToken,
} from './request'

export { default as axiosInstance } from './request'

// =============================================================================
// 服务类导出
// =============================================================================

export { AuthService, authService, authApi } from './auth'
export { UserService, userService, userApi } from './user'
export { ItemService, itemService, itemApi } from './item'
export { RecommendService, recommendService, recommendApi } from './recommend'
export { AnalyticsService, analyticsService, analyticsApi } from './analytics'
export {
  AdminUserService,
  AdminItemService,
  adminUserService,
  adminItemService,
  adminUserApi,
  adminItemApi,
} from './admin'

// =============================================================================
// Provider 导出
// =============================================================================

export {
  HttpApiProvider,
  MockApiProvider,
  getApiProvider,
  setApiProvider,
  resetApiProvider,
} from './provider'

// =============================================================================
// 默认 API Provider 工厂函数
// =============================================================================

import type { IApiProvider } from './interfaces'
import { HttpApiProvider, MockApiProvider } from './provider'

/**
 * 创建 API Provider 实例
 * 
 * @param options - 配置选项
 * @returns API Provider 实例
 * 
 * @example
 * ```typescript
 * // 生产环境
 * const api = createApiProvider()
 * 
 * // 开发环境使用 Mock
 * const api = createApiProvider({ mock: true })
 * 
 * // 使用
 * const user = await api.user.getUser('123')
 * ```
 */
export function createApiProvider(options: {
  mock?: boolean
} = {}): IApiProvider {
  const useMock = options.mock ?? (import.meta.env.DEV && import.meta.env.VITE_USE_MOCK === 'true')
  
  return useMock ? new MockApiProvider() : new HttpApiProvider()
}

// =============================================================================
// Vue 插件导出
// =============================================================================

import type { App, InjectionKey } from 'vue'

/**
 * API Provider 注入 Key
 */
export const API_PROVIDER_KEY: InjectionKey<IApiProvider> = Symbol('api')

/**
 * API Provider Vue 插件
 * 
 * @example
 * ```typescript
 * import { createApp } from 'vue'
 * import { apiPlugin } from '@shared/api'
 * 
 * const app = createApp(App)
 * app.use(apiPlugin, { mock: import.meta.env.DEV })
 * ```
 */
export const apiPlugin = {
  install(app: App, options: { mock?: boolean } = {}) {
    const api = createApiProvider(options)
    
    // 提供注入
    app.provide(API_PROVIDER_KEY, api)
    app.provide('api', api)
    
    // 全局属性
    app.config.globalProperties.$api = api
  },
}

// =============================================================================
// Composable 导出
// =============================================================================

import { inject } from 'vue'

/**
 * 使用 API Provider 的 Composable
 * 
 * @returns API Provider 实例
 * @throws Error 如果 Provider 未注入
 * 
 * @example
 * ```typescript
 * import { useApi } from '@shared/api'
 * 
 * const api = useApi()
 * const user = await api.user.getUser('123')
 * ```
 */
export function useApi(): IApiProvider {
  const api = inject<IApiProvider>(API_PROVIDER_KEY) || inject<IApiProvider>('api')
  
  if (!api) {
    throw new Error(
      'API Provider 未找到。请确保已在应用中安装 apiPlugin 或通过 app.provide() 提供 API Provider。'
    )
  }
  
  return api
}

/**
 * 安全使用 API Provider 的 Composable
 * 
 * @returns API Provider 实例或 null
 */
export function useApiSafe(): IApiProvider | null {
  try {
    return inject<IApiProvider>(API_PROVIDER_KEY) || inject<IApiProvider>('api') || null
  } catch {
    return null
  }
}

