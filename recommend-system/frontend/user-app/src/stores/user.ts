/**
 * 用户状态管理 Store
 * 
 * 负责管理用户认证状态、用户信息、画像和行为历史。
 * 使用 Pinia Composition API 风格实现。
 * 
 * @module stores/user
 * @author Person C
 */

import { defineStore } from 'pinia'
import { ref, computed, inject } from 'vue'
import type {
  User,
  LoginRequest,
  RegisterRequest,
  UpdateUserRequest,
  UserProfile,
  UserBehavior,
} from '@shared/types'
import type { IApiProvider } from '@shared/api/interfaces'

/**
 * 用户 Store
 * 
 * 使用方式:
 * ```typescript
 * const userStore = useUserStore()
 * await userStore.login({ email: 'user@example.com', password: '123456' })
 * console.log(userStore.isLoggedIn) // true
 * ```
 */
export const useUserStore = defineStore('user', () => {
  // ==========================================================================
  // 依赖注入
  // ==========================================================================
  
  // 注入 API Provider（由 Person F 提供）
  // 在实际使用时，需确保 app.provide('api', apiProvider) 已配置
  const getApi = (): IApiProvider | null => {
    try {
      return inject<IApiProvider>('api') || null
    } catch {
      return null
    }
  }

  // ==========================================================================
  // 状态定义
  // ==========================================================================
  
  /** 认证令牌 */
  const token = ref<string | null>(localStorage.getItem('token'))
  
  /** 当前用户 */
  const currentUser = ref<User | null>(null)
  
  /** 用户画像 */
  const profile = ref<UserProfile | null>(null)
  
  /** 用户行为历史 */
  const behaviors = ref<UserBehavior[]>([])
  
  /** 加载状态 */
  const isLoading = ref(false)
  
  /** 错误信息 */
  const error = ref<string | null>(null)

  // ==========================================================================
  // 计算属性
  // ==========================================================================
  
  /** 是否已登录 */
  const isLoggedIn = computed(() => !!token.value && !!currentUser.value)
  
  /** 用户ID (简化访问) */
  const userId = computed(() => currentUser.value?.id || null)
  
  /** 用户显示名称 */
  const displayName = computed(() => currentUser.value?.name || '游客')
  
  /** 用户首字母头像 */
  const avatarInitial = computed(() => {
    const name = currentUser.value?.name || ''
    return name.charAt(0).toUpperCase() || '?'
  })

  // ==========================================================================
  // 私有工具方法
  // ==========================================================================
  
  /**
   * 保存令牌到本地存储
   */
  function saveToken(newToken: string): void {
    token.value = newToken
    localStorage.setItem('token', newToken)
  }
  
  /**
   * 清除令牌
   */
  function clearToken(): void {
    token.value = null
    localStorage.removeItem('token')
  }
  
  /**
   * 设置错误信息
   */
  function setError(message: string): void {
    error.value = message
  }
  
  /**
   * 清除错误信息
   */
  function clearError(): void {
    error.value = null
  }

  // ==========================================================================
  // 认证操作
  // ==========================================================================
  
  /**
   * 用户登录
   * 
   * @param credentials - 登录凭据 (email + password)
   * @throws Error 登录失败时抛出异常
   * 
   * @example
   * ```typescript
   * try {
   *   await userStore.login({ email: 'user@example.com', password: '123456' })
   *   console.log('登录成功')
   * } catch (e) {
   *   console.error('登录失败:', e.message)
   * }
   * ```
   */
  async function login(credentials: LoginRequest): Promise<void> {
    const api = getApi()
    if (!api) {
      throw new Error('API Provider 未初始化')
    }
    
    clearError()
    isLoading.value = true
    
    try {
      const response = await api.auth.login(credentials)
      
      // 保存令牌
      saveToken(response.token)
      
      // 保存用户信息
      currentUser.value = response.user
      
    } catch (e: any) {
      const message = e?.message || '登录失败，请检查邮箱和密码'
      setError(message)
      throw new Error(message)
    } finally {
      isLoading.value = false
    }
  }
  
  /**
   * 用户注册
   * 
   * @param data - 注册数据
   * @throws Error 注册失败时抛出异常
   */
  async function register(data: RegisterRequest): Promise<void> {
    const api = getApi()
    if (!api) {
      throw new Error('API Provider 未初始化')
    }
    
    clearError()
    isLoading.value = true
    
    try {
      // 执行注册
      await api.auth.register(data)
      
      // 注册成功后自动登录
      await login({ email: data.email, password: data.password })
      
    } catch (e: any) {
      const message = e?.message || '注册失败，请稍后重试'
      setError(message)
      throw new Error(message)
    } finally {
      isLoading.value = false
    }
  }
  
  /**
   * 用户登出
   * 
   * 清除本地存储的令牌和用户信息
   */
  async function logout(): Promise<void> {
    const api = getApi()
    
    try {
      // 尝试调用服务端登出接口
      if (api) {
        await api.auth.logout()
      }
    } catch {
      // 忽略登出接口错误，继续清理本地状态
    }
    
    // 清除本地状态
    clearToken()
    currentUser.value = null
    profile.value = null
    behaviors.value = []
    clearError()
  }
  
  /**
   * 刷新令牌
   * 
   * @returns 新的令牌
   */
  async function refreshToken(): Promise<string> {
    const api = getApi()
    if (!api) {
      throw new Error('API Provider 未初始化')
    }
    
    try {
      const newToken = await api.auth.refreshToken()
      saveToken(newToken)
      return newToken
    } catch (e: any) {
      // 刷新失败，强制登出
      await logout()
      throw new Error('会话已过期，请重新登录')
    }
  }

  // ==========================================================================
  // 用户信息操作
  // ==========================================================================
  
  /**
   * 获取当前用户信息
   * 
   * 用于页面加载时恢复用户状态
   */
  async function fetchCurrentUser(): Promise<void> {
    if (!token.value) return
    
    const api = getApi()
    if (!api) return
    
    isLoading.value = true
    
    try {
      currentUser.value = await api.auth.getCurrentUser()
    } catch {
      // 获取失败，可能是令牌过期，清除状态
      await logout()
    } finally {
      isLoading.value = false
    }
  }
  
  /**
   * 获取用户画像
   */
  async function fetchProfile(): Promise<void> {
    if (!currentUser.value) return
    
    const api = getApi()
    if (!api) return
    
    isLoading.value = true
    
    try {
      profile.value = await api.user.getProfile(currentUser.value.id)
    } catch (e: any) {
      setError(e?.message || '获取用户画像失败')
    } finally {
      isLoading.value = false
    }
  }
  
  /**
   * 获取用户行为历史
   * 
   * @param limit - 获取数量限制，默认 50
   */
  async function fetchBehaviors(limit = 50): Promise<void> {
    if (!currentUser.value) return
    
    const api = getApi()
    if (!api) return
    
    isLoading.value = true
    
    try {
      behaviors.value = await api.user.getBehaviors(currentUser.value.id, limit)
    } catch (e: any) {
      setError(e?.message || '获取历史记录失败')
    } finally {
      isLoading.value = false
    }
  }
  
  /**
   * 更新用户信息
   * 
   * @param data - 更新数据
   */
  async function updateProfile(data: UpdateUserRequest): Promise<void> {
    if (!currentUser.value) return
    
    const api = getApi()
    if (!api) return
    
    clearError()
    isLoading.value = true
    
    try {
      currentUser.value = await api.user.updateUser(currentUser.value.id, data)
    } catch (e: any) {
      const message = e?.message || '更新用户信息失败'
      setError(message)
      throw new Error(message)
    } finally {
      isLoading.value = false
    }
  }

  // ==========================================================================
  // 初始化
  // ==========================================================================
  
  /**
   * 初始化用户状态
   * 
   * 在应用启动时调用，检查本地令牌并恢复用户状态
   */
  async function init(): Promise<void> {
    if (token.value) {
      await fetchCurrentUser()
    }
  }

  // ==========================================================================
  // 导出
  // ==========================================================================
  
  return {
    // 状态
    token,
    currentUser,
    profile,
    behaviors,
    isLoading,
    error,
    
    // 计算属性
    isLoggedIn,
    userId,
    displayName,
    avatarInitial,
    
    // 认证操作
    login,
    register,
    logout,
    refreshToken,
    
    // 用户信息操作
    fetchCurrentUser,
    fetchProfile,
    fetchBehaviors,
    updateProfile,
    
    // 工具方法
    clearError,
    
    // 初始化
    init,
  }
})

// 导出类型
export type UserStore = ReturnType<typeof useUserStore>

