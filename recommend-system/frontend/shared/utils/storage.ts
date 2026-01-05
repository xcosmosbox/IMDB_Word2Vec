/**
 * 本地存储工具模块
 * 
 * 提供类型安全的 localStorage 和 sessionStorage 操作
 * 支持序列化/反序列化、过期时间、前缀命名空间等功能
 * 
 * @module shared/utils/storage
 * @author Person F
 */

import type { IStorageService } from '../api/interfaces'

// =============================================================================
// 配置
// =============================================================================

/** 存储前缀 */
const STORAGE_PREFIX = 'recommend_'

/** 默认过期时间（毫秒），7 天 */
const DEFAULT_EXPIRE_TIME = 7 * 24 * 60 * 60 * 1000

// =============================================================================
// 类型定义
// =============================================================================

/**
 * 存储数据包装器
 */
interface StorageWrapper<T> {
  value: T
  expire?: number
  timestamp: number
}

/**
 * 存储选项
 */
export interface StorageOptions {
  /** 过期时间（毫秒） */
  expire?: number
  /** 存储类型 */
  type?: 'local' | 'session'
}

// =============================================================================
// 工具函数
// =============================================================================

/**
 * 获取完整的存储 key
 */
function getFullKey(key: string): string {
  return `${STORAGE_PREFIX}${key}`
}

/**
 * 获取存储实例
 */
function getStorage(type: 'local' | 'session' = 'local'): Storage {
  return type === 'session' ? sessionStorage : localStorage
}

/**
 * 检查是否已过期
 */
function isExpired(wrapper: StorageWrapper<unknown>): boolean {
  if (!wrapper.expire) return false
  return Date.now() > wrapper.expire
}

// =============================================================================
// 核心存储函数
// =============================================================================

/**
 * 存储数据
 * 
 * @param key - 存储键
 * @param value - 存储值
 * @param options - 存储选项
 * 
 * @example
 * ```typescript
 * // 基础存储
 * setStorage('user', { name: '张三' })
 * 
 * // 设置过期时间（1小时）
 * setStorage('token', 'xxx', { expire: 3600000 })
 * 
 * // 使用 sessionStorage
 * setStorage('temp', 'data', { type: 'session' })
 * ```
 */
export function setStorage<T>(key: string, value: T, options: StorageOptions = {}): void {
  const storage = getStorage(options.type)
  const fullKey = getFullKey(key)
  
  const wrapper: StorageWrapper<T> = {
    value,
    timestamp: Date.now(),
  }
  
  if (options.expire !== undefined) {
    wrapper.expire = Date.now() + options.expire
  }
  
  try {
    storage.setItem(fullKey, JSON.stringify(wrapper))
  } catch (e) {
    console.warn(`Failed to set storage for key "${key}":`, e)
  }
}

/**
 * 获取存储数据
 * 
 * @param key - 存储键
 * @param defaultValue - 默认值
 * @param options - 存储选项
 * @returns 存储的值或默认值
 * 
 * @example
 * ```typescript
 * const user = getStorage<User>('user')
 * const token = getStorage('token', '', { type: 'session' })
 * ```
 */
export function getStorage<T>(
  key: string,
  defaultValue: T | null = null,
  options: Pick<StorageOptions, 'type'> = {}
): T | null {
  const storage = getStorage(options.type)
  const fullKey = getFullKey(key)
  
  try {
    const raw = storage.getItem(fullKey)
    if (!raw) return defaultValue
    
    const wrapper: StorageWrapper<T> = JSON.parse(raw)
    
    // 检查过期
    if (isExpired(wrapper)) {
      storage.removeItem(fullKey)
      return defaultValue
    }
    
    return wrapper.value
  } catch (e) {
    console.warn(`Failed to get storage for key "${key}":`, e)
    return defaultValue
  }
}

/**
 * 移除存储数据
 * 
 * @param key - 存储键
 * @param options - 存储选项
 */
export function removeStorage(key: string, options: Pick<StorageOptions, 'type'> = {}): void {
  const storage = getStorage(options.type)
  const fullKey = getFullKey(key)
  storage.removeItem(fullKey)
}

/**
 * 检查存储数据是否存在
 * 
 * @param key - 存储键
 * @param options - 存储选项
 * @returns 是否存在
 */
export function hasStorage(key: string, options: Pick<StorageOptions, 'type'> = {}): boolean {
  return getStorage(key, undefined, options) !== undefined
}

/**
 * 清除所有带前缀的存储数据
 * 
 * @param options - 存储选项
 */
export function clearStorage(options: Pick<StorageOptions, 'type'> = {}): void {
  const storage = getStorage(options.type)
  const keysToRemove: string[] = []
  
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i)
    if (key?.startsWith(STORAGE_PREFIX)) {
      keysToRemove.push(key)
    }
  }
  
  keysToRemove.forEach(key => storage.removeItem(key))
}

/**
 * 获取所有带前缀的存储键
 * 
 * @param options - 存储选项
 * @returns 存储键数组
 */
export function getAllStorageKeys(options: Pick<StorageOptions, 'type'> = {}): string[] {
  const storage = getStorage(options.type)
  const keys: string[] = []
  
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i)
    if (key?.startsWith(STORAGE_PREFIX)) {
      keys.push(key.replace(STORAGE_PREFIX, ''))
    }
  }
  
  return keys
}

// =============================================================================
// 便捷存储对象
// =============================================================================

/**
 * localStorage 存储对象
 */
export const storage = {
  get: <T>(key: string, defaultValue: T | null = null) => 
    getStorage<T>(key, defaultValue, { type: 'local' }),
  set: <T>(key: string, value: T, expire?: number) => 
    setStorage(key, value, { type: 'local', expire }),
  remove: (key: string) => removeStorage(key, { type: 'local' }),
  has: (key: string) => hasStorage(key, { type: 'local' }),
  clear: () => clearStorage({ type: 'local' }),
  keys: () => getAllStorageKeys({ type: 'local' }),
}

/**
 * sessionStorage 存储对象
 */
export const sessionStore = {
  get: <T>(key: string, defaultValue: T | null = null) => 
    getStorage<T>(key, defaultValue, { type: 'session' }),
  set: <T>(key: string, value: T, expire?: number) => 
    setStorage(key, value, { type: 'session', expire }),
  remove: (key: string) => removeStorage(key, { type: 'session' }),
  has: (key: string) => hasStorage(key, { type: 'session' }),
  clear: () => clearStorage({ type: 'session' }),
  keys: () => getAllStorageKeys({ type: 'session' }),
}

// =============================================================================
// 服务接口实现
// =============================================================================

/**
 * 存储服务类
 * 
 * 实现 IStorageService 接口
 */
export class StorageService implements IStorageService {
  private storage: ReturnType<typeof getStorage>
  
  constructor(type: 'local' | 'session' = 'local') {
    this.storage = getStorage(type)
  }
  
  get<T>(key: string): T | null {
    return getStorage<T>(key)
  }
  
  set<T>(key: string, value: T): void {
    setStorage(key, value)
  }
  
  remove(key: string): void {
    removeStorage(key)
  }
  
  clear(): void {
    clearStorage()
  }
}

// =============================================================================
// Token 存储便捷函数
// =============================================================================

const TOKEN_KEY = 'auth_token'
const USER_KEY = 'current_user'

/**
 * 保存认证令牌
 */
export function saveAuthToken(token: string, expire = DEFAULT_EXPIRE_TIME): void {
  setStorage(TOKEN_KEY, token, { expire })
}

/**
 * 获取认证令牌
 */
export function getAuthToken(): string | null {
  return getStorage<string>(TOKEN_KEY)
}

/**
 * 移除认证令牌
 */
export function removeAuthToken(): void {
  removeStorage(TOKEN_KEY)
}

/**
 * 保存当前用户
 */
export function saveCurrentUser<T>(user: T): void {
  setStorage(USER_KEY, user)
}

/**
 * 获取当前用户
 */
export function getCurrentUser<T>(): T | null {
  return getStorage<T>(USER_KEY)
}

/**
 * 移除当前用户
 */
export function removeCurrentUser(): void {
  removeStorage(USER_KEY)
}

/**
 * 清除所有认证数据
 */
export function clearAuth(): void {
  removeAuthToken()
  removeCurrentUser()
}

