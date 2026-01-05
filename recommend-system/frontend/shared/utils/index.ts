/**
 * 工具函数统一导出
 * 
 * @module shared/utils
 * @author Person F
 */

// =============================================================================
// 存储工具
// =============================================================================

export {
  // 核心函数
  setStorage,
  getStorage,
  removeStorage,
  hasStorage,
  clearStorage,
  getAllStorageKeys,
  
  // 便捷对象
  storage,
  sessionStore,
  
  // 服务类
  StorageService,
  
  // Token 相关
  saveAuthToken,
  getAuthToken,
  removeAuthToken,
  saveCurrentUser,
  getCurrentUser,
  removeCurrentUser,
  clearAuth,
  
  // 类型
  type StorageOptions,
} from './storage'

// =============================================================================
// 格式化工具
// =============================================================================

export {
  // 日期格式化
  formatDate,
  formatRelativeTime,
  formatDateRange,
  formatDuration,
  
  // 数字格式化
  formatNumber,
  formatLargeNumber,
  formatPercent,
  formatMoney,
  formatFileSize,
  
  // 文本格式化
  truncate,
  capitalize,
  toCamelCase,
  toKebabCase,
  toSnakeCase,
  maskPhone,
  maskEmail,
  maskIdCard,
  
  // 业务格式化
  formatItemType,
  formatAction,
  formatGender,
  formatRating,
  ratingToStars,
} from './format'

// =============================================================================
// 验证工具
// =============================================================================

/**
 * 验证邮箱格式
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email)
}

/**
 * 验证手机号格式（中国大陆）
 */
export function isValidPhone(phone: string): boolean {
  const phoneRegex = /^1[3-9]\d{9}$/
  return phoneRegex.test(phone)
}

/**
 * 验证密码强度
 * - 至少 8 个字符
 * - 包含大写字母
 * - 包含小写字母
 * - 包含数字
 */
export function validatePassword(password: string): {
  valid: boolean
  message: string
} {
  if (password.length < 8) {
    return { valid: false, message: '密码至少需要 8 个字符' }
  }
  if (!/[A-Z]/.test(password)) {
    return { valid: false, message: '密码需要包含大写字母' }
  }
  if (!/[a-z]/.test(password)) {
    return { valid: false, message: '密码需要包含小写字母' }
  }
  if (!/\d/.test(password)) {
    return { valid: false, message: '密码需要包含数字' }
  }
  return { valid: true, message: '密码强度符合要求' }
}

/**
 * 验证 URL 格式
 */
export function isValidUrl(url: string): boolean {
  try {
    new URL(url)
    return true
  } catch {
    return false
  }
}

// =============================================================================
// 防抖与节流
// =============================================================================

/**
 * 防抖函数
 * 
 * @param fn - 要防抖的函数
 * @param delay - 延迟时间（毫秒）
 * @returns 防抖后的函数
 * 
 * @example
 * ```typescript
 * const debouncedSearch = debounce(search, 300)
 * input.addEventListener('input', debouncedSearch)
 * ```
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  
  return function (this: unknown, ...args: Parameters<T>) {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
    
    timeoutId = setTimeout(() => {
      fn.apply(this, args)
      timeoutId = null
    }, delay)
  }
}

/**
 * 节流函数
 * 
 * @param fn - 要节流的函数
 * @param limit - 时间限制（毫秒）
 * @returns 节流后的函数
 * 
 * @example
 * ```typescript
 * const throttledScroll = throttle(handleScroll, 100)
 * window.addEventListener('scroll', throttledScroll)
 * ```
 */
export function throttle<T extends (...args: unknown[]) => unknown>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => void {
  let lastCall = 0
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  
  return function (this: unknown, ...args: Parameters<T>) {
    const now = Date.now()
    const remaining = limit - (now - lastCall)
    
    if (remaining <= 0) {
      if (timeoutId) {
        clearTimeout(timeoutId)
        timeoutId = null
      }
      lastCall = now
      fn.apply(this, args)
    } else if (!timeoutId) {
      timeoutId = setTimeout(() => {
        lastCall = Date.now()
        timeoutId = null
        fn.apply(this, args)
      }, remaining)
    }
  }
}

// =============================================================================
// 异步工具
// =============================================================================

/**
 * 延迟执行
 * 
 * @param ms - 延迟毫秒数
 * @returns Promise
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * 重试函数
 * 
 * @param fn - 要重试的异步函数
 * @param options - 重试选项
 * @returns Promise
 * 
 * @example
 * ```typescript
 * const result = await retry(() => fetchData(), { maxAttempts: 3, delay: 1000 })
 * ```
 */
export async function retry<T>(
  fn: () => Promise<T>,
  options: {
    maxAttempts?: number
    delay?: number
    onRetry?: (error: unknown, attempt: number) => void
  } = {}
): Promise<T> {
  const { maxAttempts = 3, delay = 1000, onRetry } = options
  let lastError: unknown
  
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn()
    } catch (error) {
      lastError = error
      
      if (attempt < maxAttempts) {
        onRetry?.(error, attempt)
        await sleep(delay * attempt) // 指数退避
      }
    }
  }
  
  throw lastError
}

// =============================================================================
// 对象工具
// =============================================================================

/**
 * 深拷贝对象
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') {
    return obj
  }
  
  if (obj instanceof Date) {
    return new Date(obj.getTime()) as T
  }
  
  if (obj instanceof Array) {
    return obj.map(item => deepClone(item)) as T
  }
  
  if (obj instanceof Object) {
    const copy = {} as T
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        (copy as Record<string, unknown>)[key] = deepClone((obj as Record<string, unknown>)[key])
      }
    }
    return copy
  }
  
  return obj
}

/**
 * 深度合并对象
 */
export function deepMerge<T extends object>(target: T, ...sources: Partial<T>[]): T {
  if (!sources.length) return target
  
  const source = sources.shift()
  
  if (source === undefined) return target
  
  for (const key in source) {
    if (Object.prototype.hasOwnProperty.call(source, key)) {
      const sourceValue = source[key]
      const targetValue = target[key]
      
      if (
        sourceValue &&
        typeof sourceValue === 'object' &&
        !Array.isArray(sourceValue) &&
        targetValue &&
        typeof targetValue === 'object' &&
        !Array.isArray(targetValue)
      ) {
        target[key] = deepMerge({ ...targetValue }, sourceValue as Partial<typeof targetValue>)
      } else {
        target[key] = sourceValue as T[typeof key]
      }
    }
  }
  
  return deepMerge(target, ...sources)
}

/**
 * 移除对象中的空值
 */
export function removeEmpty<T extends object>(obj: T): Partial<T> {
  const result: Partial<T> = {}
  
  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      const value = obj[key]
      if (value !== null && value !== undefined && value !== '') {
        result[key] = value
      }
    }
  }
  
  return result
}

/**
 * 从对象中选取指定属性
 */
export function pick<T extends object, K extends keyof T>(
  obj: T,
  keys: K[]
): Pick<T, K> {
  const result = {} as Pick<T, K>
  
  for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      result[key] = obj[key]
    }
  }
  
  return result
}

/**
 * 从对象中排除指定属性
 */
export function omit<T extends object, K extends keyof T>(
  obj: T,
  keys: K[]
): Omit<T, K> {
  const result = { ...obj }
  
  for (const key of keys) {
    delete result[key]
  }
  
  return result
}

// =============================================================================
// 数组工具
// =============================================================================

/**
 * 数组去重
 */
export function unique<T>(arr: T[], key?: keyof T): T[] {
  if (!key) {
    return [...new Set(arr)]
  }
  
  const seen = new Set()
  return arr.filter(item => {
    const value = item[key]
    if (seen.has(value)) {
      return false
    }
    seen.add(value)
    return true
  })
}

/**
 * 数组分组
 */
export function groupBy<T>(
  arr: T[],
  key: keyof T | ((item: T) => string)
): Record<string, T[]> {
  return arr.reduce((groups, item) => {
    const groupKey = typeof key === 'function' ? key(item) : String(item[key])
    
    if (!groups[groupKey]) {
      groups[groupKey] = []
    }
    groups[groupKey].push(item)
    
    return groups
  }, {} as Record<string, T[]>)
}

/**
 * 数组分块
 */
export function chunk<T>(arr: T[], size: number): T[][] {
  const chunks: T[][] = []
  
  for (let i = 0; i < arr.length; i += size) {
    chunks.push(arr.slice(i, i + size))
  }
  
  return chunks
}

/**
 * 打乱数组
 */
export function shuffle<T>(arr: T[]): T[] {
  const result = [...arr]
  
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[result[i], result[j]] = [result[j], result[i]]
  }
  
  return result
}

// =============================================================================
// ID 生成
// =============================================================================

/**
 * 生成随机 ID
 */
export function generateId(length = 12): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
  let result = ''
  
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  
  return result
}

/**
 * 生成 UUID v4
 */
export function uuid(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

// =============================================================================
// URL 工具
// =============================================================================

/**
 * 构建 URL 查询字符串
 */
export function buildQueryString(params: Record<string, unknown>): string {
  const searchParams = new URLSearchParams()
  
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      searchParams.append(key, String(value))
    }
  }
  
  return searchParams.toString()
}

/**
 * 解析 URL 查询字符串
 */
export function parseQueryString(query: string): Record<string, string> {
  const params: Record<string, string> = {}
  const searchParams = new URLSearchParams(query)
  
  searchParams.forEach((value, key) => {
    params[key] = value
  })
  
  return params
}

// =============================================================================
// 剪贴板
// =============================================================================

/**
 * 复制文本到剪贴板
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text)
      return true
    }
    
    // Fallback
    const textArea = document.createElement('textarea')
    textArea.value = text
    textArea.style.position = 'fixed'
    textArea.style.left = '-9999px'
    document.body.appendChild(textArea)
    textArea.select()
    
    try {
      document.execCommand('copy')
      return true
    } finally {
      document.body.removeChild(textArea)
    }
  } catch (e) {
    console.error('Failed to copy:', e)
    return false
  }
}

// =============================================================================
// 类型检查
// =============================================================================

/**
 * 检查是否为空值
 */
export function isEmpty(value: unknown): boolean {
  if (value === null || value === undefined) return true
  if (typeof value === 'string') return value.trim() === ''
  if (Array.isArray(value)) return value.length === 0
  if (typeof value === 'object') return Object.keys(value).length === 0
  return false
}

/**
 * 检查是否为对象
 */
export function isObject(value: unknown): value is object {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

/**
 * 检查是否为函数
 */
export function isFunction(value: unknown): value is (...args: unknown[]) => unknown {
  return typeof value === 'function'
}

/**
 * 检查是否为 Promise
 */
export function isPromise<T>(value: unknown): value is Promise<T> {
  return (
    value !== null &&
    typeof value === 'object' &&
    typeof (value as Promise<T>).then === 'function'
  )
}

