/**
 * Axios HTTP 请求封装
 * 
 * 功能特性：
 * 1. 统一请求/响应拦截
 * 2. Token 自动注入
 * 3. 错误统一处理
 * 4. 请求 ID 追踪
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosError, InternalAxiosRequestConfig } from 'axios'
import type { ApiResponse, ApiError } from '../types'

// =============================================================================
// 配置
// =============================================================================

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1'
const TIMEOUT = 30000

// Token 存储 key
const TOKEN_KEY = 'recommend_token'

// =============================================================================
// 创建 Axios 实例
// =============================================================================

const instance: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
})

// =============================================================================
// 工具函数
// =============================================================================

/**
 * 生成唯一请求 ID
 */
function generateRequestId(): string {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`
}

/**
 * 获取 Token
 */
export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

/**
 * 设置 Token
 */
export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

/**
 * 移除 Token
 */
export function removeToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

// =============================================================================
// 请求拦截器
// =============================================================================

instance.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // 添加 Token
    const token = getToken()
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    
    // 添加请求 ID（用于追踪）
    config.headers['X-Request-ID'] = generateRequestId()
    
    // 添加时间戳防止缓存
    if (config.method === 'get') {
      config.params = {
        ...config.params,
        _t: Date.now(),
      }
    }
    
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// =============================================================================
// 响应拦截器
// =============================================================================

instance.interceptors.response.use(
  (response) => {
    const data = response.data as ApiResponse<unknown>
    
    // 业务错误处理（code 为 0 或 200 表示成功）
    if (data.code !== 0 && data.code !== 200) {
      const error: ApiError = {
        code: data.code,
        message: data.message || '请求失败',
      }
      return Promise.reject(error)
    }
    
    // 返回业务数据
    return data.data as any
  },
  (error: AxiosError<ApiError>) => {
    // HTTP 错误处理
    const status = error.response?.status
    const data = error.response?.data
    
    let message = '网络错误，请稍后重试'
    
    switch (status) {
      case 400:
        message = data?.message || '请求参数错误'
        break
      case 401:
        message = '登录已过期，请重新登录'
        // 清除 token 并跳转登录
        removeToken()
        // 避免在服务端渲染时报错
        if (typeof window !== 'undefined') {
          window.location.href = '/login'
        }
        break
      case 403:
        message = '没有权限访问'
        break
      case 404:
        message = '请求的资源不存在'
        break
      case 429:
        message = '请求过于频繁，请稍后重试'
        break
      case 500:
        message = '服务器错误'
        break
      case 502:
        message = '网关错误'
        break
      case 503:
        message = '服务暂不可用'
        break
      default:
        if (data?.message) {
          message = data.message
        }
    }
    
    const apiError: ApiError = {
      code: status || -1,
      message,
      details: data?.details,
    }
    
    return Promise.reject(apiError)
  }
)

// =============================================================================
// 请求方法封装
// =============================================================================

export const request = {
  /**
   * GET 请求
   */
  get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return instance.get(url, config)
  },
  
  /**
   * POST 请求
   */
  post<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    return instance.post(url, data, config)
  },
  
  /**
   * PUT 请求
   */
  put<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    return instance.put(url, data, config)
  },
  
  /**
   * DELETE 请求
   */
  delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return instance.delete(url, config)
  },
  
  /**
   * PATCH 请求
   */
  patch<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    return instance.patch(url, data, config)
  },
}

// 导出实例供高级用例使用
export default instance

