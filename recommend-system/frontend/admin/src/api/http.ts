/**
 * HTTP 客户端封装
 * 
 * 基于 Axios 封装，提供统一的请求/响应处理
 */

import axios, { type AxiosInstance, type AxiosResponse, type InternalAxiosRequestConfig } from 'axios'
import { message } from 'ant-design-vue'

/**
 * API 响应数据结构
 */
interface ApiResponse<T = any> {
  code: number
  data: T
  message?: string
}

/**
 * 创建 Axios 实例
 */
const createHttpInstance = (): AxiosInstance => {
  const instance = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || '/api/v1',
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
    },
  })

  // 请求拦截器
  instance.interceptors.request.use(
    (config: InternalAxiosRequestConfig) => {
      // 从 localStorage 获取 token
      const token = localStorage.getItem('admin_token')
      if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`
      }
      return config
    },
    (error) => {
      return Promise.reject(error)
    }
  )

  // 响应拦截器
  instance.interceptors.response.use(
    (response: AxiosResponse<ApiResponse>) => {
      const { code, data, message: msg } = response.data
      
      // 业务成功
      if (code === 0 || code === 200) {
        return { ...response, data }
      }
      
      // 业务错误
      message.error(msg || '请求失败')
      return Promise.reject(new Error(msg || '请求失败'))
    },
    (error) => {
      // HTTP 错误处理
      if (error.response) {
        const { status, data } = error.response
        
        switch (status) {
          case 401:
            message.error('登录已过期，请重新登录')
            // 清除 token 并跳转登录页
            localStorage.removeItem('admin_token')
            window.location.href = '/admin/login'
            break
          case 403:
            message.error('没有权限访问')
            break
          case 404:
            message.error('请求的资源不存在')
            break
          case 500:
            message.error('服务器内部错误')
            break
          default:
            message.error(data?.message || '请求失败')
        }
      } else if (error.request) {
        message.error('网络错误，请检查网络连接')
      } else {
        message.error('请求配置错误')
      }
      
      return Promise.reject(error)
    }
  )

  return instance
}

// 导出 HTTP 实例
export const http = createHttpInstance()

/**
 * HTTP 请求方法封装
 */
export const request = {
  get: <T = any>(url: string, config?: any) => http.get<T>(url, config),
  post: <T = any>(url: string, data?: any, config?: any) => http.post<T>(url, data, config),
  put: <T = any>(url: string, data?: any, config?: any) => http.put<T>(url, data, config),
  delete: <T = any>(url: string, config?: any) => http.delete<T>(url, config),
  patch: <T = any>(url: string, data?: any, config?: any) => http.patch<T>(url, data, config),
}

