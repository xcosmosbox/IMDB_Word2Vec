/**
 * 认证 API 模块
 */

import { request, setToken, removeToken } from './request'
import type {
  User,
  LoginRequest,
  LoginResponse,
  RegisterRequest,
} from '../types'
import type { IAuthService } from './interfaces'

/**
 * 认证服务实现
 */
export class AuthService implements IAuthService {
  /**
   * 用户登录
   */
  async login(credentials: LoginRequest): Promise<LoginResponse> {
    const response = await request.post<LoginResponse>('/auth/login', credentials)
    // 自动保存 token
    if (response.token) {
      setToken(response.token)
    }
    return response
  }
  
  /**
   * 用户注册
   */
  async register(data: RegisterRequest): Promise<void> {
    return request.post('/auth/register', data)
  }
  
  /**
   * 用户登出
   */
  async logout(): Promise<void> {
    try {
      await request.post('/auth/logout')
    } finally {
      // 无论接口是否成功，都清除本地 token
      removeToken()
    }
  }
  
  /**
   * 刷新 Token
   */
  async refreshToken(): Promise<string> {
    const response = await request.post<{ token: string }>('/auth/refresh')
    if (response.token) {
      setToken(response.token)
    }
    return response.token
  }
  
  /**
   * 获取当前用户信息
   */
  async getCurrentUser(): Promise<User> {
    return request.get('/auth/me')
  }
}

// 导出单例
export const authService = new AuthService()

// 导出便捷 API
export const authApi = {
  login: (data: LoginRequest) => authService.login(data),
  register: (data: RegisterRequest) => authService.register(data),
  logout: () => authService.logout(),
  refreshToken: () => authService.refreshToken(),
  getCurrentUser: () => authService.getCurrentUser(),
}

