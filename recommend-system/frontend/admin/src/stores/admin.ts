/**
 * Admin Store - 管理后台状态管理
 * 
 * 功能：
 * - 管理员登录状态
 * - 权限信息
 * - 全局设置
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

/** 管理员信息 */
export interface Admin {
  id: string
  name: string
  email: string
  role: 'super_admin' | 'admin' | 'operator'
  avatar?: string
  permissions: string[]
  created_at: string
  last_login_at?: string
}

/** 登录请求 */
export interface AdminLoginRequest {
  email: string
  password: string
}

/** 登录响应 */
export interface AdminLoginResponse {
  token: string
  admin: Admin
  expires_at: string
}

/**
 * Admin Store
 */
export const useAdminStore = defineStore('admin', () => {
  // 状态
  const token = ref<string | null>(localStorage.getItem('admin_token'))
  const currentAdmin = ref<Admin | null>(null)
  const expiresAt = ref<string | null>(null)

  // 计算属性
  const isLoggedIn = computed(() => !!token.value && !!currentAdmin.value)

  const isTokenExpired = computed(() => {
    if (!expiresAt.value) return true
    return new Date(expiresAt.value) < new Date()
  })

  const adminName = computed(() => currentAdmin.value?.name || '管理员')

  const adminRole = computed(() => currentAdmin.value?.role || 'operator')

  const isSuperAdmin = computed(() => currentAdmin.value?.role === 'super_admin')

  // 检查权限
  function hasPermission(permission: string): boolean {
    if (!currentAdmin.value) return false
    if (currentAdmin.value.role === 'super_admin') return true
    return currentAdmin.value.permissions.includes(permission)
  }

  // 检查多个权限（任意一个）
  function hasAnyPermission(permissions: string[]): boolean {
    return permissions.some(p => hasPermission(p))
  }

  // 检查多个权限（全部）
  function hasAllPermissions(permissions: string[]): boolean {
    return permissions.every(p => hasPermission(p))
  }

  // 登录
  async function login(credentials: AdminLoginRequest): Promise<void> {
    // 模拟登录请求
    // 实际项目中应调用 API
    await new Promise(resolve => setTimeout(resolve, 1000))

    // 模拟响应
    const response: AdminLoginResponse = {
      token: `admin_token_${Date.now()}`,
      admin: {
        id: 'admin_1',
        name: '系统管理员',
        email: credentials.email,
        role: 'super_admin',
        permissions: ['user:read', 'user:write', 'item:read', 'item:write'],
        created_at: new Date().toISOString(),
        last_login_at: new Date().toISOString(),
      },
      expires_at: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
    }

    // 保存状态
    token.value = response.token
    currentAdmin.value = response.admin
    expiresAt.value = response.expires_at

    // 持久化 token
    localStorage.setItem('admin_token', response.token)
  }

  // 登出
  async function logout(): Promise<void> {
    // 清除状态
    token.value = null
    currentAdmin.value = null
    expiresAt.value = null

    // 清除持久化数据
    localStorage.removeItem('admin_token')
  }

  // 刷新 token
  async function refreshToken(): Promise<void> {
    if (!token.value) {
      throw new Error('No token to refresh')
    }

    // 模拟刷新请求
    await new Promise(resolve => setTimeout(resolve, 500))

    // 更新过期时间
    expiresAt.value = new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
  }

  // 获取当前管理员信息
  async function fetchCurrentAdmin(): Promise<void> {
    if (!token.value) {
      throw new Error('Not logged in')
    }

    // 模拟请求
    await new Promise(resolve => setTimeout(resolve, 300))

    // 如果已经有管理员信息，直接返回
    if (currentAdmin.value) return

    // 模拟获取管理员信息
    currentAdmin.value = {
      id: 'admin_1',
      name: '系统管理员',
      email: 'admin@example.com',
      role: 'super_admin',
      permissions: ['user:read', 'user:write', 'item:read', 'item:write'],
      created_at: new Date().toISOString(),
      last_login_at: new Date().toISOString(),
    }
  }

  // 更新管理员信息
  function updateAdminInfo(info: Partial<Admin>): void {
    if (currentAdmin.value) {
      currentAdmin.value = {
        ...currentAdmin.value,
        ...info,
      }
    }
  }

  // 初始化（从本地存储恢复状态）
  async function initialize(): Promise<void> {
    const savedToken = localStorage.getItem('admin_token')
    if (savedToken) {
      token.value = savedToken
      try {
        await fetchCurrentAdmin()
      } catch {
        // Token 无效，清除
        await logout()
      }
    }
  }

  return {
    // 状态
    token,
    currentAdmin,
    expiresAt,

    // 计算属性
    isLoggedIn,
    isTokenExpired,
    adminName,
    adminRole,
    isSuperAdmin,

    // 方法
    hasPermission,
    hasAnyPermission,
    hasAllPermissions,
    login,
    logout,
    refreshToken,
    fetchCurrentAdmin,
    updateAdminInfo,
    initialize,
  }
})

export default useAdminStore

