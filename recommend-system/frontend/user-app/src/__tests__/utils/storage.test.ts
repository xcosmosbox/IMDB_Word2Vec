/**
 * 存储工具函数单元测试
 * 
 * @author Person F
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import {
  setStorage,
  getStorage,
  removeStorage,
  hasStorage,
  clearStorage,
  getAllStorageKeys,
  storage,
  saveAuthToken,
  getAuthToken,
  removeAuthToken,
  clearAuth,
} from '@shared/utils'

describe('存储工具', () => {
  beforeEach(() => {
    // 清理 mock
    vi.clearAllMocks()
    localStorage.clear()
  })

  describe('setStorage', () => {
    it('应该存储数据', () => {
      setStorage('test', { name: '张三' })

      expect(localStorage.setItem).toHaveBeenCalled()
    })

    it('应该支持过期时间', () => {
      setStorage('test', 'value', { expire: 1000 })

      const call = vi.mocked(localStorage.setItem).mock.calls[0]
      const stored = JSON.parse(call[1])

      expect(stored.expire).toBeDefined()
      expect(stored.expire).toBeGreaterThan(Date.now())
    })
  })

  describe('getStorage', () => {
    it('应该获取存储的数据', () => {
      const mockData = JSON.stringify({
        value: { name: '张三' },
        timestamp: Date.now(),
      })

      vi.mocked(localStorage.getItem).mockReturnValue(mockData)

      const result = getStorage<{ name: string }>('test')

      expect(result).toEqual({ name: '张三' })
    })

    it('不存在的键应该返回默认值', () => {
      vi.mocked(localStorage.getItem).mockReturnValue(null)

      const result = getStorage('notexist', 'default')

      expect(result).toBe('default')
    })

    it('过期的数据应该返回默认值', () => {
      const mockData = JSON.stringify({
        value: 'expired',
        timestamp: Date.now() - 10000,
        expire: Date.now() - 5000,
      })

      vi.mocked(localStorage.getItem).mockReturnValue(mockData)

      const result = getStorage('test', 'default')

      expect(result).toBe('default')
      expect(localStorage.removeItem).toHaveBeenCalled()
    })
  })

  describe('removeStorage', () => {
    it('应该移除存储的数据', () => {
      removeStorage('test')

      expect(localStorage.removeItem).toHaveBeenCalled()
    })
  })

  describe('storage 对象', () => {
    it('应该提供便捷方法', () => {
      expect(storage.get).toBeDefined()
      expect(storage.set).toBeDefined()
      expect(storage.remove).toBeDefined()
      expect(storage.has).toBeDefined()
      expect(storage.clear).toBeDefined()
      expect(storage.keys).toBeDefined()
    })
  })

  describe('Token 存储', () => {
    it('saveAuthToken 应该保存 token', () => {
      saveAuthToken('test-token')

      expect(localStorage.setItem).toHaveBeenCalled()
    })

    it('getAuthToken 应该获取 token', () => {
      const mockData = JSON.stringify({
        value: 'test-token',
        timestamp: Date.now(),
      })

      vi.mocked(localStorage.getItem).mockReturnValue(mockData)

      const token = getAuthToken()

      expect(token).toBe('test-token')
    })

    it('removeAuthToken 应该移除 token', () => {
      removeAuthToken()

      expect(localStorage.removeItem).toHaveBeenCalled()
    })

    it('clearAuth 应该清除所有认证数据', () => {
      clearAuth()

      expect(localStorage.removeItem).toHaveBeenCalledTimes(2)
    })
  })
})

