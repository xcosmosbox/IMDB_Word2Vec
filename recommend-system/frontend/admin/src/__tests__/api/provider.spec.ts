/**
 * API Provider 单元测试
 */
import { describe, it, expect, vi } from 'vitest'
import { MockApiProvider, createApiProvider } from '@/api/provider'

describe('MockApiProvider', () => {
  describe('adminUser', () => {
    it('listUsers 应该返回模拟用户列表', async () => {
      const provider = new MockApiProvider()
      
      const result = await provider.adminUser.listUsers({
        page: 1,
        page_size: 10,
      })

      expect(result.items).toBeDefined()
      expect(Array.isArray(result.items)).toBe(true)
      expect(result.total).toBeGreaterThanOrEqual(0)
    })

    it('getUser 应该返回模拟用户', async () => {
      const provider = new MockApiProvider()
      
      const result = await provider.adminUser.getUser('user_1')

      expect(result).toBeDefined()
      expect(result.id).toBe('user_1')
      expect(result.name).toBeDefined()
      expect(result.email).toBeDefined()
    })

    it('createUser 应该返回创建的用户', async () => {
      const provider = new MockApiProvider()
      
      const result = await provider.adminUser.createUser({
        name: '新用户',
        email: 'new@example.com',
      })

      expect(result).toBeDefined()
      expect(result.id).toBeDefined()
      expect(result.name).toBe('新用户')
      expect(result.email).toBe('new@example.com')
    })

    it('updateUser 应该返回更新后的用户', async () => {
      const provider = new MockApiProvider()
      
      const result = await provider.adminUser.updateUser('user_1', {
        name: '更新后的名称',
      })

      expect(result).toBeDefined()
      expect(result.id).toBe('user_1')
      expect(result.name).toBe('更新后的名称')
    })

    it('deleteUser 应该成功执行', async () => {
      const provider = new MockApiProvider()
      
      await expect(provider.adminUser.deleteUser('user_1')).resolves.toBeUndefined()
    })
  })

  describe('adminItem', () => {
    it('listItems 应该返回模拟物品列表', async () => {
      const provider = new MockApiProvider()
      
      const result = await provider.adminItem.listItems({
        page: 1,
        page_size: 10,
      })

      expect(result.items).toBeDefined()
      expect(Array.isArray(result.items)).toBe(true)
      expect(result.total).toBeGreaterThanOrEqual(0)
    })

    it('getItem 应该返回模拟物品', async () => {
      const provider = new MockApiProvider()
      
      const result = await provider.adminItem.getItem('item_1')

      expect(result).toBeDefined()
      expect(result.id).toBe('item_1')
      expect(result.title).toBeDefined()
      expect(result.type).toBeDefined()
    })

    it('createItem 应该返回创建的物品', async () => {
      const provider = new MockApiProvider()
      
      const result = await provider.adminItem.createItem({
        type: 'movie',
        title: '新物品',
      })

      expect(result).toBeDefined()
      expect(result.id).toBeDefined()
      expect(result.title).toBe('新物品')
      expect(result.type).toBe('movie')
    })

    it('updateItem 应该返回更新后的物品', async () => {
      const provider = new MockApiProvider()
      
      const result = await provider.adminItem.updateItem('item_1', {
        title: '更新后的标题',
      })

      expect(result).toBeDefined()
      expect(result.id).toBe('item_1')
      expect(result.title).toBe('更新后的标题')
    })

    it('deleteItem 应该成功执行', async () => {
      const provider = new MockApiProvider()
      
      await expect(provider.adminItem.deleteItem('item_1')).resolves.toBeUndefined()
    })
  })
})

describe('createApiProvider', () => {
  it('默认应该创建 HTTP Provider', () => {
    const provider = createApiProvider(false)
    expect(provider).toBeDefined()
    expect(provider.adminUser).toBeDefined()
    expect(provider.adminItem).toBeDefined()
  })

  it('useMock 为 true 时应该创建 Mock Provider', () => {
    const provider = createApiProvider(true)
    expect(provider).toBeDefined()
    expect(provider).toBeInstanceOf(MockApiProvider)
  })
})

