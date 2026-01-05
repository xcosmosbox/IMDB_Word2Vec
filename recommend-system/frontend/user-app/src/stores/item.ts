/**
 * 物品状态管理 Store
 * 
 * 负责物品相关的状态管理和 API 调用
 * 使用依赖注入获取 API Provider
 */
import { defineStore } from 'pinia'
import { ref, computed, inject } from 'vue'
import type { Item, ItemStats, SimilarItem, ListItemsRequest, ListItemsResponse } from '@shared/types'
import type { IApiProvider } from '@shared/api/interfaces'

export const useItemStore = defineStore('item', () => {
  // API Provider 注入
  const api = inject<IApiProvider>('api')
  
  // =========================================================================
  // 状态
  // =========================================================================
  
  /** 当前查看的物品 */
  const currentItem = ref<Item | null>(null)
  
  /** 当前物品统计 */
  const currentItemStats = ref<ItemStats | null>(null)
  
  /** 搜索结果列表 */
  const searchResults = ref<Item[]>([])
  
  /** 搜索关键词 */
  const searchQuery = ref('')
  
  /** 是否正在加载 */
  const isLoading = ref(false)
  
  /** 是否正在搜索 */
  const isSearching = ref(false)
  
  /** 相似物品列表 */
  const similarItems = ref<SimilarItem[]>([])
  
  /** 物品缓存 (id -> Item) */
  const itemCache = ref<Map<string, Item>>(new Map())
  
  /** 最近浏览记录 */
  const recentlyViewed = ref<Item[]>([])
  
  /** 错误信息 */
  const error = ref<string | null>(null)
  
  // =========================================================================
  // 计算属性
  // =========================================================================
  
  /** 是否有搜索结果 */
  const hasSearchResults = computed(() => searchResults.value.length > 0)
  
  /** 搜索结果数量 */
  const searchResultCount = computed(() => searchResults.value.length)
  
  /** 最近浏览数量 */
  const recentlyViewedCount = computed(() => recentlyViewed.value.length)
  
  // =========================================================================
  // 方法
  // =========================================================================
  
  /**
   * 获取物品详情
   * @param itemId 物品ID
   * @returns 物品详情
   */
  async function getItem(itemId: string): Promise<Item> {
    // 优先从缓存获取
    const cached = itemCache.value.get(itemId)
    if (cached) {
      currentItem.value = cached
      return cached
    }
    
    if (!api) {
      throw new Error('API Provider not injected')
    }
    
    isLoading.value = true
    error.value = null
    
    try {
      const item = await api.item.getItem(itemId)
      
      // 更新状态和缓存
      currentItem.value = item
      itemCache.value.set(itemId, item)
      
      // 添加到最近浏览
      addToRecentlyViewed(item)
      
      return item
    } catch (e) {
      error.value = e instanceof Error ? e.message : '获取物品失败'
      throw e
    } finally {
      isLoading.value = false
    }
  }
  
  /**
   * 搜索物品
   * @param query 搜索关键词
   * @param limit 返回数量限制
   * @returns 搜索结果列表
   */
  async function searchItems(query: string, limit: number = 50): Promise<Item[]> {
    if (!api) {
      throw new Error('API Provider not injected')
    }
    
    if (!query.trim()) {
      searchResults.value = []
      return []
    }
    
    isSearching.value = true
    searchQuery.value = query
    error.value = null
    
    try {
      const results = await api.item.searchItems(query, limit)
      searchResults.value = results
      
      // 缓存搜索结果
      results.forEach(item => {
        itemCache.value.set(item.id, item)
      })
      
      return results
    } catch (e) {
      error.value = e instanceof Error ? e.message : '搜索失败'
      throw e
    } finally {
      isSearching.value = false
    }
  }
  
  /**
   * 获取物品统计
   * @param itemId 物品ID
   * @returns 物品统计数据
   */
  async function getItemStats(itemId: string): Promise<ItemStats> {
    if (!api) {
      throw new Error('API Provider not injected')
    }
    
    try {
      const stats = await api.item.getItemStats(itemId)
      currentItemStats.value = stats
      return stats
    } catch (e) {
      error.value = e instanceof Error ? e.message : '获取统计失败'
      throw e
    }
  }
  
  /**
   * 获取相似物品
   * @param itemId 物品ID
   * @param limit 返回数量限制
   * @returns 相似物品列表
   */
  async function getSimilarItems(itemId: string, limit: number = 12): Promise<SimilarItem[]> {
    if (!api) {
      throw new Error('API Provider not injected')
    }
    
    try {
      const similar = await api.item.getSimilarItems(itemId, limit)
      similarItems.value = similar
      
      // 缓存相似物品
      similar.forEach(({ item }) => {
        itemCache.value.set(item.id, item)
      })
      
      return similar
    } catch (e) {
      error.value = e instanceof Error ? e.message : '获取相似物品失败'
      throw e
    }
  }
  
  /**
   * 列出物品
   * @param params 请求参数
   * @returns 物品列表响应
   */
  async function listItems(params: ListItemsRequest): Promise<ListItemsResponse> {
    if (!api) {
      throw new Error('API Provider not injected')
    }
    
    isLoading.value = true
    error.value = null
    
    try {
      const response = await api.item.listItems(params)
      
      // 缓存物品
      response.items.forEach(item => {
        itemCache.value.set(item.id, item)
      })
      
      return response
    } catch (e) {
      error.value = e instanceof Error ? e.message : '获取物品列表失败'
      throw e
    } finally {
      isLoading.value = false
    }
  }
  
  /**
   * 添加到最近浏览
   * @param item 物品
   */
  function addToRecentlyViewed(item: Item) {
    const maxRecent = 20
    
    // 移除已存在的记录
    const filtered = recentlyViewed.value.filter(i => i.id !== item.id)
    
    // 添加到头部
    recentlyViewed.value = [item, ...filtered].slice(0, maxRecent)
    
    // 持久化到 localStorage
    try {
      localStorage.setItem('recentlyViewed', JSON.stringify(recentlyViewed.value))
    } catch (e) {
      console.warn('Failed to persist recently viewed items')
    }
  }
  
  /**
   * 从 localStorage 恢复最近浏览
   */
  function loadRecentlyViewed() {
    try {
      const stored = localStorage.getItem('recentlyViewed')
      if (stored) {
        recentlyViewed.value = JSON.parse(stored)
      }
    } catch (e) {
      console.warn('Failed to load recently viewed items')
    }
  }
  
  /**
   * 清除搜索结果
   */
  function clearSearchResults() {
    searchResults.value = []
    searchQuery.value = ''
  }
  
  /**
   * 清除缓存
   */
  function clearCache() {
    itemCache.value.clear()
  }
  
  /**
   * 重置状态
   */
  function $reset() {
    currentItem.value = null
    currentItemStats.value = null
    searchResults.value = []
    searchQuery.value = ''
    similarItems.value = []
    isLoading.value = false
    isSearching.value = false
    error.value = null
    // 保留缓存和最近浏览
  }
  
  // 初始化时加载最近浏览
  loadRecentlyViewed()
  
  return {
    // 状态
    currentItem,
    currentItemStats,
    searchResults,
    searchQuery,
    isLoading,
    isSearching,
    similarItems,
    itemCache,
    recentlyViewed,
    error,
    
    // 计算属性
    hasSearchResults,
    searchResultCount,
    recentlyViewedCount,
    
    // 方法
    getItem,
    searchItems,
    getItemStats,
    getSimilarItems,
    listItems,
    addToRecentlyViewed,
    loadRecentlyViewed,
    clearSearchResults,
    clearCache,
    $reset,
  }
})

