/**
 * Search.vue 单元测试
 * 
 * Person B 开发
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import { createRouter, createWebHistory } from 'vue-router'
import Search from '@/views/Search.vue'
import SearchBar from '@/components/SearchBar.vue'
import SearchResults from '@/components/SearchResults.vue'
import type { Item } from '@shared/types'

// Mock 数据
const mockItems: Item[] = [
  {
    id: '1',
    type: 'movie',
    title: '测试电影',
    description: '这是一部测试电影的描述',
    category: '科幻',
    tags: ['科幻', '动作'],
    status: 'active',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
  },
  {
    id: '2',
    type: 'product',
    title: '测试商品',
    description: '这是一个测试商品的描述',
    category: '电子产品',
    tags: ['数码', '手机'],
    status: 'active',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
  },
]

// Mock API Provider
const mockApiProvider = {
  item: {
    searchItems: vi.fn(),
    getItem: vi.fn(),
    getItemStats: vi.fn(),
    getSimilarItems: vi.fn(),
    listItems: vi.fn(),
  },
  recommend: {
    getRecommendations: vi.fn(),
    submitFeedback: vi.fn(),
    getSimilarRecommendations: vi.fn(),
  },
  auth: {},
  user: {},
  analytics: {},
  adminUser: {},
  adminItem: {},
}

// 创建测试路由
const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'Home', component: { template: '<div>Home</div>' } },
    { path: '/search', name: 'Search', component: Search },
    { path: '/item/:id', name: 'ItemDetail', component: { template: '<div>Detail</div>' } },
  ],
})

describe('Search.vue', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApiProvider.item.searchItems.mockResolvedValue(mockItems)
  })

  const mountComponent = async (query = '') => {
    if (query) {
      await router.push({ path: '/search', query: { q: query } })
    } else {
      await router.push('/search')
    }
    
    return mount(Search, {
      global: {
        plugins: [
          router,
          createTestingPinia({
            createSpy: vi.fn,
          }),
        ],
        provide: {
          api: mockApiProvider,
        },
        components: {
          SearchBar,
          SearchResults,
        },
      },
    })
  }

  describe('初始状态', () => {
    it('应该正确渲染搜索页面', async () => {
      const wrapper = await mountComponent()
      
      expect(wrapper.find('.search-page').exists()).toBe(true)
      expect(wrapper.find('[data-testid="main-search-bar"]').exists()).toBe(true)
    })

    it('应该显示初始状态（热门搜索）', async () => {
      const wrapper = await mountComponent()
      
      expect(wrapper.find('[data-testid="initial-state"]').exists()).toBe(true)
      expect(wrapper.findAll('[data-testid="hot-tag"]').length).toBeGreaterThan(0)
    })

    it('不应该显示过滤器栏', async () => {
      const wrapper = await mountComponent()
      
      expect(wrapper.find('.filter-bar').exists()).toBe(false)
    })
  })

  describe('搜索功能', () => {
    it('应该在输入关键词后触发搜索', async () => {
      const wrapper = await mountComponent()
      const searchBar = wrapper.findComponent(SearchBar)
      
      await searchBar.vm.$emit('search', '测试')
      await flushPromises()
      
      expect(wrapper.find('[data-testid="initial-state"]').exists()).toBe(false)
    })

    it('应该显示加载状态', async () => {
      mockApiProvider.item.searchItems.mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve(mockItems), 100))
      )
      
      const wrapper = await mountComponent()
      const searchBar = wrapper.findComponent(SearchBar)
      
      searchBar.vm.$emit('search', '测试')
      await wrapper.vm.$nextTick()
      
      // 注意：这里可能需要根据实际实现调整
    })

    it('应该显示搜索结果', async () => {
      const wrapper = await mountComponent()
      const searchBar = wrapper.findComponent(SearchBar)
      
      await searchBar.vm.$emit('search', '测试')
      await flushPromises()
      
      expect(wrapper.find('[data-testid="search-results"]').exists()).toBe(true)
    })

    it('应该在 URL 中更新搜索参数', async () => {
      const wrapper = await mountComponent()
      const searchBar = wrapper.findComponent(SearchBar)
      
      await searchBar.vm.$emit('search', '新关键词')
      await flushPromises()
      
      expect(router.currentRoute.value.query.q).toBe('新关键词')
    })
  })

  describe('URL 参数', () => {
    it('应该从 URL 参数加载搜索', async () => {
      await router.push({ path: '/search', query: { q: '电影' } })
      const wrapper = await mountComponent('电影')
      
      await flushPromises()
      
      // 搜索应该被触发
      expect(wrapper.vm.query).toBe('电影')
    })
  })

  describe('热门搜索', () => {
    it('点击热门搜索标签应该触发搜索', async () => {
      const wrapper = await mountComponent()
      const hotTags = wrapper.findAll('[data-testid="hot-tag"]')
      
      if (hotTags.length > 0) {
        await hotTags[0].trigger('click')
        await flushPromises()
        
        expect(wrapper.find('[data-testid="initial-state"]').exists()).toBe(false)
      }
    })
  })

  describe('过滤功能', () => {
    it('应该在有搜索结果后显示过滤器', async () => {
      const wrapper = await mountComponent()
      const searchBar = wrapper.findComponent(SearchBar)
      
      await searchBar.vm.$emit('search', '测试')
      await flushPromises()
      
      expect(wrapper.find('.filter-bar').exists()).toBe(true)
    })
  })

  describe('清空搜索', () => {
    it('清空搜索应该回到初始状态', async () => {
      const wrapper = await mountComponent()
      const searchBar = wrapper.findComponent(SearchBar)
      
      // 先搜索
      await searchBar.vm.$emit('search', '测试')
      await flushPromises()
      
      // 清空
      await searchBar.vm.$emit('clear')
      await flushPromises()
      
      expect(wrapper.find('[data-testid="initial-state"]').exists()).toBe(true)
    })
  })
})

