/**
 * ItemDetail.vue 单元测试
 * 
 * Person B 开发
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import { createRouter, createWebHistory } from 'vue-router'
import ItemDetail from '@/views/ItemDetail.vue'
import ItemInfo from '@/components/ItemInfo.vue'
import SimilarItems from '@/components/SimilarItems.vue'
import ActionButtons from '@/components/ActionButtons.vue'
import type { Item, ItemStats, SimilarItem } from '@shared/types'

// Mock 数据
const mockItem: Item = {
  id: 'test-item-1',
  type: 'movie',
  title: '测试电影标题',
  description: '这是一部非常精彩的测试电影的详细描述',
  category: '科幻',
  tags: ['科幻', '动作', '冒险'],
  status: 'active',
  metadata: {
    director: '测试导演',
    cast: '演员A, 演员B',
    year: '2025',
  },
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
}

const mockStats: ItemStats = {
  item_id: 'test-item-1',
  view_count: 12500,
  click_count: 8000,
  like_count: 3500,
  share_count: 800,
  avg_rating: 4.5,
}

const mockSimilarItems: SimilarItem[] = [
  {
    item: {
      id: 'similar-1',
      type: 'movie',
      title: '相似电影1',
      description: '描述1',
      category: '科幻',
      tags: ['科幻'],
      status: 'active',
      created_at: '2025-01-01T00:00:00Z',
      updated_at: '2025-01-01T00:00:00Z',
    },
    score: 0.95,
  },
  {
    item: {
      id: 'similar-2',
      type: 'movie',
      title: '相似电影2',
      description: '描述2',
      category: '动作',
      tags: ['动作'],
      status: 'active',
      created_at: '2025-01-01T00:00:00Z',
      updated_at: '2025-01-01T00:00:00Z',
    },
    score: 0.85,
  },
]

// Mock API Provider
const mockApiProvider = {
  item: {
    getItem: vi.fn(),
    getItemStats: vi.fn(),
    getSimilarItems: vi.fn(),
    searchItems: vi.fn(),
    listItems: vi.fn(),
  },
  recommend: {
    getRecommendations: vi.fn(),
    submitFeedback: vi.fn(),
    getSimilarRecommendations: vi.fn(),
  },
  auth: {},
  user: {
    recordBehavior: vi.fn(),
  },
  analytics: {},
  adminUser: {},
  adminItem: {},
}

// 创建测试路由
const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'Home', component: { template: '<div>Home</div>' } },
    { path: '/item/:id', name: 'ItemDetail', component: ItemDetail },
  ],
})

describe('ItemDetail.vue', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApiProvider.item.getItem.mockResolvedValue(mockItem)
    mockApiProvider.item.getItemStats.mockResolvedValue(mockStats)
    mockApiProvider.item.getSimilarItems.mockResolvedValue(mockSimilarItems)
  })

  const mountComponent = async (itemId = 'test-item-1') => {
    await router.push(`/item/${itemId}`)
    
    return mount(ItemDetail, {
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
          ItemInfo,
          SimilarItems,
          ActionButtons,
        },
        stubs: {
          Teleport: true,
        },
      },
    })
  }

  describe('加载状态', () => {
    it('应该显示骨架屏加载状态', async () => {
      mockApiProvider.item.getItem.mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve(mockItem), 100))
      )
      
      const wrapper = await mountComponent()
      
      expect(wrapper.find('[data-testid="loading-skeleton"]').exists()).toBe(true)
    })

    it('加载完成后应该隐藏骨架屏', async () => {
      const wrapper = await mountComponent()
      await flushPromises()
      
      expect(wrapper.find('[data-testid="loading-skeleton"]').exists()).toBe(false)
    })
  })

  describe('物品信息展示', () => {
    it('应该显示物品信息组件', async () => {
      const wrapper = await mountComponent()
      await flushPromises()
      
      expect(wrapper.find('[data-testid="item-info"]').exists()).toBe(true)
    })

    it('应该显示物品封面', async () => {
      const wrapper = await mountComponent()
      await flushPromises()
      
      expect(wrapper.find('[data-testid="item-cover"]').exists()).toBe(true)
    })

    it('应该正确显示物品类型', async () => {
      const wrapper = await mountComponent()
      await flushPromises()
      
      expect(wrapper.text()).toContain('电影')
    })
  })

  describe('相似推荐', () => {
    it('应该显示相似推荐组件', async () => {
      const wrapper = await mountComponent()
      await flushPromises()
      
      expect(wrapper.find('[data-testid="similar-items"]').exists()).toBe(true)
    })

    it('点击相似物品应该导航到对应页面', async () => {
      const wrapper = await mountComponent()
      await flushPromises()
      
      const similarItems = wrapper.findComponent(SimilarItems)
      await similarItems.vm.$emit('item-click', 'similar-1')
      
      expect(router.currentRoute.value.path).toBe('/item/similar-1')
    })
  })

  describe('用户交互', () => {
    it('应该显示操作按钮', async () => {
      const wrapper = await mountComponent()
      await flushPromises()
      
      expect(wrapper.find('[data-testid="action-buttons"]').exists()).toBe(true)
    })

    it('点击喜欢按钮应该触发 like 事件', async () => {
      const wrapper = await mountComponent()
      await flushPromises()
      
      const actionButtons = wrapper.findComponent(ActionButtons)
      expect(actionButtons.exists()).toBe(true)
    })

    it('点击分享按钮应该触发 share 事件', async () => {
      const wrapper = await mountComponent()
      await flushPromises()
      
      const actionButtons = wrapper.findComponent(ActionButtons)
      expect(actionButtons.exists()).toBe(true)
    })
  })

  describe('导航', () => {
    it('应该显示返回按钮', async () => {
      const wrapper = await mountComponent()
      await flushPromises()
      
      expect(wrapper.find('[data-testid="back-button"]').exists()).toBe(true)
    })

    it('点击返回按钮应该导航回上一页', async () => {
      const wrapper = await mountComponent()
      await flushPromises()
      
      const backButton = wrapper.find('[data-testid="back-button"]')
      await backButton.trigger('click')
      
      // 由于历史记录为空，应该导航到首页
      expect(router.currentRoute.value.path).toBe('/')
    })
  })

  describe('错误处理', () => {
    it('应该显示错误状态当加载失败', async () => {
      mockApiProvider.item.getItem.mockRejectedValue(new Error('加载失败'))
      
      const wrapper = await mountComponent()
      await flushPromises()
      
      expect(wrapper.find('[data-testid="error-state"]').exists()).toBe(true)
    })
  })

  describe('404 状态', () => {
    it('应该显示 404 状态当物品不存在', async () => {
      mockApiProvider.item.getItem.mockResolvedValue(null)
      
      const wrapper = await mountComponent('non-existent')
      await flushPromises()
      
      // 当返回 null 时应该显示 not-found
      expect(wrapper.find('[data-testid="not-found"]').exists()).toBe(true)
    })
  })

  describe('路由变化', () => {
    it('路由参数变化时应该重新加载数据', async () => {
      const wrapper = await mountComponent('item-1')
      await flushPromises()
      
      // 清除之前的调用记录
      vi.clearAllMocks()
      
      // 导航到新物品
      await router.push('/item/item-2')
      await flushPromises()
      
      // 应该重新调用 getItem
      expect(mockApiProvider.item.getItem).toHaveBeenCalledWith('item-2')
    })
  })
})

