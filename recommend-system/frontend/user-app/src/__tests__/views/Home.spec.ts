/**
 * Home.vue 视图单元测试
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { createRouter, createWebHistory } from 'vue-router'
import Home from '@/views/Home.vue'
import type { IApiProvider } from '@shared/api/interfaces'
import type { RecommendResponse, Recommendation, Item } from '@shared/types'

// Mock vue-router
const mockPush = vi.fn()

vi.mock('vue-router', async () => {
  const actual = await vi.importActual('vue-router')
  return {
    ...actual,
    useRouter: () => ({
      push: mockPush,
    }),
  }
})

// Mock IntersectionObserver
const mockIntersectionObserver = vi.fn()

beforeEach(() => {
  mockIntersectionObserver.mockImplementation((callback: IntersectionObserverCallback) => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  }))
  vi.stubGlobal('IntersectionObserver', mockIntersectionObserver)
  mockPush.mockClear()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

const createMockItem = (id: string, type: Item['type'] = 'movie'): Item => ({
  id,
  type,
  title: `测试物品 ${id}`,
  description: '测试描述',
  category: '测试分类',
  tags: ['标签1'],
  status: 'active',
  created_at: '2024-01-01',
  updated_at: '2024-01-01',
})

const createMockRecommendations = (count: number): Recommendation[] => 
  Array.from({ length: count }, (_, i) => ({
    item_id: `item_${i + 1}`,
    score: 0.9 - i * 0.1,
    reason: `推荐理由 ${i + 1}`,
    item: createMockItem(`item_${i + 1}`, ['movie', 'product', 'article', 'video'][i % 4] as Item['type']),
  }))

const createMockApiProvider = (): IApiProvider => {
  const mockRecommendations = createMockRecommendations(10)
  
  return {
    auth: {} as any,
    user: {
      getUser: vi.fn(),
      updateUser: vi.fn(),
      getProfile: vi.fn(),
      getBehaviors: vi.fn(),
      recordBehavior: vi.fn().mockResolvedValue(undefined),
    },
    item: {} as any,
    recommend: {
      getRecommendations: vi.fn().mockResolvedValue({
        recommendations: mockRecommendations,
        request_id: 'req_123',
        strategy: 'generative_model',
      } as RecommendResponse),
      submitFeedback: vi.fn().mockResolvedValue(undefined),
      getSimilarRecommendations: vi.fn(),
    },
    analytics: {} as any,
    adminUser: {} as any,
    adminItem: {} as any,
  }
}

const mountHome = (apiProvider?: IApiProvider) => {
  setActivePinia(createPinia())
  
  return mount(Home, {
    global: {
      provide: {
        api: apiProvider || createMockApiProvider(),
      },
      stubs: {
        CategoryTabs: {
          template: `
            <nav class="category-tabs-stub">
              <button 
                v-for="cat in categories" 
                :key="cat.key"
                :class="{ active: active === cat.key }"
                @click="$emit('update:active', cat.key)"
              >
                {{ cat.label }}
              </button>
            </nav>
          `,
          props: ['categories', 'active'],
          emits: ['update:active'],
        },
        RecommendList: {
          template: '<div class="recommend-list-stub"></div>',
          props: ['recommendations', 'loading', 'hasMore', 'title', 'showRefresh', 'emptyText'],
          emits: ['item-click', 'item-like', 'item-share', 'refresh', 'load-more'],
        },
        LoadingSpinner: {
          template: '<div class="loading-spinner-stub"></div>',
          props: ['size', 'showText', 'text'],
        },
      },
    },
  })
}

describe('Home.vue', () => {
  describe('渲染', () => {
    it('应该正确渲染首页', async () => {
      const wrapper = mountHome()
      await flushPromises()
      
      expect(wrapper.find('.home-page').exists()).toBe(true)
      expect(wrapper.find('.hero-section').exists()).toBe(true)
    })

    it('应该渲染英雄区域标题', async () => {
      const wrapper = mountHome()
      await flushPromises()
      
      expect(wrapper.find('.hero-title-text').text()).toContain('发现你的下一个最爱')
    })

    it('应该渲染副标题', async () => {
      const wrapper = mountHome()
      await flushPromises()
      
      expect(wrapper.find('.hero-subtitle').text()).toContain('AI 生成式推荐')
    })

    it('应该渲染分类标签', async () => {
      const wrapper = mountHome()
      await flushPromises()
      
      expect(wrapper.find('.category-tabs-stub').exists()).toBe(true)
    })

    it('应该渲染推荐列表', async () => {
      const wrapper = mountHome()
      await flushPromises()
      
      expect(wrapper.find('.recommend-list-stub').exists()).toBe(true)
    })
  })

  describe('加载状态', () => {
    it('初始应该显示加载状态', () => {
      const wrapper = mountHome()
      
      // 初始加载时应该显示 LoadingSpinner
      expect(wrapper.find('.initial-loading').exists()).toBe(true)
      expect(wrapper.find('.loading-spinner-stub').exists()).toBe(true)
    })

    it('加载完成后应该隐藏加载状态', async () => {
      const wrapper = mountHome()
      await flushPromises()
      
      expect(wrapper.find('.initial-loading').exists()).toBe(false)
    })
  })

  describe('推荐策略', () => {
    it('应该显示推荐策略标签', async () => {
      const wrapper = mountHome()
      await flushPromises()
      
      expect(wrapper.find('.strategy-badge').exists()).toBe(true)
      expect(wrapper.find('.strategy-text').text()).toBe('generative_model')
    })
  })

  describe('分类过滤', () => {
    it('切换分类应该更新 activeCategory', async () => {
      const wrapper = mountHome()
      await flushPromises()
      
      // 点击"电影"分类
      const movieTab = wrapper.findAll('.category-tabs-stub button')[1]
      await movieTab.trigger('click')
      
      // 检查 activeCategory 是否更新
      expect(wrapper.vm.activeCategory).toBe('movie')
    })
  })

  describe('错误处理', () => {
    it('获取推荐失败时应该显示错误状态', async () => {
      const errorApiProvider = createMockApiProvider()
      vi.mocked(errorApiProvider.recommend.getRecommendations).mockRejectedValue(
        new Error('网络错误')
      )
      
      const wrapper = mountHome(errorApiProvider)
      await flushPromises()
      
      expect(wrapper.find('.error-state').exists()).toBe(true)
      expect(wrapper.find('.error-title').text()).toBe('加载失败')
    })

    it('点击重新加载应该重新获取推荐', async () => {
      const errorApiProvider = createMockApiProvider()
      const getRecommendationsMock = vi.mocked(errorApiProvider.recommend.getRecommendations)
      
      // 第一次失败
      getRecommendationsMock.mockRejectedValueOnce(new Error('网络错误'))
      
      const wrapper = mountHome(errorApiProvider)
      await flushPromises()
      
      // 第二次成功
      getRecommendationsMock.mockResolvedValueOnce({
        recommendations: createMockRecommendations(5),
        request_id: 'req_456',
        strategy: 'retry_success',
      })
      
      await wrapper.find('.retry-btn').trigger('click')
      await flushPromises()
      
      expect(getRecommendationsMock).toHaveBeenCalledTimes(2)
    })
  })

  describe('事件处理', () => {
    it('物品点击应该记录行为并导航', async () => {
      const apiProvider = createMockApiProvider()
      const wrapper = mountHome(apiProvider)
      await flushPromises()
      
      // 触发 item-click 事件
      const recommendList = wrapper.findComponent({ name: 'RecommendList' })
      recommendList.vm.$emit('item-click', 'item_1')
      await flushPromises()
      
      // 验证行为记录
      expect(apiProvider.user.recordBehavior).toHaveBeenCalledWith(
        expect.objectContaining({
          item_id: 'item_1',
          action: 'click',
        })
      )
      
      // 验证导航
      expect(mockPush).toHaveBeenCalledWith('/item/item_1')
    })

    it('物品喜欢应该记录行为', async () => {
      const apiProvider = createMockApiProvider()
      const wrapper = mountHome(apiProvider)
      await flushPromises()
      
      const recommendList = wrapper.findComponent({ name: 'RecommendList' })
      recommendList.vm.$emit('item-like', 'item_2')
      await flushPromises()
      
      expect(apiProvider.user.recordBehavior).toHaveBeenCalledWith(
        expect.objectContaining({
          item_id: 'item_2',
          action: 'like',
        })
      )
    })

    it('物品分享应该记录行为', async () => {
      const apiProvider = createMockApiProvider()
      const wrapper = mountHome(apiProvider)
      await flushPromises()
      
      const recommendList = wrapper.findComponent({ name: 'RecommendList' })
      recommendList.vm.$emit('item-share', 'item_3')
      await flushPromises()
      
      expect(apiProvider.user.recordBehavior).toHaveBeenCalledWith(
        expect.objectContaining({
          item_id: 'item_3',
          action: 'share',
        })
      )
    })

    it('刷新应该获取新推荐', async () => {
      const apiProvider = createMockApiProvider()
      const wrapper = mountHome(apiProvider)
      await flushPromises()
      
      const recommendList = wrapper.findComponent({ name: 'RecommendList' })
      recommendList.vm.$emit('refresh')
      await flushPromises()
      
      // getRecommendations 应该被调用两次（初始加载 + 刷新）
      expect(apiProvider.recommend.getRecommendations).toHaveBeenCalledTimes(2)
    })
  })

  describe('页脚', () => {
    it('应该渲染页脚', async () => {
      const wrapper = mountHome()
      await flushPromises()
      
      expect(wrapper.find('.page-footer').exists()).toBe(true)
    })

    it('应该显示推荐数量', async () => {
      const wrapper = mountHome()
      await flushPromises()
      
      expect(wrapper.find('.footer-stats').text()).toContain('已为你推荐')
    })
  })

  describe('快照测试', () => {
    it('首页渲染快照', async () => {
      const wrapper = mountHome()
      await flushPromises()
      
      expect(wrapper.html()).toMatchSnapshot()
    })
  })
})

