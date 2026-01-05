/**
 * RecommendList 组件单元测试
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import RecommendList from '@/components/RecommendList.vue'
import ItemCard from '@/components/ItemCard.vue'
import type { Recommendation, Item } from '@shared/types'

// Mock IntersectionObserver
const mockIntersectionObserver = vi.fn()

beforeEach(() => {
  mockIntersectionObserver.mockImplementation((callback: IntersectionObserverCallback) => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  }))
  vi.stubGlobal('IntersectionObserver', mockIntersectionObserver)
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

const createMockRecommendation = (id: string, type: Item['type'] = 'movie'): Recommendation => ({
  item_id: id,
  score: 0.9,
  reason: '推荐理由',
  item: createMockItem(id, type),
})

const createMockRecommendations = (count: number): Recommendation[] => 
  Array.from({ length: count }, (_, i) => createMockRecommendation(`item_${i + 1}`))

describe('RecommendList', () => {
  describe('渲染', () => {
    it('应该正确渲染推荐列表', () => {
      const recommendations = createMockRecommendations(3)
      const wrapper = mount(RecommendList, {
        props: { recommendations },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      expect(wrapper.find('.recommend-list').exists()).toBe(true)
      expect(wrapper.find('.list-header').exists()).toBe(true)
    })

    it('应该渲染正确数量的物品卡片', () => {
      const recommendations = createMockRecommendations(5)
      const wrapper = mount(RecommendList, {
        props: { recommendations },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      const cards = wrapper.findAllComponents({ name: 'ItemCard' })
      expect(cards).toHaveLength(5)
    })

    it('应该显示自定义标题', () => {
      const recommendations = createMockRecommendations(2)
      const wrapper = mount(RecommendList, {
        props: {
          recommendations,
          title: '热门推荐',
        },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      expect(wrapper.find('.list-title').text()).toContain('热门推荐')
    })

    it('应该显示推荐数量', () => {
      const recommendations = createMockRecommendations(10)
      const wrapper = mount(RecommendList, {
        props: { recommendations },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      expect(wrapper.find('.title-count').text()).toBe('(10)')
    })
  })

  describe('刷新按钮', () => {
    it('默认应该显示刷新按钮', () => {
      const recommendations = createMockRecommendations(2)
      const wrapper = mount(RecommendList, {
        props: { recommendations },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      expect(wrapper.find('.refresh-btn').exists()).toBe(true)
    })

    it('showRefresh 为 false 时不显示刷新按钮', () => {
      const recommendations = createMockRecommendations(2)
      const wrapper = mount(RecommendList, {
        props: {
          recommendations,
          showRefresh: false,
        },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      expect(wrapper.find('.refresh-btn').exists()).toBe(false)
    })

    it('点击刷新按钮应该触发 refresh 事件', async () => {
      const recommendations = createMockRecommendations(2)
      const wrapper = mount(RecommendList, {
        props: { recommendations },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      await wrapper.find('.refresh-btn').trigger('click')
      
      expect(wrapper.emitted('refresh')).toBeTruthy()
    })

    it('加载中时刷新按钮应该被禁用', () => {
      const recommendations = createMockRecommendations(2)
      const wrapper = mount(RecommendList, {
        props: {
          recommendations,
          loading: true,
        },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      expect(wrapper.find('.refresh-btn').attributes('disabled')).toBeDefined()
    })
  })

  describe('空状态', () => {
    it('没有推荐时应该显示空状态', () => {
      const wrapper = mount(RecommendList, {
        props: {
          recommendations: [],
          loading: false,
        },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      expect(wrapper.find('.empty-state').exists()).toBe(true)
      expect(wrapper.find('.empty-icon').exists()).toBe(true)
    })

    it('应该显示自定义空状态文字', () => {
      const wrapper = mount(RecommendList, {
        props: {
          recommendations: [],
          loading: false,
          emptyText: '暂无电影推荐',
        },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      expect(wrapper.find('.empty-text').text()).toBe('暂无电影推荐')
    })

    it('空状态点击刷新按钮应该触发 refresh 事件', async () => {
      const wrapper = mount(RecommendList, {
        props: {
          recommendations: [],
          loading: false,
        },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      await wrapper.find('.empty-refresh-btn').trigger('click')
      
      expect(wrapper.emitted('refresh')).toBeTruthy()
    })
  })

  describe('加载状态', () => {
    it('初始加载时应该显示骨架屏', () => {
      const wrapper = mount(RecommendList, {
        props: {
          recommendations: [],
          loading: true,
        },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      const skeletonCards = wrapper.findAllComponents({ name: 'ItemCard' })
      expect(skeletonCards.length).toBeGreaterThan(0)
    })

    it('应该显示正确数量的骨架屏', () => {
      const wrapper = mount(RecommendList, {
        props: {
          recommendations: [],
          loading: true,
          skeletonCount: 6,
        },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      const skeletonCards = wrapper.findAllComponents({ name: 'ItemCard' })
      expect(skeletonCards).toHaveLength(6)
    })

    it('加载更多时应该显示加载指示器', () => {
      const recommendations = createMockRecommendations(5)
      const wrapper = mount(RecommendList, {
        props: {
          recommendations,
          loading: true,
        },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      expect(wrapper.find('.loading-more').exists()).toBe(true)
      expect(wrapper.find('.loading-text').text()).toBe('加载更多...')
    })
  })

  describe('没有更多数据', () => {
    it('hasMore 为 false 时应该显示没有更多提示', () => {
      const recommendations = createMockRecommendations(5)
      const wrapper = mount(RecommendList, {
        props: {
          recommendations,
          hasMore: false,
        },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      expect(wrapper.find('.no-more-text').exists()).toBe(true)
      expect(wrapper.find('.no-more-text').text()).toContain('没有更多了')
    })
  })

  describe('事件传递', () => {
    it('ItemCard 点击应该触发 item-click 事件', async () => {
      const recommendations = createMockRecommendations(1)
      const wrapper = mount(RecommendList, {
        props: { recommendations },
      })
      
      // 找到 ItemCard 并触发点击
      const card = wrapper.findComponent(ItemCard)
      await card.trigger('click')
      
      expect(wrapper.emitted('item-click')).toBeTruthy()
      expect(wrapper.emitted('item-click')![0]).toEqual(['item_1'])
    })

    it('ItemCard 喜欢应该触发 item-like 事件', async () => {
      const recommendations = createMockRecommendations(1)
      const wrapper = mount(RecommendList, {
        props: { recommendations },
      })
      
      const card = wrapper.findComponent(ItemCard)
      card.vm.$emit('like', 'item_1')
      
      expect(wrapper.emitted('item-like')).toBeTruthy()
      expect(wrapper.emitted('item-like')![0]).toEqual(['item_1'])
    })

    it('ItemCard 分享应该触发 item-share 事件', async () => {
      const recommendations = createMockRecommendations(1)
      const wrapper = mount(RecommendList, {
        props: { recommendations },
      })
      
      const card = wrapper.findComponent(ItemCard)
      card.vm.$emit('share', 'item_1')
      
      expect(wrapper.emitted('item-share')).toBeTruthy()
      expect(wrapper.emitted('item-share')![0]).toEqual(['item_1'])
    })
  })

  describe('响应式列数', () => {
    it('应该设置 CSS 变量 --columns', () => {
      const recommendations = createMockRecommendations(5)
      const wrapper = mount(RecommendList, {
        props: { recommendations },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      
      const grid = wrapper.find('.card-grid')
      expect(grid.attributes('style')).toContain('--columns')
    })
  })

  describe('快照测试', () => {
    it('有数据时的渲染快照', () => {
      const recommendations = createMockRecommendations(2)
      const wrapper = mount(RecommendList, {
        props: { recommendations },
        global: {
          stubs: {
            ItemCard: {
              template: '<div class="item-card-stub"></div>',
            },
          },
        },
      })
      expect(wrapper.html()).toMatchSnapshot()
    })

    it('空状态渲染快照', () => {
      const wrapper = mount(RecommendList, {
        props: {
          recommendations: [],
          loading: false,
          emptyText: '暂无推荐',
        },
        global: {
          stubs: {
            ItemCard: true,
          },
        },
      })
      expect(wrapper.html()).toMatchSnapshot()
    })
  })
})

