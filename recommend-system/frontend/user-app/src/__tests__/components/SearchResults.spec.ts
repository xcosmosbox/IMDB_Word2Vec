/**
 * SearchResults.vue 单元测试
 * 
 * Person B 开发
 */
import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import SearchResults from '@/components/SearchResults.vue'
import type { Item } from '@shared/types'

// Mock 数据
const mockItems: Item[] = [
  {
    id: '1',
    type: 'movie',
    title: '科幻电影测试',
    description: '这是一部科幻电影的描述',
    category: '科幻',
    tags: ['科幻', '动作'],
    status: 'active',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
  },
  {
    id: '2',
    type: 'product',
    title: '测试商品名称',
    description: '这是一个测试商品的描述',
    category: '电子产品',
    tags: ['数码', '手机'],
    status: 'active',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
  },
  {
    id: '3',
    type: 'article',
    title: '测试文章标题',
    description: '这是一篇测试文章的内容摘要',
    category: '技术',
    tags: ['编程', 'Vue'],
    status: 'active',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
  },
]

describe('SearchResults.vue', () => {
  const mountComponent = (props = {}) => {
    return mount(SearchResults, {
      props: {
        items: mockItems,
        query: '',
        ...props,
      },
    })
  }

  describe('渲染', () => {
    it('应该正确渲染结果网格', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="results-grid"]').exists()).toBe(true)
    })

    it('应该渲染所有结果卡片', () => {
      const wrapper = mountComponent()
      
      const cards = wrapper.findAll('[data-testid="result-card"]')
      expect(cards.length).toBe(mockItems.length)
    })

    it('应该显示物品标题', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('科幻电影测试')
      expect(wrapper.text()).toContain('测试商品名称')
      expect(wrapper.text()).toContain('测试文章标题')
    })

    it('应该显示物品描述', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('这是一部科幻电影的描述')
    })

    it('应该显示物品分类', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('科幻')
      expect(wrapper.text()).toContain('电子产品')
    })

    it('应该显示物品标签', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('动作')
      expect(wrapper.text()).toContain('数码')
    })
  })

  describe('空结果状态', () => {
    it('应该显示空结果状态', () => {
      const wrapper = mountComponent({ items: [] })
      
      expect(wrapper.find('[data-testid="empty-results"]').exists()).toBe(true)
    })

    it('空结果状态应该显示提示信息', () => {
      const wrapper = mountComponent({ items: [], query: '未知关键词' })
      
      expect(wrapper.text()).toContain('未找到相关结果')
      expect(wrapper.text()).toContain('未知关键词')
    })

    it('空结果状态应该显示建议', () => {
      const wrapper = mountComponent({ items: [] })
      
      expect(wrapper.text()).toContain('建议')
      expect(wrapper.text()).toContain('检查输入是否有误')
    })
  })

  describe('关键词高亮', () => {
    it('应该高亮标题中的关键词', () => {
      const wrapper = mountComponent({ query: '科幻' })
      
      const highlight = wrapper.find('.highlight')
      expect(highlight.exists()).toBe(true)
    })

    it('应该高亮描述中的关键词', () => {
      const wrapper = mountComponent({ query: '电影' })
      
      const highlights = wrapper.findAll('.highlight')
      expect(highlights.length).toBeGreaterThan(0)
    })
  })

  describe('点击事件', () => {
    it('点击卡片应该触发 item-click 事件', async () => {
      const wrapper = mountComponent()
      
      const cards = wrapper.findAll('[data-testid="result-card"]')
      await cards[0].trigger('click')
      
      expect(wrapper.emitted('item-click')).toBeTruthy()
      expect(wrapper.emitted('item-click')![0]).toEqual(['1'])
    })

    it('点击不同卡片应该传递对应的物品ID', async () => {
      const wrapper = mountComponent()
      
      const cards = wrapper.findAll('[data-testid="result-card"]')
      await cards[1].trigger('click')
      
      expect(wrapper.emitted('item-click')![0]).toEqual(['2'])
    })
  })

  describe('类型显示', () => {
    it('应该正确显示电影类型', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('电影')
    })

    it('应该正确显示商品类型', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('商品')
    })

    it('应该正确显示文章类型', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('文章')
    })
  })

  describe('描述截断', () => {
    it('应该截断过长的描述', () => {
      const longDescItem: Item = {
        id: '99',
        type: 'movie',
        title: '测试',
        description: '这是一段非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常长的描述文本，应该被截断显示',
        category: '测试',
        tags: [],
        status: 'active',
        created_at: '2025-01-01T00:00:00Z',
        updated_at: '2025-01-01T00:00:00Z',
      }
      
      const wrapper = mountComponent({ items: [longDescItem] })
      
      // 描述应该被截断（CSS实现，这里只验证渲染）
      expect(wrapper.find('.card-description').exists()).toBe(true)
    })
  })
})

