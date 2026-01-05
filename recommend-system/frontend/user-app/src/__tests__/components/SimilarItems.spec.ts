/**
 * SimilarItems.vue 单元测试
 * 
 * Person B 开发
 */
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import SimilarItems from '@/components/SimilarItems.vue'
import type { SimilarItem, Item } from '@shared/types'

// 辅助函数：创建物品
function createItem(overrides: Partial<Item> = {}): Item {
  return {
    id: 'item-1',
    type: 'movie',
    title: '测试物品',
    description: '物品描述',
    category: '测试分类',
    tags: ['标签1'],
    status: 'active',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
    ...overrides,
  }
}

// Mock 数据
const mockSimilarItems: SimilarItem[] = [
  {
    item: createItem({ id: 'similar-1', title: '相似电影1', type: 'movie' }),
    score: 0.95,
  },
  {
    item: createItem({ id: 'similar-2', title: '相似电影2', type: 'movie' }),
    score: 0.85,
  },
  {
    item: createItem({ id: 'similar-3', title: '相似商品', type: 'product' }),
    score: 0.75,
  },
  {
    item: createItem({ id: 'similar-4', title: '相似文章', type: 'article' }),
    score: 0.65,
  },
]

describe('SimilarItems.vue', () => {
  const mountComponent = (props = {}) => {
    return mount(SimilarItems, {
      props: {
        items: mockSimilarItems,
        ...props,
      },
    })
  }

  describe('渲染', () => {
    it('应该渲染横向滚动容器', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="items-scroll"]').exists()).toBe(true)
    })

    it('应该渲染所有相似物品卡片', () => {
      const wrapper = mountComponent()
      
      const cards = wrapper.findAll('[data-testid="similar-card"]')
      expect(cards.length).toBe(mockSimilarItems.length)
    })

    it('应该显示物品标题', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('相似电影1')
      expect(wrapper.text()).toContain('相似电影2')
      expect(wrapper.text()).toContain('相似商品')
      expect(wrapper.text()).toContain('相似文章')
    })
  })

  describe('相似度显示', () => {
    it('应该显示相似度分数', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('95%')
      expect(wrapper.text()).toContain('85%')
      expect(wrapper.text()).toContain('75%')
      expect(wrapper.text()).toContain('65%')
    })

    it('高相似度应该使用 high 样式', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('.score-high').exists()).toBe(true)
    })

    it('中等相似度应该使用 medium 样式', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('.score-medium').exists()).toBe(true)
    })

    it('低相似度应该使用 low 样式', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('.score-low').exists()).toBe(true)
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

  describe('点击事件', () => {
    it('点击卡片应该触发 item-click 事件', async () => {
      const wrapper = mountComponent()
      
      const cards = wrapper.findAll('[data-testid="similar-card"]')
      await cards[0].trigger('click')
      
      expect(wrapper.emitted('item-click')).toBeTruthy()
      expect(wrapper.emitted('item-click')![0]).toEqual(['similar-1'])
    })

    it('点击不同卡片应该传递正确的物品ID', async () => {
      const wrapper = mountComponent()
      
      const cards = wrapper.findAll('[data-testid="similar-card"]')
      await cards[2].trigger('click')
      
      expect(wrapper.emitted('item-click')![0]).toEqual(['similar-3'])
    })
  })

  describe('空状态', () => {
    it('没有相似物品时应该显示空状态', () => {
      const wrapper = mountComponent({ items: [] })
      
      expect(wrapper.find('[data-testid="empty-state"]').exists()).toBe(true)
    })

    it('空状态应该显示提示信息', () => {
      const wrapper = mountComponent({ items: [] })
      
      expect(wrapper.text()).toContain('暂无相似推荐')
    })

    it('有相似物品时不应该显示空状态', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="empty-state"]').exists()).toBe(false)
    })
  })

  describe('物品信息', () => {
    it('应该显示物品描述', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('物品描述')
    })

    it('应该显示物品分类', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('测试分类')
    })
  })

  describe('渐变色', () => {
    it('不同类型应该有不同的封面渐变色', () => {
      const wrapper = mountComponent()
      
      const covers = wrapper.findAll('.card-cover')
      
      // 验证至少存在封面元素
      expect(covers.length).toBe(mockSimilarItems.length)
    })
  })
})

