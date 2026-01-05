/**
 * ItemInfo.vue 单元测试
 * 
 * Person B 开发
 */
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import ItemInfo from '@/components/ItemInfo.vue'
import type { Item, ItemStats } from '@shared/types'

// Mock 数据
const mockItem: Item = {
  id: 'test-1',
  type: 'movie',
  title: '测试电影标题',
  description: '这是测试电影的详细描述内容',
  category: '科幻',
  tags: ['科幻', '动作', '冒险'],
  status: 'active',
  metadata: {
    director: '测试导演',
    cast: '演员A, 演员B',
    year: '2025',
    duration: '120分钟',
  },
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-02T00:00:00Z',
}

const mockStats: ItemStats = {
  item_id: 'test-1',
  view_count: 12500,
  click_count: 8000,
  like_count: 3500,
  share_count: 800,
  avg_rating: 4.5,
}

describe('ItemInfo.vue', () => {
  const mountComponent = (props = {}) => {
    return mount(ItemInfo, {
      props: {
        item: mockItem,
        stats: null,
        ...props,
      },
    })
  }

  describe('基本信息展示', () => {
    it('应该显示物品标题', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="item-title"]').text()).toBe('测试电影标题')
    })

    it('应该显示物品类型', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="item-type"]').text()).toBe('电影')
    })

    it('应该显示物品状态', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="item-status"]').text()).toBe('已发布')
    })

    it('应该显示物品分类', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="item-category"]').text()).toContain('科幻')
    })

    it('应该显示物品标签', () => {
      const wrapper = mountComponent()
      
      const tagsContainer = wrapper.find('[data-testid="item-tags"]')
      expect(tagsContainer.text()).toContain('科幻')
      expect(tagsContainer.text()).toContain('动作')
      expect(tagsContainer.text()).toContain('冒险')
    })

    it('应该显示物品描述', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="item-description"]').text()).toContain('这是测试电影的详细描述内容')
    })
  })

  describe('类型显示', () => {
    it('电影类型应该显示为"电影"', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="item-type"]').text()).toBe('电影')
    })

    it('商品类型应该显示为"商品"', () => {
      const wrapper = mountComponent({
        item: { ...mockItem, type: 'product' },
      })
      
      expect(wrapper.find('[data-testid="item-type"]').text()).toBe('商品')
    })

    it('文章类型应该显示为"文章"', () => {
      const wrapper = mountComponent({
        item: { ...mockItem, type: 'article' },
      })
      
      expect(wrapper.find('[data-testid="item-type"]').text()).toBe('文章')
    })

    it('视频类型应该显示为"视频"', () => {
      const wrapper = mountComponent({
        item: { ...mockItem, type: 'video' },
      })
      
      expect(wrapper.find('[data-testid="item-type"]').text()).toBe('视频')
    })
  })

  describe('状态显示', () => {
    it('active 状态应该显示为"已发布"', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="item-status"]').text()).toBe('已发布')
      expect(wrapper.find('.status-active').exists()).toBe(true)
    })

    it('inactive 状态应该显示为"已下架"', () => {
      const wrapper = mountComponent({
        item: { ...mockItem, status: 'inactive' },
      })
      
      expect(wrapper.find('[data-testid="item-status"]').text()).toBe('已下架')
      expect(wrapper.find('.status-inactive').exists()).toBe(true)
    })
  })

  describe('统计数据展示', () => {
    it('应该显示统计数据栏', () => {
      const wrapper = mountComponent({ stats: mockStats })
      
      expect(wrapper.find('[data-testid="stats-bar"]').exists()).toBe(true)
    })

    it('不传 stats 时不应该显示统计数据栏', () => {
      const wrapper = mountComponent({ stats: null })
      
      expect(wrapper.find('[data-testid="stats-bar"]').exists()).toBe(false)
    })

    it('应该显示浏览数', () => {
      const wrapper = mountComponent({ stats: mockStats })
      
      expect(wrapper.text()).toContain('1.2万') // 12500 -> 1.2万
    })

    it('应该显示喜欢数', () => {
      const wrapper = mountComponent({ stats: mockStats })
      
      expect(wrapper.text()).toContain('3500')
    })

    it('应该显示分享数', () => {
      const wrapper = mountComponent({ stats: mockStats })
      
      expect(wrapper.text()).toContain('800')
    })

    it('应该显示评分', () => {
      const wrapper = mountComponent({ stats: mockStats })
      
      expect(wrapper.text()).toContain('4.5')
    })
  })

  describe('元数据展示', () => {
    it('应该显示元数据区域', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="item-metadata"]').exists()).toBe(true)
    })

    it('应该显示导演信息（电影类型）', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('导演')
      expect(wrapper.text()).toContain('测试导演')
    })

    it('应该显示主演信息（电影类型）', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('主演')
      expect(wrapper.text()).toContain('演员A, 演员B')
    })

    it('应该显示年份信息（电影类型）', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('年份')
      expect(wrapper.text()).toContain('2025')
    })

    it('商品类型应该显示品牌和价格', () => {
      const productItem: Item = {
        ...mockItem,
        type: 'product',
        metadata: {
          brand: '测试品牌',
          price: '999',
        },
      }
      
      const wrapper = mountComponent({ item: productItem })
      
      expect(wrapper.text()).toContain('品牌')
      expect(wrapper.text()).toContain('测试品牌')
      expect(wrapper.text()).toContain('价格')
      expect(wrapper.text()).toContain('¥999')
    })

    it('文章类型应该显示作者', () => {
      const articleItem: Item = {
        ...mockItem,
        type: 'article',
        metadata: {
          author: '测试作者',
        },
      }
      
      const wrapper = mountComponent({ item: articleItem })
      
      expect(wrapper.text()).toContain('作者')
      expect(wrapper.text()).toContain('测试作者')
    })
  })

  describe('时间信息', () => {
    it('应该显示创建时间', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('创建于')
      expect(wrapper.text()).toContain('2025')
    })

    it('应该显示更新时间（如果与创建时间不同）', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.text()).toContain('更新于')
    })

    it('创建时间和更新时间相同时不应该显示更新时间', () => {
      const wrapper = mountComponent({
        item: { ...mockItem, updated_at: mockItem.created_at },
      })
      
      // 应该只有一个时间戳
      const timestamps = wrapper.findAll('.timestamp')
      expect(timestamps.length).toBe(1)
    })
  })

  describe('无数据处理', () => {
    it('没有标签时不应该渲染标签区域', () => {
      const wrapper = mountComponent({
        item: { ...mockItem, tags: [] },
      })
      
      expect(wrapper.find('[data-testid="item-tags"]').exists()).toBe(false)
    })

    it('没有分类时不应该渲染分类', () => {
      const wrapper = mountComponent({
        item: { ...mockItem, category: '' },
      })
      
      expect(wrapper.find('[data-testid="item-category"]').exists()).toBe(false)
    })

    it('没有描述时不应该渲染描述区域', () => {
      const wrapper = mountComponent({
        item: { ...mockItem, description: '' },
      })
      
      expect(wrapper.find('[data-testid="item-description"]').exists()).toBe(false)
    })
  })
})

