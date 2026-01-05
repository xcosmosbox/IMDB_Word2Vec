/**
 * ItemCard 组件单元测试
 */

import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import ItemCard from '@/components/ItemCard.vue'
import type { Item } from '@shared/types'

const createMockItem = (overrides: Partial<Item> = {}): Item => ({
  id: 'item_1',
  type: 'movie',
  title: '肖申克的救赎',
  description: '一个关于希望和自由的故事，讲述了银行家安迪被错误地判处终身监禁后的传奇经历。',
  category: '剧情',
  tags: ['经典', '励志', '监狱'],
  status: 'active',
  created_at: '2024-01-01',
  updated_at: '2024-01-01',
  ...overrides,
})

describe('ItemCard', () => {
  describe('渲染', () => {
    it('应该正确渲染物品卡片', () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      expect(wrapper.find('.item-card').exists()).toBe(true)
      expect(wrapper.find('.card-title').text()).toBe(item.title)
    })

    it('应该渲染物品描述', () => {
      const item = createMockItem({
        description: '这是一段测试描述',
      })
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      expect(wrapper.find('.card-description').text()).toBe('这是一段测试描述')
    })

    it('长描述应该被截断', () => {
      const longDescription = 'A'.repeat(150)
      const item = createMockItem({
        description: longDescription,
      })
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      const description = wrapper.find('.card-description').text()
      expect(description.length).toBeLessThan(longDescription.length)
      expect(description.endsWith('...')).toBe(true)
    })

    it('应该渲染物品分类', () => {
      const item = createMockItem({
        category: '科幻',
      })
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      expect(wrapper.find('.card-category').text()).toBe('科幻')
    })

    it('应该渲染最多3个标签', () => {
      const item = createMockItem({
        tags: ['标签1', '标签2', '标签3', '标签4', '标签5'],
      })
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      const tags = wrapper.findAll('.tag')
      expect(tags).toHaveLength(3)
    })

    it('没有标签时不应该渲染标签区域', () => {
      const item = createMockItem({
        tags: [],
      })
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      expect(wrapper.find('.card-tags').exists()).toBe(false)
    })
  })

  describe('类型显示', () => {
    it.each([
      ['movie', '电影', '#e50914'],
      ['product', '商品', '#ff9900'],
      ['article', '文章', '#1da1f2'],
      ['video', '视频', '#ff0050'],
    ])('类型为 %s 时应该显示正确的标签', (type, label, color) => {
      const item = createMockItem({ type: type as Item['type'] })
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      const typeTag = wrapper.find('.type-tag')
      expect(typeTag.text()).toBe(label)
      expect(typeTag.attributes('style')).toContain(color)
    })

    it.each([
      ['movie', '🎬'],
      ['product', '🛒'],
      ['article', '📄'],
      ['video', '🎥'],
    ])('类型为 %s 时应该显示正确的图标', (type, icon) => {
      const item = createMockItem({ type: type as Item['type'] })
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      const typeIcon = wrapper.find('.type-icon')
      expect(typeIcon.text()).toBe(icon)
    })
  })

  describe('推荐分数', () => {
    it('有分数时应该显示匹配度', () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: {
          item,
          score: 0.95,
        },
      })
      
      const scoreEl = wrapper.find('.match-score')
      expect(scoreEl.exists()).toBe(true)
      expect(scoreEl.text()).toContain('95%')
    })

    it('没有分数时不应该显示匹配度', () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      expect(wrapper.find('.match-score').exists()).toBe(false)
    })

    it('分数应该正确四舍五入', () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: {
          item,
          score: 0.876,
        },
      })
      
      expect(wrapper.find('.match-score').text()).toContain('88%')
    })
  })

  describe('推荐理由', () => {
    it('有理由时应该显示推荐理由', () => {
      const item = createMockItem()
      const reason = '根据你的浏览历史推荐'
      const wrapper = mount(ItemCard, {
        props: {
          item,
          reason,
        },
      })
      
      const reasonEl = wrapper.find('.card-reason')
      expect(reasonEl.exists()).toBe(true)
      expect(reasonEl.text()).toContain(reason)
    })

    it('没有理由时不应该显示理由区域', () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      expect(wrapper.find('.card-reason').exists()).toBe(false)
    })
  })

  describe('交互', () => {
    it('点击卡片应该触发 click 事件', async () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      await wrapper.find('.item-card').trigger('click')
      
      expect(wrapper.emitted('click')).toBeTruthy()
      expect(wrapper.emitted('click')).toHaveLength(1)
    })

    it('按 Enter 键应该触发 click 事件', async () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      await wrapper.find('.item-card').trigger('keydown.enter')
      
      expect(wrapper.emitted('click')).toBeTruthy()
    })

    it('按空格键应该触发 click 事件', async () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      await wrapper.find('.item-card').trigger('keydown.space')
      
      expect(wrapper.emitted('click')).toBeTruthy()
    })

    it('点击喜欢按钮应该触发 like 事件', async () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      await wrapper.find('.action-btn--like').trigger('click')
      
      expect(wrapper.emitted('like')).toBeTruthy()
      expect(wrapper.emitted('like')![0]).toEqual([item.id])
    })

    it('点击分享按钮应该触发 share 事件', async () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      await wrapper.find('.action-btn--share').trigger('click')
      
      expect(wrapper.emitted('share')).toBeTruthy()
      expect(wrapper.emitted('share')![0]).toEqual([item.id])
    })

    it('点击操作按钮不应该触发卡片 click 事件', async () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      await wrapper.find('.action-btn--like').trigger('click')
      
      expect(wrapper.emitted('like')).toBeTruthy()
      expect(wrapper.emitted('click')).toBeFalsy()
    })
  })

  describe('骨架屏', () => {
    it('loading 为 true 时应该显示骨架屏', () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: {
          item,
          loading: true,
        },
      })
      
      expect(wrapper.find('.item-card--skeleton').exists()).toBe(true)
      expect(wrapper.find('.skeleton-cover').exists()).toBe(true)
      expect(wrapper.find('.skeleton-shimmer').exists()).toBe(true)
    })

    it('骨架屏状态不应该渲染实际内容', () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: {
          item,
          loading: true,
        },
      })
      
      expect(wrapper.find('.card-title').exists()).toBe(false)
      expect(wrapper.find('.type-tag').exists()).toBe(false)
    })
  })

  describe('无障碍访问', () => {
    it('应该有正确的 role 和 tabindex', () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      const card = wrapper.find('.item-card')
      expect(card.attributes('role')).toBe('button')
      expect(card.attributes('tabindex')).toBe('0')
    })

    it('操作按钮应该有 title 属性', () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: { item },
      })
      
      expect(wrapper.find('.action-btn--like').attributes('title')).toBe('喜欢')
      expect(wrapper.find('.action-btn--share').attributes('title')).toBe('分享')
    })
  })

  describe('快照测试', () => {
    it('电影类型卡片快照', () => {
      const item = createMockItem({
        type: 'movie',
        title: '测试电影',
      })
      const wrapper = mount(ItemCard, {
        props: {
          item,
          score: 0.92,
          reason: '推荐理由',
        },
      })
      expect(wrapper.html()).toMatchSnapshot()
    })

    it('骨架屏快照', () => {
      const item = createMockItem()
      const wrapper = mount(ItemCard, {
        props: {
          item,
          loading: true,
        },
      })
      expect(wrapper.html()).toMatchSnapshot()
    })
  })
})

