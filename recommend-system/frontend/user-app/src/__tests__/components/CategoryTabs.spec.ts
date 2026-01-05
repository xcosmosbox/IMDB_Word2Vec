/**
 * CategoryTabs 组件单元测试
 */

import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import CategoryTabs from '@/components/CategoryTabs.vue'

const defaultCategories = [
  { key: 'all', label: '全部' },
  { key: 'movie', label: '电影' },
  { key: 'product', label: '商品' },
  { key: 'article', label: '文章' },
]

describe('CategoryTabs', () => {
  describe('渲染', () => {
    it('应该正确渲染所有分类标签', () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'all',
        },
      })
      
      const tabs = wrapper.findAll('.tab-btn')
      expect(tabs).toHaveLength(4)
      
      expect(tabs[0].text()).toBe('全部')
      expect(tabs[1].text()).toBe('电影')
      expect(tabs[2].text()).toBe('商品')
      expect(tabs[3].text()).toBe('文章')
    })

    it('应该正确标记激活的标签', () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'movie',
        },
      })
      
      const tabs = wrapper.findAll('.tab-btn')
      
      expect(tabs[0].classes()).not.toContain('active')
      expect(tabs[1].classes()).toContain('active')
      expect(tabs[2].classes()).not.toContain('active')
    })

    it('应该渲染分类图标', () => {
      const categoriesWithIcons = [
        { key: 'all', label: '全部', icon: '✨' },
        { key: 'movie', label: '电影', icon: '🎬' },
      ]
      
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: categoriesWithIcons,
          active: 'all',
        },
      })
      
      const icons = wrapper.findAll('.tab-icon')
      expect(icons).toHaveLength(2)
      expect(icons[0].text()).toBe('✨')
      expect(icons[1].text()).toBe('🎬')
    })
  })

  describe('交互', () => {
    it('点击标签应该触发 update:active 事件', async () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'all',
        },
      })
      
      const tabs = wrapper.findAll('.tab-btn')
      await tabs[2].trigger('click') // 点击"商品"
      
      expect(wrapper.emitted('update:active')).toBeTruthy()
      expect(wrapper.emitted('update:active')![0]).toEqual(['product'])
    })

    it('多次点击应该触发多次事件', async () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'all',
        },
      })
      
      const tabs = wrapper.findAll('.tab-btn')
      await tabs[1].trigger('click')
      await tabs[3].trigger('click')
      
      const emitted = wrapper.emitted('update:active')
      expect(emitted).toHaveLength(2)
      expect(emitted![0]).toEqual(['movie'])
      expect(emitted![1]).toEqual(['article'])
    })
  })

  describe('键盘导航', () => {
    it('按右箭头应该选择下一个标签', async () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'all', // index 0
        },
      })
      
      const nav = wrapper.find('nav')
      await nav.trigger('keydown', { key: 'ArrowRight' })
      
      expect(wrapper.emitted('update:active')).toBeTruthy()
      expect(wrapper.emitted('update:active')![0]).toEqual(['movie'])
    })

    it('按左箭头应该选择上一个标签', async () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'movie', // index 1
        },
      })
      
      const nav = wrapper.find('nav')
      await nav.trigger('keydown', { key: 'ArrowLeft' })
      
      expect(wrapper.emitted('update:active')![0]).toEqual(['all'])
    })

    it('在第一个标签按左箭头不应该改变', async () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'all', // 第一个
        },
      })
      
      const nav = wrapper.find('nav')
      await nav.trigger('keydown', { key: 'ArrowLeft' })
      
      // 不应该触发事件，因为已经是第一个了
      expect(wrapper.emitted('update:active')).toBeFalsy()
    })

    it('在最后一个标签按右箭头不应该改变', async () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'article', // 最后一个
        },
      })
      
      const nav = wrapper.find('nav')
      await nav.trigger('keydown', { key: 'ArrowRight' })
      
      expect(wrapper.emitted('update:active')).toBeFalsy()
    })

    it('按 Home 键应该选择第一个标签', async () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'article',
        },
      })
      
      const nav = wrapper.find('nav')
      await nav.trigger('keydown', { key: 'Home' })
      
      expect(wrapper.emitted('update:active')![0]).toEqual(['all'])
    })

    it('按 End 键应该选择最后一个标签', async () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'all',
        },
      })
      
      const nav = wrapper.find('nav')
      await nav.trigger('keydown', { key: 'End' })
      
      expect(wrapper.emitted('update:active')![0]).toEqual(['article'])
    })
  })

  describe('无障碍访问', () => {
    it('应该有正确的 role 属性', () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'all',
        },
      })
      
      const nav = wrapper.find('nav')
      expect(nav.attributes('role')).toBe('tablist')
      expect(nav.attributes('aria-label')).toBe('内容分类')
    })

    it('标签应该有正确的 aria 属性', () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'movie',
        },
      })
      
      const tabs = wrapper.findAll('.tab-btn')
      
      // 非激活标签
      expect(tabs[0].attributes('role')).toBe('tab')
      expect(tabs[0].attributes('aria-selected')).toBe('false')
      expect(tabs[0].attributes('tabindex')).toBe('-1')
      
      // 激活标签
      expect(tabs[1].attributes('aria-selected')).toBe('true')
      expect(tabs[1].attributes('tabindex')).toBe('0')
    })
  })

  describe('粘性定位', () => {
    it('默认应该启用粘性定位', () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'all',
        },
      })
      
      expect(wrapper.find('.category-tabs--sticky').exists()).toBe(true)
    })

    it('sticky 为 false 时不应该有粘性类名', () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'all',
          sticky: false,
        },
      })
      
      expect(wrapper.find('.category-tabs--sticky').exists()).toBe(false)
    })
  })

  describe('快照测试', () => {
    it('基础渲染快照', () => {
      const wrapper = mount(CategoryTabs, {
        props: {
          categories: defaultCategories,
          active: 'movie',
        },
      })
      expect(wrapper.html()).toMatchSnapshot()
    })
  })
})

