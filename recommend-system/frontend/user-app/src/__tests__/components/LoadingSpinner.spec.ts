/**
 * LoadingSpinner 组件单元测试
 */

import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import LoadingSpinner from '@/components/LoadingSpinner.vue'

describe('LoadingSpinner', () => {
  describe('渲染', () => {
    it('应该正确渲染基础组件', () => {
      const wrapper = mount(LoadingSpinner)
      
      expect(wrapper.find('.loading-spinner').exists()).toBe(true)
      expect(wrapper.find('.spinner-container').exists()).toBe(true)
      expect(wrapper.find('.spinner-outer').exists()).toBe(true)
      expect(wrapper.find('.spinner-inner').exists()).toBe(true)
      expect(wrapper.find('.spinner-dot').exists()).toBe(true)
    })

    it('默认不显示加载文字', () => {
      const wrapper = mount(LoadingSpinner)
      
      expect(wrapper.find('.spinner-text').exists()).toBe(false)
    })

    it('showText 为 true 时应该显示加载文字', () => {
      const wrapper = mount(LoadingSpinner, {
        props: {
          showText: true,
        },
      })
      
      expect(wrapper.find('.spinner-text').exists()).toBe(true)
      expect(wrapper.find('.spinner-text').text()).toBe('加载中...')
    })

    it('应该显示自定义加载文字', () => {
      const customText = '正在加载推荐...'
      const wrapper = mount(LoadingSpinner, {
        props: {
          showText: true,
          text: customText,
        },
      })
      
      expect(wrapper.find('.spinner-text').text()).toBe(customText)
    })
  })

  describe('尺寸', () => {
    it('默认尺寸应该是 medium', () => {
      const wrapper = mount(LoadingSpinner)
      const outerSpinner = wrapper.find('.spinner-outer')
      
      // medium 对应 40px
      expect(outerSpinner.attributes('style')).toContain('width: 40px')
      expect(outerSpinner.attributes('style')).toContain('height: 40px')
    })

    it('small 尺寸应该正确应用', () => {
      const wrapper = mount(LoadingSpinner, {
        props: {
          size: 'small',
        },
      })
      const outerSpinner = wrapper.find('.spinner-outer')
      
      expect(outerSpinner.attributes('style')).toContain('width: 24px')
      expect(outerSpinner.attributes('style')).toContain('height: 24px')
    })

    it('large 尺寸应该正确应用', () => {
      const wrapper = mount(LoadingSpinner, {
        props: {
          size: 'large',
        },
      })
      const outerSpinner = wrapper.find('.spinner-outer')
      
      expect(outerSpinner.attributes('style')).toContain('width: 64px')
      expect(outerSpinner.attributes('style')).toContain('height: 64px')
    })
  })

  describe('颜色', () => {
    it('应该使用自定义颜色', () => {
      const customColor = '#ff5500'
      const wrapper = mount(LoadingSpinner, {
        props: {
          color: customColor,
        },
      })
      
      const outerSpinner = wrapper.find('.spinner-outer')
      expect(outerSpinner.attributes('style')).toContain(`border-top-color: ${customColor}`)
    })

    it('showText 时文字应该使用自定义颜色', () => {
      const customColor = '#ff5500'
      const wrapper = mount(LoadingSpinner, {
        props: {
          color: customColor,
          showText: true,
        },
      })
      
      const text = wrapper.find('.spinner-text')
      expect(text.attributes('style')).toContain(`color: ${customColor}`)
    })
  })

  describe('全屏模式', () => {
    it('默认不是全屏模式', () => {
      const wrapper = mount(LoadingSpinner)
      
      expect(wrapper.find('.loading-spinner--fullscreen').exists()).toBe(false)
    })

    it('fullscreen 为 true 时应该添加全屏类名', () => {
      const wrapper = mount(LoadingSpinner, {
        props: {
          fullscreen: true,
        },
      })
      
      expect(wrapper.find('.loading-spinner--fullscreen').exists()).toBe(true)
    })
  })

  describe('快照测试', () => {
    it('基础渲染快照', () => {
      const wrapper = mount(LoadingSpinner)
      expect(wrapper.html()).toMatchSnapshot()
    })

    it('带文字的渲染快照', () => {
      const wrapper = mount(LoadingSpinner, {
        props: {
          showText: true,
          text: '自定义加载文字',
          size: 'large',
        },
      })
      expect(wrapper.html()).toMatchSnapshot()
    })
  })
})

