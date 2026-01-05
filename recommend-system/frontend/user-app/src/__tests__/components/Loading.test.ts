/**
 * Loading 组件单元测试
 * 
 * @author Person F
 */

import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import Loading from '@/components/common/Loading.vue'

describe('Loading 组件', () => {
  it('应该正确渲染', () => {
    const wrapper = mount(Loading)

    expect(wrapper.find('.loading').exists()).toBe(true)
    expect(wrapper.find('.loading__spinner').exists()).toBe(true)
  })

  it('应该显示提示文本', () => {
    const wrapper = mount(Loading, {
      props: {
        tip: '正在加载...',
      },
    })

    expect(wrapper.find('.loading__tip').text()).toBe('正在加载...')
  })

  it('不传 tip 时不应该显示提示', () => {
    const wrapper = mount(Loading)

    expect(wrapper.find('.loading__tip').exists()).toBe(false)
  })

  it('应该支持不同尺寸', () => {
    const sizes = ['small', 'default', 'large'] as const

    sizes.forEach(size => {
      const wrapper = mount(Loading, {
        props: { size },
      })

      expect(wrapper.find('.loading__spinner').exists()).toBe(true)
    })
  })

  it('fullscreen 模式应该添加对应类', () => {
    const wrapper = mount(Loading, {
      props: {
        fullscreen: true,
      },
    })

    expect(wrapper.find('.loading--fullscreen').exists()).toBe(true)
  })

  it('应该支持自定义颜色', () => {
    const wrapper = mount(Loading, {
      props: {
        color: '#ff0000',
      },
    })

    const path = wrapper.find('.loading__path')
    expect(path.attributes('style')).toContain('stroke: rgb(255, 0, 0)')
  })

  it('应该有正确的无障碍属性', () => {
    const wrapper = mount(Loading)

    expect(wrapper.find('.loading').attributes('role')).toBe('status')
    expect(wrapper.find('.loading').attributes('aria-busy')).toBe('true')
  })
})

