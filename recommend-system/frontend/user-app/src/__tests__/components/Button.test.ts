/**
 * Button 组件单元测试
 * 
 * @author Person F
 */

import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import Button from '@/components/common/Button.vue'

describe('Button 组件', () => {
  it('应该正确渲染', () => {
    const wrapper = mount(Button, {
      slots: {
        default: '按钮文本',
      },
    })

    expect(wrapper.find('.button').exists()).toBe(true)
    expect(wrapper.text()).toContain('按钮文本')
  })

  it('应该支持不同类型', () => {
    const types = ['primary', 'secondary', 'outline', 'ghost', 'danger'] as const

    types.forEach(type => {
      const wrapper = mount(Button, {
        props: { type },
      })

      expect(wrapper.find(`.button--${type}`).exists()).toBe(true)
    })
  })

  it('应该支持不同尺寸', () => {
    const sizes = ['small', 'default', 'large'] as const

    sizes.forEach(size => {
      const wrapper = mount(Button, {
        props: { size },
      })

      expect(wrapper.find(`.button--${size}`).exists()).toBe(true)
    })
  })

  it('点击时应该触发事件', async () => {
    const wrapper = mount(Button)

    await wrapper.trigger('click')

    expect(wrapper.emitted('click')).toHaveLength(1)
  })

  it('禁用状态下不应该触发点击事件', async () => {
    const wrapper = mount(Button, {
      props: { disabled: true },
    })

    await wrapper.trigger('click')

    expect(wrapper.emitted('click')).toBeUndefined()
  })

  it('加载状态下不应该触发点击事件', async () => {
    const wrapper = mount(Button, {
      props: { loading: true },
    })

    await wrapper.trigger('click')

    expect(wrapper.emitted('click')).toBeUndefined()
  })

  it('加载状态应该显示加载动画', () => {
    const wrapper = mount(Button, {
      props: { loading: true },
    })

    expect(wrapper.find('.button__loading').exists()).toBe(true)
  })

  it('block 属性应该添加对应类', () => {
    const wrapper = mount(Button, {
      props: { block: true },
    })

    expect(wrapper.find('.button--block').exists()).toBe(true)
  })

  it('round 属性应该添加对应类', () => {
    const wrapper = mount(Button, {
      props: { round: true },
    })

    expect(wrapper.find('.button--round').exists()).toBe(true)
  })

  it('应该支持图标插槽', () => {
    const wrapper = mount(Button, {
      slots: {
        default: '按钮',
        icon: '<span class="test-icon">icon</span>',
      },
    })

    expect(wrapper.find('.button__icon').exists()).toBe(true)
    expect(wrapper.find('.test-icon').exists()).toBe(true)
  })

  it('应该支持不同的 HTML 类型', () => {
    const wrapper = mount(Button, {
      props: { htmlType: 'submit' },
    })

    expect(wrapper.find('button').attributes('type')).toBe('submit')
  })
})

