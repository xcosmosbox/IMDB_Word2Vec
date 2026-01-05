/**
 * StatCard 组件单元测试
 */
import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { h } from 'vue'
import StatCard from '../StatCard.vue'
import { UserOutlined } from '@ant-design/icons-vue'

// Mock Ant Design Vue 组件
vi.mock('ant-design-vue', () => ({
  Card: {
    name: 'ACard',
    props: ['bordered', 'bodyStyle', 'loading'],
    setup(props: any, { slots }: any) {
      return () => h('div', { class: 'ant-card' }, slots.default?.())
    },
  },
}))

describe('StatCard', () => {
  it('应正确渲染标题和数值', () => {
    const wrapper = mount(StatCard, {
      props: {
        title: '总用户数',
        value: 12345,
      },
    })

    expect(wrapper.find('.stat-title').text()).toBe('总用户数')
    expect(wrapper.find('.stat-value').text()).toContain('12,345')
  })

  it('应正确格式化大数字', () => {
    const wrapper = mount(StatCard, {
      props: {
        title: '测试',
        value: 1234567,
      },
    })

    expect(wrapper.find('.stat-value').text()).toContain('1,234,567')
  })

  it('应支持字符串类型的值', () => {
    const wrapper = mount(StatCard, {
      props: {
        title: '响应时间',
        value: '23.5ms',
      },
    })

    expect(wrapper.find('.stat-value').text()).toContain('23.5ms')
  })

  it('应正确显示趋势信息', () => {
    const wrapper = mount(StatCard, {
      props: {
        title: '测试',
        value: 100,
        trend: '+12.5%',
        trendUp: true,
      },
    })

    expect(wrapper.find('.stat-trend').exists()).toBe(true)
    expect(wrapper.find('.trend-value').text()).toBe('+12.5%')
  })

  it('应正确显示后缀', () => {
    const wrapper = mount(StatCard, {
      props: {
        title: '测试',
        value: 99,
        suffix: '%',
      },
    })

    expect(wrapper.find('.stat-suffix').text()).toBe('%')
  })

  it('应正确显示前缀', () => {
    const wrapper = mount(StatCard, {
      props: {
        title: '测试',
        value: 100,
        prefix: '¥',
      },
    })

    expect(wrapper.find('.stat-prefix').text()).toBe('¥')
  })

  it('应正确显示图标', () => {
    const wrapper = mount(StatCard, {
      props: {
        title: '测试',
        value: 100,
        icon: UserOutlined,
        color: '#1890ff',
      },
    })

    expect(wrapper.find('.stat-icon').exists()).toBe(true)
    expect(wrapper.find('.stat-icon').attributes('style')).toContain('#1890ff')
  })

  it('应支持自定义格式化函数', () => {
    const wrapper = mount(StatCard, {
      props: {
        title: '测试',
        value: 0.856,
        formatter: (v: number) => `${(v * 100).toFixed(1)}%`,
      },
    })

    expect(wrapper.find('.stat-value').text()).toContain('85.6%')
  })

  it('点击时应触发 click 事件', async () => {
    const wrapper = mount(StatCard, {
      props: {
        title: '测试',
        value: 100,
      },
    })

    await wrapper.find('.stat-card').trigger('click')
    expect(wrapper.emitted('click')).toBeTruthy()
  })

  it('趋势下降时应显示下降图标和红色', () => {
    const wrapper = mount(StatCard, {
      props: {
        title: '测试',
        value: 100,
        trend: '-5%',
        trendUp: false,
      },
    })

    const trendElement = wrapper.find('.stat-trend')
    expect(trendElement.attributes('style')).toContain('rgb(255, 77, 79)')
  })
})

