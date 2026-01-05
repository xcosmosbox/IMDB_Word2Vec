/**
 * HeatmapChart 组件单元测试
 */
import { describe, it, expect, vi } from 'vitest'
import { mount, shallowMount } from '@vue/test-utils'
import { h, defineComponent } from 'vue'
import HeatmapChart from '../HeatmapChart.vue'

// Mock vue-echarts
vi.mock('vue-echarts', () => ({
  default: defineComponent({
    name: 'VChart',
    props: ['option', 'autoresize'],
    setup(props) {
      return () => h('div', { 
        class: 'echarts-mock',
        'data-option': JSON.stringify(props.option),
      })
    },
  }),
}))

// Mock echarts
vi.mock('echarts/core', () => ({
  use: vi.fn(),
}))

vi.mock('echarts/charts', () => ({
  HeatmapChart: {},
}))

vi.mock('echarts/components', () => ({
  TitleComponent: {},
  TooltipComponent: {},
  GridComponent: {},
  VisualMapComponent: {},
}))

vi.mock('echarts/renderers', () => ({
  CanvasRenderer: {},
}))

describe('HeatmapChart', () => {
  const mockData = [
    { x: 0, y: 0, value: 100 },
    { x: 1, y: 0, value: 200 },
    { x: 0, y: 1, value: 150 },
    { x: 1, y: 1, value: 250 },
  ]

  const xLabels = ['周一', '周二']
  const yLabels = ['上午', '下午']

  it('应正确渲染组件', () => {
    const wrapper = shallowMount(HeatmapChart, {
      props: {
        data: mockData,
        xLabels,
        yLabels,
      },
    })

    expect(wrapper.find('.analytics-heatmap-chart').exists()).toBe(true)
  })

  it('应使用默认高度', () => {
    const wrapper = mount(HeatmapChart, {
      props: {
        data: mockData,
        xLabels,
        yLabels,
      },
    })

    const style = wrapper.find('.echarts-mock').attributes('style')
    expect(style).toContain('300px')
  })

  it('应支持自定义高度', () => {
    const wrapper = mount(HeatmapChart, {
      props: {
        data: mockData,
        xLabels,
        yLabels,
        height: 400,
      },
    })

    const style = wrapper.find('.echarts-mock').attributes('style')
    expect(style).toContain('400px')
  })

  it('应支持自定义颜色', () => {
    const wrapper = mount(HeatmapChart, {
      props: {
        data: mockData,
        xLabels,
        yLabels,
        minColor: '#ffffff',
        maxColor: '#ff0000',
      },
    })

    const option = wrapper.find('.echarts-mock').attributes('data-option')
    expect(option).toBeDefined()
  })

  it('应支持标题', () => {
    const wrapper = mount(HeatmapChart, {
      props: {
        data: mockData,
        xLabels,
        yLabels,
        title: '热力图标题',
      },
    })

    const option = JSON.parse(wrapper.find('.echarts-mock').attributes('data-option') || '{}')
    expect(option.title.text).toBe('热力图标题')
  })

  it('应正确计算数据范围', () => {
    const wrapper = mount(HeatmapChart, {
      props: {
        data: mockData,
        xLabels,
        yLabels,
      },
    })

    const option = JSON.parse(wrapper.find('.echarts-mock').attributes('data-option') || '{}')
    expect(option.visualMap.min).toBe(100)
    expect(option.visualMap.max).toBe(250)
  })

  it('应支持自定义最大最小值', () => {
    const wrapper = mount(HeatmapChart, {
      props: {
        data: mockData,
        xLabels,
        yLabels,
        min: 0,
        max: 500,
      },
    })

    const option = JSON.parse(wrapper.find('.echarts-mock').attributes('data-option') || '{}')
    expect(option.visualMap.min).toBe(0)
    expect(option.visualMap.max).toBe(500)
  })

  it('应支持隐藏数据标签', () => {
    const wrapper = mount(HeatmapChart, {
      props: {
        data: mockData,
        xLabels,
        yLabels,
        showLabel: false,
      },
    })

    const option = JSON.parse(wrapper.find('.echarts-mock').attributes('data-option') || '{}')
    expect(option.series[0].label.show).toBe(false)
  })

  it('应正确转换数据格式', () => {
    const dataWithStringKeys = [
      { x: '周一', y: '上午', value: 100 },
    ]

    const wrapper = mount(HeatmapChart, {
      props: {
        data: dataWithStringKeys,
        xLabels,
        yLabels,
      },
    })

    const option = JSON.parse(wrapper.find('.echarts-mock').attributes('data-option') || '{}')
    // 字符串索引应该被转换为数字索引
    expect(option.series[0].data[0][0]).toBe(0)
    expect(option.series[0].data[0][1]).toBe(0)
    expect(option.series[0].data[0][2]).toBe(100)
  })
})

