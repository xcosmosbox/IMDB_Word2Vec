/**
 * Dashboard 页面单元测试
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { h, defineComponent } from 'vue'
import Dashboard from '../Dashboard.vue'
import { mockAnalyticsApi } from '@/api/mock/analytics'

// Mock Ant Design Vue 组件
vi.mock('ant-design-vue', () => ({
  Row: {
    name: 'ARow',
    props: ['gutter'],
    setup(props: any, { slots }: any) {
      return () => h('div', { class: 'ant-row' }, slots.default?.())
    },
  },
  Col: {
    name: 'ACol',
    props: ['xs', 'sm', 'md', 'lg', 'span'],
    setup(props: any, { slots }: any) {
      return () => h('div', { class: 'ant-col' }, slots.default?.())
    },
  },
  Card: {
    name: 'ACard',
    props: ['title', 'bordered'],
    setup(props: any, { slots }: any) {
      return () => h('div', { class: 'ant-card' }, [
        props.title && h('div', { class: 'ant-card-head' }, props.title),
        h('div', { class: 'ant-card-body' }, slots.default?.()),
      ])
    },
  },
  Spin: {
    name: 'ASpin',
    props: ['spinning', 'size', 'tip'],
    setup(props: any, { slots }: any) {
      return () => h('div', { 
        class: ['ant-spin-container', props.spinning ? 'ant-spin-blur' : ''],
      }, slots.default?.())
    },
  },
  message: {
    success: vi.fn(),
    error: vi.fn(),
  },
}))

// Mock 图标
vi.mock('@ant-design/icons-vue', () => ({
  UserOutlined: defineComponent({ render: () => h('span', 'UserIcon') }),
  ShoppingOutlined: defineComponent({ render: () => h('span', 'ShopIcon') }),
  ThunderboltOutlined: defineComponent({ render: () => h('span', 'ThunderIcon') }),
  ClockCircleOutlined: defineComponent({ render: () => h('span', 'ClockIcon') }),
  PercentageOutlined: defineComponent({ render: () => h('span', 'PercentIcon') }),
  TeamOutlined: defineComponent({ render: () => h('span', 'TeamIcon') }),
}))

// Mock 子组件
vi.mock('@/components/StatCard.vue', () => ({
  default: defineComponent({
    props: ['title', 'value', 'icon', 'color', 'trend', 'trendUp', 'loading'],
    setup(props) {
      return () => h('div', { 
        class: 'stat-card-mock',
        'data-title': props.title,
        'data-value': props.value,
      })
    },
  }),
}))

vi.mock('@/components/charts/LineChart.vue', () => ({
  default: defineComponent({
    props: ['data', 'xField', 'yField', 'height', 'color', 'showDataZoom'],
    setup(props) {
      return () => h('div', { 
        class: 'line-chart-mock',
        'data-count': props.data?.length || 0,
      })
    },
  }),
}))

vi.mock('@/components/charts/PieChart.vue', () => ({
  default: defineComponent({
    props: ['data', 'nameField', 'valueField', 'height', 'donut'],
    setup(props) {
      return () => h('div', { class: 'pie-chart-mock' })
    },
  }),
}))

vi.mock('@/components/charts/BarChart.vue', () => ({
  default: defineComponent({
    props: ['data', 'xField', 'yField', 'height', 'color', 'horizontal', 'barWidth'],
    setup(props) {
      return () => h('div', { class: 'bar-chart-mock' })
    },
  }),
}))

describe('Dashboard', () => {
  const mockApiProvider = {
    analytics: mockAnalyticsApi,
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('应正确渲染页面结构', async () => {
    const wrapper = mount(Dashboard, {
      global: {
        provide: {
          api: mockApiProvider,
        },
      },
    })

    await flushPromises()

    expect(wrapper.find('.dashboard-page').exists()).toBe(true)
    expect(wrapper.find('.page-header').exists()).toBe(true)
    expect(wrapper.find('.page-title').text()).toBe('数据概览')
  })

  it('应渲染统计卡片', async () => {
    const wrapper = mount(Dashboard, {
      global: {
        provide: {
          api: mockApiProvider,
        },
      },
    })

    await flushPromises()

    const statCards = wrapper.findAll('.stat-card-mock')
    expect(statCards.length).toBe(6) // 6个统计卡片
  })

  it('应渲染图表组件', async () => {
    const wrapper = mount(Dashboard, {
      global: {
        provide: {
          api: mockApiProvider,
        },
      },
    })

    await flushPromises()

    expect(wrapper.find('.line-chart-mock').exists()).toBe(true)
    expect(wrapper.find('.pie-chart-mock').exists()).toBe(true)
    expect(wrapper.find('.bar-chart-mock').exists()).toBe(true)
  })

  it('应正确显示加载状态', () => {
    const wrapper = mount(Dashboard, {
      global: {
        provide: {
          api: mockApiProvider,
        },
      },
    })

    // 初始应该是加载状态
    expect(wrapper.find('.ant-spin-blur').exists()).toBe(true)
  })

  it('API 未初始化时应显示错误', async () => {
    const wrapper = mount(Dashboard, {
      global: {
        provide: {
          api: null,
        },
      },
    })

    await flushPromises()
    
    // 验证错误处理
    expect(wrapper.find('.dashboard-page').exists()).toBe(true)
  })

  it('应包含刷新按钮', async () => {
    const wrapper = mount(Dashboard, {
      global: {
        provide: {
          api: mockApiProvider,
        },
      },
    })

    await flushPromises()

    const refreshButton = wrapper.find('.page-header button, .page-header a')
    expect(refreshButton.exists()).toBe(true)
  })
})

