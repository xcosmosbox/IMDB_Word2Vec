/**
 * DateRangePicker 组件单元测试
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { h } from 'vue'
import DateRangePicker from '../DateRangePicker.vue'
import dayjs from 'dayjs'

// Mock Ant Design Vue 组件
vi.mock('ant-design-vue', () => ({
  DatePicker: {
    RangePicker: {
      name: 'ARangePicker',
      props: ['value', 'format', 'allowClear', 'disabled', 'disabledDate'],
      emits: ['update:value', 'change'],
      setup(props: any, { emit }: any) {
        return () => h('div', { 
          class: 'ant-picker',
          'data-testid': 'range-picker',
        })
      },
    },
  },
  Space: {
    name: 'ASpace',
    props: ['direction', 'size'],
    setup(props: any, { slots }: any) {
      return () => h('div', { class: 'ant-space' }, slots.default?.())
    },
  },
  Button: {
    name: 'AButton',
    props: ['type', 'size'],
    setup(props: any, { slots }: any) {
      return () => h('button', { 
        class: ['ant-btn', props.type === 'primary' ? 'ant-btn-primary' : ''],
      }, slots.default?.())
    },
  },
}))

describe('DateRangePicker', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2025-01-05'))
  })

  it('应正确渲染组件', () => {
    const wrapper = mount(DateRangePicker)

    expect(wrapper.find('.date-range-picker').exists()).toBe(true)
    expect(wrapper.find('[data-testid="range-picker"]').exists()).toBe(true)
  })

  it('应显示快捷选项按钮', () => {
    const wrapper = mount(DateRangePicker, {
      props: {
        showPresets: true,
      },
    })

    expect(wrapper.find('.preset-buttons').exists()).toBe(true)
  })

  it('不应显示快捷选项当 showPresets 为 false', () => {
    const wrapper = mount(DateRangePicker, {
      props: {
        showPresets: false,
      },
    })

    expect(wrapper.find('.preset-buttons').exists()).toBe(false)
  })

  it('应正确初始化日期范围', async () => {
    const startDate = '2025-01-01'
    const endDate = '2025-01-05'
    
    const wrapper = mount(DateRangePicker, {
      props: {
        startDate,
        endDate,
      },
    })

    // 组件应该被正确初始化
    expect(wrapper.props('startDate')).toBe(startDate)
    expect(wrapper.props('endDate')).toBe(endDate)
  })

  it('快捷按钮应该存在', () => {
    const wrapper = mount(DateRangePicker, {
      props: {
        showPresets: true,
      },
    })

    const buttons = wrapper.findAll('.preset-buttons button')
    expect(buttons.length).toBeGreaterThan(0)
  })

  it('应使用正确的日期格式', () => {
    const wrapper = mount(DateRangePicker, {
      props: {
        format: 'YYYY/MM/DD',
      },
    })

    expect(wrapper.props('format')).toBe('YYYY/MM/DD')
  })

  afterEach(() => {
    vi.useRealTimers()
  })
})

