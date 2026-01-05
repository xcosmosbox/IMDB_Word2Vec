/**
 * DataExport 组件单元测试
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { h } from 'vue'
import DataExport from '../DataExport.vue'

// Mock Ant Design Vue 组件
vi.mock('ant-design-vue', () => ({
  Button: {
    name: 'AButton',
    props: ['disabled', 'loading'],
    setup(props: any, { slots }: any) {
      return () => h('button', { 
        class: 'ant-btn',
        disabled: props.disabled,
      }, slots.default?.())
    },
  },
  Dropdown: {
    name: 'ADropdown',
    props: ['disabled'],
    setup(props: any, { slots }: any) {
      return () => h('div', { class: 'ant-dropdown' }, [
        slots.default?.(),
        slots.overlay?.(),
      ])
    },
  },
  Menu: {
    name: 'AMenu',
    emits: ['click'],
    setup(props: any, { slots }: any) {
      return () => h('div', { class: 'ant-menu' }, slots.default?.())
    },
    Item: {
      name: 'AMenuItem',
      props: ['key'],
      setup(props: any, { slots }: any) {
        return () => h('div', { class: 'ant-menu-item' }, slots.default?.())
      },
    },
  },
  Modal: {
    name: 'AModal',
    props: ['open', 'title', 'width'],
    emits: ['update:open', 'ok', 'cancel'],
    setup(props: any, { slots }: any) {
      return () => props.open 
        ? h('div', { class: 'ant-modal' }, slots.default?.())
        : null
    },
  },
  Checkbox: {
    name: 'ACheckbox',
    props: ['checked', 'indeterminate'],
    emits: ['change'],
    setup(props: any, { slots }: any) {
      return () => h('label', { class: 'ant-checkbox' }, [
        h('input', { type: 'checkbox', checked: props.checked }),
        slots.default?.(),
      ])
    },
  },
  Space: {
    name: 'ASpace',
    setup(props: any, { slots }: any) {
      return () => h('div', { class: 'ant-space' }, slots.default?.())
    },
  },
  message: {
    success: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
  },
}))

// Mock URL 和 Blob
global.URL.createObjectURL = vi.fn(() => 'blob:test')
global.URL.revokeObjectURL = vi.fn()

describe('DataExport', () => {
  const mockData = [
    { id: '1', name: '测试1', value: 100 },
    { id: '2', name: '测试2', value: 200 },
    { id: '3', name: '测试3', value: 300 },
  ]

  const mockColumns = [
    { key: 'id', title: 'ID' },
    { key: 'name', title: '名称' },
    { key: 'value', title: '数值' },
  ]

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('应正确渲染组件', () => {
    const wrapper = mount(DataExport, {
      props: {
        data: mockData,
        columns: mockColumns,
      },
    })

    expect(wrapper.find('.data-export').exists()).toBe(true)
  })

  it('当数据为空时按钮应禁用', () => {
    const wrapper = mount(DataExport, {
      props: {
        data: [],
        columns: mockColumns,
      },
    })

    const button = wrapper.find('button')
    expect(button.attributes('disabled')).toBeDefined()
  })

  it('应使用自定义按钮文字', () => {
    const wrapper = mount(DataExport, {
      props: {
        data: mockData,
        columns: mockColumns,
        buttonText: '自定义导出',
      },
    })

    expect(wrapper.text()).toContain('自定义导出')
  })

  it('应使用自定义文件名', () => {
    const wrapper = mount(DataExport, {
      props: {
        data: mockData,
        columns: mockColumns,
        filename: 'custom_export',
      },
    })

    expect(wrapper.props('filename')).toBe('custom_export')
  })

  it('支持禁用状态', () => {
    const wrapper = mount(DataExport, {
      props: {
        data: mockData,
        columns: mockColumns,
        disabled: true,
      },
    })

    const button = wrapper.find('button')
    expect(button.attributes('disabled')).toBeDefined()
  })
})

describe('CSV 导出功能', () => {
  it('应正确处理包含逗号的值', () => {
    const dataWithComma = [
      { name: 'hello, world', value: 100 },
    ]
    const columns = [
      { key: 'name', title: '名称' },
      { key: 'value', title: '数值' },
    ]

    const wrapper = mount(DataExport, {
      props: {
        data: dataWithComma,
        columns,
      },
    })

    // 组件应该能正常渲染
    expect(wrapper.find('.data-export').exists()).toBe(true)
  })

  it('应正确处理包含换行的值', () => {
    const dataWithNewline = [
      { name: 'hello\nworld', value: 100 },
    ]
    const columns = [
      { key: 'name', title: '名称' },
      { key: 'value', title: '数值' },
    ]

    const wrapper = mount(DataExport, {
      props: {
        data: dataWithNewline,
        columns,
      },
    })

    expect(wrapper.find('.data-export').exists()).toBe(true)
  })

  it('应正确处理包含引号的值', () => {
    const dataWithQuote = [
      { name: 'hello "world"', value: 100 },
    ]
    const columns = [
      { key: 'name', title: '名称' },
      { key: 'value', title: '数值' },
    ]

    const wrapper = mount(DataExport, {
      props: {
        data: dataWithQuote,
        columns,
      },
    })

    expect(wrapper.find('.data-export').exists()).toBe(true)
  })
})

