/**
 * ItemList 视图单元测试
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { createRouter, createWebHistory } from 'vue-router'
import { h } from 'vue'
import ItemList from '@/views/items/ItemList.vue'
import type { IApiProvider } from '@shared/api/interfaces'

// Mock Ant Design Vue 组件
vi.mock('ant-design-vue', () => ({
  Button: {
    name: 'AButton',
    props: ['type', 'loading', 'danger', 'size'],
    setup(props: any, { slots }: any) {
      return () => h('button', { class: 'ant-btn' }, slots.default?.())
    },
  },
  Space: {
    name: 'ASpace',
    props: ['size', 'wrap'],
    setup(props: any, { slots }: any) {
      return () => h('div', { class: 'ant-space' }, slots.default?.())
    },
  },
  Input: {
    name: 'AInput',
    props: ['value', 'placeholder', 'allowClear'],
    emits: ['update:value', 'pressEnter'],
    setup(props: any, { emit }: any) {
      return () => h('input', { 
        class: 'ant-input',
        value: props.value,
        onInput: (e: any) => emit('update:value', e.target.value),
      })
    },
  },
  Select: {
    name: 'ASelect',
    props: ['value', 'placeholder', 'allowClear'],
    emits: ['update:value'],
    Option: {
      name: 'ASelectOption',
      props: ['value'],
      setup(props: any, { slots }: any) {
        return () => h('option', { value: props.value }, slots.default?.())
      },
    },
    setup(props: any, { emit }: any) {
      return () => h('select', { 
        class: 'ant-select',
        value: props.value,
        onChange: (e: any) => emit('update:value', e.target.value),
      })
    },
  },
  Tag: {
    name: 'ATag',
    props: ['color'],
    setup(props: any, { slots }: any) {
      return () => h('span', { class: 'ant-tag' }, slots.default?.())
    },
  },
  Popconfirm: {
    name: 'APopconfirm',
    props: ['title', 'description', 'okText', 'cancelText', 'placement'],
    emits: ['confirm'],
    setup(props: any, { slots, emit }: any) {
      return () => h('div', { 
        class: 'ant-popconfirm',
        onClick: () => emit('confirm'),
      }, slots.default?.())
    },
  },
  Card: {
    name: 'ACard',
    props: ['bordered', 'title'],
    setup(props: any, { slots }: any) {
      return () => h('div', { class: 'ant-card' }, slots.default?.())
    },
  },
  Tooltip: {
    name: 'ATooltip',
    props: ['title'],
    setup(props: any, { slots }: any) {
      return () => h('div', { class: 'ant-tooltip' }, slots.default?.())
    },
  },
  Switch: {
    name: 'ASwitch',
    props: ['checked'],
    emits: ['update:checked'],
    setup(props: any) {
      return () => h('input', { type: 'checkbox', checked: props.checked })
    },
  },
  message: {
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
  },
}))

// Mock icons
vi.mock('@ant-design/icons-vue', () => ({
  PlusOutlined: { render: () => h('span', { class: 'icon-plus' }) },
  EditOutlined: { render: () => h('span', { class: 'icon-edit' }) },
  DeleteOutlined: { render: () => h('span', { class: 'icon-delete' }) },
  SearchOutlined: { render: () => h('span', { class: 'icon-search' }) },
  EyeOutlined: { render: () => h('span', { class: 'icon-eye' }) },
  ReloadOutlined: { render: () => h('span', { class: 'icon-reload' }) },
}))

// Mock DataTable 组件
vi.mock('@/components/DataTable.vue', () => ({
  default: {
    name: 'DataTable',
    props: ['columns', 'dataSource', 'loading', 'pagination', 'scrollX', 'rowKey'],
    emits: ['pageChange'],
    setup(props: any, { slots }: any) {
      return () => h('div', { class: 'data-table' }, [
        h('table', 
          props.dataSource?.map((row: any) => 
            h('tr', { key: row.id }, [
              h('td', row.id),
              h('td', row.title),
              h('td', row.type),
            ])
          )
        ),
        slots.bodyCell?.({ column: { key: 'action' }, record: props.dataSource?.[0] }),
      ])
    },
  },
}))

// Mock API Provider
const mockApiProvider: Partial<IApiProvider> = {
  adminItem: {
    listItems: vi.fn().mockResolvedValue({
      items: [
        { 
          id: '1', 
          title: '物品1', 
          type: 'movie', 
          category: '动作',
          tags: ['标签1'],
          status: 'active',
          description: '',
          created_at: new Date().toISOString(), 
          updated_at: new Date().toISOString() 
        },
        { 
          id: '2', 
          title: '物品2', 
          type: 'product', 
          category: '电子',
          tags: ['标签2'],
          status: 'inactive',
          description: '',
          created_at: new Date().toISOString(), 
          updated_at: new Date().toISOString() 
        },
      ],
      total: 2,
    }),
    getItem: vi.fn().mockResolvedValue({ id: '1', title: '物品1', type: 'movie' }),
    createItem: vi.fn().mockResolvedValue({ id: '3', title: '新物品' }),
    updateItem: vi.fn().mockResolvedValue({ id: '1', title: '更新物品' }),
    deleteItem: vi.fn().mockResolvedValue(undefined),
  },
}

// 创建测试路由
const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/admin/items', name: 'ItemList', component: ItemList },
    { path: '/admin/items/create', name: 'ItemCreate', component: { template: '<div />' } },
    { path: '/admin/items/:id', name: 'ItemDetail', component: { template: '<div />' } },
    { path: '/admin/items/:id/edit', name: 'ItemEdit', component: { template: '<div />' } },
  ],
})

describe('ItemList', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  function createWrapper() {
    return mount(ItemList, {
      global: {
        plugins: [router],
        provide: {
          api: mockApiProvider,
        },
        stubs: {
          RouterLink: true,
        },
      },
    })
  }

  it('应该正确渲染页面', async () => {
    const wrapper = createWrapper()
    await flushPromises()

    expect(wrapper.find('.item-list-page').exists()).toBe(true)
    expect(wrapper.find('.page-title').text()).toContain('物品管理')
  })

  it('应该在挂载时加载物品列表', async () => {
    createWrapper()
    await flushPromises()

    expect(mockApiProvider.adminItem!.listItems).toHaveBeenCalledWith({
      page: 1,
      page_size: 10,
      keyword: undefined,
      type: undefined,
    })
  })

  it('应该显示搜索表单', async () => {
    const wrapper = createWrapper()
    await flushPromises()

    expect(wrapper.find('.search-card').exists()).toBe(true)
    expect(wrapper.find('.ant-input').exists()).toBe(true)
    expect(wrapper.findAll('.ant-select').length).toBeGreaterThanOrEqual(1)
  })

  it('应该显示新增按钮', async () => {
    const wrapper = createWrapper()
    await flushPromises()

    const buttons = wrapper.findAll('.ant-btn')
    const addButton = buttons.find(btn => btn.text().includes('新增'))
    expect(addButton).toBeTruthy()
  })

  it('应该显示数据表格', async () => {
    const wrapper = createWrapper()
    await flushPromises()

    expect(wrapper.find('.data-table').exists()).toBe(true)
  })

  it('重置应该清空搜索条件', async () => {
    const wrapper = createWrapper()
    await flushPromises()

    // 清除之前的调用记录
    vi.clearAllMocks()

    // 触发重置按钮
    const resetButton = wrapper.findAll('.ant-btn').find(btn => btn.text().includes('重置'))
    await resetButton?.trigger('click')
    await flushPromises()

    expect(mockApiProvider.adminItem!.listItems).toHaveBeenCalledWith(
      expect.objectContaining({
        page: 1,
        keyword: undefined,
        type: undefined,
      })
    )
  })

  it('应该正确处理空数据', async () => {
    // 修改 mock 返回空数据
    mockApiProvider.adminItem!.listItems = vi.fn().mockResolvedValue({
      items: [],
      total: 0,
    })

    const wrapper = createWrapper()
    await flushPromises()

    expect(wrapper.find('.data-table').exists()).toBe(true)
  })
})

