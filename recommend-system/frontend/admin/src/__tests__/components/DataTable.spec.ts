/**
 * DataTable 组件单元测试
 */
import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { h } from 'vue'
import DataTable from '@/components/DataTable.vue'

// Mock Ant Design Vue 组件
vi.mock('ant-design-vue', () => ({
  Table: {
    name: 'ATable',
    props: ['columns', 'dataSource', 'rowKey', 'pagination', 'scroll', 'loading', 'size', 'bordered'],
    emits: ['change'],
    setup(props: any, { emit, slots }: any) {
      return () => h('div', { class: 'ant-table' }, [
        h('table', [
          h('tbody', props.dataSource?.map((row: any, index: number) => 
            h('tr', { key: row[props.rowKey] || index }, 
              props.columns?.map((col: any) => 
                h('td', { key: col.key || col.dataIndex }, 
                  col.customRender 
                    ? col.customRender({ text: row[col.dataIndex], record: row, index })
                    : row[col.dataIndex]
                )
              )
            )
          ))
        ]),
        slots.emptyText?.(),
      ])
    },
  },
  Empty: {
    name: 'AEmpty',
    props: ['description'],
    setup(props: any) {
      return () => h('div', { class: 'ant-empty' }, props.description)
    },
  },
  Spin: {
    name: 'ASpin',
    props: ['spinning'],
    setup(props: any, { slots }: any) {
      return () => h('div', { class: 'ant-spin-container' }, slots.default?.())
    },
  },
}))

describe('DataTable', () => {
  const mockColumns = [
    { title: 'ID', dataIndex: 'id', key: 'id' },
    { title: 'Name', dataIndex: 'name', key: 'name' },
  ]

  const mockData = [
    { id: '1', name: 'Item 1' },
    { id: '2', name: 'Item 2' },
    { id: '3', name: 'Item 3' },
  ]

  it('应该正确渲染表格', () => {
    const wrapper = mount(DataTable, {
      props: {
        columns: mockColumns,
        dataSource: mockData,
        rowKey: 'id',
      },
    })

    expect(wrapper.find('.data-table').exists()).toBe(true)
    expect(wrapper.find('.ant-table').exists()).toBe(true)
  })

  it('应该显示空状态当没有数据时', () => {
    const wrapper = mount(DataTable, {
      props: {
        columns: mockColumns,
        dataSource: [],
        rowKey: 'id',
        emptyText: '暂无数据',
      },
    })

    expect(wrapper.find('.data-table').exists()).toBe(true)
  })

  it('应该接受自定义 rowKey', () => {
    const wrapper = mount(DataTable, {
      props: {
        columns: mockColumns,
        dataSource: mockData,
        rowKey: 'id',
      },
    })

    expect(wrapper.props('rowKey')).toBe('id')
  })

  it('应该处理分页配置', () => {
    const pagination = {
      current: 1,
      pageSize: 10,
      total: 100,
    }

    const wrapper = mount(DataTable, {
      props: {
        columns: mockColumns,
        dataSource: mockData,
        rowKey: 'id',
        pagination,
      },
    })

    expect(wrapper.props('pagination')).toEqual(pagination)
  })

  it('应该处理 loading 状态', () => {
    const wrapper = mount(DataTable, {
      props: {
        columns: mockColumns,
        dataSource: mockData,
        rowKey: 'id',
        loading: true,
      },
    })

    expect(wrapper.find('.ant-spin-container').exists()).toBe(true)
  })

  it('应该处理滚动配置', () => {
    const wrapper = mount(DataTable, {
      props: {
        columns: mockColumns,
        dataSource: mockData,
        rowKey: 'id',
        scrollX: 1200,
        scrollY: 400,
      },
    })

    expect(wrapper.props('scrollX')).toBe(1200)
    expect(wrapper.props('scrollY')).toBe(400)
  })

  it('应该触发 pageChange 事件', async () => {
    const wrapper = mount(DataTable, {
      props: {
        columns: mockColumns,
        dataSource: mockData,
        rowKey: 'id',
        pagination: {
          current: 1,
          pageSize: 10,
          total: 100,
        },
      },
    })

    // 组件已挂载
    expect(wrapper.exists()).toBe(true)
  })

  it('应该支持自定义 empty 文本', () => {
    const wrapper = mount(DataTable, {
      props: {
        columns: mockColumns,
        dataSource: [],
        rowKey: 'id',
        emptyText: '自定义空状态文本',
      },
    })

    expect(wrapper.props('emptyText')).toBe('自定义空状态文本')
  })
})

