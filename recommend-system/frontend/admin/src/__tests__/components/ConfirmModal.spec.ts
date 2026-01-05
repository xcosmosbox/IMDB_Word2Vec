/**
 * ConfirmModal 组件单元测试
 */
import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { h, nextTick } from 'vue'
import ConfirmModal from '@/components/ConfirmModal.vue'

// Mock Ant Design Vue 组件
vi.mock('ant-design-vue', () => ({
  Modal: {
    name: 'AModal',
    props: ['open', 'title', 'footer', 'width', 'closable', 'maskClosable', 'centered'],
    emits: ['cancel'],
    setup(props: any, { slots, emit }: any) {
      return () => props.open 
        ? h('div', { class: 'ant-modal' }, [
            h('div', { class: 'ant-modal-content' }, slots.default?.()),
          ])
        : null
    },
  },
  Button: {
    name: 'AButton',
    props: ['type', 'danger', 'loading'],
    emits: ['click'],
    setup(props: any, { slots, emit }: any) {
      return () => h('button', { 
        class: ['ant-btn', props.type && `ant-btn-${props.type}`, props.danger && 'ant-btn-danger'],
        onClick: () => emit('click'),
      }, slots.default?.())
    },
  },
  Space: {
    name: 'ASpace',
    setup(props: any, { slots }: any) {
      return () => h('div', { class: 'ant-space' }, slots.default?.())
    },
  },
  Alert: {
    name: 'AAlert',
    props: ['message', 'type'],
    setup(props: any) {
      return () => h('div', { class: 'ant-alert' }, props.message)
    },
  },
}))

// Mock icons
vi.mock('@ant-design/icons-vue', () => ({
  ExclamationCircleOutlined: { render: () => h('span', { class: 'icon-exclamation' }) },
  DeleteOutlined: { render: () => h('span', { class: 'icon-delete' }) },
  WarningOutlined: { render: () => h('span', { class: 'icon-warning' }) },
  InfoCircleOutlined: { render: () => h('span', { class: 'icon-info' }) },
  CheckCircleOutlined: { render: () => h('span', { class: 'icon-check' }) },
}))

describe('ConfirmModal', () => {
  it('应该在 open 为 true 时显示', () => {
    const wrapper = mount(ConfirmModal, {
      props: {
        open: true,
        title: '确认删除',
        content: '确定要删除这条数据吗？',
      },
    })

    expect(wrapper.find('.ant-modal').exists()).toBe(true)
  })

  it('应该在 open 为 false 时隐藏', () => {
    const wrapper = mount(ConfirmModal, {
      props: {
        open: false,
        title: '确认删除',
        content: '确定要删除这条数据吗？',
      },
    })

    expect(wrapper.find('.ant-modal').exists()).toBe(false)
  })

  it('应该显示正确的标题和内容', () => {
    const wrapper = mount(ConfirmModal, {
      props: {
        open: true,
        title: '测试标题',
        content: '测试内容',
      },
    })

    expect(wrapper.text()).toContain('测试标题')
    expect(wrapper.text()).toContain('测试内容')
  })

  it('应该触发 ok 事件当点击确定按钮', async () => {
    const wrapper = mount(ConfirmModal, {
      props: {
        open: true,
        title: '确认',
        content: '确定吗？',
      },
    })

    const okButton = wrapper.findAll('button').find(btn => btn.classes().includes('ant-btn-primary'))
    await okButton?.trigger('click')

    expect(wrapper.emitted('ok')).toBeTruthy()
  })

  it('应该触发 cancel 事件当点击取消按钮', async () => {
    const wrapper = mount(ConfirmModal, {
      props: {
        open: true,
        title: '确认',
        content: '确定吗？',
      },
    })

    const cancelButton = wrapper.findAll('button').find(btn => !btn.classes().includes('ant-btn-primary'))
    await cancelButton?.trigger('click')

    expect(wrapper.emitted('cancel')).toBeTruthy()
  })

  it('应该支持自定义按钮文本', () => {
    const wrapper = mount(ConfirmModal, {
      props: {
        open: true,
        title: '确认',
        content: '确定吗？',
        okText: '是的',
        cancelText: '不是',
      },
    })

    expect(wrapper.text()).toContain('是的')
    expect(wrapper.text()).toContain('不是')
  })

  it('应该支持 danger 类型', () => {
    const wrapper = mount(ConfirmModal, {
      props: {
        open: true,
        title: '删除确认',
        content: '确定要删除吗？',
        type: 'error',
        okDanger: true,
      },
    })

    const dangerButton = wrapper.find('.ant-btn-danger')
    expect(dangerButton.exists()).toBe(true)
  })

  it('应该显示 loading 状态', () => {
    const wrapper = mount(ConfirmModal, {
      props: {
        open: true,
        title: '确认',
        content: '确定吗？',
        loading: true,
      },
    })

    expect(wrapper.props('loading')).toBe(true)
  })

  it('应该支持不显示取消按钮', () => {
    const wrapper = mount(ConfirmModal, {
      props: {
        open: true,
        title: '提示',
        content: '操作成功',
        showCancel: false,
      },
    })

    // showCancel 为 false 时只有一个按钮
    const buttons = wrapper.findAll('button')
    expect(buttons.length).toBe(1)
  })
})

