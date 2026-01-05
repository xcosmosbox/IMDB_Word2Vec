/**
 * AuthForm 组件单元测试
 * 
 * 测试认证表单组件的渲染、交互和事件。
 * 
 * @author Person C
 */

import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import AuthForm from '@/components/AuthForm.vue'

describe('AuthForm', () => {
  // ===========================================================================
  // 渲染测试
  // ===========================================================================

  describe('渲染', () => {
    it('登录模式应该显示正确的标题', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
        },
      })

      expect(wrapper.find('.form-title').text()).toBe('账号登录')
    })

    it('注册模式应该显示正确的标题', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'register',
        },
      })

      expect(wrapper.find('.form-title').text()).toBe('创建账号')
    })

    it('登录模式按钮应该显示"登录"', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
        },
      })

      expect(wrapper.find('.submit-btn').text()).toContain('登录')
    })

    it('注册模式按钮应该显示"注册"', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'register',
        },
      })

      expect(wrapper.find('.submit-btn').text()).toContain('注册')
    })

    it('应该支持自定义按钮文字', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
          submitText: '自定义按钮',
        },
      })

      expect(wrapper.find('.submit-btn').text()).toContain('自定义按钮')
    })
  })

  // ===========================================================================
  // 错误显示
  // ===========================================================================

  describe('错误显示', () => {
    it('应该显示错误信息', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
          error: '登录失败',
        },
      })

      expect(wrapper.find('.error-alert').exists()).toBe(true)
      expect(wrapper.find('.error-text').text()).toBe('登录失败')
    })

    it('没有错误时不应该显示错误区域', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
          error: '',
        },
      })

      expect(wrapper.find('.error-alert').exists()).toBe(false)
    })
  })

  // ===========================================================================
  // 加载状态
  // ===========================================================================

  describe('加载状态', () => {
    it('加载时按钮应该禁用', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
          loading: true,
        },
      })

      expect(wrapper.find('.submit-btn').attributes('disabled')).toBeDefined()
    })

    it('加载时应该显示加载动画', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
          loading: true,
        },
      })

      expect(wrapper.find('.loading-spinner').exists()).toBe(true)
    })

    it('按钮应该有 loading 类', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
          loading: true,
        },
      })

      expect(wrapper.find('.submit-btn').classes()).toContain('loading')
    })
  })

  // ===========================================================================
  // 插槽
  // ===========================================================================

  describe('插槽', () => {
    it('应该渲染 fields 插槽内容', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
        },
        slots: {
          fields: '<div class="test-field">测试字段</div>',
        },
      })

      expect(wrapper.find('.test-field').exists()).toBe(true)
      expect(wrapper.find('.test-field').text()).toBe('测试字段')
    })

    it('应该渲染 extra 插槽内容', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
        },
        slots: {
          extra: '<div class="test-extra">额外内容</div>',
        },
      })

      expect(wrapper.find('.test-extra').exists()).toBe(true)
    })

    it('应该渲染 footer 插槽内容', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
        },
        slots: {
          footer: '<div class="test-footer">底部内容</div>',
        },
      })

      expect(wrapper.find('.test-footer').exists()).toBe(true)
    })
  })

  // ===========================================================================
  // 事件
  // ===========================================================================

  describe('事件', () => {
    it('表单提交应该触发 submit 事件', async () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
        },
      })

      await wrapper.find('form').trigger('submit')

      expect(wrapper.emitted('submit')).toHaveLength(1)
    })

    it('点击提交按钮应该触发 submit 事件', async () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
        },
      })

      await wrapper.find('.submit-btn').trigger('click')

      expect(wrapper.emitted('submit')).toHaveLength(1)
    })

    it('加载时提交应该被阻止', async () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
          loading: true,
        },
      })

      await wrapper.find('form').trigger('submit')

      expect(wrapper.emitted('submit')).toBeUndefined()
    })
  })

  // ===========================================================================
  // Props 默认值
  // ===========================================================================

  describe('Props 默认值', () => {
    it('loading 默认为 false', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
        },
      })

      expect(wrapper.find('.submit-btn').attributes('disabled')).toBeUndefined()
    })

    it('error 默认为空', () => {
      const wrapper = mount(AuthForm, {
        props: {
          type: 'login',
        },
      })

      expect(wrapper.find('.error-alert').exists()).toBe(false)
    })
  })
})

