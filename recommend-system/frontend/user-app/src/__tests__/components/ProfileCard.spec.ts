/**
 * ProfileCard 组件单元测试
 * 
 * 测试个人信息卡片组件的渲染、编辑和事件。
 * 
 * @author Person C
 */

import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import ProfileCard from '@/components/ProfileCard.vue'
import type { User } from '@shared/types'

// Mock 用户数据
const mockUser: User = {
  id: 'user-123',
  name: '张三',
  email: 'zhangsan@example.com',
  age: 28,
  gender: 'male',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2025-01-04T00:00:00Z',
}

describe('ProfileCard', () => {
  // ===========================================================================
  // 查看模式
  // ===========================================================================

  describe('查看模式', () => {
    it('应该显示用户头像', () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
        },
      })

      expect(wrapper.find('.avatar-text').text()).toBe('张')
    })

    it('应该显示用户名称', () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
        },
      })

      expect(wrapper.find('.user-name').text()).toBe('张三')
    })

    it('应该显示用户邮箱', () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
        },
      })

      expect(wrapper.find('.user-email').text()).toBe('zhangsan@example.com')
    })

    it('应该显示用户年龄', () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
        },
      })

      const infoValues = wrapper.findAll('.info-value')
      const ageValue = infoValues.find(el => el.text() === '28')
      expect(ageValue).toBeDefined()
    })

    it('应该显示用户性别（中文）', () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
        },
      })

      const infoValues = wrapper.findAll('.info-value')
      const genderValue = infoValues.find(el => el.text() === '男')
      expect(genderValue).toBeDefined()
    })

    it('应该显示编辑按钮', () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
        },
      })

      expect(wrapper.find('.edit-btn').exists()).toBe(true)
    })

    it('点击编辑按钮应该触发 edit 事件', async () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
        },
      })

      await wrapper.find('.edit-btn').trigger('click')

      expect(wrapper.emitted('edit')).toHaveLength(1)
    })
  })

  // ===========================================================================
  // 编辑模式
  // ===========================================================================

  describe('编辑模式', () => {
    it('应该显示编辑表单', () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
          isEditing: true,
        },
      })

      expect(wrapper.find('.edit-section').exists()).toBe(true)
      expect(wrapper.find('.info-section').exists()).toBe(false)
    })

    it('应该预填充当前用户信息', () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
          isEditing: true,
        },
      })

      const nameInput = wrapper.find('input[type="text"]')
      expect((nameInput.element as HTMLInputElement).value).toBe('张三')
    })

    it('应该显示保存和取消按钮', () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
          isEditing: true,
        },
      })

      expect(wrapper.find('.save-btn').exists()).toBe(true)
      expect(wrapper.find('.cancel-btn').exists()).toBe(true)
    })

    it('点击保存应该触发 save 事件', async () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
          isEditing: true,
        },
      })

      await wrapper.find('.save-btn').trigger('click')

      expect(wrapper.emitted('save')).toHaveLength(1)
    })

    it('保存事件应该包含表单数据', async () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
          isEditing: true,
        },
      })

      // 修改名称
      const nameInput = wrapper.find('input[type="text"]')
      await nameInput.setValue('李四')

      await wrapper.find('.save-btn').trigger('click')

      const emitted = wrapper.emitted('save')
      expect(emitted).toHaveLength(1)
      expect((emitted![0][0] as any).name).toBe('李四')
    })

    it('点击取消应该触发 cancel 事件', async () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
          isEditing: true,
        },
      })

      await wrapper.find('.cancel-btn').trigger('click')

      expect(wrapper.emitted('cancel')).toHaveLength(1)
    })
  })

  // ===========================================================================
  // 加载状态
  // ===========================================================================

  describe('加载状态', () => {
    it('加载时保存按钮应该禁用', () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
          isEditing: true,
          loading: true,
        },
      })

      expect(wrapper.find('.save-btn').attributes('disabled')).toBeDefined()
    })

    it('加载时取消按钮应该禁用', () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
          isEditing: true,
          loading: true,
        },
      })

      expect(wrapper.find('.cancel-btn').attributes('disabled')).toBeDefined()
    })

    it('加载时应该显示加载文字', () => {
      const wrapper = mount(ProfileCard, {
        props: {
          user: mockUser,
          isEditing: true,
          loading: true,
        },
      })

      expect(wrapper.find('.save-btn').text()).toContain('保存中')
    })
  })

  // ===========================================================================
  // 边缘情况
  // ===========================================================================

  describe('边缘情况', () => {
    it('用户无年龄时应该显示"未设置"', () => {
      const userWithoutAge = { ...mockUser, age: 0 }
      const wrapper = mount(ProfileCard, {
        props: {
          user: userWithoutAge,
        },
      })

      const infoValues = wrapper.findAll('.info-value')
      const ageValue = infoValues.find(el => el.text() === '未设置')
      expect(ageValue).toBeDefined()
    })

    it('用户无性别时应该显示"未设置"', () => {
      const userWithoutGender = { ...mockUser, gender: '' }
      const wrapper = mount(ProfileCard, {
        props: {
          user: userWithoutGender,
        },
      })

      const infoValues = wrapper.findAll('.info-value')
      const genderValue = infoValues.find(el => el.text() === '未设置')
      expect(genderValue).toBeDefined()
    })

    it('女性用户应该显示"女"', () => {
      const femaleUser = { ...mockUser, gender: 'female' }
      const wrapper = mount(ProfileCard, {
        props: {
          user: femaleUser,
        },
      })

      const infoValues = wrapper.findAll('.info-value')
      const genderValue = infoValues.find(el => el.text() === '女')
      expect(genderValue).toBeDefined()
    })

    it('其他性别用户应该显示"其他"', () => {
      const otherGenderUser = { ...mockUser, gender: 'other' }
      const wrapper = mount(ProfileCard, {
        props: {
          user: otherGenderUser,
        },
      })

      const infoValues = wrapper.findAll('.info-value')
      const genderValue = infoValues.find(el => el.text() === '其他')
      expect(genderValue).toBeDefined()
    })
  })
})

