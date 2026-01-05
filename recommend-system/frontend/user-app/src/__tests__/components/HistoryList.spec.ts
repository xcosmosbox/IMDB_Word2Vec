/**
 * HistoryList 组件单元测试
 * 
 * 测试历史记录列表组件的渲染和交互。
 * 
 * @author Person C
 */

import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import HistoryList from '@/components/HistoryList.vue'
import type { UserBehavior } from '@shared/types'

// Mock 行为数据
const mockBehaviors: UserBehavior[] = [
  {
    user_id: 'user-123',
    item_id: 'item-001',
    action: 'view',
    timestamp: new Date().toISOString(),
  },
  {
    user_id: 'user-123',
    item_id: 'item-002',
    action: 'like',
    timestamp: new Date(Date.now() - 3600000).toISOString(), // 1小时前
  },
  {
    user_id: 'user-123',
    item_id: 'item-003',
    action: 'click',
    timestamp: new Date(Date.now() - 86400000).toISOString(), // 1天前
  },
]

describe('HistoryList', () => {
  // ===========================================================================
  // 渲染测试
  // ===========================================================================

  describe('渲染', () => {
    it('应该渲染行为列表', () => {
      const wrapper = mount(HistoryList, {
        props: {
          behaviors: mockBehaviors,
        },
      })

      const items = wrapper.findAll('.behavior-item')
      expect(items).toHaveLength(mockBehaviors.length)
    })

    it('应该显示正确的操作图标', () => {
      const wrapper = mount(HistoryList, {
        props: {
          behaviors: [mockBehaviors[0]], // view
        },
      })

      expect(wrapper.find('.action-icon').text()).toBe('👁️')
    })

    it('应该显示正确的操作标签', () => {
      const wrapper = mount(HistoryList, {
        props: {
          behaviors: [mockBehaviors[0]], // view
        },
      })

      expect(wrapper.find('.action-label').text()).toBe('浏览')
    })

    it('应该显示物品 ID', () => {
      const wrapper = mount(HistoryList, {
        props: {
          behaviors: [mockBehaviors[0]],
        },
      })

      expect(wrapper.find('.item-id').text()).toBe('item-001')
    })
  })

  // ===========================================================================
  // 时间显示
  // ===========================================================================

  describe('时间显示', () => {
    it('刚刚发生的应该显示"刚刚"', () => {
      const recentBehavior: UserBehavior = {
        user_id: 'user-123',
        item_id: 'item-001',
        action: 'view',
        timestamp: new Date().toISOString(),
      }

      const wrapper = mount(HistoryList, {
        props: {
          behaviors: [recentBehavior],
          showTimestamp: true,
        },
      })

      expect(wrapper.find('.item-time').text()).toBe('刚刚')
    })

    it('几分钟前应该显示"X 分钟前"', () => {
      const minAgo = new Date(Date.now() - 5 * 60000) // 5分钟前
      const behavior: UserBehavior = {
        user_id: 'user-123',
        item_id: 'item-001',
        action: 'view',
        timestamp: minAgo.toISOString(),
      }

      const wrapper = mount(HistoryList, {
        props: {
          behaviors: [behavior],
          showTimestamp: true,
        },
      })

      expect(wrapper.find('.item-time').text()).toContain('分钟前')
    })

    it('showTimestamp 为 false 时不应该显示时间', () => {
      const wrapper = mount(HistoryList, {
        props: {
          behaviors: mockBehaviors,
          showTimestamp: false,
        },
      })

      expect(wrapper.find('.item-time').exists()).toBe(false)
    })
  })

  // ===========================================================================
  // 日期分组
  // ===========================================================================

  describe('日期分组', () => {
    it('groupByDate 为 true 时应该显示日期标题', () => {
      const wrapper = mount(HistoryList, {
        props: {
          behaviors: mockBehaviors,
          groupByDate: true,
        },
      })

      expect(wrapper.find('.group-header').exists()).toBe(true)
    })

    it('今天的记录应该显示"今天"', () => {
      const todayBehavior: UserBehavior = {
        user_id: 'user-123',
        item_id: 'item-001',
        action: 'view',
        timestamp: new Date().toISOString(),
      }

      const wrapper = mount(HistoryList, {
        props: {
          behaviors: [todayBehavior],
          groupByDate: true,
        },
      })

      expect(wrapper.find('.group-date').text()).toBe('今天')
    })

    it('应该显示每组的记录数', () => {
      const wrapper = mount(HistoryList, {
        props: {
          behaviors: mockBehaviors,
          groupByDate: true,
        },
      })

      const countBadge = wrapper.find('.group-count')
      expect(countBadge.exists()).toBe(true)
      expect(countBadge.text()).toContain('条记录')
    })

    it('groupByDate 为 false 时不应该显示日期标题', () => {
      const wrapper = mount(HistoryList, {
        props: {
          behaviors: mockBehaviors,
          groupByDate: false,
        },
      })

      expect(wrapper.find('.group-header').exists()).toBe(false)
    })
  })

  // ===========================================================================
  // 空状态
  // ===========================================================================

  describe('空状态', () => {
    it('无数据时应该显示空状态', () => {
      const wrapper = mount(HistoryList, {
        props: {
          behaviors: [],
        },
      })

      expect(wrapper.find('.empty-state').exists()).toBe(true)
      expect(wrapper.find('.empty-text').text()).toBe('暂无历史记录')
    })

    it('空状态应该显示提示信息', () => {
      const wrapper = mount(HistoryList, {
        props: {
          behaviors: [],
        },
      })

      expect(wrapper.find('.empty-hint').exists()).toBe(true)
    })
  })

  // ===========================================================================
  // 事件
  // ===========================================================================

  describe('事件', () => {
    it('点击项目应该触发 item-click 事件', async () => {
      const wrapper = mount(HistoryList, {
        props: {
          behaviors: mockBehaviors,
        },
      })

      await wrapper.find('.behavior-item').trigger('click')

      expect(wrapper.emitted('item-click')).toHaveLength(1)
      expect(wrapper.emitted('item-click')![0]).toEqual(['item-001'])
    })
  })

  // ===========================================================================
  // 不同操作类型
  // ===========================================================================

  describe('不同操作类型', () => {
    const actionTypes = [
      { action: 'view', icon: '👁️', label: '浏览' },
      { action: 'click', icon: '👆', label: '点击' },
      { action: 'like', icon: '❤️', label: '喜欢' },
      { action: 'dislike', icon: '👎', label: '不喜欢' },
      { action: 'buy', icon: '🛒', label: '购买' },
      { action: 'share', icon: '🔗', label: '分享' },
    ]

    actionTypes.forEach(({ action, icon, label }) => {
      it(`${action} 操作应该显示正确的图标和标签`, () => {
        const behavior: UserBehavior = {
          user_id: 'user-123',
          item_id: 'item-001',
          action,
          timestamp: new Date().toISOString(),
        }

        const wrapper = mount(HistoryList, {
          props: {
            behaviors: [behavior],
          },
        })

        expect(wrapper.find('.action-icon').text()).toBe(icon)
        expect(wrapper.find('.action-label').text()).toBe(label)
      })
    })

    it('未知操作类型应该显示默认图标', () => {
      const behavior: UserBehavior = {
        user_id: 'user-123',
        item_id: 'item-001',
        action: 'unknown_action',
        timestamp: new Date().toISOString(),
      }

      const wrapper = mount(HistoryList, {
        props: {
          behaviors: [behavior],
        },
      })

      expect(wrapper.find('.action-icon').text()).toBe('📌')
      expect(wrapper.find('.action-label').text()).toBe('unknown_action')
    })
  })
})

