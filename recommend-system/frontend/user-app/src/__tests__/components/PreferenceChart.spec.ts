/**
 * PreferenceChart 组件单元测试
 * 
 * 测试偏好图表组件的渲染和数据处理。
 * 
 * @author Person C
 */

import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import PreferenceChart from '@/components/PreferenceChart.vue'

// Mock 偏好数据
const mockData: Record<string, number> = {
  movie: 50,
  product: 30,
  article: 20,
  video: 10,
}

describe('PreferenceChart', () => {
  // ===========================================================================
  // 渲染测试
  // ===========================================================================

  describe('渲染', () => {
    it('应该显示图表标题', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: mockData,
          title: '我的偏好',
        },
      })

      expect(wrapper.find('.chart-title').text()).toBe('我的偏好')
    })

    it('应该使用默认标题', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: mockData,
        },
      })

      expect(wrapper.find('.chart-title').text()).toBe('内容偏好')
    })

    it('应该显示总计数', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: mockData,
        },
      })

      // 50 + 30 + 20 + 10 = 110
      expect(wrapper.find('.total-count').text()).toContain('110')
    })

    it('应该渲染所有类型的条形', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: mockData,
        },
      })

      const items = wrapper.findAll('.chart-item')
      expect(items).toHaveLength(Object.keys(mockData).length)
    })
  })

  // ===========================================================================
  // 数据处理
  // ===========================================================================

  describe('数据处理', () => {
    it('应该按数值降序排列', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: mockData,
        },
      })

      const values = wrapper.findAll('.item-value').map(el => parseInt(el.text()))
      
      // 验证是降序
      for (let i = 0; i < values.length - 1; i++) {
        expect(values[i]).toBeGreaterThanOrEqual(values[i + 1])
      }
    })

    it('应该正确计算百分比', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: { movie: 50, product: 50 },
          showPercentage: true,
        },
      })

      const percentages = wrapper.findAll('.item-percentage')
      expect(percentages[0].text()).toContain('50.0%')
      expect(percentages[1].text()).toContain('50.0%')
    })

    it('showPercentage 为 false 时不应该显示百分比', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: mockData,
          showPercentage: false,
        },
      })

      expect(wrapper.find('.item-percentage').exists()).toBe(false)
    })

    it('应该限制最大显示数量', () => {
      const manyItems = {
        type1: 10,
        type2: 20,
        type3: 30,
        type4: 40,
        type5: 50,
        type6: 60,
        type7: 70,
        type8: 80,
      }

      const wrapper = mount(PreferenceChart, {
        props: {
          data: manyItems,
          maxItems: 3,
        },
      })

      const items = wrapper.findAll('.chart-item')
      expect(items).toHaveLength(3)
    })
  })

  // ===========================================================================
  // 类型映射
  // ===========================================================================

  describe('类型映射', () => {
    it('应该显示中文类型名称', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: { movie: 100 },
        },
      })

      expect(wrapper.find('.item-name').text()).toBe('电影')
    })

    it('未知类型应该显示原始名称', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: { unknown_type: 100 },
        },
      })

      expect(wrapper.find('.item-name').text()).toBe('unknown_type')
    })

    const typeMapping = [
      { type: 'movie', name: '电影' },
      { type: 'product', name: '商品' },
      { type: 'article', name: '文章' },
      { type: 'video', name: '视频' },
      { type: 'music', name: '音乐' },
      { type: 'book', name: '图书' },
    ]

    typeMapping.forEach(({ type, name }) => {
      it(`${type} 应该显示为 ${name}`, () => {
        const wrapper = mount(PreferenceChart, {
          props: {
            data: { [type]: 100 },
          },
        })

        expect(wrapper.find('.item-name').text()).toBe(name)
      })
    })
  })

  // ===========================================================================
  // 进度条
  // ===========================================================================

  describe('进度条', () => {
    it('最大值应该有 100% 宽度', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: { movie: 100, product: 50 },
        },
      })

      const bars = wrapper.findAll('.bar-fill')
      const firstBarStyle = bars[0].attributes('style')
      
      expect(firstBarStyle).toContain('width: 100%')
    })

    it('其他值应该相对于最大值计算宽度', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: { movie: 100, product: 50 },
        },
      })

      const bars = wrapper.findAll('.bar-fill')
      const secondBarStyle = bars[1].attributes('style')
      
      expect(secondBarStyle).toContain('width: 50%')
    })
  })

  // ===========================================================================
  // 空状态
  // ===========================================================================

  describe('空状态', () => {
    it('空数据应该显示空状态', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: {},
        },
      })

      expect(wrapper.find('.empty-state').exists()).toBe(true)
      expect(wrapper.find('.empty-text').text()).toBe('暂无偏好数据')
    })

    it('null 数据应该显示空状态', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: null as any,
        },
      })

      expect(wrapper.find('.empty-state').exists()).toBe(true)
    })

    it('空状态应该显示提示信息', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: {},
        },
      })

      expect(wrapper.find('.empty-hint').exists()).toBe(true)
    })

    it('有数据时不应该显示空状态', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: mockData,
        },
      })

      expect(wrapper.find('.empty-state').exists()).toBe(false)
    })

    it('有数据时不应该显示总计', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: {},
        },
      })

      expect(wrapper.find('.total-count').exists()).toBe(false)
    })
  })

  // ===========================================================================
  // Props 默认值
  // ===========================================================================

  describe('Props 默认值', () => {
    it('maxItems 默认为 6', () => {
      const manyItems = {
        type1: 10,
        type2: 20,
        type3: 30,
        type4: 40,
        type5: 50,
        type6: 60,
        type7: 70,
      }

      const wrapper = mount(PreferenceChart, {
        props: {
          data: manyItems,
        },
      })

      const items = wrapper.findAll('.chart-item')
      expect(items).toHaveLength(6)
    })

    it('showPercentage 默认为 true', () => {
      const wrapper = mount(PreferenceChart, {
        props: {
          data: mockData,
        },
      })

      expect(wrapper.find('.item-percentage').exists()).toBe(true)
    })
  })
})

