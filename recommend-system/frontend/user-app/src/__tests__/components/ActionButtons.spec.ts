/**
 * ActionButtons.vue 单元测试
 * 
 * Person B 开发
 */
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import ActionButtons from '@/components/ActionButtons.vue'

describe('ActionButtons.vue', () => {
  const mountComponent = (props = {}) => {
    return mount(ActionButtons, {
      props: {
        isLiked: false,
        disabled: false,
        ...props,
      },
    })
  }

  describe('渲染', () => {
    it('应该渲染喜欢按钮', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="like-button"]').exists()).toBe(true)
    })

    it('应该渲染分享按钮', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="share-button"]').exists()).toBe(true)
    })
  })

  describe('喜欢按钮', () => {
    it('未喜欢状态应该显示"喜欢"文本', () => {
      const wrapper = mountComponent({ isLiked: false })
      
      expect(wrapper.find('[data-testid="like-button"]').text()).toBe('喜欢')
    })

    it('已喜欢状态应该显示"已喜欢"文本', () => {
      const wrapper = mountComponent({ isLiked: true })
      
      expect(wrapper.find('[data-testid="like-button"]').text()).toBe('已喜欢')
    })

    it('已喜欢状态应该添加 liked 样式类', () => {
      const wrapper = mountComponent({ isLiked: true })
      
      expect(wrapper.find('[data-testid="like-button"]').classes()).toContain('liked')
    })

    it('未喜欢状态不应该有 liked 样式类', () => {
      const wrapper = mountComponent({ isLiked: false })
      
      expect(wrapper.find('[data-testid="like-button"]').classes()).not.toContain('liked')
    })

    it('点击喜欢按钮应该触发 like 事件', async () => {
      const wrapper = mountComponent()
      
      await wrapper.find('[data-testid="like-button"]').trigger('click')
      
      expect(wrapper.emitted('like')).toBeTruthy()
      expect(wrapper.emitted('like')!.length).toBe(1)
    })
  })

  describe('分享按钮', () => {
    it('应该显示"分享"文本', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="share-button"]').text()).toBe('分享')
    })

    it('点击分享按钮应该触发 share 事件', async () => {
      const wrapper = mountComponent()
      
      await wrapper.find('[data-testid="share-button"]').trigger('click')
      
      expect(wrapper.emitted('share')).toBeTruthy()
      expect(wrapper.emitted('share')!.length).toBe(1)
    })
  })

  describe('禁用状态', () => {
    it('禁用时喜欢按钮应该被禁用', () => {
      const wrapper = mountComponent({ disabled: true })
      
      const likeButton = wrapper.find('[data-testid="like-button"]')
      expect(likeButton.attributes('disabled')).toBeDefined()
    })

    it('禁用时分享按钮应该被禁用', () => {
      const wrapper = mountComponent({ disabled: true })
      
      const shareButton = wrapper.find('[data-testid="share-button"]')
      expect(shareButton.attributes('disabled')).toBeDefined()
    })

    it('禁用时点击喜欢按钮不应该触发 like 事件', async () => {
      const wrapper = mountComponent({ disabled: true })
      
      await wrapper.find('[data-testid="like-button"]').trigger('click')
      
      expect(wrapper.emitted('like')).toBeFalsy()
    })

    it('禁用时点击分享按钮不应该触发 share 事件', async () => {
      const wrapper = mountComponent({ disabled: true })
      
      await wrapper.find('[data-testid="share-button"]').trigger('click')
      
      expect(wrapper.emitted('share')).toBeFalsy()
    })
  })

  describe('多次点击', () => {
    it('多次点击喜欢按钮应该触发多次事件', async () => {
      const wrapper = mountComponent()
      
      const likeButton = wrapper.find('[data-testid="like-button"]')
      await likeButton.trigger('click')
      await likeButton.trigger('click')
      await likeButton.trigger('click')
      
      expect(wrapper.emitted('like')!.length).toBe(3)
    })

    it('多次点击分享按钮应该触发多次事件', async () => {
      const wrapper = mountComponent()
      
      const shareButton = wrapper.find('[data-testid="share-button"]')
      await shareButton.trigger('click')
      await shareButton.trigger('click')
      
      expect(wrapper.emitted('share')!.length).toBe(2)
    })
  })

  describe('样式类', () => {
    it('喜欢按钮应该有 like-btn 样式类', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="like-button"]').classes()).toContain('like-btn')
    })

    it('分享按钮应该有 share-btn 样式类', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="share-button"]').classes()).toContain('share-btn')
    })

    it('所有按钮应该有 action-btn 样式类', () => {
      const wrapper = mountComponent()
      
      expect(wrapper.find('[data-testid="like-button"]').classes()).toContain('action-btn')
      expect(wrapper.find('[data-testid="share-button"]').classes()).toContain('action-btn')
    })
  })

  describe('图标', () => {
    it('喜欢按钮应该包含图标', () => {
      const wrapper = mountComponent()
      
      const likeButton = wrapper.find('[data-testid="like-button"]')
      expect(likeButton.find('svg').exists()).toBe(true)
    })

    it('分享按钮应该包含图标', () => {
      const wrapper = mountComponent()
      
      const shareButton = wrapper.find('[data-testid="share-button"]')
      expect(shareButton.find('svg').exists()).toBe(true)
    })

    it('已喜欢状态应该显示填充的心形图标', () => {
      const wrapper = mountComponent({ isLiked: true })
      
      const likeButton = wrapper.find('[data-testid="like-button"]')
      const svg = likeButton.find('svg')
      expect(svg.attributes('fill')).toBe('currentColor')
    })

    it('未喜欢状态应该显示空心心形图标', () => {
      const wrapper = mountComponent({ isLiked: false })
      
      const likeButton = wrapper.find('[data-testid="like-button"]')
      const svg = likeButton.find('svg')
      expect(svg.attributes('fill')).toBe('none')
    })
  })
})

