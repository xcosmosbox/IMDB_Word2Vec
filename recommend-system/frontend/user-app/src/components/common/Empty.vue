<!--
  Empty 组件
  
  空状态展示组件，用于列表为空或数据未找到时显示
  
  @author Person F
-->
<script setup lang="ts">
/**
 * 组件属性
 */
interface Props {
  /** 空状态图片（可选） */
  image?: string
  /** 描述文本 */
  description?: string
  /** 是否使用小尺寸 */
  small?: boolean
}

withDefaults(defineProps<Props>(), {
  description: '暂无数据',
  small: false,
})

/**
 * 定义插槽
 */
defineSlots<{
  /** 默认插槽，用于放置操作按钮等 */
  default?: () => unknown
  /** 自定义图标插槽 */
  icon?: () => unknown
  /** 自定义描述插槽 */
  description?: () => unknown
}>()
</script>

<template>
  <div class="empty" :class="{ 'empty--small': small }">
    <!-- 图标区域 -->
    <div class="empty__icon">
      <slot name="icon">
        <svg
          v-if="image"
          class="empty__image"
          viewBox="0 0 64 64"
        >
          <image :href="image" width="64" height="64" />
        </svg>
        <svg
          v-else
          class="empty__default-icon"
          viewBox="0 0 64 64"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <!-- 默认空状态图标 -->
          <rect x="8" y="20" width="48" height="36" rx="4" stroke="currentColor" stroke-width="2" />
          <path d="M8 28H56" stroke="currentColor" stroke-width="2" />
          <circle cx="14" cy="24" r="2" fill="currentColor" />
          <circle cx="22" cy="24" r="2" fill="currentColor" />
          <circle cx="30" cy="24" r="2" fill="currentColor" />
          <path d="M24 44L32 36L40 44" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
          <path d="M20 48H44" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
        </svg>
      </slot>
    </div>
    
    <!-- 描述文本 -->
    <div class="empty__description">
      <slot name="description">
        {{ description }}
      </slot>
    </div>
    
    <!-- 操作区域 -->
    <div v-if="$slots.default" class="empty__actions">
      <slot></slot>
    </div>
  </div>
</template>

<style scoped>
.empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 48px 24px;
  text-align: center;
}

.empty--small {
  padding: 24px 16px;
}

.empty__icon {
  margin-bottom: 16px;
  color: #4a5568;
}

.empty__default-icon,
.empty__image {
  width: 64px;
  height: 64px;
}

.empty--small .empty__default-icon,
.empty--small .empty__image {
  width: 48px;
  height: 48px;
}

.empty__description {
  color: #8892b0;
  font-size: 14px;
  line-height: 1.6;
  max-width: 300px;
}

.empty--small .empty__description {
  font-size: 13px;
}

.empty__actions {
  margin-top: 24px;
}

.empty--small .empty__actions {
  margin-top: 16px;
}
</style>

