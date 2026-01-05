<script setup lang="ts">
/**
 * ActionButtons.vue - 操作按钮组组件
 * 
 * 功能：
 * - 喜欢/取消喜欢按钮
 * - 分享按钮
 * - 更多操作按钮
 * 
 * Person B 开发
 */
import { computed } from 'vue'

interface Props {
  /** 是否已喜欢 */
  isLiked?: boolean
  /** 是否禁用 */
  disabled?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  isLiked: false,
  disabled: false,
})

const emit = defineEmits<{
  /** 点击喜欢按钮 */
  'like': []
  /** 点击分享按钮 */
  'share': []
}>()

// =========================================================================
// 计算属性
// =========================================================================

/** 喜欢按钮文本 */
const likeButtonText = computed(() => {
  return props.isLiked ? '已喜欢' : '喜欢'
})

/** 喜欢按钮样式类 */
const likeButtonClass = computed(() => {
  return props.isLiked ? 'liked' : ''
})

// =========================================================================
// 方法
// =========================================================================

/**
 * 处理喜欢按钮点击
 */
function handleLike() {
  if (!props.disabled) {
    emit('like')
  }
}

/**
 * 处理分享按钮点击
 */
function handleShare() {
  if (!props.disabled) {
    emit('share')
  }
}
</script>

<template>
  <div class="action-buttons">
    <!-- 喜欢按钮 -->
    <button
      :class="['action-btn', 'like-btn', likeButtonClass]"
      :disabled="disabled"
      @click="handleLike"
      data-testid="like-button"
    >
      <span class="btn-icon">
        <svg v-if="isLiked" xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
        </svg>
        <svg v-else xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
        </svg>
      </span>
      <span class="btn-text">{{ likeButtonText }}</span>
    </button>

    <!-- 分享按钮 -->
    <button
      class="action-btn share-btn"
      :disabled="disabled"
      @click="handleShare"
      data-testid="share-button"
    >
      <span class="btn-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="18" cy="5" r="3"/>
          <circle cx="6" cy="12" r="3"/>
          <circle cx="18" cy="19" r="3"/>
          <line x1="8.59" x2="15.42" y1="13.51" y2="17.49"/>
          <line x1="15.41" x2="8.59" y1="6.51" y2="10.49"/>
        </svg>
      </span>
      <span class="btn-text">分享</span>
    </button>
  </div>
</template>

<style scoped>
.action-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
}

.action-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.85rem 1.5rem;
  border: none;
  border-radius: 0.75rem;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  font-family: 'Nunito', 'PingFang SC', sans-serif;
}

.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-icon {
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 喜欢按钮 */
.like-btn {
  background: rgba(244, 63, 94, 0.1);
  border: 1px solid rgba(244, 63, 94, 0.25);
  color: #fb7185;
}

.like-btn:hover:not(:disabled) {
  background: rgba(244, 63, 94, 0.2);
  border-color: rgba(244, 63, 94, 0.4);
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(244, 63, 94, 0.2);
}

.like-btn.liked {
  background: linear-gradient(135deg, #f43f5e, #ec4899);
  border-color: transparent;
  color: #fff;
}

.like-btn.liked:hover:not(:disabled) {
  box-shadow: 0 8px 25px rgba(244, 63, 94, 0.4);
}

/* 分享按钮 */
.share-btn {
  background: rgba(99, 102, 241, 0.1);
  border: 1px solid rgba(99, 102, 241, 0.25);
  color: #a5b4fc;
}

.share-btn:hover:not(:disabled) {
  background: rgba(99, 102, 241, 0.2);
  border-color: rgba(99, 102, 241, 0.4);
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(99, 102, 241, 0.2);
}

/* 响应式 */
@media (max-width: 480px) {
  .action-buttons {
    width: 100%;
  }
  
  .action-btn {
    flex: 1;
    justify-content: center;
  }
}
</style>

