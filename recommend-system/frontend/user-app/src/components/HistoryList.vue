/**
 * HistoryList - 历史记录列表组件
 * 
 * 展示用户行为历史记录的列表组件。
 * 支持按时间分组、操作类型过滤等功能。
 * 
 * @component
 * @author Person C
 */
<script setup lang="ts">
import { computed } from 'vue'
import type { UserBehavior } from '@shared/types'

// =============================================================================
// Props 定义
// =============================================================================

interface Props {
  /** 行为历史数据 */
  behaviors: UserBehavior[]
  /** 是否按日期分组 */
  groupByDate?: boolean
  /** 是否显示时间戳 */
  showTimestamp?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  groupByDate: true,
  showTimestamp: true,
})

// =============================================================================
// Emits 定义
// =============================================================================

const emit = defineEmits<{
  /** 点击物品 */
  'item-click': [itemId: string]
}>()

// =============================================================================
// 常量定义
// =============================================================================

/** 操作类型配置 */
const actionConfig: Record<string, { icon: string; label: string; color: string }> = {
  view: { icon: '👁️', label: '浏览', color: '#60a5fa' },
  click: { icon: '👆', label: '点击', color: '#4facfe' },
  like: { icon: '❤️', label: '喜欢', color: '#f472b6' },
  dislike: { icon: '👎', label: '不喜欢', color: '#9ca3af' },
  buy: { icon: '🛒', label: '购买', color: '#34d399' },
  share: { icon: '🔗', label: '分享', color: '#a78bfa' },
}

// =============================================================================
// 计算属性
// =============================================================================

/**
 * 格式化时间
 */
function formatTime(timestamp: string): string {
  const date = new Date(timestamp)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return '刚刚'
  if (diffMins < 60) return `${diffMins} 分钟前`
  if (diffHours < 24) return `${diffHours} 小时前`
  if (diffDays < 7) return `${diffDays} 天前`

  return date.toLocaleDateString('zh-CN', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/**
 * 获取日期标签
 */
function getDateLabel(timestamp: string): string {
  const date = new Date(timestamp)
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)

  const dateStr = date.toDateString()
  if (dateStr === today.toDateString()) return '今天'
  if (dateStr === yesterday.toDateString()) return '昨天'

  return date.toLocaleDateString('zh-CN', {
    month: 'long',
    day: 'numeric',
    weekday: 'short',
  })
}

/** 按日期分组的行为数据 */
const groupedBehaviors = computed(() => {
  if (!props.groupByDate) {
    return [{ date: '', items: props.behaviors }]
  }

  const groups: Record<string, UserBehavior[]> = {}
  
  props.behaviors.forEach((behavior) => {
    const dateKey = new Date(behavior.timestamp).toDateString()
    if (!groups[dateKey]) {
      groups[dateKey] = []
    }
    groups[dateKey].push(behavior)
  })

  return Object.entries(groups)
    .sort((a, b) => new Date(b[0]).getTime() - new Date(a[0]).getTime())
    .map(([date, items]) => ({
      date: getDateLabel(date),
      items: items.sort(
        (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      ),
    }))
})

/** 是否有数据 */
const hasData = computed(() => props.behaviors.length > 0)

// =============================================================================
// 事件处理
// =============================================================================

/**
 * 处理物品点击
 */
function handleItemClick(itemId: string) {
  emit('item-click', itemId)
}

/**
 * 获取操作配置
 */
function getActionConfig(action: string) {
  return actionConfig[action] || { icon: '📌', label: action, color: '#8892b0' }
}
</script>

<template>
  <div class="history-list">
    <!-- 有数据时 -->
    <template v-if="hasData">
      <div
        v-for="group in groupedBehaviors"
        :key="group.date"
        class="history-group"
      >
        <!-- 日期标题 -->
        <div v-if="groupByDate && group.date" class="group-header">
          <span class="group-date">{{ group.date }}</span>
          <span class="group-count">{{ group.items.length }} 条记录</span>
        </div>

        <!-- 行为列表 -->
        <ul class="behavior-list">
          <li
            v-for="(behavior, index) in group.items"
            :key="`${behavior.item_id}-${index}`"
            class="behavior-item"
            @click="handleItemClick(behavior.item_id)"
          >
            <div class="item-content">
              <!-- 操作图标 -->
              <span
                class="action-icon"
                :style="{ backgroundColor: `${getActionConfig(behavior.action).color}20` }"
              >
                {{ getActionConfig(behavior.action).icon }}
              </span>

              <!-- 主要信息 -->
              <div class="item-main">
                <div class="item-action">
                  <span
                    class="action-label"
                    :style="{ color: getActionConfig(behavior.action).color }"
                  >
                    {{ getActionConfig(behavior.action).label }}
                  </span>
                  <span class="item-id">{{ behavior.item_id }}</span>
                </div>
                <div v-if="showTimestamp" class="item-time">
                  {{ formatTime(behavior.timestamp) }}
                </div>
              </div>

              <!-- 箭头 -->
              <span class="item-arrow">→</span>
            </div>
          </li>
        </ul>
      </div>
    </template>

    <!-- 无数据时 -->
    <div v-else class="empty-state">
      <span class="empty-icon">📜</span>
      <p class="empty-text">暂无历史记录</p>
      <p class="empty-hint">开始探索内容，你的足迹会在这里显示</p>
    </div>
  </div>
</template>

<style scoped>
.history-list {
  width: 100%;
}

/* 分组 */
.history-group {
  margin-bottom: 1.5rem;
}

.history-group:last-child {
  margin-bottom: 0;
}

/* 分组头部 */
.group-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.75rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.group-date {
  font-size: 0.9rem;
  font-weight: 600;
  color: #ccd6f6;
}

.group-count {
  font-size: 0.8rem;
  color: #8892b0;
}

/* 行为列表 */
.behavior-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

/* 行为项 */
.behavior-item {
  cursor: pointer;
  transition: all 0.2s ease;
}

.behavior-item:hover {
  transform: translateX(4px);
}

.item-content {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 0.75rem;
  border: 1px solid rgba(255, 255, 255, 0.05);
  transition: all 0.2s ease;
}

.behavior-item:hover .item-content {
  background: rgba(255, 255, 255, 0.06);
  border-color: rgba(255, 255, 255, 0.1);
}

/* 操作图标 */
.action-icon {
  width: 40px;
  height: 40px;
  border-radius: 0.75rem;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.25rem;
  flex-shrink: 0;
}

/* 主要信息 */
.item-main {
  flex: 1;
  min-width: 0;
}

.item-action {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.25rem;
}

.action-label {
  font-size: 0.85rem;
  font-weight: 600;
}

.item-id {
  font-size: 0.9rem;
  color: #ccd6f6;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.item-time {
  font-size: 0.8rem;
  color: #8892b0;
}

/* 箭头 */
.item-arrow {
  color: #8892b0;
  font-size: 1rem;
  opacity: 0;
  transition: all 0.2s ease;
}

.behavior-item:hover .item-arrow {
  opacity: 1;
  color: #4facfe;
}

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 3rem 2rem;
}

.empty-icon {
  font-size: 4rem;
  display: block;
  margin-bottom: 1rem;
  opacity: 0.5;
}

.empty-text {
  font-size: 1.125rem;
  color: #ccd6f6;
  margin: 0 0 0.5rem;
}

.empty-hint {
  font-size: 0.9rem;
  color: #8892b0;
  margin: 0;
}

/* 响应式 */
@media (max-width: 480px) {
  .item-content {
    padding: 0.875rem;
    gap: 0.75rem;
  }

  .action-icon {
    width: 36px;
    height: 36px;
    font-size: 1rem;
  }
}
</style>

