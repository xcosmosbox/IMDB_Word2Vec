/**
 * PreferenceChart - 偏好图表组件
 * 
 * 展示用户偏好数据的可视化图表组件。
 * 使用 CSS 实现的条形图，不依赖外部图表库。
 * 
 * @component
 * @author Person C
 */
<script setup lang="ts">
import { computed } from 'vue'

// =============================================================================
// Props 定义
// =============================================================================

interface Props {
  /** 偏好数据 { 类型: 数量 } */
  data: Record<string, number>
  /** 图表标题 */
  title?: string
  /** 最大显示数量 */
  maxItems?: number
  /** 是否显示百分比 */
  showPercentage?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  title: '内容偏好',
  maxItems: 6,
  showPercentage: true,
})

// =============================================================================
// 计算属性
// =============================================================================

/** 类型名称映射 */
const typeNameMap: Record<string, string> = {
  movie: '电影',
  product: '商品',
  article: '文章',
  video: '视频',
  music: '音乐',
  book: '图书',
  game: '游戏',
  news: '新闻',
}

/** 类型颜色映射 */
const typeColorMap: Record<string, string> = {
  movie: '#4facfe',
  product: '#00f2fe',
  article: '#a78bfa',
  video: '#f472b6',
  music: '#34d399',
  book: '#fbbf24',
  game: '#f87171',
  news: '#60a5fa',
}

/** 处理后的图表数据 */
const chartData = computed(() => {
  if (!props.data || Object.keys(props.data).length === 0) {
    return []
  }

  const entries = Object.entries(props.data)
  const total = entries.reduce((sum, [, value]) => sum + value, 0)
  const maxValue = Math.max(...entries.map(([, value]) => value))

  return entries
    .sort((a, b) => b[1] - a[1])
    .slice(0, props.maxItems)
    .map(([type, value]) => ({
      type,
      name: typeNameMap[type] || type,
      value,
      percentage: total > 0 ? ((value / total) * 100).toFixed(1) : '0',
      barWidth: maxValue > 0 ? (value / maxValue) * 100 : 0,
      color: typeColorMap[type] || '#4facfe',
    }))
})

/** 是否有数据 */
const hasData = computed(() => chartData.value.length > 0)

/** 总计 */
const totalCount = computed(() => {
  if (!props.data) return 0
  return Object.values(props.data).reduce((sum, value) => sum + value, 0)
})
</script>

<template>
  <div class="preference-chart">
    <!-- 图表头部 -->
    <div class="chart-header">
      <h3 class="chart-title">{{ title }}</h3>
      <span v-if="hasData" class="total-count">共 {{ totalCount }} 次互动</span>
    </div>

    <!-- 图表内容 -->
    <div v-if="hasData" class="chart-content">
      <div
        v-for="item in chartData"
        :key="item.type"
        class="chart-item"
      >
        <div class="item-header">
          <span class="item-name">{{ item.name }}</span>
          <span class="item-stats">
            <span class="item-value">{{ item.value }}</span>
            <span v-if="showPercentage" class="item-percentage">
              ({{ item.percentage }}%)
            </span>
          </span>
        </div>
        <div class="bar-container">
          <div
            class="bar-fill"
            :style="{
              width: `${item.barWidth}%`,
              background: `linear-gradient(90deg, ${item.color} 0%, ${item.color}80 100%)`,
            }"
          ></div>
        </div>
      </div>
    </div>

    <!-- 空状态 -->
    <div v-else class="empty-state">
      <span class="empty-icon">📊</span>
      <p class="empty-text">暂无偏好数据</p>
      <p class="empty-hint">浏览更多内容后，这里会显示你的偏好分析</p>
    </div>
  </div>
</template>

<style scoped>
.preference-chart {
  background: rgba(255, 255, 255, 0.03);
  border-radius: 0.75rem;
  padding: 1.25rem;
}

/* 图表头部 */
.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.25rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.chart-title {
  font-size: 1rem;
  font-weight: 600;
  color: #fff;
  margin: 0;
}

.total-count {
  font-size: 0.8rem;
  color: #8892b0;
}

/* 图表内容 */
.chart-content {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.chart-item {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.item-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.item-name {
  font-size: 0.9rem;
  color: #ccd6f6;
  font-weight: 500;
}

.item-stats {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.item-value {
  font-size: 0.9rem;
  color: #fff;
  font-weight: 600;
}

.item-percentage {
  font-size: 0.8rem;
  color: #8892b0;
}

/* 进度条 */
.bar-container {
  height: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.6s ease-out;
  position: relative;
}

.bar-fill::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(255, 255, 255, 0.2) 50%,
    transparent 100%
  );
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(100%);
  }
}

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 2rem;
}

.empty-icon {
  font-size: 3rem;
  display: block;
  margin-bottom: 1rem;
  opacity: 0.5;
}

.empty-text {
  font-size: 1rem;
  color: #ccd6f6;
  margin: 0 0 0.5rem;
}

.empty-hint {
  font-size: 0.85rem;
  color: #8892b0;
  margin: 0;
}
</style>

