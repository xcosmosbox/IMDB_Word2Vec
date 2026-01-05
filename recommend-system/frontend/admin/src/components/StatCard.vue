<script setup lang="ts">
/**
 * 统计卡片组件
 * 
 * 用于展示关键统计指标，支持图标、趋势显示和自定义样式
 */
import { computed } from 'vue'
import { Card } from 'ant-design-vue'
import { RiseOutlined, FallOutlined } from '@ant-design/icons-vue'

interface Props {
  /** 标题 */
  title: string
  /** 数值 */
  value: number | string
  /** 图标组件 */
  icon?: any
  /** 主题色 */
  color?: string
  /** 趋势文本 (如 +12%, -5%) */
  trend?: string
  /** 趋势是否上升 */
  trendUp?: boolean
  /** 后缀文本 */
  suffix?: string
  /** 前缀文本 */
  prefix?: string
  /** 是否显示加载状态 */
  loading?: boolean
  /** 数值格式化 */
  formatter?: (value: number | string) => string
}

const props = withDefaults(defineProps<Props>(), {
  color: '#1890ff',
  trendUp: true,
  loading: false,
})

const emit = defineEmits<{
  click: []
}>()

// 趋势颜色
const trendColor = computed(() => 
  props.trendUp ? '#52c41a' : '#ff4d4f'
)

// 格式化显示值
const displayValue = computed(() => {
  if (props.formatter) {
    return props.formatter(props.value)
  }
  if (typeof props.value === 'number') {
    return props.value.toLocaleString()
  }
  return props.value
})

// 图标背景色（带透明度）
const iconBgColor = computed(() => `${props.color}15`)

function handleClick() {
  emit('click')
}
</script>

<template>
  <Card 
    class="stat-card" 
    :bordered="false" 
    :body-style="{ padding: '20px 24px' }"
    :loading="loading"
    @click="handleClick"
  >
    <div class="stat-content">
      <div class="stat-info">
        <div class="stat-title">{{ title }}</div>
        <div class="stat-value">
          <span v-if="prefix" class="stat-prefix">{{ prefix }}</span>
          {{ displayValue }}
          <span v-if="suffix" class="stat-suffix">{{ suffix }}</span>
        </div>
        <div v-if="trend" class="stat-trend" :style="{ color: trendColor }">
          <component :is="trendUp ? RiseOutlined : FallOutlined" />
          <span class="trend-value">{{ trend }}</span>
          <span class="trend-label">较昨日</span>
        </div>
      </div>
      <div 
        v-if="icon" 
        class="stat-icon" 
        :style="{ backgroundColor: iconBgColor, color }"
      >
        <component :is="icon" />
      </div>
    </div>
  </Card>
</template>

<style scoped>
.stat-card {
  border-radius: 8px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
  transition: all 0.3s ease;
  cursor: pointer;
}

.stat-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  transform: translateY(-2px);
}

.stat-content {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.stat-info {
  flex: 1;
  min-width: 0;
}

.stat-title {
  color: #8c8c8c;
  font-size: 14px;
  margin-bottom: 8px;
  font-weight: 400;
}

.stat-value {
  font-size: 28px;
  font-weight: 600;
  color: #262626;
  line-height: 1.2;
  word-break: break-all;
}

.stat-prefix {
  font-size: 18px;
  font-weight: normal;
  color: #595959;
  margin-right: 2px;
}

.stat-suffix {
  font-size: 16px;
  font-weight: normal;
  color: #8c8c8c;
  margin-left: 4px;
}

.stat-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-top: 8px;
  font-size: 13px;
}

.trend-value {
  font-weight: 500;
}

.trend-label {
  color: #8c8c8c;
  margin-left: 4px;
  font-weight: normal;
}

.stat-icon {
  width: 56px;
  height: 56px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  flex-shrink: 0;
}

@media (max-width: 576px) {
  .stat-value {
    font-size: 24px;
  }
  
  .stat-icon {
    width: 48px;
    height: 48px;
    font-size: 20px;
  }
}
</style>

