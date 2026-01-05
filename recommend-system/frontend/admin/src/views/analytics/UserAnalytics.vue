<script setup lang="ts">
/**
 * 用户分析页面
 * 
 * 展示用户相关的统计分析，包括用户增长、活跃度、人口统计等
 */
import { ref, computed, onMounted, inject } from 'vue'
import { Row, Col, Card, Spin, Space, message, Segmented } from 'ant-design-vue'
import {
  UserOutlined,
  TeamOutlined,
  RiseOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons-vue'
import StatCard from '@/components/StatCard.vue'
import LineChart from '@/components/charts/LineChart.vue'
import PieChart from '@/components/charts/PieChart.vue'
import BarChart from '@/components/charts/BarChart.vue'
import HeatmapChart from '@/components/charts/HeatmapChart.vue'
import DateRangePicker from '@/components/DateRangePicker.vue'
import DataExport from '@/components/DataExport.vue'
import type { IApiProvider } from '@shared/api/interfaces'
import type { TimeSeriesPoint, CategoryStats } from '@shared/types'
import type { UserActivityDistribution } from '@/api/analytics'
import dayjs from 'dayjs'

// API Provider
const api = inject<IApiProvider>('api')

// 状态
const loading = ref(true)
const timeRange = ref<'7d' | '15d' | '30d' | '90d'>('30d')

// 数据
const userTrend = ref<TimeSeriesPoint[]>([])
const genderStats = ref<CategoryStats[]>([])
const ageDistribution = ref<CategoryStats[]>([])
const activityDistribution = ref<UserActivityDistribution[]>([])

// 时间范围选项
const timeRangeOptions = [
  { label: '近7天', value: '7d' },
  { label: '近15天', value: '15d' },
  { label: '近30天', value: '30d' },
  { label: '近90天', value: '90d' },
]

// 获取天数
const getDays = computed(() => {
  const map: Record<string, number> = {
    '7d': 7,
    '15d': 15,
    '30d': 30,
    '90d': 90,
  }
  return map[timeRange.value] || 30
})

// 统计卡片数据
const statCards = computed(() => {
  const total = userTrend.value.reduce((sum, item) => sum + item.value, 0)
  const avg = userTrend.value.length > 0 ? Math.round(total / userTrend.value.length) : 0
  const latest = userTrend.value.length > 0 ? userTrend.value[userTrend.value.length - 1].value : 0
  const previous = userTrend.value.length > 1 ? userTrend.value[userTrend.value.length - 2].value : 0
  const growth = previous > 0 ? ((latest - previous) / previous * 100).toFixed(1) : '0'
  
  return [
    {
      title: '期间新增用户',
      value: total,
      icon: UserOutlined,
      color: '#1890ff',
      trend: `${growth}%`,
      trendUp: Number(growth) >= 0,
    },
    {
      title: '日均新增',
      value: avg,
      icon: TeamOutlined,
      color: '#52c41a',
    },
    {
      title: '今日新增',
      value: latest,
      icon: RiseOutlined,
      color: '#722ed1',
      trend: `${growth}%`,
      trendUp: Number(growth) >= 0,
    },
  ]
})

// 活跃时段热力图数据
const activityHeatmapData = computed(() => {
  // 生成一周每天每小时的热力图数据
  const weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
  const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`)
  
  const data: Array<{ x: number; y: number; value: number }> = []
  
  activityDistribution.value.forEach((item, hourIndex) => {
    // 为每天生成数据（模拟）
    weekdays.forEach((_, dayIndex) => {
      const weekendFactor = dayIndex >= 5 ? 1.3 : 1
      const randomFactor = 0.8 + Math.random() * 0.4
      data.push({
        x: hourIndex,
        y: dayIndex,
        value: Math.round(item.count * weekendFactor * randomFactor),
      })
    })
  })
  
  return {
    data,
    xLabels: hours,
    yLabels: weekdays,
  }
})

// 导出列定义
const exportColumns = [
  { key: 'timestamp', title: '日期' },
  { key: 'value', title: '新增用户数' },
]

// 加载数据
async function fetchData() {
  if (!api?.analytics) {
    message.error('API 服务未初始化')
    loading.value = false
    return
  }
  
  loading.value = true
  
  try {
    const analyticsApi = api.analytics as any
    
    const [trendData, genderData, ageData, activityData] = await Promise.all([
      analyticsApi.getUserTrend(getDays.value),
      analyticsApi.getUserGenderStats?.() || Promise.resolve([]),
      analyticsApi.getUserAgeDistribution?.() || Promise.resolve([]),
      analyticsApi.getUserActivityDistribution?.() || Promise.resolve([]),
    ])
    
    userTrend.value = trendData
    genderStats.value = genderData
    ageDistribution.value = ageData
    activityDistribution.value = activityData
    
  } catch (error) {
    console.error('加载用户分析数据失败:', error)
    message.error('加载数据失败，请重试')
  } finally {
    loading.value = false
  }
}

// 时间范围变化
function handleTimeRangeChange(value: string | number) {
  timeRange.value = value as '7d' | '15d' | '30d' | '90d'
  fetchData()
}

// 导出处理
function handleExport(format: string) {
  console.log('导出格式:', format)
}

onMounted(() => {
  fetchData()
})
</script>

<template>
  <div class="user-analytics-page">
    <Spin :spinning="loading" size="large" tip="加载中...">
      <!-- 页面标题 -->
      <div class="page-header">
        <div class="header-left">
          <h1 class="page-title">用户分析</h1>
          <p class="page-subtitle">用户增长与行为分析</p>
        </div>
        <div class="header-right">
          <Space>
            <Segmented 
              v-model:value="timeRange" 
              :options="timeRangeOptions"
              @change="handleTimeRangeChange"
            />
            <DataExport
              :data="userTrend"
              :columns="exportColumns"
              filename="user_analytics"
              @export="handleExport"
            />
          </Space>
        </div>
      </div>
      
      <!-- 统计卡片 -->
      <Row :gutter="[16, 16]" class="stat-row">
        <Col 
          v-for="card in statCards" 
          :key="card.title"
          :xs="24" 
          :sm="12" 
          :lg="8"
        >
          <StatCard
            :title="card.title"
            :value="card.value"
            :icon="card.icon"
            :color="card.color"
            :trend="card.trend"
            :trend-up="card.trendUp"
            :loading="loading"
          />
        </Col>
      </Row>

      <!-- 用户增长趋势 -->
      <Row :gutter="[16, 16]" class="chart-row">
        <Col :span="24">
          <Card title="用户增长趋势" :bordered="false" class="chart-card">
            <LineChart
              :data="userTrend"
              x-field="timestamp"
              y-field="value"
              :height="350"
              color="#1890ff"
              :show-data-zoom="true"
            />
          </Card>
        </Col>
      </Row>

      <!-- 用户分布图表 -->
      <Row :gutter="[16, 16]" class="chart-row">
        <Col :xs="24" :lg="12">
          <Card title="性别分布" :bordered="false" class="chart-card">
            <PieChart
              :data="genderStats"
              name-field="category"
              value-field="count"
              :height="300"
              :donut="true"
            />
          </Card>
        </Col>
        <Col :xs="24" :lg="12">
          <Card title="年龄分布" :bordered="false" class="chart-card">
            <BarChart
              :data="ageDistribution"
              x-field="category"
              y-field="count"
              :height="300"
              color="#722ed1"
              :horizontal="false"
            />
          </Card>
        </Col>
      </Row>

      <!-- 活跃时段分析 -->
      <Row :gutter="[16, 16]" class="chart-row">
        <Col :span="24">
          <Card title="用户活跃时段分布" :bordered="false" class="chart-card">
            <template #extra>
              <span class="chart-tip">
                <ClockCircleOutlined />
                <span>颜色越深表示用户越活跃</span>
              </span>
            </template>
            <HeatmapChart
              :data="activityHeatmapData.data"
              :x-labels="activityHeatmapData.xLabels"
              :y-labels="activityHeatmapData.yLabels"
              :height="280"
              min-color="#e6f7ff"
              max-color="#1890ff"
              :show-label="false"
            />
          </Card>
        </Col>
      </Row>
    </Spin>
  </div>
</template>

<style scoped>
.user-analytics-page {
  padding: 0;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 24px;
  flex-wrap: wrap;
  gap: 16px;
}

.header-left {
  flex: 1;
}

.page-title {
  font-size: 24px;
  font-weight: 600;
  color: #262626;
  margin: 0 0 4px 0;
}

.page-subtitle {
  font-size: 14px;
  color: #8c8c8c;
  margin: 0;
}

.header-right {
  display: flex;
  align-items: center;
}

.stat-row {
  margin-bottom: 24px;
}

.chart-row {
  margin-bottom: 24px;
}

.chart-card {
  border-radius: 8px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

.chart-card :deep(.ant-card-head) {
  border-bottom: 1px solid #f0f0f0;
}

.chart-card :deep(.ant-card-head-title) {
  font-size: 16px;
  font-weight: 500;
  color: #262626;
}

.chart-card :deep(.ant-card-body) {
  padding: 16px 24px 24px;
}

.chart-tip {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #8c8c8c;
  font-size: 13px;
}
</style>

