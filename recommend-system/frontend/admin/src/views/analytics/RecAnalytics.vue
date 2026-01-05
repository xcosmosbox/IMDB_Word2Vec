<script setup lang="ts">
/**
 * 推荐分析页面
 * 
 * 展示推荐系统相关的统计分析，包括推荐量、CTR、响应时间等
 */
import { ref, reactive, computed, onMounted, inject } from 'vue'
import { Row, Col, Card, Spin, Space, Table, Tag, message } from 'ant-design-vue'
import type { TableColumnsType } from 'ant-design-vue'
import {
  ThunderboltOutlined,
  PercentageOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  TrophyOutlined,
} from '@ant-design/icons-vue'
import StatCard from '@/components/StatCard.vue'
import LineChart from '@/components/charts/LineChart.vue'
import DateRangePicker from '@/components/DateRangePicker.vue'
import DataExport from '@/components/DataExport.vue'
import type { IApiProvider } from '@shared/api/interfaces'
import type { TimeSeriesPoint } from '@shared/types'
import type { RecommendationStats, TopRecommendedItem } from '@/api/analytics'
import dayjs from 'dayjs'

// API Provider
const api = inject<IApiProvider>('api')

// 日期范围
const dateRange = reactive({
  startDate: dayjs().subtract(30, 'day').format('YYYY-MM-DD'),
  endDate: dayjs().format('YYYY-MM-DD'),
})

// 状态
const loading = ref(true)

// 数据
const stats = ref<RecommendationStats>({
  totalRecommendations: 0,
  avgCTR: 0,
  avgResponseTime: 0,
  successRate: 0,
})
const ctrTrend = ref<TimeSeriesPoint[]>([])
const latencyTrend = ref<TimeSeriesPoint[]>([])
const topRecommendations = ref<(TopRecommendedItem & { rank: number })[]>([])
const recTrend = ref<TimeSeriesPoint[]>([])

// 统计卡片配置
const statCards = computed(() => [
  {
    title: '总推荐次数',
    value: stats.value.totalRecommendations,
    icon: ThunderboltOutlined,
    color: '#722ed1',
    formatter: (v: number) => v.toLocaleString(),
  },
  {
    title: '平均点击率',
    value: `${(stats.value.avgCTR * 100).toFixed(2)}%`,
    icon: PercentageOutlined,
    color: '#52c41a',
    trend: '+1.2%',
    trendUp: true,
  },
  {
    title: '平均响应时间',
    value: `${stats.value.avgResponseTime.toFixed(1)}ms`,
    icon: ClockCircleOutlined,
    color: '#fa8c16',
    trend: '-3.5%',
    trendUp: false,
  },
  {
    title: '推荐成功率',
    value: `${(stats.value.successRate * 100).toFixed(2)}%`,
    icon: CheckCircleOutlined,
    color: '#1890ff',
  },
])

// 热门推荐表格列
const columns: TableColumnsType = [
  {
    title: '排名',
    dataIndex: 'rank',
    width: 80,
    customRender: ({ text }) => {
      const colors: Record<number, string> = {
        1: '#ffd700',
        2: '#c0c0c0',
        3: '#cd7f32',
      }
      return text <= 3 
        ? h(TrophyOutlined, { style: { color: colors[text], fontSize: '18px' } })
        : text
    },
  },
  {
    title: '物品ID',
    dataIndex: 'item_id',
    width: 120,
  },
  {
    title: '标题',
    dataIndex: 'title',
    ellipsis: true,
  },
  {
    title: '推荐次数',
    dataIndex: 'count',
    sorter: (a: TopRecommendedItem, b: TopRecommendedItem) => a.count - b.count,
    customRender: ({ text }) => text.toLocaleString(),
  },
  {
    title: 'CTR',
    dataIndex: 'ctr',
    sorter: (a: TopRecommendedItem, b: TopRecommendedItem) => a.ctr - b.ctr,
    customRender: ({ text }) => {
      const percentage = (text * 100).toFixed(2)
      const color = text >= 0.1 ? 'green' : text >= 0.05 ? 'blue' : 'default'
      return h(Tag, { color }, () => `${percentage}%`)
    },
  },
]

// 导出列定义
const exportColumns = [
  { key: 'rank', title: '排名' },
  { key: 'item_id', title: '物品ID' },
  { key: 'title', title: '标题' },
  { key: 'count', title: '推荐次数' },
  { 
    key: 'ctr', 
    title: 'CTR (%)',
    formatter: (v: number) => (v * 100).toFixed(2),
  },
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
    const { startDate, endDate } = dateRange
    
    const [statsData, ctrData, latencyData, topData, trendData] = await Promise.all([
      analyticsApi.getRecommendationStats?.(startDate, endDate) || Promise.resolve({
        totalRecommendations: 0,
        avgCTR: 0,
        avgResponseTime: 0,
        successRate: 0,
      }),
      analyticsApi.getCTRTrend(startDate, endDate),
      analyticsApi.getLatencyTrend?.(startDate, endDate) || Promise.resolve([]),
      analyticsApi.getTopRecommendedItems?.(20) || Promise.resolve([]),
      analyticsApi.getRecommendationTrend(30),
    ])
    
    stats.value = statsData
    ctrTrend.value = ctrData
    latencyTrend.value = latencyData
    topRecommendations.value = topData.map((item: TopRecommendedItem, index: number) => ({
      ...item,
      rank: index + 1,
    }))
    recTrend.value = trendData
    
  } catch (error) {
    console.error('加载推荐分析数据失败:', error)
    message.error('加载数据失败，请重试')
  } finally {
    loading.value = false
  }
}

// 日期范围变化
function handleDateChange(start: string, end: string) {
  dateRange.startDate = start
  dateRange.endDate = end
  if (start && end) {
    fetchData()
  }
}

// 渲染函数
import { h } from 'vue'

onMounted(() => {
  fetchData()
})
</script>

<template>
  <div class="rec-analytics-page">
    <Spin :spinning="loading" size="large" tip="加载中...">
      <!-- 页面标题 -->
      <div class="page-header">
        <div class="header-left">
          <h1 class="page-title">推荐分析</h1>
          <p class="page-subtitle">推荐效果与性能分析</p>
        </div>
        <div class="header-right">
          <Space>
            <DateRangePicker
              :start-date="dateRange.startDate"
              :end-date="dateRange.endDate"
              :show-presets="true"
              @change="handleDateChange"
            />
            <DataExport
              :data="topRecommendations"
              :columns="exportColumns"
              filename="recommendation_analytics"
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
          :lg="6"
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

      <!-- 推荐量趋势 -->
      <Row :gutter="[16, 16]" class="chart-row">
        <Col :span="24">
          <Card title="推荐量趋势" :bordered="false" class="chart-card">
            <LineChart
              :data="recTrend"
              x-field="timestamp"
              y-field="value"
              :height="320"
              color="#722ed1"
              :show-data-zoom="true"
            />
          </Card>
        </Col>
      </Row>

      <!-- CTR 和响应时间趋势 -->
      <Row :gutter="[16, 16]" class="chart-row">
        <Col :xs="24" :lg="12">
          <Card title="点击率趋势" :bordered="false" class="chart-card">
            <LineChart
              :data="ctrTrend"
              x-field="timestamp"
              y-field="value"
              :height="280"
              color="#52c41a"
              :show-area="true"
            />
          </Card>
        </Col>
        <Col :xs="24" :lg="12">
          <Card title="响应时间趋势" :bordered="false" class="chart-card">
            <LineChart
              :data="latencyTrend"
              x-field="timestamp"
              y-field="value"
              :height="280"
              color="#fa8c16"
              :show-area="true"
            />
          </Card>
        </Col>
      </Row>

      <!-- 热门推荐表 -->
      <Row :gutter="[16, 16]" class="chart-row">
        <Col :span="24">
          <Card title="热门推荐 Top 20" :bordered="false" class="chart-card">
            <Table
              :columns="columns"
              :data-source="topRecommendations"
              :loading="loading"
              :pagination="false"
              size="middle"
              row-key="item_id"
              :scroll="{ y: 400 }"
            />
          </Card>
        </Col>
      </Row>
    </Spin>
  </div>
</template>

<style scoped>
.rec-analytics-page {
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

.chart-card :deep(.ant-table) {
  font-size: 14px;
}

.chart-card :deep(.ant-table-thead > tr > th) {
  background: #fafafa;
  font-weight: 500;
}
</style>

