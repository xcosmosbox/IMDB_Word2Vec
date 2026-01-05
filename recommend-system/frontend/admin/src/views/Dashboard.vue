<script setup lang="ts">
/**
 * 仪表盘主页
 * 
 * 展示系统核心统计指标和趋势图表
 */
import { ref, computed, onMounted, inject } from 'vue'
import { Row, Col, Card, Spin, message } from 'ant-design-vue'
import {
  UserOutlined,
  ShoppingOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined,
  PercentageOutlined,
  TeamOutlined,
} from '@ant-design/icons-vue'
import StatCard from '@/components/StatCard.vue'
import LineChart from '@/components/charts/LineChart.vue'
import PieChart from '@/components/charts/PieChart.vue'
import BarChart from '@/components/charts/BarChart.vue'
import type { IApiProvider } from '@shared/api/interfaces'
import type { DashboardStats, TimeSeriesPoint, CategoryStats } from '@shared/types'

// 通过依赖注入获取 API Provider
const api = inject<IApiProvider>('api')

// 状态
const loading = ref(true)
const stats = ref<DashboardStats | null>(null)
const userTrend = ref<TimeSeriesPoint[]>([])
const itemTypeStats = ref<CategoryStats[]>([])
const recTrend = ref<TimeSeriesPoint[]>([])
const topCategories = ref<CategoryStats[]>([])

// 统计卡片配置
const statCards = computed(() => [
  {
    title: '总用户数',
    value: stats.value?.total_users || 0,
    icon: UserOutlined,
    color: '#1890ff',
    trend: '+12.5%',
    trendUp: true,
  },
  {
    title: '总物品数',
    value: stats.value?.total_items || 0,
    icon: ShoppingOutlined,
    color: '#52c41a',
    trend: '+8.3%',
    trendUp: true,
  },
  {
    title: '今日推荐数',
    value: stats.value?.total_recommendations || 0,
    icon: ThunderboltOutlined,
    color: '#722ed1',
    trend: '+25.6%',
    trendUp: true,
  },
  {
    title: '日活跃用户',
    value: stats.value?.daily_active_users || 0,
    icon: TeamOutlined,
    color: '#13c2c2',
    trend: '+5.2%',
    trendUp: true,
  },
  {
    title: '推荐点击率',
    value: stats.value ? `${(stats.value.recommendation_ctr * 100).toFixed(2)}%` : '0%',
    icon: PercentageOutlined,
    color: '#fa8c16',
    trend: '+1.8%',
    trendUp: true,
  },
  {
    title: '平均响应时间',
    value: stats.value ? `${stats.value.avg_response_time}ms` : '0ms',
    icon: ClockCircleOutlined,
    color: '#eb2f96',
    trend: '-5.3%',
    trendUp: false,  // 响应时间下降是好事
  },
])

// 加载仪表盘数据
async function fetchDashboardData() {
  if (!api?.analytics) {
    message.error('API 服务未初始化')
    loading.value = false
    return
  }
  
  loading.value = true
  
  try {
    const [statsData, userTrendData, itemTypeData, recTrendData, categoryData] = 
      await Promise.all([
        api.analytics.getDashboardStats(),
        api.analytics.getUserTrend(30),
        api.analytics.getItemTypeStats(),
        api.analytics.getRecommendationTrend(30),
        api.analytics.getTopCategories(10),
      ])
    
    stats.value = statsData
    userTrend.value = userTrendData
    itemTypeStats.value = itemTypeData
    recTrend.value = recTrendData
    topCategories.value = categoryData
    
  } catch (error) {
    console.error('加载仪表盘数据失败:', error)
    message.error('加载数据失败，请刷新重试')
  } finally {
    loading.value = false
  }
}

// 刷新数据
async function handleRefresh() {
  await fetchDashboardData()
  message.success('数据已刷新')
}

onMounted(() => {
  fetchDashboardData()
})
</script>

<template>
  <div class="dashboard-page">
    <Spin :spinning="loading" size="large" tip="加载中...">
      <!-- 页面标题 -->
      <div class="page-header">
        <h1 class="page-title">数据概览</h1>
        <a-button type="link" @click="handleRefresh">
          刷新数据
        </a-button>
      </div>
      
      <!-- 统计卡片 -->
      <Row :gutter="[16, 16]" class="stat-row">
        <Col 
          v-for="card in statCards" 
          :key="card.title"
          :xs="24" 
          :sm="12" 
          :md="8" 
          :lg="4"
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

      <!-- 趋势图表 -->
      <Row :gutter="[16, 16]" class="chart-row">
        <Col :xs="24" :lg="12">
          <Card title="用户增长趋势（近30天）" :bordered="false" class="chart-card">
            <LineChart
              :data="userTrend"
              x-field="timestamp"
              y-field="value"
              :height="320"
              color="#1890ff"
              :show-data-zoom="true"
            />
          </Card>
        </Col>
        <Col :xs="24" :lg="12">
          <Card title="推荐量趋势（近30天）" :bordered="false" class="chart-card">
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

      <!-- 分布图表 -->
      <Row :gutter="[16, 16]" class="chart-row">
        <Col :xs="24" :lg="12">
          <Card title="物品类型分布" :bordered="false" class="chart-card">
            <PieChart
              :data="itemTypeStats"
              name-field="category"
              value-field="count"
              :height="320"
              :donut="true"
            />
          </Card>
        </Col>
        <Col :xs="24" :lg="12">
          <Card title="热门分类 Top 10" :bordered="false" class="chart-card">
            <BarChart
              :data="topCategories"
              x-field="category"
              y-field="count"
              :height="320"
              color="#52c41a"
              :horizontal="true"
              :bar-width="14"
            />
          </Card>
        </Col>
      </Row>
    </Spin>
  </div>
</template>

<style scoped>
.dashboard-page {
  padding: 0;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-title {
  font-size: 24px;
  font-weight: 600;
  color: #262626;
  margin: 0;
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
</style>

