<script setup lang="ts">
/**
 * 物品分析页面
 * 
 * 展示物品相关的统计分析，包括物品数量、类型分布、分类统计等
 */
import { ref, computed, onMounted, inject } from 'vue'
import { Row, Col, Card, Spin, Space, message, Segmented, Table, Tag } from 'ant-design-vue'
import type { TableColumnsType } from 'ant-design-vue'
import {
  ShoppingOutlined,
  AppstoreOutlined,
  TagsOutlined,
  RiseOutlined,
} from '@ant-design/icons-vue'
import StatCard from '@/components/StatCard.vue'
import LineChart from '@/components/charts/LineChart.vue'
import PieChart from '@/components/charts/PieChart.vue'
import BarChart from '@/components/charts/BarChart.vue'
import DataExport from '@/components/DataExport.vue'
import type { IApiProvider } from '@shared/api/interfaces'
import type { TimeSeriesPoint, CategoryStats } from '@shared/types'
import type { ItemGrowthTrend } from '@/api/analytics'

// API Provider
const api = inject<IApiProvider>('api')

// 状态
const loading = ref(true)
const timeRange = ref<'7d' | '15d' | '30d' | '90d'>('30d')

// 数据
const itemTypeStats = ref<CategoryStats[]>([])
const itemStatusStats = ref<CategoryStats[]>([])
const itemCategoryStats = ref<CategoryStats[]>([])
const itemGrowthTrend = ref<ItemGrowthTrend[]>([])

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

// 类型颜色映射
const typeColors: Record<string, string> = {
  movie: '#1890ff',
  product: '#52c41a',
  article: '#fa8c16',
  video: '#722ed1',
}

// 类型标签
const typeLabels: Record<string, string> = {
  movie: '电影',
  product: '商品',
  article: '文章',
  video: '视频',
  active: '上架',
  inactive: '下架',
}

// 统计卡片数据
const statCards = computed(() => {
  const totalItems = itemTypeStats.value.reduce((sum, item) => sum + item.count, 0)
  const activeItems = itemStatusStats.value.find(s => s.category === 'active')?.count || 0
  const categories = itemCategoryStats.value.length
  
  return [
    {
      title: '物品总数',
      value: totalItems,
      icon: ShoppingOutlined,
      color: '#1890ff',
      trend: '+8.3%',
      trendUp: true,
    },
    {
      title: '上架物品',
      value: activeItems,
      icon: RiseOutlined,
      color: '#52c41a',
      trend: '+5.2%',
      trendUp: true,
    },
    {
      title: '物品类型',
      value: itemTypeStats.value.length,
      icon: AppstoreOutlined,
      color: '#722ed1',
    },
    {
      title: '分类数量',
      value: categories,
      icon: TagsOutlined,
      color: '#fa8c16',
    },
  ]
})

// 增长趋势图数据（转换为多系列）
const growthChartData = computed(() => {
  return itemGrowthTrend.value.map(item => ({
    timestamp: item.date,
    movie: item.movie,
    product: item.product,
    article: item.article,
    video: item.video,
    total: item.movie + item.product + item.article + item.video,
  }))
})

// 分类表格列定义
const categoryColumns: TableColumnsType = [
  {
    title: '排名',
    dataIndex: 'rank',
    width: 80,
    customRender: ({ index }) => index + 1,
  },
  {
    title: '分类名称',
    dataIndex: 'category',
  },
  {
    title: '物品数量',
    dataIndex: 'count',
    sorter: (a: CategoryStats, b: CategoryStats) => a.count - b.count,
    customRender: ({ text }) => text.toLocaleString(),
  },
  {
    title: '占比',
    dataIndex: 'percentage',
    customRender: ({ text }) => `${text.toFixed(1)}%`,
  },
]

// 导出列定义
const exportColumns = [
  { key: 'category', title: '分类' },
  { key: 'count', title: '数量' },
  { key: 'percentage', title: '占比 (%)' },
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
    
    const [typeData, statusData, categoryData, growthData] = await Promise.all([
      analyticsApi.getItemTypeStats(),
      analyticsApi.getItemStatusStats?.() || Promise.resolve([]),
      analyticsApi.getItemCategoryStats?.() || Promise.resolve([]),
      analyticsApi.getItemGrowthTrend?.(getDays.value) || Promise.resolve([]),
    ])
    
    itemTypeStats.value = typeData
    itemStatusStats.value = statusData
    itemCategoryStats.value = categoryData
    itemGrowthTrend.value = growthData
    
  } catch (error) {
    console.error('加载物品分析数据失败:', error)
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

onMounted(() => {
  fetchData()
})
</script>

<template>
  <div class="item-analytics-page">
    <Spin :spinning="loading" size="large" tip="加载中...">
      <!-- 页面标题 -->
      <div class="page-header">
        <div class="header-left">
          <h1 class="page-title">物品分析</h1>
          <p class="page-subtitle">物品库存与分布分析</p>
        </div>
        <div class="header-right">
          <Space>
            <Segmented 
              v-model:value="timeRange" 
              :options="timeRangeOptions"
              @change="handleTimeRangeChange"
            />
            <DataExport
              :data="itemCategoryStats"
              :columns="exportColumns"
              filename="item_analytics"
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

      <!-- 物品增长趋势 -->
      <Row :gutter="[16, 16]" class="chart-row">
        <Col :span="24">
          <Card title="物品增长趋势" :bordered="false" class="chart-card">
            <template #extra>
              <Space>
                <Tag v-for="(color, type) in typeColors" :key="type" :color="color">
                  {{ typeLabels[type] || type }}
                </Tag>
              </Space>
            </template>
            <LineChart
              :data="growthChartData"
              x-field="timestamp"
              y-field="total"
              :height="350"
              color="#1890ff"
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
              :height="300"
              :donut="true"
              :colors="Object.values(typeColors)"
            />
          </Card>
        </Col>
        <Col :xs="24" :lg="12">
          <Card title="物品状态分布" :bordered="false" class="chart-card">
            <PieChart
              :data="itemStatusStats"
              name-field="category"
              value-field="count"
              :height="300"
              :donut="false"
              :colors="['#52c41a', '#ff4d4f']"
            />
          </Card>
        </Col>
      </Row>

      <!-- 分类统计表格 -->
      <Row :gutter="[16, 16]" class="chart-row">
        <Col :span="24">
          <Card title="分类统计排行" :bordered="false" class="chart-card">
            <Table
              :columns="categoryColumns"
              :data-source="itemCategoryStats"
              :pagination="{ pageSize: 10, showSizeChanger: true }"
              :loading="loading"
              row-key="category"
              size="middle"
            />
          </Card>
        </Col>
      </Row>
    </Spin>
  </div>
</template>

<style scoped>
.item-analytics-page {
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
</style>

