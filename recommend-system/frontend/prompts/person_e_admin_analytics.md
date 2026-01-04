# Person E: 管理后台 - 数据分析看板

## 你的角色
你是一名前端工程师，负责实现生成式推荐系统的 **管理后台数据分析看板** 模块，包括仪表盘、图表可视化、统计报表等。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读以下文件：**

1. **数据类型定义：**
```
frontend/shared/types/index.ts
```

2. **服务接口定义（核心）：**
```
frontend/shared/api/interfaces.ts
```

你需要使用的核心接口：

```typescript
// 分析服务接口
interface IAnalyticsService {
  getDashboardStats(): Promise<DashboardStats>
  getUserTrend(days: number): Promise<TimeSeriesPoint[]>
  getItemTypeStats(): Promise<CategoryStats[]>
  getRecommendationTrend(days: number): Promise<TimeSeriesPoint[]>
  getTopCategories(limit: number): Promise<CategoryStats[]>
  getCTRTrend(startDate: string, endDate: string): Promise<TimeSeriesPoint[]>
}
```

相关数据类型：

```typescript
interface DashboardStats {
  total_users: number;
  total_items: number;
  total_recommendations: number;
  daily_active_users: number;
  recommendation_ctr: number;
  avg_response_time: number;
}

interface TimeSeriesPoint { timestamp: string; value: number; }
interface CategoryStats { category: string; count: number; percentage: number; }
```

**⚠️ 不要直接导入具体实现！** 使用依赖注入：

```typescript
// ✅ 正确：通过 inject 获取接口
const api = inject<IApiProvider>('api')!
const stats = await api.analytics.getDashboardStats()
const trend = await api.analytics.getUserTrend(30)

// ❌ 错误：直接导入具体实现
import { analyticsApi } from '@/api/admin/analytics'
```

---

## 技术栈

- **框架**: Vue 3 + Composition API + TypeScript
- **构建**: Vite
- **UI 组件库**: Ant Design Vue 4.x
- **图表库**: ECharts 5.x + vue-echarts
- **日期处理**: dayjs
- **HTTP**: Axios

---

## 你的任务

```
frontend/admin/
├── src/
│   ├── views/
│   │   ├── Dashboard.vue          # 仪表盘主页
│   │   └── analytics/
│   │       ├── UserAnalytics.vue  # 用户分析
│   │       ├── ItemAnalytics.vue  # 物品分析
│   │       └── RecAnalytics.vue   # 推荐分析
│   ├── components/
│   │   ├── charts/
│   │   │   ├── LineChart.vue      # 折线图
│   │   │   ├── BarChart.vue       # 柱状图
│   │   │   ├── PieChart.vue       # 饼图
│   │   │   └── HeatmapChart.vue   # 热力图
│   │   ├── StatCard.vue           # 统计卡片
│   │   ├── DateRangePicker.vue    # 日期选择器
│   │   └── DataExport.vue         # 数据导出
│   └── ...
```

---

## 1. 仪表盘主页 (Dashboard.vue)

```vue
<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { Row, Col, Card, Statistic, Spin } from 'ant-design-vue'
import {
  UserOutlined,
  ShoppingOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined,
  RiseOutlined,
  FallOutlined,
} from '@ant-design/icons-vue'
import StatCard from '@/components/StatCard.vue'
import LineChart from '@/components/charts/LineChart.vue'
import PieChart from '@/components/charts/PieChart.vue'
import BarChart from '@/components/charts/BarChart.vue'
import type { DashboardStats, TimeSeriesPoint, CategoryStats } from '@shared/types'
import { analyticsApi } from '@/api/admin/analytics'

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
    trend: '+12%',
    trendUp: true,
  },
  {
    title: '总物品数',
    value: stats.value?.total_items || 0,
    icon: ShoppingOutlined,
    color: '#52c41a',
    trend: '+8%',
    trendUp: true,
  },
  {
    title: '今日推荐',
    value: stats.value?.total_recommendations || 0,
    icon: ThunderboltOutlined,
    color: '#722ed1',
    trend: '+25%',
    trendUp: true,
  },
  {
    title: '平均响应时间',
    value: `${stats.value?.avg_response_time || 0}ms`,
    icon: ClockCircleOutlined,
    color: '#fa8c16',
    trend: '-5%',
    trendUp: false,
  },
])

// 加载数据
async function fetchDashboardData() {
  loading.value = true
  try {
    const [statsData, userTrendData, itemTypeData, recTrendData, categoryData] = 
      await Promise.all([
        analyticsApi.getDashboardStats(),
        analyticsApi.getUserTrend(30),
        analyticsApi.getItemTypeStats(),
        analyticsApi.getRecommendationTrend(30),
        analyticsApi.getTopCategories(10),
      ])
    
    stats.value = statsData
    userTrend.value = userTrendData
    itemTypeStats.value = itemTypeData
    recTrend.value = recTrendData
    topCategories.value = categoryData
  } catch (error) {
    console.error('Failed to load dashboard data:', error)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchDashboardData()
})
</script>

<template>
  <div class="dashboard-page">
    <Spin :spinning="loading" size="large">
      <!-- 统计卡片 -->
      <Row :gutter="[16, 16]" class="stat-row">
        <Col :xs="24" :sm="12" :lg="6" v-for="card in statCards" :key="card.title">
          <StatCard
            :title="card.title"
            :value="card.value"
            :icon="card.icon"
            :color="card.color"
            :trend="card.trend"
            :trend-up="card.trendUp"
          />
        </Col>
      </Row>

      <!-- 趋势图表 -->
      <Row :gutter="[16, 16]" class="chart-row">
        <Col :xs="24" :lg="12">
          <Card title="用户增长趋势" :bordered="false">
            <LineChart
              :data="userTrend"
              x-field="timestamp"
              y-field="value"
              :height="300"
              color="#1890ff"
            />
          </Card>
        </Col>
        <Col :xs="24" :lg="12">
          <Card title="推荐量趋势" :bordered="false">
            <LineChart
              :data="recTrend"
              x-field="timestamp"
              y-field="value"
              :height="300"
              color="#722ed1"
            />
          </Card>
        </Col>
      </Row>

      <!-- 分布图表 -->
      <Row :gutter="[16, 16]" class="chart-row">
        <Col :xs="24" :lg="12">
          <Card title="物品类型分布" :bordered="false">
            <PieChart
              :data="itemTypeStats"
              name-field="category"
              value-field="count"
              :height="300"
            />
          </Card>
        </Col>
        <Col :xs="24" :lg="12">
          <Card title="热门分类 Top 10" :bordered="false">
            <BarChart
              :data="topCategories"
              x-field="category"
              y-field="count"
              :height="300"
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

.stat-row {
  margin-bottom: 16px;
}

.chart-row {
  margin-bottom: 16px;
}
</style>
```

---

## 2. 统计卡片 (StatCard.vue)

```vue
<script setup lang="ts">
import { computed } from 'vue'
import { Card, Statistic } from 'ant-design-vue'
import { RiseOutlined, FallOutlined } from '@ant-design/icons-vue'

interface Props {
  title: string
  value: number | string
  icon: any
  color: string
  trend?: string
  trendUp?: boolean
  suffix?: string
}

const props = withDefaults(defineProps<Props>(), {
  trendUp: true,
})

const trendColor = computed(() => 
  props.trendUp ? '#52c41a' : '#ff4d4f'
)
</script>

<template>
  <Card class="stat-card" :bordered="false" :body-style="{ padding: '20px 24px' }">
    <div class="stat-content">
      <div class="stat-info">
        <div class="stat-title">{{ title }}</div>
        <div class="stat-value">
          {{ value }}
          <span v-if="suffix" class="stat-suffix">{{ suffix }}</span>
        </div>
        <div v-if="trend" class="stat-trend" :style="{ color: trendColor }">
          <component :is="trendUp ? RiseOutlined : FallOutlined" />
          <span>{{ trend }}</span>
          <span class="trend-label">较昨日</span>
        </div>
      </div>
      <div class="stat-icon" :style="{ backgroundColor: `${color}15`, color }">
        <component :is="icon" />
      </div>
    </div>
  </Card>
</template>

<style scoped>
.stat-card {
  border-radius: 8px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
  transition: all 0.3s;
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
}

.stat-title {
  color: #8c8c8c;
  font-size: 14px;
  margin-bottom: 8px;
}

.stat-value {
  font-size: 28px;
  font-weight: 600;
  color: #262626;
  line-height: 1.2;
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

.trend-label {
  color: #8c8c8c;
  margin-left: 4px;
}

.stat-icon {
  width: 56px;
  height: 56px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
}
</style>
```

---

## 3. 折线图 (LineChart.vue)

```vue
<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart as EChartsLineChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsOption } from 'echarts'

// 注册组件
use([
  EChartsLineChart,
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  CanvasRenderer,
])

interface Props {
  data: Array<Record<string, any>>
  xField: string
  yField: string
  height?: number
  color?: string
  smooth?: boolean
  showArea?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  height: 300,
  color: '#1890ff',
  smooth: true,
  showArea: true,
})

const option = computed<EChartsOption>(() => ({
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderColor: '#e8e8e8',
    textStyle: {
      color: '#262626',
    },
    formatter: (params: any) => {
      const item = params[0]
      const date = new Date(item.axisValue).toLocaleDateString()
      return `
        <div style="padding: 4px 8px;">
          <div style="color: #8c8c8c; margin-bottom: 4px;">${date}</div>
          <div style="font-weight: 600; color: ${props.color};">
            ${item.value.toLocaleString()}
          </div>
        </div>
      `
    },
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    top: '10%',
    containLabel: true,
  },
  xAxis: {
    type: 'category',
    data: props.data.map(item => item[props.xField]),
    boundaryGap: false,
    axisLine: {
      lineStyle: {
        color: '#d9d9d9',
      },
    },
    axisLabel: {
      color: '#8c8c8c',
      formatter: (value: string) => {
        const date = new Date(value)
        return `${date.getMonth() + 1}/${date.getDate()}`
      },
    },
  },
  yAxis: {
    type: 'value',
    axisLine: {
      show: false,
    },
    axisTick: {
      show: false,
    },
    splitLine: {
      lineStyle: {
        color: '#f0f0f0',
        type: 'dashed',
      },
    },
    axisLabel: {
      color: '#8c8c8c',
    },
  },
  series: [
    {
      type: 'line',
      data: props.data.map(item => item[props.yField]),
      smooth: props.smooth,
      symbol: 'circle',
      symbolSize: 6,
      showSymbol: false,
      lineStyle: {
        width: 3,
        color: props.color,
      },
      itemStyle: {
        color: props.color,
      },
      areaStyle: props.showArea ? {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: `${props.color}40` },
            { offset: 1, color: `${props.color}05` },
          ],
        },
      } : undefined,
    },
  ],
}))
</script>

<template>
  <VChart
    :option="option"
    :style="{ height: `${height}px` }"
    autoresize
  />
</template>
```

---

## 4. 饼图 (PieChart.vue)

```vue
<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { PieChart as EChartsPieChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsOption } from 'echarts'

use([
  EChartsPieChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  CanvasRenderer,
])

interface Props {
  data: Array<Record<string, any>>
  nameField: string
  valueField: string
  height?: number
  colors?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  height: 300,
  colors: () => ['#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1', '#13c2c2'],
})

// 类型映射
const typeLabels: Record<string, string> = {
  movie: '电影',
  product: '商品',
  article: '文章',
  video: '视频',
}

const option = computed<EChartsOption>(() => ({
  tooltip: {
    trigger: 'item',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderColor: '#e8e8e8',
    formatter: (params: any) => {
      return `
        <div style="padding: 4px 8px;">
          <div style="font-weight: 600;">${typeLabels[params.name] || params.name}</div>
          <div style="color: #8c8c8c;">
            ${params.value.toLocaleString()} (${params.percent}%)
          </div>
        </div>
      `
    },
  },
  legend: {
    orient: 'vertical',
    right: '5%',
    top: 'center',
    formatter: (name: string) => typeLabels[name] || name,
  },
  color: props.colors,
  series: [
    {
      type: 'pie',
      radius: ['45%', '70%'],
      center: ['40%', '50%'],
      avoidLabelOverlap: false,
      itemStyle: {
        borderRadius: 8,
        borderColor: '#fff',
        borderWidth: 2,
      },
      label: {
        show: false,
      },
      emphasis: {
        label: {
          show: true,
          fontSize: 16,
          fontWeight: 'bold',
        },
      },
      data: props.data.map(item => ({
        name: item[props.nameField],
        value: item[props.valueField],
      })),
    },
  ],
}))
</script>

<template>
  <VChart
    :option="option"
    :style="{ height: `${height}px` }"
    autoresize
  />
</template>
```

---

## 5. 柱状图 (BarChart.vue)

```vue
<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { BarChart as EChartsBarChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsOption } from 'echarts'

use([
  EChartsBarChart,
  TitleComponent,
  TooltipComponent,
  GridComponent,
  CanvasRenderer,
])

interface Props {
  data: Array<Record<string, any>>
  xField: string
  yField: string
  height?: number
  color?: string
  horizontal?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  height: 300,
  color: '#1890ff',
  horizontal: true,
})

const option = computed<EChartsOption>(() => {
  const sortedData = [...props.data].sort((a, b) => b[props.yField] - a[props.yField])
  
  const baseConfig = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderColor: '#e8e8e8',
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '3%',
      containLabel: true,
    },
  }
  
  if (props.horizontal) {
    return {
      ...baseConfig,
      xAxis: {
        type: 'value',
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: {
          lineStyle: { color: '#f0f0f0', type: 'dashed' },
        },
      },
      yAxis: {
        type: 'category',
        data: sortedData.map(item => item[props.xField]),
        axisLine: { lineStyle: { color: '#d9d9d9' } },
        inverse: true,
      },
      series: [{
        type: 'bar',
        data: sortedData.map(item => item[props.yField]),
        barWidth: 16,
        itemStyle: {
          borderRadius: [0, 4, 4, 0],
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 1, y2: 0,
            colorStops: [
              { offset: 0, color: props.color },
              { offset: 1, color: `${props.color}80` },
            ],
          },
        },
      }],
    } as EChartsOption
  }
  
  return {
    ...baseConfig,
    xAxis: {
      type: 'category',
      data: sortedData.map(item => item[props.xField]),
      axisLine: { lineStyle: { color: '#d9d9d9' } },
      axisLabel: {
        rotate: 45,
        color: '#8c8c8c',
      },
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: {
        lineStyle: { color: '#f0f0f0', type: 'dashed' },
      },
    },
    series: [{
      type: 'bar',
      data: sortedData.map(item => item[props.yField]),
      barWidth: 24,
      itemStyle: {
        borderRadius: [4, 4, 0, 0],
        color: props.color,
      },
    }],
  } as EChartsOption
})
</script>

<template>
  <VChart
    :option="option"
    :style="{ height: `${height}px` }"
    autoresize
  />
</template>
```

---

## 6. 推荐分析页 (RecAnalytics.vue)

```vue
<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { Row, Col, Card, DatePicker, Space, Table, Tag } from 'ant-design-vue'
import LineChart from '@/components/charts/LineChart.vue'
import StatCard from '@/components/StatCard.vue'
import { analyticsApi } from '@/api/admin/analytics'
import dayjs, { Dayjs } from 'dayjs'

const dateRange = ref<[Dayjs, Dayjs]>([
  dayjs().subtract(30, 'day'),
  dayjs(),
])

const loading = ref(false)

// 推荐统计
const stats = ref({
  totalRecommendations: 0,
  avgCTR: 0,
  avgResponseTime: 0,
  successRate: 0,
})

// 趋势数据
const ctrTrend = ref<any[]>([])
const latencyTrend = ref<any[]>([])

// 热门推荐
const topRecommendations = ref<any[]>([])

// 表格列
const columns = [
  { title: '排名', dataIndex: 'rank', width: 60 },
  { title: '物品ID', dataIndex: 'item_id', width: 150 },
  { title: '标题', dataIndex: 'title', ellipsis: true },
  { 
    title: '推荐次数', 
    dataIndex: 'count',
    sorter: (a: any, b: any) => a.count - b.count,
  },
  { 
    title: 'CTR', 
    dataIndex: 'ctr',
    customRender: ({ text }: { text: number }) => `${(text * 100).toFixed(2)}%`,
  },
]

// 加载数据
async function fetchData() {
  loading.value = true
  try {
    const [start, end] = dateRange.value
    const startDate = start.format('YYYY-MM-DD')
    const endDate = end.format('YYYY-MM-DD')
    
    const [statsData, ctrData, latencyData, topData] = await Promise.all([
      analyticsApi.getRecommendationStats(startDate, endDate),
      analyticsApi.getCTRTrend(startDate, endDate),
      analyticsApi.getLatencyTrend(startDate, endDate),
      analyticsApi.getTopRecommendedItems(20),
    ])
    
    stats.value = statsData
    ctrTrend.value = ctrData
    latencyTrend.value = latencyData
    topRecommendations.value = topData.map((item: any, index: number) => ({
      ...item,
      rank: index + 1,
    }))
  } finally {
    loading.value = false
  }
}

// 日期变化
function handleDateChange() {
  fetchData()
}

onMounted(() => {
  fetchData()
})
</script>

<template>
  <div class="rec-analytics-page">
    <!-- 日期筛选 -->
    <div class="filter-bar">
      <Space>
        <span>时间范围：</span>
        <DatePicker.RangePicker
          v-model:value="dateRange"
          @change="handleDateChange"
        />
      </Space>
    </div>

    <!-- 统计卡片 -->
    <Row :gutter="[16, 16]" class="stat-row">
      <Col :xs="24" :sm="12" :lg="6">
        <Card :bordered="false">
          <Statistic title="总推荐次数" :value="stats.totalRecommendations" />
        </Card>
      </Col>
      <Col :xs="24" :sm="12" :lg="6">
        <Card :bordered="false">
          <Statistic 
            title="平均点击率" 
            :value="stats.avgCTR * 100" 
            suffix="%" 
            :precision="2" 
          />
        </Card>
      </Col>
      <Col :xs="24" :sm="12" :lg="6">
        <Card :bordered="false">
          <Statistic 
            title="平均响应时间" 
            :value="stats.avgResponseTime" 
            suffix="ms" 
          />
        </Card>
      </Col>
      <Col :xs="24" :sm="12" :lg="6">
        <Card :bordered="false">
          <Statistic 
            title="推荐成功率" 
            :value="stats.successRate * 100" 
            suffix="%" 
            :precision="2" 
          />
        </Card>
      </Col>
    </Row>

    <!-- 趋势图 -->
    <Row :gutter="[16, 16]} class="chart-row">
      <Col :xs="24" :lg="12">
        <Card title="点击率趋势" :bordered="false" :loading="loading">
          <LineChart
            :data="ctrTrend"
            x-field="date"
            y-field="ctr"
            :height="280"
            color="#52c41a"
          />
        </Card>
      </Col>
      <Col :xs="24" :lg="12">
        <Card title="响应时间趋势" :bordered="false" :loading="loading">
          <LineChart
            :data="latencyTrend"
            x-field="date"
            y-field="latency"
            :height="280"
            color="#fa8c16"
          />
        </Card>
      </Col>
    </Row>

    <!-- 热门推荐表 -->
    <Card title="热门推荐 Top 20" :bordered="false" class="table-card">
      <Table
        :columns="columns"
        :data-source="topRecommendations"
        :loading="loading"
        :pagination="false"
        size="small"
        row-key="item_id"
      />
    </Card>
  </div>
</template>

<style scoped>
.rec-analytics-page {
  padding: 0;
}

.filter-bar {
  margin-bottom: 16px;
  padding: 16px;
  background: #fafafa;
  border-radius: 8px;
}

.stat-row {
  margin-bottom: 16px;
}

.chart-row {
  margin-bottom: 16px;
}

.table-card {
  margin-top: 16px;
}
</style>
```

---

## 注意事项

1. 使用 ECharts + vue-echarts 绑定
2. 图表需要响应式调整
3. 支持日期范围筛选
4. 数据加载时显示骨架屏
5. 支持数据导出功能

## 输出要求

请输出完整的可运行代码，包含：
1. Dashboard 主页
2. 所有图表组件
3. 分析页面
4. API 接口封装

