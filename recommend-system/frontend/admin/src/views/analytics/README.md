# 数据分析看板模块开发文档

> Person E 开发模块 - 管理后台数据分析看板

## 📋 模块概述

本模块是生成式推荐系统管理后台的数据分析看板，提供系统核心指标的可视化展示和分析功能。

### 功能特性

- **仪表盘概览**: 展示系统关键指标（用户数、物品数、推荐量、CTR等）
- **用户分析**: 用户增长趋势、人口统计分布、活跃时段分析
- **物品分析**: 物品类型分布、状态统计、分类排行
- **推荐分析**: 推荐效果追踪、CTR趋势、响应时间监控

## 🛠️ 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Vue 3 | ^3.4 | 核心框架 |
| TypeScript | ^5.3 | 类型安全 |
| Ant Design Vue | ^4.0 | UI 组件库 |
| ECharts | ^5.4 | 图表可视化 |
| vue-echarts | ^6.6 | ECharts Vue 绑定 |
| dayjs | ^1.11 | 日期处理 |
| Vitest | ^1.0 | 单元测试 |

## 📁 目录结构

```
admin/src/
├── api/
│   ├── analytics.ts           # Analytics API 接口封装
│   ├── mock/
│   │   └── analytics.ts       # Mock 数据实现
│   └── __tests__/
│       └── analytics.spec.ts  # API 单元测试
├── components/
│   ├── StatCard.vue           # 统计卡片组件
│   ├── DateRangePicker.vue    # 日期范围选择器
│   ├── DataExport.vue         # 数据导出组件
│   ├── charts/
│   │   ├── LineChart.vue      # 折线图（已有）
│   │   ├── BarChart.vue       # 柱状图（已有）
│   │   ├── PieChart.vue       # 饼图（已有）
│   │   └── HeatmapChart.vue   # 热力图（新增）
│   └── __tests__/
│       ├── StatCard.spec.ts
│       ├── DateRangePicker.spec.ts
│       └── DataExport.spec.ts
└── views/
    ├── Dashboard.vue          # 仪表盘主页
    ├── analytics/
    │   ├── UserAnalytics.vue  # 用户分析页
    │   ├── ItemAnalytics.vue  # 物品分析页
    │   ├── RecAnalytics.vue   # 推荐分析页
    │   └── README.md          # 本文档
    └── __tests__/
        └── Dashboard.spec.ts
```

## 🚀 快速开始

### 1. 依赖注入配置

在应用入口文件中配置 API Provider：

```typescript
// main.ts
import { createApp } from 'vue'
import App from './App.vue'
import { createApiProvider } from './api/provider'

const app = createApp(App)

// 根据环境选择使用 Mock 或真实 API
const useMock = import.meta.env.DEV
const apiProvider = createApiProvider(useMock)

// 注入 API Provider
app.provide('api', apiProvider)

app.mount('#app')
```

### 2. 路由配置

添加分析页面路由：

```typescript
// router/index.ts
const routes = [
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: () => import('@/views/Dashboard.vue'),
    meta: { title: '数据概览' },
  },
  {
    path: '/analytics',
    name: 'Analytics',
    redirect: '/analytics/users',
    children: [
      {
        path: 'users',
        name: 'UserAnalytics',
        component: () => import('@/views/analytics/UserAnalytics.vue'),
        meta: { title: '用户分析' },
      },
      {
        path: 'items',
        name: 'ItemAnalytics',
        component: () => import('@/views/analytics/ItemAnalytics.vue'),
        meta: { title: '物品分析' },
      },
      {
        path: 'recommendations',
        name: 'RecAnalytics',
        component: () => import('@/views/analytics/RecAnalytics.vue'),
        meta: { title: '推荐分析' },
      },
    ],
  },
]
```

## 📊 组件使用指南

### StatCard 统计卡片

```vue
<template>
  <StatCard
    title="总用户数"
    :value="12345"
    :icon="UserOutlined"
    color="#1890ff"
    trend="+12.5%"
    :trend-up="true"
    suffix="人"
  />
</template>
```

**Props:**

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| title | string | - | 标题（必填） |
| value | number \| string | - | 数值（必填） |
| icon | Component | - | 图标组件 |
| color | string | '#1890ff' | 主题色 |
| trend | string | - | 趋势文本 |
| trendUp | boolean | true | 趋势是否上升 |
| suffix | string | - | 后缀 |
| prefix | string | - | 前缀 |
| loading | boolean | false | 加载状态 |
| formatter | Function | - | 自定义格式化函数 |

### DateRangePicker 日期选择器

```vue
<template>
  <DateRangePicker
    :start-date="startDate"
    :end-date="endDate"
    :show-presets="true"
    @change="handleDateChange"
  />
</template>

<script setup>
function handleDateChange(start: string, end: string) {
  console.log('日期范围:', start, end)
}
</script>
```

**Props:**

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| startDate | string | - | 开始日期 |
| endDate | string | - | 结束日期 |
| format | string | 'YYYY-MM-DD' | 日期格式 |
| showPresets | boolean | true | 显示快捷选项 |
| allowClear | boolean | true | 允许清除 |
| disabled | boolean | false | 禁用状态 |

### DataExport 数据导出

```vue
<template>
  <DataExport
    :data="tableData"
    :columns="columns"
    filename="user_analytics"
    @export="handleExport"
  />
</template>

<script setup>
const columns = [
  { key: 'date', title: '日期' },
  { key: 'count', title: '数量' },
  { 
    key: 'rate', 
    title: '比率',
    formatter: (v) => `${(v * 100).toFixed(2)}%`
  },
]
</script>
```

### HeatmapChart 热力图

```vue
<template>
  <HeatmapChart
    :data="heatmapData"
    :x-labels="hours"
    :y-labels="weekdays"
    :height="300"
    min-color="#e6f7ff"
    max-color="#1890ff"
    title="用户活跃时段"
  />
</template>

<script setup>
const heatmapData = [
  { x: 0, y: 0, value: 100 },
  { x: 1, y: 0, value: 200 },
  // ...
]
const hours = ['0:00', '1:00', /* ... */ '23:00']
const weekdays = ['周一', '周二', /* ... */ '周日']
</script>
```

## 🔌 API 接口

### IAnalyticsService 接口

```typescript
interface IAnalyticsService {
  // 获取仪表盘统计
  getDashboardStats(): Promise<DashboardStats>
  
  // 获取用户增长趋势
  getUserTrend(days: number): Promise<TimeSeriesPoint[]>
  
  // 获取物品类型统计
  getItemTypeStats(): Promise<CategoryStats[]>
  
  // 获取推荐量趋势
  getRecommendationTrend(days: number): Promise<TimeSeriesPoint[]>
  
  // 获取热门分类
  getTopCategories(limit: number): Promise<CategoryStats[]>
  
  // 获取CTR趋势
  getCTRTrend(startDate: string, endDate: string): Promise<TimeSeriesPoint[]>
}
```

### 扩展接口

本模块还提供以下扩展接口：

```typescript
// 推荐统计
getRecommendationStats(startDate: string, endDate: string): Promise<RecommendationStats>

// 响应延迟趋势
getLatencyTrend(startDate: string, endDate: string): Promise<TimeSeriesPoint[]>

// 热门推荐物品
getTopRecommendedItems(limit: number): Promise<TopRecommendedItem[]>

// 用户活跃度分布
getUserActivityDistribution(): Promise<UserActivityDistribution[]>

// 用户性别统计
getUserGenderStats(): Promise<CategoryStats[]>

// 用户年龄分布
getUserAgeDistribution(): Promise<CategoryStats[]>

// 物品增长趋势
getItemGrowthTrend(days: number): Promise<ItemGrowthTrend[]>

// 物品状态统计
getItemStatusStats(): Promise<CategoryStats[]>

// 物品分类统计
getItemCategoryStats(): Promise<CategoryStats[]>
```

## 🧪 运行测试

```bash
# 运行所有测试
npm run test

# 运行特定测试文件
npm run test -- src/components/__tests__/StatCard.spec.ts

# 查看测试覆盖率
npm run test:coverage
```

## 📝 开发规范

### 1. 组件规范

- 使用 Vue 3 Composition API + `<script setup>` 语法
- 所有 Props 必须定义 TypeScript 类型
- 使用 `withDefaults` 设置默认值
- 样式使用 `scoped` 避免污染

```vue
<script setup lang="ts">
interface Props {
  title: string
  value: number
}

const props = withDefaults(defineProps<Props>(), {
  value: 0,
})
</script>
```

### 2. API 调用规范

- 使用依赖注入获取 API Provider
- 不直接导入具体 API 实现
- 统一错误处理

```typescript
// ✅ 正确
const api = inject<IApiProvider>('api')
const data = await api?.analytics.getDashboardStats()

// ❌ 错误
import { analyticsApi } from '@/api/analytics'
```

### 3. 样式规范

- 使用 CSS 变量实现主题化
- 遵循 Ant Design 设计规范
- 响应式适配移动端

## 🔄 后续开发计划

- [ ] 添加实时数据刷新功能
- [ ] 支持自定义仪表盘布局
- [ ] 添加报表生成功能
- [ ] 支持数据对比分析
- [ ] 添加告警阈值配置

## 📞 联系方式

如有问题，请联系 Person E 或提交 Issue。

---

*最后更新: 2025-01-05*

