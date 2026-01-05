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
  title?: string
  showLegend?: boolean
  donut?: boolean
  roseType?: 'radius' | 'area' | false
}

const props = withDefaults(defineProps<Props>(), {
  height: 300,
  colors: () => ['#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1', '#13c2c2', '#eb2f96', '#fa541c'],
  title: '',
  showLegend: true,
  donut: true,
  roseType: false,
})

// 类型映射
const typeLabels: Record<string, string> = {
  movie: '电影',
  product: '商品',
  article: '文章',
  video: '视频',
  active: '活跃',
  inactive: '非活跃',
  male: '男性',
  female: '女性',
  unknown: '未知',
}

const getLabel = (name: string): string => {
  return typeLabels[name] || name
}

const option = computed<EChartsOption>(() => ({
  title: props.title ? {
    text: props.title,
    left: 'center',
    textStyle: {
      fontSize: 14,
      fontWeight: 500,
      color: '#262626',
    },
  } : undefined,
  tooltip: {
    trigger: 'item',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderColor: '#e8e8e8',
    borderWidth: 1,
    textStyle: {
      color: '#262626',
    },
    formatter: (params: any) => {
      return `
        <div style="padding: 4px 8px;">
          <div style="font-weight: 600; margin-bottom: 4px;">${getLabel(params.name)}</div>
          <div style="color: #8c8c8c;">
            数量: <span style="color: #262626; font-weight: 500;">${params.value.toLocaleString()}</span>
          </div>
          <div style="color: #8c8c8c;">
            占比: <span style="color: #262626; font-weight: 500;">${params.percent}%</span>
          </div>
        </div>
      `
    },
  },
  legend: props.showLegend ? {
    orient: 'vertical',
    right: '5%',
    top: 'center',
    itemWidth: 14,
    itemHeight: 14,
    itemGap: 12,
    textStyle: {
      color: '#595959',
    },
    formatter: (name: string) => getLabel(name),
  } : undefined,
  color: props.colors,
  series: [
    {
      type: 'pie',
      radius: props.donut ? ['45%', '70%'] : '70%',
      center: props.showLegend ? ['40%', '50%'] : ['50%', '50%'],
      roseType: props.roseType || undefined,
      avoidLabelOverlap: false,
      itemStyle: {
        borderRadius: 8,
        borderColor: '#fff',
        borderWidth: 2,
      },
      label: {
        show: false,
      },
      labelLine: {
        show: false,
      },
      emphasis: {
        label: {
          show: true,
          fontSize: 16,
          fontWeight: 'bold',
          formatter: (params: any) => getLabel(params.name),
        },
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.2)',
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
    :style="{ height: `${height}px`, width: '100%' }"
    autoresize
    class="analytics-pie-chart"
  />
</template>

<style scoped>
.analytics-pie-chart {
  min-height: 200px;
}
</style>

