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
  DataZoomComponent,
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
  DataZoomComponent,
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
  title?: string
  showDataZoom?: boolean
  dateFormat?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  height: 300,
  color: '#1890ff',
  smooth: true,
  showArea: true,
  title: '',
  showDataZoom: false,
  dateFormat: true,
})

const formatDate = (value: string): string => {
  if (!props.dateFormat) return value
  const date = new Date(value)
  return `${date.getMonth() + 1}/${date.getDate()}`
}

const formatTooltipDate = (value: string): string => {
  const date = new Date(value)
  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  })
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
    trigger: 'axis',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderColor: '#e8e8e8',
    borderWidth: 1,
    textStyle: {
      color: '#262626',
    },
    formatter: (params: any) => {
      const item = params[0]
      const date = formatTooltipDate(item.axisValue)
      return `
        <div style="padding: 4px 8px;">
          <div style="color: #8c8c8c; margin-bottom: 4px;">${date}</div>
          <div style="font-weight: 600; color: ${props.color};">
            ${typeof item.value === 'number' ? item.value.toLocaleString() : item.value}
          </div>
        </div>
      `
    },
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: props.showDataZoom ? '15%' : '3%',
    top: props.title ? '15%' : '10%',
    containLabel: true,
  },
  dataZoom: props.showDataZoom ? [
    {
      type: 'inside',
      start: 0,
      end: 100,
    },
    {
      type: 'slider',
      start: 0,
      end: 100,
      height: 20,
      bottom: '2%',
    },
  ] : undefined,
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
      formatter: formatDate,
    },
    axisTick: {
      show: false,
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
      formatter: (value: number) => {
        if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`
        if (value >= 1000) return `${(value / 1000).toFixed(1)}K`
        return value.toString()
      },
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
      emphasis: {
        focus: 'series',
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.2)',
        },
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
    :style="{ height: `${height}px`, width: '100%' }"
    autoresize
    class="analytics-line-chart"
  />
</template>

<style scoped>
.analytics-line-chart {
  min-height: 200px;
}
</style>

