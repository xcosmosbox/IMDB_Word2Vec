<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { BarChart as EChartsBarChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsOption } from 'echarts'

use([
  EChartsBarChart,
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
  horizontal?: boolean
  showLabel?: boolean
  barWidth?: number
  title?: string
}

const props = withDefaults(defineProps<Props>(), {
  height: 300,
  color: '#1890ff',
  horizontal: true,
  showLabel: false,
  barWidth: 16,
  title: '',
})

const option = computed<EChartsOption>(() => {
  const sortedData = [...props.data].sort((a, b) => b[props.yField] - a[props.yField])
  
  const baseConfig = {
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
      axisPointer: {
        type: 'shadow',
      },
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderColor: '#e8e8e8',
      borderWidth: 1,
      textStyle: {
        color: '#262626',
      },
      formatter: (params: any) => {
        const item = params[0]
        return `
          <div style="padding: 4px 8px;">
            <div style="font-weight: 500; margin-bottom: 4px;">${item.name}</div>
            <div style="color: ${props.color}; font-weight: 600;">
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
      top: props.title ? '15%' : '3%',
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
        axisLabel: {
          color: '#8c8c8c',
          formatter: (value: number) => {
            if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`
            if (value >= 1000) return `${(value / 1000).toFixed(1)}K`
            return value.toString()
          },
        },
      },
      yAxis: {
        type: 'category',
        data: sortedData.map(item => item[props.xField]),
        axisLine: { lineStyle: { color: '#d9d9d9' } },
        axisTick: { show: false },
        axisLabel: {
          color: '#595959',
          width: 80,
          overflow: 'truncate',
        },
        inverse: true,
      },
      series: [{
        type: 'bar',
        data: sortedData.map(item => item[props.yField]),
        barWidth: props.barWidth,
        label: props.showLabel ? {
          show: true,
          position: 'right',
          color: '#595959',
          formatter: '{c}',
        } : { show: false },
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
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.2)',
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
      axisTick: { show: false },
      axisLabel: {
        rotate: sortedData.length > 6 ? 45 : 0,
        color: '#8c8c8c',
        interval: 0,
      },
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: {
        lineStyle: { color: '#f0f0f0', type: 'dashed' },
      },
      axisLabel: {
        color: '#8c8c8c',
      },
    },
    series: [{
      type: 'bar',
      data: sortedData.map(item => item[props.yField]),
      barWidth: props.barWidth + 8,
      label: props.showLabel ? {
        show: true,
        position: 'top',
        color: '#595959',
        formatter: '{c}',
      } : { show: false },
      itemStyle: {
        borderRadius: [4, 4, 0, 0],
        color: props.color,
      },
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.2)',
        },
      },
    }],
  } as EChartsOption
})
</script>

<template>
  <VChart
    :option="option"
    :style="{ height: `${height}px`, width: '100%' }"
    autoresize
    class="analytics-bar-chart"
  />
</template>

<style scoped>
.analytics-bar-chart {
  min-height: 200px;
}
</style>

