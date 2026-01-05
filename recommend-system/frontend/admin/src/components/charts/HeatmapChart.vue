<script setup lang="ts">
/**
 * 热力图组件
 * 
 * 用于展示二维数据分布，如用户活跃时间分布、行为热力图等
 */
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { HeatmapChart as EChartsHeatmapChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  VisualMapComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsOption } from 'echarts'

// 注册组件
use([
  EChartsHeatmapChart,
  TitleComponent,
  TooltipComponent,
  GridComponent,
  VisualMapComponent,
  CanvasRenderer,
])

interface HeatmapDataItem {
  x: number | string
  y: number | string
  value: number
}

interface Props {
  /** 数据数组 */
  data: HeatmapDataItem[]
  /** X轴标签 */
  xLabels: string[]
  /** Y轴标签 */
  yLabels: string[]
  /** 高度 */
  height?: number
  /** 最小值颜色 */
  minColor?: string
  /** 最大值颜色 */
  maxColor?: string
  /** 标题 */
  title?: string
  /** 是否显示数值 */
  showLabel?: boolean
  /** 最大值（用于色阶计算） */
  max?: number
  /** 最小值（用于色阶计算） */
  min?: number
}

const props = withDefaults(defineProps<Props>(), {
  height: 300,
  minColor: '#f0f5ff',
  maxColor: '#1890ff',
  title: '',
  showLabel: true,
})

// 转换数据格式为 ECharts 需要的格式
const chartData = computed(() => {
  return props.data.map(item => {
    const xIndex = typeof item.x === 'number' ? item.x : props.xLabels.indexOf(item.x)
    const yIndex = typeof item.y === 'number' ? item.y : props.yLabels.indexOf(item.y)
    return [xIndex, yIndex, item.value]
  })
})

// 计算数据范围
const dataRange = computed(() => {
  const values = props.data.map(d => d.value)
  return {
    min: props.min ?? Math.min(...values),
    max: props.max ?? Math.max(...values),
  }
})

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
      const xLabel = props.xLabels[params.data[0]] || params.data[0]
      const yLabel = props.yLabels[params.data[1]] || params.data[1]
      return `
        <div style="padding: 4px 8px;">
          <div style="margin-bottom: 4px;">
            <span style="color: #8c8c8c;">${yLabel}</span>
            <span style="margin: 0 4px;">-</span>
            <span style="color: #8c8c8c;">${xLabel}</span>
          </div>
          <div style="font-weight: 600; color: #1890ff;">
            ${params.data[2].toLocaleString()}
          </div>
        </div>
      `
    },
  },
  grid: {
    left: '10%',
    right: '15%',
    bottom: '15%',
    top: props.title ? '15%' : '10%',
    containLabel: true,
  },
  xAxis: {
    type: 'category',
    data: props.xLabels,
    splitArea: {
      show: true,
    },
    axisLine: {
      lineStyle: {
        color: '#d9d9d9',
      },
    },
    axisLabel: {
      color: '#8c8c8c',
      fontSize: 11,
    },
  },
  yAxis: {
    type: 'category',
    data: props.yLabels,
    splitArea: {
      show: true,
    },
    axisLine: {
      lineStyle: {
        color: '#d9d9d9',
      },
    },
    axisLabel: {
      color: '#8c8c8c',
      fontSize: 11,
    },
  },
  visualMap: {
    min: dataRange.value.min,
    max: dataRange.value.max,
    calculable: true,
    orient: 'vertical',
    right: '2%',
    top: 'center',
    inRange: {
      color: [props.minColor, props.maxColor],
    },
    textStyle: {
      color: '#8c8c8c',
    },
  },
  series: [
    {
      type: 'heatmap',
      data: chartData.value,
      label: {
        show: props.showLabel,
        color: '#262626',
        fontSize: 10,
        formatter: (params: any) => {
          const value = params.data[2]
          if (value >= 1000) {
            return `${(value / 1000).toFixed(1)}k`
          }
          return value.toString()
        },
      },
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowColor: 'rgba(0, 0, 0, 0.3)',
        },
      },
      itemStyle: {
        borderColor: '#fff',
        borderWidth: 1,
      },
    },
  ],
}))
</script>

<template>
  <VChart
    :option="option"
    :style="{ height: `${height}px`, width: '100%' }"
    autoresize
    class="analytics-heatmap-chart"
  />
</template>

<style scoped>
.analytics-heatmap-chart {
  min-height: 200px;
}
</style>

