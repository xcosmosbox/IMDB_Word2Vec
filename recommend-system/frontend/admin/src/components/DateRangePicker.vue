<script setup lang="ts">
/**
 * 日期范围选择器组件
 * 
 * 封装 Ant Design Vue 的日期选择器，提供常用快捷选项
 */
import { ref, computed, watch } from 'vue'
import { DatePicker, Space, Button } from 'ant-design-vue'
import dayjs, { Dayjs } from 'dayjs'

interface Props {
  /** 开始日期 */
  startDate?: string
  /** 结束日期 */
  endDate?: string
  /** 日期格式 */
  format?: string
  /** 是否显示快捷选项 */
  showPresets?: boolean
  /** 是否允许清除 */
  allowClear?: boolean
  /** 是否禁用 */
  disabled?: boolean
  /** 最大可选日期 */
  maxDate?: string
  /** 最小可选日期 */
  minDate?: string
}

const props = withDefaults(defineProps<Props>(), {
  format: 'YYYY-MM-DD',
  showPresets: true,
  allowClear: true,
  disabled: false,
})

const emit = defineEmits<{
  'update:startDate': [value: string]
  'update:endDate': [value: string]
  change: [startDate: string, endDate: string]
}>()

// 内部日期范围状态
const dateRange = ref<[Dayjs, Dayjs] | null>(null)

// 初始化日期范围
watch(
  () => [props.startDate, props.endDate],
  ([start, end]) => {
    if (start && end) {
      dateRange.value = [dayjs(start), dayjs(end)]
    } else {
      dateRange.value = null
    }
  },
  { immediate: true }
)

// 快捷选项
const presets = computed(() => {
  const today = dayjs()
  return [
    { label: '今天', value: [today, today] as [Dayjs, Dayjs] },
    { label: '近7天', value: [today.subtract(6, 'day'), today] as [Dayjs, Dayjs] },
    { label: '近15天', value: [today.subtract(14, 'day'), today] as [Dayjs, Dayjs] },
    { label: '近30天', value: [today.subtract(29, 'day'), today] as [Dayjs, Dayjs] },
    { label: '本月', value: [today.startOf('month'), today] as [Dayjs, Dayjs] },
    { label: '上月', value: [
      today.subtract(1, 'month').startOf('month'),
      today.subtract(1, 'month').endOf('month')
    ] as [Dayjs, Dayjs] },
  ]
})

// 当前选中的快捷选项
const activePreset = computed(() => {
  if (!dateRange.value) return null
  
  const [start, end] = dateRange.value
  for (const preset of presets.value) {
    const [presetStart, presetEnd] = preset.value
    if (start.isSame(presetStart, 'day') && end.isSame(presetEnd, 'day')) {
      return preset.label
    }
  }
  return null
})

// 日期变化处理
function handleChange(dates: [Dayjs, Dayjs] | null) {
  dateRange.value = dates
  
  if (dates) {
    const [start, end] = dates
    const startStr = start.format(props.format)
    const endStr = end.format(props.format)
    
    emit('update:startDate', startStr)
    emit('update:endDate', endStr)
    emit('change', startStr, endStr)
  } else {
    emit('update:startDate', '')
    emit('update:endDate', '')
    emit('change', '', '')
  }
}

// 选择快捷选项
function selectPreset(preset: { label: string; value: [Dayjs, Dayjs] }) {
  handleChange(preset.value)
}

// 禁用日期判断
function disabledDate(current: Dayjs): boolean {
  if (!current) return false
  
  const today = dayjs().endOf('day')
  
  // 不能选择未来日期
  if (current.isAfter(today)) return true
  
  // 检查最大日期
  if (props.maxDate && current.isAfter(dayjs(props.maxDate))) return true
  
  // 检查最小日期
  if (props.minDate && current.isBefore(dayjs(props.minDate))) return true
  
  return false
}
</script>

<template>
  <div class="date-range-picker">
    <Space direction="vertical" :size="12" style="width: 100%">
      <!-- 快捷选项 -->
      <div v-if="showPresets" class="preset-buttons">
        <Button
          v-for="preset in presets"
          :key="preset.label"
          :type="activePreset === preset.label ? 'primary' : 'default'"
          size="small"
          @click="selectPreset(preset)"
        >
          {{ preset.label }}
        </Button>
      </div>
      
      <!-- 日期选择器 -->
      <DatePicker.RangePicker
        v-model:value="dateRange"
        :format="format"
        :allow-clear="allowClear"
        :disabled="disabled"
        :disabled-date="disabledDate"
        style="width: 100%"
        @change="handleChange"
      />
    </Space>
  </div>
</template>

<style scoped>
.date-range-picker {
  display: inline-block;
  min-width: 300px;
}

.preset-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.preset-buttons :deep(.ant-btn) {
  border-radius: 4px;
}

.preset-buttons :deep(.ant-btn-primary) {
  font-weight: 500;
}
</style>

