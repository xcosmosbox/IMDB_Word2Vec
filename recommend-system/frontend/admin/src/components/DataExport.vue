<script setup lang="ts">
/**
 * 数据导出组件
 * 
 * 支持将数据导出为 CSV、JSON、Excel 等格式
 */
import { ref, computed } from 'vue'
import { Button, Dropdown, Menu, Modal, Checkbox, Space, message } from 'ant-design-vue'
import { 
  DownloadOutlined, 
  FileExcelOutlined, 
  FileTextOutlined,
  SettingOutlined 
} from '@ant-design/icons-vue'

interface Column {
  key: string
  title: string
  formatter?: (value: any) => string
}

interface Props {
  /** 要导出的数据 */
  data: Record<string, any>[]
  /** 列定义 */
  columns: Column[]
  /** 文件名（不含扩展名） */
  filename?: string
  /** 是否显示导出设置弹窗 */
  showSettings?: boolean
  /** 是否禁用 */
  disabled?: boolean
  /** 按钮文字 */
  buttonText?: string
}

const props = withDefaults(defineProps<Props>(), {
  filename: 'export_data',
  showSettings: true,
  disabled: false,
  buttonText: '导出数据',
})

const emit = defineEmits<{
  export: [format: string, data: Record<string, any>[]]
}>()

// 设置弹窗状态
const settingsVisible = ref(false)
const selectedColumns = ref<string[]>([])
const exportLoading = ref(false)

// 初始化选中的列
selectedColumns.value = props.columns.map(col => col.key)

// 当前可用的列
const availableColumns = computed(() => {
  return props.columns.filter(col => selectedColumns.value.includes(col.key))
})

// 导出菜单项
const exportMenuItems = [
  { key: 'csv', label: 'CSV 格式', icon: FileTextOutlined },
  { key: 'json', label: 'JSON 格式', icon: FileTextOutlined },
  { key: 'excel', label: 'Excel 格式', icon: FileExcelOutlined },
]

// 导出为 CSV
function exportToCsv(data: Record<string, any>[], columns: Column[]): string {
  const headers = columns.map(col => col.title).join(',')
  const rows = data.map(row => {
    return columns.map(col => {
      let value = row[col.key]
      if (col.formatter) {
        value = col.formatter(value)
      }
      // 处理包含逗号或换行的值
      if (typeof value === 'string' && (value.includes(',') || value.includes('\n') || value.includes('"'))) {
        value = `"${value.replace(/"/g, '""')}"`
      }
      return value ?? ''
    }).join(',')
  })
  
  // 添加 BOM 以支持中文
  return '\ufeff' + [headers, ...rows].join('\n')
}

// 导出为 JSON
function exportToJson(data: Record<string, any>[], columns: Column[]): string {
  const filteredData = data.map(row => {
    const newRow: Record<string, any> = {}
    columns.forEach(col => {
      let value = row[col.key]
      if (col.formatter) {
        value = col.formatter(value)
      }
      newRow[col.title] = value
    })
    return newRow
  })
  return JSON.stringify(filteredData, null, 2)
}

// 下载文件
function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

// 处理导出
async function handleExport(format: string) {
  if (props.data.length === 0) {
    message.warning('没有可导出的数据')
    return
  }
  
  exportLoading.value = true
  
  try {
    const columns = availableColumns.value
    const timestamp = new Date().toISOString().slice(0, 10).replace(/-/g, '')
    const baseFilename = `${props.filename}_${timestamp}`
    
    let content: string
    let filename: string
    let mimeType: string
    
    switch (format) {
      case 'csv':
        content = exportToCsv(props.data, columns)
        filename = `${baseFilename}.csv`
        mimeType = 'text/csv;charset=utf-8'
        break
        
      case 'json':
        content = exportToJson(props.data, columns)
        filename = `${baseFilename}.json`
        mimeType = 'application/json;charset=utf-8'
        break
        
      case 'excel':
        // Excel 格式使用 CSV 兼容方式（真正的 Excel 需要额外库）
        content = exportToCsv(props.data, columns)
        filename = `${baseFilename}.csv`
        mimeType = 'text/csv;charset=utf-8'
        message.info('已导出为 CSV 格式，可用 Excel 打开')
        break
        
      default:
        throw new Error(`不支持的导出格式: ${format}`)
    }
    
    downloadFile(content, filename, mimeType)
    emit('export', format, props.data)
    message.success(`导出成功：${filename}`)
    
  } catch (error) {
    console.error('导出失败:', error)
    message.error('导出失败，请重试')
  } finally {
    exportLoading.value = false
    settingsVisible.value = false
  }
}

// 菜单点击
function handleMenuClick({ key }: { key: string }) {
  if (props.showSettings) {
    settingsVisible.value = true
  } else {
    handleExport(key)
  }
}

// 全选/取消全选列
function toggleAllColumns(checked: boolean) {
  if (checked) {
    selectedColumns.value = props.columns.map(col => col.key)
  } else {
    selectedColumns.value = []
  }
}

// 是否全选
const isAllSelected = computed(() => 
  selectedColumns.value.length === props.columns.length
)

// 是否部分选中
const isIndeterminate = computed(() => 
  selectedColumns.value.length > 0 && selectedColumns.value.length < props.columns.length
)
</script>

<template>
  <div class="data-export">
    <Dropdown :disabled="disabled || data.length === 0">
      <template #overlay>
        <Menu @click="handleMenuClick">
          <Menu.Item v-for="item in exportMenuItems" :key="item.key">
            <Space>
              <component :is="item.icon" />
              <span>{{ item.label }}</span>
            </Space>
          </Menu.Item>
        </Menu>
      </template>
      <Button :disabled="disabled || data.length === 0" :loading="exportLoading">
        <template #icon><DownloadOutlined /></template>
        {{ buttonText }}
      </Button>
    </Dropdown>
    
    <!-- 导出设置弹窗 -->
    <Modal
      v-model:open="settingsVisible"
      title="导出设置"
      :width="400"
      @ok="handleExport('csv')"
      @cancel="settingsVisible = false"
    >
      <div class="export-settings">
        <div class="settings-header">
          <span class="settings-title">选择导出列</span>
          <Checkbox
            :checked="isAllSelected"
            :indeterminate="isIndeterminate"
            @change="(e: any) => toggleAllColumns(e.target.checked)"
          >
            全选
          </Checkbox>
        </div>
        
        <div class="column-list">
          <Checkbox
            v-for="column in columns"
            :key="column.key"
            :checked="selectedColumns.includes(column.key)"
            @change="(e: any) => {
              if (e.target.checked) {
                selectedColumns.push(column.key)
              } else {
                selectedColumns = selectedColumns.filter(k => k !== column.key)
              }
            }"
          >
            {{ column.title }}
          </Checkbox>
        </div>
        
        <div class="export-info">
          <SettingOutlined />
          <span>将导出 {{ data.length }} 条数据，{{ selectedColumns.length }} 列</span>
        </div>
      </div>
      
      <template #footer>
        <Space>
          <Button @click="settingsVisible = false">取消</Button>
          <Button @click="handleExport('json')">
            <template #icon><FileTextOutlined /></template>
            导出 JSON
          </Button>
          <Button type="primary" @click="handleExport('csv')">
            <template #icon><FileExcelOutlined /></template>
            导出 CSV
          </Button>
        </Space>
      </template>
    </Modal>
  </div>
</template>

<style scoped>
.data-export {
  display: inline-block;
}

.export-settings {
  padding: 8px 0;
}

.settings-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #f0f0f0;
}

.settings-title {
  font-weight: 500;
  color: #262626;
}

.column-list {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  margin-bottom: 16px;
}

.export-info {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  background: #fafafa;
  border-radius: 6px;
  color: #8c8c8c;
  font-size: 13px;
}
</style>

