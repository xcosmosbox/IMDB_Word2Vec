<script setup lang="ts">
/**
 * DataTable - 通用数据表格组件
 * 
 * 封装 Ant Design Vue Table，提供统一的表格功能：
 * - 分页
 * - 排序
 * - 加载状态
 * - 空状态
 * - 行选择
 */
import { computed } from 'vue'
import { Table, Empty, Spin } from 'ant-design-vue'
import type { TablePaginationConfig, TableProps } from 'ant-design-vue'
import type { SorterResult, FilterValue } from 'ant-design-vue/es/table/interface'

export interface Column {
  title: string
  dataIndex?: string
  key?: string
  width?: number | string
  fixed?: 'left' | 'right'
  ellipsis?: boolean
  sorter?: boolean | ((a: any, b: any) => number)
  customRender?: (opt: { text: any; record: any; index: number }) => any
  align?: 'left' | 'center' | 'right'
}

export interface Pagination {
  current: number
  pageSize: number
  total: number
  showSizeChanger?: boolean
  showQuickJumper?: boolean
  showTotal?: (total: number, range: [number, number]) => string
}

const props = withDefaults(defineProps<{
  /** 表格列定义 */
  columns: Column[]
  /** 数据源 */
  dataSource: any[]
  /** 行唯一键 */
  rowKey?: string | ((record: any) => string)
  /** 加载状态 */
  loading?: boolean
  /** 分页配置 */
  pagination?: Pagination | false
  /** 横向滚动宽度 */
  scrollX?: number | string
  /** 纵向滚动高度 */
  scrollY?: number | string
  /** 是否可选择行 */
  rowSelection?: TableProps['rowSelection']
  /** 表格大小 */
  size?: 'large' | 'middle' | 'small'
  /** 是否显示边框 */
  bordered?: boolean
  /** 空状态描述 */
  emptyText?: string
}>(), {
  rowKey: 'id',
  loading: false,
  size: 'middle',
  bordered: false,
  emptyText: '暂无数据',
})

const emit = defineEmits<{
  /** 表格变化事件（分页、排序、筛选） */
  change: [
    pagination: TablePaginationConfig,
    filters: Record<string, FilterValue | null>,
    sorter: SorterResult<any> | SorterResult<any>[]
  ]
  /** 分页变化事件 */
  pageChange: [page: number, pageSize: number]
}>()

// 计算分页配置
const paginationConfig = computed(() => {
  if (props.pagination === false) {
    return false
  }
  
  if (!props.pagination) {
    return false
  }

  return {
    ...props.pagination,
    showSizeChanger: props.pagination.showSizeChanger ?? true,
    showQuickJumper: props.pagination.showQuickJumper ?? true,
    showTotal: props.pagination.showTotal ?? ((total: number) => `共 ${total} 条`),
    pageSizeOptions: ['10', '20', '50', '100'],
  }
})

// 计算滚动配置
const scrollConfig = computed(() => {
  const config: { x?: number | string; y?: number | string } = {}
  if (props.scrollX) {
    config.x = props.scrollX
  }
  if (props.scrollY) {
    config.y = props.scrollY
  }
  return Object.keys(config).length > 0 ? config : undefined
})

// 处理表格变化
function handleChange(
  pagination: TablePaginationConfig,
  filters: Record<string, FilterValue | null>,
  sorter: SorterResult<any> | SorterResult<any>[]
) {
  emit('change', pagination, filters, sorter)
  
  if (pagination.current && pagination.pageSize) {
    emit('pageChange', pagination.current, pagination.pageSize)
  }
}
</script>

<template>
  <div class="data-table">
    <Spin :spinning="loading">
      <Table
        :columns="columns"
        :data-source="dataSource"
        :row-key="rowKey"
        :pagination="paginationConfig"
        :scroll="scrollConfig"
        :row-selection="rowSelection"
        :size="size"
        :bordered="bordered"
        @change="handleChange"
      >
        <!-- 空状态 -->
        <template #emptyText>
          <Empty :description="emptyText" />
        </template>

        <!-- 透传所有插槽 -->
        <template
          v-for="(_, name) in $slots"
          :key="name"
          #[name]="slotData"
        >
          <slot :name="name" v-bind="slotData || {}" />
        </template>
      </Table>
    </Spin>
  </div>
</template>

<style scoped>
.data-table {
  width: 100%;
}

.data-table :deep(.ant-table-thead > tr > th) {
  background: #fafafa;
  font-weight: 600;
}

.data-table :deep(.ant-table-tbody > tr:hover > td) {
  background: #f5f7fa;
}

.data-table :deep(.ant-pagination) {
  margin-top: 16px;
  margin-bottom: 0;
}
</style>

