<script setup lang="ts">
/**
 * SearchForm - 通用搜索表单组件
 * 
 * 提供统一的搜索表单布局和功能：
 * - 多字段搜索
 * - 展开/收起
 * - 重置功能
 */
import { ref, computed } from 'vue'
import { Form, FormItem, Row, Col, Button, Space } from 'ant-design-vue'
import { SearchOutlined, ReloadOutlined, DownOutlined, UpOutlined } from '@ant-design/icons-vue'

export interface SearchField {
  /** 字段名 */
  name: string
  /** 标签 */
  label: string
  /** 默认值 */
  defaultValue?: any
  /** 占据的栅格数 */
  span?: number
}

const props = withDefaults(defineProps<{
  /** 搜索字段配置 */
  fields?: SearchField[]
  /** 表单数据 */
  modelValue?: Record<string, any>
  /** 是否加载中 */
  loading?: boolean
  /** 默认展开的字段数量 */
  defaultExpandedCount?: number
  /** 是否显示展开按钮 */
  showExpand?: boolean
  /** 每行显示的字段数 */
  columnsPerRow?: number
}>(), {
  fields: () => [],
  modelValue: () => ({}),
  loading: false,
  defaultExpandedCount: 3,
  showExpand: true,
  columnsPerRow: 4,
})

const emit = defineEmits<{
  'update:modelValue': [value: Record<string, any>]
  search: [value: Record<string, any>]
  reset: []
}>()

// 是否展开
const expanded = ref(false)

// 计算每个字段的栅格宽度
const colSpan = computed(() => Math.floor(24 / props.columnsPerRow))

// 需要展示的字段
const visibleFields = computed(() => {
  if (!props.showExpand || expanded.value) {
    return props.fields
  }
  return props.fields.slice(0, props.defaultExpandedCount)
})

// 是否显示展开按钮
const showExpandButton = computed(() => {
  return props.showExpand && props.fields.length > props.defaultExpandedCount
})

// 更新表单数据
function updateValue(field: string, value: any) {
  emit('update:modelValue', {
    ...props.modelValue,
    [field]: value,
  })
}

// 搜索
function handleSearch() {
  emit('search', props.modelValue)
}

// 重置
function handleReset() {
  const resetValue: Record<string, any> = {}
  props.fields.forEach(field => {
    resetValue[field.name] = field.defaultValue ?? ''
  })
  emit('update:modelValue', resetValue)
  emit('reset')
}

// 切换展开状态
function toggleExpand() {
  expanded.value = !expanded.value
}
</script>

<template>
  <div class="search-form">
    <Form layout="inline" class="search-form-inner">
      <Row :gutter="16" class="search-row">
        <!-- 搜索字段 -->
        <Col
          v-for="field in visibleFields"
          :key="field.name"
          :span="field.span || colSpan"
          class="search-col"
        >
          <FormItem :label="field.label" class="search-item">
            <slot
              :name="field.name"
              :value="modelValue[field.name]"
              :update="(value: any) => updateValue(field.name, value)"
            >
              <!-- 默认插槽内容 -->
            </slot>
          </FormItem>
        </Col>

        <!-- 操作按钮 -->
        <Col :span="colSpan" class="search-col actions-col">
          <FormItem class="search-item actions-item">
            <Space>
              <Button
                type="primary"
                :loading="loading"
                @click="handleSearch"
              >
                <SearchOutlined />
                搜索
              </Button>
              <Button @click="handleReset">
                <ReloadOutlined />
                重置
              </Button>
              <Button
                v-if="showExpandButton"
                type="link"
                class="expand-btn"
                @click="toggleExpand"
              >
                {{ expanded ? '收起' : '展开' }}
                <component :is="expanded ? UpOutlined : DownOutlined" />
              </Button>
            </Space>
          </FormItem>
        </Col>
      </Row>
    </Form>
  </div>
</template>

<style scoped>
.search-form {
  margin-bottom: 16px;
  padding: 16px 16px 0;
  background: #fafafa;
  border-radius: 8px;
}

.search-form-inner {
  width: 100%;
}

.search-row {
  width: 100%;
}

.search-col {
  margin-bottom: 16px;
}

.search-item {
  width: 100%;
  margin-right: 0;
}

.search-item :deep(.ant-form-item-label) {
  width: 80px;
  text-align: right;
}

.search-item :deep(.ant-form-item-control) {
  flex: 1;
}

.actions-col {
  display: flex;
  justify-content: flex-end;
}

.actions-item {
  width: auto;
}

.expand-btn {
  padding: 0;
}
</style>

