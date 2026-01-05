<script setup lang="ts">
/**
 * ConfirmModal - 确认弹窗组件
 * 
 * 封装常用的确认操作弹窗：
 * - 删除确认
 * - 操作确认
 * - 自定义内容
 */
import { ref, computed } from 'vue'
import { Modal, Button, Space, Alert } from 'ant-design-vue'
import {
  ExclamationCircleOutlined,
  DeleteOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons-vue'

export type ConfirmType = 'info' | 'warning' | 'error' | 'success' | 'confirm'

const props = withDefaults(defineProps<{
  /** 是否显示 */
  open?: boolean
  /** 标题 */
  title?: string
  /** 内容 */
  content?: string
  /** 确认类型 */
  type?: ConfirmType
  /** 确认按钮文字 */
  okText?: string
  /** 取消按钮文字 */
  cancelText?: string
  /** 确认按钮是否为危险按钮 */
  okDanger?: boolean
  /** 是否加载中 */
  loading?: boolean
  /** 宽度 */
  width?: number | string
  /** 是否显示图标 */
  showIcon?: boolean
  /** 是否显示取消按钮 */
  showCancel?: boolean
}>(), {
  open: false,
  title: '确认',
  content: '确定要执行此操作吗？',
  type: 'confirm',
  okText: '确定',
  cancelText: '取消',
  okDanger: false,
  loading: false,
  width: 416,
  showIcon: true,
  showCancel: true,
})

const emit = defineEmits<{
  'update:open': [value: boolean]
  ok: []
  cancel: []
}>()

// 图标映射
const iconMap = {
  info: InfoCircleOutlined,
  warning: WarningOutlined,
  error: DeleteOutlined,
  success: CheckCircleOutlined,
  confirm: ExclamationCircleOutlined,
}

// 图标颜色映射
const iconColorMap = {
  info: '#1890ff',
  warning: '#faad14',
  error: '#ff4d4f',
  success: '#52c41a',
  confirm: '#faad14',
}

// 当前图标
const currentIcon = computed(() => iconMap[props.type])
const currentIconColor = computed(() => iconColorMap[props.type])

// 关闭弹窗
function handleClose() {
  emit('update:open', false)
}

// 取消
function handleCancel() {
  emit('cancel')
  handleClose()
}

// 确认
function handleOk() {
  emit('ok')
}
</script>

<template>
  <Modal
    :open="open"
    :title="null"
    :footer="null"
    :width="width"
    :closable="false"
    :mask-closable="false"
    centered
    class="confirm-modal"
    @cancel="handleCancel"
  >
    <div class="confirm-content">
      <!-- 图标 -->
      <div v-if="showIcon" class="confirm-icon" :style="{ color: currentIconColor }">
        <component :is="currentIcon" />
      </div>

      <!-- 内容区 -->
      <div class="confirm-body">
        <div class="confirm-title">{{ title }}</div>
        <div class="confirm-message">
          <slot>{{ content }}</slot>
        </div>
      </div>
    </div>

    <!-- 操作按钮 -->
    <div class="confirm-footer">
      <Space>
        <Button v-if="showCancel" @click="handleCancel">
          {{ cancelText }}
        </Button>
        <Button
          type="primary"
          :danger="okDanger || type === 'error'"
          :loading="loading"
          @click="handleOk"
        >
          {{ okText }}
        </Button>
      </Space>
    </div>
  </Modal>
</template>

<style scoped>
.confirm-modal :deep(.ant-modal-body) {
  padding: 24px;
}

.confirm-content {
  display: flex;
  gap: 16px;
}

.confirm-icon {
  flex-shrink: 0;
  font-size: 22px;
  line-height: 1;
}

.confirm-body {
  flex: 1;
}

.confirm-title {
  font-size: 16px;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.85);
  margin-bottom: 8px;
}

.confirm-message {
  color: rgba(0, 0, 0, 0.65);
  line-height: 1.6;
}

.confirm-footer {
  display: flex;
  justify-content: flex-end;
  margin-top: 24px;
}
</style>

