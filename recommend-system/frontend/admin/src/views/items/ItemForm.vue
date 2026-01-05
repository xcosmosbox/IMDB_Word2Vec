<script setup lang="ts">
/**
 * ItemForm - 物品表单页面
 * 
 * 功能：
 * - 新增物品
 * - 编辑物品
 * - 表单验证
 * - 标签管理
 */
import { ref, reactive, onMounted, computed, inject } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  Form,
  FormItem,
  Input,
  Select,
  Button,
  Card,
  Space,
  message,
  Spin,
  Divider,
  Tag,
  Tooltip,
} from 'ant-design-vue'
import { SaveOutlined, ArrowLeftOutlined, PlusOutlined } from '@ant-design/icons-vue'
import type { Rule } from 'ant-design-vue/es/form'
import type { Item, CreateItemRequest, UpdateItemRequest } from '@shared/types'
import type { IApiProvider } from '@shared/api/interfaces'

const route = useRoute()
const router = useRouter()

// 通过依赖注入获取 API Provider
const api = inject<IApiProvider>('api')!

// 页面状态
const itemId = computed(() => route.params.id as string | undefined)
const isEdit = computed(() => !!itemId.value && route.path.includes('/edit'))
const isCreate = computed(() => route.path.includes('/create'))
const pageTitle = computed(() => (isEdit.value ? '编辑物品' : '新增物品'))

const loading = ref(false)
const submitLoading = ref(false)
const formRef = ref()

// 标签输入相关
const tagInputVisible = ref(false)
const tagInputValue = ref('')
const tagInputRef = ref()

// 表单数据
interface FormData {
  type: 'movie' | 'product' | 'article' | 'video' | undefined
  title: string
  description: string
  category: string
  tags: string[]
}

const formData = reactive<FormData>({
  type: undefined,
  title: '',
  description: '',
  category: '',
  tags: [],
})

// 类型选项
const typeOptions = [
  { value: 'movie', label: '电影' },
  { value: 'product', label: '商品' },
  { value: 'article', label: '文章' },
  { value: 'video', label: '视频' },
]

// 分类选项（可根据类型动态加载）
const categoryOptions = computed(() => {
  const baseOptions = [
    { value: '科技', label: '科技' },
    { value: '娱乐', label: '娱乐' },
    { value: '教育', label: '教育' },
    { value: '生活', label: '生活' },
    { value: '其他', label: '其他' },
  ]

  if (formData.type === 'movie') {
    return [
      { value: '动作', label: '动作' },
      { value: '喜剧', label: '喜剧' },
      { value: '剧情', label: '剧情' },
      { value: '科幻', label: '科幻' },
      { value: '恐怖', label: '恐怖' },
    ]
  }

  if (formData.type === 'product') {
    return [
      { value: '电子产品', label: '电子产品' },
      { value: '服装', label: '服装' },
      { value: '家居', label: '家居' },
      { value: '食品', label: '食品' },
      { value: '其他', label: '其他' },
    ]
  }

  return baseOptions
})

// 表单校验规则
const rules: Record<string, Rule[]> = {
  type: [{ required: true, message: '请选择物品类型', trigger: 'change' }],
  title: [
    { required: true, message: '请输入标题', trigger: 'blur' },
    { min: 2, max: 200, message: '标题长度 2-200 个字符', trigger: 'blur' },
  ],
  description: [
    { max: 2000, message: '描述不能超过 2000 个字符', trigger: 'blur' },
  ],
  category: [{ required: true, message: '请选择分类', trigger: 'change' }],
}

// 加载物品数据（编辑模式）
async function fetchItem() {
  if (!itemId.value) return

  loading.value = true
  try {
    const item = await api.adminItem.getItem(itemId.value)
    Object.assign(formData, {
      type: item.type,
      title: item.title,
      description: item.description || '',
      category: item.category || '',
      tags: item.tags || [],
    })
  } catch (error: any) {
    message.error(error.message || '加载物品信息失败')
    router.back()
  } finally {
    loading.value = false
  }
}

// 提交表单
async function handleSubmit() {
  try {
    await formRef.value?.validate()
  } catch {
    return
  }

  submitLoading.value = true

  try {
    if (isEdit.value && itemId.value) {
      // 更新物品
      const updateData: UpdateItemRequest = {
        title: formData.title,
        description: formData.description,
        category: formData.category,
        tags: formData.tags,
      }
      await api.adminItem.updateItem(itemId.value, updateData)
      message.success('更新成功')
    } else {
      // 创建物品
      const createData: CreateItemRequest = {
        type: formData.type!,
        title: formData.title,
        description: formData.description,
        category: formData.category,
        tags: formData.tags,
      }
      await api.adminItem.createItem(createData)
      message.success('创建成功')
    }
    router.push('/admin/items')
  } catch (error: any) {
    message.error(error.message || '提交失败')
  } finally {
    submitLoading.value = false
  }
}

// 返回列表
function handleBack() {
  router.push('/admin/items')
}

// 重置表单
function handleReset() {
  formRef.value?.resetFields()
  formData.tags = []
}

// 显示标签输入框
function showTagInput() {
  tagInputVisible.value = true
  setTimeout(() => {
    tagInputRef.value?.focus()
  }, 0)
}

// 处理标签输入确认
function handleTagInputConfirm() {
  if (tagInputValue.value && !formData.tags.includes(tagInputValue.value)) {
    formData.tags.push(tagInputValue.value)
  }
  tagInputVisible.value = false
  tagInputValue.value = ''
}

// 移除标签
function handleTagClose(removedTag: string) {
  formData.tags = formData.tags.filter(tag => tag !== removedTag)
}

onMounted(() => {
  if (isEdit.value) {
    fetchItem()
  }
})
</script>

<template>
  <div class="item-form-page">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-left">
        <Button type="text" class="back-btn" @click="handleBack">
          <ArrowLeftOutlined />
        </Button>
        <div class="header-info">
          <h2 class="page-title">{{ pageTitle }}</h2>
          <span class="page-desc">
            {{ isEdit ? '修改物品的基本信息' : '创建一个新的物品' }}
          </span>
        </div>
      </div>
    </div>

    <!-- 表单区域 -->
    <Spin :spinning="loading">
      <Card :bordered="false" class="form-card">
        <Form
          ref="formRef"
          :model="formData"
          :rules="rules"
          layout="vertical"
          class="item-form"
          @finish="handleSubmit"
        >
          <!-- 基本信息 -->
          <div class="form-section">
            <h3 class="section-title">基本信息</h3>
            <Divider />

            <div class="form-grid">
              <FormItem
                label="物品类型"
                name="type"
                class="form-item"
              >
                <Select
                  v-model:value="formData.type"
                  :disabled="isEdit"
                  placeholder="请选择物品类型"
                  :options="typeOptions"
                />
              </FormItem>

              <FormItem label="分类" name="category" class="form-item">
                <Select
                  v-model:value="formData.category"
                  placeholder="请选择分类"
                  :options="categoryOptions"
                  allow-clear
                />
              </FormItem>

              <FormItem
                label="标题"
                name="title"
                class="form-item full-width"
              >
                <Input
                  v-model:value="formData.title"
                  placeholder="请输入物品标题"
                  :maxlength="200"
                  show-count
                />
              </FormItem>

              <FormItem
                label="描述"
                name="description"
                class="form-item full-width"
              >
                <Input.TextArea
                  v-model:value="formData.description"
                  placeholder="请输入物品描述"
                  :rows="4"
                  :maxlength="2000"
                  show-count
                />
              </FormItem>
            </div>
          </div>

          <!-- 标签管理 -->
          <div class="form-section">
            <h3 class="section-title">标签</h3>
            <Divider />

            <div class="tags-container">
              <Space wrap>
                <Tag
                  v-for="tag in formData.tags"
                  :key="tag"
                  closable
                  color="blue"
                  @close="handleTagClose(tag)"
                >
                  {{ tag }}
                </Tag>
                <Input
                  v-if="tagInputVisible"
                  ref="tagInputRef"
                  v-model:value="tagInputValue"
                  type="text"
                  size="small"
                  style="width: 100px"
                  @blur="handleTagInputConfirm"
                  @keyup.enter="handleTagInputConfirm"
                />
                <Tag
                  v-else
                  class="add-tag"
                  @click="showTagInput"
                >
                  <PlusOutlined />
                  添加标签
                </Tag>
              </Space>
              <div class="tags-hint">
                最多可添加 10 个标签，每个标签不超过 20 个字符
              </div>
            </div>
          </div>

          <!-- 操作按钮 -->
          <div class="form-actions">
            <Space :size="16">
              <Button @click="handleBack">取消</Button>
              <Button @click="handleReset">重置</Button>
              <Button
                type="primary"
                html-type="submit"
                :loading="submitLoading"
              >
                <SaveOutlined />
                {{ isEdit ? '保存修改' : '创建物品' }}
              </Button>
            </Space>
          </div>
        </Form>
      </Card>
    </Spin>
  </div>
</template>

<style scoped>
.item-form-page {
  max-width: 900px;
}

.page-header {
  display: flex;
  align-items: center;
  margin-bottom: 24px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.back-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  padding: 0;
  font-size: 16px;
}

.header-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.page-title {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.85);
}

.page-desc {
  font-size: 14px;
  color: rgba(0, 0, 0, 0.45);
}

.form-card {
  background: #fff;
  border-radius: 8px;
}

.form-card :deep(.ant-card-body) {
  padding: 24px 32px;
}

.form-section {
  margin-bottom: 24px;
}

.section-title {
  margin: 0 0 8px;
  font-size: 16px;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.85);
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px 24px;
}

.form-item {
  margin-bottom: 0;
}

.form-item.full-width {
  grid-column: 1 / -1;
}

.tags-container {
  padding: 8px 0;
}

.add-tag {
  border-style: dashed;
  cursor: pointer;
  background: #fafafa;
}

.add-tag:hover {
  border-color: #1890ff;
  color: #1890ff;
}

.tags-hint {
  margin-top: 8px;
  font-size: 12px;
  color: rgba(0, 0, 0, 0.45);
}

.form-actions {
  display: flex;
  justify-content: flex-end;
  padding-top: 24px;
  border-top: 1px solid #f0f0f0;
}

@media (max-width: 768px) {
  .form-grid {
    grid-template-columns: 1fr;
  }

  .form-card :deep(.ant-card-body) {
    padding: 16px;
  }
}
</style>

