<script setup lang="ts">
/**
 * ItemList - 物品列表页面
 * 
 * 功能：
 * - 物品列表展示
 * - 分页搜索筛选
 * - 新增/编辑/删除操作
 */
import { ref, reactive, onMounted, h, inject } from 'vue'
import { useRouter } from 'vue-router'
import {
  Button,
  Space,
  Input,
  Select,
  Tag,
  Popconfirm,
  message,
  Card,
  Tooltip,
  Switch,
} from 'ant-design-vue'
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  SearchOutlined,
  EyeOutlined,
  ReloadOutlined,
} from '@ant-design/icons-vue'
import type { Item } from '@shared/types'
import type { IApiProvider } from '@shared/api/interfaces'
import DataTable from '@/components/DataTable.vue'

const router = useRouter()

// 通过依赖注入获取 API Provider
const api = inject<IApiProvider>('api')!

// 表格数据
const loading = ref(false)
const data = ref<Item[]>([])
const pagination = reactive({
  current: 1,
  pageSize: 10,
  total: 0,
})

// 搜索条件
const searchForm = reactive({
  keyword: '',
  type: undefined as string | undefined,
  status: undefined as string | undefined,
})

// 类型配置
const typeConfig: Record<string, { color: string; label: string }> = {
  movie: { color: 'red', label: '电影' },
  product: { color: 'orange', label: '商品' },
  article: { color: 'blue', label: '文章' },
  video: { color: 'purple', label: '视频' },
}

// 状态配置
const statusConfig: Record<string, { color: string; label: string }> = {
  active: { color: 'success', label: '上架' },
  inactive: { color: 'default', label: '下架' },
}

// 表格列定义
const columns = [
  {
    title: 'ID',
    dataIndex: 'id',
    width: 120,
    ellipsis: true,
  },
  {
    title: '标题',
    dataIndex: 'title',
    width: 250,
    ellipsis: true,
  },
  {
    title: '类型',
    dataIndex: 'type',
    width: 100,
    align: 'center' as const,
    customRender: ({ text }: { text: string }) => {
      const config = typeConfig[text]
      if (config) {
        return h(Tag, { color: config.color }, () => config.label)
      }
      return text || '-'
    },
  },
  {
    title: '分类',
    dataIndex: 'category',
    width: 120,
    ellipsis: true,
    customRender: ({ text }: { text: string }) => text || '-',
  },
  {
    title: '标签',
    dataIndex: 'tags',
    width: 200,
    customRender: ({ text }: { text: string[] }) => {
      if (!text || text.length === 0) return '-'
      const displayTags = text.slice(0, 3)
      const hasMore = text.length > 3

      return h(Space, { size: 4, wrap: true }, () => [
        ...displayTags.map(tag => h(Tag, { key: tag }, () => tag)),
        hasMore ? h(Tag, { color: 'default' }, () => `+${text.length - 3}`) : null,
      ])
    },
  },
  {
    title: '状态',
    dataIndex: 'status',
    width: 100,
    align: 'center' as const,
    customRender: ({ text }: { text: string }) => {
      const config = statusConfig[text]
      if (config) {
        return h(Tag, { color: config.color }, () => config.label)
      }
      return text || '-'
    },
  },
  {
    title: '创建时间',
    dataIndex: 'created_at',
    width: 180,
    customRender: ({ text }: { text: string }) => {
      if (!text) return '-'
      return new Date(text).toLocaleString('zh-CN')
    },
  },
  {
    title: '操作',
    key: 'action',
    width: 200,
    fixed: 'right' as const,
  },
]

// 加载数据
async function fetchData() {
  loading.value = true
  try {
    const response = await api.adminItem.listItems({
      page: pagination.current,
      page_size: pagination.pageSize,
      keyword: searchForm.keyword || undefined,
      type: searchForm.type,
    })
    data.value = response.items
    pagination.total = response.total
  } catch (error: any) {
    message.error(error.message || '加载物品列表失败')
  } finally {
    loading.value = false
  }
}

// 翻页
function handlePageChange(page: number, pageSize: number) {
  pagination.current = page
  pagination.pageSize = pageSize
  fetchData()
}

// 搜索
function handleSearch() {
  pagination.current = 1
  fetchData()
}

// 重置
function handleReset() {
  searchForm.keyword = ''
  searchForm.type = undefined
  searchForm.status = undefined
  pagination.current = 1
  fetchData()
}

// 新增
function handleAdd() {
  router.push('/admin/items/create')
}

// 查看详情
function handleView(record: Item) {
  router.push(`/admin/items/${record.id}`)
}

// 编辑
function handleEdit(record: Item) {
  router.push(`/admin/items/${record.id}/edit`)
}

// 删除
async function handleDelete(record: Item) {
  try {
    await api.adminItem.deleteItem(record.id)
    message.success('删除成功')
    // 如果当前页只有一条数据且不是第一页，则返回上一页
    if (data.value.length === 1 && pagination.current > 1) {
      pagination.current--
    }
    fetchData()
  } catch (error: any) {
    message.error(error.message || '删除失败')
  }
}

// 刷新
function handleRefresh() {
  fetchData()
}

onMounted(() => {
  fetchData()
})
</script>

<template>
  <div class="item-list-page">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-left">
        <h2 class="page-title">物品管理</h2>
        <span class="page-desc">管理系统中的所有物品内容</span>
      </div>
      <div class="header-right">
        <Space>
          <Button @click="handleRefresh">
            <ReloadOutlined />
            刷新
          </Button>
          <Button type="primary" @click="handleAdd">
            <PlusOutlined />
            新增物品
          </Button>
        </Space>
      </div>
    </div>

    <!-- 搜索表单 -->
    <Card class="search-card" :bordered="false">
      <div class="search-form">
        <Space :size="16" wrap>
          <div class="form-item">
            <span class="form-label">关键词</span>
            <Input
              v-model:value="searchForm.keyword"
              placeholder="搜索标题"
              allow-clear
              style="width: 200px"
              @press-enter="handleSearch"
            >
              <template #prefix>
                <SearchOutlined />
              </template>
            </Input>
          </div>

          <div class="form-item">
            <span class="form-label">类型</span>
            <Select
              v-model:value="searchForm.type"
              placeholder="全部"
              allow-clear
              style="width: 120px"
            >
              <Select.Option value="movie">电影</Select.Option>
              <Select.Option value="product">商品</Select.Option>
              <Select.Option value="article">文章</Select.Option>
              <Select.Option value="video">视频</Select.Option>
            </Select>
          </div>

          <div class="form-item">
            <span class="form-label">状态</span>
            <Select
              v-model:value="searchForm.status"
              placeholder="全部"
              allow-clear
              style="width: 120px"
            >
              <Select.Option value="active">上架</Select.Option>
              <Select.Option value="inactive">下架</Select.Option>
            </Select>
          </div>

          <div class="form-actions">
            <Space>
              <Button type="primary" @click="handleSearch">
                <SearchOutlined />
                搜索
              </Button>
              <Button @click="handleReset">重置</Button>
            </Space>
          </div>
        </Space>
      </div>
    </Card>

    <!-- 数据表格 -->
    <Card class="table-card" :bordered="false">
      <DataTable
        :columns="columns"
        :data-source="data"
        :loading="loading"
        :pagination="pagination"
        :scroll-x="1300"
        row-key="id"
        @page-change="handlePageChange"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'action'">
            <Space>
              <Tooltip title="查看详情">
                <Button type="link" size="small" @click="handleView(record)">
                  <EyeOutlined />
                </Button>
              </Tooltip>
              <Tooltip title="编辑">
                <Button type="link" size="small" @click="handleEdit(record)">
                  <EditOutlined />
                </Button>
              </Tooltip>
              <Popconfirm
                title="确定要删除这个物品吗？"
                description="删除后将无法恢复"
                ok-text="确定"
                cancel-text="取消"
                placement="topRight"
                @confirm="handleDelete(record)"
              >
                <Tooltip title="删除">
                  <Button type="link" danger size="small">
                    <DeleteOutlined />
                  </Button>
                </Tooltip>
              </Popconfirm>
            </Space>
          </template>
        </template>
      </DataTable>
    </Card>
  </div>
</template>

<style scoped>
.item-list-page {
  padding: 0;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.header-left {
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

.search-card {
  margin-bottom: 16px;
  background: #fafafa;
}

.search-card :deep(.ant-card-body) {
  padding: 16px 24px;
}

.search-form {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
}

.form-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.form-label {
  color: rgba(0, 0, 0, 0.65);
  white-space: nowrap;
}

.form-actions {
  margin-left: auto;
}

.table-card :deep(.ant-card-body) {
  padding: 0;
}

@media (max-width: 768px) {
  .page-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }

  .form-actions {
    margin-left: 0;
    margin-top: 8px;
  }
}
</style>

