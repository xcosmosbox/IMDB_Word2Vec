<script setup lang="ts">
/**
 * ItemDetail - 物品详情页面
 * 
 * 功能：
 * - 展示物品详细信息
 * - 物品统计数据
 * - 编辑/删除操作
 */
import { ref, onMounted, inject, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  Card,
  Descriptions,
  DescriptionsItem,
  Tag,
  Button,
  Space,
  Spin,
  message,
  Popconfirm,
  Divider,
  Empty,
  Statistic,
  Row,
  Col,
} from 'ant-design-vue'
import {
  ArrowLeftOutlined,
  EditOutlined,
  DeleteOutlined,
  ShoppingOutlined,
  CalendarOutlined,
  EyeOutlined,
  LikeOutlined,
  ShareAltOutlined,
  TagOutlined,
} from '@ant-design/icons-vue'
import type { Item, ItemStats } from '@shared/types'
import type { IApiProvider } from '@shared/api/interfaces'

const route = useRoute()
const router = useRouter()

// 通过依赖注入获取 API Provider
const api = inject<IApiProvider>('api')!

const itemId = route.params.id as string
const loading = ref(false)
const item = ref<Item | null>(null)

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

// 模拟统计数据
const stats = computed(() => ({
  view_count: Math.floor(Math.random() * 10000),
  click_count: Math.floor(Math.random() * 5000),
  like_count: Math.floor(Math.random() * 1000),
  share_count: Math.floor(Math.random() * 500),
}))

// 加载物品信息
async function fetchItem() {
  loading.value = true
  try {
    item.value = await api.adminItem.getItem(itemId)
  } catch (error: any) {
    message.error(error.message || '加载物品信息失败')
    router.back()
  } finally {
    loading.value = false
  }
}

// 编辑物品
function handleEdit() {
  router.push(`/admin/items/${itemId}/edit`)
}

// 删除物品
async function handleDelete() {
  try {
    await api.adminItem.deleteItem(itemId)
    message.success('删除成功')
    router.push('/admin/items')
  } catch (error: any) {
    message.error(error.message || '删除失败')
  }
}

// 返回列表
function handleBack() {
  router.push('/admin/items')
}

// 格式化日期
function formatDate(dateStr: string) {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString('zh-CN')
}

onMounted(() => {
  fetchItem()
})
</script>

<template>
  <div class="item-detail-page">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-left">
        <Button type="text" class="back-btn" @click="handleBack">
          <ArrowLeftOutlined />
        </Button>
        <div class="header-info">
          <h2 class="page-title">物品详情</h2>
          <span class="page-desc">查看物品的详细信息</span>
        </div>
      </div>
      <div class="header-right">
        <Space>
          <Button type="primary" @click="handleEdit">
            <EditOutlined />
            编辑
          </Button>
          <Popconfirm
            title="确定要删除这个物品吗？"
            description="删除后将无法恢复"
            ok-text="确定"
            cancel-text="取消"
            @confirm="handleDelete"
          >
            <Button danger>
              <DeleteOutlined />
              删除
            </Button>
          </Popconfirm>
        </Space>
      </div>
    </div>

    <Spin :spinning="loading">
      <template v-if="item">
        <!-- 统计数据 -->
        <Card :bordered="false" class="stats-card">
          <Row :gutter="24">
            <Col :span="6">
              <Statistic
                title="浏览量"
                :value="stats.view_count"
                :value-style="{ color: '#1890ff' }"
              >
                <template #prefix>
                  <EyeOutlined />
                </template>
              </Statistic>
            </Col>
            <Col :span="6">
              <Statistic
                title="点击量"
                :value="stats.click_count"
                :value-style="{ color: '#52c41a' }"
              >
                <template #prefix>
                  <ShoppingOutlined />
                </template>
              </Statistic>
            </Col>
            <Col :span="6">
              <Statistic
                title="点赞数"
                :value="stats.like_count"
                :value-style="{ color: '#eb2f96' }"
              >
                <template #prefix>
                  <LikeOutlined />
                </template>
              </Statistic>
            </Col>
            <Col :span="6">
              <Statistic
                title="分享数"
                :value="stats.share_count"
                :value-style="{ color: '#722ed1' }"
              >
                <template #prefix>
                  <ShareAltOutlined />
                </template>
              </Statistic>
            </Col>
          </Row>
        </Card>

        <!-- 基本信息 -->
        <Card :bordered="false" class="info-card">
          <template #title>
            <div class="card-title">
              <ShoppingOutlined />
              <span>基本信息</span>
            </div>
          </template>

          <div class="item-header">
            <div class="item-info">
              <h3 class="item-title">{{ item.title }}</h3>
              <div class="item-meta">
                <Tag :color="typeConfig[item.type]?.color">
                  {{ typeConfig[item.type]?.label || item.type }}
                </Tag>
                <Tag :color="statusConfig[item.status]?.color">
                  {{ statusConfig[item.status]?.label || item.status }}
                </Tag>
                <span v-if="item.category" class="category">
                  {{ item.category }}
                </span>
              </div>
            </div>
          </div>

          <Divider />

          <Descriptions :column="2" :label-style="{ fontWeight: 500 }">
            <DescriptionsItem label="物品 ID">
              <code>{{ item.id }}</code>
            </DescriptionsItem>
            <DescriptionsItem label="物品类型">
              <Tag :color="typeConfig[item.type]?.color">
                {{ typeConfig[item.type]?.label || item.type }}
              </Tag>
            </DescriptionsItem>
            <DescriptionsItem label="分类">
              {{ item.category || '-' }}
            </DescriptionsItem>
            <DescriptionsItem label="状态">
              <Tag :color="statusConfig[item.status]?.color">
                {{ statusConfig[item.status]?.label || item.status }}
              </Tag>
            </DescriptionsItem>
            <DescriptionsItem label="创建时间">
              <CalendarOutlined style="margin-right: 8px" />
              {{ formatDate(item.created_at) }}
            </DescriptionsItem>
            <DescriptionsItem label="更新时间">
              <CalendarOutlined style="margin-right: 8px" />
              {{ formatDate(item.updated_at) }}
            </DescriptionsItem>
            <DescriptionsItem label="描述" :span="2">
              {{ item.description || '暂无描述' }}
            </DescriptionsItem>
          </Descriptions>
        </Card>

        <!-- 标签信息 -->
        <Card :bordered="false" class="info-card" style="margin-top: 16px">
          <template #title>
            <div class="card-title">
              <TagOutlined />
              <span>标签</span>
            </div>
          </template>

          <div v-if="item.tags && item.tags.length > 0" class="tags-container">
            <Space wrap>
              <Tag
                v-for="tag in item.tags"
                :key="tag"
                color="blue"
              >
                {{ tag }}
              </Tag>
            </Space>
          </div>
          <Empty v-else description="暂无标签" :image="Empty.PRESENTED_IMAGE_SIMPLE" />
        </Card>
      </template>

      <Empty v-else-if="!loading" description="暂无物品信息" />
    </Spin>
  </div>
</template>

<style scoped>
.item-detail-page {
  max-width: 1000px;
}

.page-header {
  display: flex;
  justify-content: space-between;
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

.stats-card {
  margin-bottom: 16px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 8px;
}

.stats-card :deep(.ant-card-body) {
  padding: 24px;
}

.stats-card :deep(.ant-statistic-title) {
  color: rgba(255, 255, 255, 0.85);
}

.stats-card :deep(.ant-statistic-content) {
  color: #fff;
}

.stats-card :deep(.ant-statistic-content-prefix) {
  margin-right: 8px;
}

.info-card {
  background: #fff;
  border-radius: 8px;
}

.card-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
}

.item-header {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 16px 0;
}

.item-info {
  flex: 1;
}

.item-title {
  margin: 0 0 12px;
  font-size: 24px;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.85);
}

.item-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.category {
  color: rgba(0, 0, 0, 0.65);
  font-size: 14px;
}

.tags-container {
  padding: 8px 0;
}

@media (max-width: 768px) {
  .page-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }

  .stats-card :deep(.ant-col) {
    margin-bottom: 16px;
  }

  .stats-card :deep(.ant-col:last-child) {
    margin-bottom: 0;
  }
}
</style>

