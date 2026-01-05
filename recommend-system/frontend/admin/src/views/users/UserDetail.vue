<script setup lang="ts">
/**
 * UserDetail - 用户详情页面
 * 
 * 功能：
 * - 展示用户详细信息
 * - 用户行为记录
 * - 编辑/删除操作
 */
import { ref, onMounted, inject } from 'vue'
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
  Timeline,
  TimelineItem,
} from 'ant-design-vue'
import {
  ArrowLeftOutlined,
  EditOutlined,
  DeleteOutlined,
  UserOutlined,
  MailOutlined,
  CalendarOutlined,
  ManOutlined,
  WomanOutlined,
} from '@ant-design/icons-vue'
import type { User, UserBehavior } from '@shared/types'
import type { IApiProvider } from '@shared/api/interfaces'

const route = useRoute()
const router = useRouter()

// 通过依赖注入获取 API Provider
const api = inject<IApiProvider>('api')!

const userId = route.params.id as string
const loading = ref(false)
const user = ref<User | null>(null)
const behaviors = ref<UserBehavior[]>([])

// 性别配置
const genderConfig: Record<string, { color: string; label: string; icon: any }> = {
  male: { color: 'blue', label: '男', icon: ManOutlined },
  female: { color: 'magenta', label: '女', icon: WomanOutlined },
  other: { color: 'default', label: '其他', icon: UserOutlined },
}

// 行为类型配置
const actionConfig: Record<string, { color: string; label: string }> = {
  view: { color: 'blue', label: '浏览' },
  click: { color: 'green', label: '点击' },
  like: { color: 'red', label: '喜欢' },
  dislike: { color: 'default', label: '不喜欢' },
  buy: { color: 'gold', label: '购买' },
  share: { color: 'purple', label: '分享' },
}

// 加载用户信息
async function fetchUser() {
  loading.value = true
  try {
    user.value = await api.adminUser.getUser(userId)
  } catch (error: any) {
    message.error(error.message || '加载用户信息失败')
    router.back()
  } finally {
    loading.value = false
  }
}

// 编辑用户
function handleEdit() {
  router.push(`/admin/users/${userId}/edit`)
}

// 删除用户
async function handleDelete() {
  try {
    await api.adminUser.deleteUser(userId)
    message.success('删除成功')
    router.push('/admin/users')
  } catch (error: any) {
    message.error(error.message || '删除失败')
  }
}

// 返回列表
function handleBack() {
  router.push('/admin/users')
}

// 格式化日期
function formatDate(dateStr: string) {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString('zh-CN')
}

// 格式化相对时间
function formatRelativeTime(dateStr: string) {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  const now = new Date()
  const diff = now.getTime() - date.getTime()
  const days = Math.floor(diff / (1000 * 60 * 60 * 24))
  const hours = Math.floor(diff / (1000 * 60 * 60))
  const minutes = Math.floor(diff / (1000 * 60))

  if (days > 0) return `${days}天前`
  if (hours > 0) return `${hours}小时前`
  if (minutes > 0) return `${minutes}分钟前`
  return '刚刚'
}

onMounted(() => {
  fetchUser()
})
</script>

<template>
  <div class="user-detail-page">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-left">
        <Button type="text" class="back-btn" @click="handleBack">
          <ArrowLeftOutlined />
        </Button>
        <div class="header-info">
          <h2 class="page-title">用户详情</h2>
          <span class="page-desc">查看用户的详细信息</span>
        </div>
      </div>
      <div class="header-right">
        <Space>
          <Button type="primary" @click="handleEdit">
            <EditOutlined />
            编辑
          </Button>
          <Popconfirm
            title="确定要删除这个用户吗？"
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
      <!-- 基本信息 -->
      <Card :bordered="false" class="info-card">
        <template #title>
          <div class="card-title">
            <UserOutlined />
            <span>基本信息</span>
          </div>
        </template>

        <template v-if="user">
          <div class="user-profile">
            <div class="user-avatar">
              {{ user.name?.[0] || 'U' }}
            </div>
            <div class="user-info">
              <h3 class="user-name">{{ user.name }}</h3>
              <div class="user-meta">
                <span class="meta-item">
                  <MailOutlined />
                  {{ user.email }}
                </span>
                <span v-if="user.gender" class="meta-item">
                  <component
                    :is="genderConfig[user.gender]?.icon || UserOutlined"
                  />
                  <Tag :color="genderConfig[user.gender]?.color">
                    {{ genderConfig[user.gender]?.label || user.gender }}
                  </Tag>
                </span>
                <span v-if="user.age" class="meta-item">
                  {{ user.age }} 岁
                </span>
              </div>
            </div>
          </div>

          <Divider />

          <Descriptions :column="2" :label-style="{ fontWeight: 500 }">
            <DescriptionsItem label="用户 ID">
              <code>{{ user.id }}</code>
            </DescriptionsItem>
            <DescriptionsItem label="邮箱">
              {{ user.email }}
            </DescriptionsItem>
            <DescriptionsItem label="年龄">
              {{ user.age || '-' }}
            </DescriptionsItem>
            <DescriptionsItem label="性别">
              <Tag
                v-if="user.gender"
                :color="genderConfig[user.gender]?.color"
              >
                {{ genderConfig[user.gender]?.label || user.gender }}
              </Tag>
              <span v-else>-</span>
            </DescriptionsItem>
            <DescriptionsItem label="创建时间">
              <CalendarOutlined style="margin-right: 8px" />
              {{ formatDate(user.created_at) }}
            </DescriptionsItem>
            <DescriptionsItem label="更新时间">
              <CalendarOutlined style="margin-right: 8px" />
              {{ formatDate(user.updated_at) }}
            </DescriptionsItem>
          </Descriptions>
        </template>
        <Empty v-else description="暂无用户信息" />
      </Card>

      <!-- 用户行为记录 -->
      <Card :bordered="false" class="info-card" style="margin-top: 16px">
        <template #title>
          <div class="card-title">
            <CalendarOutlined />
            <span>行为记录</span>
          </div>
        </template>

        <template v-if="behaviors.length > 0">
          <Timeline>
            <TimelineItem
              v-for="behavior in behaviors"
              :key="`${behavior.item_id}-${behavior.timestamp}`"
              :color="actionConfig[behavior.action]?.color || 'blue'"
            >
              <div class="behavior-item">
                <Tag :color="actionConfig[behavior.action]?.color">
                  {{ actionConfig[behavior.action]?.label || behavior.action }}
                </Tag>
                <span class="behavior-desc">
                  物品 ID: {{ behavior.item_id }}
                </span>
                <span class="behavior-time">
                  {{ formatRelativeTime(behavior.timestamp) }}
                </span>
              </div>
            </TimelineItem>
          </Timeline>
        </template>
        <Empty v-else description="暂无行为记录" />
      </Card>
    </Spin>
  </div>
</template>

<style scoped>
.user-detail-page {
  max-width: 900px;
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

.user-profile {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 16px 0;
}

.user-avatar {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 32px;
  font-weight: 600;
  color: #fff;
  flex-shrink: 0;
}

.user-info {
  flex: 1;
}

.user-name {
  margin: 0 0 8px;
  font-size: 24px;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.85);
}

.user-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  color: rgba(0, 0, 0, 0.65);
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.behavior-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.behavior-desc {
  color: rgba(0, 0, 0, 0.65);
}

.behavior-time {
  color: rgba(0, 0, 0, 0.45);
  font-size: 12px;
}

@media (max-width: 768px) {
  .page-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }

  .user-profile {
    flex-direction: column;
    text-align: center;
  }

  .user-meta {
    justify-content: center;
  }
}
</style>

