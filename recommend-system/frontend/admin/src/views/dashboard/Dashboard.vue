<script setup lang="ts">
/**
 * Dashboard - 仪表盘页面
 * 
 * 展示系统概览和统计数据
 */
import { ref, onMounted } from 'vue'
import { Card, Row, Col, Statistic, Skeleton } from 'ant-design-vue'
import {
  UserOutlined,
  ShoppingOutlined,
  LineChartOutlined,
  ClockCircleOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
} from '@ant-design/icons-vue'

const loading = ref(true)

// 模拟统计数据
const stats = ref({
  totalUsers: 0,
  totalItems: 0,
  todayRecommendations: 0,
  avgResponseTime: 0,
})

// 模拟趋势数据
const trends = ref({
  usersChange: 12.5,
  itemsChange: 8.3,
  recommendationsChange: 25.8,
  responseTimeChange: -15.2,
})

async function fetchStats() {
  loading.value = true
  try {
    // 模拟 API 请求
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    stats.value = {
      totalUsers: 12580,
      totalItems: 45680,
      todayRecommendations: 158900,
      avgResponseTime: 42,
    }
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchStats()
})
</script>

<template>
  <div class="dashboard-page">
    <div class="page-header">
      <h2 class="page-title">仪表盘</h2>
      <span class="page-desc">欢迎回来，查看系统概览</span>
    </div>

    <!-- 统计卡片 -->
    <Row :gutter="[16, 16]" class="stats-row">
      <Col :xs="24" :sm="12" :lg="6">
        <Card class="stat-card users" :bordered="false">
          <Skeleton :loading="loading" active :paragraph="{ rows: 1 }">
            <div class="stat-content">
              <div class="stat-icon">
                <UserOutlined />
              </div>
              <div class="stat-info">
                <Statistic
                  title="总用户数"
                  :value="stats.totalUsers"
                  :value-style="{ fontWeight: 600 }"
                />
                <div class="stat-trend positive">
                  <ArrowUpOutlined />
                  <span>{{ trends.usersChange }}%</span>
                  <span class="trend-text">较上周</span>
                </div>
              </div>
            </div>
          </Skeleton>
        </Card>
      </Col>

      <Col :xs="24" :sm="12" :lg="6">
        <Card class="stat-card items" :bordered="false">
          <Skeleton :loading="loading" active :paragraph="{ rows: 1 }">
            <div class="stat-content">
              <div class="stat-icon">
                <ShoppingOutlined />
              </div>
              <div class="stat-info">
                <Statistic
                  title="总物品数"
                  :value="stats.totalItems"
                  :value-style="{ fontWeight: 600 }"
                />
                <div class="stat-trend positive">
                  <ArrowUpOutlined />
                  <span>{{ trends.itemsChange }}%</span>
                  <span class="trend-text">较上周</span>
                </div>
              </div>
            </div>
          </Skeleton>
        </Card>
      </Col>

      <Col :xs="24" :sm="12" :lg="6">
        <Card class="stat-card recommendations" :bordered="false">
          <Skeleton :loading="loading" active :paragraph="{ rows: 1 }">
            <div class="stat-content">
              <div class="stat-icon">
                <LineChartOutlined />
              </div>
              <div class="stat-info">
                <Statistic
                  title="今日推荐数"
                  :value="stats.todayRecommendations"
                  :value-style="{ fontWeight: 600 }"
                />
                <div class="stat-trend positive">
                  <ArrowUpOutlined />
                  <span>{{ trends.recommendationsChange }}%</span>
                  <span class="trend-text">较昨日</span>
                </div>
              </div>
            </div>
          </Skeleton>
        </Card>
      </Col>

      <Col :xs="24" :sm="12" :lg="6">
        <Card class="stat-card response-time" :bordered="false">
          <Skeleton :loading="loading" active :paragraph="{ rows: 1 }">
            <div class="stat-content">
              <div class="stat-icon">
                <ClockCircleOutlined />
              </div>
              <div class="stat-info">
                <Statistic
                  title="平均响应时间"
                  :value="stats.avgResponseTime"
                  suffix="ms"
                  :value-style="{ fontWeight: 600 }"
                />
                <div class="stat-trend negative">
                  <ArrowDownOutlined />
                  <span>{{ Math.abs(trends.responseTimeChange) }}%</span>
                  <span class="trend-text">较上周</span>
                </div>
              </div>
            </div>
          </Skeleton>
        </Card>
      </Col>
    </Row>

    <!-- 快捷入口 -->
    <Card title="快捷操作" :bordered="false" class="quick-actions">
      <Row :gutter="[16, 16]">
        <Col :xs="12" :sm="6">
          <router-link to="/admin/users/create" class="action-item">
            <div class="action-icon users">
              <UserOutlined />
            </div>
            <span>新增用户</span>
          </router-link>
        </Col>
        <Col :xs="12" :sm="6">
          <router-link to="/admin/items/create" class="action-item">
            <div class="action-icon items">
              <ShoppingOutlined />
            </div>
            <span>新增物品</span>
          </router-link>
        </Col>
        <Col :xs="12" :sm="6">
          <router-link to="/admin/users" class="action-item">
            <div class="action-icon list">
              <UserOutlined />
            </div>
            <span>用户列表</span>
          </router-link>
        </Col>
        <Col :xs="12" :sm="6">
          <router-link to="/admin/items" class="action-item">
            <div class="action-icon list">
              <ShoppingOutlined />
            </div>
            <span>物品列表</span>
          </router-link>
        </Col>
      </Row>
    </Card>
  </div>
</template>

<style scoped>
.dashboard-page {
  padding: 0;
}

.page-header {
  margin-bottom: 24px;
}

.page-title {
  margin: 0 0 4px;
  font-size: 20px;
  font-weight: 600;
}

.page-desc {
  color: rgba(0, 0, 0, 0.45);
}

.stats-row {
  margin-bottom: 24px;
}

.stat-card {
  border-radius: 8px;
  overflow: hidden;
}

.stat-card.users {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.stat-card.items {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.stat-card.recommendations {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

.stat-card.response-time {
  background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
}

.stat-card :deep(.ant-card-body) {
  padding: 20px;
}

.stat-content {
  display: flex;
  align-items: center;
  gap: 16px;
}

.stat-icon {
  width: 56px;
  height: 56px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: #fff;
}

.stat-info {
  flex: 1;
}

.stat-info :deep(.ant-statistic-title) {
  color: rgba(255, 255, 255, 0.85);
  font-size: 14px;
}

.stat-info :deep(.ant-statistic-content) {
  color: #fff;
}

.stat-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-top: 8px;
  font-size: 12px;
  color: rgba(255, 255, 255, 0.85);
}

.stat-trend.positive {
  color: #95f204;
}

.stat-trend.negative {
  color: #95f204;
}

.trend-text {
  opacity: 0.65;
}

.quick-actions {
  border-radius: 8px;
}

.action-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 16px;
  border-radius: 8px;
  background: #f5f5f5;
  transition: all 0.3s;
  color: rgba(0, 0, 0, 0.85);
  text-decoration: none;
}

.action-item:hover {
  background: #e6f7ff;
  color: #1890ff;
}

.action-icon {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  color: #fff;
}

.action-icon.users {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.action-icon.items {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.action-icon.list {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}
</style>

