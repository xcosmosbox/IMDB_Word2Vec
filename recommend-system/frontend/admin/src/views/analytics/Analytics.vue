<script setup lang="ts">
/**
 * Analytics - 数据分析页面
 * 
 * 占位页面，由 Person E 实现
 */
import { Card, Empty } from 'ant-design-vue'
import { BarChartOutlined } from '@ant-design/icons-vue'
</script>

<template>
  <div class="analytics-page">
    <div class="page-header">
      <h2 class="page-title">数据分析</h2>
      <span class="page-desc">查看系统数据和分析报告</span>
    </div>

    <Card :bordered="false">
      <Empty description="数据分析功能由 Person E 实现..." />
    </Card>
  </div>
</template>

<style scoped>
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
</style>

