<script setup lang="ts">
/**
 * Settings - 系统设置页面
 * 
 * 占位页面，可根据需要扩展
 */
import { Card, Empty } from 'ant-design-vue'
import { SettingOutlined } from '@ant-design/icons-vue'
</script>

<template>
  <div class="settings-page">
    <div class="page-header">
      <h2 class="page-title">系统设置</h2>
      <span class="page-desc">管理系统配置和参数</span>
    </div>

    <Card :bordered="false">
      <Empty description="系统设置功能开发中..." />
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

