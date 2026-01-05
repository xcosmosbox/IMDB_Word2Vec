<script setup lang="ts">
/**
 * NotFound - 404 页面
 */
import { useRouter } from 'vue-router'
import { Button, Result } from 'ant-design-vue'
import { HomeOutlined, ArrowLeftOutlined } from '@ant-design/icons-vue'

const router = useRouter()

function goHome() {
  router.push('/admin/dashboard')
}

function goBack() {
  router.back()
}
</script>

<template>
  <div class="not-found-page">
    <Result
      status="404"
      title="404"
      sub-title="抱歉，您访问的页面不存在"
    >
      <template #extra>
        <Button type="primary" @click="goHome">
          <HomeOutlined />
          返回首页
        </Button>
        <Button @click="goBack">
          <ArrowLeftOutlined />
          返回上一页
        </Button>
      </template>
    </Result>
  </div>
</template>

<style scoped>
.not-found-page {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
}
</style>

