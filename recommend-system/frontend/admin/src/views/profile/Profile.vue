<script setup lang="ts">
/**
 * Profile - 个人信息页面
 */
import { ref, reactive } from 'vue'
import {
  Card,
  Form,
  FormItem,
  Input,
  Button,
  Space,
  Avatar,
  Divider,
  message,
} from 'ant-design-vue'
import { UserOutlined, SaveOutlined, LockOutlined } from '@ant-design/icons-vue'
import { useAdminStore } from '@/stores/admin'

const adminStore = useAdminStore()

const loading = ref(false)

const formData = reactive({
  name: adminStore.currentAdmin?.name || '',
  email: adminStore.currentAdmin?.email || '',
})

const passwordForm = reactive({
  oldPassword: '',
  newPassword: '',
  confirmPassword: '',
})

async function handleUpdateProfile() {
  loading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 500))
    adminStore.updateAdminInfo({
      name: formData.name,
      email: formData.email,
    })
    message.success('更新成功')
  } finally {
    loading.value = false
  }
}

async function handleChangePassword() {
  if (passwordForm.newPassword !== passwordForm.confirmPassword) {
    message.error('两次密码输入不一致')
    return
  }

  loading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 500))
    message.success('密码修改成功')
    passwordForm.oldPassword = ''
    passwordForm.newPassword = ''
    passwordForm.confirmPassword = ''
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="profile-page">
    <div class="page-header">
      <h2 class="page-title">个人信息</h2>
      <span class="page-desc">管理您的账号信息</span>
    </div>

    <!-- 头像和基本信息 -->
    <Card :bordered="false" class="profile-card">
      <div class="profile-header">
        <Avatar :size="80" class="user-avatar">
          {{ adminStore.currentAdmin?.name?.[0] || 'A' }}
        </Avatar>
        <div class="profile-info">
          <h3>{{ adminStore.currentAdmin?.name || '管理员' }}</h3>
          <p>{{ adminStore.currentAdmin?.email || 'admin@example.com' }}</p>
          <p class="role">
            角色: {{ adminStore.currentAdmin?.role === 'super_admin' ? '超级管理员' : '管理员' }}
          </p>
        </div>
      </div>

      <Divider />

      <Form
        :model="formData"
        layout="vertical"
        class="profile-form"
      >
        <FormItem label="姓名">
          <Input
            v-model:value="formData.name"
            placeholder="请输入姓名"
            :prefix="h(UserOutlined)"
          />
        </FormItem>
        <FormItem label="邮箱">
          <Input
            v-model:value="formData.email"
            placeholder="请输入邮箱"
            type="email"
          />
        </FormItem>
        <FormItem>
          <Button type="primary" :loading="loading" @click="handleUpdateProfile">
            <SaveOutlined />
            保存修改
          </Button>
        </FormItem>
      </Form>
    </Card>

    <!-- 修改密码 -->
    <Card :bordered="false" class="profile-card" style="margin-top: 16px">
      <template #title>
        <div class="card-title">
          <LockOutlined />
          <span>修改密码</span>
        </div>
      </template>

      <Form
        :model="passwordForm"
        layout="vertical"
        class="profile-form"
      >
        <FormItem label="当前密码">
          <Input.Password
            v-model:value="passwordForm.oldPassword"
            placeholder="请输入当前密码"
          />
        </FormItem>
        <FormItem label="新密码">
          <Input.Password
            v-model:value="passwordForm.newPassword"
            placeholder="请输入新密码"
          />
        </FormItem>
        <FormItem label="确认密码">
          <Input.Password
            v-model:value="passwordForm.confirmPassword"
            placeholder="请再次输入新密码"
          />
        </FormItem>
        <FormItem>
          <Button type="primary" :loading="loading" @click="handleChangePassword">
            修改密码
          </Button>
        </FormItem>
      </Form>
    </Card>
  </div>
</template>

<script lang="ts">
import { h } from 'vue'
</script>

<style scoped>
.profile-page {
  max-width: 600px;
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

.profile-card {
  border-radius: 8px;
}

.profile-header {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 16px 0;
}

.user-avatar {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  font-size: 32px;
}

.profile-info h3 {
  margin: 0 0 4px;
  font-size: 20px;
  font-weight: 600;
}

.profile-info p {
  margin: 0;
  color: rgba(0, 0, 0, 0.65);
}

.profile-info .role {
  margin-top: 4px;
  font-size: 12px;
  color: rgba(0, 0, 0, 0.45);
}

.profile-form {
  max-width: 400px;
}

.card-title {
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>

