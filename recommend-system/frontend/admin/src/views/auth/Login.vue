<script setup lang="ts">
/**
 * Login - 管理员登录页面
 */
import { ref, reactive } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { Form, FormItem, Input, Button, Card, Checkbox, message } from 'ant-design-vue'
import { UserOutlined, LockOutlined, LoginOutlined } from '@ant-design/icons-vue'
import { useAdminStore } from '@/stores/admin'

const router = useRouter()
const route = useRoute()
const adminStore = useAdminStore()

const loading = ref(false)
const formRef = ref()

const formData = reactive({
  email: '',
  password: '',
  remember: true,
})

const rules = {
  email: [
    { required: true, message: '请输入邮箱' },
    { type: 'email', message: '邮箱格式不正确' },
  ],
  password: [
    { required: true, message: '请输入密码' },
    { min: 6, message: '密码至少 6 位' },
  ],
}

async function handleSubmit() {
  try {
    await formRef.value?.validate()
  } catch {
    return
  }

  loading.value = true

  try {
    await adminStore.login({
      email: formData.email,
      password: formData.password,
    })

    message.success('登录成功')

    // 跳转到之前的页面或首页
    const redirect = (route.query.redirect as string) || '/admin/dashboard'
    router.push(redirect)
  } catch (error: any) {
    message.error(error.message || '登录失败')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <div class="login-container">
      <div class="login-header">
        <div class="logo">
          <LoginOutlined class="logo-icon" />
        </div>
        <h1 class="title">推荐系统管理后台</h1>
        <p class="subtitle">管理员登录</p>
      </div>

      <Card class="login-card" :bordered="false">
        <Form
          ref="formRef"
          :model="formData"
          :rules="rules"
          layout="vertical"
          @finish="handleSubmit"
        >
          <FormItem name="email">
            <Input
              v-model:value="formData.email"
              size="large"
              placeholder="请输入邮箱"
            >
              <template #prefix>
                <UserOutlined />
              </template>
            </Input>
          </FormItem>

          <FormItem name="password">
            <Input.Password
              v-model:value="formData.password"
              size="large"
              placeholder="请输入密码"
            >
              <template #prefix>
                <LockOutlined />
              </template>
            </Input.Password>
          </FormItem>

          <FormItem>
            <div class="login-options">
              <Checkbox v-model:checked="formData.remember">
                记住我
              </Checkbox>
              <a class="forgot-link">忘记密码？</a>
            </div>
          </FormItem>

          <FormItem>
            <Button
              type="primary"
              size="large"
              html-type="submit"
              :loading="loading"
              block
            >
              登录
            </Button>
          </FormItem>
        </Form>
      </Card>

      <div class="login-footer">
        <p>© 2025 推荐系统 - 版权所有</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 24px;
}

.login-container {
  width: 100%;
  max-width: 400px;
}

.login-header {
  text-align: center;
  margin-bottom: 32px;
}

.logo {
  width: 64px;
  height: 64px;
  margin: 0 auto 16px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.logo-icon {
  font-size: 32px;
  color: #fff;
}

.title {
  margin: 0 0 8px;
  font-size: 24px;
  font-weight: 600;
  color: #fff;
}

.subtitle {
  margin: 0;
  font-size: 14px;
  color: rgba(255, 255, 255, 0.85);
}

.login-card {
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.login-card :deep(.ant-card-body) {
  padding: 32px;
}

.login-options {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.forgot-link {
  color: #1890ff;
}

.login-footer {
  text-align: center;
  margin-top: 24px;
  color: rgba(255, 255, 255, 0.65);
  font-size: 12px;
}
</style>

