<script setup lang="ts">
/**
 * UserForm - 用户表单页面
 * 
 * 功能：
 * - 新增用户
 * - 编辑用户
 * - 表单验证
 */
import { ref, reactive, onMounted, computed, inject } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  Form,
  FormItem,
  Input,
  InputNumber,
  Select,
  Button,
  Card,
  Space,
  message,
  Spin,
  Divider,
} from 'ant-design-vue'
import { SaveOutlined, ArrowLeftOutlined } from '@ant-design/icons-vue'
import type { Rule } from 'ant-design-vue/es/form'
import type { User, CreateUserRequest, UpdateUserRequest } from '@shared/types'
import type { IApiProvider } from '@shared/api/interfaces'

const route = useRoute()
const router = useRouter()

// 通过依赖注入获取 API Provider
const api = inject<IApiProvider>('api')!

// 页面状态
const userId = computed(() => route.params.id as string | undefined)
const isEdit = computed(() => !!userId.value && route.path.includes('/edit'))
const isCreate = computed(() => route.path.includes('/create'))
const pageTitle = computed(() => (isEdit.value ? '编辑用户' : '新增用户'))

const loading = ref(false)
const submitLoading = ref(false)
const formRef = ref()

// 表单数据
interface FormData {
  name: string
  email: string
  password: string
  age: number | undefined
  gender: string | undefined
}

const formData = reactive<FormData>({
  name: '',
  email: '',
  password: '',
  age: undefined,
  gender: undefined,
})

// 表单校验规则
const rules: Record<string, Rule[]> = {
  name: [
    { required: true, message: '请输入姓名', trigger: 'blur' },
    { min: 2, max: 50, message: '姓名长度 2-50 个字符', trigger: 'blur' },
  ],
  email: [
    { required: true, message: '请输入邮箱', trigger: 'blur' },
    { type: 'email', message: '邮箱格式不正确', trigger: 'blur' },
  ],
  password: [
    {
      required: isCreate.value,
      message: '请输入密码',
      trigger: 'blur',
    },
    { min: 6, max: 32, message: '密码长度 6-32 个字符', trigger: 'blur' },
  ],
  age: [
    { type: 'number', min: 1, max: 150, message: '年龄范围 1-150', trigger: 'blur' },
  ],
}

// 加载用户数据（编辑模式）
async function fetchUser() {
  if (!userId.value) return

  loading.value = true
  try {
    const user = await api.adminUser.getUser(userId.value)
    Object.assign(formData, {
      name: user.name,
      email: user.email,
      age: user.age,
      gender: user.gender,
      password: '', // 编辑时不显示密码
    })
  } catch (error: any) {
    message.error(error.message || '加载用户信息失败')
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
    if (isEdit.value && userId.value) {
      // 更新用户
      const updateData: UpdateUserRequest = {
        name: formData.name,
        email: formData.email,
        age: formData.age,
        gender: formData.gender,
      }
      await api.adminUser.updateUser(userId.value, updateData)
      message.success('更新成功')
    } else {
      // 创建用户
      const createData: CreateUserRequest = {
        name: formData.name,
        email: formData.email,
        age: formData.age,
        gender: formData.gender,
      }
      await api.adminUser.createUser(createData)
      message.success('创建成功')
    }
    router.push('/admin/users')
  } catch (error: any) {
    message.error(error.message || '提交失败')
  } finally {
    submitLoading.value = false
  }
}

// 返回列表
function handleBack() {
  router.push('/admin/users')
}

// 重置表单
function handleReset() {
  formRef.value?.resetFields()
}

onMounted(() => {
  if (isEdit.value) {
    fetchUser()
  }
})
</script>

<template>
  <div class="user-form-page">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-left">
        <Button type="text" class="back-btn" @click="handleBack">
          <ArrowLeftOutlined />
        </Button>
        <div class="header-info">
          <h2 class="page-title">{{ pageTitle }}</h2>
          <span class="page-desc">
            {{ isEdit ? '修改用户的基本信息' : '创建一个新的用户账号' }}
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
          class="user-form"
          @finish="handleSubmit"
        >
          <div class="form-section">
            <h3 class="section-title">基本信息</h3>
            <Divider />

            <div class="form-grid">
              <FormItem label="姓名" name="name" class="form-item">
                <Input
                  v-model:value="formData.name"
                  placeholder="请输入用户姓名"
                  :maxlength="50"
                  show-count
                />
              </FormItem>

              <FormItem label="邮箱" name="email" class="form-item">
                <Input
                  v-model:value="formData.email"
                  placeholder="请输入邮箱地址"
                  type="email"
                />
              </FormItem>

              <FormItem
                v-if="isCreate"
                label="密码"
                name="password"
                class="form-item"
              >
                <Input.Password
                  v-model:value="formData.password"
                  placeholder="请输入登录密码"
                  :maxlength="32"
                />
              </FormItem>

              <FormItem label="年龄" name="age" class="form-item">
                <InputNumber
                  v-model:value="formData.age"
                  :min="1"
                  :max="150"
                  :precision="0"
                  placeholder="请输入年龄"
                  style="width: 100%"
                />
              </FormItem>

              <FormItem label="性别" name="gender" class="form-item">
                <Select
                  v-model:value="formData.gender"
                  placeholder="请选择性别"
                  allow-clear
                >
                  <Select.Option value="male">男</Select.Option>
                  <Select.Option value="female">女</Select.Option>
                  <Select.Option value="other">其他</Select.Option>
                </Select>
              </FormItem>
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
                {{ isEdit ? '保存修改' : '创建用户' }}
              </Button>
            </Space>
          </div>
        </Form>
      </Card>
    </Spin>
  </div>
</template>

<style scoped>
.user-form-page {
  max-width: 800px;
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

