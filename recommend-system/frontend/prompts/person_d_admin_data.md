# Person D: 管理后台 - 数据管理

## 你的角色
你是一名前端工程师，负责实现生成式推荐系统的 **管理后台数据管理** 模块，包括用户管理、物品管理、CRUD 操作等。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读以下文件：**

1. **数据类型定义：**
```
frontend/shared/types/index.ts
```

2. **服务接口定义（核心）：**
```
frontend/shared/api/interfaces.ts
```

你需要使用的核心接口：

```typescript
// 管理员用户服务接口
interface IAdminUserService {
  listUsers(params: { page, page_size, keyword?, gender? }): Promise<{ items: User[]; total: number }>
  getUser(userId: string): Promise<User>
  createUser(data: CreateUserRequest): Promise<User>
  updateUser(userId: string, data: UpdateUserRequest): Promise<User>
  deleteUser(userId: string): Promise<void>
}

// 管理员物品服务接口
interface IAdminItemService {
  listItems(params: { page, page_size, type?, keyword? }): Promise<{ items: Item[]; total: number }>
  getItem(itemId: string): Promise<Item>
  createItem(data: CreateItemRequest): Promise<Item>
  updateItem(itemId: string, data: UpdateItemRequest): Promise<Item>
  deleteItem(itemId: string): Promise<void>
}
```

**⚠️ 不要直接导入具体实现！** 使用依赖注入：

```typescript
// ✅ 正确：通过 inject 获取接口
const api = inject<IApiProvider>('api')!
const { items, total } = await api.adminUser.listUsers({ page: 1, page_size: 10 })
await api.adminItem.createItem(data)

// ❌ 错误：直接导入具体实现
import { adminUserApi, adminItemApi } from '@/api/admin'
```

---

## 技术栈

- **框架**: Vue 3 + Composition API + TypeScript
- **构建**: Vite
- **UI 组件库**: Ant Design Vue 4.x
- **路由**: Vue Router
- **状态管理**: Pinia
- **HTTP**: Axios
- **图标**: @ant-design/icons-vue

---

## 你的任务

```
frontend/admin/
├── src/
│   ├── views/
│   │   ├── users/
│   │   │   ├── UserList.vue       # 用户列表
│   │   │   └── UserForm.vue       # 用户表单
│   │   └── items/
│   │       ├── ItemList.vue       # 物品列表
│   │       └── ItemForm.vue       # 物品表单
│   ├── components/
│   │   ├── DataTable.vue          # 数据表格封装
│   │   ├── SearchForm.vue         # 搜索表单
│   │   └── ConfirmModal.vue       # 确认弹窗
│   ├── layouts/
│   │   └── AdminLayout.vue        # 后台布局
│   └── ...
```

---

## 1. 后台布局 (AdminLayout.vue)

```vue
<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  Layout,
  LayoutSider,
  LayoutHeader,
  LayoutContent,
  Menu,
  MenuItem,
  Breadcrumb,
  BreadcrumbItem,
  Avatar,
  Dropdown,
} from 'ant-design-vue'
import {
  UserOutlined,
  ShoppingOutlined,
  DashboardOutlined,
  SettingOutlined,
  LogoutOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
} from '@ant-design/icons-vue'
import { useAdminStore } from '@/stores/admin'

const route = useRoute()
const router = useRouter()
const adminStore = useAdminStore()

const collapsed = ref(false)

// 菜单配置
const menuItems = [
  {
    key: '/admin/dashboard',
    icon: DashboardOutlined,
    label: '仪表盘',
  },
  {
    key: '/admin/users',
    icon: UserOutlined,
    label: '用户管理',
  },
  {
    key: '/admin/items',
    icon: ShoppingOutlined,
    label: '物品管理',
  },
  {
    key: '/admin/settings',
    icon: SettingOutlined,
    label: '系统设置',
  },
]

// 当前选中的菜单
const selectedKeys = computed(() => [route.path])

// 面包屑
const breadcrumbs = computed(() => {
  const matched = route.matched.filter(r => r.meta?.title)
  return matched.map(r => ({
    path: r.path,
    title: r.meta?.title as string,
  }))
})

// 菜单点击
function handleMenuClick({ key }: { key: string }) {
  router.push(key)
}

// 退出登录
function handleLogout() {
  adminStore.logout()
  router.push('/admin/login')
}
</script>

<template>
  <Layout class="admin-layout">
    <!-- 侧边栏 -->
    <LayoutSider
      v-model:collapsed="collapsed"
      :trigger="null"
      collapsible
      class="admin-sider"
    >
      <div class="logo">
        <span v-if="!collapsed">推荐系统管理</span>
        <span v-else>RS</span>
      </div>
      
      <Menu
        :selected-keys="selectedKeys"
        theme="dark"
        mode="inline"
        @click="handleMenuClick"
      >
        <MenuItem v-for="item in menuItems" :key="item.key">
          <component :is="item.icon" />
          <span>{{ item.label }}</span>
        </MenuItem>
      </Menu>
    </LayoutSider>

    <Layout>
      <!-- 顶部栏 -->
      <LayoutHeader class="admin-header">
        <div class="header-left">
          <component
            :is="collapsed ? MenuUnfoldOutlined : MenuFoldOutlined"
            class="trigger"
            @click="collapsed = !collapsed"
          />
          
          <Breadcrumb class="breadcrumb">
            <BreadcrumbItem v-for="item in breadcrumbs" :key="item.path">
              {{ item.title }}
            </BreadcrumbItem>
          </Breadcrumb>
        </div>

        <div class="header-right">
          <Dropdown>
            <template #overlay>
              <Menu>
                <MenuItem key="profile">
                  <UserOutlined /> 个人信息
                </MenuItem>
                <MenuItem key="logout" @click="handleLogout">
                  <LogoutOutlined /> 退出登录
                </MenuItem>
              </Menu>
            </template>
            <div class="user-info">
              <Avatar :size="32">
                {{ adminStore.currentAdmin?.name?.[0] || 'A' }}
              </Avatar>
              <span class="username">{{ adminStore.currentAdmin?.name || 'Admin' }}</span>
            </div>
          </Dropdown>
        </div>
      </LayoutHeader>

      <!-- 内容区 -->
      <LayoutContent class="admin-content">
        <router-view />
      </LayoutContent>
    </Layout>
  </Layout>
</template>

<style scoped>
.admin-layout {
  min-height: 100vh;
}

.admin-sider {
  background: #001529;
}

.logo {
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.25rem;
  font-weight: 700;
  color: #fff;
  background: rgba(255, 255, 255, 0.1);
}

.admin-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #fff;
  padding: 0 24px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
}

.trigger {
  font-size: 18px;
  cursor: pointer;
  transition: color 0.3s;
}

.trigger:hover {
  color: #1890ff;
}

.header-right {
  display: flex;
  align-items: center;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.username {
  font-weight: 500;
}

.admin-content {
  margin: 24px;
  padding: 24px;
  background: #fff;
  border-radius: 8px;
  min-height: 280px;
}
</style>
```

---

## 2. 用户列表 (UserList.vue)

```vue
<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import {
  Table,
  Button,
  Space,
  Input,
  Select,
  Tag,
  Popconfirm,
  message,
} from 'ant-design-vue'
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  SearchOutlined,
} from '@ant-design/icons-vue'
import type { User, PaginatedResponse } from '@shared/types'
import { adminUserApi } from '@/api/admin/user'

const router = useRouter()

// 表格数据
const loading = ref(false)
const data = ref<User[]>([])
const pagination = reactive({
  current: 1,
  pageSize: 10,
  total: 0,
})

// 搜索条件
const searchForm = reactive({
  keyword: '',
  gender: '',
})

// 表格列定义
const columns = [
  {
    title: 'ID',
    dataIndex: 'id',
    width: 100,
    ellipsis: true,
  },
  {
    title: '姓名',
    dataIndex: 'name',
    width: 150,
  },
  {
    title: '邮箱',
    dataIndex: 'email',
    width: 200,
  },
  {
    title: '年龄',
    dataIndex: 'age',
    width: 80,
  },
  {
    title: '性别',
    dataIndex: 'gender',
    width: 80,
    customRender: ({ text }: { text: string }) => {
      const colorMap: Record<string, string> = {
        male: 'blue',
        female: 'pink',
        other: 'default',
      }
      const labelMap: Record<string, string> = {
        male: '男',
        female: '女',
        other: '其他',
      }
      return h(Tag, { color: colorMap[text] }, () => labelMap[text] || text)
    },
  },
  {
    title: '创建时间',
    dataIndex: 'created_at',
    width: 180,
    customRender: ({ text }: { text: string }) => {
      return new Date(text).toLocaleString()
    },
  },
  {
    title: '操作',
    key: 'action',
    width: 150,
    fixed: 'right',
  },
]

// 加载数据
async function fetchData() {
  loading.value = true
  try {
    const response: PaginatedResponse<User> = await adminUserApi.listUsers({
      page: pagination.current,
      page_size: pagination.pageSize,
      keyword: searchForm.keyword,
      gender: searchForm.gender,
    })
    data.value = response.items
    pagination.total = response.total
  } catch (error) {
    message.error('加载用户列表失败')
  } finally {
    loading.value = false
  }
}

// 翻页
function handleTableChange(pag: any) {
  pagination.current = pag.current
  pagination.pageSize = pag.pageSize
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
  searchForm.gender = ''
  pagination.current = 1
  fetchData()
}

// 新增
function handleAdd() {
  router.push('/admin/users/create')
}

// 编辑
function handleEdit(record: User) {
  router.push(`/admin/users/${record.id}/edit`)
}

// 删除
async function handleDelete(record: User) {
  try {
    await adminUserApi.deleteUser(record.id)
    message.success('删除成功')
    fetchData()
  } catch (error) {
    message.error('删除失败')
  }
}

onMounted(() => {
  fetchData()
})
</script>

<template>
  <div class="user-list-page">
    <div class="page-header">
      <h2>用户管理</h2>
      <Button type="primary" @click="handleAdd">
        <PlusOutlined /> 新增用户
      </Button>
    </div>

    <!-- 搜索表单 -->
    <div class="search-form">
      <Space>
        <Input
          v-model:value="searchForm.keyword"
          placeholder="搜索姓名/邮箱"
          style="width: 200px"
          @press-enter="handleSearch"
        >
          <template #prefix>
            <SearchOutlined />
          </template>
        </Input>
        
        <Select
          v-model:value="searchForm.gender"
          placeholder="性别"
          style="width: 100px"
          allow-clear
        >
          <Select.Option value="male">男</Select.Option>
          <Select.Option value="female">女</Select.Option>
          <Select.Option value="other">其他</Select.Option>
        </Select>
        
        <Button type="primary" @click="handleSearch">搜索</Button>
        <Button @click="handleReset">重置</Button>
      </Space>
    </div>

    <!-- 数据表格 -->
    <Table
      :columns="columns"
      :data-source="data"
      :loading="loading"
      :pagination="pagination"
      :scroll="{ x: 1000 }"
      row-key="id"
      @change="handleTableChange"
    >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'action'">
          <Space>
            <Button type="link" size="small" @click="handleEdit(record)">
              <EditOutlined /> 编辑
            </Button>
            <Popconfirm
              title="确定要删除这个用户吗？"
              ok-text="确定"
              cancel-text="取消"
              @confirm="handleDelete(record)"
            >
              <Button type="link" danger size="small">
                <DeleteOutlined /> 删除
              </Button>
            </Popconfirm>
          </Space>
        </template>
      </template>
    </Table>
  </div>
</template>

<style scoped>
.user-list-page {
  padding: 0;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-header h2 {
  margin: 0;
  font-size: 20px;
}

.search-form {
  margin-bottom: 16px;
}
</style>
```

---

## 3. 用户表单 (UserForm.vue)

```vue
<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  Form,
  FormItem,
  Input,
  InputNumber,
  Select,
  Button,
  Card,
  message,
} from 'ant-design-vue'
import type { User, CreateUserRequest, UpdateUserRequest } from '@shared/types'
import { adminUserApi } from '@/api/admin/user'

const route = useRoute()
const router = useRouter()

const userId = computed(() => route.params.id as string | undefined)
const isEdit = computed(() => !!userId.value)
const loading = ref(false)
const submitLoading = ref(false)

// 表单数据
const formData = reactive<CreateUserRequest & { password?: string }>({
  name: '',
  email: '',
  age: undefined,
  gender: undefined,
  password: '',
})

// 表单校验规则
const rules = {
  name: [
    { required: true, message: '请输入姓名' },
    { min: 2, max: 50, message: '姓名长度 2-50 个字符' },
  ],
  email: [
    { required: true, message: '请输入邮箱' },
    { type: 'email', message: '邮箱格式不正确' },
  ],
  password: [
    { required: !isEdit.value, message: '请输入密码' },
    { min: 6, message: '密码至少 6 位' },
  ],
}

// 加载用户数据（编辑模式）
async function fetchUser() {
  if (!userId.value) return
  
  loading.value = true
  try {
    const user = await adminUserApi.getUser(userId.value)
    Object.assign(formData, {
      name: user.name,
      email: user.email,
      age: user.age,
      gender: user.gender,
    })
  } catch (error) {
    message.error('加载用户信息失败')
    router.back()
  } finally {
    loading.value = false
  }
}

// 提交表单
async function handleSubmit() {
  submitLoading.value = true
  
  try {
    if (isEdit.value) {
      await adminUserApi.updateUser(userId.value!, {
        name: formData.name,
        email: formData.email,
        age: formData.age,
        gender: formData.gender,
      })
      message.success('更新成功')
    } else {
      await adminUserApi.createUser(formData as CreateUserRequest)
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
function handleCancel() {
  router.push('/admin/users')
}

onMounted(() => {
  if (isEdit.value) {
    fetchUser()
  }
})
</script>

<template>
  <div class="user-form-page">
    <Card :title="isEdit ? '编辑用户' : '新增用户'" :loading="loading">
      <Form
        :model="formData"
        :rules="rules"
        layout="vertical"
        style="max-width: 500px"
        @finish="handleSubmit"
      >
        <FormItem label="姓名" name="name">
          <Input v-model:value="formData.name" placeholder="请输入姓名" />
        </FormItem>

        <FormItem label="邮箱" name="email">
          <Input v-model:value="formData.email" placeholder="请输入邮箱" />
        </FormItem>

        <FormItem v-if="!isEdit" label="密码" name="password">
          <Input.Password v-model:value="formData.password" placeholder="请输入密码" />
        </FormItem>

        <FormItem label="年龄" name="age">
          <InputNumber
            v-model:value="formData.age"
            :min="1"
            :max="150"
            placeholder="请输入年龄"
            style="width: 100%"
          />
        </FormItem>

        <FormItem label="性别" name="gender">
          <Select v-model:value="formData.gender" placeholder="请选择性别">
            <Select.Option value="male">男</Select.Option>
            <Select.Option value="female">女</Select.Option>
            <Select.Option value="other">其他</Select.Option>
          </Select>
        </FormItem>

        <FormItem>
          <Button type="primary" html-type="submit" :loading="submitLoading">
            {{ isEdit ? '更新' : '创建' }}
          </Button>
          <Button style="margin-left: 12px" @click="handleCancel">
            取消
          </Button>
        </FormItem>
      </Form>
    </Card>
  </div>
</template>

<style scoped>
.user-form-page {
  max-width: 800px;
}
</style>
```

---

## 4. 物品列表 (ItemList.vue)

```vue
<script setup lang="ts">
import { ref, reactive, onMounted, h } from 'vue'
import { useRouter } from 'vue-router'
import {
  Table,
  Button,
  Space,
  Input,
  Select,
  Tag,
  Popconfirm,
  Image,
  message,
} from 'ant-design-vue'
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  SearchOutlined,
  EyeOutlined,
} from '@ant-design/icons-vue'
import type { Item, PaginatedResponse } from '@shared/types'
import { adminItemApi } from '@/api/admin/item'

const router = useRouter()

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
  type: '',
  status: '',
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
  active: { color: 'green', label: '上架' },
  inactive: { color: 'default', label: '下架' },
}

// 表格列定义
const columns = [
  {
    title: 'ID',
    dataIndex: 'id',
    width: 100,
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
    customRender: ({ text }: { text: string }) => {
      const config = typeConfig[text]
      return config
        ? h(Tag, { color: config.color }, () => config.label)
        : text
    },
  },
  {
    title: '分类',
    dataIndex: 'category',
    width: 120,
  },
  {
    title: '标签',
    dataIndex: 'tags',
    width: 200,
    customRender: ({ text }: { text: string[] }) => {
      return h(
        Space,
        { size: 4, wrap: true },
        () => text?.slice(0, 3).map(tag => h(Tag, { key: tag }, () => tag)) || []
      )
    },
  },
  {
    title: '状态',
    dataIndex: 'status',
    width: 80,
    customRender: ({ text }: { text: string }) => {
      const config = statusConfig[text]
      return config
        ? h(Tag, { color: config.color }, () => config.label)
        : text
    },
  },
  {
    title: '创建时间',
    dataIndex: 'created_at',
    width: 180,
    customRender: ({ text }: { text: string }) => {
      return new Date(text).toLocaleString()
    },
  },
  {
    title: '操作',
    key: 'action',
    width: 180,
    fixed: 'right',
  },
]

// 加载数据
async function fetchData() {
  loading.value = true
  try {
    const response: PaginatedResponse<Item> = await adminItemApi.listItems({
      page: pagination.current,
      page_size: pagination.pageSize,
      type: searchForm.type,
      keyword: searchForm.keyword,
    })
    data.value = response.items
    pagination.total = response.total
  } catch (error) {
    message.error('加载物品列表失败')
  } finally {
    loading.value = false
  }
}

// 翻页
function handleTableChange(pag: any) {
  pagination.current = pag.current
  pagination.pageSize = pag.pageSize
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
  searchForm.type = ''
  searchForm.status = ''
  pagination.current = 1
  fetchData()
}

// 新增
function handleAdd() {
  router.push('/admin/items/create')
}

// 查看
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
    await adminItemApi.deleteItem(record.id)
    message.success('删除成功')
    fetchData()
  } catch (error) {
    message.error('删除失败')
  }
}

onMounted(() => {
  fetchData()
})
</script>

<template>
  <div class="item-list-page">
    <div class="page-header">
      <h2>物品管理</h2>
      <Button type="primary" @click="handleAdd">
        <PlusOutlined /> 新增物品
      </Button>
    </div>

    <!-- 搜索表单 -->
    <div class="search-form">
      <Space>
        <Input
          v-model:value="searchForm.keyword"
          placeholder="搜索标题"
          style="width: 200px"
          @press-enter="handleSearch"
        >
          <template #prefix>
            <SearchOutlined />
          </template>
        </Input>
        
        <Select
          v-model:value="searchForm.type"
          placeholder="类型"
          style="width: 100px"
          allow-clear
        >
          <Select.Option value="movie">电影</Select.Option>
          <Select.Option value="product">商品</Select.Option>
          <Select.Option value="article">文章</Select.Option>
          <Select.Option value="video">视频</Select.Option>
        </Select>
        
        <Select
          v-model:value="searchForm.status"
          placeholder="状态"
          style="width: 100px"
          allow-clear
        >
          <Select.Option value="active">上架</Select.Option>
          <Select.Option value="inactive">下架</Select.Option>
        </Select>
        
        <Button type="primary" @click="handleSearch">搜索</Button>
        <Button @click="handleReset">重置</Button>
      </Space>
    </div>

    <!-- 数据表格 -->
    <Table
      :columns="columns"
      :data-source="data"
      :loading="loading"
      :pagination="pagination"
      :scroll="{ x: 1200 }"
      row-key="id"
      @change="handleTableChange"
    >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'action'">
          <Space>
            <Button type="link" size="small" @click="handleView(record)">
              <EyeOutlined /> 查看
            </Button>
            <Button type="link" size="small" @click="handleEdit(record)">
              <EditOutlined /> 编辑
            </Button>
            <Popconfirm
              title="确定要删除这个物品吗？"
              ok-text="确定"
              cancel-text="取消"
              @confirm="handleDelete(record)"
            >
              <Button type="link" danger size="small">
                <DeleteOutlined /> 删除
              </Button>
            </Popconfirm>
          </Space>
        </template>
      </template>
    </Table>
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

.page-header h2 {
  margin: 0;
  font-size: 20px;
}

.search-form {
  margin-bottom: 16px;
}
</style>
```

---

## 注意事项

1. 使用 Ant Design Vue 4.x 组件库
2. 表格支持分页、排序、筛选
3. 表单需要完整验证
4. 删除操作需要二次确认
5. 所有 API 请求需要处理错误

## 输出要求

请输出完整的可运行代码，包含：
1. AdminLayout 布局组件
2. 用户管理 CRUD
3. 物品管理 CRUD
4. 所有必要的 API 封装

