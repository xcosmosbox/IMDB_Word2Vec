# 管理后台数据管理模块

> Person D 开发 - 生成式推荐系统管理后台

## 📋 模块概述

本模块是推荐系统管理后台的数据管理部分，主要负责：

- **用户管理**：用户的增删改查 (CRUD) 操作
- **物品管理**：物品的增删改查 (CRUD) 操作
- **通用组件**：可复用的数据表格、搜索表单、确认弹窗等组件
- **状态管理**：管理员登录状态、权限控制
- **路由配置**：管理后台的路由定义和权限守卫

---

## 🏗️ 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Vue 3 | ^3.4 | 核心框架 |
| TypeScript | ^5.3 | 类型安全 |
| Vite | ^5.0 | 构建工具 |
| Ant Design Vue | ^4.0 | UI 组件库 |
| Vue Router | ^4.2 | 路由管理 |
| Pinia | ^2.1 | 状态管理 |
| Vitest | ^1.0 | 单元测试 |

---

## 📁 目录结构

```
frontend/admin/src/
├── __tests__/                 # 单元测试
│   ├── api/                   # API 测试
│   │   └── provider.spec.ts
│   ├── components/            # 组件测试
│   │   ├── ConfirmModal.spec.ts
│   │   └── DataTable.spec.ts
│   ├── router/                # 路由测试
│   │   └── index.spec.ts
│   ├── stores/                # Store 测试
│   │   └── admin.spec.ts
│   ├── views/                 # 视图测试
│   │   ├── ItemList.spec.ts
│   │   └── UserList.spec.ts
│   └── setup.ts               # 测试环境配置
├── api/                       # API 封装
│   ├── http.ts                # HTTP 客户端
│   ├── index.ts               # 统一导出
│   ├── item.ts                # 物品 API
│   ├── provider.ts            # API Provider
│   └── user.ts                # 用户 API
├── components/                # 公共组件
│   ├── charts/                # 图表组件
│   │   ├── BarChart.vue
│   │   ├── LineChart.vue
│   │   └── PieChart.vue
│   ├── ConfirmModal.vue       # 确认弹窗
│   ├── DataTable.vue          # 数据表格
│   ├── index.ts               # 组件导出
│   └── SearchForm.vue         # 搜索表单
├── layouts/                   # 布局组件
│   └── AdminLayout.vue        # 后台主布局
├── router/                    # 路由配置
│   └── index.ts
├── stores/                    # 状态管理
│   └── admin.ts               # Admin Store
└── views/                     # 页面组件
    ├── analytics/
    │   └── Analytics.vue      # 数据分析（占位）
    ├── auth/
    │   └── Login.vue          # 登录页
    ├── dashboard/
    │   └── Dashboard.vue      # 仪表盘
    ├── error/
    │   └── NotFound.vue       # 404 页面
    ├── items/                 # 物品管理
    │   ├── ItemDetail.vue     # 物品详情
    │   ├── ItemForm.vue       # 物品表单
    │   └── ItemList.vue       # 物品列表
    ├── profile/
    │   └── Profile.vue        # 个人信息
    ├── settings/
    │   └── Settings.vue       # 系统设置
    └── users/                 # 用户管理
        ├── UserDetail.vue     # 用户详情
        ├── UserForm.vue       # 用户表单
        └── UserList.vue       # 用户列表
```

---

## 🚀 快速开始

### 安装依赖

```bash
cd recommend-system/frontend/admin
npm install
```

### 开发模式

```bash
npm run dev
```

### 运行测试

```bash
npm run test
```

### 构建生产版本

```bash
npm run build
```

---

## 📖 核心概念

### 1. 接口驱动开发 (Interface-Driven Development)

本模块遵循接口驱动开发模式，所有 API 调用都通过接口进行，确保可插拔设计。

#### 使用方式

```typescript
// ✅ 正确：通过依赖注入获取 API
import { inject } from 'vue'
import type { IApiProvider } from '@shared/api/interfaces'

const api = inject<IApiProvider>('api')!
const { items, total } = await api.adminUser.listUsers({ page: 1, page_size: 10 })

// ❌ 错误：直接导入具体实现
import { adminUserApi } from '@/api/admin'
```

### 2. API Provider

`IApiProvider` 是所有服务的统一入口，支持 Mock 模式和 HTTP 模式：

```typescript
// 生产环境
const api = new HttpApiProvider()

// 开发/测试环境
const api = new MockApiProvider()

// 使用
const user = await api.adminUser.getUser('123')
await api.adminItem.createItem({ type: 'movie', title: '新电影' })
```

### 3. 状态管理 (Admin Store)

使用 Pinia 管理管理员状态：

```typescript
import { useAdminStore } from '@/stores/admin'

const adminStore = useAdminStore()

// 登录
await adminStore.login({ email: 'admin@example.com', password: 'password' })

// 检查权限
if (adminStore.hasPermission('user:write')) {
  // 有权限执行操作
}

// 登出
await adminStore.logout()
```

---

## 📦 组件说明

### DataTable 数据表格

封装 Ant Design Vue Table，提供统一的表格功能。

```vue
<template>
  <DataTable
    :columns="columns"
    :data-source="data"
    :loading="loading"
    :pagination="pagination"
    :scroll-x="1200"
    row-key="id"
    @page-change="handlePageChange"
  >
    <template #bodyCell="{ column, record }">
      <template v-if="column.key === 'action'">
        <Button @click="handleEdit(record)">编辑</Button>
      </template>
    </template>
  </DataTable>
</template>
```

#### Props

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| columns | Column[] | - | 列定义 |
| dataSource | any[] | - | 数据源 |
| rowKey | string \| Function | 'id' | 行唯一键 |
| loading | boolean | false | 加载状态 |
| pagination | Pagination \| false | - | 分页配置 |
| scrollX | number \| string | - | 横向滚动宽度 |
| emptyText | string | '暂无数据' | 空状态描述 |

### ConfirmModal 确认弹窗

封装常用的确认操作弹窗。

```vue
<template>
  <ConfirmModal
    v-model:open="visible"
    title="删除确认"
    content="确定要删除这条数据吗？"
    type="error"
    ok-danger
    :loading="loading"
    @ok="handleConfirm"
    @cancel="handleCancel"
  />
</template>
```

#### Props

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| open | boolean | false | 是否显示 |
| title | string | '确认' | 标题 |
| content | string | - | 内容 |
| type | 'info' \| 'warning' \| 'error' \| 'success' | 'confirm' | 类型 |
| okText | string | '确定' | 确认按钮文字 |
| cancelText | string | '取消' | 取消按钮文字 |
| okDanger | boolean | false | 确认按钮是否为危险按钮 |
| loading | boolean | false | 加载状态 |

### SearchForm 搜索表单

提供统一的搜索表单布局。

```vue
<template>
  <SearchForm
    v-model="searchData"
    :fields="fields"
    :loading="loading"
    @search="handleSearch"
    @reset="handleReset"
  >
    <template #keyword="{ value, update }">
      <Input :value="value" @update:value="update" placeholder="搜索关键词" />
    </template>
  </SearchForm>
</template>
```

---

## 🛣️ 路由配置

### 路由结构

```
/admin
├── /login                 # 登录页 (无需认证)
├── /dashboard             # 仪表盘
├── /users                 # 用户列表
│   ├── /create           # 新增用户
│   ├── /:id              # 用户详情
│   └── /:id/edit         # 编辑用户
├── /items                 # 物品列表
│   ├── /create           # 新增物品
│   ├── /:id              # 物品详情
│   └── /:id/edit         # 编辑物品
├── /analytics             # 数据分析
├── /settings              # 系统设置
└── /profile               # 个人信息
```

### 路由守卫

路由守卫自动处理：
- 认证检查：未登录重定向到登录页
- Token 刷新：Token 过期时自动刷新
- 权限验证：无权限时跳转到首页

```typescript
// 路由元信息
{
  path: 'users',
  meta: {
    title: '用户列表',
    permission: 'user:read',  // 需要的权限
  },
}
```

---

## 🔐 权限系统

### 权限定义

| 权限 | 说明 |
|------|------|
| user:read | 查看用户 |
| user:write | 创建/编辑/删除用户 |
| item:read | 查看物品 |
| item:write | 创建/编辑/删除物品 |
| analytics:read | 查看数据分析 |
| settings:read | 查看系统设置 |

### 权限检查

```typescript
const adminStore = useAdminStore()

// 检查单个权限
if (adminStore.hasPermission('user:write')) {
  // ...
}

// 检查多个权限（任意一个）
if (adminStore.hasAnyPermission(['user:read', 'item:read'])) {
  // ...
}

// 检查多个权限（全部）
if (adminStore.hasAllPermissions(['user:read', 'user:write'])) {
  // ...
}

// 超级管理员
if (adminStore.isSuperAdmin) {
  // 超级管理员拥有所有权限
}
```

---

## 🧪 单元测试

### 测试覆盖

| 模块 | 测试文件 | 覆盖内容 |
|------|----------|----------|
| Admin Store | admin.spec.ts | 登录、登出、权限检查 |
| DataTable | DataTable.spec.ts | 渲染、分页、加载状态 |
| ConfirmModal | ConfirmModal.spec.ts | 显示/隐藏、事件触发 |
| UserList | UserList.spec.ts | 列表加载、搜索、删除 |
| ItemList | ItemList.spec.ts | 列表加载、搜索、重置 |
| Router | index.spec.ts | 路由配置、导航 |
| API Provider | provider.spec.ts | Mock 数据返回 |

### 运行测试

```bash
# 运行所有测试
npm run test

# 运行测试并生成覆盖率报告
npm run test:coverage

# 监听模式
npm run test:watch
```

---

## 📝 开发指南

### 添加新页面

1. 在 `views/` 下创建新的 Vue 组件
2. 在 `router/index.ts` 中添加路由配置
3. 如需权限控制，在路由 meta 中添加 `permission` 字段

### 添加新的 API

1. 在 `@shared/api/interfaces.ts` 中定义接口
2. 在 `api/` 目录下实现 HTTP 版本
3. 在 `api/provider.ts` 中添加 Mock 实现

### 添加新组件

1. 在 `components/` 下创建组件
2. 在 `components/index.ts` 中导出
3. 编写对应的单元测试

---

## 🔗 相关文档

- [前端开发任务分配](../prompts/README.md)
- [接口定义](../shared/api/interfaces.ts)
- [类型定义](../shared/types/index.ts)
- [系统架构设计](../../docs/生成式推荐系统架构设计.md)

---

## 👥 协作说明

本模块由 **Person D** 负责开发，与其他模块的协作关系：

| 依赖方 | 说明 |
|--------|------|
| Person F | 依赖基础设施（Axios、Pinia、Router） |
| Person E | 数据分析模块在同一布局下 |

---

## 📄 License

MIT License

