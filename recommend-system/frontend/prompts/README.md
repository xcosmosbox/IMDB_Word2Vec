# 前端开发任务分配

## 概述

本目录包含前端层开发的 6 个独立任务，每个任务由一位开发者独立完成。

---

## ⚠️ 重要：类型驱动开发

所有开发者必须先阅读共享类型定义：

```
frontend/shared/types/index.ts
```

---

## 任务分配

| 角色 | 负责模块 | 提示词文件 | 技术重点 |
|------|----------|------------|----------|
| **Person A** | 用户端-首页与推荐展示 | `person_a_user_home.md` | Vue3, 自定义组件, 暗色主题 |
| **Person B** | 用户端-搜索与详情页 | `person_b_search_detail.md` | Vue Router, 搜索, 骨架屏 |
| **Person C** | 用户端-用户中心 | `person_c_user_center.md` | Pinia, 表单验证, 认证 |
| **Person D** | 管理后台-数据管理 | `person_d_admin_data.md` | Ant Design Vue, CRUD |
| **Person E** | 管理后台-数据分析 | `person_e_admin_analytics.md` | ECharts, 图表可视化 |
| **Person F** | 前端基础设施 | `person_f_infra.md` | Axios, Pinia, Router, Vite |

---

## 项目结构

```
frontend/
├── shared/                  # 共享代码
│   ├── api/                 # API 封装 (Person F)
│   ├── types/               # 类型定义 (已提供)
│   └── utils/               # 工具函数 (Person F)
├── user-app/                # 用户端应用
│   ├── src/
│   │   ├── views/           # 页面组件 (Person A/B/C)
│   │   ├── components/      # 通用组件 (Person A/B/C/F)
│   │   ├── stores/          # 状态管理 (Person C/F)
│   │   └── router/          # 路由配置 (Person F)
│   └── ...
├── admin/                   # 管理后台
│   ├── src/
│   │   ├── views/           # 页面组件 (Person D/E)
│   │   ├── components/      # 组件 (Person D/E)
│   │   ├── stores/          # 状态管理 (Person D)
│   │   └── router/          # 路由配置 (Person D)
│   └── ...
└── prompts/                 # 提示词文件 (本目录)
```

---

## 技术栈

### 用户端 (user-app)

| 技术 | 版本 | 用途 |
|------|------|------|
| Vue 3 | ^3.4 | 核心框架 |
| Vite | ^5.0 | 构建工具 |
| Vue Router | ^4.2 | 路由 |
| Pinia | ^2.1 | 状态管理 |
| Axios | ^1.6 | HTTP 客户端 |
| TypeScript | ^5.3 | 类型安全 |

### 管理后台 (admin)

| 技术 | 版本 | 用途 |
|------|------|------|
| Vue 3 | ^3.4 | 核心框架 |
| Ant Design Vue | ^4.0 | UI 组件库 |
| ECharts | ^5.4 | 图表可视化 |
| vue-echarts | ^6.6 | ECharts Vue 绑定 |
| dayjs | ^1.11 | 日期处理 |

---

## 开发规范

### 1. 组件规范

```vue
<script setup lang="ts">
// 使用 Composition API + TypeScript
import { ref, computed, onMounted } from 'vue'
import type { Item } from '@shared/types'

// Props 定义
interface Props {
  item: Item
}
const props = defineProps<Props>()

// Emits 定义
const emit = defineEmits<{
  'click': [id: string]
}>()
</script>

<template>
  <!-- 模板内容 -->
</template>

<style scoped>
/* 使用 scoped 样式 */
</style>
```

### 2. API 调用

```typescript
import { itemApi } from '@shared/api'
import type { Item } from '@shared/types'

// 使用 async/await
const item = await itemApi.getItem(itemId)

// 错误处理
try {
  await itemApi.createItem(data)
} catch (error) {
  // 处理错误
}
```

### 3. 状态管理

```typescript
import { useItemStore } from '@/stores/item'

const itemStore = useItemStore()

// 访问状态
console.log(itemStore.items)

// 调用方法
await itemStore.fetchItems()
```

---

## 依赖关系

```
Person F (基础设施) ──────────────────────────────────────────────┐
                                                                  │
Person A (首页) ←───────────── shared/api ←──────────────────────┤
Person B (搜索) ←───────────── shared/types ←────────────────────┤
Person C (用户中心) ←───────── stores/ ←─────────────────────────┤
Person D (数据管理) ←───────── router/ ←─────────────────────────┘
Person E (数据分析) ←──────────────────────────────────────────────
```

**建议开发顺序：**
1. Person F 先完成基础设施
2. 其他成员并行开发各自模块
3. 最后集成测试

---

## 注意事项

1. **类型安全** - 所有代码必须使用 TypeScript
2. **样式隔离** - 使用 `scoped` 避免样式污染
3. **错误处理** - API 调用必须有错误处理
4. **响应式** - 所有页面需支持移动端
5. **暗色主题** - 用户端使用暗色主题

