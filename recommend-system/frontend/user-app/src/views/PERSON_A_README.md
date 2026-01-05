# Person A: 用户端首页与推荐展示模块

> 开发者：Person A  
> 模块：用户端首页 (Home) + 推荐展示组件  
> 技术栈：Vue 3 + TypeScript + Pinia + Vite

---

## 📋 目录

1. [模块概述](#模块概述)
2. [文件结构](#文件结构)
3. [核心组件](#核心组件)
4. [状态管理](#状态管理)
5. [接口驱动开发](#接口驱动开发)
6. [设计系统](#设计系统)
7. [单元测试](#单元测试)
8. [开发指南](#开发指南)
9. [常见问题](#常见问题)

---

## 模块概述

本模块实现了生成式推荐系统的 **用户端首页** 和 **推荐展示** 功能，主要包括：

- 🏠 **首页视图** - 用户进入系统的主页面
- 📦 **推荐列表** - 展示个性化推荐内容
- 🎴 **物品卡片** - 单个推荐物品的展示
- 🏷️ **分类标签** - 按类型筛选推荐内容
- ⏳ **加载动画** - 优雅的加载状态反馈

### 核心特性

| 特性 | 描述 |
|------|------|
| 🌙 暗色主题 | 深蓝色渐变背景，霓虹色彩点缀 |
| 📱 响应式设计 | 支持桌面/平板/手机多端适配 |
| ♿ 无障碍支持 | 键盘导航、ARIA 属性 |
| 🚀 性能优化 | 虚拟滚动、懒加载、骨架屏 |
| 🔌 可插拔设计 | 通过接口注入 API 服务 |

---

## 文件结构

```
frontend/user-app/src/
├── views/
│   └── Home.vue                    # 首页视图 ⭐
├── components/
│   ├── RecommendList.vue           # 推荐列表组件
│   ├── ItemCard.vue                # 物品卡片组件
│   ├── CategoryTabs.vue            # 分类标签组件
│   └── LoadingSpinner.vue          # 加载动画组件
├── stores/
│   └── recommend.ts                # 推荐状态管理 (Pinia)
├── __tests__/
│   ├── setup.ts                    # 测试环境配置
│   ├── stores/
│   │   └── recommend.spec.ts       # Store 单元测试
│   ├── components/
│   │   ├── LoadingSpinner.spec.ts
│   │   ├── CategoryTabs.spec.ts
│   │   ├── ItemCard.spec.ts
│   │   └── RecommendList.spec.ts
│   └── views/
│       └── Home.spec.ts            # 首页单元测试
└── vitest.config.ts                # Vitest 配置
```

---

## 核心组件

### 1. Home.vue (首页视图)

**位置**: `src/views/Home.vue`

首页是用户进入系统的主入口，负责整合所有子组件。

#### 功能

- 展示英雄区域 (Hero Section) 标题和副标题
- 渲染分类标签导航
- 加载和展示推荐列表
- 处理用户交互事件 (点击、喜欢、分享)
- 错误状态处理和重试

#### 使用示例

```vue
<template>
  <Home />
</template>

<script setup lang="ts">
import Home from '@/views/Home.vue'
</script>
```

#### 关键代码

```typescript
// 依赖注入 API Provider
const apiProvider = inject<IApiProvider>('api')

// 初始化 Store
const recommendStore = useRecommendStore()
if (apiProvider) {
  recommendStore.setApiProvider(apiProvider)
}

// 加载推荐
async function loadRecommendations() {
  await recommendStore.fetchRecommendations({
    user_id: currentUserId.value,
    limit: 50,
    scene: 'home',
  })
}

// 处理物品点击
function handleItemClick(itemId: string) {
  recommendStore.recordBehavior({
    user_id: currentUserId.value,
    item_id: itemId,
    action: 'click',
  })
  router.push(`/item/${itemId}`)
}
```

---

### 2. RecommendList.vue (推荐列表)

**位置**: `src/components/RecommendList.vue`

展示推荐物品的瀑布流/网格布局。

#### Props

| 属性 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `recommendations` | `Recommendation[]` | 必填 | 推荐列表数据 |
| `loading` | `boolean` | `false` | 加载状态 |
| `hasMore` | `boolean` | `true` | 是否有更多数据 |
| `title` | `string` | `'为你推荐'` | 列表标题 |
| `showRefresh` | `boolean` | `true` | 是否显示刷新按钮 |
| `emptyText` | `string` | `'暂无推荐内容'` | 空状态提示 |
| `skeletonCount` | `number` | `8` | 骨架屏数量 |

#### Events

| 事件 | 参数 | 描述 |
|------|------|------|
| `item-click` | `itemId: string` | 物品点击 |
| `item-like` | `itemId: string` | 物品喜欢 |
| `item-share` | `itemId: string` | 物品分享 |
| `refresh` | - | 请求刷新 |
| `load-more` | - | 请求加载更多 |

#### 使用示例

```vue
<RecommendList
  :recommendations="recommendations"
  :loading="isLoading"
  :has-more="hasMore"
  title="热门推荐"
  @item-click="handleItemClick"
  @refresh="handleRefresh"
  @load-more="handleLoadMore"
/>
```

---

### 3. ItemCard.vue (物品卡片)

**位置**: `src/components/ItemCard.vue`

展示单个推荐物品的卡片组件。

#### Props

| 属性 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `item` | `Item` | 必填 | 物品数据 |
| `score` | `number` | - | 推荐分数 (0-1) |
| `reason` | `string` | - | 推荐理由 |
| `loading` | `boolean` | `false` | 骨架屏模式 |

#### Events

| 事件 | 参数 | 描述 |
|------|------|------|
| `click` | - | 卡片点击 |
| `like` | `itemId: string` | 喜欢按钮点击 |
| `share` | `itemId: string` | 分享按钮点击 |

#### 物品类型配色

| 类型 | 颜色 | 图标 |
|------|------|------|
| movie | `#e50914` | 🎬 |
| product | `#ff9900` | 🛒 |
| article | `#1da1f2` | 📄 |
| video | `#ff0050` | 🎥 |

---

### 4. CategoryTabs.vue (分类标签)

**位置**: `src/components/CategoryTabs.vue`

分类导航标签页，支持键盘导航。

#### Props

| 属性 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `categories` | `Category[]` | 必填 | 分类列表 |
| `active` | `string` | 必填 | 当前激活分类 |
| `sticky` | `boolean` | `true` | 是否粘性定位 |

#### Category 类型

```typescript
interface Category {
  key: string      // 分类唯一标识
  label: string    // 显示名称
  icon?: string    // 图标 (可选)
}
```

#### 键盘导航

| 按键 | 功能 |
|------|------|
| `←` / `→` | 切换相邻标签 |
| `Home` | 跳转到第一个标签 |
| `End` | 跳转到最后一个标签 |

---

### 5. LoadingSpinner.vue (加载动画)

**位置**: `src/components/LoadingSpinner.vue`

优雅的加载指示器。

#### Props

| 属性 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `size` | `'small' \| 'medium' \| 'large'` | `'medium'` | 尺寸 |
| `color` | `string` | `'#4facfe'` | 颜色 |
| `showText` | `boolean` | `false` | 是否显示文字 |
| `text` | `string` | `'加载中...'` | 加载文字 |
| `fullscreen` | `boolean` | `false` | 全屏覆盖模式 |

---

## 状态管理

### useRecommendStore

**位置**: `src/stores/recommend.ts`

使用 Pinia 管理推荐相关状态。

#### State

```typescript
{
  recommendations: Recommendation[]  // 推荐列表
  isLoading: boolean                // 加载状态
  error: string | null              // 错误信息
  currentRequestId: string          // 当前请求 ID
  strategy: string                  // 推荐策略
  viewedItemIds: Set<string>        // 已浏览物品
}
```

#### Getters

| 名称 | 返回类型 | 描述 |
|------|----------|------|
| `totalCount` | `number` | 推荐总数 |
| `topRecommendations` | `Recommendation[]` | 高分推荐 (≥0.8) |
| `recommendationsByType` | `Record<string, Recommendation[]>` | 按类型分组 |
| `hasRecommendations` | `boolean` | 是否有推荐 |

#### Actions

| 方法 | 参数 | 描述 |
|------|------|------|
| `setApiProvider` | `provider: IApiProvider` | 设置 API 服务 |
| `fetchRecommendations` | `request: RecommendRequest` | 获取推荐 |
| `refreshRecommendations` | `userId, limit` | 刷新推荐 |
| `loadMoreRecommendations` | `userId, limit` | 加载更多 |
| `recordBehavior` | `data: RecordBehaviorRequest` | 记录行为 |
| `submitFeedback` | `itemId, action` | 提交反馈 |
| `clearRecommendations` | - | 清空推荐 |
| `$reset` | - | 重置状态 |

#### 使用示例

```typescript
import { useRecommendStore } from '@/stores/recommend'

const recommendStore = useRecommendStore()

// 设置 API Provider (依赖注入)
recommendStore.setApiProvider(apiProvider)

// 获取推荐
await recommendStore.fetchRecommendations({
  user_id: 'user_123',
  limit: 50,
  scene: 'home',
})

// 访问状态
console.log(recommendStore.recommendations)
console.log(recommendStore.totalCount)

// 记录行为
await recommendStore.recordBehavior({
  item_id: 'item_1',
  action: 'click',
})
```

---

## 接口驱动开发

### 核心原则

⚠️ **不要直接导入具体实现！** 通过依赖注入使用 API 服务。

```typescript
// ✅ 正确：通过 inject 获取接口
const api = inject<IApiProvider>('api')!
await api.recommend.getRecommendations(request)

// ❌ 错误：直接导入具体实现
import { recommendApi } from '@shared/api'
```

### 接口定义

详见 `frontend/shared/api/interfaces.ts`

```typescript
// 推荐服务接口
interface IRecommendService {
  getRecommendations(request: RecommendRequest): Promise<RecommendResponse>
  submitFeedback(feedback: FeedbackRequest): Promise<void>
  getSimilarRecommendations(itemId: string, limit?: number): Promise<Recommendation[]>
}

// 用户服务接口
interface IUserService {
  recordBehavior(data: RecordBehaviorRequest): Promise<void>
  // ...
}
```

### 在组件中注入

```typescript
// main.ts 或 App.vue 中提供
import { createApp } from 'vue'
import type { IApiProvider } from '@shared/api/interfaces'

const app = createApp(App)

// 开发环境使用 Mock
const apiProvider: IApiProvider = new MockApiProvider()
// 生产环境使用真实 API
// const apiProvider: IApiProvider = new HttpApiProvider()

app.provide('api', apiProvider)
```

---

## 设计系统

### 颜色变量

```css
/* 主色调 */
--color-primary: #4facfe;
--color-secondary: #00f2fe;
--color-accent: #64ffda;

/* 背景色 */
--bg-dark: #0f0f23;
--bg-card: rgba(255, 255, 255, 0.03);

/* 文字颜色 */
--text-primary: #ffffff;
--text-secondary: #8892b0;
--text-muted: #5a6378;
```

### 渐变

```css
/* 主渐变 */
background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);

/* 页面背景 */
background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
```

### 响应式断点

| 断点 | 宽度 | 列数 |
|------|------|------|
| 手机 | < 640px | 1 列 |
| 平板 | 640-899px | 2 列 |
| 小桌面 | 900-1199px | 3 列 |
| 桌面 | 1200-1599px | 4 列 |
| 大屏 | ≥ 1600px | 5 列 |

### 动效

- 卡片悬停：`translateY(-8px) + scale(1.02)`
- 过渡时长：`0.3s - 0.4s`
- 缓动函数：`cubic-bezier(0.4, 0, 0.2, 1)`

---

## 单元测试

### 运行测试

```bash
# 运行所有测试
npm run test

# 运行并监听变化
npm run test:watch

# 生成覆盖率报告
npm run test:coverage

# 运行特定文件
npm run test -- src/__tests__/stores/recommend.spec.ts
```

### 测试覆盖

| 文件 | 测试用例数 | 覆盖内容 |
|------|------------|----------|
| `recommend.spec.ts` | 18 | Store 状态、Actions、Getters |
| `LoadingSpinner.spec.ts` | 10 | 渲染、尺寸、颜色、全屏 |
| `CategoryTabs.spec.ts` | 15 | 渲染、交互、键盘导航、无障碍 |
| `ItemCard.spec.ts` | 22 | 渲染、类型、分数、交互、骨架屏 |
| `RecommendList.spec.ts` | 18 | 渲染、刷新、空状态、加载、事件 |
| `Home.spec.ts` | 15 | 渲染、加载、错误、事件 |

### 测试工具

- **Vitest** - 测试框架
- **@vue/test-utils** - Vue 组件测试工具
- **jsdom** - DOM 环境模拟

---

## 开发指南

### 环境准备

```bash
# 安装依赖
cd frontend/user-app
npm install

# 开发模式
npm run dev

# 构建
npm run build
```

### 添加新组件

1. 在 `src/components/` 创建组件文件
2. 使用 Composition API + TypeScript
3. 添加对应的单元测试
4. 在需要的地方导入使用

```vue
<script setup lang="ts">
// 使用 Composition API
import { ref, computed } from 'vue'

// Props 定义
interface Props {
  // ...
}
const props = defineProps<Props>()

// Emits 定义
const emit = defineEmits<{
  'event-name': [param: string]
}>()
</script>

<template>
  <!-- 模板 -->
</template>

<style scoped>
/* 使用 scoped 样式 */
</style>
```

### 代码规范

- ✅ 使用 TypeScript 类型
- ✅ 使用 Composition API
- ✅ 使用 scoped 样式
- ✅ 组件名使用 PascalCase
- ✅ 事件名使用 kebab-case
- ✅ 添加 JSDoc 注释

---

## 常见问题

### Q: 如何切换 Mock/真实 API？

在 `main.ts` 中切换 API Provider：

```typescript
// 开发环境
const apiProvider = new MockApiProvider()

// 生产环境
const apiProvider = new HttpApiProvider()

app.provide('api', apiProvider)
```

### Q: 如何添加新的物品类型？

1. 更新 `@shared/types` 中的 `Item.type` 类型
2. 在 `ItemCard.vue` 中添加类型配色和图标映射
3. 在 `CategoryTabs` 的分类列表中添加新类型

### Q: 如何自定义主题颜色？

修改组件中的 CSS 变量或直接修改颜色值。建议将颜色变量提取到全局样式文件中统一管理。

### Q: 测试失败怎么办？

1. 检查是否有 Mock 未正确设置
2. 确保 `beforeEach` 中正确初始化 Pinia
3. 异步操作后使用 `await flushPromises()`

---

## 相关文档

- [前端开发任务分配](../../prompts/README.md)
- [API 接口定义](../../shared/api/interfaces.ts)
- [类型定义](../../shared/types/index.ts)
- [生成式推荐系统架构](../../../../docs/生成式推荐系统架构设计.md)

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0.0 | 2025-01-04 | 初始版本，完成首页和推荐展示模块 |

---

*Made with ❤️ by Person A*

