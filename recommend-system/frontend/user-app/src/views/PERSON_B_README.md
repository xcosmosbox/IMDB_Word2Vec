# Person B: 搜索与详情页模块开发文档

## 概述

本模块由 Person B 负责开发，包含生成式推荐系统的 **搜索功能** 和 **物品详情页** 两个核心功能模块。

---

## 目录结构

```
user-app/src/
├── views/
│   ├── Search.vue              # 搜索页面
│   └── ItemDetail.vue          # 物品详情页
├── components/
│   ├── SearchBar.vue           # 搜索栏组件
│   ├── SearchResults.vue       # 搜索结果展示组件
│   ├── ItemInfo.vue            # 物品信息展示组件
│   ├── SimilarItems.vue        # 相似推荐组件
│   └── ActionButtons.vue       # 操作按钮组组件
└── __tests__/
    ├── views/
    │   ├── Search.spec.ts      # 搜索页面测试
    │   └── ItemDetail.spec.ts  # 详情页测试
    └── components/
        ├── SearchResults.spec.ts
        ├── ItemInfo.spec.ts
        ├── SimilarItems.spec.ts
        └── ActionButtons.spec.ts
```

---

## 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Vue 3 | ^3.4 | 核心框架 |
| TypeScript | ^5.3 | 类型安全 |
| Vue Router | ^4.2 | 页面路由 |
| Pinia | ^2.1 | 状态管理 |
| Vitest | ^1.0 | 单元测试 |

---

## 组件详解

### 1. Search.vue - 搜索页面

**文件路径**: `views/Search.vue`

**功能特性**:
- 搜索输入和触发
- 搜索结果展示与过滤
- 热门搜索推荐
- URL 参数同步（支持分享搜索链接）
- 最近浏览记录展示

**使用的 Store**:
- `useItemStore`: 物品搜索和数据管理

**路由配置**:
```typescript
{
  path: '/search',
  name: 'Search',
  component: () => import('@/views/Search.vue')
}
```

**URL 参数**:
- `q`: 搜索关键词，例如 `/search?q=科幻电影`

**关键方法**:

```typescript
// 执行搜索
async function handleSearch(searchQuery: string) {
  // 更新 URL 参数
  router.replace({ query: { q: searchQuery } })
  // 调用 store 搜索方法
  searchResults.value = await itemStore.searchItems(searchQuery, 50)
}
```

---

### 2. ItemDetail.vue - 物品详情页

**文件路径**: `views/ItemDetail.vue`

**功能特性**:
- 物品详细信息展示
- 物品统计数据展示（浏览数、点赞数等）
- 相似物品推荐
- 用户交互（喜欢、分享）
- 浏览行为自动记录
- 骨架屏加载状态
- 分享面板

**使用的 Store**:
- `useItemStore`: 获取物品详情和相似物品
- `useRecommendStore`: 记录用户行为、管理喜欢状态

**路由配置**:
```typescript
{
  path: '/item/:id',
  name: 'ItemDetail',
  component: () => import('@/views/ItemDetail.vue')
}
```

**关键方法**:

```typescript
// 加载物品数据（并行加载）
async function loadItemData() {
  const [itemData, statsData, similar] = await Promise.all([
    itemStore.getItem(itemId.value),
    itemStore.getItemStats(itemId.value),
    itemStore.getSimilarItems(itemId.value, 12),
  ])
  
  // 记录浏览行为
  await recommendStore.recordBehavior({
    item_id: itemId.value,
    action: 'view',
  })
}
```

---

### 3. SearchBar.vue - 搜索栏组件

**文件路径**: `components/SearchBar.vue`

**Props**:
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| modelValue | string | - | v-model 绑定值 |
| placeholder | string | '搜索电影、商品、文章...' | 占位文本 |
| loading | boolean | false | 加载状态 |
| autofocus | boolean | false | 自动聚焦 |

**Events**:
| 事件名 | 参数 | 说明 |
|--------|------|------|
| update:modelValue | (value: string) | 输入值变化 |
| search | (query: string) | 触发搜索 |
| clear | - | 清空输入 |

**使用示例**:
```vue
<SearchBar
  v-model="query"
  @search="handleSearch"
  @clear="handleClear"
  placeholder="搜索..."
  :loading="isSearching"
/>
```

---

### 4. SearchResults.vue - 搜索结果组件

**文件路径**: `components/SearchResults.vue`

**Props**:
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| items | Item[] | - | 搜索结果列表 |
| query | string | '' | 搜索关键词（用于高亮） |

**Events**:
| 事件名 | 参数 | 说明 |
|--------|------|------|
| item-click | (itemId: string) | 点击结果项 |

**功能特性**:
- 网格布局展示
- 关键词高亮
- 空结果状态提示
- 类型图标和颜色区分

---

### 5. ItemInfo.vue - 物品信息组件

**文件路径**: `components/ItemInfo.vue`

**Props**:
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| item | Item | - | 物品信息 |
| stats | ItemStats \| null | null | 统计数据 |

**展示内容**:
- 标题和类型徽章
- 状态标签（已发布/已下架）
- 统计数据（浏览、点赞、分享、评分）
- 分类和标签
- 描述信息
- 元数据（根据类型显示不同字段）
- 时间信息

**元数据字段映射**:
| 物品类型 | 显示字段 |
|----------|----------|
| movie | 导演、主演、年份、时长、地区 |
| product | 品牌、价格、库存 |
| article | 作者、字数、来源 |
| video | 创作者、时长、画质 |

---

### 6. SimilarItems.vue - 相似推荐组件

**文件路径**: `components/SimilarItems.vue`

**Props**:
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| items | SimilarItem[] | - | 相似物品列表 |

**Events**:
| 事件名 | 参数 | 说明 |
|--------|------|------|
| item-click | (itemId: string) | 点击物品 |

**功能特性**:
- 横向滚动展示
- 相似度分数显示（百分比）
- 相似度等级样式（高/中/低）

---

### 7. ActionButtons.vue - 操作按钮组

**文件路径**: `components/ActionButtons.vue`

**Props**:
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| isLiked | boolean | false | 是否已喜欢 |
| disabled | boolean | false | 是否禁用 |

**Events**:
| 事件名 | 参数 | 说明 |
|--------|------|------|
| like | - | 点击喜欢 |
| share | - | 点击分享 |

---

## 接口依赖

本模块使用依赖注入方式获取 API 服务，需要在应用入口注入 `IApiProvider`。

### 使用的接口

```typescript
// 物品服务接口
interface IItemService {
  getItem(itemId: string): Promise<Item>
  searchItems(query: string, limit?: number): Promise<Item[]>
  getItemStats(itemId: string): Promise<ItemStats>
  getSimilarItems(itemId: string, limit?: number): Promise<SimilarItem[]>
}

// 推荐服务接口
interface IRecommendService {
  submitFeedback(feedback: FeedbackRequest): Promise<void>
}
```

### Store 中的使用方式

```typescript
// 在 Store 中通过 inject 获取 API
const api = inject<IApiProvider>('api')

// 调用接口
const item = await api.item.getItem(itemId)
```

---

## 类型定义

核心类型定义位于 `shared/types/index.ts`：

```typescript
// 物品类型
interface Item {
  id: string
  type: 'movie' | 'product' | 'article' | 'video'
  title: string
  description: string
  category: string
  tags: string[]
  metadata?: Record<string, any>
  status: 'active' | 'inactive'
  created_at: string
  updated_at: string
}

// 物品统计
interface ItemStats {
  item_id: string
  view_count: number
  click_count: number
  like_count: number
  share_count: number
  avg_rating: number
}

// 相似物品
interface SimilarItem {
  item: Item
  score: number  // 0-1 相似度分数
}
```

---

## 状态管理

### useItemStore

物品相关状态管理，包含：
- `currentItem`: 当前查看的物品
- `currentItemStats`: 当前物品统计
- `searchResults`: 搜索结果
- `searchQuery`: 搜索关键词
- `similarItems`: 相似物品
- `recentlyViewed`: 最近浏览记录

### useRecommendStore

推荐和行为相关状态管理，包含：
- `likedItems`: 已喜欢物品集合
- `recordBehavior()`: 记录用户行为
- `toggleLike()`: 切换喜欢状态

---

## 单元测试

### 运行测试

```bash
# 运行所有测试
npm run test

# 运行特定文件测试
npm run test -- Search.spec.ts

# 监听模式
npm run test:watch

# 生成覆盖率报告
npm run test:coverage
```

### 测试覆盖的场景

**Search.vue 测试**:
- 初始状态渲染
- 搜索功能触发
- URL 参数同步
- 热门搜索点击
- 过滤功能
- 清空搜索

**ItemDetail.vue 测试**:
- 加载状态（骨架屏）
- 物品信息展示
- 相似推荐展示
- 用户交互
- 导航功能
- 错误处理
- 404 状态
- 路由变化响应

**组件测试**:
- Props 渲染
- 事件触发
- 条件渲染
- 样式类应用

---

## 样式规范

### 颜色主题

组件采用暗色主题，主要颜色：
- 背景: `#0f0f23`, `#1a1a3e`
- 主色: `#6366f1` (Indigo)
- 强调色: `#8b5cf6`, `#a855f7` (Purple)
- 文字: `#f1f5f9`, `#e2e8f0`, `#94a3b8`

### 物品类型配色

| 类型 | 渐变色 |
|------|--------|
| movie | `#dc2626 → #9333ea` |
| product | `#059669 → #0891b2` |
| article | `#ea580c → #ca8a04` |
| video | `#db2777 → #9333ea` |

### 响应式断点

- 移动端: `max-width: 640px`
- 平板: `max-width: 768px`
- 桌面: `max-width: 900px`

---

## 开发指南

### 1. 添加新的物品类型

1. 在 `shared/types/index.ts` 中更新 `Item` 类型
2. 在相关组件中添加类型映射：
   - `typeNames`: 类型显示名
   - `typeGradients`: 类型渐变色
   - `typeIcons`: 类型图标

```typescript
// 示例：添加音乐类型
const typeNames = {
  // ...
  music: '音乐',
}

const typeGradients = {
  // ...
  music: 'linear-gradient(135deg, #10b981 0%, #06b6d4 100%)',
}
```

### 2. 扩展元数据展示

在 `ItemInfo.vue` 的 `metadataItems` 计算属性中添加新类型的元数据字段：

```typescript
const metadataItems = computed(() => {
  // ...
  if (props.item.type === 'music') {
    if (metadata.artist) items.push({ label: '艺人', value: metadata.artist })
    if (metadata.album) items.push({ label: '专辑', value: metadata.album })
  }
  // ...
})
```

### 3. 自定义搜索过滤

在 `Search.vue` 中添加新的过滤条件：

```typescript
const filters = ref({
  type: '',
  category: '',
  // 添加新过滤条件
  year: '',
})

const filteredResults = computed(() => {
  let results = searchResults.value
  // ...
  if (filters.value.year) {
    results = results.filter(item => 
      item.metadata?.year === filters.value.year
    )
  }
  return results
})
```

---

## 常见问题

### Q: 如何处理 API 加载失败？

所有组件都有错误状态处理，会显示友好的错误提示和重试按钮。

### Q: 如何实现无限滚动？

可以在 `SearchResults.vue` 中添加 Intersection Observer：

```typescript
const observer = new IntersectionObserver((entries) => {
  if (entries[0].isIntersecting) {
    loadMore()
  }
})
```

### Q: 如何添加搜索历史功能？

可以在 `useItemStore` 中添加搜索历史管理，存储到 localStorage。

---

## 后续优化建议

1. **性能优化**
   - 添加虚拟滚动支持大量搜索结果
   - 图片懒加载
   - 组件按需加载

2. **功能增强**
   - 搜索建议/自动补全
   - 高级搜索（多条件组合）
   - 搜索历史记录
   - 语音搜索

3. **用户体验**
   - 添加骨架屏动画
   - 优化移动端手势交互
   - 添加返回顶部按钮

---

## 联系方式

如有问题，请联系 Person B 或在项目中提交 Issue。

