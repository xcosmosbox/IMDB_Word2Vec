<script setup lang="ts">
/**
 * AdminLayout - 管理后台主布局
 * 
 * 提供侧边栏导航、顶部栏、面包屑等功能
 */
import { ref, computed, h } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  Layout,
  LayoutSider,
  LayoutHeader,
  LayoutContent,
  Menu,
  MenuItem,
  SubMenu,
  Breadcrumb,
  BreadcrumbItem,
  Avatar,
  Dropdown,
  Badge,
  theme,
} from 'ant-design-vue'
import {
  UserOutlined,
  ShoppingOutlined,
  DashboardOutlined,
  SettingOutlined,
  LogoutOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  BarChartOutlined,
  BellOutlined,
} from '@ant-design/icons-vue'
import { useAdminStore } from '@/stores/admin'

const route = useRoute()
const router = useRouter()
const adminStore = useAdminStore()

// 侧边栏折叠状态
const collapsed = ref(false)

// 菜单配置
const menuItems = [
  {
    key: '/admin/dashboard',
    icon: () => h(DashboardOutlined),
    label: '仪表盘',
  },
  {
    key: '/admin/users',
    icon: () => h(UserOutlined),
    label: '用户管理',
    children: [
      {
        key: '/admin/users',
        label: '用户列表',
      },
      {
        key: '/admin/users/create',
        label: '新增用户',
      },
    ],
  },
  {
    key: '/admin/items',
    icon: () => h(ShoppingOutlined),
    label: '物品管理',
    children: [
      {
        key: '/admin/items',
        label: '物品列表',
      },
      {
        key: '/admin/items/create',
        label: '新增物品',
      },
    ],
  },
  {
    key: '/admin/analytics',
    icon: () => h(BarChartOutlined),
    label: '数据分析',
  },
  {
    key: '/admin/settings',
    icon: () => h(SettingOutlined),
    label: '系统设置',
  },
]

// 当前选中的菜单项
const selectedKeys = computed(() => {
  const path = route.path
  // 处理子路由匹配
  if (path.includes('/users/')) {
    return ['/admin/users']
  }
  if (path.includes('/items/')) {
    return ['/admin/items']
  }
  return [path]
})

// 展开的子菜单
const openKeys = ref<string[]>(['/admin/users', '/admin/items'])

// 面包屑导航
const breadcrumbs = computed(() => {
  const matched = route.matched.filter(r => r.meta?.title)
  return matched.map(r => ({
    path: r.path,
    title: r.meta?.title as string,
  }))
})

// 处理菜单点击
function handleMenuClick({ key }: { key: string }) {
  router.push(key)
}

// 切换侧边栏折叠状态
function toggleCollapsed() {
  collapsed.value = !collapsed.value
}

// 退出登录
async function handleLogout() {
  await adminStore.logout()
  router.push('/admin/login')
}

// 跳转到个人信息页
function goToProfile() {
  router.push('/admin/profile')
}

// 用户下拉菜单
const userDropdownItems = [
  {
    key: 'profile',
    icon: () => h(UserOutlined),
    label: '个人信息',
    onClick: goToProfile,
  },
  {
    type: 'divider',
  },
  {
    key: 'logout',
    icon: () => h(LogoutOutlined),
    label: '退出登录',
    danger: true,
    onClick: handleLogout,
  },
]
</script>

<template>
  <Layout class="admin-layout">
    <!-- 侧边栏 -->
    <LayoutSider
      v-model:collapsed="collapsed"
      :trigger="null"
      collapsible
      :width="240"
      :collapsed-width="80"
      class="admin-sider"
      theme="dark"
    >
      <!-- Logo -->
      <div class="logo" @click="router.push('/admin/dashboard')">
        <div class="logo-icon">
          <DashboardOutlined />
        </div>
        <transition name="fade">
          <span v-if="!collapsed" class="logo-text">推荐系统管理</span>
        </transition>
      </div>

      <!-- 导航菜单 -->
      <Menu
        v-model:selectedKeys="selectedKeys"
        v-model:openKeys="openKeys"
        theme="dark"
        mode="inline"
        :inline-collapsed="collapsed"
        @click="handleMenuClick"
      >
        <template v-for="item in menuItems" :key="item.key">
          <!-- 有子菜单 -->
          <SubMenu v-if="item.children" :key="item.key">
            <template #icon>
              <component :is="item.icon" />
            </template>
            <template #title>{{ item.label }}</template>
            <MenuItem v-for="child in item.children" :key="child.key">
              {{ child.label }}
            </MenuItem>
          </SubMenu>
          <!-- 无子菜单 -->
          <MenuItem v-else :key="item.key">
            <template #icon>
              <component :is="item.icon" />
            </template>
            <span>{{ item.label }}</span>
          </MenuItem>
        </template>
      </Menu>
    </LayoutSider>

    <Layout class="main-layout">
      <!-- 顶部栏 -->
      <LayoutHeader class="admin-header">
        <div class="header-left">
          <!-- 折叠按钮 -->
          <div class="trigger" @click="toggleCollapsed">
            <component :is="collapsed ? MenuUnfoldOutlined : MenuFoldOutlined" />
          </div>

          <!-- 面包屑 -->
          <Breadcrumb class="breadcrumb">
            <BreadcrumbItem>
              <router-link to="/admin/dashboard">首页</router-link>
            </BreadcrumbItem>
            <BreadcrumbItem v-for="item in breadcrumbs" :key="item.path">
              <router-link v-if="item.path !== route.path" :to="item.path">
                {{ item.title }}
              </router-link>
              <span v-else>{{ item.title }}</span>
            </BreadcrumbItem>
          </Breadcrumb>
        </div>

        <div class="header-right">
          <!-- 通知图标 -->
          <div class="header-action">
            <Badge :count="5" :offset="[-2, 2]">
              <BellOutlined class="action-icon" />
            </Badge>
          </div>

          <!-- 用户信息 -->
          <Dropdown placement="bottomRight">
            <div class="user-info">
              <Avatar :size="32" class="user-avatar">
                {{ adminStore.currentAdmin?.name?.[0] || 'A' }}
              </Avatar>
              <span class="user-name">
                {{ adminStore.currentAdmin?.name || '管理员' }}
              </span>
            </div>
            <template #overlay>
              <Menu>
                <MenuItem key="profile" @click="goToProfile">
                  <UserOutlined />
                  <span style="margin-left: 8px">个人信息</span>
                </MenuItem>
                <MenuItem key="logout" @click="handleLogout">
                  <LogoutOutlined />
                  <span style="margin-left: 8px">退出登录</span>
                </MenuItem>
              </Menu>
            </template>
          </Dropdown>
        </div>
      </LayoutHeader>

      <!-- 内容区 -->
      <LayoutContent class="admin-content">
        <div class="content-wrapper">
          <router-view v-slot="{ Component }">
            <transition name="fade-slide" mode="out-in">
              <component :is="Component" />
            </transition>
          </router-view>
        </div>
      </LayoutContent>
    </Layout>
  </Layout>
</template>

<style scoped>
.admin-layout {
  min-height: 100vh;
}

.admin-sider {
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  z-index: 100;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.15);
}

.logo {
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 0 16px;
  background: rgba(255, 255, 255, 0.05);
  cursor: pointer;
  transition: all 0.3s;
  overflow: hidden;
}

.logo:hover {
  background: rgba(255, 255, 255, 0.1);
}

.logo-icon {
  font-size: 24px;
  color: #1890ff;
  flex-shrink: 0;
}

.logo-text {
  font-size: 16px;
  font-weight: 600;
  color: #fff;
  white-space: nowrap;
}

.main-layout {
  margin-left: 240px;
  transition: margin-left 0.2s;
}

.admin-sider:deep(.ant-layout-sider-collapsed) + .main-layout {
  margin-left: 80px;
}

.admin-header {
  position: sticky;
  top: 0;
  z-index: 99;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  background: #fff;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
}

.trigger {
  font-size: 18px;
  padding: 8px;
  cursor: pointer;
  transition: all 0.3s;
  border-radius: 4px;
}

.trigger:hover {
  background: #f5f5f5;
  color: #1890ff;
}

.breadcrumb {
  margin: 0;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.header-action {
  padding: 8px 12px;
  cursor: pointer;
  transition: background 0.3s;
  border-radius: 4px;
}

.header-action:hover {
  background: #f5f5f5;
}

.action-icon {
  font-size: 18px;
  color: #666;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 8px;
  cursor: pointer;
  border-radius: 4px;
  transition: background 0.3s;
}

.user-info:hover {
  background: #f5f5f5;
}

.user-avatar {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.user-name {
  font-weight: 500;
  color: #333;
}

.admin-content {
  margin: 24px;
  min-height: calc(100vh - 64px - 48px);
}

.content-wrapper {
  background: #fff;
  border-radius: 8px;
  padding: 24px;
  min-height: 100%;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.fade-slide-enter-active,
.fade-slide-leave-active {
  transition: all 0.3s ease;
}

.fade-slide-enter-from {
  opacity: 0;
  transform: translateX(20px);
}

.fade-slide-leave-to {
  opacity: 0;
  transform: translateX(-20px);
}

/* 响应式调整 */
@media (max-width: 768px) {
  .main-layout {
    margin-left: 0;
  }

  .admin-sider {
    position: fixed;
    z-index: 200;
  }

  .user-name {
    display: none;
  }
}
</style>

