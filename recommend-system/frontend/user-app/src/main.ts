/**
 * 应用入口文件
 * 
 * 负责初始化 Vue 应用、插件和全局配置
 * 
 * @module user-app/main
 * @author Person F
 */

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate'

import App from './App.vue'
import router from './router'
import { apiPlugin, MockApiProvider, HttpApiProvider } from '@shared/api'

// 全局样式
import './assets/styles/main.css'

// =============================================================================
// 创建应用实例
// =============================================================================

const app = createApp(App)

// =============================================================================
// Pinia 状态管理
// =============================================================================

const pinia = createPinia()

// 添加持久化插件
pinia.use(piniaPluginPersistedstate)

app.use(pinia)

// =============================================================================
// API Provider
// =============================================================================

// 根据环境决定使用 Mock 还是 HTTP
const useMock = import.meta.env.DEV && import.meta.env.VITE_USE_MOCK === 'true'

app.use(apiPlugin, { mock: useMock })

// 开发环境日志
if (import.meta.env.DEV) {
  console.log(`[App] API Provider: ${useMock ? 'Mock' : 'HTTP'}`)
  console.log(`[App] Environment: ${import.meta.env.MODE}`)
}

// =============================================================================
// Vue Router
// =============================================================================

app.use(router)

// =============================================================================
// 全局错误处理
// =============================================================================

/**
 * Vue 组件错误处理
 */
app.config.errorHandler = (err, instance, info) => {
  console.error('[Vue Error]', err)
  console.error('[Component]', instance)
  console.error('[Info]', info)
  
  // 可以在这里上报错误到监控服务
  // reportError({ err, instance, info })
}

/**
 * Vue 警告处理（仅开发环境）
 */
if (import.meta.env.DEV) {
  app.config.warnHandler = (msg, instance, trace) => {
    console.warn('[Vue Warning]', msg)
    if (trace) {
      console.warn('[Trace]', trace)
    }
  }
}

// =============================================================================
// 全局属性
// =============================================================================

// 版本号
app.config.globalProperties.$version = __APP_VERSION__

// =============================================================================
// 性能监控（开发环境）
// =============================================================================

if (import.meta.env.DEV) {
  app.config.performance = true
}

// =============================================================================
// 全局未捕获错误处理
// =============================================================================

window.addEventListener('error', (event) => {
  console.error('[Global Error]', event.error)
})

window.addEventListener('unhandledrejection', (event) => {
  console.error('[Unhandled Promise Rejection]', event.reason)
})

// =============================================================================
// 挂载应用
// =============================================================================

// 等待路由准备就绪后挂载
router.isReady().then(() => {
  app.mount('#app')
  
  // 移除加载动画
  const loading = document.querySelector('.app-loading')
  if (loading) {
    loading.remove()
  }
  
  if (import.meta.env.DEV) {
    console.log('[App] Mounted successfully')
  }
})

// =============================================================================
// 类型声明
// =============================================================================

declare global {
  const __APP_VERSION__: string
}

declare module 'vue' {
  interface ComponentCustomProperties {
    $version: string
  }
}

