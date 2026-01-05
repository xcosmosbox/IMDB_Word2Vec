/**
 * Vite 配置文件 - 用户端应用
 * 
 * @author Person F
 */

import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig(({ mode }) => {
  // 加载环境变量
  const env = loadEnv(mode, process.cwd(), '')
  
  return {
    // 插件配置
    plugins: [
      vue(),
    ],
    
    // 路径别名
    resolve: {
      alias: {
        '@': resolve(__dirname, 'src'),
        '@shared': resolve(__dirname, '../shared'),
      },
    },
    
    // 开发服务器配置
    server: {
      port: 3000,
      host: true,
      open: true,
      cors: true,
      
      // API 代理
      proxy: {
        '/api': {
          target: env.VITE_API_TARGET || 'http://localhost:8080',
          changeOrigin: true,
          rewrite: (path) => path,
        },
      },
    },
    
    // 预览服务器配置
    preview: {
      port: 4173,
      host: true,
    },
    
    // 构建配置
    build: {
      target: 'es2020',
      outDir: 'dist',
      assetsDir: 'assets',
      
      // 生产环境移除 console
      minify: 'terser',
      terserOptions: {
        compress: {
          drop_console: true,
          drop_debugger: true,
        },
      },
      
      // 代码分割
      rollupOptions: {
        output: {
          // 手动分包
          manualChunks: {
            'vue-vendor': ['vue', 'vue-router', 'pinia'],
            'utils-vendor': ['axios', '@vueuse/core'],
          },
          
          // 文件命名
          entryFileNames: 'js/[name]-[hash].js',
          chunkFileNames: 'js/[name]-[hash].js',
          assetFileNames: (assetInfo) => {
            const name = assetInfo.name || ''
            if (name.endsWith('.css')) {
              return 'css/[name]-[hash][extname]'
            }
            if (/\.(png|jpe?g|gif|svg|webp|ico)$/.test(name)) {
              return 'images/[name]-[hash][extname]'
            }
            if (/\.(woff2?|eot|ttf|otf)$/.test(name)) {
              return 'fonts/[name]-[hash][extname]'
            }
            return 'assets/[name]-[hash][extname]'
          },
        },
      },
      
      // 分块大小警告阈值
      chunkSizeWarningLimit: 1000,
    },
    
    // CSS 配置
    css: {
      devSourcemap: true,
    },
    
    // 优化依赖
    optimizeDeps: {
      include: ['vue', 'vue-router', 'pinia', 'axios'],
    },
    
    // 定义全局常量
    define: {
      __APP_VERSION__: JSON.stringify(process.env.npm_package_version || '1.0.0'),
    },
  }
})

